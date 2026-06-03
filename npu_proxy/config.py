"""Authoritative runtime and bootstrap configuration helpers."""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"
DEFAULT_LLM_MODEL: str = "tinyllama-1.1b-chat-int4-ov"
DEFAULT_DEVICE: str = "NPU"
DEFAULT_INFERENCE_TIMEOUT: int = 180
DEFAULT_MAX_PROMPT_LEN: int = 4096
DEFAULT_PREFIX_CACHE_MODE: str = "auto"
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8080
DEFAULT_TOKEN_LIMIT: int = 1800
DEFAULT_WORKERS: int = 1
DEFAULT_LOG_LEVEL: str = "info"
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1", "[::1]", "testserver", "test")

_HOSTNAME_RE = re.compile(r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$")

VALID_COMPILE_CACHE_MODES = {"OPTIMIZE_SIZE", "OPTIMIZE_SPEED"}
VALID_PREFIX_CACHE_MODES = {"auto", "on", "off"}
VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
VALID_CLI_DEVICE_CHOICES = {"AUTO", "NPU", "GPU", "CPU"}

ENV_HOST = "NPU_PROXY_HOST"
ENV_PORT = "NPU_PROXY_PORT"
ENV_DEVICE = "NPU_PROXY_DEVICE"
ENV_TOKEN_LIMIT = "NPU_PROXY_TOKEN_LIMIT"
ENV_REAL_INFERENCE = "NPU_PROXY_REAL_INFERENCE"
ENV_INFERENCE_TIMEOUT = "NPU_PROXY_INFERENCE_TIMEOUT"
ENV_MAX_PROMPT_LEN = "NPU_PROXY_MAX_PROMPT_LEN"
ENV_COMPILE_CACHE_DIR = "NPU_PROXY_COMPILE_CACHE_DIR"
ENV_COMPILE_CACHE_MODE = "NPU_PROXY_COMPILE_CACHE_MODE"
ENV_PREFIX_CACHE_MODE = "NPU_PROXY_PREFIX_CACHE_MODE"
ENV_LLM_BACKEND = "NPU_PROXY_LLM_BACKEND"
ENV_ENABLE_ALPHA_BACKENDS = "NPU_PROXY_ENABLE_ALPHA_BACKENDS"
ENV_LLAMACPP_MODEL_PATH = "NPU_PROXY_LLAMACPP_MODEL_PATH"
ENV_ALLOWED_HOSTS = "NPU_PROXY_ALLOWED_HOSTS"
ENV_PREFERRED_DEVICE = "NPU_PROXY_PREFERRED_DEVICE"
ENV_FALLBACK_DEVICE = "NPU_PROXY_FALLBACK_DEVICE"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

_active_bootstrap_config: "ProxyBootstrapConfig | None" = None
_active_bootstrap_lock = threading.Lock()


class LLMRuntimeConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


class LLMBackend(str, Enum):
    """Supported LLM backend identifiers."""

    OPENVINO = "openvino"
    LLAMA_CPP = "llama_cpp"

    @classmethod
    def parse(cls, value: str | None) -> "LLMBackend":
        """Parse an environment or user-provided backend value."""
        if value is None or not value.strip():
            return cls.OPENVINO

        normalized = value.strip().lower().replace("-", "_").replace(".", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = ", ".join(member.value for member in cls)
            raise LLMRuntimeConfigError(
                f"Unsupported LLM backend {value!r}. Expected one of: {valid}."
            ) from exc


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Resolved LLM runtime configuration."""

    backend: LLMBackend = LLMBackend.OPENVINO
    model_path: Path = DEFAULT_MODEL_DIR / DEFAULT_LLM_MODEL
    device: str = DEFAULT_DEVICE
    inference_timeout: int = DEFAULT_INFERENCE_TIMEOUT
    max_prompt_len: int = DEFAULT_MAX_PROMPT_LEN
    compile_cache_dir: Path | None = None
    compile_cache_mode: str | None = None
    prefix_cache_mode: str = DEFAULT_PREFIX_CACHE_MODE
    enable_alpha_backends: bool = False
    llama_cpp_model_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        object.__setattr__(self, "device", str(self.device).strip().upper())

        if self.inference_timeout <= 0:
            raise LLMRuntimeConfigError("inference_timeout must be greater than zero")
        if self.max_prompt_len <= 0:
            raise LLMRuntimeConfigError("max_prompt_len must be greater than zero")

        if self.compile_cache_dir is not None:
            object.__setattr__(self, "compile_cache_dir", Path(self.compile_cache_dir))

        object.__setattr__(
            self,
            "compile_cache_mode",
            normalize_compile_cache_mode(self.compile_cache_mode),
        )
        object.__setattr__(
            self,
            "prefix_cache_mode",
            normalize_prefix_cache_mode(self.prefix_cache_mode),
        )

        if self.llama_cpp_model_path is not None:
            object.__setattr__(
                self,
                "llama_cpp_model_path",
                Path(self.llama_cpp_model_path),
            )

    @property
    def is_alpha_backend(self) -> bool:
        """Return True when the selected backend requires alpha opt-in."""
        return self.backend is LLMBackend.LLAMA_CPP

    def backend_model_path(self) -> Path:
        """Return the active backend's configured model path."""
        if self.backend is LLMBackend.LLAMA_CPP:
            if self.llama_cpp_model_path is None:
                raise LLMRuntimeConfigError(
                    "llama.cpp backend requires NPU_PROXY_LLAMACPP_MODEL_PATH "
                    "to point to a local GGUF file."
                )
            return self.llama_cpp_model_path
        return self.model_path


@dataclass(frozen=True)
class ProxyBootstrapConfig:
    """Authoritative startup and runtime control-plane configuration."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    workers: int = DEFAULT_WORKERS
    reload: bool = False
    token_limit: int = DEFAULT_TOKEN_LIMIT
    real_inference: bool = False
    log_level: str = DEFAULT_LOG_LEVEL
    log_file: str | None = None
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", normalize_bind_host(self.host))
        object.__setattr__(self, "log_level", normalize_log_level(self.log_level))
        object.__setattr__(self, "allowed_hosts", normalize_allowed_hosts(self.allowed_hosts))
        validate_port(self.port)
        if self.workers <= 0:
            raise LLMRuntimeConfigError("workers must be greater than zero")
        if self.token_limit <= 0:
            raise LLMRuntimeConfigError("token_limit must be greater than zero")

    def as_environment(self) -> dict[str, str]:
        """Serialize the effective runtime config for compatibility shims."""
        env = {
            ENV_HOST: self.host,
            ENV_PORT: str(self.port),
            ENV_DEVICE: self.llm.device,
            ENV_TOKEN_LIMIT: str(self.token_limit),
            ENV_REAL_INFERENCE: "1" if self.real_inference else "0",
            ENV_INFERENCE_TIMEOUT: str(self.llm.inference_timeout),
            ENV_MAX_PROMPT_LEN: str(self.llm.max_prompt_len),
            ENV_PREFIX_CACHE_MODE: self.llm.prefix_cache_mode,
            ENV_LLM_BACKEND: self.llm.backend.value,
            ENV_ENABLE_ALPHA_BACKENDS: "1" if self.llm.enable_alpha_backends else "0",
            ENV_ALLOWED_HOSTS: ",".join(self.allowed_hosts),
        }
        if self.llm.compile_cache_dir is not None:
            env[ENV_COMPILE_CACHE_DIR] = str(self.llm.compile_cache_dir)
        if self.llm.compile_cache_mode is not None:
            env[ENV_COMPILE_CACHE_MODE] = self.llm.compile_cache_mode
        if self.llm.llama_cpp_model_path is not None:
            env[ENV_LLAMACPP_MODEL_PATH] = str(self.llm.llama_cpp_model_path)
        return env


@dataclass(frozen=True)
class CLIEnvironmentDefaults:
    """Centralized CLI defaults resolved from the environment."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str = "AUTO"
    token_limit: int = DEFAULT_TOKEN_LIMIT
    compile_cache_dir: str | None = None
    compile_cache_mode: str | None = None
    prefix_cache_mode: str = DEFAULT_PREFIX_CACHE_MODE
    real_inference: bool = False
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS


@dataclass(frozen=True)
class ContextRoutingConfig:
    """Forgiving advisory routing config resolved through the config layer."""

    token_limit: int = DEFAULT_TOKEN_LIMIT
    preferred_device: str = DEFAULT_DEVICE
    fallback_device: str | None = None


def validate_port(port: int) -> int:
    """Validate and return a TCP port number."""
    if not 1 <= int(port) <= 65535:
        raise LLMRuntimeConfigError("port must be between 1 and 65535")
    return int(port)


def _strip_ipv6_brackets(host: str) -> str:
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def normalize_bind_host(value: str | None) -> str:
    """Validate a uvicorn bind host as an IP address or DNS hostname."""
    host = str(value or "").strip() or DEFAULT_HOST
    if "://" in host or "/" in host or any(ch.isspace() for ch in host):
        raise LLMRuntimeConfigError(f"host must be an IP address or hostname, got {value!r}.")
    candidate = _strip_ipv6_brackets(host)
    try:
        ipaddress.ip_address(candidate)
        return candidate
    except ValueError:
        pass
    if ":" in candidate or not _HOSTNAME_RE.match(candidate):
        raise LLMRuntimeConfigError(f"host must be an IP address or hostname, got {value!r}.")
    return candidate.rstrip(".").lower()


def is_loopback_host(value: str | None) -> bool:
    """Return whether a bind/Host name is loopback-only."""
    host = str(value or "").strip().lower()
    candidate = _strip_ipv6_brackets(host)
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def normalize_allowed_hosts(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Normalize the Host-header allow-list used to block DNS rebinding."""
    if value is None:
        return DEFAULT_ALLOWED_HOSTS
    if isinstance(value, str):
        raw_hosts = [item.strip() for item in value.split(",")]
    else:
        raw_hosts = [str(item).strip() for item in value]
    hosts: list[str] = []
    for raw in raw_hosts:
        if not raw:
            continue
        if raw.startswith("[") and "]" in raw:
            closing = raw.index("]")
            host_part = raw[: closing + 1]
            suffix = raw[closing + 1 :]
            if suffix and not (suffix.startswith(":") and suffix[1:].isdigit()):
                raise LLMRuntimeConfigError(f"allowed host must be a hostname or IP, got {raw!r}.")
            normalized_host = normalize_bind_host(host_part)
            normalized = f"[{normalized_host}]{suffix}" if suffix else normalized_host
        else:
            host_part, sep, port_part = raw.rpartition(":")
            if sep and port_part.isdigit() and ":" not in host_part:
                normalized_host = normalize_bind_host(host_part)
                normalized = f"{normalized_host}:{validate_port(int(port_part))}"
            else:
                normalized = normalize_bind_host(raw)
        if normalized not in hosts:
            hosts.append(normalized)
    return tuple(hosts or DEFAULT_ALLOWED_HOSTS)


def _parse_int_or_default(name: str, value: str | None, default: int) -> int:
    """Parse a positive integer for lazy/observational paths, warning on fallback."""
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using default %s", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Invalid %s=%s, using default %s", name, value, default)
        return default
    return parsed


def _parse_int(name: str, value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise LLMRuntimeConfigError(f"{name} must be an integer, got {value!r}.") from exc


def parse_bool(name: str, value: str | None, default: bool = False) -> bool:
    """Parse a boolean-like string from the environment or CLI."""
    if value is None or value == "":
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    valid = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise LLMRuntimeConfigError(
        f"{name} must be a boolean-like value ({valid}), got {value!r}."
    )


def normalize_compile_cache_mode(
    value: str | None,
    *,
    default: str | None = None,
    strict: bool = True,
) -> str | None:
    """Normalize compile cache mode values."""
    if value is None or value == "":
        return default
    normalized = str(value).strip().upper().replace("-", "_")
    if normalized in VALID_COMPILE_CACHE_MODES:
        return normalized
    if strict:
        raise LLMRuntimeConfigError(
            "compile_cache_mode must be one of "
            f"{sorted(VALID_COMPILE_CACHE_MODES)}"
        )
    logger.warning("Ignoring invalid %s=%s", ENV_COMPILE_CACHE_MODE, value)
    return default


def normalize_prefix_cache_mode(
    value: str | None,
    *,
    default: str = DEFAULT_PREFIX_CACHE_MODE,
    strict: bool = True,
) -> str:
    """Normalize prefix cache mode values."""
    if value is None or value == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in VALID_PREFIX_CACHE_MODES:
        return normalized
    if strict:
        raise LLMRuntimeConfigError(
            "prefix_cache_mode must be one of "
            f"{sorted(VALID_PREFIX_CACHE_MODES)}"
        )
    logger.warning("Ignoring invalid %s=%s, using %s", ENV_PREFIX_CACHE_MODE, value, default)
    return default


def normalize_cli_device(
    value: str | None,
    *,
    default: str = "AUTO",
    strict: bool = True,
) -> str:
    """Normalize CLI-facing device selections, including AUTO."""
    if value is None or value == "":
        return default
    normalized = str(value).strip().upper()
    if normalized in VALID_CLI_DEVICE_CHOICES:
        return normalized
    if strict:
        raise LLMRuntimeConfigError(
            f"device must be one of {sorted(VALID_CLI_DEVICE_CHOICES)}"
        )
    logger.warning("Ignoring invalid %s=%s, using %s", ENV_DEVICE, value, default)
    return default


def normalize_log_level(value: str | None, default: str = DEFAULT_LOG_LEVEL) -> str:
    """Normalize log-level values."""
    if value is None or value == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in VALID_LOG_LEVELS:
        return normalized
    raise LLMRuntimeConfigError(
        f"log_level must be one of {sorted(VALID_LOG_LEVELS)}, got {value!r}."
    )


def normalize_context_device(value: str | None, *, default: str = DEFAULT_DEVICE, field_name: str = "device") -> str:
    """Normalize advisory routing devices without raising on lazy paths."""
    if value is None or value == "":
        return default
    normalized = str(value).strip().upper()
    if normalized in {"NPU", "GPU", "CPU"}:
        return normalized
    logger.warning("Ignoring invalid %s=%s, using %s", field_name, value, default)
    return default


def _env_overlay_bootstrap_config(current: ProxyBootstrapConfig) -> ProxyBootstrapConfig | None:
    """Return a non-persistent env overlay for lazy readers when tests mutate env."""
    overlay = current

    raw_device = os.environ.get(ENV_DEVICE)
    if raw_device:
        device = normalize_context_device(raw_device, default=current.llm.device, field_name=ENV_DEVICE)
        if device != current.llm.device:
            overlay = replace(overlay, llm=replace(overlay.llm, device=device))

    raw_token_limit = os.environ.get(ENV_TOKEN_LIMIT)
    if raw_token_limit:
        token_limit = _parse_int_or_default(ENV_TOKEN_LIMIT, raw_token_limit, current.token_limit)
        if token_limit != current.token_limit:
            overlay = replace(overlay, token_limit=token_limit)

    return overlay if overlay != current else None


def load_context_routing_config(env: Mapping[str, str] | None = None) -> ContextRoutingConfig:
    """Load forgiving advisory router config, with env overrides taking precedence."""
    env_map = os.environ if env is None else env
    with _active_bootstrap_lock:
        current = _active_bootstrap_config

    default_token_limit = current.token_limit if current is not None else DEFAULT_TOKEN_LIMIT
    default_preferred = current.llm.device if current is not None else DEFAULT_DEVICE

    token_limit = _parse_int_or_default(
        ENV_TOKEN_LIMIT,
        env_map.get(ENV_TOKEN_LIMIT),
        default_token_limit if env_map.get(ENV_TOKEN_LIMIT) is None else DEFAULT_TOKEN_LIMIT,
    )
    preferred_device = normalize_context_device(
        env_map.get(ENV_PREFERRED_DEVICE) or env_map.get(ENV_DEVICE),
        default=default_preferred,
        field_name=ENV_PREFERRED_DEVICE,
    )
    fallback_device = None
    if env_map.get(ENV_FALLBACK_DEVICE):
        fallback_device = normalize_context_device(
            env_map.get(ENV_FALLBACK_DEVICE),
            default="CPU",
            field_name=ENV_FALLBACK_DEVICE,
        )

    return ContextRoutingConfig(
        token_limit=token_limit,
        preferred_device=preferred_device,
        fallback_device=fallback_device,
    )


def load_cli_environment_defaults(
    env: Mapping[str, str] | None = None,
) -> CLIEnvironmentDefaults:
    """Load CLI defaults from the environment in one place."""
    env_map = os.environ if env is None else env
    return CLIEnvironmentDefaults(
        host=normalize_bind_host(env_map.get(ENV_HOST, DEFAULT_HOST)),
        port=validate_port(_parse_int(ENV_PORT, env_map.get(ENV_PORT), DEFAULT_PORT)),
        device=normalize_cli_device(
            env_map.get(ENV_DEVICE),
            default="AUTO",
            strict=False,
        ),
        token_limit=_parse_int(
            ENV_TOKEN_LIMIT,
            env_map.get(ENV_TOKEN_LIMIT),
            DEFAULT_TOKEN_LIMIT,
        ),
        compile_cache_dir=env_map.get(ENV_COMPILE_CACHE_DIR),
        compile_cache_mode=normalize_compile_cache_mode(
            env_map.get(ENV_COMPILE_CACHE_MODE),
            strict=False,
        ),
        prefix_cache_mode=normalize_prefix_cache_mode(
            env_map.get(ENV_PREFIX_CACHE_MODE),
            strict=False,
        ),
        real_inference=parse_bool(
            ENV_REAL_INFERENCE,
            env_map.get(ENV_REAL_INFERENCE),
            default=False,
        ),
        allowed_hosts=normalize_allowed_hosts(env_map.get(ENV_ALLOWED_HOSTS)),
    )


def load_llm_runtime_config(
    env: Mapping[str, str] | None = None,
    *,
    model_path: str | Path | None = None,
    backend: str | LLMBackend | None = None,
    device: str | None = None,
    inference_timeout: int | None = None,
    max_prompt_len: int | None = None,
    compile_cache_dir: str | Path | None = None,
    compile_cache_mode: str | None = None,
    prefix_cache_mode: str | None = None,
    enable_alpha_backends: bool | None = None,
    llama_cpp_model_path: str | Path | None = None,
) -> LLMRuntimeConfig:
    """Load LLM runtime configuration from environment with optional overrides."""
    env_map = os.environ if env is None else env

    resolved_backend = (
        backend
        if isinstance(backend, LLMBackend)
        else LLMBackend.parse(backend or env_map.get(ENV_LLM_BACKEND))
    )
    resolved_model_path = (
        Path(model_path) if model_path is not None else DEFAULT_MODEL_DIR / DEFAULT_LLM_MODEL
    )
    resolved_device = device or env_map.get(ENV_DEVICE, DEFAULT_DEVICE)
    resolved_timeout = (
        inference_timeout
        if inference_timeout is not None
        else _parse_int(
            ENV_INFERENCE_TIMEOUT,
            env_map.get(ENV_INFERENCE_TIMEOUT),
            DEFAULT_INFERENCE_TIMEOUT,
        )
    )
    resolved_max_prompt_len = (
        max_prompt_len
        if max_prompt_len is not None
        else _parse_int(
            ENV_MAX_PROMPT_LEN,
            env_map.get(ENV_MAX_PROMPT_LEN),
            DEFAULT_MAX_PROMPT_LEN,
        )
    )

    resolved_compile_cache_dir = compile_cache_dir
    if resolved_compile_cache_dir is None:
        raw_compile_cache_dir = env_map.get(ENV_COMPILE_CACHE_DIR)
        resolved_compile_cache_dir = Path(raw_compile_cache_dir) if raw_compile_cache_dir else None

    resolved_compile_cache_mode = (
        compile_cache_mode
        if compile_cache_mode is not None
        else env_map.get(ENV_COMPILE_CACHE_MODE)
    )
    resolved_prefix_cache_mode = (
        prefix_cache_mode
        if prefix_cache_mode is not None
        else env_map.get(ENV_PREFIX_CACHE_MODE, DEFAULT_PREFIX_CACHE_MODE)
    )
    resolved_alpha = (
        enable_alpha_backends
        if enable_alpha_backends is not None
        else parse_bool(
            ENV_ENABLE_ALPHA_BACKENDS,
            env_map.get(ENV_ENABLE_ALPHA_BACKENDS),
            default=False,
        )
    )

    resolved_llama_path = llama_cpp_model_path
    if resolved_llama_path is None:
        raw_llama_path = env_map.get(ENV_LLAMACPP_MODEL_PATH)
        resolved_llama_path = Path(raw_llama_path) if raw_llama_path else None

    return LLMRuntimeConfig(
        backend=resolved_backend,
        model_path=resolved_model_path,
        device=resolved_device,
        inference_timeout=resolved_timeout,
        max_prompt_len=resolved_max_prompt_len,
        compile_cache_dir=resolved_compile_cache_dir,
        compile_cache_mode=resolved_compile_cache_mode,
        prefix_cache_mode=resolved_prefix_cache_mode,
        enable_alpha_backends=resolved_alpha,
        llama_cpp_model_path=resolved_llama_path,
    )


def load_proxy_bootstrap_config(
    env: Mapping[str, str] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    workers: int | None = None,
    reload: bool | None = None,
    token_limit: int | None = None,
    real_inference: bool | None = None,
    log_level: str | None = None,
    log_file: str | None = None,
    allowed_hosts: str | tuple[str, ...] | list[str] | None = None,
    model_path: str | Path | None = None,
    backend: str | LLMBackend | None = None,
    device: str | None = None,
    inference_timeout: int | None = None,
    max_prompt_len: int | None = None,
    compile_cache_dir: str | Path | None = None,
    compile_cache_mode: str | None = None,
    prefix_cache_mode: str | None = None,
    enable_alpha_backends: bool | None = None,
    llama_cpp_model_path: str | Path | None = None,
) -> ProxyBootstrapConfig:
    """Load the authoritative bootstrap config from env and explicit overrides."""
    env_map = os.environ if env is None else env
    llm_device_override = None if device == "AUTO" else device
    return ProxyBootstrapConfig(
        host=host or str(env_map.get(ENV_HOST, DEFAULT_HOST)).strip() or DEFAULT_HOST,
        port=port
        if port is not None
        else _parse_int(ENV_PORT, env_map.get(ENV_PORT), DEFAULT_PORT),
        workers=workers if workers is not None else DEFAULT_WORKERS,
        reload=bool(reload) if reload is not None else False,
        token_limit=token_limit
        if token_limit is not None
        else _parse_int(ENV_TOKEN_LIMIT, env_map.get(ENV_TOKEN_LIMIT), DEFAULT_TOKEN_LIMIT),
        real_inference=real_inference
        if real_inference is not None
        else parse_bool(
            ENV_REAL_INFERENCE,
            env_map.get(ENV_REAL_INFERENCE),
            default=False,
        ),
        log_level=log_level or DEFAULT_LOG_LEVEL,
        log_file=log_file,
        allowed_hosts=normalize_allowed_hosts(
            allowed_hosts if allowed_hosts is not None else env_map.get(ENV_ALLOWED_HOSTS)
        ),
        llm=load_llm_runtime_config(
            env_map,
            model_path=model_path,
            backend=backend,
            device=llm_device_override,
            inference_timeout=inference_timeout,
            max_prompt_len=max_prompt_len,
            compile_cache_dir=compile_cache_dir,
            compile_cache_mode=compile_cache_mode,
            prefix_cache_mode=prefix_cache_mode,
            enable_alpha_backends=enable_alpha_backends,
            llama_cpp_model_path=llama_cpp_model_path,
        ),
    )


def apply_proxy_bootstrap_config_to_env(
    config: ProxyBootstrapConfig,
    environ: dict[str, str] | None = None,
) -> None:
    """Mirror the authoritative config into environment variables for legacy readers."""
    env_map = os.environ if environ is None else environ
    serialized = config.as_environment()
    mirrored_keys = {
        ENV_HOST,
        ENV_PORT,
        ENV_DEVICE,
        ENV_TOKEN_LIMIT,
        ENV_REAL_INFERENCE,
        ENV_INFERENCE_TIMEOUT,
        ENV_MAX_PROMPT_LEN,
        ENV_COMPILE_CACHE_DIR,
        ENV_COMPILE_CACHE_MODE,
        ENV_PREFIX_CACHE_MODE,
        ENV_LLM_BACKEND,
        ENV_ENABLE_ALPHA_BACKENDS,
        ENV_LLAMACPP_MODEL_PATH,
        ENV_ALLOWED_HOSTS,
    }
    for key in mirrored_keys:
        if key in serialized:
            env_map[key] = serialized[key]
        else:
            env_map.pop(key, None)


def activate_proxy_bootstrap_config(
    config: ProxyBootstrapConfig | None = None,
) -> ProxyBootstrapConfig:
    """Set the active authoritative bootstrap config."""
    global _active_bootstrap_config
    resolved = config or load_proxy_bootstrap_config()
    with _active_bootstrap_lock:
        _active_bootstrap_config = resolved
    return resolved


def get_active_proxy_bootstrap_config() -> ProxyBootstrapConfig:
    """Get the active bootstrap config, lazily applying explicit env overlays."""
    with _active_bootstrap_lock:
        current = _active_bootstrap_config
    if current is None:
        return load_proxy_bootstrap_config()
    return _env_overlay_bootstrap_config(current) or current


def get_active_llm_runtime_config() -> LLMRuntimeConfig:
    """Get the active LLM runtime config from the bootstrap control plane."""
    return get_active_proxy_bootstrap_config().llm


def reset_active_proxy_bootstrap_config() -> None:
    """Reset the active bootstrap config singleton."""
    global _active_bootstrap_config
    with _active_bootstrap_lock:
        _active_bootstrap_config = None
