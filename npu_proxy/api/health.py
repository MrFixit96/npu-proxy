"""Health endpoint handlers for the NPU Proxy service.

This module provides health check endpoints for monitoring service status,
device availability, and engine readiness.

Endpoints:
    GET /health: Observational service summary with engine info.
    GET /health/liveness: Cheap process-up probe.
    GET /health/readiness: Observational readiness gate for warmed runtime state.
    GET /health/devices: Detailed device information and fallback chain.

Deployment Note:
    NPU Proxy runs as a **native host service**, not in containers.
    Intel NPU drivers require direct hardware access and cannot be
    containerized (no Docker, Kubernetes, or WSL2 passthrough for NPU).
    
    For WSL2 workloads, the proxy runs on Windows and WSL2 clients
    connect via HTTP bridge (see NPU_WSL2_PROXY_SPEC.md).

Service Management:
    - **Windows**: Use Windows Service (sc.exe) or Task Scheduler
    - **Linux**: Use systemd (see packaging/npu-proxy.service)

Load Balancer Health Checks:
    Configure your load balancer to probe:
    - ``GET /health/liveness`` for basic availability
    - ``GET /health/readiness`` for warmed runtime readiness
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from npu_proxy import __version__
from npu_proxy.config import get_active_llm_runtime_config
from npu_proxy.inference import embedding_config
from npu_proxy.inference.devices import DEVICE_FALLBACK_CHAIN
from npu_proxy.inference.embedding_engine import (
    DEFAULT_CACHE_SIZE,
    get_loaded_embedding_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Module-level cached OpenVINO Core instance for efficient device queries
_ov_core: Any | None = None
_ENGINE_HEALTH_STATUSES = {"not_loaded", "loaded", "error"}


@dataclass(frozen=True)
class EmbeddingCacheSummary:
    """Typed embedding-cache summary surfaced through health payloads."""

    enabled: bool
    kind: str
    configured_max_entries: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"lru", "in_memory"}:
            raise ValueError(f"unsupported embedding cache kind {self.kind!r}")
        if self.configured_max_entries is not None and self.configured_max_entries <= 0:
            raise ValueError("configured_max_entries must be greater than zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "kind": self.kind,
            "configured_max_entries": self.configured_max_entries,
        }


@dataclass(frozen=True)
class LLMEngineHealthState:
    """Normalized LLM health state shared across health surfaces."""

    status: str = "not_loaded"
    device: str | None = None
    model: str | None = None
    backend: str | None = None
    requested_device: str | None = None
    fallback_device: str | None = None
    used_fallback: bool | None = None
    available_devices: tuple[str, ...] | None = None
    is_warmed_up: bool = False
    compile_cache_dir: str | None = None
    compile_cache_mode: str | None = None
    prefix_cache_mode: str | None = None
    runtime_features: dict[str, Any] | None = None
    model_load_seconds: float | None = None
    load_diagnostics: tuple[dict[str, Any], ...] = ()
    last_generation_stats: dict[str, Any] | None = None
    model_path: str | None = None
    model_format: str | None = None
    alpha_backend: bool | None = None
    config: dict[str, Any] | None = None
    message: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _ENGINE_HEALTH_STATUSES:
            raise ValueError(f"unsupported health status {self.status!r}")
        if self.model_load_seconds is not None and self.model_load_seconds < 0:
            raise ValueError("model_load_seconds must not be negative")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LLMEngineHealthState":
        return cls(
            status=str(payload.get("status") or "not_loaded"),
            device=payload.get("device"),
            model=payload.get("model"),
            backend=payload.get("backend"),
            requested_device=payload.get("requested_device"),
            fallback_device=payload.get("fallback_device"),
            used_fallback=payload.get("used_fallback"),
            available_devices=(
                tuple(str(device) for device in payload.get("available_devices") or ())
                if payload.get("available_devices") is not None
                else None
            ),
            is_warmed_up=bool(payload.get("is_warmed_up", False)),
            compile_cache_dir=payload.get("compile_cache_dir"),
            compile_cache_mode=payload.get("compile_cache_mode"),
            prefix_cache_mode=payload.get("prefix_cache_mode"),
            runtime_features=dict(payload.get("runtime_features") or {})
            if payload.get("runtime_features") is not None
            else None,
            model_load_seconds=payload.get("model_load_seconds"),
            load_diagnostics=tuple(dict(item) for item in payload.get("load_diagnostics") or ()),
            last_generation_stats=dict(payload.get("last_generation_stats") or {})
            if payload.get("last_generation_stats") is not None
            else None,
            model_path=payload.get("model_path"),
            model_format=payload.get("model_format"),
            alpha_backend=payload.get("alpha_backend"),
            config=dict(payload.get("config") or {}) if payload.get("config") is not None else None,
            message=payload.get("message"),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "device": self.device,
            "model": self.model,
            "backend": self.backend,
            "requested_device": self.requested_device,
            "fallback_device": self.fallback_device,
            "used_fallback": self.used_fallback,
            "available_devices": list(self.available_devices) if self.available_devices is not None else None,
            "is_warmed_up": self.is_warmed_up,
            "compile_cache_dir": self.compile_cache_dir,
            "compile_cache_mode": self.compile_cache_mode,
            "prefix_cache_mode": self.prefix_cache_mode,
            "runtime_features": self.runtime_features,
            "model_load_seconds": self.model_load_seconds,
            "load_diagnostics": [dict(item) for item in self.load_diagnostics],
            "last_generation_stats": self.last_generation_stats,
            "model_path": self.model_path,
            "model_format": self.model_format,
            "alpha_backend": self.alpha_backend,
            "config": self.config,
            "message": self.message,
            "error": self.error,
        }

    def runtime_state_payload(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "fallback_device": self.fallback_device,
            "used_fallback": self.used_fallback,
            "compile_cache_dir": self.compile_cache_dir,
            "compile_cache_mode": self.compile_cache_mode,
            "prefix_cache_mode": self.prefix_cache_mode,
            "runtime_features": self.runtime_features,
            "model_load_seconds": self.model_load_seconds,
            "last_generation_stats": self.last_generation_stats,
        }


@dataclass(frozen=True)
class EmbeddingEngineHealthState:
    """Normalized embedding health state shared across health surfaces."""

    status: str = "not_loaded"
    device: str | None = None
    model: str | None = None
    backend: str | None = None
    requested_device: str | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    dimensions: int | None = None
    is_production: bool | None = None
    is_fallback: bool | None = None
    fallback_reason: str | None = None
    fallback_mode: str | None = None
    load_error: str | None = None
    model_path: str | None = None
    canonical_model_path: str | None = None
    repo_id: str | None = None
    cache: EmbeddingCacheSummary | None = None
    downloaded: bool | None = None
    message: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _ENGINE_HEALTH_STATUSES:
            raise ValueError(f"unsupported health status {self.status!r}")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EmbeddingEngineHealthState":
        cache_payload = payload.get("cache")
        cache = None
        if isinstance(cache_payload, EmbeddingCacheSummary):
            cache = cache_payload
        elif isinstance(cache_payload, dict):
            cache = EmbeddingCacheSummary(
                enabled=bool(cache_payload.get("enabled", True)),
                kind=str(cache_payload.get("kind") or "in_memory"),
                configured_max_entries=cache_payload.get("configured_max_entries"),
            )
        return cls(
            status=str(payload.get("status") or "not_loaded"),
            device=payload.get("device"),
            model=payload.get("model"),
            backend=payload.get("backend"),
            requested_device=payload.get("requested_device"),
            requested_model=payload.get("requested_model"),
            resolved_model=payload.get("resolved_model"),
            dimensions=payload.get("dimensions"),
            is_production=payload.get("is_production"),
            is_fallback=payload.get("is_fallback"),
            fallback_reason=payload.get("fallback_reason"),
            fallback_mode=payload.get("fallback_mode"),
            load_error=payload.get("load_error"),
            model_path=payload.get("model_path"),
            canonical_model_path=payload.get("canonical_model_path"),
            repo_id=payload.get("repo_id"),
            cache=cache,
            downloaded=payload.get("downloaded"),
            message=payload.get("message"),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "device": self.device,
            "model": self.model,
            "backend": self.backend,
            "requested_device": self.requested_device,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "dimensions": self.dimensions,
            "is_production": self.is_production,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "fallback_mode": self.fallback_mode,
            "load_error": self.load_error,
            "model_path": self.model_path,
            "canonical_model_path": self.canonical_model_path,
            "repo_id": self.repo_id,
            "cache": self.cache.to_dict() if self.cache is not None else None,
            "downloaded": self.downloaded,
            "message": self.message,
            "error": self.error,
        }

    def runtime_state_payload(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "dimensions": self.dimensions,
            "is_production": self.is_production,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "fallback_mode": self.fallback_mode,
            "cache": self.cache.to_dict() if self.cache is not None else None,
        }


def _get_openvino_module() -> Any:
    """Import OpenVINO lazily so route registration stays side-effect free."""
    return importlib.import_module("openvino")


def get_ov_core() -> Any:
    """Get or create a cached OpenVINO Core instance.

    Uses module-level caching to avoid repeated Core initialization,
    which can be expensive on systems with many devices.

    Returns:
        OpenVINO Core instance for device queries.

    Note:
        The Core instance is created lazily on first access and reused
        for all subsequent calls within the module lifetime.
    """
    global _ov_core
    if _ov_core is None:
        _ov_core = _get_openvino_module().Core()
    return _ov_core


def is_model_loaded() -> bool:
    """Return whether the singleton LLM engine has been loaded."""
    from npu_proxy.inference.engine import is_model_loaded as _is_model_loaded

    return _is_model_loaded()


def get_loaded_models() -> dict[str, Any]:
    """Return loaded LLM engine instances without importing the backend at startup."""
    from npu_proxy.inference.engine import get_loaded_models as _get_loaded_models

    return _get_loaded_models()


def check_npu_available() -> tuple[bool, str | None]:
    """Check if an Intel NPU device is available via OpenVINO.

    Queries the OpenVINO runtime to determine if an NPU (Neural Processing
    Unit) is present and accessible on the current system.

    Returns:
        Tuple of availability and a stable error code when probing fails.
    """
    try:
        return "NPU" in get_ov_core().available_devices, None
    except Exception:
        logger.exception("NPU availability probe failed")
        return False, "device_probe_failed"


def check_gpu_available() -> tuple[bool, str | None]:
    """Check if an Intel integrated GPU is available via OpenVINO.

    Queries the OpenVINO runtime to determine if an Intel GPU (iGPU or
    discrete) is present and accessible for inference.

    Returns:
        Tuple of availability and a stable error code when probing fails.
    """
    try:
        return "GPU" in get_ov_core().available_devices, None
    except Exception:
        logger.exception("GPU availability probe failed")
        return False, "device_probe_failed"


def _safe_backend_model_path() -> str | None:
    """Return the configured backend model path when it can be resolved."""
    try:
        config = get_active_llm_runtime_config()
    except Exception:
        logger.exception("Failed to read LLM runtime config while resolving model path")
        return None

    try:
        return str(config.backend_model_path())
    except Exception:
        logger.exception("Failed to resolve backend model path")
        if config.llama_cpp_model_path is not None:
            return str(config.llama_cpp_model_path)
        return str(config.model_path)


def _path_exists(path_value: str | None) -> bool | None:
    """Return whether a configured path exists, preserving unknown paths as None."""
    if not path_value:
        return None
    try:
        return Path(path_value).exists()
    except OSError:
        return None


def _get_llm_runtime_config() -> dict[str, Any]:
    """Return additive LLM runtime configuration for health surfaces."""
    try:
        config = get_active_llm_runtime_config()
    except Exception:
        logger.exception("Failed to read LLM runtime config for health response")
        return {"error": "runtime_config_unavailable"}

    return {
        "backend": config.backend.value,
        "requested_device": config.device,
        "model_path": _safe_backend_model_path(),
        "compile_cache_dir": (
            str(config.compile_cache_dir) if config.compile_cache_dir is not None else None
        ),
        "compile_cache_mode": config.compile_cache_mode,
        "prefix_cache_mode": config.prefix_cache_mode,
        "alpha_backend": config.is_alpha_backend,
        "llama_cpp_model_path": (
            str(config.llama_cpp_model_path) if config.llama_cpp_model_path is not None else None
        ),
    }


def _get_available_devices() -> tuple[list[str], str | None]:
    """Return available OpenVINO devices without failing the health surface."""
    try:
        return list(get_ov_core().available_devices), None
    except Exception:
        logger.exception("OpenVINO device enumeration failed")
        return [], "device_enumeration_failed"


def _get_openvino_version() -> str | None:
    """Return the OpenVINO version without breaking degraded health responses."""
    try:
        return str(_get_openvino_module().__version__)
    except Exception:
        logger.debug("OpenVINO version probe failed", exc_info=True)
        return None


def _summarize_embedding_cache(engine_info: dict[str, Any]) -> EmbeddingCacheSummary:
    """Summarize the embedding cache without depending on engine internals."""
    is_production = bool(engine_info.get("is_production", False))
    return EmbeddingCacheSummary(
        enabled=True,
        kind="lru" if is_production else "in_memory",
        configured_max_entries=DEFAULT_CACHE_SIZE if is_production else None,
    )


def _get_device_pool_snapshot() -> list[dict[str, Any]]:
    """Return loaded per-device LLM engine pool state without side effects."""
    try:
        from npu_proxy.inference.engine import get_engine_pool_snapshot

        return get_engine_pool_snapshot()
    except Exception:
        logger.exception("Failed to inspect LLM device pool")
        return []


def _get_llm_engine_state() -> tuple[LLMEngineHealthState, dict[str, Any] | None]:
    """Collect additive LLM engine state while preserving legacy fields."""
    runtime_config = _get_llm_runtime_config()
    state: dict[str, Any] = {
        "status": "not_loaded",
        "device": None,
        "model": None,
        "backend": runtime_config.get("backend"),
        "requested_device": runtime_config.get("requested_device"),
        "fallback_device": None,
        "used_fallback": None,
        "available_devices": None,
        "is_warmed_up": False,
        "compile_cache_dir": runtime_config.get("compile_cache_dir"),
        "compile_cache_mode": runtime_config.get("compile_cache_mode"),
        "prefix_cache_mode": runtime_config.get("prefix_cache_mode"),
        "runtime_features": None,
        "model_load_seconds": None,
        "load_diagnostics": [],
        "last_generation_stats": None,
        "model_path": runtime_config.get("model_path"),
        "model_format": None,
        "alpha_backend": runtime_config.get("alpha_backend"),
        "config": runtime_config,
        "message": None,
    }
    device_info: dict[str, Any] | None = None

    if runtime_config.get("error"):
        state["status"] = "error"
        state["message"] = "Failed to read LLM runtime configuration."
        state["error"] = "runtime_config_unavailable"
        return LLMEngineHealthState.from_payload(state), device_info

    if not is_model_loaded():
        model_path = state.get("model_path")
        if _path_exists(model_path):
            state["message"] = (
                "LLM model is not loaded. /health is observational only and does not auto-load models."
            )
        else:
            state["message"] = (
                "LLM model is not loaded. /health is observational only and the configured model "
                f"path is missing: {model_path}"
            )
        return LLMEngineHealthState.from_payload(state), device_info

    try:
        loaded = get_loaded_models()
        if not loaded:
            return LLMEngineHealthState.from_payload(state), device_info

        name, engine = next(iter(loaded.items()))
        raw_device_info = engine.get_device_info() if hasattr(engine, "get_device_info") else {}
        device_info = dict(raw_device_info or {})

        state.update(
            {
                "status": "loaded",
                "device": device_info.get("actual_device", "unknown"),
                "model": getattr(engine, "model_name", name),
                "backend": device_info.get("backend", state["backend"]),
                "requested_device": device_info.get("requested_device", state["requested_device"]),
                "fallback_device": device_info.get("fallback_device"),
                "used_fallback": device_info.get("used_fallback"),
                "available_devices": device_info.get("available_devices"),
                "is_warmed_up": device_info.get("is_warmed_up", False),
                "compile_cache_dir": device_info.get(
                    "compile_cache_dir", state["compile_cache_dir"]
                ),
                "compile_cache_mode": device_info.get(
                    "compile_cache_mode", state["compile_cache_mode"]
                ),
                "prefix_cache_mode": device_info.get(
                    "prefix_cache_mode", state["prefix_cache_mode"]
                ),
                "runtime_features": device_info.get("runtime_features"),
                "model_load_seconds": device_info.get("model_load_seconds"),
                "load_diagnostics": list(device_info.get("load_diagnostics") or []),
                "last_generation_stats": device_info.get("last_generation_stats"),
                "model_path": device_info.get("model_path", state["model_path"]),
                "model_format": device_info.get("model_format"),
                "alpha_backend": device_info.get("alpha_backend", state["alpha_backend"]),
                "message": None,
            }
        )
    except Exception:
        logger.exception("Failed to inspect loaded LLM engine")
        state["status"] = "error"
        state["message"] = "Failed to inspect loaded LLM engine."
        state["error"] = "llm_engine_inspection_failed"

    return LLMEngineHealthState.from_payload(state), device_info


def _get_embedding_engine_state() -> tuple[EmbeddingEngineHealthState, dict[str, Any] | None]:
    """Collect additive embedding engine state for health surfaces."""
    state: dict[str, Any] = {
        "status": "not_loaded",
        "device": None,
        "model": None,
        "backend": None,
        "requested_device": None,
        "requested_model": None,
        "resolved_model": None,
        "dimensions": None,
        "is_production": None,
        "is_fallback": None,
        "fallback_reason": None,
        "fallback_mode": None,
        "load_error": None,
        "model_path": None,
        "canonical_model_path": None,
        "repo_id": None,
        "cache": None,
        "downloaded": None,
        "message": None,
    }

    try:
        config = embedding_config.resolve_embedding_model_config()
        state.update(
            {
                "requested_device": config.requested_device,
                "requested_model": config.requested_model,
                "resolved_model": config.resolved_model,
                "dimensions": config.dimensions,
                "model_path": str(config.model_path),
                "canonical_model_path": str(config.canonical_path),
                "repo_id": config.repo_id,
                "downloaded": config.is_downloaded,
            }
        )
    except Exception:
        logger.exception("Failed to resolve embedding configuration")
        state["status"] = "error"
        state["message"] = "Failed to resolve embedding configuration."
        state["error"] = "embedding_config_unavailable"
        return EmbeddingEngineHealthState.from_payload(state), None

    emb_engine = get_loaded_embedding_engine(
        model_name=str(state["requested_model"]) if state["requested_model"] else None,
        device=str(state["requested_device"]) if state["requested_device"] else None,
    )
    if emb_engine is None:
        if state["downloaded"]:
            state["message"] = (
                "Embedding model is not loaded. /health is observational only and does not "
                "auto-load embedding engines."
            )
        else:
            state["message"] = (
                "Embedding model is not loaded. /health is observational only and the configured "
                f"embedding files are missing from {state['canonical_model_path']}."
            )
        return EmbeddingEngineHealthState.from_payload(state), None

    try:
        engine_info = dict(emb_engine.get_engine_info())
        backend = engine_info.get("backend")
        if backend is None:
            backend = "hash" if engine_info.get("is_fallback") else "openvino"
        engine_info["backend"] = backend

        state.update(
            {
                "status": "loaded",
                "device": engine_info.get("device", "CPU"),
                "model": engine_info.get("model_name", "unknown"),
                "backend": backend,
                "requested_device": engine_info.get("requested_device"),
                "requested_model": engine_info.get("requested_model"),
                "resolved_model": engine_info.get("resolved_model", engine_info.get("model_name")),
                "dimensions": engine_info.get("dimensions"),
                "is_production": engine_info.get("is_production"),
                "is_fallback": engine_info.get("is_fallback"),
                "fallback_reason": engine_info.get("fallback_reason"),
                "fallback_mode": engine_info.get("fallback_mode"),
                "load_error": engine_info.get("load_error"),
                "model_path": engine_info.get("model_path"),
                "canonical_model_path": engine_info.get("canonical_model_path"),
                "repo_id": engine_info.get("repo_id"),
                "cache": _summarize_embedding_cache(engine_info),
                "downloaded": state["downloaded"],
                "message": None,
            }
        )
        return EmbeddingEngineHealthState.from_payload(state), engine_info
    except Exception:
        logger.exception("Failed to inspect loaded embedding engine")
        state["status"] = "error"
        state["message"] = "Failed to inspect loaded embedding engine."
        state["error"] = "embedding_engine_inspection_failed"
        return EmbeddingEngineHealthState.from_payload(state), None


def _summarize_health(
    *,
    llm_state: LLMEngineHealthState,
    embedding_state: EmbeddingEngineHealthState,
    device_error: str | None,
) -> tuple[str, list[str]]:
    """Return top-level health status and human-readable messages."""
    messages: list[str] = []

    for label, state in (("llm", llm_state), ("embedding", embedding_state)):
        message = state.message
        if isinstance(message, str) and message:
            messages.append(f"{label}: {message}")

    if device_error:
        messages.append("device probe error: device_enumeration_failed")
        return "degraded", messages

    if llm_state.status == "error" or embedding_state.status == "error":
        return "degraded", messages

    return "healthy", messages


def _readiness_reasons(
    *,
    llm_state: LLMEngineHealthState,
    embedding_state: EmbeddingEngineHealthState,
    device_error: str | None,
) -> list[str]:
    """Return reasons that the service is not ready to serve warmed traffic."""
    reasons: list[str] = []
    if device_error:
        reasons.append(f"OpenVINO device enumeration failed: {device_error}")

    if llm_state.status != "loaded":
        reasons.append(str(llm_state.message or "LLM model is not loaded."))

    if embedding_state.status != "loaded":
        reasons.append(str(embedding_state.message or "Embedding model is not loaded."))

    return reasons


@router.get("/health")
async def health() -> dict:
    """Check service health status and device availability.

    Provides a comprehensive observational summary for the NPU Proxy service,
    including device detection, engine status, and version information.
    This endpoint does **not** auto-load models or warm runtimes.

    Returns:
        dict: Health response containing service status and diagnostics.

    Response Fields:
        status (str): Overall service status. Values:
            - ``'healthy'``: Service is fully operational
            - ``'degraded'``: Service running with reduced capability
            - ``'unhealthy'``: Service is not functional
        engines (dict): Status of inference engines:
            - ``llm``: LLM engine status object
            - ``embedding``: Embedding engine status object
        version (str): API version string (e.g., ``'0.2.0'``)
        npu_available (bool): Whether Intel NPU device is detected
        gpu_available (bool): Whether Intel GPU device is detected
        cpu_available (bool): Whether CPU is available (always True)
        devices (list[str]): List of all available OpenVINO devices
        openvino_version (str): OpenVINO runtime version

    Engine Status Object:
        status (str): Engine state. Values:
            - ``'loaded'``: Model loaded and ready for inference
            - ``'not_loaded'``: No model currently loaded
            - ``'fallback'``: Using fallback model (embedding only)
            - ``'error'``: Engine in error state
        device (str | None): Device running the model (e.g., ``'NPU'``, ``'CPU'``)
        model (str | None): Name/path of the loaded model

    Health Check Usage:
        - **Load Balancer**: Prefer ``/health/liveness`` for availability
        - **Readiness**: Prefer ``/health/readiness`` for warmed-model readiness
        - **Monitoring**: Inspect ``engines.*.message`` for not-loaded or error details

    Note:
        NPU Proxy runs as a native host service (Windows Service or systemd),
        not in containers. NPU hardware cannot be virtualized or containerized.

    Example:
        Response for an observational summary with no auto-load side effects::

            {
                "status": "healthy",
                "engines": {
                    "llm": {
                        "status": "loaded",
                        "device": "NPU",
                        "model": "Phi-3-mini-4k-instruct-int4-ov"
                    },
                    "embedding": {
                        "status": "loaded",
                        "device": "CPU",
                        "model": "all-MiniLM-L6-v2"
                    }
                },
                "version": "0.2.0",
                "npu_available": true,
                "gpu_available": true,
                "cpu_available": true,
                "devices": ["CPU", "GPU", "NPU"],
                "openvino_version": "2026.1.0"
            }
    """
    devices, device_error = _get_available_devices()
    llm_state, _ = _get_llm_engine_state()
    device_pool = _get_device_pool_snapshot()
    embedding_state, _ = _get_embedding_engine_state()
    status, messages = _summarize_health(
        llm_state=llm_state,
        embedding_state=embedding_state,
        device_error=device_error,
    )

    return {
        "status": status,
        "engines": {
            "llm": llm_state.to_dict(),
            "embedding": embedding_state.to_dict(),
        },
        "runtime": {
            "device_pool": device_pool,
        },
        "messages": messages,
        "version": __version__,
        # Maintain backward compatibility
        "npu_available": "NPU" in devices,
        "gpu_available": "GPU" in devices,
        "cpu_available": "CPU" in devices,
        "devices": devices,
        "openvino_version": _get_openvino_version(),
        "device_probe_error": device_error,
    }


@router.get("/health/liveness")
async def liveness() -> dict:
    """Return a cheap, non-mutating process liveness signal."""
    return {
        "status": "alive",
        "alive": True,
        "version": __version__,
    }


@router.get("/health/readiness")
async def readiness() -> JSONResponse:
    """Return whether warmed runtime state is ready to serve traffic."""
    devices, device_error = _get_available_devices()
    llm_state, _ = _get_llm_engine_state()
    embedding_state, _ = _get_embedding_engine_state()
    reasons = _readiness_reasons(
        llm_state=llm_state,
        embedding_state=embedding_state,
        device_error=device_error,
    )
    ready = not reasons

    payload = {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "reasons": reasons,
        "engines": {
            "llm": llm_state.to_dict(),
            "embedding": embedding_state.to_dict(),
        },
        "devices": devices,
        "version": __version__,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@router.get("/health/devices")
async def get_devices() -> dict:
    """Get detailed device information and fallback chain status.

    Provides comprehensive device diagnostics including all available
    OpenVINO devices, the currently active inference device, and the
    device fallback priority chain.

    Returns:
        dict: Device information and status.

    Response Fields:
        available_devices (list[str]): All OpenVINO-compatible devices
            detected on the system (e.g., ``['CPU', 'GPU', 'NPU']``).
        active_device (str | None): The device currently running inference,
            or None if no model is loaded.
        device_info (dict | None): Detailed info about the active device:
            - ``actual_device``: Device identifier string
            - ``device_name``: Human-readable device name
            - Additional device-specific properties
        fallback_chain (list[str]): Priority order for device selection
            when loading models (``['NPU', 'GPU', 'CPU']``).

    Use Cases:
        - Debugging device detection issues
        - Verifying NPU/GPU is being utilized
        - Monitoring device fallback behavior

    Example:
        Response when LLM is running on NPU::

            {
                "available_devices": ["CPU", "GPU", "NPU"],
                "active_device": "NPU",
                "device_info": {
                    "actual_device": "NPU",
                    "device_name": "Intel(R) AI Boost"
                },
                "fallback_chain": ["NPU", "GPU", "CPU"]
            }
    """
    from npu_proxy.inference.engine import get_available_devices

    devices = get_available_devices()
    llm_state, device_info = _get_llm_engine_state()
    device_pool = _get_device_pool_snapshot()
    embedding_state, embedding_device_info = _get_embedding_engine_state()

    return {
        "available_devices": devices,
        "active_device": llm_state.device,
        "device_info": device_info,
        "fallback_chain": DEVICE_FALLBACK_CHAIN,
        "device_pool": device_pool,
        "active_backend": llm_state.backend,
        "llm": {
            "status": llm_state.status,
            "active_device": llm_state.device,
            "backend": llm_state.backend,
            "model": llm_state.model,
            "device_info": device_info,
            "runtime_state": llm_state.runtime_state_payload(),
        },
        "embedding": {
            "status": embedding_state.status,
            "active_device": embedding_state.device,
            "backend": embedding_state.backend,
            "model": embedding_state.model,
            "device_info": embedding_device_info,
            "runtime_state": embedding_state.runtime_state_payload(),
        },
    }
