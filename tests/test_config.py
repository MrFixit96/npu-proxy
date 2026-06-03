from __future__ import annotations

import logging

import pytest

from npu_proxy.config import (
    ENV_COMPILE_CACHE_DIR,
    ENV_COMPILE_CACHE_MODE,
    ENV_DEVICE,
    ENV_ENABLE_ALPHA_BACKENDS,
    ENV_FALLBACK_DEVICE,
    ENV_HOST,
    ENV_INFERENCE_TIMEOUT,
    ENV_LLAMACPP_MODEL_PATH,
    ENV_LLM_BACKEND,
    ENV_MAX_PROMPT_LEN,
    ENV_PORT,
    ENV_PREFIX_CACHE_MODE,
    ENV_PREFERRED_DEVICE,
    ENV_REAL_INFERENCE,
    ENV_TOKEN_LIMIT,
    LLMBackend,
    LLMRuntimeConfig,
    LLMRuntimeConfigError,
    ProxyBootstrapConfig,
    activate_proxy_bootstrap_config,
    apply_proxy_bootstrap_config_to_env,
    get_active_llm_runtime_config,
    get_active_proxy_bootstrap_config,
    load_cli_environment_defaults,
    load_context_routing_config,
    load_llm_runtime_config,
    load_proxy_bootstrap_config,
    normalize_cli_device,
    normalize_compile_cache_mode,
    normalize_prefix_cache_mode,
    reset_active_proxy_bootstrap_config,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_bool_env_values_are_accepted(raw: str, expected: bool) -> None:
    config = load_llm_runtime_config(env={ENV_ENABLE_ALPHA_BACKENDS: raw})

    assert config.enable_alpha_backends is expected


def test_invalid_bool_env_raises_for_startup_config() -> None:
    with pytest.raises(LLMRuntimeConfigError, match="NPU_PROXY_REAL_INFERENCE must be a boolean-like value"):
        load_proxy_bootstrap_config(env={ENV_REAL_INFERENCE: "maybe"})


@pytest.mark.parametrize(
    ("key", "loader", "message"),
    [
        (ENV_PORT, load_cli_environment_defaults, "NPU_PROXY_PORT must be an integer, got 'abc'"),
        (ENV_TOKEN_LIMIT, load_proxy_bootstrap_config, "NPU_PROXY_TOKEN_LIMIT must be an integer, got 'abc'"),
        (ENV_INFERENCE_TIMEOUT, load_llm_runtime_config, "NPU_PROXY_INFERENCE_TIMEOUT must be an integer, got 'abc'"),
        (ENV_MAX_PROMPT_LEN, load_llm_runtime_config, "NPU_PROXY_MAX_PROMPT_LEN must be an integer, got 'abc'"),
    ],
)
def test_invalid_integer_env_raises_on_strict_startup_paths(key: str, loader, message: str) -> None:
    with pytest.raises(LLMRuntimeConfigError, match=message):
        loader(env={key: "abc"})


def test_invalid_log_level_raises_with_allowed_values() -> None:
    with pytest.raises(LLMRuntimeConfigError, match="log_level must be one of .* got 'trace'"):
        load_proxy_bootstrap_config(env={}, log_level="trace")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("optimize-size", "OPTIMIZE_SIZE"),
        ("OPTIMIZE_SPEED", "OPTIMIZE_SPEED"),
    ],
)
def test_compile_cache_mode_normalizes_valid_values(raw: str, expected: str) -> None:
    assert normalize_compile_cache_mode(raw) == expected


def test_compile_cache_mode_strict_raises_and_non_strict_warns(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(LLMRuntimeConfigError, match="compile_cache_mode must be one of"):
        normalize_compile_cache_mode("fastest")

    caplog.set_level(logging.WARNING, logger="npu_proxy.config")
    assert normalize_compile_cache_mode("fastest", strict=False) is None
    assert "Ignoring invalid NPU_PROXY_COMPILE_CACHE_MODE=fastest" in caplog.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("AUTO", "auto"), ("On", "on"), ("off", "off")],
)
def test_prefix_cache_mode_normalizes_valid_values(raw: str, expected: str) -> None:
    assert normalize_prefix_cache_mode(raw) == expected


def test_prefix_cache_mode_strict_raises_and_non_strict_warns(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(LLMRuntimeConfigError, match="prefix_cache_mode must be one of"):
        normalize_prefix_cache_mode("enabled")

    caplog.set_level(logging.WARNING, logger="npu_proxy.config")
    assert normalize_prefix_cache_mode("enabled", strict=False) == "auto"
    assert "Ignoring invalid NPU_PROXY_PREFIX_CACHE_MODE=enabled, using auto" in caplog.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("auto", "AUTO"), ("npu", "NPU"), ("gpu", "GPU"), ("cpu", "CPU")],
)
def test_cli_device_normalizes_valid_values(raw: str, expected: str) -> None:
    assert normalize_cli_device(raw) == expected


def test_cli_device_strict_raises_and_non_strict_warns(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(LLMRuntimeConfigError, match="device must be one of"):
        normalize_cli_device("tpu")

    caplog.set_level(logging.WARNING, logger="npu_proxy.config")
    assert normalize_cli_device("tpu", strict=False) == "AUTO"
    assert "Ignoring invalid NPU_PROXY_DEVICE=tpu, using AUTO" in caplog.text


def test_cli_environment_defaults_forgive_invalid_cache_and_device_env(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="npu_proxy.config")

    defaults = load_cli_environment_defaults(
        env={
            ENV_DEVICE: "tpu",
            ENV_COMPILE_CACHE_MODE: "fastest",
            ENV_PREFIX_CACHE_MODE: "enabled",
        }
    )

    assert defaults.device == "AUTO"
    assert defaults.compile_cache_mode is None
    assert defaults.prefix_cache_mode == "auto"
    assert "Ignoring invalid NPU_PROXY_DEVICE=tpu" in caplog.text
    assert "Ignoring invalid NPU_PROXY_COMPILE_CACHE_MODE=fastest" in caplog.text
    assert "Ignoring invalid NPU_PROXY_PREFIX_CACHE_MODE=enabled" in caplog.text


def test_context_routing_config_falls_back_with_warnings(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="npu_proxy.config")

    config = load_context_routing_config(
        env={
            ENV_TOKEN_LIMIT: "not-an-int",
            ENV_PREFERRED_DEVICE: "tpu",
            ENV_FALLBACK_DEVICE: "asic",
        }
    )

    assert config.token_limit == 1800
    assert config.preferred_device == "NPU"
    assert config.fallback_device == "CPU"
    assert "Invalid NPU_PROXY_TOKEN_LIMIT=not-an-int, using default 1800" in caplog.text
    assert "Ignoring invalid NPU_PROXY_PREFERRED_DEVICE=tpu, using NPU" in caplog.text
    assert "Ignoring invalid NPU_PROXY_FALLBACK_DEVICE=asic, using CPU" in caplog.text


def test_active_config_lazy_env_overlay_falls_back_with_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    activate_proxy_bootstrap_config(load_proxy_bootstrap_config(env={}, token_limit=1234, device="CPU"))
    monkeypatch.setenv(ENV_TOKEN_LIMIT, "bad")
    monkeypatch.setenv(ENV_DEVICE, "tpu")
    caplog.set_level(logging.WARNING, logger="npu_proxy.config")

    active = get_active_proxy_bootstrap_config()

    assert active.token_limit == 1234
    assert active.llm.device == "CPU"
    assert "Invalid NPU_PROXY_TOKEN_LIMIT=bad, using default 1234" in caplog.text
    assert "Ignoring invalid NPU_PROXY_DEVICE=tpu, using CPU" in caplog.text


def test_apply_proxy_bootstrap_config_to_env_removes_stale_optional_values(tmp_path) -> None:
    env = {
        ENV_COMPILE_CACHE_DIR: "stale-cache",
        ENV_COMPILE_CACHE_MODE: "OPTIMIZE_SPEED",
        ENV_LLAMACPP_MODEL_PATH: "stale.gguf",
    }
    llama_path = tmp_path / "model.gguf"
    llama_path.write_text("gguf")
    configured = load_proxy_bootstrap_config(
        env={},
        compile_cache_dir=tmp_path / "cache",
        compile_cache_mode="OPTIMIZE_SIZE",
        backend="llama.cpp",
        enable_alpha_backends=True,
        llama_cpp_model_path=llama_path,
    )

    apply_proxy_bootstrap_config_to_env(configured, env)
    assert env[ENV_COMPILE_CACHE_DIR] == str(tmp_path / "cache")
    assert env[ENV_COMPILE_CACHE_MODE] == "OPTIMIZE_SIZE"
    assert env[ENV_LLAMACPP_MODEL_PATH] == str(llama_path)

    apply_proxy_bootstrap_config_to_env(load_proxy_bootstrap_config(env={}), env)

    assert ENV_COMPILE_CACHE_DIR not in env
    assert ENV_COMPILE_CACHE_MODE not in env
    assert ENV_LLAMACPP_MODEL_PATH not in env
    assert env[ENV_HOST] == "127.0.0.1"
    assert env[ENV_PORT] == "8080"
    assert env[ENV_LLM_BACKEND] == "openvino"


def test_active_config_activation_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_proxy_bootstrap_config(env={}, device="CPU", token_limit=99)

    activated = activate_proxy_bootstrap_config(config)
    assert activated is config
    assert get_active_proxy_bootstrap_config() is config
    assert get_active_llm_runtime_config().device == "CPU"

    reset_active_proxy_bootstrap_config()
    monkeypatch.setenv(ENV_DEVICE, "GPU")

    assert get_active_llm_runtime_config().device == "GPU"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"inference_timeout": 0}, "inference_timeout must be greater than zero"),
        ({"inference_timeout": -1}, "inference_timeout must be greater than zero"),
        ({"max_prompt_len": 0}, "max_prompt_len must be greater than zero"),
        ({"max_prompt_len": -1}, "max_prompt_len must be greater than zero"),
    ],
)
def test_runtime_config_rejects_non_positive_limits(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(LLMRuntimeConfigError, match=message):
        LLMRuntimeConfig(**kwargs)


def test_missing_llama_cpp_model_path_raises_when_backend_path_is_requested() -> None:
    config = load_llm_runtime_config(
        env={
            ENV_LLM_BACKEND: "llama.cpp",
            ENV_ENABLE_ALPHA_BACKENDS: "1",
        }
    )

    assert config.backend is LLMBackend.LLAMA_CPP
    with pytest.raises(LLMRuntimeConfigError, match="llama.cpp backend requires NPU_PROXY_LLAMACPP_MODEL_PATH"):
        config.backend_model_path()
