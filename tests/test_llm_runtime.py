from __future__ import annotations

from pathlib import Path

import pytest

from npu_proxy.config import LLMBackend, LLMRuntimeConfig, load_llm_runtime_config
from npu_proxy.inference.backends.base import (
    BackendConfigurationError,
    BackendDependencyError,
    BaseLLMBackend,
)
from npu_proxy.inference.backends.llama_cpp_backend import LlamaCppBackend
from npu_proxy.inference.backends.openvino_backend import OpenVINOBackend
from npu_proxy.inference.engine import ModelNotFoundError
from npu_proxy.inference.llm_runtime import LLMRuntime, get_llm_runtime, reset_llm_runtime


class FakeOpenVINOEngine:
    def __init__(self, model_path: str | Path, device: str, **kwargs) -> None:
        self.model_name = Path(model_path).name
        self.requested_device = device
        self.actual_device = "CPU"
        self.is_warmed_up = False
        self.init_kwargs = kwargs
        self.generate_calls: list[dict[str, object]] = []

    def warmup(self, warmup_tokens: int = 16) -> None:
        self.is_warmed_up = True
        self.generate_calls.append({"warmup_tokens": warmup_tokens})

    def get_device_info(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "used_fallback": True,
        }

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return "openvino-ok"

    def generate_stream(self, **kwargs):
        self.generate_calls.append(kwargs)
        yield from ["hello", " world"]


class FakeRuntimeBackend(BaseLLMBackend):
    backend = LLMBackend.OPENVINO

    def __init__(self, config: LLMRuntimeConfig) -> None:
        self._config = config

    @property
    def model_name(self) -> str:
        return "runtime-model"

    @property
    def requested_device(self) -> str:
        return self._config.device

    @property
    def actual_device(self) -> str:
        return "CPU"

    @property
    def is_warmed_up(self) -> bool:
        return True

    def get_device_info(self) -> dict[str, object]:
        return {"actual_device": "CPU"}

    def generate(self, **kwargs) -> str:
        return f"runtime:{kwargs['prompt']}"

    def generate_stream(self, **kwargs):
        yield from ["r", "t"]


class FakeLlama:
    def __init__(self, *args, **kwargs) -> None:
        self.init_kwargs = kwargs

    def create_completion(self, **kwargs):
        if kwargs["stream"]:
            return iter(
                [
                    {"choices": [{"text": "Hel"}]},
                    {"choices": [{"text": "lo"}]},
                ]
            )
        return {"choices": [{"text": "llama-ok"}]}


class AbortAfterFirstTokenLlama:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def create_completion(self, **kwargs):
        return iter(
            [
                {"choices": [{"text": "A"}]},
                {"choices": [{"text": "B"}]},
            ]
        )



def test_load_llm_runtime_config_defaults_to_openvino() -> None:
    config = load_llm_runtime_config(env={})

    assert config.backend is LLMBackend.OPENVINO
    assert config.device == "NPU"
    assert config.enable_alpha_backends is False
    assert config.prefix_cache_mode == "auto"



def test_load_llm_runtime_config_reads_openvino_cache_env(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    config = load_llm_runtime_config(
        env={
            "NPU_PROXY_COMPILE_CACHE_DIR": str(cache_dir),
            "NPU_PROXY_COMPILE_CACHE_MODE": "optimize-speed",
            "NPU_PROXY_PREFIX_CACHE_MODE": "on",
        }
    )

    assert config.compile_cache_dir == cache_dir
    assert config.compile_cache_mode == "OPTIMIZE_SPEED"
    assert config.prefix_cache_mode == "on"



def test_load_llm_runtime_config_reads_llama_cpp_env(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    config = load_llm_runtime_config(
        env={
            "NPU_PROXY_LLM_BACKEND": "llama.cpp",
            "NPU_PROXY_ENABLE_ALPHA_BACKENDS": "1",
            "NPU_PROXY_DEVICE": "cpu",
            "NPU_PROXY_LLAMACPP_MODEL_PATH": str(gguf_path),
        }
    )

    assert config.backend is LLMBackend.LLAMA_CPP
    assert config.enable_alpha_backends is True
    assert config.device == "CPU"
    assert config.backend_model_path() == gguf_path



def test_openvino_backend_wraps_existing_engine(tmp_path) -> None:
    model_path = tmp_path / "tinyllama"
    model_path.mkdir()
    cache_dir = tmp_path / "compile-cache"

    backend = OpenVINOBackend(
        LLMRuntimeConfig(
            model_path=model_path,
            inference_timeout=23,
            max_prompt_len=2048,
            compile_cache_dir=cache_dir,
            compile_cache_mode="OPTIMIZE_SPEED",
            prefix_cache_mode="on",
        ),
        engine_factory=FakeOpenVINOEngine,
    )

    assert backend.generate("hi") == "openvino-ok"
    assert list(backend.generate_stream("hi")) == ["hello", " world"]

    backend.warmup(8)
    info = backend.get_device_info()

    assert backend.is_warmed_up is True
    assert info["model_path"] == str(model_path)
    assert info["actual_device"] == "CPU"
    assert backend._engine.init_kwargs["inference_timeout"] == 23
    assert backend._engine.init_kwargs["max_prompt_len"] == 2048
    assert backend._engine.init_kwargs["compile_cache_dir"] == cache_dir
    assert backend._engine.init_kwargs["compile_cache_mode"] == "OPTIMIZE_SPEED"
    assert backend._engine.init_kwargs["prefix_cache_mode"] == "on"



def test_openvino_backend_requires_existing_model_path(tmp_path) -> None:
    missing_path = tmp_path / "missing-model"

    with pytest.raises(ModelNotFoundError):
        OpenVINOBackend(
            LLMRuntimeConfig(model_path=missing_path),
            engine_factory=FakeOpenVINOEngine,
        )



def test_llama_cpp_backend_requires_alpha_opt_in(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    with pytest.raises(BackendConfigurationError, match="alpha-gated"):
        LlamaCppBackend(
            LLMRuntimeConfig(
                backend=LLMBackend.LLAMA_CPP,
                device="CPU",
                llama_cpp_model_path=gguf_path,
            ),
            llama_factory=FakeLlama,
        )



def test_llama_cpp_backend_requires_cpu_device(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    with pytest.raises(BackendConfigurationError, match="CPU-only"):
        LlamaCppBackend(
            LLMRuntimeConfig(
                backend=LLMBackend.LLAMA_CPP,
                device="NPU",
                enable_alpha_backends=True,
                llama_cpp_model_path=gguf_path,
            ),
            llama_factory=FakeLlama,
        )



def test_llama_cpp_backend_requires_optional_dependency(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    def missing_loader():
        raise BackendDependencyError("missing llama.cpp dependency")

    with pytest.raises(BackendDependencyError, match="missing llama.cpp dependency"):
        LlamaCppBackend(
            LLMRuntimeConfig(
                backend=LLMBackend.LLAMA_CPP,
                device="CPU",
                enable_alpha_backends=True,
                llama_cpp_model_path=gguf_path,
            ),
            dependency_loader=missing_loader,
        )



def test_llama_cpp_backend_streams_with_fake_runtime(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    backend = LlamaCppBackend(
        LLMRuntimeConfig(
            backend=LLMBackend.LLAMA_CPP,
            device="CPU",
            enable_alpha_backends=True,
            llama_cpp_model_path=gguf_path,
            max_prompt_len=1024,
        ),
        llama_factory=FakeLlama,
    )

    assert backend.generate("Hello") == "llama-ok"
    assert list(backend.generate_stream("Hello")) == ["Hel", "lo"]

    backend.warmup(4)
    assert backend.is_warmed_up is True
    assert backend.get_device_info()["model_format"] == "gguf"



def test_llama_cpp_backend_yields_aborting_token(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")
    observed_tokens: list[str] = []

    backend = LlamaCppBackend(
        LLMRuntimeConfig(
            backend=LLMBackend.LLAMA_CPP,
            device="CPU",
            enable_alpha_backends=True,
            llama_cpp_model_path=gguf_path,
        ),
        llama_factory=AbortAfterFirstTokenLlama,
    )

    def stop_after_first(token: str) -> bool:
        observed_tokens.append(token)
        return True

    assert list(backend.generate_stream("Hello", streamer_callback=stop_after_first)) == ["A"]
    assert observed_tokens == ["A"]


def test_llama_cpp_backend_honors_generate_timeout(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    backend = LlamaCppBackend(
        LLMRuntimeConfig(
            backend=LLMBackend.LLAMA_CPP,
            device="CPU",
            enable_alpha_backends=True,
            llama_cpp_model_path=gguf_path,
        ),
        llama_factory=FakeLlama,
    )

    with pytest.raises(TimeoutError):
        backend.generate("Hello", timeout=0)


def test_llama_cpp_backend_honors_stream_timeout(tmp_path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    backend = LlamaCppBackend(
        LLMRuntimeConfig(
            backend=LLMBackend.LLAMA_CPP,
            device="CPU",
            enable_alpha_backends=True,
            llama_cpp_model_path=gguf_path,
        ),
        llama_factory=FakeLlama,
    )

    with pytest.raises(TimeoutError):
        list(backend.generate_stream("Hello", timeout=0))



def test_llm_runtime_uses_backend_factory(tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    runtime = LLMRuntime(
        LLMRuntimeConfig(model_path=model_path),
        backend_factories={LLMBackend.OPENVINO: FakeRuntimeBackend},
    )

    assert runtime.backend_name == "openvino"
    assert runtime.generate("ping") == "runtime:ping"
    assert list(runtime.generate_stream("ping")) == ["r", "t"]
    assert runtime.get_device_info()["backend"] == "openvino"



def test_get_llm_runtime_returns_singleton(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = LLMRuntimeConfig(model_path=model_path)

    import npu_proxy.inference.llm_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "OpenVINOBackend", FakeRuntimeBackend)
    monkeypatch.setattr(
        runtime_module,
        "_RUNTIME_FACTORIES",
        {
            LLMBackend.OPENVINO: FakeRuntimeBackend,
            LLMBackend.LLAMA_CPP: FakeRuntimeBackend,
        },
    )

    reset_llm_runtime()
    first = get_llm_runtime(config=config)
    second = get_llm_runtime()

    assert first is second
    assert first.generate("pong") == "runtime:pong"
