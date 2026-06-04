from __future__ import annotations

from collections.abc import Mapping

from npu_proxy.config import LLMBackend, LLMRuntimeConfig
from npu_proxy.inference.backends.base import BaseLLMBackend
from npu_proxy.inference.llm_runtime import LLMRuntime, get_llm_runtime, reset_llm_runtime
from npu_proxy.inference.runtime_pool import LLMRuntimePool, reset_runtime_pool


class FakeRuntime:
    def __init__(
        self,
        config: LLMRuntimeConfig,
        *,
        backend_factories: Mapping[LLMBackend, object] | None = None,
        fail_shutdown: bool = False,
    ) -> None:
        self.config = config
        self.backend_factories = backend_factories
        self.fail_shutdown = fail_shutdown
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError(f"shutdown failed for {self.config.device}")


class FakeRuntimeBackend(BaseLLMBackend):
    backend = LLMBackend.OPENVINO

    def __init__(self, config: LLMRuntimeConfig) -> None:
        self._config = config
        self.shutdown_calls = 0

    @property
    def model_name(self) -> str:
        return "runtime-model"

    @property
    def requested_device(self) -> str:
        return self._config.device

    @property
    def actual_device(self) -> str:
        return self._config.device

    @property
    def is_warmed_up(self) -> bool:
        return True

    def get_device_info(self) -> dict[str, object]:
        return {"actual_device": self.actual_device}

    def generate(self, **kwargs) -> str:
        return f"runtime:{kwargs['prompt']}"

    def generate_stream(self, **kwargs):
        yield from ["r", "t"]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_pool_lazily_creates_and_caches_runtime_per_device() -> None:
    created: list[str] = []

    def factory(config: LLMRuntimeConfig, **kwargs) -> FakeRuntime:
        created.append(config.device)
        return FakeRuntime(config, **kwargs)

    pool = LLMRuntimePool(runtime_factory=factory)

    first = pool.get_runtime("CPU")
    second = pool.get_runtime("CPU")

    assert first is second
    assert first.config.device == "CPU"
    assert created == ["CPU"]


def test_pool_returns_different_runtimes_for_different_devices() -> None:
    pool = LLMRuntimePool(runtime_factory=FakeRuntime)

    npu_runtime = pool.get_runtime("NPU")
    gpu_runtime = pool.get_runtime("GPU")

    assert npu_runtime is not gpu_runtime
    assert npu_runtime.config.device == "NPU"
    assert gpu_runtime.config.device == "GPU"


def test_pool_normalizes_device_names() -> None:
    pool = LLMRuntimePool(runtime_factory=FakeRuntime)

    runtime = pool.get_runtime("npu")

    assert runtime.config.device == "NPU"
    assert pool.loaded_devices() == ["NPU"]


def test_loaded_devices_reflects_requested_runtimes() -> None:
    pool = LLMRuntimePool(runtime_factory=FakeRuntime)

    assert pool.loaded_devices() == []

    pool.get_runtime("cpu")
    pool.get_runtime("gpu")

    assert pool.loaded_devices() == ["CPU", "GPU"]


def test_shutdown_closes_all_runtimes_and_clears_even_when_one_raises() -> None:
    runtimes: dict[str, FakeRuntime] = {}

    def factory(config: LLMRuntimeConfig, **kwargs) -> FakeRuntime:
        runtime = FakeRuntime(
            config,
            fail_shutdown=config.device == "GPU",
            **kwargs,
        )
        runtimes[config.device] = runtime
        return runtime

    pool = LLMRuntimePool(runtime_factory=factory)
    pool.get_runtime("CPU")
    pool.get_runtime("GPU")
    pool.get_runtime("NPU")

    pool.shutdown()

    assert runtimes["CPU"].shutdown_calls == 1
    assert runtimes["GPU"].shutdown_calls == 1
    assert runtimes["NPU"].shutdown_calls == 1
    assert pool.loaded_devices() == []


def test_pool_passes_backend_factories_to_runtime_factory() -> None:
    backend_factories = {LLMBackend.OPENVINO: FakeRuntimeBackend}
    pool = LLMRuntimePool(
        backend_factories=backend_factories,
        runtime_factory=FakeRuntime,
    )

    runtime = pool.get_runtime("CPU")

    assert runtime.backend_factories is backend_factories


def test_get_llm_runtime_with_device_returns_pool_runtime(monkeypatch) -> None:
    import npu_proxy.inference.runtime_pool as pool_module

    pool = LLMRuntimePool(runtime_factory=FakeRuntime)
    monkeypatch.setattr(pool_module, "_runtime_pool", pool)

    runtime = get_llm_runtime(device="CPU")

    assert isinstance(runtime, FakeRuntime)
    assert runtime.config.device == "CPU"
    assert runtime is pool.get_runtime("cpu")


def test_get_llm_runtime_without_device_preserves_legacy_singleton(monkeypatch, tmp_path) -> None:
    import npu_proxy.inference.llm_runtime as runtime_module

    reset_llm_runtime()
    reset_runtime_pool()
    monkeypatch.setattr(
        runtime_module,
        "_RUNTIME_FACTORIES",
        {
            LLMBackend.OPENVINO: FakeRuntimeBackend,
            LLMBackend.LLAMA_CPP: FakeRuntimeBackend,
        },
    )
    config = LLMRuntimeConfig(model_path=tmp_path / "model")

    first = get_llm_runtime(config=config)
    second = get_llm_runtime()

    assert isinstance(first, LLMRuntime)
    assert first is second
    assert first.requested_device == "NPU"
