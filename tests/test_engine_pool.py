from __future__ import annotations

from pathlib import Path

import pytest

from npu_proxy.config import LLMRuntimeConfig
from npu_proxy.inference.engine import InferenceError, get_llm_engine, reset_engine


class FakeInferenceEngine:
    instances: list["FakeInferenceEngine"] = []

    def __init__(
        self,
        model_path: str | Path,
        device: str = "NPU",
        **kwargs,
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = device.upper()
        self.actual_device = self.requested_device
        self.model_name = self.model_path.name
        self.init_kwargs = kwargs
        self.shutdown_calls = 0
        self.active = False
        FakeInferenceEngine.instances.append(self)

    def get_device_info(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "used_fallback": False,
        }

    def has_active_inference(self) -> bool:
        return self.active

    def shutdown(self, wait: bool = False) -> None:
        self.shutdown_calls += 1
        self.shutdown_wait = wait


@pytest.fixture
def fake_engine(monkeypatch):
    import npu_proxy.inference.engine as engine_module

    FakeInferenceEngine.instances.clear()
    monkeypatch.setattr(engine_module, "InferenceEngine", FakeInferenceEngine)
    reset_engine(force=True)
    yield FakeInferenceEngine
    reset_engine(force=True)
    FakeInferenceEngine.instances.clear()


def test_get_llm_engine_pools_by_device(fake_engine, tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    cpu_engine = get_llm_engine(model_path=model_path, device="CPU")
    gpu_engine = get_llm_engine(model_path=model_path, device="GPU")

    assert cpu_engine is not gpu_engine
    assert cpu_engine.requested_device == "CPU"
    assert gpu_engine.requested_device == "GPU"


def test_get_llm_engine_returns_cached_instance_for_same_device(fake_engine, tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    first = get_llm_engine(model_path=model_path, device="cpu")
    second = get_llm_engine(model_path=model_path, device="CPU")

    assert first is second
    assert len(fake_engine.instances) == 1


def test_get_llm_engine_without_device_uses_configured_default(fake_engine, tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = LLMRuntimeConfig(model_path=model_path, device="GPU")

    engine = get_llm_engine(config=config)

    assert engine.requested_device == "GPU"
    assert get_llm_engine(config=config, device="gpu") is engine


def test_reset_engine_force_shuts_down_all_pooled_engines_and_clears(fake_engine, tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cpu_engine = get_llm_engine(model_path=model_path, device="CPU")
    gpu_engine = get_llm_engine(model_path=model_path, device="GPU")

    reset_engine(force=True)

    assert cpu_engine.shutdown_calls == 1
    assert gpu_engine.shutdown_calls == 1
    assert cpu_engine.shutdown_wait is False
    assert gpu_engine.shutdown_wait is False

    fresh_cpu_engine = get_llm_engine(model_path=model_path, device="CPU")
    assert fresh_cpu_engine is not cpu_engine


def test_reset_engine_without_force_raises_409_if_any_pooled_engine_is_active(
    fake_engine,
    tmp_path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cpu_engine = get_llm_engine(model_path=model_path, device="CPU")
    gpu_engine = get_llm_engine(model_path=model_path, device="GPU")
    gpu_engine.active = True

    with pytest.raises(InferenceError) as exc_info:
        reset_engine()

    assert exc_info.value.status_code == 409
    assert cpu_engine.shutdown_calls == 0
    assert gpu_engine.shutdown_calls == 0


def test_get_llm_execution_target_reports_loaded_default_engine(fake_engine, tmp_path) -> None:
    from npu_proxy.inference.engine import get_llm_execution_target

    model_path = tmp_path / "model"
    model_path.mkdir()
    config = LLMRuntimeConfig(model_path=model_path, device="CPU")
    get_llm_engine(config=config)

    target = get_llm_execution_target(load_if_needed=False)

    assert target["loaded"] is True
    assert target["model"] == "model"
    assert target["requested_device"] == "CPU"
