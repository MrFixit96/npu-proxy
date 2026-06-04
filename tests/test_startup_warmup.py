from __future__ import annotations

from types import SimpleNamespace

from npu_proxy.config import ProxyBootstrapConfig, load_warmup_devices
from npu_proxy.main import _warmup_configured_devices


def test_load_warmup_devices_uses_forgiving_unique_uppercase_parse():
    devices = load_warmup_devices({"NPU_PROXY_WARMUP_DEVICES": "npu, bad, cpu, NPU, ,gpu"})

    assert devices == ("NPU", "CPU", "GPU")


def test_startup_warmup_is_noop_in_mock_mode(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        "npu_proxy.inference.engine.get_llm_engine",
        lambda device: calls.append(device),
    )

    _warmup_configured_devices(ProxyBootstrapConfig(real_inference=False, warmup_devices=("NPU",)))

    assert calls == []


def test_startup_warmup_loads_available_configured_devices(monkeypatch):
    warmed: list[str] = []

    def fake_get_llm_engine(device: str):
        return SimpleNamespace(warmup=lambda: warmed.append(device))

    monkeypatch.setattr("npu_proxy.inference.engine.get_available_devices", lambda: ["CPU", "NPU"])
    monkeypatch.setattr("npu_proxy.inference.engine.get_llm_engine", fake_get_llm_engine)

    _warmup_configured_devices(
        ProxyBootstrapConfig(real_inference=True, warmup_devices=("NPU", "GPU", "CPU"))
    )

    assert warmed == ["NPU", "CPU"]


def test_startup_warmup_continues_after_device_failure(monkeypatch):
    warmed: list[str] = []

    def fake_get_llm_engine(device: str):
        if device == "NPU":
            raise RuntimeError("compile failed")
        return SimpleNamespace(warmup=lambda: warmed.append(device))

    monkeypatch.setattr("npu_proxy.inference.engine.get_available_devices", lambda: ["CPU", "NPU"])
    monkeypatch.setattr("npu_proxy.inference.engine.get_llm_engine", fake_get_llm_engine)

    _warmup_configured_devices(ProxyBootstrapConfig(real_inference=True, warmup_devices=("NPU", "CPU")))

    assert warmed == ["CPU"]
