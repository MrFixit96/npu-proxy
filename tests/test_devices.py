"""Tests for mock-safe OpenVINO device discovery."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from npu_proxy.inference import devices as devices_api


class FakeOpenVINOModule:
    def __init__(self, available_devices=None, core_error: Exception | None = None):
        self._available_devices = list(available_devices or [])
        self._core_error = core_error

    def Core(self):
        if self._core_error is not None:
            raise self._core_error
        return SimpleNamespace(available_devices=self._available_devices)


def test_get_available_devices_mock_mode_returns_cpu_without_importing_openvino(monkeypatch):
    """Default mock mode should not touch the native OpenVINO runtime."""
    original_import = builtins.__import__

    def fail_openvino_import(name, *args, **kwargs):
        if name == "openvino":
            raise AssertionError("OpenVINO should not be imported in mock mode")
        return original_import(name, *args, **kwargs)

    monkeypatch.delenv("NPU_PROXY_REAL_INFERENCE", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_openvino_import)

    assert devices_api.get_available_devices() == ["CPU"]


@pytest.mark.parametrize(
    ("reported_devices", "expected"),
    [
        (["NPU", "GPU", "CPU"], ["NPU", "GPU", "CPU"]),
        (["GPU", "CPU"], ["GPU", "CPU"]),
        (["CPU"], ["CPU"]),
        (["NPU"], ["NPU"]),
    ],
)
def test_get_available_devices_real_inference_uses_openvino_core(monkeypatch, reported_devices, expected):
    """Real-inference mode should return the OpenVINO Core device list unchanged."""
    monkeypatch.setenv("NPU_PROXY_REAL_INFERENCE", "1")
    monkeypatch.setitem(sys.modules, "openvino", FakeOpenVINOModule(reported_devices))

    assert devices_api.get_available_devices() == expected


def test_get_available_devices_real_inference_import_failure_falls_back_to_cpu(monkeypatch, caplog):
    """Import failures should be logged and degrade to CPU-only discovery."""
    monkeypatch.setenv("NPU_PROXY_REAL_INFERENCE", "1")
    monkeypatch.setitem(sys.modules, "openvino", None)

    assert devices_api.get_available_devices() == ["CPU"]
    assert "Failed to enumerate OpenVINO devices" in caplog.text


def test_get_available_devices_real_inference_core_failure_falls_back_to_cpu(monkeypatch, caplog):
    """Core construction failures should be logged and degrade to CPU-only discovery."""
    monkeypatch.setenv("NPU_PROXY_REAL_INFERENCE", "1")
    monkeypatch.setitem(
        sys.modules,
        "openvino",
        FakeOpenVINOModule(["NPU"], core_error=RuntimeError("driver unavailable")),
    )

    assert devices_api.get_available_devices() == ["CPU"]
    assert "Failed to enumerate OpenVINO devices" in caplog.text
