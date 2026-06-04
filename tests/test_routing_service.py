"""Unit tests for the shared routing-execution service and device wire types."""

from __future__ import annotations

from npu_proxy.inference import routing_service
from npu_proxy.inference.devices import (
    Device,
    FallbackReason,
    normalize_device,
)


class _Slot:
    def __init__(self, fallback_reason=None):
        self.fallback_reason = fallback_reason


def test_device_and_fallback_wire_strings_are_stable():
    """These strings are part of the public header/metric contract."""
    assert str(Device.NPU) == "NPU"
    assert str(Device.GPU) == "GPU"
    assert str(Device.CPU) == "CPU"
    assert f"{FallbackReason.BUSY}" == "busy"
    assert f"{FallbackReason.DEVICE_FALLBACK}" == "device_fallback"
    assert FallbackReason.BUSY.value == "busy"
    assert FallbackReason.DEVICE_FALLBACK.value == "device_fallback"


def test_normalize_device_canonicalizes_and_defaults():
    assert normalize_device(" npu ") == "NPU"
    assert normalize_device("gpu") == "GPU"
    assert normalize_device(None) is None
    assert normalize_device("") is None
    assert normalize_device(None, default="CPU") == "CPU"


def test_fallback_reason_none_when_devices_match():
    assert routing_service.fallback_reason(routed_device="NPU", execution_device="NPU") is None
    assert routing_service.fallback_reason(routed_device="npu", execution_device="NPU") is None


def test_fallback_reason_busy_from_slot_metadata():
    slot = _Slot(fallback_reason="busy")
    reason = routing_service.fallback_reason(
        routed_device="NPU", execution_device="CPU", engine_slot=slot
    )
    assert reason == "busy"


def test_fallback_reason_defaults_to_device_fallback():
    reason = routing_service.fallback_reason(routed_device="NPU", execution_device="CPU")
    assert reason == "device_fallback"


def test_fallback_reason_suppressed_when_execution_unknown():
    """An unresolved execution device must not be reported as a deliberate fallback."""
    assert routing_service.fallback_reason(routed_device="NPU", execution_device="unknown") is None
    assert routing_service.fallback_reason(routed_device="NPU", execution_device="") is None


def test_fallback_reason_reports_slot_busy_even_when_execution_unknown():
    """A known busy fallback stays truthful even if the engine can't report its device."""
    slot = _Slot(fallback_reason="busy")
    reason = routing_service.fallback_reason(
        routed_device="NPU", execution_device="unknown", engine_slot=slot
    )
    assert reason == "busy"


def test_execution_device_from_engine_reads_device_info():
    class Engine:
        actual_device = "CPU"

        def get_device_info(self):
            return {"actual_device": "GPU.0"}

    assert routing_service.execution_device_from_engine(Engine()) == "GPU.0"


def test_execution_device_from_engine_falls_back_on_error():
    class Engine:
        actual_device = "CPU"

        def get_device_info(self):
            raise RuntimeError("boom")

    assert routing_service.execution_device_from_engine(Engine()) == "CPU"


def test_execution_device_from_engine_unknown_when_unavailable():
    class Engine:
        actual_device = None

    assert routing_service.execution_device_from_engine(Engine()) == "unknown"
