"""Shared device discovery helpers decoupled from backend engine imports."""

from __future__ import annotations

import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover - exercised only on Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal StrEnum backport: str() and format() yield the member value."""

        def __str__(self) -> str:
            return str(self.value)

        __format__ = str.__format__


class Device(StrEnum):
    """Canonical compute device identifiers used across routing and inference."""

    NPU = "NPU"
    GPU = "GPU"
    CPU = "CPU"


class FallbackReason(StrEnum):
    """Why an execution device differs from the routed device.

    The string values are part of the public wire/metric contract
    (``X-NPU-Proxy-Fallback-Reason`` header and the Prometheus
    ``fallback_reason`` label), so they must stay stable.
    """

    BUSY = "busy"
    DEVICE_FALLBACK = "device_fallback"


# Device priority for fallback (NPU -> GPU -> CPU)
DEVICE_FALLBACK_CHAIN: list[str] = [Device.NPU.value, Device.GPU.value, Device.CPU.value]

# Canonical accelerator identifiers accepted by routing/inference.
VALID_DEVICES: frozenset[str] = frozenset(member.value for member in Device)


def normalize_device(value: object, *, default: str | None = None) -> str | None:
    """Return the canonical upper-cased device string.

    This is the single place that canonicalizes a free-form device value
    (trimming whitespace and upper-casing) so that ``npu`` and ``NPU`` cannot
    create two distinct pool entries or locks. It does not validate membership;
    callers that need validation should compare against :data:`VALID_DEVICES`.

    Args:
        value: A device-like value (string or :class:`Device`).
        default: Returned when ``value`` is empty/None.
    """
    if value is None:
        return default
    normalized = str(value).strip().upper()
    if not normalized:
        return default
    return normalized


def device_class(value: object) -> str:
    """Collapse an enumerated accelerator id to its canonical device class.

    OpenVINO enumerates multiple accelerators of the same kind with a numeric
    suffix (e.g. ``GPU.0``, ``GPU.1``). Routing and availability decisions reason
    about the device *class* (``GPU``), and OpenVINO accepts the bare class name
    as an alias for the first instance when compiling. Returns ``""`` for empty
    input so callers can filter it out.
    """
    normalized = normalize_device(value) or ""
    return re.sub(r"\.\d+$", "", normalized)


def get_available_devices() -> list[str]:
    """Return available compute devices with a mock-safe CPU fallback.

    In the default mock mode, the service does not execute real inference, so it
    should not import the native OpenVINO runtime just to compute advisory
    routing defaults. Real device probing is reserved for real-inference mode.
    """
    if os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") != "1":
        return ["CPU"]

    try:
        import openvino as ov

        core = ov.Core()
        return list(core.available_devices)
    except Exception:
        logger.exception("Failed to enumerate OpenVINO devices in real-inference mode")
        return ["CPU"]
