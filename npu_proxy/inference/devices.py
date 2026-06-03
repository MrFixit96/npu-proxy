"""Shared device discovery helpers decoupled from backend engine imports."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Device priority for fallback (NPU -> GPU -> CPU)
DEVICE_FALLBACK_CHAIN: list[str] = ["NPU", "GPU", "CPU"]


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
