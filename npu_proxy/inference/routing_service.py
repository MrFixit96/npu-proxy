"""Shared routing-execution helpers used by the chat and Ollama API layers.

These helpers were previously duplicated verbatim in ``npu_proxy/api/chat.py``
and ``npu_proxy/api/ollama.py``. Centralizing them keeps the routing
truthfulness logic (which device actually executed, and why it differs from the
routed device) consistent across API surfaces. The API modules keep thin,
patchable wrappers that delegate here so existing test seams are preserved.
"""

from __future__ import annotations

import logging
from typing import Any

from npu_proxy.inference.devices import FallbackReason, normalize_device
from npu_proxy.metrics import record_routing_execution as _metrics_record_routing_execution

logger = logging.getLogger(__name__)


def execution_device_from_engine(engine: Any) -> str:
    """Return the actual device reported by an acquired engine."""
    get_device_info = getattr(engine, "get_device_info", None)
    if callable(get_device_info):
        try:
            info = get_device_info()
        except Exception:
            logger.warning(
                "Failed to read device info from engine; falling back to actual_device attribute",
                exc_info=True,
            )
            info = {}
        if isinstance(info, dict) and info.get("actual_device"):
            return str(info["actual_device"])
    return str(getattr(engine, "actual_device", None) or "unknown")


def fallback_reason(
    *,
    routed_device: str,
    execution_device: str,
    engine_slot: object | None = None,
) -> str | None:
    """Return why execution differs from routing, or None when they match.

    Returns None when the execution device is unknown/empty: an unresolved
    execution device is not evidence of a deliberate fallback, so reporting
    ``device_fallback`` there would be untruthful.
    """
    routed = normalize_device(routed_device) or ""
    executed = normalize_device(execution_device) or ""
    if not routed or executed == routed:
        return None
    # An explicit slot reason (e.g. a busy fallback) is authoritative even when
    # the execution device couldn't be resolved, so report it before suppressing.
    reason = getattr(engine_slot, "fallback_reason", None)
    if reason:
        return str(reason)
    if not executed or executed == "UNKNOWN":
        return None
    return FallbackReason.DEVICE_FALLBACK.value


def record_routing_execution(
    routed_device: str,
    execution_device: str,
    fallback_reason: str | None,
) -> None:
    """Record a routing-execution metric, normalizing an absent reason to 'none'."""
    _metrics_record_routing_execution(routed_device, execution_device, fallback_reason or "none")


def close_engine_slot(slot: object) -> None:
    """Release an acquired engine slot via its context-manager exit hook."""
    exit_method = getattr(slot, "__exit__")
    exit_method(None, None, None)
