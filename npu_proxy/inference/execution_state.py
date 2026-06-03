"""Helpers for reporting execution state without forcing backend imports."""

from __future__ import annotations

import logging
import sys

from npu_proxy.config import get_active_llm_runtime_config

logger = logging.getLogger(__name__)


def get_reportable_execution_device(*, load_if_needed: bool) -> str:
    """Return the truthful execution device without eagerly importing backends."""
    runtime_module = sys.modules.get("npu_proxy.inference.llm_runtime")
    runtime = getattr(runtime_module, "_llm_runtime", None) if runtime_module else None
    if runtime is not None:
        try:
            return str(runtime.actual_device or "unknown")
        except Exception as exc:
            logger.warning("Unable to read LLMRuntime execution device: %s", exc)
            return "unknown"

    if load_if_needed or "npu_proxy.inference.engine" in sys.modules:
        try:
            from npu_proxy.inference.engine import get_llm_execution_target

            execution_target = get_llm_execution_target(load_if_needed=load_if_needed)
            device = str(execution_target.get("device") or "unknown")
            if not execution_target.get("loaded"):
                logger.info("Reporting configured execution device %s before engine load", device)
            return device
        except Exception as exc:
            logger.warning("Unable to read legacy engine execution target: %s", exc)
            return "unknown"

    try:
        config = get_active_llm_runtime_config()
        if config.backend.value == "mock":
            return "mock"
        logger.info("Runtime not loaded; reporting configured execution device %s", config.device)
        return str(config.device or "unknown")
    except Exception as exc:
        logger.warning("Unable to read runtime configuration for execution device: %s", exc)
        return "unknown"
