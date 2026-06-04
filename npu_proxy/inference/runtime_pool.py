"""Thread-safe pool of per-device LLM runtimes."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace

from npu_proxy.config import LLMBackend, LLMRuntimeConfig, get_active_llm_runtime_config
from npu_proxy.inference.llm_runtime import BackendFactory, LLMRuntime

logger = logging.getLogger(__name__)

RuntimeFactory = Callable[..., LLMRuntime]

_runtime_pool: "LLMRuntimePool | None" = None
_pool_lock = threading.Lock()


class LLMRuntimePool:
    """Lazily create and cache one LLM runtime per requested device."""

    def __init__(
        self,
        *,
        backend_factories: Mapping[LLMBackend, BackendFactory] | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self._backend_factories = backend_factories
        self._runtime_factory = runtime_factory or LLMRuntime
        self._runtimes: dict[str, LLMRuntime] = {}
        self._lock = threading.Lock()

    def get_runtime(self, device: str) -> LLMRuntime:
        """Get or create the runtime for a normalized device key."""
        device_key = self._normalize_device(device)
        runtime = self._runtimes.get(device_key)
        if runtime is not None:
            return runtime

        with self._lock:
            runtime = self._runtimes.get(device_key)
            if runtime is None:
                config = replace(get_active_llm_runtime_config(), device=device_key)
                kwargs = {}
                if self._backend_factories is not None:
                    kwargs["backend_factories"] = self._backend_factories
                runtime = self._runtime_factory(config, **kwargs)
                self._runtimes[device_key] = runtime
        return runtime

    def loaded_devices(self) -> list[str]:
        """Return device keys for runtimes that have been instantiated."""
        with self._lock:
            return list(self._runtimes.keys())

    def shutdown(self) -> None:
        """Shutdown all loaded runtimes and clear the pool."""
        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()

        for runtime in runtimes:
            try:
                runtime.shutdown()
            except Exception:
                logger.exception("Failed to shutdown LLM runtime")

    @staticmethod
    def _normalize_device(device: str) -> str:
        device_key = str(device).strip().upper()
        if not device_key:
            raise ValueError("device must not be empty")
        return device_key


def get_runtime_pool() -> LLMRuntimePool:
    """Get or create the process-wide LLM runtime pool."""
    global _runtime_pool

    if _runtime_pool is None:
        with _pool_lock:
            if _runtime_pool is None:
                _runtime_pool = LLMRuntimePool()

    return _runtime_pool


def reset_runtime_pool() -> None:
    """Reset and shutdown the process-wide LLM runtime pool."""
    global _runtime_pool
    with _pool_lock:
        pool = _runtime_pool
        _runtime_pool = None
    if pool is not None:
        pool.shutdown()
