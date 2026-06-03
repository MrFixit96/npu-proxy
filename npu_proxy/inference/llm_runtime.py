"""Backend-neutral facade for LLM inference runtimes."""

from __future__ import annotations

import threading
from typing import Callable, Mapping

from npu_proxy.config import (
    LLMBackend,
    LLMRuntimeConfig,
    get_active_llm_runtime_config,
)
from npu_proxy.inference.backends.base import BaseLLMBackend

BackendFactory = Callable[[LLMRuntimeConfig], BaseLLMBackend]

# Test hooks/backward-compatible module attributes. They intentionally do not
# import backend modules; factories below import lazily if these are unset.
OpenVINOBackend = None
LlamaCppBackend = None


def _create_openvino_backend(config: LLMRuntimeConfig) -> BaseLLMBackend:
    backend_cls = OpenVINOBackend
    if backend_cls is None:
        from npu_proxy.inference.backends.openvino_backend import OpenVINOBackend as backend_cls

    return backend_cls(config)


def _create_llama_cpp_backend(config: LLMRuntimeConfig) -> BaseLLMBackend:
    backend_cls = LlamaCppBackend
    if backend_cls is None:
        from npu_proxy.inference.backends.llama_cpp_backend import LlamaCppBackend as backend_cls

    return backend_cls(config)


_RUNTIME_FACTORIES: dict[LLMBackend, BackendFactory] = {
    LLMBackend.OPENVINO: _create_openvino_backend,
    LLMBackend.LLAMA_CPP: _create_llama_cpp_backend,
}

_llm_runtime: "LLMRuntime | None" = None
_runtime_lock = threading.Lock()


class LLMRuntime:
    """Facade that hides the concrete LLM backend implementation."""

    def __init__(
        self,
        config: LLMRuntimeConfig | None = None,
        *,
        backend_factories: Mapping[LLMBackend, BackendFactory] | None = None,
    ) -> None:
        self.config = config or get_active_llm_runtime_config()
        self._backend_factories = dict(_RUNTIME_FACTORIES)
        if backend_factories is not None:
            self._backend_factories.update(backend_factories)
        self._backend = self._create_backend(self.config)

    @property
    def backend(self) -> BaseLLMBackend:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.backend.value

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @property
    def requested_device(self) -> str:
        return self._backend.requested_device

    @property
    def actual_device(self) -> str:
        return self._backend.actual_device

    @property
    def is_warmed_up(self) -> bool:
        return self._backend.is_warmed_up

    def warmup(self, warmup_tokens: int = 16) -> None:
        self._backend.warmup(warmup_tokens=warmup_tokens)

    def get_device_info(self) -> dict[str, object]:
        info = dict(self._backend.get_device_info())
        info.setdefault("backend", self.backend_name)
        return info

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int | None = None,
    ) -> str:
        return self._backend.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        streamer_callback=None,
        abort_callback=None,
        timeout: int | None = None,
    ):
        return self._backend.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            streamer_callback=streamer_callback,
            abort_callback=abort_callback,
            timeout=timeout,
        )

    def shutdown(self) -> None:
        """Release resources owned by the active backend."""
        self._backend.shutdown()

    def close(self) -> None:
        """Alias for shutdown()."""
        self.shutdown()

    def _create_backend(self, config: LLMRuntimeConfig) -> BaseLLMBackend:
        try:
            backend_factory = self._backend_factories[config.backend]
        except KeyError as exc:
            raise ValueError(f"No backend factory registered for {config.backend.value}") from exc
        return backend_factory(config)



def get_llm_runtime(config: LLMRuntimeConfig | None = None) -> LLMRuntime:
    """Get or create the singleton LLM runtime instance."""
    global _llm_runtime

    if _llm_runtime is None:
        with _runtime_lock:
            if _llm_runtime is None:
                _llm_runtime = LLMRuntime(config=config)

    return _llm_runtime



def reset_llm_runtime() -> None:
    """Reset the singleton LLM runtime instance and close its backend."""
    global _llm_runtime
    with _runtime_lock:
        runtime = _llm_runtime
        _llm_runtime = None
    if runtime is not None:
        runtime.shutdown()
