"""Alpha-gated llama.cpp backend scaffold for local GGUF models."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from npu_proxy.config import LLMBackend, LLMRuntimeConfig
from npu_proxy.inference.backends.base import (
    AbortCallback,
    BackendConfigurationError,
    BackendDependencyError,
    BaseLLMBackend,
    LLMBackendError,
    TokenCallback,
)
from npu_proxy.inference.engine import InferenceTimeoutError

_SUPPORTED_ALPHA_DEVICES = {"AUTO", "CPU"}
FinishReason = Literal["stop", "length"]


class LlamaCppBackend(BaseLLMBackend):
    """Backend scaffold for optional llama.cpp GGUF execution.

    Timeout handling is cooperative: deadlines are checked between emitted
    chunks/tokens, not by forcibly interrupting llama.cpp mid-call.
    """

    backend = LLMBackend.LLAMA_CPP

    def __init__(
        self,
        config: LLMRuntimeConfig,
        llama_factory: Callable[..., Any] | None = None,
        dependency_loader: Callable[[], Callable[..., Any]] | None = None,
    ) -> None:
        self._config = config
        self._model_path = self._validate_config(config)
        self._model_lock = threading.Lock()
        self._requested_device = config.device
        self._actual_device = "CPU"
        self._is_warmed_up = False
        self._last_finish_reason: FinishReason | None = None

        factory = llama_factory
        if factory is None:
            loader = dependency_loader or _load_llama_factory
            factory = loader()

        try:
            self._llama = factory(
                model_path=str(self._model_path),
                n_ctx=config.max_prompt_len,
                verbose=False,
            )
        except TypeError:
            self._llama = factory(model_path=str(self._model_path))

    @property
    def model_name(self) -> str:
        return self._model_path.stem

    @property
    def requested_device(self) -> str:
        return self._requested_device

    @property
    def actual_device(self) -> str:
        return self._actual_device

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def warmup(self, warmup_tokens: int = 16) -> None:
        self.generate("Hello", max_new_tokens=warmup_tokens, temperature=0.0, top_p=1.0)
        self._is_warmed_up = True

    def get_device_info(self) -> dict[str, object]:
        return {
            "requested_device": self._requested_device,
            "actual_device": self._actual_device,
            "fallback_device": None,
            "used_fallback": False,
            "available_devices": ["CPU"],
            "is_warmed_up": self._is_warmed_up,
            "model_path": str(self._model_path),
            "model_format": "gguf",
            "alpha_backend": True,
            "last_finish_reason": self._last_finish_reason,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int | None = None,
    ) -> str:
        with self._model_lock:
            if timeout is None:
                response = self._create_completion(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                )
                self._last_finish_reason = _extract_finish_reason(response)
                return _extract_text(response)

            chunks: list[str] = []
            for token in self._iterate_tokens(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
            ):
                chunks.append(token)
            if self._last_finish_reason is None:
                self._last_finish_reason = "length" if len(chunks) >= max_new_tokens else "stop"
            return "".join(chunks)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        streamer_callback: TokenCallback | None = None,
        abort_callback: AbortCallback | None = None,
        timeout: int | None = None,
    ) -> Iterator[str]:
        with self._model_lock:
            yield from self._iterate_tokens(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                streamer_callback=streamer_callback,
                abort_callback=abort_callback,
                timeout=timeout,
            )

    def _create_completion(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> Any:
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": max(temperature, 0.0),
            "top_p": top_p,
            "stream": stream,
        }

        if hasattr(self._llama, "create_completion"):
            return self._llama.create_completion(**kwargs)
        return self._llama(**kwargs)

    def _iterate_tokens(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        streamer_callback: TokenCallback | None = None,
        abort_callback: AbortCallback | None = None,
        timeout: int | None = None,
    ) -> Iterator[str]:
        """Iterate llama.cpp output with cooperative deadline checks."""
        deadline = None if timeout is None else time.monotonic() + timeout
        emitted_tokens = 0
        native_finish_reason: FinishReason | None = None
        stream = iter(
            self._create_completion(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
        )

        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise InferenceTimeoutError(timeout)

                try:
                    chunk = next(stream)
                except StopIteration:
                    break

                native_finish_reason = _extract_finish_reason(chunk) or native_finish_reason
                token = _extract_text(chunk)
                if not token:
                    continue

                should_abort = False
                if abort_callback is not None and abort_callback():
                    should_abort = True
                if streamer_callback is not None and streamer_callback(token):
                    should_abort = True

                emitted_tokens += 1
                yield token

                if should_abort:
                    break
        finally:
            self._last_finish_reason = (
                native_finish_reason
                or ("length" if max_new_tokens > 0 and emitted_tokens >= max_new_tokens else "stop")
            )
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def shutdown(self) -> None:
        with self._model_lock:
            llama = getattr(self, "_llama", None)
            close = getattr(llama, "close", None)
            if callable(close):
                close()
            self._llama = None

    @staticmethod
    def _validate_config(config: LLMRuntimeConfig) -> Path:
        if not config.enable_alpha_backends:
            raise BackendConfigurationError(
                "llama.cpp backend is alpha-gated. Set "
                "NPU_PROXY_ENABLE_ALPHA_BACKENDS=1 and "
                "NPU_PROXY_LLM_BACKEND=llama_cpp to opt in."
            )

        if config.device not in _SUPPORTED_ALPHA_DEVICES:
            supported = ", ".join(sorted(_SUPPORTED_ALPHA_DEVICES))
            raise BackendConfigurationError(
                "llama.cpp alpha currently supports CPU-only execution. "
                f"Set NPU_PROXY_DEVICE to one of: {supported}."
            )

        model_path = config.backend_model_path().expanduser().resolve(strict=False)
        if not model_path.is_file():
            raise BackendConfigurationError(
                f"llama.cpp GGUF model file not found at {model_path}."
            )

        if model_path.suffix.lower() != ".gguf":
            raise BackendConfigurationError(
                "llama.cpp backend requires a local .gguf model file."
            )
        return model_path



def _load_llama_factory() -> Callable[..., Any]:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise BackendDependencyError(
            "llama.cpp backend requires the optional 'llama-cpp-python' package."
        ) from exc
    return Llama



def _normalize_finish_reason(reason: Any) -> FinishReason | None:
    if reason is None:
        return None
    normalized = str(reason).strip().lower()
    if any(marker in normalized for marker in ("length", "max", "limit")):
        return "length"
    if any(marker in normalized for marker in ("stop", "eos", "end")):
        return "stop"
    return None


def _extract_finish_reason(chunk: Any) -> FinishReason | None:
    if not isinstance(chunk, dict):
        return None
    reason = _normalize_finish_reason(chunk.get("finish_reason") or chunk.get("stop_reason"))
    if reason is not None:
        return reason
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return _normalize_finish_reason(
            choices[0].get("finish_reason") or choices[0].get("stop_reason")
        )
    return None


def _extract_text(chunk: Any) -> str:
    if not isinstance(chunk, dict):
        raise LLMBackendError(
            f"llama.cpp returned unsupported completion shape: {_redacted_shape(chunk)}"
        )

    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMBackendError(
            f"llama.cpp returned unsupported completion shape: {_redacted_shape(chunk)}"
        )

    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMBackendError(
            f"llama.cpp returned unsupported completion shape: {_redacted_shape(chunk)}"
        )

    text = choice.get("text")
    if isinstance(text, str):
        return text

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    raise LLMBackendError(
        f"llama.cpp returned unsupported completion shape: {_redacted_shape(chunk)}"
    )


def _redacted_shape(value: Any) -> str:
    if isinstance(value, dict):
        parts = [f"keys={sorted(str(key) for key in value.keys())}"]
        choices = value.get("choices")
        if isinstance(choices, list):
            parts.append(f"choices=list[{len(choices)}]")
            if choices and isinstance(choices[0], dict):
                parts.append(f"choice_keys={sorted(str(key) for key in choices[0].keys())}")
        return f"dict({', '.join(parts)})"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__
