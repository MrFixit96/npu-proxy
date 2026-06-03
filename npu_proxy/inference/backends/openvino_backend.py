"""Backend adapter for the existing OpenVINO inference engine."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Callable, Iterator, Protocol

from npu_proxy.config import LLMBackend, LLMRuntimeConfig
from npu_proxy.inference.backends.base import BaseLLMBackend

logger = logging.getLogger(__name__)


class OpenVINOEngineProtocol(Protocol):
    """Structural protocol for the wrapped OpenVINO engine."""

    model_name: str
    requested_device: str
    actual_device: str

    @property
    def is_warmed_up(self) -> bool: ...

    def warmup(self, warmup_tokens: int = 16) -> None: ...

    def get_device_info(self) -> dict[str, object]: ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int | None = None,
    ) -> str: ...

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        streamer_callback=None,
        abort_callback=None,
        timeout: int | None = None,
    ) -> Iterator[str]: ...


class OpenVINOBackend(BaseLLMBackend):
    """Backend-neutral adapter around :class:`InferenceEngine`."""

    backend = LLMBackend.OPENVINO

    def __init__(
        self,
        config: LLMRuntimeConfig,
        engine_factory: Callable[..., OpenVINOEngineProtocol] | None = None,
    ) -> None:
        self._config = config
        if engine_factory is None:
            from npu_proxy.inference.engine import InferenceEngine

            engine_factory = InferenceEngine
        if not config.model_path.exists():
            from npu_proxy.inference.engine import ModelNotFoundError

            raise ModelNotFoundError(config.model_path)
        self._engine = self._create_engine(engine_factory)

    @property
    def model_name(self) -> str:
        return self._engine.model_name

    @property
    def requested_device(self) -> str:
        return self._engine.requested_device

    @property
    def actual_device(self) -> str:
        return self._engine.actual_device

    @property
    def is_warmed_up(self) -> bool:
        return bool(self._engine.is_warmed_up)

    def warmup(self, warmup_tokens: int = 16) -> None:
        self._engine.warmup(warmup_tokens=warmup_tokens)

    def get_device_info(self) -> dict[str, object]:
        info = dict(self._engine.get_device_info())
        last_finish_reason = getattr(self._engine, "last_finish_reason", None)
        if last_finish_reason is not None:
            info["last_finish_reason"] = last_finish_reason
        info["model_path"] = str(self._config.model_path)
        return info

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int | None = None,
    ) -> str:
        resolved_timeout = self._config.inference_timeout if timeout is None else timeout
        return self._engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=resolved_timeout,
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
    ) -> Iterator[str]:
        resolved_timeout = self._config.inference_timeout if timeout is None else timeout
        return self._engine.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            streamer_callback=streamer_callback,
            abort_callback=abort_callback,
            timeout=resolved_timeout,
        )

    def shutdown(self) -> None:
        shutdown = getattr(self._engine, "shutdown", None)
        if callable(shutdown):
            shutdown(wait=False)

    def _create_engine(
        self,
        engine_factory: Callable[..., OpenVINOEngineProtocol],
    ) -> OpenVINOEngineProtocol:
        raw_kwargs = {
            "inference_timeout": self._config.inference_timeout,
            "max_prompt_len": self._config.max_prompt_len,
            "compile_cache_dir": self._config.compile_cache_dir,
            "compile_cache_mode": self._config.compile_cache_mode,
            "prefix_cache_mode": self._config.prefix_cache_mode,
        }
        supported_kwargs = _filter_supported_kwargs(engine_factory, raw_kwargs)
        return engine_factory(
            self._config.model_path,
            self._config.device,
            **supported_kwargs,
        )



def _filter_supported_kwargs(
    engine_factory: Callable[..., OpenVINOEngineProtocol],
    raw_kwargs: dict[str, object],
) -> dict[str, object]:
    try:
        signature = inspect.signature(engine_factory)
    except (TypeError, ValueError) as exc:
        if raw_kwargs:
            logger.warning(
                "Could not inspect OpenVINO engine factory %r; skipping runtime options %s: %s",
                engine_factory,
                sorted(raw_kwargs),
                exc,
            )
        return {}

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return raw_kwargs

    return {
        name: value
        for name, value in raw_kwargs.items()
        if name in signature.parameters
    }
