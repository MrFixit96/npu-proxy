"""Shared pytest fixtures for npu-proxy tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

_ORIGINAL_NPU_PROXY_ENV = {
    key: value for key, value in os.environ.items() if key.startswith("NPU_PROXY_")
}


def _restore_npu_proxy_env() -> None:
    for key in [key for key in os.environ if key.startswith("NPU_PROXY_")]:
        if key not in _ORIGINAL_NPU_PROXY_ENV:
            os.environ.pop(key, None)
    os.environ.update(_ORIGINAL_NPU_PROXY_ENV)


def _call_reset_if_loaded(module_name: str, function_name: str, **kwargs: Any) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    reset = getattr(module, function_name, None)
    if not callable(reset):
        return
    try:
        reset(**kwargs)
    except Exception:
        pass


def _reset_loaded_global_state() -> None:
    _call_reset_if_loaded("npu_proxy.inference.llm_runtime", "reset_llm_runtime")
    _call_reset_if_loaded("npu_proxy.inference.engine", "reset_engine", force=True)
    _call_reset_if_loaded("npu_proxy.inference.embedding_engine", "_reset_embedding_engine")
    _call_reset_if_loaded("npu_proxy.routing.context_router", "reset_context_router")
    _call_reset_if_loaded("npu_proxy.config", "reset_active_proxy_bootstrap_config")
    _call_reset_if_loaded("npu_proxy.metrics", "reset_metrics")
    _call_reset_if_loaded("npu_proxy.metrics", "_reset_metrics")


@pytest.fixture(autouse=True)
def reset_process_test_state() -> Iterator[None]:
    """Restore mutable process-wide state around every test."""
    _restore_npu_proxy_env()
    _reset_loaded_global_state()
    yield
    _reset_loaded_global_state()
    _restore_npu_proxy_env()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Return a function-scoped synchronous FastAPI test client."""
    from npu_proxy.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> Iterator[httpx.AsyncClient]:
    """Return a function-scoped async FastAPI test client."""
    from npu_proxy.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


class FakeLLMEngine:
    """Deterministic test double for the current LLM engine interface."""

    def __init__(
        self,
        *,
        response_text: str,
        stream_tokens: Iterable[str] | None,
        finish_reason: str | None,
        model_name: str,
        actual_device: str,
        requested_device: str | None,
    ) -> None:
        self.response_text = response_text
        self.stream_tokens = list(stream_tokens) if stream_tokens is not None else None
        self._configured_finish_reason = finish_reason
        self.model_name = model_name
        self.actual_device = actual_device
        self.requested_device = requested_device or actual_device
        self.used_fallback = self.requested_device != self.actual_device
        self.last_finish_reason: str | None = None
        self.last_generation_stats: dict[str, Any] | None = None
        self.generate_calls: list[dict[str, Any]] = []
        self.generate_stream_calls: list[dict[str, Any]] = []
        self.shutdown_calls = 0

    @staticmethod
    def _words(text: str) -> list[str]:
        return text.split()

    def _finish_reason(self, token_count: int, max_new_tokens: int) -> str:
        if self._configured_finish_reason is not None:
            return self._configured_finish_reason
        return "length" if max_new_tokens > 0 and token_count >= max_new_tokens else "stop"

    def _completion(self, max_new_tokens: int) -> tuple[str, list[str], str]:
        tokens = self.stream_tokens if self.stream_tokens is not None else self._words(self.response_text)
        if max_new_tokens > 0:
            tokens = tokens[:max_new_tokens]
        text = " ".join(tokens) if self.stream_tokens is None else "".join(tokens)
        return text, list(tokens), self._finish_reason(len(tokens), max_new_tokens)

    def _record_stats(self, prompt: str, generated_tokens: int, finish_reason: str) -> None:
        self.last_finish_reason = finish_reason
        self.last_generation_stats = {
            "device": self.actual_device,
            "input_tokens": len(prompt.split()),
            "generated_tokens": generated_tokens,
            "finish_reason": finish_reason,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int | None = None,
    ) -> str:
        text, tokens, finish_reason = self._completion(max_new_tokens)
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "timeout": timeout,
            }
        )
        self._record_stats(prompt, len(tokens), finish_reason)
        return text

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        streamer_callback: Callable[[str], bool] | None = None,
        abort_callback: Callable[[], bool] | None = None,
        timeout: int | None = None,
    ) -> Iterator[str]:
        _, tokens, finish_reason = self._completion(max_new_tokens)
        self.generate_stream_calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "timeout": timeout,
            }
        )

        emitted: list[str] = []
        for token in tokens:
            if abort_callback is not None and abort_callback():
                finish_reason = "stop"
                break
            emitted.append(token)
            should_abort = streamer_callback(token) if streamer_callback is not None else False
            yield token
            if should_abort:
                finish_reason = "stop"
                break

        self._record_stats(prompt, len(emitted), finish_reason)

    def get_device_info(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "actual_device": self.actual_device,
            "requested_device": self.requested_device,
            "used_fallback": self.used_fallback,
            "loaded": True,
            "last_generation_stats": self.last_generation_stats,
            "last_finish_reason": self.last_finish_reason,
        }

    def has_active_inference(self) -> bool:
        return False

    def shutdown(self, wait: bool = False) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def fake_llm_engine_factory() -> Callable[..., FakeLLMEngine]:
    """Return a factory for deterministic fake LLM engines."""

    def factory(
        *,
        response_text: str = "Hello from the fake NPU proxy.",
        stream_tokens: Iterable[str] | None = None,
        finish_reason: str | None = None,
        model_name: str = "tinyllama",
        actual_device: str = "CPU",
        requested_device: str | None = None,
    ) -> FakeLLMEngine:
        return FakeLLMEngine(
            response_text=response_text,
            stream_tokens=stream_tokens,
            finish_reason=finish_reason,
            model_name=model_name,
            actual_device=actual_device,
            requested_device=requested_device,
        )

    return factory


@pytest.fixture
def known_ollama_model() -> str:
    """Return a known Ollama-style LLM model name."""
    return "tinyllama"


@pytest.fixture
def known_llm_model() -> str:
    """Return a known canonical LLM model identifier."""
    return "tinyllama"


@pytest.fixture
def known_huggingface_repo() -> str:
    """Return a known HuggingFace LLM repository."""
    return "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"


@pytest.fixture
def known_embedding_model() -> str:
    """Return a known Ollama-style embedding model name."""
    return "all-minilm"


@pytest.fixture
def known_embedding_repo() -> str:
    """Return a known HuggingFace embedding repository."""
    return "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def unknown_model() -> str:
    """Return an unknown model name."""
    return "nonexistent-model-xyz"
