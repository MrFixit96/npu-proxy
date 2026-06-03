"""Focused tests for tokenizer-backed accounting."""

import pytest
from httpx import ASGITransport, AsyncClient

from npu_proxy.inference.chat_templates import RenderedChatPrompt
from npu_proxy.inference.tokenizer import (
    TokenCountPrecision,
    TokenCountResult,
    count_tokens,
    count_tokens_best_effort,
)
from npu_proxy.main import app


class FakeTokenizer:
    """Minimal tokenizer stub for exact counting tests."""

    def __call__(self, text, add_special_tokens=False, **kwargs):
        assert add_special_tokens is False
        return {"input_ids": [101, 102, 103, 104]}


def test_count_tokens_best_effort_uses_model_tokenizer(monkeypatch):
    """Best-effort counting should use exact tokenizer counts when possible."""

    import npu_proxy.inference.tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module, "get_model_tokenizer", lambda model: FakeTokenizer())

    result = count_tokens_best_effort("Hello there", model="tinyllama-1.1b-chat-int4-ov")

    assert result.count == 4
    assert result.achieved_precision == TokenCountPrecision.EXACT
    assert result.exact is True
    assert result.tokenizer_backend == "transformers"


def test_count_tokens_best_effort_falls_back_when_tokenizer_missing(monkeypatch):
    """Best-effort counting should fall back without raising."""

    import npu_proxy.inference.tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module, "get_model_tokenizer", lambda model: None)
    text = "Hello, world!"

    result = count_tokens_best_effort(text, model="tinyllama-1.1b-chat-int4-ov")

    assert result.count == count_tokens(text)
    assert result.achieved_precision == TokenCountPrecision.APPROXIMATE
    assert result.fallback_reason


@pytest.mark.asyncio
async def test_chat_usage_uses_lightweight_mock_accounting(monkeypatch):
    """Mock-mode chat usage should use lightweight prompt and token accounting."""

    import npu_proxy.api.chat as chat_api

    prompt_tokens = 17
    completion_tokens = 11
    response_text = "Hello! I'm a helpful AI assistant running on Intel NPU via OpenVINO."

    monkeypatch.setattr(chat_api, "format_chat_prompt", lambda messages, model=None: "<mock prompt>")
    monkeypatch.setattr(
        chat_api,
        "count_tokens",
        lambda text, precision=TokenCountPrecision.APPROXIMATE, model=None: {
            "<mock prompt>": prompt_tokens,
            response_text: completion_tokens,
        }[text],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    data = response.json()
    assert response.status_code == 200
    assert data["usage"]["prompt_tokens"] == prompt_tokens
    assert data["usage"]["completion_tokens"] == completion_tokens
    assert data["usage"]["total_tokens"] == prompt_tokens + completion_tokens
    assert response.headers["X-NPU-Proxy-Token-Count"] == str(prompt_tokens)
