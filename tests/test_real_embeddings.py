"""Contract tests for truthful embedding availability and explicit fallback."""

import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from npu_proxy.main import app
from npu_proxy.inference.embedding_engine import (
    EMBEDDING_FALLBACK_MODE_ENV_VAR,
    EmbeddingUnavailableError,
    get_embedding_engine,
    _reset_embedding_engine,
)


@pytest.fixture(autouse=True)
def reset_embedding_engine():
    """Keep embedding engine cache isolated across fallback-policy tests."""
    _reset_embedding_engine()
    yield
    _reset_embedding_engine()


@pytest.mark.asyncio
async def test_embeddings_default_to_503_for_missing_model():
    """OpenAI embeddings should fail closed when no real model is available."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "org-test/unavailable-real-embed",
                "input": "Hello world",
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "embedding_unavailable"


@pytest.mark.asyncio
async def test_ollama_embed_defaults_to_503_for_missing_model():
    """Ollama embeddings should also fail closed when no real model is available."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/embed",
            json={
                "model": "org-test/unavailable-real-ollama-embed",
                "input": "Hello world",
            },
        )

    assert response.status_code == 503


def test_embedding_engine_raises_when_fallback_is_disabled():
    """Direct engine access should no longer silently synthesize embeddings by default."""
    with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "disabled"}):
        with pytest.raises(EmbeddingUnavailableError):
            get_embedding_engine(model_name="org-test/direct-missing-embed")


def test_embedding_engine_can_still_use_explicit_operator_fallback():
    """Explicit operator fallback remains available behind the env gate."""
    with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}):
        engine = get_embedding_engine(model_name="org-test/direct-missing-embed-fallback")
        embedding = engine.embed("Hello world")

    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_embedding_engine_batch_fallback_requires_explicit_opt_in():
    """Explicit fallback should still support batch behavior for operator workflows."""
    with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}):
        engine = get_embedding_engine(model_name="org-test/direct-missing-embed-batch")
        embeddings = engine.embed_batch(["Hello", "World"])

    assert len(embeddings) == 2
    assert all(len(embedding) == 384 for embedding in embeddings)
