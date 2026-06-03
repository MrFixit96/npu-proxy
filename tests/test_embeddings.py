"""Tests for embeddings endpoint - TDD Phase 1.5"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_embeddings_returns_503_when_unavailable():
    """POST /v1/embeddings should hard-fail when the real model is unavailable."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "org-test/unavailable-openai-embed",
                "input": "Hello world",
            },
        )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_embeddings_returns_error_detail_when_unavailable():
    """Unavailable embeddings should use the structured error detail payload."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "org-test/unavailable-openai-embed-detail",
                "input": "Test embedding",
            },
        )
    data = response.json()

    assert response.status_code == 503
    assert "error" in data
    assert data["error"]["type"] == "service_unavailable_error"
    assert data["error"]["code"] == "embedding_unavailable"


@pytest.mark.asyncio
async def test_embeddings_batch_input_returns_503_when_unavailable():
    """Batch embedding requests should also fail closed when real embeddings are unavailable."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "org-test/unavailable-openai-embed-batch",
                "input": ["First text", "Second text", "Third text"],
            },
        )

    assert response.status_code == 503
