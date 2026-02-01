"""Tests for embeddings endpoint - TDD Phase 1.5"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_embeddings_returns_200():
    """POST /v1/embeddings should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "all-minilm-l6-v2",
                "input": "Hello world",
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_embeddings_returns_openai_format():
    """Response should match OpenAI embeddings format"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "all-minilm-l6-v2",
                "input": "Test embedding",
            },
        )
    data = response.json()
    
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert "model" in data
    assert "usage" in data
    
    assert len(data["data"]) > 0
    embedding = data["data"][0]
    assert "object" in embedding
    assert embedding["object"] == "embedding"
    assert "embedding" in embedding
    assert "index" in embedding
    assert isinstance(embedding["embedding"], list)


@pytest.mark.asyncio
async def test_embeddings_batch_input():
    """Should handle batch input (list of strings)"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "all-minilm-l6-v2",
                "input": ["First text", "Second text", "Third text"],
            },
        )
    data = response.json()
    
    assert response.status_code == 200
    assert len(data["data"]) == 3
    
    for i, embedding in enumerate(data["data"]):
        assert embedding["index"] == i
        assert len(embedding["embedding"]) > 0
