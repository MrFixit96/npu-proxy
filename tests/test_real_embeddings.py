"""Tests for real embeddings - TDD P1-1"""
import pytest
import os


# 🔴 RED: These tests should FAIL initially

@pytest.mark.asyncio
async def test_embeddings_returns_correct_dimensions():
    """Embeddings should return 384 dimensions for MiniLM model"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "all-minilm-l6-v2",
                "input": "Hello world",
            },
        )
    
    data = response.json()
    embedding = data["data"][0]["embedding"]
    
    assert len(embedding) == 384


@pytest.mark.asyncio
async def test_embeddings_are_deterministic():
    """Same input should produce same embedding"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response1 = await client.post(
            "/v1/embeddings",
            json={"model": "all-minilm-l6-v2", "input": "test input"},
        )
        response2 = await client.post(
            "/v1/embeddings",
            json={"model": "all-minilm-l6-v2", "input": "test input"},
        )
    
    emb1 = response1.json()["data"][0]["embedding"]
    emb2 = response2.json()["data"][0]["embedding"]
    
    assert emb1 == emb2


@pytest.mark.asyncio
async def test_embeddings_different_for_different_inputs():
    """Different inputs should produce different embeddings"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response1 = await client.post(
            "/v1/embeddings",
            json={"model": "all-minilm-l6-v2", "input": "Hello"},
        )
        response2 = await client.post(
            "/v1/embeddings",
            json={"model": "all-minilm-l6-v2", "input": "Goodbye"},
        )
    
    emb1 = response1.json()["data"][0]["embedding"]
    emb2 = response2.json()["data"][0]["embedding"]
    
    assert emb1 != emb2


@pytest.mark.asyncio
async def test_embeddings_batch_returns_multiple():
    """Batch embeddings should return one embedding per input"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "all-minilm-l6-v2",
                "input": ["Hello", "World", "Test"],
            },
        )
    
    data = response.json()
    
    assert len(data["data"]) == 3
    for i, item in enumerate(data["data"]):
        assert item["index"] == i
        assert len(item["embedding"]) == 384


def test_embedding_engine_loads():
    """EmbeddingEngine should load without errors"""
    from npu_proxy.inference.embedding_engine import EmbeddingEngine
    
    # Should not raise
    engine = EmbeddingEngine()
    assert engine is not None


def test_embedding_engine_generates_vectors():
    """EmbeddingEngine should generate embedding vectors"""
    from npu_proxy.inference.embedding_engine import EmbeddingEngine
    
    engine = EmbeddingEngine()
    embedding = engine.embed("Hello world")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


def test_embedding_engine_batch():
    """EmbeddingEngine should handle batch inputs"""
    from npu_proxy.inference.embedding_engine import EmbeddingEngine
    
    engine = EmbeddingEngine()
    embeddings = engine.embed_batch(["Hello", "World"])
    
    assert len(embeddings) == 2
    assert all(len(e) == 384 for e in embeddings)
