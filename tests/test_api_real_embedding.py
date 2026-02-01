"""
API integration tests for embedding endpoints using the real model.

These tests verify that the embedding endpoints work correctly with the
actual embedding model and meet performance requirements.
"""

import time
import pytest
from httpx import AsyncClient, ASGITransport

from npu_proxy.main import app
from npu_proxy.inference.embedding_engine import (
    is_embedding_model_downloaded,
    DEFAULT_EMBEDDING_MODEL,
)


@pytest.fixture
def skip_if_model_not_downloaded():
    """Skip test if the embedding model is not downloaded."""
    if not is_embedding_model_downloaded(DEFAULT_EMBEDDING_MODEL):
        pytest.skip(
            f"Embedding model {DEFAULT_EMBEDDING_MODEL} not downloaded. "
            "Run 'python -m npu_proxy.inference.embedding_engine' to download."
        )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_openai_endpoint_real_model(skip_if_model_not_downloaded):
    """
    Test the OpenAI-compatible /v1/embeddings endpoint with the real model.

    Verifies:
    - Status code is 200
    - Response contains "data" list with at least 1 item
    - Each item has an "embedding" list with 384 float values
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "bge-small",
                "input": "Hello world",
            },
        )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) >= 1

    # Verify embedding structure
    first_item = data["data"][0]
    assert "embedding" in first_item
    assert isinstance(first_item["embedding"], list)
    assert len(first_item["embedding"]) == 384
    assert all(isinstance(val, float) for val in first_item["embedding"])


@pytest.mark.slow
@pytest.mark.asyncio
async def test_ollama_embed_real_model(skip_if_model_not_downloaded):
    """
    Test the Ollama-compatible /api/embed endpoint with the real model.

    Verifies:
    - Status code is 200
    - Response contains "embeddings" list with at least 1 item
    - Each embedding has 384 dimensions
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/embed",
            json={
                "model": "bge-small",
                "input": "Hello world",
            },
        )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) >= 1

    # Verify embedding dimensions
    first_embedding = data["embeddings"][0]
    assert isinstance(first_embedding, list)
    assert len(first_embedding) == 384
    assert all(isinstance(val, float) for val in first_embedding)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_embedding_performance(skip_if_model_not_downloaded):
    """
    Test that embedding generation meets performance requirements.

    Verifies:
    - Single embedding request completes within 500ms
    - This ensures production-grade latency requirements are met
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        test_text = "The quick brown fox jumps over the lazy dog"

        start_time = time.perf_counter()
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "bge-small",
                "input": test_text,
            },
        )
        elapsed_time = time.perf_counter() - start_time

    assert response.status_code == 200
    assert elapsed_time < 0.5, f"Embedding latency {elapsed_time:.3f}s exceeds 500ms limit"
