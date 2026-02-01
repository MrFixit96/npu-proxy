"""
Integration tests for real OpenVINO embedding inference.

These tests use the actual downloaded model and verify end-to-end embedding functionality.
"""

import math
import pytest

from npu_proxy.inference.embedding_engine import (
    ProductionEmbeddingEngine,
    is_embedding_model_downloaded,
    get_embedding_model_path,
    DEFAULT_EMBEDDING_MODEL,
)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# Skip tests if the production model is not downloaded
skip_if_no_model = pytest.mark.skipif(
    not is_embedding_model_downloaded(DEFAULT_EMBEDDING_MODEL),
    reason="Production embedding model not downloaded",
)


@pytest.mark.slow
@skip_if_no_model
def test_production_engine_loads_real_model():
    """Test that ProductionEmbeddingEngine loads the real model correctly."""
    model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
    engine = ProductionEmbeddingEngine(str(model_path))

    info = engine.get_engine_info()
    assert info["is_production"] is True, "Engine should be in production mode"


@pytest.mark.slow
@skip_if_no_model
def test_real_embedding_dimensions():
    """Test that embeddings have the correct dimensions (384 for BGE-small)."""
    model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
    engine = ProductionEmbeddingEngine(str(model_path))

    embedding = engine.embed("Hello world")
    assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"


@pytest.mark.slow
@skip_if_no_model
def test_real_embedding_deterministic():
    """Test that embedding the same text twice produces identical results."""
    model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
    engine = ProductionEmbeddingEngine(str(model_path))

    text = "The quick brown fox jumps over the lazy dog"
    embedding1 = engine.embed(text)
    embedding2 = engine.embed(text)

    assert embedding1 == embedding2, "Embeddings should be deterministic"


@pytest.mark.slow
@skip_if_no_model
def test_real_batch_embedding():
    """Test batch embedding with multiple texts."""
    model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
    engine = ProductionEmbeddingEngine(str(model_path))

    texts = [
        "The first text for embedding",
        "The second text for embedding",
        "The third text for embedding",
    ]

    embeddings = engine.embed_batch(texts)

    assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"

    for i, embedding in enumerate(embeddings):
        assert (
            len(embedding) == 384
        ), f"Embedding {i} should have 384 dimensions, got {len(embedding)}"


@pytest.mark.slow
@skip_if_no_model
def test_semantic_similarity():
    """Test semantic similarity between related and unrelated texts."""
    model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
    engine = ProductionEmbeddingEngine(str(model_path))

    # Semantically similar texts
    text1 = "The cat sat on the mat"
    text2 = "A feline rested on the rug"

    embedding1 = engine.embed(text1)
    embedding2 = engine.embed(text2)

    similarity_similar = cosine_similarity(embedding1, embedding2)
    assert (
        similarity_similar > 0.7
    ), f"Similar texts should have similarity > 0.7, got {similarity_similar}"

    # Semantically unrelated text
    text3 = "Python programming language"

    embedding3 = engine.embed(text3)
    similarity_unrelated = cosine_similarity(embedding1, embedding3)

    assert (
        similarity_unrelated < 0.5
    ), f"Unrelated texts should have similarity < 0.5, got {similarity_unrelated}"
