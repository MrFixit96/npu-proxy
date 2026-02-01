"""Tests for Ollama-native embedding endpoints (/api/embed and /api/embeddings)."""
import pytest
from fastapi.testclient import TestClient
from npu_proxy.main import app

client = TestClient(app)


class TestOllamaEmbedEndpoint:
    """Tests for POST /api/embed (Ollama current format)."""

    def test_embed_single_input(self):
        """POST /api/embed with single input returns nested embeddings array."""
        response = client.post("/api/embed", json={
            "model": "bge-small",
            "input": "test text"
        })
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 384

    def test_embed_multiple_inputs(self):
        """POST /api/embed with multiple inputs returns multiple embeddings."""
        response = client.post("/api/embed", json={
            "model": "bge-small",
            "input": ["text1", "text2", "text3"]
        })
        assert response.status_code == 200
        assert len(response.json()["embeddings"]) == 3

    def test_embed_returns_duration_stats(self):
        """POST /api/embed returns timing statistics."""
        response = client.post("/api/embed", json={
            "model": "bge-small",
            "input": "test"
        })
        assert response.status_code == 200
        data = response.json()
        assert "total_duration" in data
        assert "prompt_eval_count" in data

    def test_embed_model_in_response(self):
        """POST /api/embed includes model name in response."""
        response = client.post("/api/embed", json={
            "model": "bge-small",
            "input": "test"
        })
        assert response.status_code == 200
        assert response.json()["model"] == "bge-small"


class TestOllamaEmbeddingsLegacyEndpoint:
    """Tests for POST /api/embeddings (Ollama legacy format)."""

    def test_embeddings_legacy_single_prompt(self):
        """POST /api/embeddings with prompt returns single flat embedding."""
        response = client.post("/api/embeddings", json={
            "model": "bge-small",
            "prompt": "test text"
        })
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data  # NOT "embeddings"
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) == 384

    def test_embeddings_legacy_returns_single_array(self):
        """POST /api/embeddings returns flat array, not nested."""
        response = client.post("/api/embeddings", json={
            "model": "bge-small",
            "prompt": "single prompt only"
        })
        assert response.status_code == 200
        # Verify it's a flat array, not nested
        embedding = response.json()["embedding"]
        assert isinstance(embedding[0], float)  # First element is float, not list
