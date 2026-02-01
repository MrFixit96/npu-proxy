"""Integration tests for OpenAI-compatible /v1/embeddings endpoint."""
import pytest
from fastapi.testclient import TestClient
from npu_proxy.main import app

client = TestClient(app)


class TestEmbeddingsEndpoint:
    def test_embeddings_returns_correct_dimensions(self):
        """POST /v1/embeddings returns embeddings with 384 dimensions"""
        response = client.post("/v1/embeddings", json={
            "model": "bge-small",
            "input": "test text"
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"][0]["embedding"]) == 384

    def test_embeddings_batch_input(self):
        """POST /v1/embeddings handles batch input"""
        response = client.post("/v1/embeddings", json={
            "model": "bge-small",
            "input": ["text1", "text2", "text3"]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        for item in data["data"]:
            assert len(item["embedding"]) == 384

    def test_embeddings_usage_reports_tokens(self):
        """POST /v1/embeddings reports token usage"""
        response = client.post("/v1/embeddings", json={
            "model": "test",
            "input": "hello world"
        })
        assert response.status_code == 200
        data = response.json()
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0
