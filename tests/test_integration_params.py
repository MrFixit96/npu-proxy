"""Integration tests for parameter handling in API endpoints."""
import pytest
from fastapi.testclient import TestClient
from npu_proxy.main import app

client = TestClient(app)


class TestChatParameterIntegration:
    def test_chat_accepts_temperature(self):
        """Verify /api/chat accepts temperature parameter without error."""
        response = client.post("/api/chat", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {"temperature": 0.5}
        })
        # Should not error on the parameter (may error on model, that's ok)
        assert response.status_code in [200, 404]  # 404 if model not found

    def test_chat_accepts_mirostat_without_error(self):
        """Verify /api/chat accepts unsupported mirostat without error."""
        response = client.post("/api/chat", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {"mirostat": 1, "mirostat_tau": 5.0}
        })
        assert response.status_code in [200, 404]

    def test_chat_accepts_unknown_param_without_error(self):
        """Verify /api/chat handles unknown params gracefully."""
        response = client.post("/api/chat", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {"some_future_param": 123}
        })
        assert response.status_code in [200, 404]


class TestGenerateParameterIntegration:
    def test_generate_accepts_temperature(self):
        """Verify /api/generate accepts temperature parameter."""
        response = client.post("/api/generate", json={
            "model": "test-model",
            "prompt": "Hello",
            "options": {"temperature": 0.7}
        })
        assert response.status_code in [200, 404]

    def test_generate_accepts_num_predict(self):
        """Verify /api/generate accepts num_predict parameter."""
        response = client.post("/api/generate", json={
            "model": "test-model",
            "prompt": "Hello",
            "options": {"num_predict": 50}
        })
        assert response.status_code in [200, 404]


class TestOpenAIParameterIntegration:
    def test_openai_accepts_temperature(self):
        """Verify /v1/chat/completions accepts temperature."""
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5
        })
        assert response.status_code in [200, 404]

    def test_openai_accepts_max_tokens(self):
        """Verify /v1/chat/completions accepts max_tokens."""
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100
        })
        assert response.status_code in [200, 404]

    def test_openai_accepts_presence_penalty(self):
        """Verify /v1/chat/completions accepts presence_penalty."""
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "presence_penalty": 0.5
        })
        assert response.status_code in [200, 404]
