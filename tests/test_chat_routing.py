"""Tests for chat endpoint routing integration."""
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with fresh router state."""
    from npu_proxy.routing.context_router import reset_context_router
    reset_context_router()
    from npu_proxy.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_router():
    """Reset context router before and after each test."""
    from npu_proxy.routing.context_router import reset_context_router
    reset_context_router()
    yield
    reset_context_router()


class TestChatEndpointRouting:
    """Tests for /v1/chat/completions routing."""
    
    def test_chat_endpoint_includes_routing_headers(self, client):
        """Chat endpoint should include X-NPU-Proxy headers."""
        response = client.post("/v1/chat/completions", json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hello!"}],
        })
        
        assert response.status_code == 200
        assert "X-NPU-Proxy-Device" in response.headers
        assert "X-NPU-Proxy-Route-Reason" in response.headers
        assert "X-NPU-Proxy-Token-Count" in response.headers
    
    def test_short_prompt_routes_to_npu(self, client):
        """Short prompts should route to NPU."""
        with patch.dict(os.environ, {"NPU_PROXY_TOKEN_LIMIT": "1800"}):
            from npu_proxy.routing.context_router import reset_context_router
            reset_context_router()
            
            response = client.post("/v1/chat/completions", json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello!"}],
            })
        
        assert response.status_code == 200
        assert response.headers.get("X-NPU-Proxy-Device") == "NPU"
        assert response.headers.get("X-NPU-Proxy-Route-Reason") == "within_npu_limit"
    
    def test_long_prompt_routes_to_cpu(self, client):
        """Long prompts should route to CPU."""
        # Create a very long prompt that exceeds token limit
        long_content = "word " * 2000
        
        with patch.dict(os.environ, {"NPU_PROXY_TOKEN_LIMIT": "1800"}):
            from npu_proxy.routing.context_router import reset_context_router
            reset_context_router()
            
            response = client.post("/v1/chat/completions", json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": long_content}],
            })
        
        assert response.status_code == 200
        assert response.headers.get("X-NPU-Proxy-Device") == "CPU"
        assert response.headers.get("X-NPU-Proxy-Route-Reason") == "prompt_exceeds_npu_limit"
    
    def test_token_count_header_is_integer(self, client):
        """X-NPU-Proxy-Token-Count should be a valid integer."""
        response = client.post("/v1/chat/completions", json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hello world!"}],
        })
        
        assert response.status_code == 200
        token_count = response.headers.get("X-NPU-Proxy-Token-Count")
        assert token_count is not None
        assert int(token_count) > 0


class TestOllamaEndpointRouting:
    """Tests for Ollama /api endpoints routing."""
    
    def test_generate_includes_routing_headers(self, client):
        """Ollama /api/generate should include routing headers."""
        response = client.post("/api/generate", json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "prompt": "Hello!",
            "stream": False,
        })
        
        assert response.status_code == 200
        assert "X-NPU-Proxy-Device" in response.headers
    
    def test_generate_long_prompt_routes_to_cpu(self, client):
        """Long prompts in /api/generate should route to CPU."""
        long_prompt = "word " * 2000
        
        with patch.dict(os.environ, {"NPU_PROXY_TOKEN_LIMIT": "1800"}):
            from npu_proxy.routing.context_router import reset_context_router
            reset_context_router()
            
            response = client.post("/api/generate", json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": long_prompt,
                "stream": False,
            })
        
        assert response.status_code == 200
        assert response.headers.get("X-NPU-Proxy-Device") == "CPU"
    
    def test_chat_api_includes_routing_headers(self, client):
        """Ollama /api/chat should include routing headers."""
        response = client.post("/api/chat", json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        })
        
        assert response.status_code == 200
        assert "X-NPU-Proxy-Device" in response.headers


class TestRoutingConfiguration:
    """Tests for routing configuration via environment variables."""
    
    def test_custom_token_limit(self):
        """Custom token limit should be respected."""
        # Set a very low limit before creating client
        with patch.dict(os.environ, {"NPU_PROXY_TOKEN_LIMIT": "10"}):
            from npu_proxy.routing.context_router import reset_context_router
            reset_context_router()
            from npu_proxy.main import app
            client = TestClient(app)
            
            # This message has >10 tokens so should exceed the limit
            response = client.post("/v1/chat/completions", json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "This is a message that should definitely have more than ten tokens in it."}],
            })
        
        assert response.status_code == 200
        assert response.headers.get("X-NPU-Proxy-Device") == "CPU"
    
    def test_custom_fallback_device(self):
        """Custom fallback device should be used."""
        with patch.dict(os.environ, {
            "NPU_PROXY_TOKEN_LIMIT": "10",
            "NPU_PROXY_FALLBACK_DEVICE": "GPU"
        }):
            from npu_proxy.routing.context_router import reset_context_router
            reset_context_router()
            from npu_proxy.main import app
            client = TestClient(app)
            
            # This message has >10 tokens so should fall back to GPU
            response = client.post("/v1/chat/completions", json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "This is a message that should definitely have more than ten tokens in it."}],
            })
        
        assert response.status_code == 200
        assert response.headers.get("X-NPU-Proxy-Device") == "GPU"
