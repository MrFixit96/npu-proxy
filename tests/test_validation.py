"""Tests for input validation and error handling"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_chat_invalid_max_tokens_rejected():
    """max_tokens outside range should return 422"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": -1,
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_invalid_temperature_rejected():
    """temperature outside range should return 422"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 5.0,
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_empty_messages_rejected():
    """Empty messages array should return 422"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [],
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_invalid_role_rejected():
    """Invalid message role should return 422"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [{"role": "invalid_role", "content": "Hi"}],
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_max_tokens_at_boundary():
    """max_tokens at valid boundaries should be accepted"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test minimum (1)
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            },
        )
        assert response.status_code == 200
        
        # Test maximum (4096)
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4096,
            },
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_temperature_at_boundary():
    """temperature at valid boundaries should be accepted"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test minimum (0.0)
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.0,
            },
        )
        assert response.status_code == 200
        
        # Test maximum (2.0)
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 2.0,
            },
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_invalid_model_returns_404():
    """Non-existent model should return 404"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ollama_generate_invalid_model_returns_404():
    """Ollama generate with non-existent model should return 404"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "nonexistent-model",
                "prompt": "Hello",
            },
        )
    assert response.status_code == 404
