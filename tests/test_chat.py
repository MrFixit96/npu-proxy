"""Tests for chat completions endpoint - TDD Phase 1.4"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_chat_completions_returns_200():
    """POST /v1/chat/completions should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_returns_openai_format():
    """Response should match OpenAI chat completion format"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
    data = response.json()
    
    assert "id" in data
    assert "object" in data
    assert data["object"] == "chat.completion"
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    
    choice = data["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert "role" in choice["message"]
    assert "content" in choice["message"]
    assert "finish_reason" in choice


@pytest.mark.asyncio
async def test_chat_completions_streaming():
    """POST with stream=true should return SSE stream"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
            
            assert len(chunks) > 0
