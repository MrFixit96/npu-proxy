"""Tests for Ollama-compatible endpoints - TDD Phase 1.7-2.0"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_ps_returns_200():
    """GET /api/ps should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/ps")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ps_returns_models_array():
    """GET /api/ps should return models array"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/ps")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_version_returns_200():
    """GET /api/version should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/version")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_version_returns_version_field():
    """GET /api/version should return version field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/version")
    data = response.json()
    assert "version" in data
    assert "npu-proxy" in data["version"]


@pytest.mark.asyncio
async def test_show_returns_200():
    """POST /api/show should return 200 for valid model"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/show",
            json={"model": "tinyllama-1.1b-chat-int4-ov"},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_show_returns_model_details():
    """POST /api/show should return model details"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/show",
            json={"model": "tinyllama-1.1b-chat-int4-ov"},
        )
    data = response.json()
    assert "details" in data
    assert "modelfile" in data
    assert "model_info" in data


@pytest.mark.asyncio
async def test_show_returns_404_for_invalid_model():
    """POST /api/show should return 404 for invalid model"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/show",
            json={"model": "nonexistent-model"},
        )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_generate_returns_200():
    """POST /api/generate should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": False,
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_generate_returns_response():
    """POST /api/generate should return response field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": False,
            },
        )
    data = response.json()
    assert "response" in data
    assert "done" in data
    assert data["done"] is True
    assert "model" in data


@pytest.mark.asyncio
async def test_chat_returns_200():
    """POST /api/chat should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_returns_message():
    """POST /api/chat should return message field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
    data = response.json()
    assert "message" in data
    assert "done" in data
    assert data["done"] is True
    assert data["message"]["role"] == "assistant"
    assert len(data["message"]["content"]) > 0


# Phase 2.5: Model Management Endpoint Tests


@pytest.mark.asyncio
async def test_pull_unknown_model_returns_404():
    """POST /api/pull with unknown model returns 404"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/pull", json={"name": "nonexistent-model-xyz"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_pull_requires_name():
    """POST /api/pull without name returns 422 (validation error)"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/pull", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_returns_200():
    """GET /api/search returns 200"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_search_returns_models_list():
    """GET /api/search returns models array"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_search_returns_pagination_fields():
    """GET /api/search returns pagination fields"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search")
    data = response.json()
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    assert "has_more" in data


@pytest.mark.asyncio
async def test_search_invalid_sort_returns_400():
    """GET /api/search with invalid sort returns 400"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search?sort=invalid")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_search_invalid_type_returns_400():
    """GET /api/search with invalid type returns 400"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search?type=invalid")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_known_models_returns_200():
    """GET /api/models/known returns 200"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models/known")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_known_models_returns_list():
    """GET /api/models/known returns models list"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models/known")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


@pytest.mark.asyncio
async def test_known_models_have_required_fields():
    """Each known model has ollama_name, huggingface_repo, quantization"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models/known")
    data = response.json()
    for model in data["models"]:
        assert "ollama_name" in model
        assert "huggingface_repo" in model
        assert "quantization" in model
