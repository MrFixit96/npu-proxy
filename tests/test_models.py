"""Tests for models endpoint - TDD Phase 1.2"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_models_returns_200():
    """GET /v1/models should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_models_returns_openai_format():
    """GET /v1/models should return OpenAI-compatible format"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")
    data = response.json()
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_models_data_has_required_fields():
    """Each model in data should have id, object, created, owned_by"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")
    data = response.json()
    
    # Should have at least one model
    assert len(data["data"]) > 0
    
    model = data["data"][0]
    assert "id" in model
    assert "object" in model
    assert model["object"] == "model"
    assert "created" in model
    assert "owned_by" in model
