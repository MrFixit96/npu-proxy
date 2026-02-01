"""Tests for health endpoint - TDD Phase 1.1"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_health_returns_200():
    """GET /health should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_status():
    """GET /health should return status field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] in ("ok", "healthy")


@pytest.mark.asyncio
async def test_health_includes_npu_status():
    """GET /health should include NPU availability"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    data = response.json()
    assert "npu_available" in data
    assert isinstance(data["npu_available"], bool)
