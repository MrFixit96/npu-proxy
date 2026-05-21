"""Tests for health endpoint - TDD Phase 1.1"""
import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy import __version__
from npu_proxy.api import health as health_api
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


@pytest.mark.asyncio
async def test_health_reports_package_version():
    """GET /health should report the shared package version."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    data = response.json()
    assert data["version"] == __version__


@pytest.mark.asyncio
async def test_health_reports_loaded_llm_device(monkeypatch):
    """GET /health should report the actual device for a loaded LLM engine."""

    class FakeCore:
        available_devices = ["CPU", "NPU"]

    class FakeLlmEngine:
        model_name = "tinyllama-1.1b-chat-int4-ov"

        def get_device_info(self):
            return {"actual_device": "NPU", "used_fallback": False}

    monkeypatch.setattr(health_api, "get_ov_core", lambda: FakeCore())
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: True)
    monkeypatch.setattr(
        health_api,
        "get_loaded_models",
        lambda: {"tinyllama-1.1b-chat-int4-ov": FakeLlmEngine()},
    )
    monkeypatch.setattr(health_api, "get_embedding_engine", lambda: None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert data["engines"]["llm"]["status"] == "loaded"
    assert data["engines"]["llm"]["device"] == "NPU"
    assert data["engines"]["llm"]["model"] == "tinyllama-1.1b-chat-int4-ov"
