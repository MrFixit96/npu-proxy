"""Tests for device fallback functionality"""
import pytest
from unittest.mock import patch, MagicMock


def test_get_available_devices_returns_list():
    """get_available_devices should return list of device strings"""
    from npu_proxy.inference.engine import get_available_devices
    
    devices = get_available_devices()
    assert isinstance(devices, list)
    assert "CPU" in devices  # CPU is always available


def test_select_best_device_prefers_npu():
    """select_best_device should prefer NPU when available"""
    from npu_proxy.inference.engine import select_best_device
    
    with patch('npu_proxy.inference.engine.get_available_devices') as mock:
        mock.return_value = ["NPU", "GPU", "CPU"]
        device, fallback = select_best_device()
        assert device == "NPU"
        assert fallback == "GPU"


def test_select_best_device_falls_back_to_gpu():
    """select_best_device should use GPU if NPU unavailable"""
    from npu_proxy.inference.engine import select_best_device
    
    with patch('npu_proxy.inference.engine.get_available_devices') as mock:
        mock.return_value = ["GPU", "CPU"]
        device, fallback = select_best_device()
        assert device == "GPU"
        assert fallback == "CPU"


def test_select_best_device_falls_back_to_cpu():
    """select_best_device should use CPU as last resort"""
    from npu_proxy.inference.engine import select_best_device
    
    with patch('npu_proxy.inference.engine.get_available_devices') as mock:
        mock.return_value = ["CPU"]
        device, fallback = select_best_device()
        assert device == "CPU"
        assert fallback is None


def test_select_best_device_respects_user_preference():
    """select_best_device should honor user's device choice"""
    from npu_proxy.inference.engine import select_best_device
    
    with patch('npu_proxy.inference.engine.get_available_devices') as mock:
        mock.return_value = ["NPU", "GPU", "CPU"]
        device, fallback = select_best_device(preferred="GPU")
        assert device == "GPU"
        assert fallback == "CPU"


def test_select_best_device_ignores_unavailable_preference():
    """select_best_device should ignore unavailable preference"""
    from npu_proxy.inference.engine import select_best_device
    
    with patch('npu_proxy.inference.engine.get_available_devices') as mock:
        mock.return_value = ["CPU"]
        device, fallback = select_best_device(preferred="NPU")
        assert device == "CPU"
        assert fallback is None


@pytest.mark.asyncio
async def test_health_devices_endpoint():
    """GET /health/devices should return device info"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/devices")
    
    assert response.status_code == 200
    data = response.json()
    assert "available_devices" in data
    assert "fallback_chain" in data
    assert data["fallback_chain"] == ["NPU", "GPU", "CPU"]


@pytest.mark.asyncio
async def test_health_shows_gpu_available():
    """GET /health should show GPU availability"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "gpu_available" in data
    assert "npu_available" in data
    assert "cpu_available" in data
