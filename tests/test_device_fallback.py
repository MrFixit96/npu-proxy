"""Tests for device fallback functionality"""
import pytest
from types import SimpleNamespace
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


def test_select_best_device_returns_none_tuple_for_unavailable_custom_device():
    """Custom devices outside the fallback chain should fail closed without advisory CPU routing."""
    from npu_proxy.inference.engine import select_best_device

    with patch("npu_proxy.inference.engine.get_available_devices", return_value=["CPU"]):
        assert select_best_device(preferred="AUTO:GPU") == (None, None)


@pytest.mark.parametrize(
    ("available", "preferred", "expected"),
    [
        (["NPU", "GPU", "CPU"], "NPU", ("NPU", "GPU")),
        (["GPU", "CPU"], "NPU", ("GPU", "CPU")),
        (["CPU"], "GPU", ("CPU", None)),
        (["MYRIAD", "CPU"], None, ("CPU", None)),
    ],
)
def test_select_best_device_fallback_chain_contract(available, preferred, expected):
    """NPU→GPU→CPU ordering is advisory and deterministic for known devices."""
    from npu_proxy.inference.engine import select_best_device

    with patch("npu_proxy.inference.engine.get_available_devices", return_value=available):
        assert select_best_device(preferred=preferred) == expected


@pytest.mark.asyncio
async def test_health_devices_reports_advisory_fallback_runtime_state(monkeypatch, async_client):
    """Device health should report requested, actual, and fallback devices without rerouting requests."""
    from npu_proxy.api import health as health_api

    class FakeLlmEngine:
        model_name = "tinyllama"

        def get_device_info(self):
            return {
                "actual_device": "GPU",
                "requested_device": "NPU",
                "fallback_device": "CPU",
                "used_fallback": True,
                "backend": "openvino",
            }

    monkeypatch.setattr("npu_proxy.inference.engine.get_available_devices", lambda: ["NPU", "GPU", "CPU"])
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: SimpleNamespace(
        backend=SimpleNamespace(value="openvino"),
        device="NPU",
        compile_cache_dir=None,
        compile_cache_mode=None,
        prefix_cache_mode="auto",
        is_alpha_backend=False,
        llama_cpp_model_path=None,
        model_path="C:\\models\\tinyllama",
        backend_model_path=lambda: "C:\\models\\tinyllama",
    ))
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: True)
    monkeypatch.setattr(health_api, "get_loaded_models", lambda: {"tinyllama": FakeLlmEngine()})
    monkeypatch.setattr(health_api, "get_primary_loaded_engine", lambda: FakeLlmEngine())
    monkeypatch.setattr(
        health_api.embedding_config,
        "resolve_embedding_model_config",
        lambda: SimpleNamespace(
            requested_model="all-minilm",
            requested_device="CPU",
            resolved_model="all-minilm-l6-v2",
            dimensions=384,
            model_path="C:\\models\\embeddings\\all-minilm",
            canonical_path="C:\\models\\embeddings\\all-minilm",
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            is_downloaded=False,
        ),
    )
    monkeypatch.setattr(health_api, "get_loaded_embedding_engine", lambda model_name=None, device=None: None)

    response = await async_client.get("/health/devices")

    assert response.status_code == 200
    data = response.json()
    assert data["active_device"] == "GPU"
    assert data["llm"]["runtime_state"]["requested_device"] == "NPU"
    assert data["llm"]["runtime_state"]["fallback_device"] == "CPU"
    assert data["llm"]["runtime_state"]["used_fallback"] is True
