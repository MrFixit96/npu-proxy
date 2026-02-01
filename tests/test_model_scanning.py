"""Tests for dynamic model scanning - TDD P2-1"""
import pytest
from pathlib import Path
import tempfile
import shutil


# 🔴 RED: These tests should FAIL initially

@pytest.mark.asyncio
async def test_scan_finds_models_in_directory():
    """scan_available_models should find models with openvino_model.xml"""
    from npu_proxy.models.registry import scan_available_models
    
    # Create temp directory with fake model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test-model-1b"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").write_text("<model/>")
        (model_dir / "openvino_model.bin").write_text("binary")
        
        models = scan_available_models(Path(tmpdir))
        
        assert len(models) == 1
        assert models[0]["id"] == "test-model-1b"


@pytest.mark.asyncio
async def test_scan_ignores_non_model_directories():
    """scan_available_models should skip directories without openvino_model.xml"""
    from npu_proxy.models.registry import scan_available_models
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory without model files
        non_model_dir = Path(tmpdir) / "not-a-model"
        non_model_dir.mkdir()
        (non_model_dir / "readme.txt").write_text("not a model")
        
        # Create valid model directory
        model_dir = Path(tmpdir) / "valid-model"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").write_text("<model/>")
        
        models = scan_available_models(Path(tmpdir))
        
        assert len(models) == 1
        assert models[0]["id"] == "valid-model"


@pytest.mark.asyncio
async def test_scan_returns_empty_for_missing_directory():
    """scan_available_models should return empty list if directory doesn't exist"""
    from npu_proxy.models.registry import scan_available_models
    
    models = scan_available_models(Path("/nonexistent/path/12345"))
    
    assert models == []


@pytest.mark.asyncio
async def test_models_endpoint_includes_scanned_models():
    """GET /v1/models should include dynamically scanned models"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")
    
    data = response.json()
    
    # Should have at least the built-in models
    assert len(data["data"]) >= 1
    
    # Each model should have required fields
    for model in data["data"]:
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
