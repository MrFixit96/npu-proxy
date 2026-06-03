"""Tests for dynamic model scanning."""

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_scan_finds_models_in_directory(tmp_path):
    from npu_proxy.models.registry import scan_available_models

    model_dir = tmp_path / "test-model-1b"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_text("binary", encoding="utf-8")

    models = scan_available_models(tmp_path)

    assert len(models) == 1
    assert models[0]["id"] == "test-model-1b"


@pytest.mark.asyncio
async def test_scan_ignores_non_model_directories(tmp_path):
    from npu_proxy.models.registry import scan_available_models

    non_model_dir = tmp_path / "not-a-model"
    non_model_dir.mkdir()
    (non_model_dir / "readme.txt").write_text("not a model", encoding="utf-8")

    xml_only_dir = tmp_path / "xml-only-model"
    xml_only_dir.mkdir()
    (xml_only_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")

    model_dir = tmp_path / "valid-model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_text("binary", encoding="utf-8")

    models = scan_available_models(tmp_path)

    assert len(models) == 1
    assert models[0]["id"] == "valid-model"


@pytest.mark.asyncio
async def test_scan_rejects_xml_only_model_directories(tmp_path):
    from npu_proxy.models.registry import scan_available_models

    xml_only_dir = tmp_path / "xml-only-model"
    xml_only_dir.mkdir()
    (xml_only_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")

    assert scan_available_models(tmp_path) == []


@pytest.mark.asyncio
async def test_scan_returns_empty_for_missing_directory():
    from npu_proxy.models.registry import scan_available_models

    models = scan_available_models(Path("/nonexistent/path/12345"))

    assert models == []


@pytest.mark.asyncio
async def test_scan_derives_metadata_for_unknown_models(tmp_path):
    from npu_proxy.models.registry import scan_available_models

    model_dir = tmp_path / "granite-3.3-2b-fp8-ov"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_text("binary", encoding="utf-8")

    models = scan_available_models(tmp_path)

    assert len(models) == 1
    assert models[0]["family"] == "granite"
    assert models[0]["quantization"] == "FP8"
    assert models[0]["parameter_size"] == "2B"
    assert models[0]["backend"] == "openvino"
    assert models[0]["format"] == "openvino-ir"
    assert models[0]["task"] == "text-generation"


@pytest.mark.asyncio
async def test_scan_marks_vision_language_models_as_vision(tmp_path):
    from npu_proxy.models.registry import scan_available_models

    model_dir = tmp_path / "qwen2.5-vl-3b-int4-ov"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_text("binary", encoding="utf-8")

    models = scan_available_models(tmp_path)

    assert len(models) == 1
    assert models[0]["type"] == "vision"
    assert models[0]["task"] == "image-text-to-text"


@pytest.mark.asyncio
async def test_models_endpoint_includes_scanned_models():
    from httpx import ASGITransport, AsyncClient

    from npu_proxy.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")

    data = response.json()

    assert len(data["data"]) >= 1
    for model in data["data"]:
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
