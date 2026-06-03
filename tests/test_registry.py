"""Tests for model registry metadata and catalog derivation."""

import pytest


class TestEmbeddingModelRegistry:
    """Tests for embedding model metadata in registry."""

    def test_bge_small_in_registry(self):
        from npu_proxy.models.registry import MODELS_INFO

        assert "bge-small" in MODELS_INFO
        assert MODELS_INFO["bge-small"]["type"] == "embedding"
        assert MODELS_INFO["bge-small"]["dimensions"] == 384

    def test_embedding_model_has_dimensions(self):
        from npu_proxy.models.registry import get_model_info

        info = get_model_info("bge-small")
        assert info is not None
        assert info["dimensions"] == 384

        info2 = get_model_info("all-minilm-l6-v2")
        assert info2 is not None
        assert info2["dimensions"] == 384

        info3 = get_model_info("e5-large")
        assert info3 is not None
        assert info3["dimensions"] == 1024

    def test_list_embedding_models(self):
        from npu_proxy.models.registry import list_embedding_models

        embedding_models = list_embedding_models()
        assert len(embedding_models) >= 3

        for model in embedding_models:
            assert model.get("type") == "embedding"
            assert "dimensions" in model
            assert model.get("task") == "feature-extraction"

        ids = [model["id"] for model in embedding_models]
        assert "bge-small" in ids
        assert "all-minilm-l6-v2" in ids
        assert "e5-large" in ids

    def test_alias_only_embedding_catalog_entries_keep_repo_storage_key(self):
        from npu_proxy.models.registry import find_catalog_entry, get_model_info

        entry = find_catalog_entry("bge-base")
        info = get_model_info("bge-base")

        assert entry is not None
        assert info is not None
        assert entry.storage_key == "BAAI%2Fbge-base-en-v1.5"
        assert info["storage_key"] == "BAAI%2Fbge-base-en-v1.5"

    def test_catalog_entry_rejects_missing_embedding_dimensions(self):
        from npu_proxy.models.registry import CatalogEntry

        with pytest.raises(ValueError):
            CatalogEntry(
                repo_id="org/example-embedding",
                ollama_name="example-embedding",
                name="Example Embedding",
                type="embedding",
                backend="openvino",
                format="openvino-ir",
                task="feature-extraction",
            )


@pytest.mark.asyncio
async def test_registry_returns_model_info():
    from npu_proxy.models.registry import get_model_info

    info = get_model_info("tinyllama-1.1b-chat-int4-ov")

    assert info is not None
    assert info["id"] == "tinyllama-1.1b-chat-int4-ov"
    assert "size" in info
    assert "family" in info
    assert "quantization" in info


@pytest.mark.asyncio
async def test_registry_returns_none_for_unknown():
    from npu_proxy.models.registry import get_model_info

    assert get_model_info("nonexistent-model-xyz") is None


@pytest.mark.asyncio
async def test_registry_list_all_models():
    from npu_proxy.models.registry import list_all_models

    models = list_all_models()

    assert isinstance(models, list)
    assert len(models) >= 1
    assert any(model["id"] == "tinyllama-1.1b-chat-int4-ov" for model in models)


@pytest.mark.asyncio
async def test_registry_adds_backend_format_and_task_metadata():
    from npu_proxy.models.registry import get_model_info

    info = get_model_info("tinyllama-1.1b-chat-int4-ov")
    assert info is not None
    assert info["backend"] == "openvino"
    assert info["format"] == "openvino-ir"
    assert info["task"] == "text-generation"
    assert info["repo_id"] == "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"


@pytest.mark.asyncio
async def test_registry_exposes_alias_only_catalog_entries():
    from npu_proxy.models.registry import get_model_info

    phi3 = get_model_info("phi-3")
    assert phi3 is not None
    assert phi3["task"] == "text-generation"
    assert phi3["repo_id"] == "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"

    tinyllama_fp16 = get_model_info("tinyllama:fp16")
    assert tinyllama_fp16 is not None
    assert tinyllama_fp16["quantization"] == "FP16"


@pytest.mark.asyncio
async def test_registry_recognizes_canonical_storage_keys():
    from npu_proxy.models.registry import get_model_info

    info = get_model_info("OpenVINO%2FPhi-3-mini-4k-instruct-int4-ov")
    assert info is not None
    assert info["repo_id"] == "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"


@pytest.mark.asyncio
async def test_registry_preserves_embedding_task_metadata():
    from npu_proxy.models.registry import get_model_info

    info = get_model_info("bge-small")
    assert info is not None
    assert info["backend"] == "openvino"
    assert info["format"] == "openvino-ir"
    assert info["task"] == "feature-extraction"
    assert info["hf_repo"] == "BAAI/bge-small-en-v1.5"


@pytest.mark.asyncio
async def test_registry_secondary_alias_resolves_to_primary_canonical_info():
    from npu_proxy.models.registry import get_model_info

    primary = get_model_info("tinyllama:fp16")
    secondary = get_model_info("tinyllama-fp16")

    assert primary is not None
    assert secondary is not None
    assert secondary == primary
    assert secondary["id"] == "tinyllama:fp16"
    assert secondary["repo_id"] == "OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov"


@pytest.mark.asyncio
async def test_registry_rejects_xml_only_local_model_dirs(monkeypatch, tmp_path):
    from npu_proxy.models import registry

    xml_only_dir = tmp_path / "xml-only-model"
    xml_only_dir.mkdir()
    (xml_only_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")

    valid_dir = tmp_path / "valid-model"
    valid_dir.mkdir()
    (valid_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
    (valid_dir / "openvino_model.bin").write_bytes(b"weights")

    monkeypatch.setattr(registry, "DEFAULT_MODEL_DIR", tmp_path)

    assert registry.get_model_info("xml-only-model") is None
    assert registry.get_model_info("valid-model") is not None
    assert all(model["id"] != "xml-only-model" for model in registry.list_all_models())
    assert any(model["id"] == "valid-model" for model in registry.list_all_models())


@pytest.mark.asyncio
async def test_models_endpoint_uses_registry():
    from httpx import ASGITransport, AsyncClient

    from npu_proxy.main import app
    from npu_proxy.models.registry import list_all_models

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")

    data = response.json()
    registry_models = list_all_models()

    endpoint_ids = {model["id"] for model in data["data"]}
    registry_ids = {model["id"] for model in registry_models}

    assert endpoint_ids == registry_ids


@pytest.mark.asyncio
async def test_ollama_endpoints_use_registry():
    from npu_proxy.api.ollama import MODELS_INFO as OLLAMA_MODELS_INFO
    from npu_proxy.models.registry import MODELS_INFO, get_model_info

    assert OLLAMA_MODELS_INFO
    for model_id in ["tinyllama-1.1b-chat-int4-ov"]:
        registry_info = get_model_info(model_id)
        assert registry_info is not None
        assert model_id in MODELS_INFO
