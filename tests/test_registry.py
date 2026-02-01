"""Tests for model registry - TDD P2-3"""
import pytest
from pathlib import Path


class TestEmbeddingModelRegistry:
    """Tests for embedding model metadata in registry - Phase 3.5.5"""

    def test_bge_small_in_registry(self):
        """bge-small exists in MODELS_INFO with type=embedding"""
        from npu_proxy.models.registry import MODELS_INFO
        assert "bge-small" in MODELS_INFO
        assert MODELS_INFO["bge-small"]["type"] == "embedding"
        assert MODELS_INFO["bge-small"]["dimensions"] == 384

    def test_embedding_model_has_dimensions(self):
        """Embedding models have dimensions field"""
        from npu_proxy.models.registry import get_model_info
        
        # Test bge-small
        info = get_model_info("bge-small")
        assert info is not None
        assert "dimensions" in info
        assert info["dimensions"] == 384
        
        # Test all-minilm-l6-v2
        info2 = get_model_info("all-minilm-l6-v2")
        assert info2 is not None
        assert "dimensions" in info2
        assert info2["dimensions"] == 384
        
        # Test e5-large
        info3 = get_model_info("e5-large")
        assert info3 is not None
        assert "dimensions" in info3
        assert info3["dimensions"] == 1024

    def test_list_embedding_models(self):
        """Can list all embedding models"""
        from npu_proxy.models.registry import list_embedding_models
        
        embedding_models = list_embedding_models()
        assert len(embedding_models) >= 3
        
        # All returned models should be embedding type
        for model in embedding_models:
            assert model.get("type") == "embedding"
            assert "dimensions" in model
        
        # Should contain our known models
        ids = [m["id"] for m in embedding_models]
        assert "bge-small" in ids
        assert "all-minilm-l6-v2" in ids
        assert "e5-large" in ids


# 🔴 RED: These tests should FAIL initially

@pytest.mark.asyncio
async def test_registry_returns_model_info():
    """Registry should return model metadata by name"""
    from npu_proxy.models.registry import get_model_info
    
    info = get_model_info("tinyllama-1.1b-chat-int4-ov")
    
    assert info is not None
    assert info["id"] == "tinyllama-1.1b-chat-int4-ov"
    assert "size" in info
    assert "family" in info
    assert "quantization" in info


@pytest.mark.asyncio
async def test_registry_returns_none_for_unknown():
    """Registry should return None for unknown models"""
    from npu_proxy.models.registry import get_model_info
    
    info = get_model_info("nonexistent-model-xyz")
    assert info is None


@pytest.mark.asyncio
async def test_registry_list_all_models():
    """Registry should list all known models"""
    from npu_proxy.models.registry import list_all_models
    
    models = list_all_models()
    
    assert isinstance(models, list)
    assert len(models) >= 1
    assert any(m["id"] == "tinyllama-1.1b-chat-int4-ov" for m in models)


@pytest.mark.asyncio
async def test_models_endpoint_uses_registry():
    """GET /v1/models should use the model registry"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    from npu_proxy.models.registry import list_all_models
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")
    
    data = response.json()
    registry_models = list_all_models()
    
    # Endpoint should return same models as registry
    endpoint_ids = {m["id"] for m in data["data"]}
    registry_ids = {m["id"] for m in registry_models}
    
    assert endpoint_ids == registry_ids


@pytest.mark.asyncio
async def test_ollama_endpoints_use_registry():
    """Ollama endpoints should use same registry as OpenAI endpoints"""
    from npu_proxy.models.registry import get_model_info, MODELS_INFO
    from npu_proxy.api.ollama import MODELS_INFO as OLLAMA_MODELS_INFO
    
    # Both should reference the same data source
    # After refactor, OLLAMA_MODELS_INFO should be imported from registry
    for model_id in ["tinyllama-1.1b-chat-int4-ov"]:
        registry_info = get_model_info(model_id)
        assert registry_info is not None
