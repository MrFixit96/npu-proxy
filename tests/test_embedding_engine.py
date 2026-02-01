"""Tests for embedding engine - TDD Phase 3.5"""
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from npu_proxy.inference.embedding_engine import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_DEVICE,
    get_embedding_model_name,
    get_embedding_device,
    get_embedding_model_path,
    is_embedding_model_downloaded,
    ProductionEmbeddingEngine,
    get_embedding_engine,
    _reset_embedding_engine,
)


class TestEmbeddingEngineConfig:
    """Tests for embedding engine configuration constants and functions."""

    def test_default_model_is_bge_small(self):
        """Default model should be BAAI/bge-small-en-v1.5."""
        assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"

    def test_default_dimensions_is_384(self):
        """Default embedding dimensions should be 384."""
        assert DEFAULT_EMBEDDING_DIMENSIONS == 384

    def test_model_configurable_via_env(self):
        """Model should be configurable via NPU_PROXY_EMBEDDING_MODEL env var."""
        with patch.dict(os.environ, {"NPU_PROXY_EMBEDDING_MODEL": "custom/model-name"}):
            assert get_embedding_model_name() == "custom/model-name"
        # Without env var, should return default
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NPU_PROXY_EMBEDDING_MODEL", None)
            assert get_embedding_model_name() == DEFAULT_EMBEDDING_MODEL


class TestProductionEmbeddingEngine:
    """Tests for the ProductionEmbeddingEngine class."""

    def test_init_with_model_path(self, tmp_path):
        """Engine should initialize with model path and device."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                device="NPU"
            )
            
            assert engine._model_path == str(model_path)
            assert engine._device == "NPU"

    def test_embed_single_returns_list_of_floats(self, tmp_path):
        """embed() should return a list of floats."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            # Simulate OpenVINO returning a tensor-like object
            mock_result = MagicMock()
            mock_result.data = [[0.1, 0.2, 0.3, 0.4]]
            mock_pipeline.generate.return_value = mock_result
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(model_path=str(model_path))
            result = engine.embed("test text")
            
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)

    def test_embed_batch_returns_list_of_embeddings(self, tmp_path):
        """embed_batch() should return a list of embedding lists."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_result = MagicMock()
            mock_result.data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_pipeline.generate.return_value = mock_result
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(model_path=str(model_path))
            results = engine.embed_batch(["text1", "text2"])
            
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(emb, list) for emb in results)
            assert all(isinstance(x, float) for emb in results for x in emb)

    def test_embed_empty_string_returns_zeros(self, tmp_path):
        """embed() with empty string should return zero vector."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                dimensions=384
            )
            result = engine.embed("")
            
            assert result == [0.0] * 384
            assert len(result) == 384


class TestEmbeddingEngineFallback:
    """Tests for fallback behavior when model is not available."""

    def test_falls_back_to_hash_if_model_missing(self, tmp_path):
        """Should fall back to hash-based embedding if model not downloaded."""
        model_path = tmp_path / "nonexistent_model"
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_ov.TextEmbeddingPipeline.side_effect = Exception("Model not found")
            
            engine = ProductionEmbeddingEngine(model_path=str(model_path))
            result = engine.embed("test text")
            
            # Should still return valid embedding from fallback
            assert isinstance(result, list)
            assert len(result) == DEFAULT_EMBEDDING_DIMENSIONS
            assert all(isinstance(x, float) for x in result)

    def test_get_engine_info_returns_dict(self, tmp_path):
        """get_engine_info() should return dict with required keys."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(model_path=str(model_path))
            info = engine.get_engine_info()
            
            assert isinstance(info, dict)
            assert "model_name" in info
            assert "dimensions" in info
            assert "is_production" in info
            assert "device" in info

    def test_dimensions_property(self, tmp_path):
        """dimensions property should return embedding dimensions."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                dimensions=512
            )
            
            assert engine.dimensions == 512

    def test_model_name_property(self, tmp_path):
        """model_name property should return the model name."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                model_name="my-custom-model"
            )
            
            assert engine.model_name == "my-custom-model"


class TestEmbeddingDeviceSelection:
    """Tests for device selection configuration."""

    def test_default_device_is_cpu(self):
        """Default device should be CPU."""
        assert DEFAULT_EMBEDDING_DEVICE == "CPU"

    def test_device_configurable_via_env(self):
        """Device should be configurable via NPU_PROXY_EMBEDDING_DEVICE env var."""
        with patch.dict(os.environ, {"NPU_PROXY_EMBEDDING_DEVICE": "NPU"}):
            assert get_embedding_device() == "NPU"
        # Without env var, should return default
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NPU_PROXY_EMBEDDING_DEVICE", None)
            assert get_embedding_device() == DEFAULT_EMBEDDING_DEVICE


class TestEmbeddingModelPath:
    """Tests for model path resolution."""

    def test_get_embedding_model_path_returns_path(self):
        """get_embedding_model_path() should return a Path object."""
        result = get_embedding_model_path()
        assert isinstance(result, Path)

    def test_model_path_in_cache_directory(self):
        """Model path should be in ~/.cache/npu-proxy/models/embeddings/."""
        result = get_embedding_model_path()
        path_str = str(result)
        assert "npu-proxy" in path_str
        assert "models" in path_str
        assert "embeddings" in path_str

    def test_is_embedding_model_downloaded_false_when_missing(self):
        """is_embedding_model_downloaded() should return False for missing model."""
        result = is_embedding_model_downloaded("nonexistent/model-xyz-12345")
        assert result is False


class TestEmbeddingEngineSingleton:
    """Tests for the singleton get_embedding_engine pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        _reset_embedding_engine()

    def teardown_method(self):
        """Reset singleton after each test."""
        _reset_embedding_engine()

    def test_get_embedding_engine_returns_engine(self):
        """get_embedding_engine() should return an engine instance."""
        with patch("npu_proxy.inference.embedding_engine.is_embedding_model_downloaded", return_value=False):
            engine = get_embedding_engine()
            assert engine is not None
            assert hasattr(engine, "embed")
            assert hasattr(engine, "embed_batch")

    def test_get_embedding_engine_returns_same_instance(self):
        """get_embedding_engine() should return the same instance (singleton)."""
        with patch("npu_proxy.inference.embedding_engine.is_embedding_model_downloaded", return_value=False):
            engine1 = get_embedding_engine()
            engine2 = get_embedding_engine()
            assert engine1 is engine2
