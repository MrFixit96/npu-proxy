"""Tests for embedding device selection (CPU/NPU/GPU)."""

import os
import tempfile
from pathlib import Path

import pytest

from npu_proxy.inference.embedding_engine import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    ProductionEmbeddingEngine,
    get_embedding_model_path,
    is_embedding_model_downloaded,
)

# Skip tests if model not downloaded
skip_if_no_model = pytest.mark.skipif(
    not is_embedding_model_downloaded(DEFAULT_EMBEDDING_MODEL),
    reason="Production embedding model not downloaded",
)


class TestCPUDeviceSelection:
    """Tests for CPU device selection."""

    @pytest.mark.slow
    @skip_if_no_model
    def test_cpu_device_selection(self, monkeypatch):
        """Test that CPU device selection works correctly."""
        monkeypatch.setenv("NPU_PROXY_EMBEDDING_DEVICE", "CPU")
        
        model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
        engine = ProductionEmbeddingEngine(
            model_path=str(model_path),
            device="CPU",
        )

        engine_info = engine.get_engine_info()
        assert engine_info["device"] == "CPU"
        assert engine_info["is_production"] is True
        assert engine_info["dimensions"] == DEFAULT_EMBEDDING_DIMENSIONS

        embedding = engine.embed("This is a test embedding")
        assert len(embedding) == DEFAULT_EMBEDDING_DIMENSIONS


class TestNPUDeviceSelection:
    """Tests for NPU device selection."""

    @pytest.mark.slow
    @skip_if_no_model
    def test_npu_device_selection(self, monkeypatch):
        """Test that NPU device selection handles unavailability gracefully."""
        model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
        
        try:
            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                device="NPU",
            )
            
            engine_info = engine.get_engine_info()
            # If NPU loaded successfully, verify it
            if engine_info["is_production"]:
                assert engine_info["device"] == "NPU"
            # If it fell back, that's ok too
            embedding = engine.embed("This is a test for NPU device")
            assert len(embedding) == DEFAULT_EMBEDDING_DIMENSIONS
            
        except RuntimeError as e:
            pytest.skip(f"NPU not available on this system: {e}")


class TestDeviceFallback:
    """Tests for device fallback behavior."""

    def test_device_fallback_on_error(self, tmp_path):
        """Test that engine falls back gracefully with invalid path."""
        invalid_model_path = str(tmp_path / "nonexistent" / "model")
        
        engine = ProductionEmbeddingEngine(
            model_path=invalid_model_path,
            device="CPU",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        )

        engine_info = engine.get_engine_info()
        assert engine_info["is_production"] is False, "Should be in fallback mode"
        assert engine_info["dimensions"] == DEFAULT_EMBEDDING_DIMENSIONS

        embedding = engine.embed("This is a test for fallback behavior")
        assert len(embedding) == DEFAULT_EMBEDDING_DIMENSIONS

    @pytest.mark.slow
    @skip_if_no_model
    def test_default_device_selection(self, monkeypatch):
        """Test that default device selection works correctly."""
        monkeypatch.delenv("NPU_PROXY_EMBEDDING_DEVICE", raising=False)
        
        model_path = get_embedding_model_path(DEFAULT_EMBEDDING_MODEL)
        engine = ProductionEmbeddingEngine(model_path=str(model_path))

        engine_info = engine.get_engine_info()
        assert engine_info["device"] in ["CPU", "GPU", "NPU"]
        assert engine_info["is_production"] is True
        assert engine_info["dimensions"] == DEFAULT_EMBEDDING_DIMENSIONS

        embedding = engine.embed("Default device selection test")
        assert len(embedding) == DEFAULT_EMBEDDING_DIMENSIONS
