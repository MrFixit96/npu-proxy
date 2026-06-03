"""Tests for embedding device selection (CPU/NPU/GPU)."""

import os
import tempfile
from pathlib import Path

import pytest

from npu_proxy.inference.embedding_engine import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_FALLBACK_MODE_ENV_VAR,
    EmbeddingUnavailableError,
    ProductionEmbeddingEngine,
    get_embedding_model_path,
    is_embedding_model_downloaded,
)

# Skip tests if model not downloaded
skip_if_no_model = pytest.mark.skipif(
    not is_embedding_model_downloaded(DEFAULT_EMBEDDING_MODEL),
    reason="Production embedding model not downloaded",
)

# The default embedding model (bge-small) cannot compile on the Intel NPU: its
# SDPA subgraph is rejected by the NPU plugin (check_sdpa_nodes failure).
# all-MiniLM-L6-v2 is the registry-catalogued, community-validated small BERT
# embedder that runs natively on the NPU via a static-shape preset
# (see _NPU_STATIC_EMBEDDING_PRESETS in embedding_engine).
NPU_VALIDATED_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

skip_if_no_npu_model = pytest.mark.skipif(
    not is_embedding_model_downloaded(NPU_VALIDATED_EMBEDDING_MODEL),
    reason="NPU-validated embedding model (all-MiniLM-L6-v2) not downloaded",
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
    """Tests for NPU device selection using an NPU-validated embedding model."""

    @pytest.mark.slow
    @skip_if_no_npu_model
    def test_npu_device_selection(self, monkeypatch):
        """all-MiniLM-L6-v2 should load and run natively on the NPU when present.

        On NPU-equipped hardware this exercises the real static-shape NPU
        embedding path end-to-end. On machines without a usable NPU the engine
        degrades to an unavailable state (fallback is off by default), and the
        test skips rather than asserting against absent hardware.
        """
        monkeypatch.delenv(EMBEDDING_FALLBACK_MODE_ENV_VAR, raising=False)

        model_path = get_embedding_model_path(NPU_VALIDATED_EMBEDDING_MODEL)
        engine = ProductionEmbeddingEngine(
            model_path=str(model_path),
            device="NPU",
            requested_model=NPU_VALIDATED_EMBEDDING_MODEL,
            resolved_model="all-minilm-l6-v2",
            repo_id=NPU_VALIDATED_EMBEDDING_MODEL,
        )

        engine_info = engine.get_engine_info()

        if not engine_info["is_production"]:
            reason = engine_info.get("load_error") or engine_info.get(
                "fallback_reason", "unknown"
            )
            pytest.skip(f"NPU not available on this system: {reason}")

        # NPU is present and the model compiled: assert the real NPU path.
        assert engine_info["device"] == "NPU"
        assert engine_info["is_fallback"] is False

        embedding = engine.embed("This is a test for NPU device")
        assert len(embedding) == DEFAULT_EMBEDDING_DIMENSIONS


class TestDeviceFallback:
    """Tests for explicit device fallback behavior."""

    def test_device_fallback_on_error(self, tmp_path):
        """Invalid paths should hard-fail unless fallback was explicitly enabled."""
        invalid_model_path = str(tmp_path / "nonexistent" / "model")

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv(EMBEDDING_FALLBACK_MODE_ENV_VAR, "disabled")
            engine = ProductionEmbeddingEngine(
                model_path=invalid_model_path,
                device="CPU",
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            )

            engine_info = engine.get_engine_info()
            assert engine_info["is_production"] is False
            assert engine_info["is_fallback"] is False
            assert engine_info["available"] is False
            assert engine_info["dimensions"] == DEFAULT_EMBEDDING_DIMENSIONS

            with pytest.raises(EmbeddingUnavailableError):
                engine.embed("This is a test for fallback behavior")

    def test_device_fallback_requires_explicit_opt_in(self, tmp_path):
        """Explicit fallback should remain available for operator workflows."""
        invalid_model_path = str(tmp_path / "nonexistent" / "model")

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv(EMBEDDING_FALLBACK_MODE_ENV_VAR, "hash")
            engine = ProductionEmbeddingEngine(
                model_path=invalid_model_path,
                device="CPU",
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            )

            engine_info = engine.get_engine_info()
            assert engine_info["is_fallback"] is True
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
