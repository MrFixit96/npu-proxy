"""Tests for embedding engine - TDD Phase 3.5"""
import os
import threading
import time
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from npu_proxy.inference.embedding_engine import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_DEVICE,
    EMBEDDING_FALLBACK_MODE_ENV_VAR,
    EmbeddingInferenceError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
    get_embedding_model_name,
    get_embedding_device,
    get_embedding_model_path,
    is_embedding_model_downloaded,
    ProductionEmbeddingEngine,
    get_embedding_engine,
    _reset_embedding_engine,
)
from npu_proxy.inference.embedding_config import (
    InvalidEmbeddingModelError,
    is_known_embedding_model,
    resolve_embedding_model_config,
)


class BlockingPipelineLoader:
    def __init__(self, pipeline=None):
        self.release = threading.Event()
        self.pipeline = pipeline or MagicMock()

    def __call__(self, *args, **kwargs):
        self.release.wait(timeout=5)
        return self.pipeline


class BlockingEmbeddingPipeline:
    def __init__(self):
        self.query_release = threading.Event()
        self.batch_release = threading.Event()

    def embed_query(self, text):
        self.query_release.wait(timeout=5)
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        self.batch_release.wait(timeout=5)
        return [[0.1, 0.2, 0.3] for _ in texts]


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
            mock_config = SimpleNamespace()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            mock_ov.TextEmbeddingPipeline.Config.return_value = mock_config

            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                device="NPU"
            )
            
            assert engine._model_path == str(model_path)
            assert engine._device == "NPU"

    def test_npu_uses_static_shape_profile_for_validated_model(self, tmp_path):
        """Validated NPU models should load with static-shape pipeline properties."""
        model_path = tmp_path / "all-minilm-l6-v2"
        model_path.mkdir()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_config = SimpleNamespace()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            mock_ov.TextEmbeddingPipeline.Config.return_value = mock_config

            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                device="NPU",
                model_name="all-minilm-l6-v2",
                requested_model="sentence-transformers/all-MiniLM-L6-v2",
                resolved_model="all-minilm-l6-v2",
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
            )

            assert engine.get_engine_info()["pipeline_properties"] == {
                "batch_size": 1,
                "max_length": 256,
                "pad_to_max_length": True,
            }
            assert mock_config.batch_size == 1
            assert mock_config.max_length == 256
            assert mock_config.pad_to_max_length is True
            mock_ov.TextEmbeddingPipeline.assert_called_once_with(
                str(model_path),
                "NPU",
                mock_config,
            )

    def test_npu_without_validated_profile_uses_default_pipeline_shape(self, tmp_path):
        """Unvalidated NPU models should keep the existing default pipeline call."""
        model_path = tmp_path / "bge-small"
        model_path.mkdir()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline

            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                device="NPU",
                model_name="bge-small",
                requested_model="BAAI/bge-small-en-v1.5",
                resolved_model="bge-small",
                repo_id="BAAI/bge-small-en-v1.5",
            )

            assert "pipeline_properties" not in engine.get_engine_info()
            mock_ov.TextEmbeddingPipeline.assert_called_once_with(
                str(model_path),
                "NPU",
            )

    def test_embed_single_returns_list_of_floats(self, tmp_path):
        """embed() should return a list of floats."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_pipeline.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=4)
            result = engine.embed("test text")
            
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)

    def test_embed_batch_returns_list_of_embeddings(self, tmp_path):
        """embed_batch() should return a list of embedding lists."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_pipeline.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            
            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=3)
            results = engine.embed_batch(["text1", "text2"])
            
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(emb, list) for emb in results)
            assert all(isinstance(x, float) for emb in results for x in emb)

    def test_embed_batch_rejects_backend_result_count_mismatch(self, tmp_path):
        """embed_batch() must not leave missing non-empty results as zero vectors."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_pipeline.embed_documents.return_value = [[0.1, 0.2, 0.3]]
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline

            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=3)
            with pytest.raises(EmbeddingInferenceError, match="returned 1 result"):
                engine.embed_batch(["text1", "text2"])

    def test_embed_rejects_wrong_dimension_or_non_finite_values(self, tmp_path):
        """Production embeddings must match configured dimensions and be finite."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_pipeline.embed_query.return_value = [0.1, float("nan"), 0.3]
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline

            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=3)
            with pytest.raises(EmbeddingInferenceError, match="non-finite"):
                engine.embed("text")

    def test_npu_static_batch_size_chunks_documents(self, tmp_path):
        """Static NPU batch size should be honored during document embedding."""
        model_path = tmp_path / "all-minilm-l6-v2"
        model_path.mkdir()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_pipeline.embed_documents.side_effect = lambda texts: [[float(len(text))] for text in texts]
            mock_config = SimpleNamespace()
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline
            mock_ov.TextEmbeddingPipeline.Config.return_value = mock_config

            engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                device="NPU",
                model_name="all-minilm-l6-v2",
                dimensions=1,
            )
            assert engine.embed_batch(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]

            assert mock_pipeline.embed_documents.call_args_list == [
                ((["a"],),),
                ((["bb"],),),
                ((["ccc"],),),
            ]

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

    def test_init_timeout_returns_promptly(self, tmp_path):
        """Model load timeout should return without waiting for worker completion."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        loader = BlockingPipelineLoader()

        with patch(
            "npu_proxy.inference.embedding_engine.openvino_genai",
            SimpleNamespace(TextEmbeddingPipeline=loader),
        ), patch("npu_proxy.inference.embedding_engine.DEFAULT_LOAD_TIMEOUT", 0.05):
            started_at = time.perf_counter()
            engine = ProductionEmbeddingEngine(model_path=str(model_path))
            assert time.perf_counter() - started_at < 0.5

        try:
            info = engine.get_engine_info()
            assert info["available"] is False
            assert "Model load timed out after 0.05s" in info["load_error"]
            with pytest.raises(EmbeddingUnavailableError):
                engine.embed("test text")
        finally:
            loader.release.set()
            if engine._active_future is not None:
                engine._active_future.result(timeout=1)
            engine.shutdown(wait=False)

    def test_embed_timeout_returns_promptly(self, tmp_path):
        """Embedding timeout should return promptly and keep runtime timeout semantics."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        pipeline = BlockingEmbeddingPipeline()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_ov.TextEmbeddingPipeline.return_value = pipeline
            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=3)

        try:
            started_at = time.perf_counter()
            with pytest.raises(EmbeddingTimeoutError, match="Embedding timed out after 0.05s"):
                engine.embed("hello", timeout=0.05)
            assert time.perf_counter() - started_at < 0.5

            started_at = time.perf_counter()
            with pytest.raises(EmbeddingTimeoutError, match="still running"):
                engine.embed("again", timeout=0.05)
            assert time.perf_counter() - started_at < 0.2

            pipeline.query_release.set()
            if engine._active_future is not None:
                engine._active_future.result(timeout=1)

            assert engine.embed("recovered", timeout=0.2) == [0.1, 0.2, 0.3]
        finally:
            pipeline.query_release.set()
            engine.shutdown(wait=False)

    def test_embed_batch_timeout_returns_promptly(self, tmp_path):
        """Batch timeout should return promptly instead of waiting for worker shutdown."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        pipeline = BlockingEmbeddingPipeline()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_ov.TextEmbeddingPipeline.return_value = pipeline
            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=3)

        try:
            started_at = time.perf_counter()
            with pytest.raises(
                EmbeddingTimeoutError,
                match="Batch embedding timed out after 0.05s",
            ):
                engine.embed_batch(["hello", "world"], timeout=0.05)
            assert time.perf_counter() - started_at < 0.5

            pipeline.batch_release.set()
            if engine._active_future is not None:
                engine._active_future.result(timeout=1)

            assert engine.embed_batch(["hello", "world"], timeout=0.2) == [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
            ]
        finally:
            pipeline.batch_release.set()
            engine.shutdown(wait=False)

    def test_embed_batch_optimized_uses_one_overall_deadline(self, tmp_path):
        """Optimized batch timeout should shrink across items instead of resetting per text."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_ov.TextEmbeddingPipeline.return_value = MagicMock()
            engine = ProductionEmbeddingEngine(model_path=str(model_path))

        observed_timeouts = []

        def fake_embed(text, timeout=None):
            observed_timeouts.append(timeout)
            time.sleep(0.06)
            if timeout is not None and timeout < 0.06:
                raise EmbeddingTimeoutError(f"Embedding timed out after {timeout:g}s")
            return [float(len(text))]

        try:
            with patch.object(engine, "embed", side_effect=fake_embed):
                started_at = time.perf_counter()
                with pytest.raises(EmbeddingTimeoutError):
                    engine.embed_batch_optimized(["a", "bb", "ccc"], timeout=0.15)
                assert time.perf_counter() - started_at < 0.35

            assert len(observed_timeouts) == 3
            assert observed_timeouts[0] == pytest.approx(0.15, rel=0.3)
            assert observed_timeouts[1] < observed_timeouts[0]
            assert observed_timeouts[2] < 0.06
        finally:
            engine.shutdown(wait=False)

    def test_embedding_cache_returns_copies(self, tmp_path):
        """Cached production embeddings should not expose shared mutable lists."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        with patch("npu_proxy.inference.embedding_engine.openvino_genai") as mock_ov:
            mock_pipeline = MagicMock()
            mock_pipeline.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline

            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=3)
            first = engine.embed("cached text")
            first[0] = 999.0
            second = engine.embed("cached text")

            assert second == [0.1, 0.2, 0.3]
            assert mock_pipeline.embed_query.call_count == 1


class TestEmbeddingEngineFallback:
    """Tests for explicit fallback behavior when embeddings are unavailable."""

    def test_unavailable_model_raises_by_default(self, tmp_path):
        """Real embedding failures should hard-fail by default."""
        model_path = tmp_path / "nonexistent_model"

        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "disabled"}), patch(
            "npu_proxy.inference.embedding_engine.openvino_genai"
        ) as mock_ov:
            mock_ov.TextEmbeddingPipeline.side_effect = Exception("Model not found")

            engine = ProductionEmbeddingEngine(model_path=str(model_path))
            info = engine.get_engine_info()

            assert info["is_production"] is False
            assert info["is_fallback"] is False
            assert info["available"] is False
            assert "Model not found" in info["load_error"]
            with pytest.raises(EmbeddingUnavailableError):
                engine.embed("test text")

    def test_missing_openvino_raises_unavailable_by_default(self, tmp_path):
        """Missing OpenVINO should raise EmbeddingUnavailableError by default."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "disabled"}), patch(
            "npu_proxy.inference.embedding_engine._get_openvino_genai",
            side_effect=ImportError("no openvino"),
        ):
            engine = ProductionEmbeddingEngine(model_path=str(model_path))

            info = engine.get_engine_info()
            assert info["available"] is False
            assert info["is_fallback"] is False
            assert "OpenVINO GenAI not available" in info["load_error"]
            with pytest.raises(EmbeddingUnavailableError):
                engine.embed_batch(["test text"])

    def test_runtime_batch_failure_raises_inference_error_without_degrading(self, tmp_path):
        """Runtime batch failures should stay 500-class and not permanently degrade by default."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "disabled"}), patch(
            "npu_proxy.inference.embedding_engine.openvino_genai"
        ) as mock_ov:
            mock_pipeline = MagicMock()
            mock_pipeline.embed_documents.side_effect = Exception("boom")
            mock_ov.TextEmbeddingPipeline.return_value = mock_pipeline

            engine = ProductionEmbeddingEngine(model_path=str(model_path), dimensions=2)

            with pytest.raises(EmbeddingInferenceError):
                engine.embed_batch(["text1", "text2"])

            info = engine.get_engine_info()
            assert info["is_fallback"] is False
            assert info["available"] is True

            mock_pipeline.embed_documents.side_effect = None
            mock_pipeline.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
            results = engine.embed_batch(["text1", "text2"])
            assert results == [[0.1, 0.2], [0.3, 0.4]]

    def test_fallback_requires_explicit_opt_in(self, tmp_path):
        """Hash fallback remains available only when explicitly enabled."""
        model_path = tmp_path / "nonexistent_model"

        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}), patch(
            "npu_proxy.inference.embedding_engine.openvino_genai"
        ) as mock_ov:
            mock_ov.TextEmbeddingPipeline.side_effect = Exception("Model not found")

            engine = ProductionEmbeddingEngine(model_path=str(model_path))
            result = engine.embed("test text")
            info = engine.get_engine_info()

            assert isinstance(result, list)
            assert len(result) == DEFAULT_EMBEDDING_DIMENSIONS
            assert info["is_fallback"] is True
            assert info["fallback_allowed"] is True

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
        assert result.name in {"bge-small", "BAAI_bge-small-en-v1.5"}

    def test_is_embedding_model_downloaded_false_when_missing(self):
        """is_embedding_model_downloaded() should return False for missing model."""
        result = is_embedding_model_downloaded("nonexistent/model-xyz-12345")
        assert result is False

    def test_unknown_repo_uses_sanitized_full_repo_path(self):
        """Unknown Hugging Face repos should keep unique canonical storage keys."""
        result = get_embedding_model_path("org-one/custom-embed")

        assert result.name == "org-one%2Fcustom-embed"

    def test_embedding_keyword_repo_is_classified_as_embedding(self):
        assert is_known_embedding_model("org-one/custom-embed") is True

    def test_catalog_alias_without_registry_id_uses_repo_storage_key(self, tmp_path):
        """Alias-only catalog entries should resolve to canonical repo-derived storage keys."""
        config = resolve_embedding_model_config(model_name="bge-base", model_dir=tmp_path)

        assert config.resolved_model == "bge-base"
        assert config.storage_key == "BAAI%2Fbge-base-en-v1.5"
        assert config.canonical_path.name == "BAAI%2Fbge-base-en-v1.5"

    def test_embedding_config_detects_legacy_alias_directory(self, tmp_path):
        """Canonical embedding resolution should still honor older alias-based cache paths."""
        legacy_dir = tmp_path / "embeddings" / "bge-base"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (legacy_dir / "openvino_model.bin").write_bytes(b"weights")

        config = resolve_embedding_model_config(model_name="bge-base", model_dir=tmp_path)

        assert config.is_downloaded is True
        assert config.model_path == legacy_dir
        assert legacy_dir in config.legacy_paths

    @pytest.mark.parametrize(
        "model_name",
        [
            "../outside",
            "..\\outside",
            "C:\\models\\outside",
            "\\\\server\\share\\model",
            "tinyllama",
        ],
    )
    def test_invalid_model_identifier_is_rejected(self, model_name):
        """Embedding resolution must reject path-like model identifiers."""
        with pytest.raises(InvalidEmbeddingModelError):
            resolve_embedding_model_config(model_name=model_name)

    @pytest.mark.parametrize(
        "model_name",
        [
            "org/name/extra",
            "https://huggingface.co/org/name",
            "bad namespace/model",
            "org/bad..model",
            "org/bad--model",
        ],
    )
    def test_invalid_unknown_repo_shape_is_rejected(self, model_name):
        """Unknown repo IDs must be exactly namespace/name, not arbitrary slash strings."""
        with pytest.raises(InvalidEmbeddingModelError):
            resolve_embedding_model_config(model_name=model_name)


class TestEmbeddingEngineSingleton:
    """Tests for the singleton get_embedding_engine pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        _reset_embedding_engine()

    def teardown_method(self):
        """Reset singleton after each test."""
        _reset_embedding_engine()


    def test_failed_production_load_is_cached_temporarily_and_shutdown(self):
        """Failed production loads should not be recreated on every request."""
        config = SimpleNamespace(
            resolved_model="bge-small",
            device="CPU",
            requested_model="bge-small",
            requested_device="CPU",
            model_path=Path(r"C:\models\bge-small"),
            canonical_path=Path(r"C:\models\bge-small"),
            dimensions=384,
            repo_id="BAAI/bge-small-en-v1.5",
            is_downloaded=True,
        )
        pipeline_factory = Mock(side_effect=Exception(r"Failed to load C:\secret\path"))

        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "disabled"}), patch(
            "npu_proxy.inference.embedding_engine.embedding_config.resolve_embedding_model_config",
            return_value=config,
        ), patch(
            "npu_proxy.inference.embedding_engine.openvino_genai",
            SimpleNamespace(TextEmbeddingPipeline=pipeline_factory),
        ):
            with pytest.raises(EmbeddingUnavailableError):
                get_embedding_engine(model_name="bge-small")
            with pytest.raises(EmbeddingUnavailableError):
                get_embedding_engine(model_name="bge-small")

        assert pipeline_factory.call_count == 1

    def test_failed_production_load_uses_cached_hash_fallback_during_cooldown(self):
        """Cooldown should prevent retry storms even when hash fallback can serve."""
        config = SimpleNamespace(
            resolved_model="bge-small",
            device="CPU",
            requested_model="bge-small",
            requested_device="CPU",
            model_path=Path(r"C:\models\bge-small"),
            canonical_path=Path(r"C:\models\bge-small"),
            dimensions=384,
            repo_id="BAAI/bge-small-en-v1.5",
            is_downloaded=True,
        )
        pipeline_factory = Mock(side_effect=Exception(r"Failed to load C:\secret\path"))

        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}), patch(
            "npu_proxy.inference.embedding_engine.embedding_config.resolve_embedding_model_config",
            return_value=config,
        ), patch(
            "npu_proxy.inference.embedding_engine.openvino_genai",
            SimpleNamespace(TextEmbeddingPipeline=pipeline_factory),
        ):
            engine1 = get_embedding_engine(model_name="bge-small")
            engine2 = get_embedding_engine(model_name="bge-small")

        assert engine1 is engine2
        assert pipeline_factory.call_count == 1

    def test_get_embedding_engine_returns_engine(self):
        """get_embedding_engine() should hard-fail when model is unavailable by default."""
        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "disabled"}):
            with pytest.raises(EmbeddingUnavailableError):
                get_embedding_engine(model_name="org-test/missing-embed-default")

    def test_get_embedding_engine_returns_engine_when_fallback_enabled(self):
        """get_embedding_engine() should still support explicit operator fallback."""
        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}):
            engine = get_embedding_engine(model_name="org-test/missing-embed-enabled")
            assert engine is not None
            assert hasattr(engine, "embed")
            assert hasattr(engine, "embed_batch")

    def test_get_embedding_engine_returns_same_instance(self):
        """get_embedding_engine() should return the same instance (singleton)."""
        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}):
            engine1 = get_embedding_engine(model_name="org-test/missing-embed-singleton")
            engine2 = get_embedding_engine(model_name="org-test/missing-embed-singleton")
            assert engine1 is engine2

    def test_get_embedding_engine_keys_by_model_and_device(self):
        """Different model/device pairs should get distinct cached engines."""
        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}):
            engine1 = get_embedding_engine(model_name="org-test/cache-cpu", device="CPU")
            engine2 = get_embedding_engine(model_name="org-test/cache-cpu", device="CPU")
            engine3 = get_embedding_engine(model_name="org-test/cache-cpu", device="NPU")
            engine4 = get_embedding_engine(model_name="org-test/cache-other", device="CPU")

            assert engine1 is engine2
            assert engine1 is not engine3
            assert engine1 is not engine4

    def test_get_embedding_engine_uses_model_dimensions(self):
        """Requested embedding model metadata should drive engine dimensions."""
        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}):
            engine = get_embedding_engine(model_name="e5-large")

            assert engine.dimensions == 1024
            info = engine.get_engine_info()
            assert info["resolved_model"] == "e5-large"
            assert info["requested_model"] == "e5-large"

    def test_get_embedding_engine_separates_unknown_repo_ids(self):
        """Different uncatalogued repos should not share an engine cache entry."""
        with patch.dict(os.environ, {EMBEDDING_FALLBACK_MODE_ENV_VAR: "hash"}):
            engine1 = get_embedding_engine(model_name="org-one/custom-embed", device="CPU")
            engine2 = get_embedding_engine(model_name="org-two/custom-embed", device="CPU")

            assert engine1 is not engine2
