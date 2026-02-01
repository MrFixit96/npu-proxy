"""Tests for the mapper module."""

import pytest

from npu_proxy.models.mapper import (
    OLLAMA_TO_HUGGINGFACE,
    resolve_model_repo,
    get_ollama_name,
    list_known_models,
)


class TestResolveModelRepo:
    """Tests for resolve_model_repo function."""

    def test_resolve_known_model(self, known_ollama_model: str) -> None:
        """tinyllama should resolve to OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov."""
        result = resolve_model_repo(known_ollama_model)
        assert result is not None
        repo_id, local_name = result
        assert local_name == "tinyllama"
        assert repo_id == "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"

    def test_resolve_with_quantization_suffix(self) -> None:
        """tinyllama:fp16 should resolve to fp16 variant."""
        result = resolve_model_repo("tinyllama:fp16")
        assert result is not None
        repo_id, local_name = result
        assert "fp16" in repo_id.lower()
        assert local_name == "tinyllama:fp16"

    def test_resolve_direct_huggingface_repo(self) -> None:
        """OpenVINO/phi-2-int4-ov should work directly as a HuggingFace repo."""
        result = resolve_model_repo("OpenVINO/phi-2-int4-ov")
        assert result is not None
        repo_id, local_name = result
        assert repo_id == "OpenVINO/phi-2-int4-ov"
        assert local_name == "phi-2-int4-ov"

    def test_resolve_unknown_returns_none(self, unknown_model: str) -> None:
        """nonexistent-model-xyz should return None."""
        result = resolve_model_repo(unknown_model)
        assert result is None


class TestGetOllamaName:
    """Tests for get_ollama_name function."""

    def test_get_ollama_name_reverse_lookup(self, known_huggingface_repo: str) -> None:
        """Reverse lookup should work for known repos."""
        result = get_ollama_name(known_huggingface_repo)
        assert result is not None
        assert result == "tinyllama"

    def test_get_ollama_name_for_unknown_returns_none(self) -> None:
        """Unknown HuggingFace repo should return None."""
        result = get_ollama_name("unknown/nonexistent-repo")
        assert result is None


class TestListKnownModels:
    """Tests for list_known_models function."""

    def test_list_known_models_returns_list(self) -> None:
        """Should return a non-empty list."""
        result = list_known_models()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_known_models_dict_keys(self) -> None:
        """Each item should have ollama_name, huggingface_repo, quantization."""
        result = list_known_models()
        required_fields = {"ollama_name", "huggingface_repo", "quantization"}

        for item in result:
            assert isinstance(item, dict)
            assert required_fields.issubset(item.keys()), (
                f"Missing fields: {required_fields - set(item.keys())}"
            )
            assert item["ollama_name"] is not None
            assert item["huggingface_repo"] is not None
            assert item["quantization"] is not None


class TestEmbeddingModelMapping:
    """Tests for embedding model mappings."""

    def test_bge_small_mapped(self) -> None:
        """bge-small resolves to BAAI/bge-small-en-v1.5."""
        result = resolve_model_repo("bge-small")
        assert result is not None
        assert "bge-small-en-v1.5" in result[0].lower() or "bge-small-en-v1.5" in str(result).lower()

    def test_e5_large_mapped(self) -> None:
        """e5-large resolves to multilingual-e5-large."""
        result = resolve_model_repo("e5-large")
        assert result is not None
        repo_id = result[0]
        assert "e5-large" in repo_id.lower()

    def test_list_known_models_includes_embeddings(self) -> None:
        """list_known_models includes embedding models."""
        models = list_known_models()
        names = [m["ollama_name"] for m in models]
        assert "bge-small" in names or any("bge" in n.lower() for n in names)

    def test_embedding_model_has_type_field(self) -> None:
        """Embedding models have type='embedding' in list_known_models."""
        models = list_known_models()
        embedding_models = [m for m in models if m.get("type") == "embedding"]
        assert len(embedding_models) >= 1
