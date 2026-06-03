"""Tests for the mapper module."""

from npu_proxy.models.mapper import (
    OLLAMA_TO_HUGGINGFACE,
    get_ollama_name,
    list_known_models,
    resolve_model_repo,
    resolve_model_storage_key,
    resolve_runtime_model_name,
)


class TestResolveModelRepo:
    """Tests for resolve_model_repo function."""

    def test_resolve_known_model(self, known_ollama_model: str) -> None:
        result = resolve_model_repo(known_ollama_model)
        assert result is not None
        repo_id, local_name = result
        assert local_name == "tinyllama"
        assert repo_id == "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"

    def test_resolve_with_quantization_suffix(self) -> None:
        result = resolve_model_repo("tinyllama:fp16")
        assert result is not None
        repo_id, local_name = result
        assert "fp16" in repo_id.lower()
        assert local_name == "tinyllama:fp16"

    def test_resolve_direct_huggingface_repo(self) -> None:
        result = resolve_model_repo("OpenVINO/phi-2-int4-ov")
        assert result is not None
        repo_id, local_name = result
        assert repo_id == "OpenVINO/phi-2-int4-ov"
        assert local_name == "phi-2-int4-ov"

    def test_resolve_registry_model_id(self) -> None:
        result = resolve_model_repo("tinyllama-1.1b-chat-int4-ov")
        assert result == (
            "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
            "tinyllama-1.1b-chat-int4-ov",
        )

    def test_resolve_unknown_returns_none(self, unknown_model: str) -> None:
        assert resolve_model_repo(unknown_model) is None


class TestGetOllamaName:
    """Tests for get_ollama_name function."""

    def test_get_ollama_name_reverse_lookup(self, known_huggingface_repo: str) -> None:
        assert get_ollama_name(known_huggingface_repo) == "tinyllama"

    def test_get_ollama_name_for_unknown_returns_none(self) -> None:
        assert get_ollama_name("unknown/nonexistent-repo") is None


class TestListKnownModels:
    """Tests for list_known_models function."""

    def test_list_known_models_returns_list(self) -> None:
        result = list_known_models()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_known_models_dict_keys(self) -> None:
        result = list_known_models()
        required_fields = {
            "ollama_name",
            "huggingface_repo",
            "quantization",
            "type",
            "backend",
            "format",
            "task",
        }

        for item in result:
            assert isinstance(item, dict)
            assert required_fields.issubset(item.keys())
            assert item["ollama_name"] is not None
            assert item["huggingface_repo"] is not None
            assert item["quantization"] is not None

    def test_static_mapping_is_derived(self) -> None:
        assert OLLAMA_TO_HUGGINGFACE["tinyllama"][0] == "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"


class TestEmbeddingModelMapping:
    """Tests for embedding model mappings."""

    def test_bge_small_mapped(self) -> None:
        result = resolve_model_repo("bge-small")
        assert result is not None
        assert "bge-small-en-v1.5" in result[0].lower()

    def test_e5_large_mapped(self) -> None:
        result = resolve_model_repo("e5-large")
        assert result is not None
        assert "e5-large" in result[0].lower()

    def test_list_known_models_includes_embeddings(self) -> None:
        models = list_known_models()
        names = [model["ollama_name"] for model in models]
        assert "bge-small" in names or any("bge" in name.lower() for name in names)

    def test_embedding_model_has_type_field(self) -> None:
        models = list_known_models()
        embedding_models = [model for model in models if model.get("type") == "embedding"]
        assert len(embedding_models) >= 1
        assert all(model["task"] == "feature-extraction" for model in embedding_models)


class TestModelStorageIdentity:
    def test_catalog_alias_registry_and_repo_share_storage_key(self) -> None:
        alias_key = resolve_model_storage_key("tinyllama")
        registry_key = resolve_model_storage_key("tinyllama-1.1b-chat-int4-ov")
        repo_key = resolve_model_storage_key("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov")

        assert alias_key == "tinyllama-1.1b-chat-int4-ov"
        assert alias_key == registry_key == repo_key

    def test_unknown_repo_uses_encoded_full_repo_storage_key(self) -> None:
        assert resolve_model_storage_key("attacker/tinyllama") == "attacker%2Ftinyllama"

    def test_unknown_repo_runtime_name_round_trips_to_storage_key(self) -> None:
        assert resolve_runtime_model_name("attacker/tinyllama") == "attacker%2Ftinyllama"

    def test_secondary_alias_shares_canonical_repo_and_storage_with_primary_alias(self) -> None:
        primary_repo = resolve_model_repo("tinyllama:fp16")
        secondary_repo = resolve_model_repo("tinyllama-fp16")

        assert primary_repo is not None
        assert secondary_repo is not None
        assert primary_repo[0] == secondary_repo[0] == "OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov"
        assert resolve_model_storage_key("tinyllama:fp16") == resolve_model_storage_key(
            "tinyllama-fp16"
        )
