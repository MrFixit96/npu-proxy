"""Tests for the search module."""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from npu_proxy.models.search import (
    SearchResult,
    _cached_search,
    extract_architecture,
    extract_model_metadata,
    extract_parameters,
    extract_quantization,
    get_model_details,
    is_openvino_compatible,
    search_openvino_models,
)


@pytest.fixture(autouse=True)
def clear_search_cache() -> None:
    _cached_search.cache_clear()
    yield
    _cached_search.cache_clear()


class TestSearchResultDataclass:
    """Tests for SearchResult dataclass."""

    def test_search_result_dataclass_fields(self):
        expected_fields = {
            "id",
            "name",
            "author",
            "downloads",
            "likes",
            "last_modified",
            "quantization",
            "parameters",
            "architecture",
        }
        actual_fields = {f.name for f in fields(SearchResult)}
        assert expected_fields == actual_fields


class TestExtractQuantization:
    """Tests for extract_quantization function."""

    def test_extract_quantization_int4(self):
        assert extract_quantization("model-int4-ov") == "INT4"

    def test_extract_quantization_fp16(self):
        assert extract_quantization("model-fp16-ov") == "FP16"

    def test_extract_quantization_fp8(self):
        assert extract_quantization("model-fp8-ov") == "FP8"

    def test_extract_quantization_gguf_variant(self):
        assert extract_quantization("DeepSeek-R1-Q4_K_M-GGUF") == "Q4_K_M"


class TestExtractParameters:
    """Tests for extract_parameters function."""

    def test_extract_parameters_1b(self):
        assert extract_parameters("TinyLlama-1.1B") == "1.1B"

    def test_extract_parameters_7b(self):
        assert extract_parameters("Llama-7B") == "7B"

    def test_extract_parameters_22m(self):
        assert extract_parameters("all-MiniLM-L6-v2-22M") == "22M"


class TestExtractArchitecture:
    """Tests for extract_architecture function."""

    def test_extract_architecture_tinyllama(self):
        assert extract_architecture("tinyllama") == "TinyLLaMA"

    def test_extract_architecture_granite(self):
        assert extract_architecture("granite-3.3") == "Granite"


class TestExtractModelMetadata:
    """Tests for extract_model_metadata function."""

    def test_extract_model_metadata_returns_dict(self):
        result = extract_model_metadata("OpenVINO/TinyLlama-1.1B-int4-ov")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"quantization", "parameters", "architecture"}


class TestIsOpenvinoCompatible:
    """Tests for is_openvino_compatible function."""

    def test_is_openvino_compatible_true(self):
        with patch("npu_proxy.models.search.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance
            mock_instance.model_info.return_value = MagicMock(
                author="OpenVINO",
                tags=["text-generation"],
            )

            assert is_openvino_compatible("OpenVINO/some-model") is True

    def test_is_openvino_compatible_false(self):
        with patch("npu_proxy.models.search.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance
            mock_instance.model_info.return_value = MagicMock(
                author="random-user",
                tags=["pytorch"],
            )

            assert is_openvino_compatible("random-user/some-model") is False


class TestSearchOpenvinoModels:
    """Tests for search_openvino_models function."""

    def test_search_returns_tuple(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_list.return_value = []

            result = search_openvino_models()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], list)
            assert isinstance(result[1], int)

    def test_search_sort_popular(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_model_1 = MagicMock(
                id="OpenVINO/model-a-int4-ov",
                author="OpenVINO",
                downloads=100,
                likes=5,
                last_modified=None,
            )
            mock_model_2 = MagicMock(
                id="OpenVINO/model-b-int4-ov",
                author="OpenVINO",
                downloads=500,
                likes=10,
                last_modified=None,
            )
            mock_list.return_value = [mock_model_2, mock_model_1]

            results, total = search_openvino_models(sort="popular")
            assert total >= 0
            assert results[0].downloads >= results[-1].downloads

    def test_search_filter_quantization(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_model_int4 = MagicMock(
                id="OpenVINO/model-int4-ov",
                author="OpenVINO",
                downloads=100,
                likes=5,
                last_modified=None,
            )
            mock_model_fp16 = MagicMock(
                id="OpenVINO/model-fp16-ov",
                author="OpenVINO",
                downloads=200,
                likes=10,
                last_modified=None,
            )
            mock_list.return_value = [mock_model_int4, mock_model_fp16]

            results, _ = search_openvino_models(quantization="INT4")
            assert {result.quantization for result in results} == {"INT4"}

    def test_search_filter_quantization_fp8(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_model_fp8 = MagicMock(
                id="OpenVINO/DeepSeek-R1-Distill-Qwen-7B-fp8-ov",
                author="OpenVINO",
                downloads=100,
                likes=5,
                last_modified=None,
            )
            mock_model_int4 = MagicMock(
                id="OpenVINO/TinyLlama-1.1B-Chat-int4-ov",
                author="OpenVINO",
                downloads=200,
                likes=10,
                last_modified=None,
            )
            mock_list.return_value = [mock_model_fp8, mock_model_int4]

            results, _ = search_openvino_models(quantization="FP8")
            assert len(results) == 1
            assert results[0].quantization == "FP8"

    def test_search_filter_model_type(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_model_llm = MagicMock(
                id="OpenVINO/llama-7b-int4-ov",
                author="OpenVINO",
                downloads=100,
                likes=5,
                last_modified=None,
            )
            mock_model_embed = MagicMock(
                id="OpenVINO/bge-embedding-int8-ov",
                author="OpenVINO",
                downloads=50,
                likes=2,
                last_modified=None,
            )
            mock_list.return_value = [mock_model_llm, mock_model_embed]

            results, total = search_openvino_models(model_type="llm")
            assert total >= 0
            assert all(result.architecture for result in results)

    def test_search_filter_model_type_with_newer_family(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_model_llm = MagicMock(
                id="OpenVINO/DeepSeek-R1-Distill-Qwen-7B-fp8-ov",
                author="OpenVINO",
                downloads=100,
                likes=5,
                last_modified=None,
            )
            mock_model_embed = MagicMock(
                id="OpenVINO/Qwen3-Embedding-0.6B-int4-ov",
                author="OpenVINO",
                downloads=50,
                likes=2,
                last_modified=None,
            )
            mock_list.return_value = [mock_model_llm, mock_model_embed]

            results, _ = search_openvino_models(model_type="llm")
            assert len(results) == 1
            assert results[0].architecture in {"DeepSeek", "Qwen"}

    def test_search_min_downloads(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_model_low = MagicMock(
                id="OpenVINO/model-a-int4-ov",
                author="OpenVINO",
                downloads=10,
                likes=1,
                last_modified=None,
            )
            mock_model_high = MagicMock(
                id="OpenVINO/model-b-int4-ov",
                author="OpenVINO",
                downloads=1000,
                likes=50,
                last_modified=None,
            )
            mock_list.return_value = [mock_model_low, mock_model_high]

            results, _ = search_openvino_models(min_downloads=500)
            assert all(result.downloads >= 500 for result in results)

    def test_search_pagination(self):
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_models = [
                MagicMock(
                    id=f"OpenVINO/model-{index}-int4-ov",
                    author="OpenVINO",
                    downloads=100 * index,
                    likes=index,
                    last_modified=None,
                )
                for index in range(1, 11)
            ]
            mock_list.return_value = mock_models

            results, _ = search_openvino_models(limit=3, offset=2)
            assert len(results) <= 3


class TestGetModelDetails:
    """Tests for get_model_details function."""

    def test_get_model_details(self):
        with patch("npu_proxy.models.search.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance
            mock_instance.model_info.return_value = MagicMock(
                id="OpenVINO/TinyLlama-1.1B-int4-ov",
                author="OpenVINO",
                downloads=1000,
                likes=50,
                last_modified=None,
            )

            result = get_model_details("OpenVINO/TinyLlama-1.1B-int4-ov")
            assert result is None or isinstance(result, SearchResult)
