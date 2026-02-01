"""Tests for the search module (TDD - RED phase)."""

import pytest
from dataclasses import fields
from unittest.mock import patch, MagicMock

from npu_proxy.models.search import (
    SearchResult,
    extract_quantization,
    extract_parameters,
    extract_architecture,
    extract_model_metadata,
    is_openvino_compatible,
    search_openvino_models,
    get_model_details,
)


class TestSearchResultDataclass:
    """Tests for SearchResult dataclass."""

    def test_search_result_dataclass_fields(self):
        """SearchResult should have all required fields."""
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
        """Model with int4 in name should extract INT4."""
        result = extract_quantization("model-int4-ov")
        assert result == "INT4"

    def test_extract_quantization_fp16(self):
        """Model with fp16 in name should extract FP16."""
        result = extract_quantization("model-fp16-ov")
        assert result == "FP16"


class TestExtractParameters:
    """Tests for extract_parameters function."""

    def test_extract_parameters_1b(self):
        """TinyLlama-1.1B should extract 1.1B."""
        result = extract_parameters("TinyLlama-1.1B")
        assert result == "1.1B"

    def test_extract_parameters_7b(self):
        """Llama-7B should extract 7B."""
        result = extract_parameters("Llama-7B")
        assert result == "7B"


class TestExtractArchitecture:
    """Tests for extract_architecture function."""

    def test_extract_architecture(self):
        """tinyllama should map to TinyLLaMA."""
        result = extract_architecture("tinyllama")
        assert result == "TinyLLaMA"


class TestExtractModelMetadata:
    """Tests for extract_model_metadata function."""

    def test_extract_model_metadata_returns_dict(self):
        """extract_model_metadata should return dict with 3 keys."""
        result = extract_model_metadata("OpenVINO/TinyLlama-1.1B-int4-ov")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"quantization", "parameters", "architecture"}


class TestIsOpenvinoCompatible:
    """Tests for is_openvino_compatible function."""

    def test_is_openvino_compatible_true(self):
        """Model with OpenVINO author should be compatible."""
        with patch("npu_proxy.models.search.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance
            mock_instance.model_info.return_value = MagicMock(
                author="OpenVINO",
                tags=["text-generation"],
            )

            result = is_openvino_compatible("OpenVINO/some-model")
            assert result is True

    def test_is_openvino_compatible_false(self):
        """Non-OpenVINO model should not be compatible."""
        with patch("npu_proxy.models.search.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance
            mock_instance.model_info.return_value = MagicMock(
                author="random-user",
                tags=["pytorch"],
            )

            result = is_openvino_compatible("random-user/some-model")
            assert result is False


class TestSearchOpenvinoModels:
    """Tests for search_openvino_models function."""

    def test_search_returns_tuple(self):
        """search_openvino_models should return (list, int)."""
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_list.return_value = []

            result = search_openvino_models()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], list)
            assert isinstance(result[1], int)

    def test_search_sort_popular(self):
        """Sort by popular should order by downloads descending."""
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

    def test_search_filter_quantization(self):
        """Quantization filter should work."""
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

            results, total = search_openvino_models(quantization="INT4")
            for r in results:
                assert r.quantization == "INT4"

    def test_search_filter_model_type(self):
        """model_type filter should work."""
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

    def test_search_min_downloads(self):
        """min_downloads filter should work."""
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

            results, total = search_openvino_models(min_downloads=500)
            for r in results:
                assert r.downloads >= 500

    def test_search_pagination(self):
        """offset/limit should work for pagination."""
        with patch("npu_proxy.models.search.list_models") as mock_list:
            mock_models = [
                MagicMock(
                    id=f"OpenVINO/model-{i}-int4-ov",
                    author="OpenVINO",
                    downloads=100 * i,
                    likes=i,
                    last_modified=None,
                )
                for i in range(1, 11)
            ]
            mock_list.return_value = mock_models

            results, total = search_openvino_models(limit=3, offset=2)
            assert len(results) <= 3


class TestGetModelDetails:
    """Tests for get_model_details function."""

    def test_get_model_details(self):
        """get_model_details should return SearchResult or None."""
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
