"""Focused tests for shared model metadata detectors."""

import pytest

from npu_proxy.models.metadata import (
    DetectedModelMetadata,
    detect_family,
    detect_format,
    detect_model_metadata,
    detect_parameters,
    detect_quantization,
)


def test_detect_family_for_newer_llm_family() -> None:
    assert detect_family("OpenVINO/granite-3.3-2b-instruct-fp8-ov") == "granite"


def test_detect_format_for_gguf_artifacts() -> None:
    assert detect_format("deepseek-r1-q4_k_m.gguf") == "gguf"


def test_detect_quantization_for_fp8() -> None:
    assert detect_quantization("OpenVINO/DeepSeek-R1-Distill-Qwen-7B-fp8-ov") == "FP8"


def test_detect_quantization_for_gguf_variant() -> None:
    assert detect_quantization("DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-GGUF") == "Q4_K_M"


def test_detect_parameters_for_million_scale_models() -> None:
    assert detect_parameters("all-MiniLM-L6-v2-22M") == "22M"


def test_detect_parameters_ignores_context_length_markers() -> None:
    assert detect_parameters("Phi-3-mini-4k-instruct-int4-ov") == ""


def test_detect_model_metadata_for_embedding_task() -> None:
    metadata = detect_model_metadata("Qwen/Qwen3-Embedding-0.6B-int4-ov")
    assert metadata["family"] == "qwen"
    assert metadata["quantization"] == "INT4"
    assert metadata["parameters"] == "0.6B"
    assert metadata["task"] == "feature-extraction"


def test_detect_model_metadata_for_vision_language_models() -> None:
    metadata = detect_model_metadata("OpenVINO/Qwen2.5-VL-3B-Instruct-int4-ov")
    assert metadata["type"] == "vision"
    assert metadata["task"] == "image-text-to-text"


def test_detected_model_metadata_rejects_incompatible_task_contract() -> None:
    with pytest.raises(ValueError):
        DetectedModelMetadata(type="embedding", task="text-generation")
