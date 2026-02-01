"""Pytest fixtures for npu-proxy tests."""

import pytest


@pytest.fixture
def known_ollama_model() -> str:
    """Return a known Ollama-style model name."""
    return "tinyllama"


@pytest.fixture
def known_huggingface_repo() -> str:
    """Return a known HuggingFace repository."""
    return "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"


@pytest.fixture
def unknown_model() -> str:
    """Return an unknown model name."""
    return "nonexistent-model-xyz"
