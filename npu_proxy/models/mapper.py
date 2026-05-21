"""Model name mapping between Ollama-style names and HuggingFace OpenVINO repositories.

This module provides utilities for resolving model names between Ollama-style
short names (e.g., "tinyllama", "phi-2") and their corresponding HuggingFace
repository paths (e.g., "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov").

The mapping enables users to reference models using familiar Ollama naming
conventions while the system internally resolves to the correct HuggingFace
repository for downloading OpenVINO-optimized models.

Model Categories:
    - LLM: Language models for text generation (Llama, Phi, Mistral, etc.)
    - Embedding: Text embedding models for semantic search (BGE, E5, MiniLM)

Mapping Behavior:
    1. Ollama-style names (e.g., "tinyllama") -> mapped to HuggingFace repo
    2. Direct HuggingFace format (e.g., "OpenVINO/phi-2-int4-ov") -> passed through
    3. Version tags (e.g., "tinyllama:fp16") -> mapped to variant if available

Example:
    >>> from npu_proxy.models.mapper import resolve_model_repo
    >>> resolve_model_repo("tinyllama")
    ('OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov', 'tinyllama')

    >>> from npu_proxy.models.mapper import list_known_models
    >>> models = list_known_models()
    >>> len(models) > 0
    True

    >>> from npu_proxy.models.mapper import get_ollama_name
    >>> get_ollama_name("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov")
    'tinyllama'

Note:
    To add new model mappings, update the OLLAMA_TO_HUGGINGFACE dictionary.
    The reverse mapping (REVERSE_MAPPING) is generated automatically.
"""

from __future__ import annotations

# Ollama-style model names to HuggingFace OpenVINO repository mappings
# Values are tuples of (huggingface_repo, model_type) where model_type is "llm" or "embedding"
OLLAMA_TO_HUGGINGFACE: dict[str, tuple[str, str]] = {
    # LLM models
    "tinyllama": ("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", "llm"),
    "tinyllama:fp16": ("OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov", "llm"),
    "phi-2": ("OpenVINO/phi-2-int4-ov", "llm"),
    "phi-3": ("OpenVINO/Phi-3-mini-4k-instruct-int4-ov", "llm"),
    "llama2": ("OpenVINO/llama-2-7b-chat-int4-ov", "llm"),
    "llama2:13b": ("OpenVINO/llama-2-13b-chat-int4-ov", "llm"),
    "llama3.2": ("OpenVINO/Llama-3.2-3B-Instruct-int4-ov", "llm"),
    "mistral": ("OpenVINO/mistral-7b-instruct-v0.1-int4-ov", "llm"),
    "qwen2": ("OpenVINO/Qwen2-1.5B-Instruct-int4-ov", "llm"),
    "gemma": ("OpenVINO/gemma-2b-it-int4-ov", "llm"),
    # Embedding models
    "bge-small": ("BAAI/bge-small-en-v1.5", "embedding"),
    "bge-base": ("BAAI/bge-base-en-v1.5", "embedding"),
    "bge-large": ("BAAI/bge-large-en-v1.5", "embedding"),
    "e5-small": ("intfloat/e5-small-v2", "embedding"),
    "e5-large": ("intfloat/multilingual-e5-large", "embedding"),
    "all-minilm": ("sentence-transformers/all-MiniLM-L6-v2", "embedding"),
    "nomic-embed-text": ("nomic-ai/nomic-embed-text-v1.5", "embedding"),
}

# Reverse mapping: HuggingFace repo to Ollama-style name
REVERSE_MAPPING: dict[str, str] = {v[0]: k for k, v in OLLAMA_TO_HUGGINGFACE.items()}


def resolve_model_repo(name: str) -> tuple[str, str] | None:
    """Resolve a model name to a HuggingFace repository.

    Args:
        name: Ollama-style model name or direct HuggingFace repo (org/repo format).

    Returns:
        Tuple of (repo_id, local_model_name) or None if not found.

    Examples:
        >>> resolve_model_repo("tinyllama")
        ('OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov', 'tinyllama')

        >>> resolve_model_repo("OpenVINO/phi-2-int4-ov")
        ('OpenVINO/phi-2-int4-ov', 'phi-2-int4-ov')
    """
    # Check static mappings first
    if name in OLLAMA_TO_HUGGINGFACE:
        repo_id, _ = OLLAMA_TO_HUGGINGFACE[name]
        return (repo_id, name)

    # Support direct HuggingFace repo format (org/repo)
    if "/" in name:
        # Extract local name from repo path
        local_name = name.split("/")[-1]
        return (name, local_name)

    return None


def get_ollama_name(repo_id: str) -> str | None:
    """Get the Ollama-style name for a HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID
            (e.g., "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov").

    Returns:
        Ollama-style name or None if not found.

    Examples:
        >>> get_ollama_name("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov")
        'tinyllama'
    """
    return REVERSE_MAPPING.get(repo_id)


def _extract_quantization(repo_name: str) -> str:
    """Extract quantization type from repository name.

    Args:
        repo_name: Repository name containing quantization info.

    Returns:
        Quantization type string (fp16, int4, int8, or unknown).
    """
    repo_lower = repo_name.lower()
    if "fp16" in repo_lower:
        return "fp16"
    if "int4" in repo_lower:
        return "int4"
    if "int8" in repo_lower:
        return "int8"
    return "unknown"


def list_known_models() -> list[dict[str, str]]:
    """List all known model mappings.

    Returns:
        List of dicts with ollama_name, huggingface_repo, quantization, and type keys.

    Examples:
        >>> models = list_known_models()
        >>> models[0].keys()
        dict_keys(['ollama_name', 'huggingface_repo', 'quantization', 'type'])
    """
    models: list[dict[str, str]] = []
    for ollama_name, (hf_repo, model_type) in OLLAMA_TO_HUGGINGFACE.items():
        models.append({
            "ollama_name": ollama_name,
            "huggingface_repo": hf_repo,
            "quantization": _extract_quantization(hf_repo),
            "type": model_type,
        })
    return models
