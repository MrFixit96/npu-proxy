"""Model registry for NPU Proxy.

This module maintains a registry of available OpenVINO-optimized models.
The registry provides model metadata, aliases, and path resolution.

Model Information:
    Each model entry contains:
    - id: Unique model identifier (e.g., 'tinyllama-1.1b-chat-int4-ov')
    - family: Model architecture family (llama, mistral, phi, bert, qwen, granite)
    - parameter_size: Human-readable size (1.1B, 7B, 22M, etc.)
    - quantization: Quantization format (INT4, INT8, FP16, FP32)
    - size: Approximate file size in bytes
    - context_length: Maximum context window (optional, model-dependent)
    - type: Model type - 'llm' for language models, 'embedding' for embeddings
    - owned_by: Organization that created/optimized the model
    - digest: SHA256 digest for model verification
    
    Embedding models also include:
    - dimensions: Output embedding vector dimensions (384, 1024, 4096)
    - description: Human-readable model description
    - hf_repo: HuggingFace repository path for downloads

Data Structures:
    MODELS_INFO (dict[str, dict]):
        Primary registry mapping model IDs to their metadata dictionaries.
        Keys are canonical model identifiers (e.g., 'tinyllama-1.1b-chat-int4-ov').
        Values are dictionaries containing all model metadata fields.

    DEFAULT_MODEL_DIR (Path):
        Default filesystem path where models are cached locally.
        Defaults to ~/.cache/npu-proxy/models

Adding New Models:
    1. Add entry to MODELS_INFO dictionary with unique model ID as key
    2. Include all required metadata fields:
       - id, size, digest, parameter_size, quantization, family, type, owned_by
    3. For embedding models, also include: dimensions, context_length, description, hf_repo
    4. For LLM models, optionally include: context_length

Example:
    >>> from npu_proxy.models.registry import get_model_info
    >>> info = get_model_info("tinyllama-1.1b-chat-int4-ov")
    >>> info["family"]
    'llama'
    >>> info["quantization"]
    'INT4'

    >>> from npu_proxy.models.registry import list_embedding_models
    >>> embeddings = list_embedding_models()
    >>> len(embeddings) >= 1
    True
"""

import time
from pathlib import Path
from typing import Optional

# Default model directory - where models are cached locally
DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"
"""Default filesystem path for cached models (~/.cache/npu-proxy/models)."""

# Built-in model metadata (known models with full info)
# Each entry maps a model ID to its complete metadata dictionary
MODELS_INFO: dict[str, dict] = {
    "tinyllama-1.1b-chat-int4-ov": {
        "id": "tinyllama-1.1b-chat-int4-ov",
        "size": 637_000_000,
        "digest": "sha256:tinyllama1.1b",
        "parameter_size": "1.1B",
        "quantization": "INT4",
        "family": "llama",
        "type": "llm",
        "owned_by": "openvino",
    },
    "mistral-7b-int4-ov": {
        "id": "mistral-7b-int4-ov",
        "size": 4_000_000_000,
        "digest": "sha256:mistral7b",
        "parameter_size": "7B",
        "quantization": "INT4",
        "family": "mistral",
        "type": "llm",
        "context_length": 8192,
        "owned_by": "openvino",
    },
    "granite-4-micro-ov": {
        "id": "granite-4-micro-ov",
        "size": 3_406_770_588,
        "digest": "sha256:granite4micro",
        "parameter_size": "1B",
        "quantization": "FP32",
        "family": "granite",
        "type": "llm",
        "context_length": 8192,
        "owned_by": "openvino",
    },
    "phi-2-int4-ov": {
        "id": "phi-2-int4-ov",
        "size": 1_400_000_000,
        "digest": "sha256:phi2",
        "parameter_size": "2.7B",
        "quantization": "INT4",
        "family": "phi",
        "type": "llm",
        "owned_by": "openvino",
    },
    "all-minilm-l6-v2": {
        "id": "all-minilm-l6-v2",
        "name": "All-MiniLM-L6-v2",
        "size": 91_000_000,
        "digest": "sha256:minilm",
        "parameter_size": "22M",
        "quantization": "FP32",
        "family": "bert",
        "type": "embedding",
        "dimensions": 384,
        "context_length": 256,
        "description": "Lightweight sentence embeddings",
        "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "owned_by": "sentence-transformers",
    },
    "bge-small": {
        "id": "bge-small",
        "name": "BGE Small English",
        "size": 133_000_000,
        "digest": "sha256:bgesmall",
        "parameter_size": "33M",
        "quantization": "FP32",
        "family": "bert",
        "type": "embedding",
        "dimensions": 384,
        "context_length": 512,
        "description": "Fast, high-quality English embeddings",
        "hf_repo": "BAAI/bge-small-en-v1.5",
        "owned_by": "BAAI",
    },
    "e5-large": {
        "id": "e5-large",
        "name": "E5 Large Multilingual",
        "size": 2_200_000_000,
        "digest": "sha256:e5large",
        "parameter_size": "560M",
        "quantization": "FP32",
        "family": "bert",
        "type": "embedding",
        "dimensions": 1024,
        "context_length": 512,
        "description": "High-quality multilingual embeddings",
        "hf_repo": "intfloat/multilingual-e5-large",
        "owned_by": "intfloat",
    },
    "qwen3-embedding-0.6b-int4-ov": {
        "id": "qwen3-embedding-0.6b-int4-ov",
        "name": "Qwen3 Embedding 0.6B",
        "size": 400_000_000,
        "digest": "sha256:qwen3emb06b",
        "parameter_size": "0.6B",
        "quantization": "INT4",
        "family": "qwen",
        "type": "embedding",
        "dimensions": 1024,
        "context_length": 8192,
        "description": "Qwen3 text embeddings - small",
        "hf_repo": "Qwen/Qwen3-Embedding-0.6B",
        "owned_by": "Qwen",
    },
    "qwen3-embedding-8b-int4-ov": {
        "id": "qwen3-embedding-8b-int4-ov",
        "name": "Qwen3 Embedding 8B",
        "size": 4_500_000_000,
        "digest": "sha256:qwen3emb8b",
        "parameter_size": "8B",
        "quantization": "INT4",
        "family": "qwen",
        "type": "embedding",
        "dimensions": 4096,
        "context_length": 8192,
        "description": "Qwen3 text embeddings - large",
        "hf_repo": "Qwen/Qwen3-Embedding-8B",
        "owned_by": "Qwen",
    },
}


def get_model_info(model_id: str) -> Optional[dict]:
    """Get model metadata by ID.

    Retrieves complete metadata for a model by its identifier. First checks
    the built-in MODELS_INFO registry, then falls back to scanning the local
    model directory for dynamically discovered models.

    Args:
        model_id: The unique model identifier (e.g., 'tinyllama-1.1b-chat-int4-ov').
            Must match either a key in MODELS_INFO or a directory name in the
            model cache that contains an 'openvino_model.xml' file.

    Returns:
        A copy of the model metadata dictionary if found, containing fields like:
        - id: Model identifier
        - size: File size in bytes
        - family: Model architecture family
        - quantization: Quantization format
        - type: 'llm' or 'embedding'
        - owned_by: Creator organization
        
        Returns None if the model is not found in registry or on disk.

    Example:
        >>> info = get_model_info("tinyllama-1.1b-chat-int4-ov")
        >>> info["parameter_size"]
        '1.1B'
        >>> info = get_model_info("nonexistent-model")
        >>> info is None
        True
    """
    # Check built-in models first
    if model_id in MODELS_INFO:
        return MODELS_INFO[model_id].copy()
    
    # Check if model exists on disk (scanned model)
    model_path = DEFAULT_MODEL_DIR / model_id
    if model_path.exists() and (model_path / "openvino_model.xml").exists():
        return {
            "id": model_id,
            "size": sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()),
            "digest": f"sha256:{model_id[:12]}",
            "parameter_size": "unknown",
            "quantization": "unknown",
            "family": "unknown",
            "type": "llm",
            "owned_by": "local",
        }
    
    return None


def scan_available_models(model_dir: Optional[Path] = None) -> list[dict]:
    """Scan directory for available OpenVINO models.

    Scans the specified directory (or default model cache) for subdirectories
    containing OpenVINO model files. Each valid model directory must contain
    an 'openvino_model.xml' file.

    For models found on disk that match entries in MODELS_INFO, the built-in
    metadata is used. For unknown models, basic metadata is generated from
    the filesystem (size calculated, other fields set to 'unknown').

    Args:
        model_dir: Directory path to scan for models. If None, uses
            DEFAULT_MODEL_DIR (~/.cache/npu-proxy/models).

    Returns:
        List of model metadata dictionaries for all valid models found.
        Each dictionary contains at minimum: id, size, digest, parameter_size,
        quantization, family, type, owned_by.
        
        Returns empty list if the directory doesn't exist or contains no models.

    Example:
        >>> from pathlib import Path
        >>> models = scan_available_models()
        >>> isinstance(models, list)
        True
        >>> custom_models = scan_available_models(Path("/custom/path"))
        >>> isinstance(custom_models, list)
        True
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    if not model_dir.exists():
        return []
    
    models = []
    for path in model_dir.iterdir():
        if path.is_dir() and (path / "openvino_model.xml").exists():
            model_id = path.name
            
            # Use built-in info if available, otherwise create basic entry
            if model_id in MODELS_INFO:
                models.append(MODELS_INFO[model_id].copy())
            else:
                models.append({
                    "id": model_id,
                    "size": sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
                    "digest": f"sha256:{model_id[:12]}",
                    "parameter_size": "unknown",
                    "quantization": "unknown",
                    "family": "unknown",
                    "type": "llm",
                    "owned_by": "local",
                })
    
    return models


def list_all_models() -> list[dict]:
    """List all known models (built-in + scanned).

    Combines models from the built-in MODELS_INFO registry with any
    additional models discovered by scanning the local model directory.
    Built-in model definitions take precedence over scanned metadata.

    This function provides a complete view of all available models,
    whether they are pre-registered or dynamically discovered on disk.

    Returns:
        List of model metadata dictionaries. Each dictionary contains
        complete model information including id, size, family, type, etc.
        Models are deduplicated by ID, with built-in definitions preferred.

    Example:
        >>> all_models = list_all_models()
        >>> isinstance(all_models, list)
        True
        >>> all(isinstance(m, dict) for m in all_models)
        True
    """
    # Start with built-in models
    all_models = {m["id"]: m.copy() for m in MODELS_INFO.values()}
    
    # Add/update with scanned models
    for model in scan_available_models():
        if model["id"] not in all_models:
            all_models[model["id"]] = model
    
    return list(all_models.values())


def get_openai_model_list() -> list[dict]:
    """Get models in OpenAI API format.

    Converts the internal model registry format to the format expected by
    the OpenAI API's /v1/models endpoint. This enables compatibility with
    OpenAI API clients and tools.

    Returns:
        List of dictionaries in OpenAI model format, each containing:
        - id (str): The model identifier
        - object (str): Always "model"
        - created (int): Unix timestamp of when list was generated
        - owned_by (str): Organization that owns/created the model

    Example:
        >>> models = get_openai_model_list()
        >>> all(m["object"] == "model" for m in models)
        True
        >>> all("id" in m and "created" in m for m in models)
        True
    """
    models = list_all_models()
    return [
        {
            "id": m["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": m.get("owned_by", "local"),
        }
        for m in models
    ]


def list_embedding_models() -> list[dict]:
    """Return only embedding models from registry.

    Filters the MODELS_INFO registry to return only models with
    type='embedding'. Useful for populating embedding-specific
    endpoints and model selection UIs.

    Embedding models include specialized metadata fields:
    - dimensions: Vector output size (e.g., 384, 1024, 4096)
    - context_length: Maximum input token length
    - description: Human-readable model description
    - hf_repo: HuggingFace repository for downloads

    Returns:
        List of copies of embedding model metadata dictionaries.
        Returns empty list if no embedding models are registered.

    Example:
        >>> embeddings = list_embedding_models()
        >>> all(m["type"] == "embedding" for m in embeddings)
        True
        >>> all("dimensions" in m for m in embeddings)
        True
    """
    return [
        m.copy() for m in MODELS_INFO.values()
        if m.get("type") == "embedding"
    ]
