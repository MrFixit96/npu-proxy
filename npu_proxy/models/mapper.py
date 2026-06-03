"""Mapping helpers derived from the canonical model catalog."""

from __future__ import annotations

import re

from huggingface_hub.utils import HFValidationError, validate_repo_id

from npu_proxy.models.metadata import detect_quantization
from npu_proxy.models.registry import (
    MODEL_CATALOG,
    encode_repo_storage_key,
    find_catalog_entry,
    get_catalog_storage_key,
)

OLLAMA_TO_HUGGINGFACE: dict[str, tuple[str, str]] = {
    entry["ollama_name"]: (entry["repo_id"], entry["type"])
    for entry in MODEL_CATALOG
    if entry.get("ollama_name")
}

REVERSE_MAPPING: dict[str, str] = {
    entry["repo_id"]: entry["ollama_name"]
    for entry in MODEL_CATALOG
    if entry.get("ollama_name")
}

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_URL_LIKE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def is_valid_repo_id(repo_id: str) -> bool:
    """Return True when a request string is an acceptable Hugging Face repo ID."""
    if not isinstance(repo_id, str):
        return False
    candidate = repo_id.strip()
    if candidate != repo_id or not candidate:
        return False
    if "\\" in candidate or _CONTROL_CHARS.search(candidate) or _URL_LIKE.match(candidate):
        return False
    if any(part in {"", ".", ".."} for part in candidate.split("/")):
        return False
    try:
        validate_repo_id(candidate)
    except HFValidationError:
        return False
    return True


def resolve_model_repo(name: str) -> tuple[str, str] | None:
    """Resolve an Ollama-style name or registry ID to a HuggingFace repository."""
    if name in OLLAMA_TO_HUGGINGFACE:
        repo_id, _ = OLLAMA_TO_HUGGINGFACE[name]
        return (repo_id, name)

    if "/" in name:
        if not is_valid_repo_id(name):
            return None
        return (name, name.split("/")[-1])

    catalog_entry = find_catalog_entry(name)
    if catalog_entry:
        return (catalog_entry["repo_id"], name)

    return None


def resolve_model_storage_key(name: str) -> str | None:
    """Resolve a request to the canonical on-disk cache directory name."""
    catalog_entry = find_catalog_entry(name)
    if catalog_entry:
        return get_catalog_storage_key(catalog_entry)

    if "/" in name:
        if not is_valid_repo_id(name):
            return None
        return encode_repo_storage_key(name)

    return None


def resolve_runtime_model_name(name: str) -> str | None:
    """Resolve the model identifier that should be returned to clients."""
    if find_catalog_entry(name):
        return name

    if "/" in name:
        if not is_valid_repo_id(name):
            return None
        return encode_repo_storage_key(name)

    return None


def get_ollama_name(repo_id: str) -> str | None:
    """Get the primary Ollama-style alias for a HuggingFace repository."""
    return REVERSE_MAPPING.get(repo_id)


def _extract_quantization(repo_name: str) -> str:
    """Extract quantization type from a repository name for display helpers."""
    return detect_quantization(repo_name).lower() or "unknown"


def list_known_models() -> list[dict[str, str]]:
    """List all known alias-to-repository mappings derived from the catalog."""
    models: list[dict[str, str]] = []
    for entry in sorted(MODEL_CATALOG, key=lambda item: item["ollama_name"]):
        models.append(
            {
                "ollama_name": entry["ollama_name"],
                "huggingface_repo": entry["repo_id"],
                "quantization": _extract_quantization(entry["repo_id"]),
                "type": entry["type"],
                "family": entry["family"],
                "parameter_size": entry["parameter_size"],
                "backend": entry["backend"],
                "format": entry["format"],
                "task": entry["task"],
            }
        )
    return models
