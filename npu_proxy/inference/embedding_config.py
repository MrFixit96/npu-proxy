"""Canonical embedding model resolution and storage helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from npu_proxy.models.registry import (
    DEFAULT_MODEL_DIR,
    encode_repo_storage_key,
    find_catalog_entry,
    get_model_info,
)
from npu_proxy.models.metadata import detect_model_metadata

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_EMBEDDING_DEVICE = "CPU"
EMBEDDING_MODELS_SUBDIR = "embeddings"
EMBEDDING_REQUIRED_FILES: tuple[str, ...] = ("openvino_model.xml", "openvino_model.bin")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class InvalidEmbeddingModelError(ValueError):
    """Raised when an embedding model identifier is not safe to resolve."""


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Resolved embedding runtime/download configuration."""

    requested_model: str
    requested_device: str
    resolved_model: str
    storage_key: str
    repo_id: str | None
    dimensions: int
    device: str
    canonical_path: Path
    model_path: Path
    legacy_paths: tuple[Path, ...]
    is_downloaded: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_model", _normalize_embedding_model_name(self.requested_model))
        object.__setattr__(self, "requested_device", get_configured_embedding_device(self.requested_device))
        object.__setattr__(self, "resolved_model", str(self.resolved_model or "").strip())
        object.__setattr__(self, "storage_key", str(self.storage_key or "").strip())
        object.__setattr__(self, "repo_id", str(self.repo_id).strip() if self.repo_id else None)
        object.__setattr__(self, "device", get_configured_embedding_device(self.device))
        object.__setattr__(self, "canonical_path", Path(self.canonical_path).resolve(strict=False))
        object.__setattr__(self, "model_path", Path(self.model_path).resolve(strict=False))
        object.__setattr__(
            self,
            "legacy_paths",
            tuple(dict.fromkeys(Path(path).resolve(strict=False) for path in self.legacy_paths)),
        )

        if not self.resolved_model:
            raise InvalidEmbeddingModelError("Resolved embedding model identity must not be empty.")
        if not self.storage_key:
            raise InvalidEmbeddingModelError("Embedding storage key must not be empty.")
        if "/" in self.storage_key or "\\" in self.storage_key:
            raise InvalidEmbeddingModelError(
                "Embedding storage key must be a single safe path segment."
            )
        if self.dimensions <= 0:
            raise InvalidEmbeddingModelError("Embedding dimensions must be greater than zero.")
        if self.requested_device != self.device:
            raise InvalidEmbeddingModelError(
                "Embedding config device must match the normalized requested device."
            )
        if self.canonical_path.name != self.storage_key:
            raise InvalidEmbeddingModelError(
                "Canonical embedding path must end with the canonical storage key."
            )
        allowed_paths = {self.canonical_path, *self.legacy_paths}
        if self.model_path not in allowed_paths:
            raise InvalidEmbeddingModelError(
                "Embedding model path must be the canonical path or a known legacy path."
            )
        embeddings_root = self.canonical_path.parent.resolve(strict=False)
        for path in allowed_paths:
            resolved_path = path.resolve(strict=False)
            if resolved_path.parent != embeddings_root or not _is_relative_to(resolved_path, embeddings_root):
                raise InvalidEmbeddingModelError(
                    "Embedding resolution paths must stay within the embeddings model root."
                )


def get_configured_embedding_model_name(model_name: str | None = None) -> str:
    """Return the explicitly requested or configured embedding model name."""
    return model_name or os.environ.get("NPU_PROXY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_configured_embedding_device(device: str | None = None) -> str:
    """Return the explicitly requested or configured embedding device."""
    candidate = device or os.environ.get("NPU_PROXY_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    candidate = candidate.strip().upper() if candidate and candidate.strip() else DEFAULT_EMBEDDING_DEVICE
    return candidate


def get_embedding_models_dir(model_dir: Path | str | None = None) -> Path:
    """Return the canonical directory for embedding models."""
    if model_dir is None:
        base_dir = DEFAULT_MODEL_DIR
    elif isinstance(model_dir, str):
        base_dir = Path(model_dir)
    else:
        base_dir = model_dir
    return base_dir / EMBEDDING_MODELS_SUBDIR


def is_known_embedding_model(model_name: str) -> bool:
    """Return True when the requested name resolves to a catalogued embedding model."""
    info = get_model_info(model_name)
    return bool((info and info.get("type") == "embedding") or _is_explicit_embedding_repo(model_name))


def get_embedding_model_path(model_name: str | None = None, model_dir: Path | str | None = None) -> Path:
    """Return the active filesystem path for an embedding model."""
    return resolve_embedding_model_config(model_name=model_name, model_dir=model_dir).model_path


def is_embedding_model_downloaded(model_name: str, model_dir: Path | str | None = None) -> bool:
    """Return True if the embedding model exists at the canonical or legacy path."""
    return resolve_embedding_model_config(model_name=model_name, model_dir=model_dir).is_downloaded


def resolve_embedding_model_config(
    model_name: str | None = None,
    *,
    device: str | None = None,
    model_dir: Path | str | None = None,
) -> EmbeddingModelConfig:
    """Resolve model identity, dimensions, and storage paths for embeddings."""
    requested_model = _normalize_embedding_model_name(get_configured_embedding_model_name(model_name))
    requested_device = get_configured_embedding_device(device)

    catalog_entry = find_catalog_entry(requested_model)
    model_info = get_model_info(requested_model)
    repo_id: str | None = None
    resolved_model = requested_model
    storage_key = encode_repo_storage_key(requested_model) if "/" in requested_model else _sanitize_model_id(requested_model)
    dimensions = DEFAULT_EMBEDDING_DIMENSIONS
    catalog_type = catalog_entry.get("type") if catalog_entry else None

    if model_info and model_info.get("type") == "embedding":
        repo_id = model_info.get("hf_repo") or model_info.get("repo_id")
        resolved_model = model_info["id"]
        storage_key = model_info["storage_key"]
        dimensions = int(model_info.get("dimensions") or DEFAULT_EMBEDDING_DIMENSIONS)
    else:
        if catalog_entry and catalog_type != "embedding":
            raise InvalidEmbeddingModelError(
                f"Model '{requested_model}' is not a supported embedding model."
            )
        if catalog_entry:
            repo_id = catalog_entry.get("repo_id")
            resolved_model = catalog_entry.get("registry_id") or catalog_entry.get("ollama_name") or requested_model
            storage_key = catalog_entry.storage_key
        elif "/" in requested_model:
            _validate_hugging_face_repo_id(requested_model)
            repo_id = requested_model
            resolved_model = requested_model
            storage_key = encode_repo_storage_key(requested_model)

    embeddings_root = get_embedding_models_dir(model_dir=model_dir).resolve(strict=False)
    canonical_path = embeddings_root / storage_key
    legacy_root = DEFAULT_MODEL_DIR if model_dir is None else Path(model_dir)
    legacy_paths = tuple(
        path
        for path in _legacy_model_paths(
            legacy_root=legacy_root,
            requested_model=requested_model,
            resolved_model=resolved_model,
            storage_key=storage_key,
            repo_id=repo_id,
            catalog_entry=catalog_entry,
        )
        if path != canonical_path
    )

    model_path = canonical_path
    is_downloaded = False
    for candidate in (canonical_path, *legacy_paths):
        if _has_required_files(candidate, embeddings_root):
            model_path = candidate
            is_downloaded = True
            break

    return EmbeddingModelConfig(
        requested_model=requested_model,
        requested_device=requested_device,
        resolved_model=resolved_model,
        storage_key=storage_key,
        repo_id=repo_id,
        dimensions=dimensions,
        device=requested_device,
        canonical_path=canonical_path,
        model_path=model_path,
        legacy_paths=legacy_paths,
        is_downloaded=is_downloaded,
    )


def _legacy_model_paths(
    *,
    legacy_root: Path,
    requested_model: str,
    resolved_model: str,
    storage_key: str,
    repo_id: str | None,
    catalog_entry: object | None,
) -> list[Path]:
    candidates: list[str] = []
    trusted_catalog_values = catalog_entry is not None

    def add(value: str | None) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(storage_key)
    add(resolved_model)
    if trusted_catalog_values:
        add(requested_model)
    if catalog_entry:
        add(catalog_entry.get("ollama_name"))
        add(catalog_entry.get("registry_id"))
        for alias in catalog_entry.get("aliases") or ():
            add(alias)
    if trusted_catalog_values and repo_id and "/" in repo_id:
        add(repo_id.split("/")[-1])
        add(encode_repo_storage_key(repo_id))
        add(_sanitize_model_id(repo_id))
    if trusted_catalog_values and "/" in requested_model:
        add(requested_model.split("/")[-1])
        add(encode_repo_storage_key(requested_model))
        add(_sanitize_model_id(requested_model))
    elif repo_id and "/" in repo_id:
        add(encode_repo_storage_key(repo_id))
        add(_sanitize_model_id(repo_id))

    legacy_paths: list[Path] = []
    embeddings_root = legacy_root / EMBEDDING_MODELS_SUBDIR
    for candidate in candidates:
        safe_candidate = _sanitize_model_id(candidate)
        legacy_paths.append(embeddings_root / safe_candidate)
    return legacy_paths


def _has_required_files(model_path: Path, embeddings_root: Path) -> bool:
    """Return True only for regular required files under the embeddings root."""
    root = embeddings_root.resolve(strict=False)
    resolved_model_path = model_path.resolve(strict=False)
    if not _is_relative_to(resolved_model_path, root):
        return False
    if not resolved_model_path.exists() or not resolved_model_path.is_dir():
        return False

    for name in EMBEDDING_REQUIRED_FILES:
        required_file = (resolved_model_path / name).resolve(strict=False)
        if not _is_relative_to(required_file, resolved_model_path):
            return False
        if not _is_relative_to(required_file, root):
            return False
        if not required_file.is_file():
            return False
    return True


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_hugging_face_repo_id(repo_id: str) -> None:
    """Validate the limited namespace/repo shape accepted for unknown repos."""
    if repo_id.startswith(("http://", "https://")) or repo_id.count("/") != 1:
        raise InvalidEmbeddingModelError(
            "Embedding model identifier must be a Hugging Face repo ID of the form namespace/name."
        )
    if not _HF_REPO_ID_PATTERN.match(repo_id):
        raise InvalidEmbeddingModelError(
            "Embedding model identifier contains unsupported characters."
        )
    for segment in repo_id.split("/"):
        if ".." in segment or "--" in segment:
            raise InvalidEmbeddingModelError(
                "Embedding model identifier contains unsupported segment syntax."
            )


def _sanitize_model_id(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def _is_explicit_embedding_repo(model_name: str) -> bool:
    metadata = detect_model_metadata(model_name)
    return metadata["type"] == "embedding"


def _normalize_embedding_model_name(model_name: str) -> str:
    """Normalize and validate a model identifier before path resolution."""
    normalized = model_name.strip()
    if not normalized:
        raise InvalidEmbeddingModelError("Embedding model identifier must not be empty.")

    if "\\" in normalized:
        raise InvalidEmbeddingModelError(
            "Embedding model identifier must use model IDs or Hugging Face repo IDs, not filesystem paths."
        )

    if normalized.startswith("/") or normalized.startswith("//") or _WINDOWS_DRIVE_PATTERN.match(normalized):
        raise InvalidEmbeddingModelError(
            "Embedding model identifier must not be an absolute filesystem path."
        )

    for segment in normalized.split("/"):
        if segment in {".", ".."}:
            raise InvalidEmbeddingModelError(
                "Embedding model identifier must not contain path traversal segments."
            )

    return normalized
