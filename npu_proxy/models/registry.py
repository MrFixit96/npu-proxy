"""Canonical model catalog and registry helpers for NPU Proxy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from npu_proxy.models.metadata import (
    DictLikeDataclass,
    detect_backend,
    detect_model_metadata,
    detect_task,
)

DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"
LOCAL_MODEL_REQUIRED_FILES: tuple[str, str] = ("openvino_model.xml", "openvino_model.bin")
MAX_SCAN_FILES = 10_000
MAX_SCAN_DEPTH = 16
logger = logging.getLogger(__name__)

VALID_MODEL_TYPES = {"llm", "embedding", "vision"}


def _normalize_required_text(field_name: str, value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _normalize_slug(value: str | None) -> str:
    normalized = _normalize_required_text("identifier", value)
    if normalized in {".", ".."}:
        raise ValueError(f"identifier {normalized!r} must not be a relative path segment")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"identifier {normalized!r} must not contain path separators")
    return normalized


@dataclass(frozen=True)
class CatalogEntry(DictLikeDataclass):
    """Canonical catalog metadata with construction-time invariant checks."""

    repo_id: str
    ollama_name: str
    registry_id: str | None = None
    aliases: tuple[str, ...] = ()
    name: str = ""
    size: int = 0
    digest: str = ""
    parameter_size: str = "unknown"
    quantization: str = "unknown"
    family: str = "unknown"
    architecture: str = "unknown"
    type: str = "llm"
    backend: str = "openvino"
    format: str = "openvino-ir"
    task: str = "text-generation"
    owned_by: str = "local"
    context_length: int | None = None
    dimensions: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_id", _normalize_required_text("repo_id", self.repo_id))
        object.__setattr__(
            self, "ollama_name", _normalize_required_text("ollama_name", self.ollama_name)
        )
        object.__setattr__(self, "registry_id", _normalize_optional_text(self.registry_id))
        object.__setattr__(self, "name", _normalize_required_text("name", self.name))
        object.__setattr__(self, "digest", str(self.digest or "").strip())
        object.__setattr__(
            self,
            "parameter_size",
            _normalize_required_text("parameter_size", self.parameter_size),
        )
        object.__setattr__(
            self,
            "quantization",
            _normalize_required_text("quantization", self.quantization),
        )
        object.__setattr__(self, "family", _normalize_required_text("family", self.family))
        object.__setattr__(
            self,
            "architecture",
            _normalize_required_text("architecture", self.architecture),
        )
        object.__setattr__(self, "type", _normalize_required_text("type", self.type))
        object.__setattr__(self, "backend", _normalize_required_text("backend", self.backend))
        object.__setattr__(self, "format", _normalize_required_text("format", self.format))
        object.__setattr__(self, "task", _normalize_required_text("task", self.task))
        object.__setattr__(
            self,
            "owned_by",
            _normalize_required_text("owned_by", self.owned_by).lower(),
        )
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(
            self,
            "aliases",
            tuple(
                dict.fromkeys(
                    alias
                    for alias in (
                        _normalize_optional_text(alias) for alias in self.aliases
                    )
                    if alias
                )
            ),
        )

        if self.size < 0:
            raise ValueError("size must be zero or greater")
        if self.type not in VALID_MODEL_TYPES:
            raise ValueError(f"unsupported model type {self.type!r}")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be greater than zero")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        if self.type == "embedding" and self.dimensions is None:
            raise ValueError("embedding catalog entries must declare dimensions")
        if self.type != "embedding" and self.dimensions is not None:
            raise ValueError("only embedding catalog entries may declare dimensions")
        expected_task = detect_task("", model_type=self.type)
        if expected_task and self.task != expected_task:
            raise ValueError(f"task {self.task!r} is not valid for model type {self.type!r}")
        expected_backend = detect_backend("", model_format=self.format)
        if expected_backend and self.backend != expected_backend:
            raise ValueError(
                f"backend {self.backend!r} is not valid for model format {self.format!r}"
            )

    @property
    def canonical_id(self) -> str:
        return self.registry_id or self.ollama_name

    @property
    def storage_key(self) -> str:
        return self.registry_id or encode_repo_storage_key(self.repo_id)

    def to_registry_model(self) -> "RegistryModelInfo":
        aliases = tuple(dict.fromkeys((self.ollama_name, *self.aliases)))
        return RegistryModelInfo(
            id=self.canonical_id,
            storage_key=self.storage_key,
            name=self.name,
            size=self.size,
            digest=self.digest,
            parameter_size=self.parameter_size,
            quantization=self.quantization,
            family=self.family,
            architecture=self.architecture,
            type=self.type,
            backend=self.backend,
            format=self.format,
            task=self.task,
            repo_id=self.repo_id,
            owned_by=self.owned_by,
            aliases=aliases,
            context_length=self.context_length,
            dimensions=self.dimensions,
            description=self.description,
            hf_repo=self.repo_id if self.type == "embedding" else None,
        )


@dataclass(frozen=True)
class RegistryModelInfo(DictLikeDataclass):
    """Normalized model metadata exposed by the registry."""

    id: str
    storage_key: str
    name: str
    size: int
    digest: str
    parameter_size: str
    quantization: str
    family: str
    architecture: str
    type: str
    backend: str
    format: str
    task: str
    repo_id: str
    owned_by: str
    aliases: tuple[str, ...] = ()
    context_length: int | None = None
    dimensions: int | None = None
    description: str = ""
    hf_repo: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_slug(self.id))
        object.__setattr__(self, "storage_key", _normalize_slug(self.storage_key))
        object.__setattr__(self, "name", _normalize_required_text("name", self.name))
        object.__setattr__(self, "digest", str(self.digest or "").strip())
        object.__setattr__(
            self,
            "parameter_size",
            _normalize_required_text("parameter_size", self.parameter_size),
        )
        object.__setattr__(
            self,
            "quantization",
            _normalize_required_text("quantization", self.quantization),
        )
        object.__setattr__(self, "family", _normalize_required_text("family", self.family))
        object.__setattr__(
            self,
            "architecture",
            _normalize_required_text("architecture", self.architecture),
        )
        object.__setattr__(self, "type", _normalize_required_text("type", self.type))
        object.__setattr__(self, "backend", _normalize_required_text("backend", self.backend))
        object.__setattr__(self, "format", _normalize_required_text("format", self.format))
        object.__setattr__(self, "task", _normalize_required_text("task", self.task))
        object.__setattr__(self, "repo_id", str(self.repo_id or "").strip())
        object.__setattr__(
            self,
            "owned_by",
            _normalize_required_text("owned_by", self.owned_by).lower(),
        )
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "hf_repo", _normalize_optional_text(self.hf_repo))
        object.__setattr__(
            self,
            "aliases",
            tuple(
                dict.fromkeys(
                    alias
                    for alias in (
                        _normalize_optional_text(alias) for alias in self.aliases
                    )
                    if alias
                )
            ),
        )

        if self.size < 0:
            raise ValueError("size must be zero or greater")
        if self.type not in VALID_MODEL_TYPES:
            raise ValueError(f"unsupported model type {self.type!r}")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be greater than zero")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        if self.hf_repo and self.type != "embedding":
            raise ValueError("hf_repo is only valid for embedding registry models")
        if self.type == "embedding" and self.repo_id and self.dimensions is None:
            raise ValueError("catalogued embedding registry models must declare dimensions")
        if self.type != "embedding" and self.hf_repo:
            raise ValueError("non-embedding registry models must not declare hf_repo")
        expected_task = detect_task("", model_type=self.type)
        if expected_task and self.task != expected_task:
            raise ValueError(f"task {self.task!r} is not valid for model type {self.type!r}")
        expected_backend = detect_backend("", model_format=self.format)
        if expected_backend and self.backend != expected_backend:
            raise ValueError(
                f"backend {self.backend!r} is not valid for model format {self.format!r}"
            )


def _catalog_entry(
    *,
    repo_id: str,
    ollama_name: str,
    registry_id: str | None = None,
    aliases: tuple[str, ...] = (),
    name: str | None = None,
    size: int = 0,
    digest: str = "",
    parameter_size: str = "",
    quantization: str = "",
    family: str = "",
    model_type: str = "",
    backend: str = "openvino",
    model_format: str = "openvino-ir",
    task: str = "",
    owned_by: str = "",
    context_length: int | None = None,
    dimensions: int | None = None,
    description: str = "",
) -> CatalogEntry:
    hint_text = " ".join(
        part for part in (repo_id, registry_id or "", ollama_name, name or "") if part
    )
    detected = detect_model_metadata(
        hint_text,
        default_type=model_type,
        default_backend=backend,
        default_format=model_format,
        default_task=task,
    )
    org = repo_id.split("/", 1)[0] if "/" in repo_id else owned_by
    return CatalogEntry(
        repo_id=repo_id,
        registry_id=registry_id,
        ollama_name=ollama_name,
        aliases=tuple(dict.fromkeys(aliases)),
        name=name or (registry_id or repo_id.split("/")[-1]),
        size=size,
        digest=digest,
        parameter_size=parameter_size or detected["parameters"] or "unknown",
        quantization=quantization or detected["quantization"] or "unknown",
        family=family or detected["family"] or "unknown",
        architecture=detected["architecture"] or (family or detected["family"] or "unknown"),
        type=model_type or detected["type"] or "llm",
        backend=detected["backend"] or backend or "openvino",
        format=detected["format"] or model_format or "openvino-ir",
        task=detected["task"] or task or "text-generation",
        owned_by=(owned_by or org or "local").lower(),
        context_length=context_length,
        dimensions=dimensions,
        description=description,
    )


def encode_repo_storage_key(repo_id: str) -> str:
    """Encode a fully-qualified repo ID into an unambiguous cache directory name."""
    return quote(repo_id, safe="")


def get_catalog_storage_key(entry: CatalogEntry | RegistryModelInfo | dict[str, Any]) -> str:
    """Return the canonical cache directory name for a catalogued model."""
    if isinstance(entry, CatalogEntry):
        return entry.storage_key
    if isinstance(entry, RegistryModelInfo):
        return entry.storage_key
    return str(entry.get("storage_key") or entry.get("registry_id") or encode_repo_storage_key(str(entry["repo_id"])))


MODEL_CATALOG: tuple[CatalogEntry, ...] = (
    _catalog_entry(
        repo_id="OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        registry_id="tinyllama-1.1b-chat-int4-ov",
        ollama_name="tinyllama",
        aliases=("tinyllama-1.1b-chat-int4-ov",),
        name="TinyLlama 1.1B Chat INT4",
        size=637_000_000,
        digest="sha256:tinyllama1.1b",
        parameter_size="1.1B",
        quantization="INT4",
        family="llama",
        model_type="llm",
        task="text-generation",
        context_length=2048,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov",
        ollama_name="tinyllama:fp16",
        aliases=("tinyllama-fp16",),
        name="TinyLlama 1.1B Chat FP16",
        parameter_size="1.1B",
        quantization="FP16",
        family="llama",
        model_type="llm",
        task="text-generation",
        context_length=2048,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/phi-2-int4-ov",
        registry_id="phi-2-int4-ov",
        ollama_name="phi-2",
        aliases=("phi-2-int4-ov",),
        name="Phi-2 INT4",
        size=1_400_000_000,
        digest="sha256:phi2",
        parameter_size="2.7B",
        quantization="INT4",
        family="phi",
        model_type="llm",
        task="text-generation",
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/Phi-3-mini-4k-instruct-int4-ov",
        ollama_name="phi-3",
        aliases=("phi-3-mini",),
        name="Phi-3 Mini 4K Instruct INT4",
        parameter_size="3.8B",
        quantization="INT4",
        family="phi",
        model_type="llm",
        task="text-generation",
        context_length=4096,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/llama-2-7b-chat-int4-ov",
        ollama_name="llama2",
        name="Llama 2 7B Chat INT4",
        parameter_size="7B",
        quantization="INT4",
        family="llama",
        model_type="llm",
        task="text-generation",
        context_length=4096,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/llama-2-13b-chat-int4-ov",
        ollama_name="llama2:13b",
        name="Llama 2 13B Chat INT4",
        parameter_size="13B",
        quantization="INT4",
        family="llama",
        model_type="llm",
        task="text-generation",
        context_length=4096,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
        ollama_name="llama3.2",
        name="Llama 3.2 3B Instruct INT4",
        parameter_size="3B",
        quantization="INT4",
        family="llama",
        model_type="llm",
        task="text-generation",
        context_length=8192,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/mistral-7b-instruct-v0.1-int4-ov",
        registry_id="mistral-7b-int4-ov",
        ollama_name="mistral",
        aliases=("mistral-7b-int4-ov",),
        name="Mistral 7B Instruct INT4",
        size=4_000_000_000,
        digest="sha256:mistral7b",
        parameter_size="7B",
        quantization="INT4",
        family="mistral",
        model_type="llm",
        task="text-generation",
        context_length=8192,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/Qwen2-1.5B-Instruct-int4-ov",
        ollama_name="qwen2",
        name="Qwen2 1.5B Instruct INT4",
        parameter_size="1.5B",
        quantization="INT4",
        family="qwen",
        model_type="llm",
        task="text-generation",
        context_length=8192,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/gemma-2b-it-int4-ov",
        ollama_name="gemma",
        name="Gemma 2B IT INT4",
        parameter_size="2B",
        quantization="INT4",
        family="gemma",
        model_type="llm",
        task="text-generation",
        context_length=8192,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="OpenVINO/granite-4-micro-ov",
        registry_id="granite-4-micro-ov",
        ollama_name="granite-4-micro",
        aliases=("granite-4-micro-ov",),
        name="Granite 4 Micro",
        size=3_406_770_588,
        digest="sha256:granite4micro",
        parameter_size="1B",
        quantization="FP32",
        family="granite",
        model_type="llm",
        task="text-generation",
        context_length=8192,
        owned_by="openvino",
    ),
    _catalog_entry(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        registry_id="all-minilm-l6-v2",
        ollama_name="all-minilm",
        aliases=("all-minilm-l6-v2",),
        name="All-MiniLM-L6-v2",
        size=91_000_000,
        digest="sha256:minilm",
        parameter_size="22M",
        quantization="FP32",
        family="bert",
        model_type="embedding",
        task="feature-extraction",
        context_length=256,
        dimensions=384,
        description="Lightweight sentence embeddings",
        owned_by="sentence-transformers",
    ),
    _catalog_entry(
        repo_id="BAAI/bge-small-en-v1.5",
        registry_id="bge-small",
        ollama_name="bge-small",
        name="BGE Small English",
        size=133_000_000,
        digest="sha256:bgesmall",
        parameter_size="33M",
        quantization="FP32",
        family="bert",
        model_type="embedding",
        task="feature-extraction",
        context_length=512,
        dimensions=384,
        description="Fast, high-quality English embeddings",
        owned_by="baai",
    ),
    _catalog_entry(
        repo_id="BAAI/bge-base-en-v1.5",
        ollama_name="bge-base",
        name="BGE Base English",
        parameter_size="109M",
        quantization="FP32",
        family="bert",
        model_type="embedding",
        task="feature-extraction",
        context_length=512,
        dimensions=768,
        owned_by="baai",
    ),
    _catalog_entry(
        repo_id="BAAI/bge-large-en-v1.5",
        ollama_name="bge-large",
        name="BGE Large English",
        parameter_size="335M",
        quantization="FP32",
        family="bert",
        model_type="embedding",
        task="feature-extraction",
        context_length=512,
        dimensions=1024,
        owned_by="baai",
    ),
    _catalog_entry(
        repo_id="intfloat/e5-small-v2",
        ollama_name="e5-small",
        name="E5 Small",
        parameter_size="33M",
        quantization="FP32",
        family="bert",
        model_type="embedding",
        task="feature-extraction",
        context_length=512,
        dimensions=384,
        owned_by="intfloat",
    ),
    _catalog_entry(
        repo_id="intfloat/multilingual-e5-large",
        registry_id="e5-large",
        ollama_name="e5-large",
        name="E5 Large Multilingual",
        size=2_200_000_000,
        digest="sha256:e5large",
        parameter_size="560M",
        quantization="FP32",
        family="bert",
        model_type="embedding",
        task="feature-extraction",
        context_length=512,
        dimensions=1024,
        description="High-quality multilingual embeddings",
        owned_by="intfloat",
    ),
    _catalog_entry(
        repo_id="nomic-ai/nomic-embed-text-v1.5",
        ollama_name="nomic-embed-text",
        name="Nomic Embed Text",
        parameter_size="137M",
        quantization="FP32",
        family="bert",
        model_type="embedding",
        task="feature-extraction",
        context_length=8192,
        dimensions=768,
        owned_by="nomic-ai",
    ),
    _catalog_entry(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        registry_id="qwen3-embedding-0.6b-int4-ov",
        ollama_name="qwen3-embedding:0.6b",
        aliases=("qwen3-embedding-0.6b-int4-ov",),
        name="Qwen3 Embedding 0.6B",
        size=400_000_000,
        digest="sha256:qwen3emb06b",
        parameter_size="0.6B",
        quantization="INT4",
        family="qwen",
        model_type="embedding",
        task="feature-extraction",
        context_length=8192,
        dimensions=1024,
        description="Qwen3 text embeddings - small",
        owned_by="qwen",
    ),
    _catalog_entry(
        repo_id="Qwen/Qwen3-Embedding-8B",
        registry_id="qwen3-embedding-8b-int4-ov",
        ollama_name="qwen3-embedding:8b",
        aliases=("qwen3-embedding-8b-int4-ov",),
        name="Qwen3 Embedding 8B",
        size=4_500_000_000,
        digest="sha256:qwen3emb8b",
        parameter_size="8B",
        quantization="INT4",
        family="qwen",
        model_type="embedding",
        task="feature-extraction",
        context_length=8192,
        dimensions=4096,
        description="Qwen3 text embeddings - large",
        owned_by="qwen",
    ),
)

_CATALOG_BY_OLLAMA_NAME: dict[str, CatalogEntry] = {
    entry["ollama_name"]: entry for entry in MODEL_CATALOG if entry.get("ollama_name")
}
_CATALOG_BY_REPO_ID: dict[str, CatalogEntry] = {entry["repo_id"]: entry for entry in MODEL_CATALOG}
_CATALOG_BY_REGISTRY_ID: dict[str, CatalogEntry] = {
    entry["registry_id"]: entry for entry in MODEL_CATALOG if entry.get("registry_id")
}
_CATALOG_BY_NAME: dict[str, CatalogEntry] = {}
for _entry in MODEL_CATALOG:
    for _candidate in (
        _entry.get("ollama_name"),
        _entry.get("repo_id"),
        _entry.get("registry_id"),
        get_catalog_storage_key(_entry),
        *(_entry.get("aliases") or ()),
    ):
        if _candidate:
            _CATALOG_BY_NAME.setdefault(_candidate, _entry)


def _entry_to_registry_model(entry: CatalogEntry) -> RegistryModelInfo:
    return entry.to_registry_model()


MODELS_INFO: dict[str, RegistryModelInfo] = {
    entry["registry_id"] or entry["ollama_name"]: _entry_to_registry_model(entry)
    for entry in MODEL_CATALOG
}


def find_catalog_entry(name: str) -> Optional[CatalogEntry]:
    """Return a canonical catalog entry for an alias, repo, or registry ID."""
    return _CATALOG_BY_NAME.get(name)


def _is_regular_nonempty_file(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        return path.stat().st_size > 0
    except OSError:
        return False


def _is_within_directory(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _scan_size(model_path: Path) -> int:
    root = model_path.resolve()
    total = 0
    scanned = 0
    stack: list[tuple[Path, int]] = [(model_path, 0)]

    while stack:
        current, depth = stack.pop()
        if depth > MAX_SCAN_DEPTH or not _is_within_directory(current, root):
            continue
        try:
            entries = list(current.iterdir())
        except OSError:
            logger.debug("Unable to scan model path %s", current, exc_info=True)
            continue

        for entry in entries:
            if scanned >= MAX_SCAN_FILES:
                logger.warning("Stopped size scan for %s after %d files", model_path, scanned)
                return total
            if entry.is_symlink() or not _is_within_directory(entry, root):
                continue
            try:
                if entry.is_dir():
                    stack.append((entry, depth + 1))
                elif entry.is_file():
                    total += entry.stat().st_size
                    scanned += 1
            except OSError:
                logger.debug("Unable to stat model path %s", entry, exc_info=True)
                continue
    return total


def _is_valid_local_model_dir(model_path: Path) -> bool:
    return model_path.is_dir() and all(
        _is_regular_nonempty_file(model_path / file_name)
        for file_name in LOCAL_MODEL_REQUIRED_FILES
    )


def _build_scanned_model_info(model_id: str, model_path: Path) -> RegistryModelInfo:
    detected = detect_model_metadata(
        model_id,
        default_backend="openvino",
        default_format="openvino-ir",
    )
    model_type = detected["type"] or "llm"
    task = detected["task"] or ("feature-extraction" if model_type == "embedding" else "text-generation")
    family = detected["family"] or "unknown"
    return RegistryModelInfo(
        id=model_id,
        storage_key=model_id,
        name=model_id,
        size=_scan_size(model_path),
        digest=f"sha256:{model_id[:12]}",
        parameter_size=detected["parameters"] or "unknown",
        quantization=detected["quantization"] or "unknown",
        family=family,
        architecture=detected["architecture"] or family,
        type=model_type,
        backend="openvino",
        format="openvino-ir",
        task=task,
        owned_by="local",
        repo_id="",
        aliases=(),
    )


def get_model_info(model_id: str) -> Optional[RegistryModelInfo]:
    """Get model metadata by registry ID."""
    if model_id in MODELS_INFO:
        return MODELS_INFO[model_id]

    catalog_entry = _CATALOG_BY_NAME.get(model_id)
    if catalog_entry:
        registry_id = catalog_entry.get("registry_id")
        if registry_id in MODELS_INFO:
            return MODELS_INFO[registry_id]
        return _entry_to_registry_model(catalog_entry)

    try:
        safe_model_id = _normalize_slug(model_id)
    except ValueError:
        # Request routes pass raw model IDs here. Direct repo IDs are stored under
        # percent-encoded cache keys, so raw separators are never valid local paths.
        return None

    root = DEFAULT_MODEL_DIR.resolve()
    model_path = (root / safe_model_id).resolve()
    if not _is_within_directory(model_path, root):
        return None
    if _is_valid_local_model_dir(model_path):
        return _build_scanned_model_info(safe_model_id, model_path)

    return None


def scan_available_models(model_dir: Optional[Path] = None) -> list[RegistryModelInfo]:
    """Scan a directory for locally available OpenVINO models."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    if not model_dir.exists():
        return []

    models: list[RegistryModelInfo] = []
    for path in model_dir.iterdir():
        if not _is_valid_local_model_dir(path):
            continue

        model_id = path.name
        model_info = get_model_info(model_id)
        if model_info is None:
            model_info = _build_scanned_model_info(model_id, path)
        models.append(model_info)

    return models


def list_all_models() -> list[RegistryModelInfo]:
    """List all registry models plus any scanned local additions."""
    all_models = dict(MODELS_INFO)
    for model in scan_available_models():
        all_models.setdefault(model["id"], model)
    return list(all_models.values())


def get_openai_model_list() -> list[dict]:
    """Get models in the OpenAI /v1/models response format."""
    return [
        {
            "id": model["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": model.get("owned_by", "local"),
        }
        for model in list_all_models()
    ]


def list_embedding_models() -> list[RegistryModelInfo]:
    """Return only embedding models from the registry."""
    return [model for model in MODELS_INFO.values() if model.get("type") == "embedding"]
