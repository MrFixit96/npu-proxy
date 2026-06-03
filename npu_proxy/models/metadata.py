"""Shared model metadata detectors used across catalog, registry, and search."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any
import re

FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\btinyllama\b", re.IGNORECASE), "llama", "TinyLLaMA"),
    (re.compile(r"\bcodellama\b", re.IGNORECASE), "llama", "CodeLLaMA"),
    (re.compile(r"\bllama(?:[-\s]?\d+(?:\.\d+)?)?\b", re.IGNORECASE), "llama", "LLaMA"),
    (re.compile(r"\bmixtral\b", re.IGNORECASE), "mistral", "Mixtral"),
    (re.compile(r"\bmistral\b", re.IGNORECASE), "mistral", "Mistral"),
    (re.compile(r"\bphi(?:[-\s]?\d+(?:\.\d+)?)?\b", re.IGNORECASE), "phi", "Phi"),
    (re.compile(r"\bqwen(?:[-\s]?\d+(?:\.\d+)?)?\b|\bqwq\b", re.IGNORECASE), "qwen", "Qwen"),
    (re.compile(r"\bgranite\b", re.IGNORECASE), "granite", "Granite"),
    (re.compile(r"\bgemma\b", re.IGNORECASE), "gemma", "Gemma"),
    (re.compile(r"\bdeepseek\b", re.IGNORECASE), "deepseek", "DeepSeek"),
    (re.compile(r"\bexaone\b", re.IGNORECASE), "exaone", "EXAONE"),
    (re.compile(r"\bsmollm\b", re.IGNORECASE), "smollm", "SmolLM"),
    (re.compile(r"\bminicpm\b", re.IGNORECASE), "minicpm", "MiniCPM"),
    (re.compile(r"\bolmo\b", re.IGNORECASE), "olmo", "OLMo"),
    (re.compile(r"\bchatglm\b|\bglm\b", re.IGNORECASE), "glm", "ChatGLM"),
    (re.compile(r"\bbaichuan\b", re.IGNORECASE), "baichuan", "Baichuan"),
    (re.compile(r"\bfalcon\b", re.IGNORECASE), "falcon", "Falcon"),
    (re.compile(r"\binternlm\b", re.IGNORECASE), "internlm", "InternLM"),
    (re.compile(r"\byi\b", re.IGNORECASE), "yi", "Yi"),
    (re.compile(r"\b(?:bge|e5|gte|minilm|bert)\b", re.IGNORECASE), "bert", "BERT"),
)

ARCHITECTURE_PATTERNS: dict[str, str] = {
    pattern.pattern: architecture for pattern, _, architecture in FAMILY_PATTERNS
}

MODEL_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "llm": (
        "chat",
        "instruct",
        "reason",
        "reasoning",
        "coder",
        "code",
        "assistant",
        "dialog",
        "generate",
        "completion",
    ),
    "embedding": (
        "embedding",
        "embed",
        "bge",
        "e5",
        "gte",
        "sentence",
        "minilm",
        "bert",
        "retriever",
        "rerank",
    ),
    "vision": (
        "vision",
        "image",
        "clip",
        "vit",
        "llava",
        "blip",
        "sam",
    ),
}

_FORMAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgguf\b|\bggml\b", re.IGNORECASE), "gguf"),
    (
        re.compile(
            r"openvino_model\.(?:xml|bin)|openvino/|(?:^|[-_/])ov(?:$|[-_/])",
            re.IGNORECASE,
        ),
        "openvino-ir",
    ),
    (re.compile(r"\bonnx\b", re.IGNORECASE), "onnx"),
    (re.compile(r"\bsafetensors?\b", re.IGNORECASE), "safetensors"),
    (re.compile(r"\bpytorch\b|(?:^|[-_/])pt(?:$|[-_/])", re.IGNORECASE), "pytorch"),
)

_QUANTIZATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnf4\b", re.IGNORECASE), "NF4"),
    (re.compile(r"\bfp8\b|\bf8\b|\be4m3\b|\be5m2\b", re.IGNORECASE), "FP8"),
    (re.compile(r"\bbf16\b", re.IGNORECASE), "BF16"),
    (re.compile(r"\bfp16\b|\bf16\b|\bfloat16\b", re.IGNORECASE), "FP16"),
    (re.compile(r"\bfp32\b|\bf32\b|\bfloat32\b", re.IGNORECASE), "FP32"),
    (re.compile(r"\bint4\b|\bi4\b|\b4bit\b", re.IGNORECASE), "INT4"),
    (re.compile(r"\bint8\b|\bi8\b|\b8bit\b", re.IGNORECASE), "INT8"),
)


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class DictLikeDataclass(Mapping[str, Any]):
    """Small helper for dataclass payloads that still support legacy dict access."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            yield field.name

    def __len__(self) -> int:
        return len(fields(self))

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def copy(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class DetectedModelMetadata(DictLikeDataclass):
    """Normalized metadata inferred from a model identifier."""

    family: str = ""
    architecture: str = ""
    quantization: str = ""
    parameters: str = ""
    type: str = ""
    format: str = ""
    backend: str = ""
    task: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "family",
            "architecture",
            "quantization",
            "parameters",
            "type",
            "format",
            "backend",
            "task",
        ):
            object.__setattr__(self, field_name, _normalize_text(getattr(self, field_name)))

        expected_task = detect_task("", model_type=self.type) if self.type else ""
        if expected_task and self.task and self.task != expected_task:
            raise ValueError(
                f"task {self.task!r} is not valid for detected model type {self.type!r}"
            )

        expected_backend = detect_backend("", model_format=self.format) if self.format else ""
        if expected_backend and self.backend and self.backend != expected_backend:
            raise ValueError(
                f"backend {self.backend!r} is not valid for detected format {self.format!r}"
            )


def detect_quantization(text: str) -> str:
    """Detect a quantization string from a model name, repo, or filename."""
    if not text:
        return ""

    gguf_match = re.search(r"\b(q[2-8](?:[-_][a-z0-9]+)*)\b", text, re.IGNORECASE)
    if gguf_match and re.search(r"\bgguf\b|\bggml\b", text, re.IGNORECASE):
        quantization = gguf_match.group(1).upper().replace("-", "_")
        return quantization.removesuffix("_GGUF").removesuffix("_GGML")

    for pattern, quantization in _QUANTIZATION_PATTERNS:
        if pattern.search(text):
            return quantization

    return ""


def detect_parameters(text: str) -> str:
    """Detect a parameter count such as 7B, 1.1B, 560M, or 22M."""
    if not text:
        return ""

    match = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*([BM])\b", text, re.IGNORECASE)
    if not match:
        return ""

    amount, unit = match.groups()
    return f"{amount}{unit.upper()}"


def detect_family(text: str) -> str:
    """Detect the canonical model family slug."""
    if not text:
        return ""

    for pattern, family, _ in FAMILY_PATTERNS:
        if pattern.search(text):
            return family

    return ""


def detect_architecture(text: str) -> str:
    """Detect the human-readable model architecture/family name."""
    if not text:
        return ""

    for pattern, _, architecture in FAMILY_PATTERNS:
        if pattern.search(text):
            return architecture

    return ""


def detect_format(text: str) -> str:
    """Detect a model packaging/runtime format such as openvino-ir or gguf."""
    if not text:
        return ""

    for pattern, model_format in _FORMAT_PATTERNS:
        if pattern.search(text):
            return model_format

    return ""


def detect_model_type(text: str) -> str:
    """Detect a broad model type such as llm, embedding, or vision."""
    if not text:
        return ""

    lowered = text.lower()

    for keyword in MODEL_TYPE_KEYWORDS["embedding"]:
        if keyword in lowered:
            return "embedding"

    if re.search(r"(?:^|[-_/])vl(?:$|[-_/])|vision-language", lowered):
        return "vision"

    for keyword in MODEL_TYPE_KEYWORDS["vision"]:
        if keyword in lowered:
            return "vision"

    if detect_family(text):
        return "llm"

    for keyword in MODEL_TYPE_KEYWORDS["llm"]:
        if keyword in lowered:
            return "llm"

    return ""


def detect_task(text: str, model_type: str | None = None) -> str:
    """Detect an inference task name from metadata-bearing text."""
    inferred_type = model_type or detect_model_type(text)
    if inferred_type == "embedding":
        return "feature-extraction"
    if inferred_type == "vision":
        return "image-text-to-text"
    if inferred_type == "llm":
        return "text-generation"
    return ""


def detect_backend(text: str, model_format: str | None = None) -> str:
    """Detect the likely runtime backend for a model artifact."""
    inferred_format = model_format or detect_format(text)
    if inferred_format == "openvino-ir":
        return "openvino"
    if inferred_format == "gguf":
        return "llama.cpp"
    if inferred_format in {"onnx", "safetensors", "pytorch"}:
        return "transformers"
    return ""


def detect_model_metadata(
    text: str,
    *,
    default_type: str = "",
    default_backend: str = "",
    default_format: str = "",
    default_task: str = "",
) -> DetectedModelMetadata:
    """Detect shared model metadata fields from free-form model identifiers."""
    family = detect_family(text)
    architecture = detect_architecture(text)
    quantization = detect_quantization(text)
    parameters = detect_parameters(text)
    model_type = default_type or detect_model_type(text)
    model_format = default_format or detect_format(text)
    backend = default_backend or detect_backend(text, model_format)
    task = default_task or detect_task(text, model_type)

    return DetectedModelMetadata(
        family=family,
        architecture=architecture,
        quantization=quantization,
        parameters=parameters,
        type=model_type,
        format=model_format,
        backend=backend,
        task=task,
    )
