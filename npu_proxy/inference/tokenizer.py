"""Tokenizer-backed token counting helpers."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from npu_proxy.models.registry import DEFAULT_MODEL_DIR

logger = logging.getLogger(__name__)


class TokenCountPrecision(Enum):
    """Precision levels for token counting."""

    FAST = "fast"
    APPROXIMATE = "approximate"
    EXACT = "exact"


@dataclass(frozen=True)
class TokenCountResult:
    """Detailed token counting result."""

    count: int
    requested_precision: TokenCountPrecision
    achieved_precision: TokenCountPrecision
    model: str | None = None
    tokenizer_backend: str | None = None
    fallback_reason: str | None = None

    @property
    def exact(self) -> bool:
        """Return True when exact tokenizer accounting was used."""

        return self.achieved_precision == TokenCountPrecision.EXACT


_TOKEN_PATTERN = re.compile(
    r"""
    '[a-zA-Z]+
    |[a-zA-Z]+
    |[0-9]+
    |[^\s\w]
""",
    re.VERBOSE,
)
_CHARS_PER_TOKEN = 4
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def _count_tokens_fast(text: str) -> int:
    """Estimate token count using a character ratio."""

    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0


def _count_tokens_regex(text: str) -> int:
    """Approximate token count using a regex split."""

    return len(_TOKEN_PATTERN.findall(text))


def _model_root() -> Path:
    return Path(os.environ.get("NPU_PROXY_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser().resolve()


def _resolve_trusted_local_model_path(model_path: str | Path | None) -> Path | None:
    """Resolve an operator-provided local tokenizer path.

    This helper is intentionally separate from request-facing model-id
    resolution: callers must only use it for trusted local configuration.
    """
    if not model_path:
        return None
    resolved = Path(model_path).expanduser().resolve()
    return resolved if resolved.exists() else None


def _resolve_model_path(model: str | None) -> Path | None:
    """Resolve a request-facing model id under the configured model root."""

    if not model:
        return None

    raw_model = str(model)
    candidate = Path(raw_model)
    if candidate.is_absolute() or ".." in candidate.parts:
        logger.warning("Rejected unsafe tokenizer model id: %s", raw_model)
        return None

    model_root = _model_root()
    model_path = (model_root / raw_model).resolve()
    try:
        model_path.relative_to(model_root)
    except ValueError:
        logger.warning("Rejected tokenizer model id outside model root: %s", raw_model)
        return None
    return model_path if model_path.exists() else None


def _has_tokenizer_assets(model_path: Path | None) -> bool:
    """Return True when a model directory looks tokenizer-capable."""

    return bool(model_path and model_path.is_dir() and any((model_path / name).exists() for name in _TOKENIZER_FILES))


@lru_cache(maxsize=8)
def get_model_tokenizer(model: str | None) -> Any | None:
    """Load and cache a local tokenizer for a model."""

    model_path = _resolve_model_path(model)
    if not _has_tokenizer_assets(model_path):
        return None

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific
        logger.debug("transformers unavailable for tokenizer loading: %s", exc)
        return None

    try:
        return AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as exc:
        logger.warning("Unable to load tokenizer for %s from %s: %s", model, model_path, exc)
        return None


def clear_tokenizer_cache() -> None:
    """Clear the cached tokenizer instances."""

    get_model_tokenizer.cache_clear()


def _count_tokens_exact(text: str, model: str | None = None) -> int:
    """Count tokens with the model tokenizer."""

    tokenizer = get_model_tokenizer(model)
    if tokenizer is None:
        raise LookupError(f"No tokenizer available for model '{model}'")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def count_tokens_with_details(
    text: str,
    precision: TokenCountPrecision = TokenCountPrecision.APPROXIMATE,
    model: str | None = None,
) -> TokenCountResult:
    """Count tokens and report how the count was produced."""

    if not text or not text.strip():
        return TokenCountResult(
            count=0,
            requested_precision=precision,
            achieved_precision=precision,
            model=model,
        )

    if precision == TokenCountPrecision.FAST:
        return TokenCountResult(
            count=_count_tokens_fast(text),
            requested_precision=precision,
            achieved_precision=TokenCountPrecision.FAST,
            model=model,
        )

    if precision == TokenCountPrecision.EXACT:
        try:
            return TokenCountResult(
                count=_count_tokens_exact(text, model=model),
                requested_precision=precision,
                achieved_precision=TokenCountPrecision.EXACT,
                model=model,
                tokenizer_backend="transformers",
            )
        except Exception as exc:
            logger.warning("Exact token counting failed for model %s; using approximate count: %s", model, exc)
            return TokenCountResult(
                count=_count_tokens_regex(text),
                requested_precision=precision,
                achieved_precision=TokenCountPrecision.APPROXIMATE,
                model=model,
                fallback_reason=str(exc),
            )

    return TokenCountResult(
        count=_count_tokens_regex(text),
        requested_precision=precision,
        achieved_precision=TokenCountPrecision.APPROXIMATE,
        model=model,
    )


def count_tokens_best_effort(text: str, model: str | None = None) -> TokenCountResult:
    """Prefer exact tokenizer counts and fall back safely when unavailable."""

    return count_tokens_with_details(
        text,
        precision=TokenCountPrecision.EXACT,
        model=model,
    )


def count_tokens(
    text: str,
    precision: TokenCountPrecision = TokenCountPrecision.APPROXIMATE,
    model: str | None = None,
) -> int:
    """Count tokens in text."""

    return count_tokens_with_details(text, precision=precision, model=model).count


def count_tokens_safe(
    text: str,
    precision: TokenCountPrecision = TokenCountPrecision.APPROXIMATE,
    fallback_to_words: bool = True,
    model: str | None = None,
) -> int:
    """Count tokens with optional word-count fallback."""

    try:
        return count_tokens(text, precision=precision, model=model)
    except Exception as exc:
        logger.warning("Token counting failed; using word-count fallback: %s", exc)
        if fallback_to_words:
            return len(text.split()) if text else 0
        raise


def count_prompt_tokens(
    prompt: str,
    precision: TokenCountPrecision = TokenCountPrecision.APPROXIMATE,
    model: Optional[str] = None,
) -> int:
    """Count tokens in a prompt string."""

    return count_tokens(prompt, precision=precision, model=model)


def count_completion_tokens(
    completion: str,
    precision: TokenCountPrecision = TokenCountPrecision.APPROXIMATE,
    model: Optional[str] = None,
) -> int:
    """Count tokens in generated completion text."""

    return count_tokens(completion, precision=precision, model=model)
