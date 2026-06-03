"""Shared chat prompt rendering helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from npu_proxy.inference.tokenizer import get_model_tokenizer

logger = logging.getLogger(__name__)

_DISABLE_CHAT_TEMPLATES_ENV = "NPU_PROXY_DISABLE_CHAT_TEMPLATES"


@dataclass(frozen=True)
class RenderedChatPrompt:
    """Rendered chat prompt and metadata about how it was produced."""

    prompt: str
    used_chat_template: bool
    model: str | None = None
    source: str = "legacy"
    fallback_reason: str | None = None


def _is_truthy(value: str | None) -> bool:
    """Return True for common environment variable truthy values."""

    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def chat_templates_enabled() -> bool:
    """Return whether chat template rendering is enabled."""

    return not _is_truthy(os.environ.get(_DISABLE_CHAT_TEMPLATES_ENV))


def _message_value(message: Mapping[str, Any] | Any, key: str) -> Any:
    """Read a message field from either a mapping or an object."""

    if isinstance(message, Mapping):
        return message.get(key)
    return getattr(message, key, None)


def _normalized_messages(messages: Sequence[Mapping[str, Any] | Any]) -> list[dict[str, str]]:
    """Normalize chat messages for shared rendering."""

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(_message_value(message, "role") or "")
        content = str(_message_value(message, "content") or "")
        normalized.append({"role": role, "content": content})
    return normalized


def legacy_format_chat_messages(messages: Sequence[Mapping[str, Any] | Any]) -> str:
    """Render chat messages using the historical role-prefixed format."""

    prompt_parts: list[str] = []
    for message in _normalized_messages(messages):
        role = message["role"].lower()
        if role in {"system", "developer"}:
            label = "Developer" if role == "developer" else "System"
            prompt_parts.append(f"{label}: {message['content']}")
        elif role == "user":
            prompt_parts.append(f"User: {message['content']}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {message['content']}")
        elif role == "tool":
            prompt_parts.append(f"Tool: {message['content']}")
        else:
            logger.warning("Legacy chat formatter preserving unknown role %r", message["role"])
            prompt_parts.append(f"{message['role'] or 'Unknown'}: {message['content']}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def render_chat_prompt(
    messages: Sequence[Mapping[str, Any] | Any],
    model: str | None = None,
    add_generation_prompt: bool = True,
) -> RenderedChatPrompt:
    """Render a chat prompt using a model tokenizer when available."""

    legacy_prompt = legacy_format_chat_messages(messages)

    if not chat_templates_enabled():
        return RenderedChatPrompt(
            prompt=legacy_prompt,
            used_chat_template=False,
            model=model,
            source="legacy",
            fallback_reason="chat_templates_disabled",
        )

    tokenizer = get_model_tokenizer(model)
    chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    if tokenizer is None or not chat_template:
        return RenderedChatPrompt(
            prompt=legacy_prompt,
            used_chat_template=False,
            model=model,
            source="legacy",
            fallback_reason="chat_template_unavailable",
        )

    try:
        rendered_prompt = tokenizer.apply_chat_template(
            _normalized_messages(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered_prompt, str) or not rendered_prompt.strip():
            raise ValueError("Tokenizer chat template returned an empty prompt")
        return RenderedChatPrompt(
            prompt=rendered_prompt,
            used_chat_template=True,
            model=model,
            source="tokenizer",
        )
    except Exception as exc:
        logger.debug("Falling back to legacy chat formatting for %s: %s", model, exc)
        return RenderedChatPrompt(
            prompt=legacy_prompt,
            used_chat_template=False,
            model=model,
            source="legacy",
            fallback_reason="chat_template_render_failed",
        )
