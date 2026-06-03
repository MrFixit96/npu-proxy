"""Shared API helpers for headers, request IDs, validation, prompts, and errors."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import orjson
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import Response

from npu_proxy.inference.chat_templates import render_chat_prompt
from npu_proxy.inference.embedding_engine import EmbeddingInferenceError
from npu_proxy.models.registry import RegistryModelInfo, get_model_info

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")
_REQUEST_ID_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
MAX_HEADER_VALUE_LENGTH = 1024
SINGLE_ENGINE_ROUTE_REASON = "single_engine_runtime"


def to_http_header_value(value: Any) -> str | None:
    """Convert arbitrary values into single-line HTTP-safe header strings."""
    if value is None:
        return None

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(part.strip() for part in text.split("\n") if part.strip())
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > MAX_HEADER_VALUE_LENGTH:
        text = text[:MAX_HEADER_VALUE_LENGTH]
    return text or None


def generate_request_id(*, prefix: str, hex_chars: int) -> str:
    """Generate a prefixed request ID with a fixed number of UUID hex chars."""

    return f"{prefix}{uuid.uuid4().hex[:hex_chars]}"


def resolve_request_id(
    explicit_request_id: str | None,
    *,
    prefix: str,
    hex_chars: int,
    generator: Callable[[], str] | None = None,
) -> str:
    """Use a provided request ID when present, otherwise generate one."""

    if explicit_request_id is not None:
        sanitized = to_http_header_value(explicit_request_id)
        if sanitized:
            sanitized = _REQUEST_ID_RE.sub("-", sanitized)[:MAX_HEADER_VALUE_LENGTH]
            if sanitized.strip("-_.:"):
                return sanitized
    return generator() if generator is not None else generate_request_id(prefix=prefix, hex_chars=hex_chars)


def apply_headers(response: Response, headers: Mapping[str, str]) -> None:
    """Apply a header mapping to a mutable HTTP response."""

    for name, value in headers.items():
        sanitized = to_http_header_value(value)
        if sanitized is not None:
            response.headers[name] = sanitized


def add_request_id_header(response: Response, request_id: str) -> None:
    """Attach an X-Request-ID header to a response."""

    apply_headers(response, {"X-Request-ID": _REQUEST_ID_RE.sub("-", request_id)})


def build_single_engine_execution_headers(
    token_count: int,
    *,
    execution_device: str,
    request_id: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build truthful single-engine execution headers."""

    headers = {
        "X-NPU-Proxy-Device": execution_device,
        "X-NPU-Proxy-Route-Reason": SINGLE_ENGINE_ROUTE_REASON,
        "X-NPU-Proxy-Token-Count": str(token_count),
    }
    if request_id:
        headers["X-Request-ID"] = request_id
    if extra_headers:
        headers.update(extra_headers)
    return headers


def add_single_engine_execution_headers(
    response: Response,
    token_count: int,
    *,
    execution_device: str,
    request_id: str | None = None,
) -> None:
    """Attach truthful single-engine execution headers to a response."""

    apply_headers(
        response,
        build_single_engine_execution_headers(
            token_count,
            execution_device=execution_device,
            request_id=request_id,
        ),
    )


def build_embedding_engine_headers(engine_info: Mapping[str, Any]) -> dict[str, str]:
    """Build additive embedding execution headers from engine metadata."""

    headers = {
        "X-NPU-Proxy-Device": to_http_header_value(engine_info.get("device", "unknown")) or "unknown",
        "X-NPU-Proxy-Model": (
            to_http_header_value(
                engine_info.get("resolved_model") or engine_info.get("model_name") or "unknown"
            )
            or "unknown"
        ),
        "X-NPU-Proxy-Fallback": "true" if engine_info.get("is_fallback") else "false",
    }
    fallback_reason = to_http_header_value(
        engine_info.get("fallback_reason") or engine_info.get("load_error")
    )
    if fallback_reason:
        headers["X-NPU-Proxy-Fallback-Reason"] = fallback_reason
    return headers


def apply_embedding_engine_headers(response: Response, engine_info: Mapping[str, Any]) -> None:
    """Attach embedding execution headers to a response."""

    apply_headers(response, build_embedding_engine_headers(engine_info))


def embedding_failure_reason(
    engine_info: Mapping[str, Any],
    *,
    default: str,
) -> str:
    """Extract a stable human-readable embedding failure reason."""

    return str(
        engine_info.get("fallback_reason")
        or engine_info.get("load_error")
        or default
    )


def get_registered_model_or_404(model: str) -> RegistryModelInfo:
    """Resolve a registry model or raise the standard 404 contract."""

    model_info = get_model_info(model)
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    return model_info


def validate_registered_model(model: str) -> None:
    """Validate that a registry-backed model exists."""

    get_registered_model_or_404(model)


def render_api_chat_prompt(
    messages: Sequence[Mapping[str, Any] | Any],
    *,
    model: str | None = None,
) -> str:
    """Render request messages through the shared chat template integration point."""

    return render_chat_prompt(messages, model=model).prompt


def validate_embedding_batch_result_count(
    inputs: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> None:
    """Reject embedding responses that do not align 1:1 with inputs."""

    expected_count = len(inputs)
    actual_count = len(embeddings)
    if actual_count != expected_count:
        raise EmbeddingInferenceError(
            f"Embedding batch returned {actual_count} result(s) for {expected_count} input(s)"
        )


def openai_error_content(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, dict[str, str | None]]:
    """Build an OpenAI-style error envelope."""

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def openai_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """Create an OpenAI-compatible handled error response."""

    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(
        status_code=status_code,
        content=openai_error_content(
            message,
            error_type=error_type,
            param=param,
            code=code,
        ),
        headers=headers,
    )


def openai_error_response_from_http_exception(
    exc: HTTPException,
    *,
    request_id: str | None = None,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Convert a handled HTTPException into OpenAI's error envelope."""

    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    error_type = "server_error" if exc.status_code >= 500 else "invalid_request_error"
    return openai_error_response(
        status_code=exc.status_code,
        message=message,
        error_type=error_type,
        param=param,
        code=code,
        request_id=request_id,
    )


def build_openai_stream_error_chunk(message: str) -> str:
    """Build an OpenAI-compatible SSE error frame for stream failures."""

    return (
        f"data: {orjson.dumps(openai_error_content(message, error_type='server_error', code='streaming_error')).decode()}\n\n"
    )


def ollama_error_message(detail: object) -> str:
    """Normalize handled errors into Ollama's flat error string."""

    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("error")
        if isinstance(message, str) and message:
            return message
    return "Request failed"


def ollama_error_response(
    exc: HTTPException,
    *,
    response_headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Convert handled endpoint errors into Ollama's flat error envelope."""

    headers = dict(response_headers or {})
    if exc.headers:
        headers.update(exc.headers)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": ollama_error_message(exc.detail)},
        headers=headers or None,
    )


def build_ollama_stream_error_chunk(model: str, message: str) -> str:
    """Build a terminal Ollama NDJSON error frame for stream failures."""

    return orjson.dumps({"model": model, "error": message}).decode() + "\n"
