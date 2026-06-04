"""Ollama-compatible API endpoints.

This module provides Ollama API compatibility for the NPU Proxy, enabling
drop-in replacement for Ollama clients. All endpoints follow the Ollama API
specification while routing inference to Intel NPU hardware via OpenVINO.

Ollama API Reference:
    https://github.com/ollama/ollama/blob/main/docs/api.md

Supported Endpoints:
    - POST /api/generate - Raw text completion
    - POST /api/chat - Chat completion with message history
    - POST /api/embed - Generate embeddings (current format)
    - POST /api/embeddings - Generate embeddings (legacy format)
    - GET /api/tags - List locally available models
    - GET /api/ps - List running models
    - GET /api/version - Get version information
    - POST /api/show - Show model information
    - POST /api/pull - Download models from HuggingFace
    - GET /api/search - Search for OpenVINO models
    - GET /api/models/known - List pre-mapped model names

Headers:
    All inference endpoints return the following headers:
    - X-Request-ID: Unique identifier for request tracing
    - X-NPU-Proxy-Device: Device used for inference (npu, cpu, gpu)
    - X-NPU-Proxy-Route-Reason: Why this device was selected
    - X-NPU-Proxy-Token-Count: Estimated token count for routing

Example:
    >>> import requests
    >>> response = requests.post("http://localhost:8080/api/generate", json={
    ...     "model": "tinyllama",
    ...     "prompt": "Why is the sky blue?",
    ...     "stream": False
    ... })
    >>> print(response.json()["response"])
"""
import time
import os
import asyncio
import logging
import json
import orjson
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.responses import Response
from pydantic import BaseModel, Field, SecretStr
from typing import Any, Literal

from npu_proxy import OLLAMA_VERSION
from npu_proxy.routing.context_router import get_context_router, RoutingResult
from npu_proxy.metrics import record_routing_decision
from npu_proxy.api.header_utils import (
    add_request_id_header,
    add_single_engine_execution_headers,
    apply_embedding_engine_headers as shared_apply_embedding_engine_headers,
    build_single_engine_execution_headers,
    embedding_failure_reason,
    generate_request_id as shared_generate_request_id,
    get_registered_model_or_404,
    validate_embedding_batch_result_count,
    validate_registered_model,
)
from npu_proxy.models.ollama_defaults import merge_with_defaults
from npu_proxy.models.parameter_mapper import map_parameters
from npu_proxy.inference.streaming import FinishReason, determine_finish_reason, stream_engine_tokens
from npu_proxy.inference import routing_service
from npu_proxy.inference.chat_templates import render_chat_prompt
from npu_proxy.inference.tokenizer import count_tokens_best_effort
from npu_proxy.config import DEFAULT_INFERENCE_TIMEOUT, load_device_queue_timeout, load_fallback_on_busy
from npu_proxy.api.embeddings import (
    MAX_EMBEDDING_BATCH_SIZE,
    MAX_EMBEDDING_TEXT_CHARS,
    MAX_EMBEDDING_TOTAL_CHARS,
)

router = APIRouter(prefix="/api", tags=["ollama"])
logger = logging.getLogger(__name__)


class ModelDetails(BaseModel):
    """Model metadata details in Ollama format.

    Attributes:
        parent_model: Base model this was derived from.
        format: Model format (gguf, openvino, etc.).
        family: Model architecture family (llama, phi, etc.).
        families: List of architecture families if hybrid.
        parameter_size: Human-readable parameter count (e.g., "1.1B").
        quantization_level: Quantization format (Q4_0, INT4, etc.).
    """
    parent_model: str = ""
    format: str = "gguf"
    family: str = "llama"
    families: list[str] | None = None
    parameter_size: str = "1.1B"
    quantization_level: str = "Q4_0"


class TagsModel(BaseModel):
    """Locally available model entry in Ollama /api/tags format."""

    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails


class TagsResponse(BaseModel):
    """Response for /api/tags endpoint listing local models."""

    models: list[TagsModel]


class RunningModel(BaseModel):
    """Information about a currently loaded model.

    Attributes:
        name: Model name as specified in pull/run command.
        model: Canonical model identifier.
        size: Model size in bytes.
        digest: SHA256 digest of model weights.
        details: Model metadata details.
        expires_at: ISO 8601 timestamp when model will be unloaded.
        size_vram: Amount of VRAM used (0 for NPU models).
    """
    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    expires_at: str
    size_vram: int


class PsResponse(BaseModel):
    """Response for /api/ps endpoint listing running models.

    Attributes:
        models: List of currently loaded models.
    """
    models: list[RunningModel]


class VersionResponse(BaseModel):
    """Response for /api/version endpoint.

    Attributes:
        version: Version string in format "X.Y.Z-npu-proxy".
    """
    version: str


class ShowRequest(BaseModel):
    """Request body for /api/show endpoint.

    Attributes:
        model: Name of the model to show details for.
        verbose: If true, include full model configuration.
    """
    model: str
    verbose: bool = False


class ShowResponse(BaseModel):
    """Response for /api/show endpoint with model information.

    Attributes:
        modelfile: Modelfile content in Ollama format.
        parameters: Model parameters as string.
        template: Chat template if applicable.
        details: Model metadata details.
        model_info: Extended model information dictionary.
    """
    modelfile: str
    parameters: str
    template: str
    details: ModelDetails
    model_info: dict


class GenerateRequest(BaseModel):
    """Request body for /api/generate endpoint.

    Ollama-compatible request for raw text generation without
    chat formatting.

    Attributes:
        model: Name of the model to use.
        prompt: Raw prompt text to send to the model.
        stream: If true, stream response tokens. Default: True.
        options: Generation options (temperature, top_p, etc.).

    Example:
        >>> request = GenerateRequest(
        ...     model="tinyllama",
        ...     prompt="Explain quantum computing",
        ...     stream=False,
        ...     options={"temperature": 0.7}
        ... )
    """
    model: str
    prompt: str
    stream: bool = True
    options: dict | None = None


class GenerateResponse(BaseModel):
    """Response for /api/generate endpoint.

    For streaming responses, multiple GenerateResponse objects are
    returned as newline-delimited JSON, with done=False for intermediate
    chunks and done=True for the final chunk.

    Attributes:
        model: Model name used for generation.
        created_at: ISO 8601 timestamp of response creation.
        response: Generated text (partial for streaming, full otherwise).
        done: True if this is the final response chunk.
        context: Token context for continuing conversation (optional).
        total_duration: Total generation time in nanoseconds.
        load_duration: Model loading time in nanoseconds.
        prompt_eval_count: Number of tokens in the prompt.
        prompt_eval_duration: Prompt processing time in nanoseconds.
        eval_count: Number of generated tokens.
        eval_duration: Token generation time in nanoseconds.
    """
    model: str
    created_at: str
    response: str
    done: bool
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    done_reason: FinishReason | None = None


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Message role (system, user, or assistant).
        content: Message content text.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    """Request body for /api/chat endpoint.

    Ollama-compatible request for chat completion with message history.

    Attributes:
        model: Name of the model to use.
        messages: List of chat messages in conversation order.
        stream: If true, stream response tokens. Default: True.
        options: Generation options (temperature, top_p, etc.).

    Example:
        >>> request = ChatRequest(
        ...     model="tinyllama",
        ...     messages=[
        ...         ChatMessage(role="system", content="You are helpful."),
        ...         ChatMessage(role="user", content="Hello!")
        ...     ],
        ...     stream=False
        ... )
    """
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    options: dict | None = None


class ChatResponse(BaseModel):
    """Response for /api/chat endpoint.

    For streaming responses, multiple ChatResponse objects are returned
    as newline-delimited JSON, with done=False for intermediate chunks
    and done=True for the final chunk.

    Attributes:
        model: Model name used for generation.
        created_at: ISO 8601 timestamp of response creation.
        message: Assistant's response message.
        done: True if this is the final response chunk.
        total_duration: Total generation time in nanoseconds.
        eval_count: Number of generated tokens.
    """
    model: str
    created_at: str
    message: ChatMessage
    done: bool
    total_duration: int | None = None
    eval_count: int | None = None
    done_reason: FinishReason | None = None


# Import from registry for single source of truth
from npu_proxy.models.registry import MODELS_INFO as REGISTRY_MODELS_INFO, get_model_info


def _get_ollama_model_info(model_id: str) -> dict:
    """Get model info in Ollama format from registry.

    Converts internal registry format to Ollama-compatible model info.

    Args:
        model_id: The model identifier (e.g., "tinyllama").

    Returns:
        Dictionary with size, digest, parameter_size, quantization_level,
        and family fields in Ollama format.
    """
    info = get_model_info(model_id) or {}
    return {
        "size": info.get("size", 0),
        "digest": info.get("digest", ""),
        "parameter_size": info.get("parameter_size", ""),
        "quantization_level": info.get("quantization", ""),
        "family": info.get("family", "unknown"),
    }


# Re-export for backward compatibility
MODELS_INFO = {k: _get_ollama_model_info(k) for k in REGISTRY_MODELS_INFO.keys()}


def _generate_request_id() -> str:
    """Generate a unique request ID for tracing.

    Returns:
        A UUID-based request ID string prefixed with 'req-'.
    """
    return shared_generate_request_id(prefix="req-", hex_chars=12)


def add_request_headers(response: Response, request_id: str) -> None:
    """Add request ID header for tracing.

    Args:
        response: FastAPI response object.
        request_id: Unique request identifier.
    """
    add_request_id_header(response, request_id)


_ALLOWED_EMBEDDING_DEVICES = {"NPU", "CPU", "GPU", "AUTO"}


def _get_embedding_device_from_options(options: dict | None) -> str | None:
    """Extract and validate an optional embedding device override from Ollama options."""
    if not isinstance(options, dict):
        return None
    device = options.get("device") or options.get("embedding_device")
    if not device:
        return None
    normalized = str(device).strip().upper()
    if normalized not in _ALLOWED_EMBEDDING_DEVICES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported embedding device override",
                "code": "invalid_embedding_device",
            },
        )
    return normalized


def _apply_embedding_engine_headers(response: Response, engine_info: dict) -> None:
    """Expose embedding routing details additively via headers."""
    shared_apply_embedding_engine_headers(response, engine_info)


def _ollama_model_format(info: dict) -> str:
    model_format = str(info.get("format") or "")
    backend = str(info.get("backend") or "")
    if backend == "openvino" or model_format.startswith("openvino"):
        return "openvino"
    return model_format or "unknown"


def _ollama_model_details(info: dict) -> ModelDetails:
    family = info.get("family", "unknown")
    return ModelDetails(
        format=_ollama_model_format(info),
        family=family,
        families=[family] if family and family != "unknown" else None,
        parameter_size=info.get("parameter_size", "unknown"),
        quantization_level=info.get("quantization", "unknown"),
    )


def _ollama_model_info_payload(info: dict, *, verbose: bool) -> dict:
    payload = {
        "general.architecture": info.get("architecture") or info.get("family", "unknown"),
        "general.parameter_count": info.get("parameter_size", "unknown"),
        "general.quantization_version": info.get("quantization", "unknown"),
        "general.file_type": _ollama_model_format(info),
        "npu_proxy.backend": info.get("backend", "unknown"),
        "npu_proxy.task": info.get("task", "unknown"),
        "npu_proxy.model_type": info.get("type", "unknown"),
    }
    if info.get("context_length"):
        payload["npu_proxy.context_length"] = info["context_length"]
    if info.get("dimensions"):
        payload["npu_proxy.embedding_dimensions"] = info["dimensions"]
    if verbose:
        payload.update(
            {
                "npu_proxy.id": info.get("id", ""),
                "npu_proxy.name": info.get("name", ""),
                "npu_proxy.repo_id": info.get("repo_id", ""),
                "npu_proxy.storage_key": info.get("storage_key", ""),
                "npu_proxy.aliases": list(info.get("aliases") or ()),
                "npu_proxy.description": info.get("description", ""),
            }
        )
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _ollama_parameters(info: dict) -> str:
    parameters: list[str] = []
    if info.get("context_length"):
        parameters.append(f"PARAMETER num_ctx {info['context_length']}")
    return "\n".join(parameters)


def _get_model_chat_template(model: str) -> str:
    try:
        from npu_proxy.inference.tokenizer import get_model_tokenizer

        tokenizer = get_model_tokenizer(model)
    except Exception as exc:
        logger.debug("Unable to load chat template for %s", model, exc_info=True)
        return ""
    template = getattr(tokenizer, "chat_template", "") if tokenizer is not None else ""
    return template if isinstance(template, str) else ""


def _build_modelfile(model: str, parameters: str, template: str) -> str:
    lines = [f"FROM {model}"]
    if parameters:
        lines.extend(parameters.splitlines())
    if template:
        lines.append(f'TEMPLATE """\n{template}\n"""')
    return "\n".join(lines)


@router.get("/tags", response_model=TagsResponse)
async def list_local_models():
    """List locally available models in Ollama /api/tags format."""
    from npu_proxy.models.registry import DEFAULT_MODEL_DIR, scan_available_models

    models: list[TagsModel] = []
    for info in await asyncio.to_thread(scan_available_models):
        model_path = DEFAULT_MODEL_DIR / info["storage_key"]
        try:
            modified_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(model_path.stat().st_mtime))
        except OSError:
            modified_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        models.append(
            TagsModel(
                name=info["id"],
                model=info["id"],
                modified_at=modified_at,
                size=info.get("size", 0),
                digest=info.get("digest", ""),
                details=_ollama_model_details(info),
            )
        )
    return TagsResponse(models=models)


@router.get("/ps", response_model=PsResponse)
async def list_running_models():
    """List currently loaded models.

    Ollama-compatible endpoint that returns information about all models
    currently loaded in memory and ready for inference.

    Returns:
        PsResponse: List of running models with their details.

    Ollama Compatibility:
        - Matches GET /api/ps from Ollama API
        - Returns same response structure as Ollama
        - size_vram is always 0 (NPU uses system memory)
        - expires_at is empty (NPU Proxy doesn't auto-unload)

    Example:
        >>> response = client.get("/api/ps")
        >>> for model in response.json()["models"]:
        ...     print(f"{model['name']}: {model['details']['parameter_size']}")
    """
    from npu_proxy.inference.engine import get_loaded_models, is_model_loaded

    models = []

    if is_model_loaded():
        loaded = get_loaded_models()
        for name, engine in loaded.items():
            info = _get_ollama_model_info(name)
            models.append(RunningModel(
                name=name,
                model=name,
                size=info.get("size", 0),
                digest=info.get("digest", ""),
                details=ModelDetails(
                    parameter_size=info.get("parameter_size", ""),
                    quantization_level=info.get("quantization_level", ""),
                    family=info.get("family", "llama"),
                ),
                expires_at="",
                size_vram=0,
            ))

    return PsResponse(models=models)


@router.get("/version", response_model=VersionResponse)
async def get_version():
    """Get NPU Proxy version information.

    Ollama-compatible endpoint that returns the version string.
    Version includes "-npu-proxy" suffix to identify this implementation.

    Returns:
        VersionResponse: Version string.

    Ollama Compatibility:
        - Matches GET /api/version from Ollama API
        - Returns same response structure as Ollama
        - Version suffix indicates NPU Proxy implementation

    Example:
        >>> response = client.get("/api/version")
        >>> print(response.json()["version"])
        0.2.0-npu-proxy
    """
    return VersionResponse(version=OLLAMA_VERSION)


@router.post("/show", response_model=ShowResponse)
async def show_model(request: ShowRequest, response: Response):
    """Show detailed information about a model.

    Ollama-compatible endpoint that returns model metadata including
    architecture, parameter count, quantization, and modelfile.

    Args:
        request: ShowRequest with model name and verbose flag.

    Returns:
        ShowResponse: Model details and configuration.

    Raises:
        HTTPException: 404 if model not found in registry.

    Ollama Compatibility:
        - Matches POST /api/show from Ollama API
        - Returns modelfile in Ollama format
        - model_info contains architecture details

    Example:
        >>> response = client.post("/api/show", json={"model": "tinyllama"})
        >>> print(response.json()["details"]["parameter_size"])
        1.1B
    """
    request_id = _generate_request_id()
    add_request_headers(response, request_id)

    try:
        info = get_registered_model_or_404(request.model)
    except HTTPException as exc:
        logger.warning(
            "Ollama show model lookup failed",
            extra={"request_id": request_id, "model": request.model, "error": _error_message(exc.detail)},
        )
        return _ollama_error_response(
            HTTPException(
                status_code=exc.status_code,
                detail=_ollama_error_detail("Model not found", "model_not_found", request_id),
            ),
            response_headers=dict(response.headers),
        )

    parameters = _ollama_parameters(info)
    template = await asyncio.to_thread(_get_model_chat_template, request.model)

    return ShowResponse(
        modelfile=_build_modelfile(request.model, parameters, template),
        parameters=parameters,
        template=template,
        details=_ollama_model_details(info),
        model_info=_ollama_model_info_payload(info, verbose=request.verbose),
    )


def validate_ollama_model(model: str) -> None:
    """Validate that a model exists in the registry.

    Args:
        model: Model name to validate.

    Raises:
        HTTPException: 404 if model not found in registry.
    """
    validate_registered_model(model)


def _get_execution_device(*, load_if_needed: bool) -> str:
    """Return the actual singleton execution device for response reporting."""
    from npu_proxy.inference.execution_state import get_reportable_execution_device

    return get_reportable_execution_device(load_if_needed=load_if_needed)


def _execution_device_from_engine(engine: Any) -> str:
    """Return the actual device reported by an acquired engine."""
    return routing_service.execution_device_from_engine(engine)


def _fallback_reason(
    *,
    routed_device: str,
    execution_device: str,
    engine_slot: object | None = None,
) -> str | None:
    return routing_service.fallback_reason(
        routed_device=routed_device,
        execution_device=execution_device,
        engine_slot=engine_slot,
    )


def _record_routing_execution(routed_device: str, execution_device: str, fallback_reason: str | None) -> None:
    routing_service.record_routing_execution(routed_device, execution_device, fallback_reason)


def _open_routed_engine_slot(device: str) -> tuple[Any, Any]:
    """Acquire the routed device slot and return its engine plus release handle."""
    from npu_proxy.inference.engine import open_routed_engine_slot

    return open_routed_engine_slot(
        device,
        timeout=load_device_queue_timeout(),
        fallback_on_busy=load_fallback_on_busy(),
    )


def _close_engine_slot(slot: object) -> None:
    routing_service.close_engine_slot(slot)


def _device_busy_http_exception(exc: Exception, request_id: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=_ollama_error_detail(str(exc), "device_busy", request_id),
    )


def add_execution_headers(
    response: Response,
    routing_result: RoutingResult,
    *,
    execution_device: str,
    fallback_reason: str | None = None,
) -> None:
    """Add truthful single-engine execution headers to response headers.

    Adds headers indicating which device was selected and why,
    useful for debugging and monitoring routing decisions.

    Args:
        response: FastAPI response object.
        routing_result: The routing decision from context router.
    """
    add_single_engine_execution_headers(
        response,
        routing_result.token_count,
        execution_device=execution_device,
        routed_device=routing_result.device,
        fallback_reason=fallback_reason,
    )


def build_execution_headers(
    routing_result: RoutingResult,
    *,
    execution_device: str,
    request_id: str,
    fallback_reason: str | None = None,
) -> dict[str, str]:
    """Build truthful single-engine execution headers for streaming responses."""
    return build_single_engine_execution_headers(
        routing_result.token_count,
        execution_device=execution_device,
        routed_device=routing_result.device,
        fallback_reason=fallback_reason,
        request_id=request_id,
        extra_headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _ollama_error_detail(message: str, code: str, request_id: str | None = None) -> dict[str, str]:
    """Build a sanitized Ollama error detail with a stable code."""
    safe_message = message
    if request_id:
        safe_message = f"{message} (request id: {request_id})"
    return {"message": safe_message, "code": code}


def _safe_error_message(message: str, request_id: str | None = None) -> str:
    """Append request ID to a sanitized user-facing error message."""
    return _ollama_error_detail(message, "error", request_id)["message"]


def _error_message(detail: object) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("error")
        if isinstance(message, str) and message:
            return message
    return "Request failed"


def _error_code(detail: object) -> str | None:
    if isinstance(detail, dict):
        code = detail.get("code")
        if isinstance(code, str) and code:
            return code
    return None


def _build_ollama_stream_error_chunk(
    model: str,
    message: str,
    *,
    code: str = "inference_failed",
    request_id: str | None = None,
) -> str:
    """Build a terminal NDJSON error frame without leaking raw backend details."""
    return orjson.dumps(
        {
            "model": model,
            "error": _safe_error_message(message, request_id),
            "code": code,
        }
    ).decode() + "\n"


def _ollama_error_response(
    exc: HTTPException,
    *,
    response_headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Convert handled endpoint errors into Ollama's flat error envelope."""
    headers = dict(response_headers or {})
    if exc.headers:
        headers.update(exc.headers)
    content = {"error": _error_message(exc.detail)}
    code = _error_code(exc.detail)
    if code:
        content["code"] = code
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers or None,
    )



def _effective_max_tokens(mapped_options: dict | None, fallback: int = 256) -> int:
    value = (mapped_options or {}).get("max_new_tokens", fallback)
    try:
        token_limit = int(value)
    except (TypeError, ValueError):
        return fallback
    return token_limit if token_limit > 0 else fallback


def _effective_top_p(mapped_options: dict | None, fallback: float = 0.9) -> float:
    value = (mapped_options or {}).get("top_p", fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _validate_ollama_embedding_inputs(texts: list[str], request_id: str) -> None:
    if not texts:
        raise HTTPException(
            status_code=400,
            detail=_ollama_error_detail("Embedding input must contain at least one item", "empty_input", request_id),
        )
    if len(texts) > MAX_EMBEDDING_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=_ollama_error_detail("Embedding input batch is too large", "embedding_batch_too_large", request_id),
        )

    total_chars = 0
    for text in texts:
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=_ollama_error_detail("Embedding input text must not be empty", "empty_input", request_id),
            )
        text_length = len(text)
        if text_length > MAX_EMBEDDING_TEXT_CHARS:
            raise HTTPException(
                status_code=413,
                detail=_ollama_error_detail("Embedding input text is too large", "embedding_input_too_large", request_id),
            )
        total_chars += text_length

    if total_chars > MAX_EMBEDDING_TOTAL_CHARS:
        raise HTTPException(
            status_code=413,
            detail=_ollama_error_detail("Embedding request is too large", "embedding_request_too_large", request_id),
        )


def _mock_tokens(text: str, max_tokens: int) -> tuple[list[str], FinishReason]:
    words = text.split()
    if max_tokens > 0 and len(words) > max_tokens:
        words = words[:max_tokens]
        finish_reason: FinishReason = "length"
    else:
        finish_reason = "stop"
    return [word + " " for word in words], finish_reason


def _mock_text(text: str, max_tokens: int) -> tuple[str, FinishReason]:
    tokens, finish_reason = _mock_tokens(text, max_tokens)
    return "".join(tokens).rstrip(), finish_reason

def _next_sync_iterator_item(iterator):
    """Advance a blocking iterator safely from a worker thread."""
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _validate_embedding_batch_result_count(
    texts: list[str],
    embeddings: list[list[float]],
) -> None:
    """Reject batch embedding responses that do not align 1:1 with inputs."""
    validate_embedding_batch_result_count(texts, embeddings)


def _raise_for_runtime_embedding_fallback(
    engine,
    response: Response,
    *,
    request_id: str,
) -> None:
    """Reject runtime failover that would otherwise look like a successful embed."""
    engine_info = engine.get_engine_info()
    if not (engine_info.get("is_fallback") and engine_info.get("fallback_mode") == "runtime"):
        return

    _apply_embedding_engine_headers(response, engine_info)
    reason = embedding_failure_reason(
        engine_info,
        default="Embedding failed during runtime",
    )
    logger.error(
        "Embedding runtime fallback rejected",
        extra={"request_id": request_id, "error": reason},
    )
    raise HTTPException(
        status_code=500,
        detail=_ollama_error_detail("Embedding failed", "embedding_failed", request_id),
        headers=dict(response.headers),
    )


def get_routing_for_prompt(prompt: str) -> RoutingResult:
    """Get routing decision for a raw prompt.

    Analyzes the prompt to determine optimal device (NPU, CPU, GPU)
    based on token count and context length.

    Args:
        prompt: The raw prompt text.

    Returns:
        RoutingResult with device selection and reasoning.
    """
    router = get_context_router()
    return router.select_device(prompt)


@router.post("/generate")
async def generate(request: GenerateRequest, response: Response):
    """Generate text completion.

    Ollama-compatible endpoint for raw text generation without chat
    formatting. Supports both streaming and non-streaming responses.

    Args:
        request: Generation request with model, prompt, and options.
        response: FastAPI response for setting headers.

    Returns:
        GenerateResponse for non-streaming requests.
        StreamingResponse with newline-delimited JSON for streaming.

    Raises:
        HTTPException: 404 if model not found.
        HTTPException: 503 if inference engine unavailable.
        HTTPException: 504 if inference times out.

    Ollama Compatibility:
        - Matches POST /api/generate from Ollama API
        - Supports all Ollama options (temperature, top_p, top_k, etc.)
        - Returns done=true on final response chunk
        - Streaming uses newline-delimited JSON (not SSE)

    Headers:
        X-Request-ID: Unique request identifier for tracing.
        X-NPU-Proxy-Device: Device used (npu, cpu, gpu).
        X-NPU-Proxy-Route-Reason: Why device was selected.
        X-NPU-Proxy-Token-Count: Estimated prompt token count.

    Streaming Response Format:
        Each line is a JSON object with partial response::

            {"model":"tinyllama","response":"Hello","done":false}
            {"model":"tinyllama","response":" world","done":false}
            {"model":"tinyllama","response":"","done":true,"eval_count":10}

    Example:
        >>> # Non-streaming
        >>> response = client.post("/api/generate", json={
        ...     "model": "tinyllama",
        ...     "prompt": "Why is the sky blue?",
        ...     "stream": False,
        ...     "options": {"temperature": 0.7}
        ... })
        >>> print(response.json()["response"])

        >>> # Streaming
        >>> with client.post("/api/generate", json={
        ...     "model": "tinyllama",
        ...     "prompt": "Hello",
        ...     "stream": True
        ... }, stream=True) as response:
        ...     for line in response.iter_lines():
        ...         chunk = json.loads(line)
        ...         print(chunk["response"], end="")
    """
    # Generate request ID for tracing
    request_id = _generate_request_id()
    add_request_headers(response, request_id)

    try:
        # Validate model exists
        validate_ollama_model(request.model)

        # Route based on context size
        routing = get_routing_for_prompt(request.prompt)
        record_routing_decision(routing.device, routing.reason)

        # Apply parameter pipeline: merge defaults, then map to OpenVINO format
        user_options = request.options or {}
        full_options = merge_with_defaults(user_options)
        mapped_options = map_parameters(full_options)

        use_real = os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") == "1"

        if request.stream:
            engine = None
            engine_slot = None
            engine_setup_error: Exception | None = None
            if use_real:
                try:
                    engine, engine_slot = await asyncio.to_thread(
                        lambda: _open_routed_engine_slot(routing.device)
                    )
                    execution_device = _execution_device_from_engine(engine)
                    fallback_reason = _fallback_reason(
                        routed_device=routing.device,
                        execution_device=execution_device,
                        engine_slot=engine_slot,
                    )
                    _record_routing_execution(routing.device, execution_device, fallback_reason)
                except Exception as exc:
                    from npu_proxy.inference.engine import DeviceBusyError

                    if isinstance(exc, DeviceBusyError):
                        return _ollama_error_response(
                            _device_busy_http_exception(exc, request_id),
                            response_headers=dict(response.headers),
                        )
                    engine_setup_error = exc
                    execution_device = routing.device
                    fallback_reason = None
            else:
                try:
                    execution_device = _get_execution_device(load_if_needed=False)
                    fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
                    _record_routing_execution(routing.device, execution_device, fallback_reason)
                except Exception as exc:
                    logger.error(
                        "Failed to resolve streaming execution device; using routing device",
                        extra={"request_id": request_id, "error": str(exc)},
                    )
                    execution_device = routing.device
                    fallback_reason = None

            async def stream_generate():
                try:
                    max_new_tokens = _effective_max_tokens(mapped_options)
                    if use_real:
                        if engine_setup_error is not None:
                            logger.exception(
                                "Streaming inference setup failed",
                                extra={"request_id": request_id},
                                exc_info=engine_setup_error,
                            )
                            yield _build_ollama_stream_error_chunk(
                                request.model,
                                "Inference failed",
                                code="inference_failed",
                                request_id=request_id,
                            )
                            return
                        stream_error_message: str | None = None
                        finish_reason: FinishReason = "stop"

                        def set_finish_reason(reason: FinishReason) -> None:
                            nonlocal finish_reason
                            finish_reason = reason
                        try:
                            async for token in stream_engine_tokens(
                                engine_factory=lambda: engine,
                                prompt=request.prompt,
                                max_new_tokens=max_new_tokens,
                                temperature=mapped_options.get("temperature", 0.8),
                                request_id=request_id,
                                top_p=_effective_top_p(mapped_options),
                                timeout=float(DEFAULT_INFERENCE_TIMEOUT),
                                finish_reason_callback=set_finish_reason,
                            ):
                                chunk = GenerateResponse(
                                    model=request.model,
                                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    response=token,
                                    done=False,
                                )
                                yield orjson.dumps(chunk.model_dump()).decode() + "\n"
                        except asyncio.CancelledError:
                            raise
                        except TimeoutError as exc:
                            stream_error_message = "Inference timed out"
                            logger.exception("Streaming inference timed out", extra={"request_id": request_id})
                            stream_error_code = "inference_timeout"
                        except Exception as exc:
                            stream_error_message = "Inference failed"
                            logger.exception("Streaming inference failed", extra={"request_id": request_id})
                            stream_error_code = "inference_failed"

                        if stream_error_message is not None:
                            yield _build_ollama_stream_error_chunk(
                                request.model,
                                stream_error_message,
                                code=stream_error_code,
                                request_id=request_id,
                            )
                            return
                    else:
                        tokens, finish_reason = _mock_tokens("Hello! I'm running on Intel NPU.", max_new_tokens)
                        for token in tokens:
                            chunk = GenerateResponse(
                                model=request.model,
                                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                response=token,
                                done=False,
                            )
                            yield orjson.dumps(chunk.model_dump()).decode() + "\n"

                    final = GenerateResponse(
                        model=request.model,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        response="",
                        done=True,
                        total_duration=1000000,
                        eval_count=max_new_tokens if finish_reason == "length" else 10,
                        done_reason=finish_reason,
                    )
                    yield orjson.dumps(final.model_dump()).decode() + "\n"
                finally:
                    if engine_slot is not None:
                        await asyncio.to_thread(_close_engine_slot, engine_slot)

            try:
                stream_response = StreamingResponse(
                    stream_generate(),
                    media_type="application/x-ndjson",
                    headers=build_execution_headers(
                        routing,
                        execution_device=execution_device,
                        request_id=request_id,
                        fallback_reason=fallback_reason,
                    ),
                )
            except Exception:
                if engine_slot is not None:
                    await asyncio.to_thread(_close_engine_slot, engine_slot)
                raise
            return stream_response

        # Non-streaming
        if use_real:
            try:
                def run_generation():
                    engine, engine_slot = _open_routed_engine_slot(routing.device)
                    try:
                        text = engine.generate(
                            request.prompt,
                            max_new_tokens=_effective_max_tokens(mapped_options),
                            temperature=mapped_options.get("temperature", 0.8),
                            top_p=_effective_top_p(mapped_options),
                            timeout=float(DEFAULT_INFERENCE_TIMEOUT),
                        )
                        return engine, text, getattr(engine_slot, "fallback_reason", None)
                    finally:
                        _close_engine_slot(engine_slot)

                engine, response_text, slot_fallback_reason = await asyncio.to_thread(run_generation)
                execution_device = _execution_device_from_engine(engine)
                fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
                fallback_reason = slot_fallback_reason or fallback_reason
                _record_routing_execution(routing.device, execution_device, fallback_reason)
                add_execution_headers(
                    response,
                    routing,
                    execution_device=execution_device,
                    fallback_reason=fallback_reason,
                )
            except TimeoutError as e:
                logger.exception("Non-streaming inference timed out", extra={"request_id": request_id})
                raise HTTPException(
                    status_code=504,
                    detail=_ollama_error_detail("Inference timed out", "inference_timeout", request_id),
                ) from e
            except RuntimeError as e:
                logger.exception("Non-streaming inference failed", extra={"request_id": request_id})
                raise HTTPException(
                    status_code=503,
                    detail=_ollama_error_detail("Inference service unavailable", "inference_unavailable", request_id),
                ) from e
            except Exception as e:
                from npu_proxy.inference.engine import DeviceBusyError

                if isinstance(e, DeviceBusyError):
                    raise _device_busy_http_exception(e, request_id) from e
                logger.exception("Unexpected inference error", extra={"request_id": request_id})
                raise HTTPException(
                    status_code=500,
                    detail=_ollama_error_detail("Internal inference error", "inference_failed", request_id),
                ) from e
        else:
            execution_device = _get_execution_device(load_if_needed=False)
            fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
            _record_routing_execution(routing.device, execution_device, fallback_reason)
            add_execution_headers(
                response,
                routing,
                execution_device=execution_device,
                fallback_reason=fallback_reason,
            )
            response_text, finish_reason = _mock_text(
                "Hello! I'm running on Intel NPU via OpenVINO.",
                _effective_max_tokens(mapped_options),
            )

        if use_real:
            finish_reason = determine_finish_reason(
                completion_tokens=count_tokens_best_effort(response_text, model=request.model).count,
                max_new_tokens=_effective_max_tokens(mapped_options),
                native_finish_reason=getattr(locals().get("engine", None), "last_finish_reason", None),
            )

        return GenerateResponse(
            model=request.model,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            response=response_text,
            done=True,
            total_duration=1000000,
            eval_count=len(response_text.split()),
            done_reason=finish_reason,
        )
    except HTTPException as exc:
        if request.stream:
            raise
        return _ollama_error_response(exc, response_headers=dict(response.headers))


def format_chat_prompt(messages: list[ChatMessage], model: str | None = None) -> str:
    """Format chat messages into a prompt string for the model.

    Converts structured chat messages into a simple text format
    that works with base language models.

    Args:
        messages: List of ChatMessage objects.

    Returns:
        Formatted prompt string with role prefixes.

    Example:
        >>> messages = [
        ...     ChatMessage(role="system", content="Be helpful"),
        ...     ChatMessage(role="user", content="Hi")
        ... ]
        >>> format_chat_prompt(messages)
        'System: Be helpful\\nUser: Hi\\nAssistant:'
    """
    return render_chat_prompt(messages, model=model).prompt


@router.post("/chat")
async def chat(request: ChatRequest, response: Response):
    """Chat completion with message history.

    Ollama-compatible endpoint for conversational AI with full message
    history support. Supports both streaming and non-streaming responses.

    Args:
        request: Chat request with model, messages, and options.
        response: FastAPI response for setting headers.

    Returns:
        ChatResponse for non-streaming requests.
        StreamingResponse with newline-delimited JSON for streaming.

    Raises:
        HTTPException: 404 if model not found.
        HTTPException: 503 if inference engine unavailable.
        HTTPException: 504 if inference times out.

    Ollama Compatibility:
        - Matches POST /api/chat from Ollama API
        - Supports system, user, and assistant message roles
        - Supports all Ollama options (temperature, top_p, etc.)
        - Returns done=true on final response chunk
        - Streaming uses newline-delimited JSON (not SSE)

    Headers:
        X-Request-ID: Unique request identifier for tracing.
        X-NPU-Proxy-Device: Device used (npu, cpu, gpu).
        X-NPU-Proxy-Route-Reason: Why device was selected.
        X-NPU-Proxy-Token-Count: Estimated total token count.

    Streaming Response Format:
        Each line is a JSON object with partial message::

            {"model":"tinyllama","message":{"role":"assistant","content":"Hello"},"done":false}
            {"model":"tinyllama","message":{"role":"assistant","content":" there"},"done":false}
            {"model":"tinyllama","message":{"role":"assistant","content":""},"done":true}

    Example:
        >>> # Non-streaming
        >>> response = client.post("/api/chat", json={
        ...     "model": "tinyllama",
        ...     "messages": [
        ...         {"role": "system", "content": "You are helpful."},
        ...         {"role": "user", "content": "Hello!"}
        ...     ],
        ...     "stream": False
        ... })
        >>> print(response.json()["message"]["content"])

        >>> # Streaming
        >>> with client.post("/api/chat", json={
        ...     "model": "tinyllama",
        ...     "messages": [{"role": "user", "content": "Hi"}],
        ...     "stream": True
        ... }, stream=True) as response:
        ...     for line in response.iter_lines():
        ...         chunk = json.loads(line)
        ...         print(chunk["message"]["content"], end="")
    """
    # Generate request ID for tracing
    request_id = _generate_request_id()
    add_request_headers(response, request_id)

    try:
        # Validate model exists
        validate_ollama_model(request.model)

        use_real = os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") == "1"
        prompt_model = request.model if use_real else None
        prompt = format_chat_prompt(request.messages, model=prompt_model)

        # Route based on the rendered prompt that will actually be executed
        routing = get_routing_for_prompt(prompt)
        record_routing_decision(routing.device, routing.reason)

        # Apply parameter pipeline: merge defaults, then map to OpenVINO format
        user_options = request.options or {}
        full_options = merge_with_defaults(user_options)
        mapped_options = map_parameters(full_options)

        if request.stream:
            engine = None
            engine_slot = None
            engine_setup_error: Exception | None = None
            if use_real:
                try:
                    engine, engine_slot = await asyncio.to_thread(
                        lambda: _open_routed_engine_slot(routing.device)
                    )
                    execution_device = _execution_device_from_engine(engine)
                    fallback_reason = _fallback_reason(
                        routed_device=routing.device,
                        execution_device=execution_device,
                        engine_slot=engine_slot,
                    )
                    _record_routing_execution(routing.device, execution_device, fallback_reason)
                except Exception as exc:
                    from npu_proxy.inference.engine import DeviceBusyError

                    if isinstance(exc, DeviceBusyError):
                        return _ollama_error_response(
                            _device_busy_http_exception(exc, request_id),
                            response_headers=dict(response.headers),
                        )
                    engine_setup_error = exc
                    execution_device = routing.device
                    fallback_reason = None
            else:
                try:
                    execution_device = _get_execution_device(load_if_needed=False)
                    fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
                    _record_routing_execution(routing.device, execution_device, fallback_reason)
                except Exception as exc:
                    logger.error(
                        "Failed to resolve streaming execution device; using routing device",
                        extra={"request_id": request_id, "error": str(exc)},
                    )
                    execution_device = routing.device
                    fallback_reason = None

            async def stream_chat():
                try:
                    max_new_tokens = _effective_max_tokens(mapped_options)
                    if use_real:
                        if engine_setup_error is not None:
                            logger.exception(
                                "Streaming chat setup failed",
                                extra={"request_id": request_id},
                                exc_info=engine_setup_error,
                            )
                            yield _build_ollama_stream_error_chunk(
                                request.model,
                                "Inference failed",
                                code="inference_failed",
                                request_id=request_id,
                            )
                            return
                        stream_error_message: str | None = None
                        finish_reason: FinishReason = "stop"

                        def set_finish_reason(reason: FinishReason) -> None:
                            nonlocal finish_reason
                            finish_reason = reason
                        try:
                            async for token in stream_engine_tokens(
                                engine_factory=lambda: engine,
                                prompt=prompt,
                                max_new_tokens=max_new_tokens,
                                temperature=mapped_options.get("temperature", 0.8),
                                request_id=request_id,
                                top_p=_effective_top_p(mapped_options),
                                timeout=float(DEFAULT_INFERENCE_TIMEOUT),
                                finish_reason_callback=set_finish_reason,
                            ):
                                chunk = ChatResponse(
                                    model=request.model,
                                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    message=ChatMessage(role="assistant", content=token),
                                    done=False,
                                )
                                yield orjson.dumps(chunk.model_dump()).decode() + "\n"
                        except asyncio.CancelledError:
                            raise
                        except TimeoutError as exc:
                            stream_error_message = "Inference timed out"
                            logger.exception("Streaming chat inference timed out", extra={"request_id": request_id})
                            stream_error_code = "inference_timeout"
                        except Exception as exc:
                            stream_error_message = "Inference failed"
                            logger.exception("Streaming chat inference failed", extra={"request_id": request_id})
                            stream_error_code = "inference_failed"

                        if stream_error_message is not None:
                            yield _build_ollama_stream_error_chunk(
                                request.model,
                                stream_error_message,
                                code=stream_error_code,
                                request_id=request_id,
                            )
                            return
                    else:
                        tokens, finish_reason = _mock_tokens("Hello! I'm running on Intel NPU.", max_new_tokens)
                        for token in tokens:
                            chunk = ChatResponse(
                                model=request.model,
                                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                message=ChatMessage(role="assistant", content=token),
                                done=False,
                            )
                            yield orjson.dumps(chunk.model_dump()).decode() + "\n"

                    final = ChatResponse(
                        model=request.model,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        message=ChatMessage(role="assistant", content=""),
                        done=True,
                        total_duration=1000000,
                        eval_count=max_new_tokens if finish_reason == "length" else 10,
                        done_reason=finish_reason,
                    )
                    yield orjson.dumps(final.model_dump()).decode() + "\n"
                finally:
                    if engine_slot is not None:
                        await asyncio.to_thread(_close_engine_slot, engine_slot)

            try:
                stream_response = StreamingResponse(
                    stream_chat(),
                    media_type="application/x-ndjson",
                    headers=build_execution_headers(
                        routing,
                        execution_device=execution_device,
                        request_id=request_id,
                        fallback_reason=fallback_reason,
                    ),
                )
            except Exception:
                if engine_slot is not None:
                    await asyncio.to_thread(_close_engine_slot, engine_slot)
                raise
            return stream_response

        # Non-streaming
        if use_real:
            try:
                def run_generation():
                    engine, engine_slot = _open_routed_engine_slot(routing.device)
                    try:
                        text = engine.generate(
                            prompt,
                            max_new_tokens=_effective_max_tokens(mapped_options),
                            temperature=mapped_options.get("temperature", 0.8),
                            top_p=_effective_top_p(mapped_options),
                            timeout=float(DEFAULT_INFERENCE_TIMEOUT),
                        )
                        return engine, text, getattr(engine_slot, "fallback_reason", None)
                    finally:
                        _close_engine_slot(engine_slot)

                engine, response_text, slot_fallback_reason = await asyncio.to_thread(run_generation)
                execution_device = _execution_device_from_engine(engine)
                fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
                fallback_reason = slot_fallback_reason or fallback_reason
                _record_routing_execution(routing.device, execution_device, fallback_reason)
                add_execution_headers(
                    response,
                    routing,
                    execution_device=execution_device,
                    fallback_reason=fallback_reason,
                )
            except TimeoutError as e:
                logger.exception("Non-streaming chat inference timed out", extra={"request_id": request_id})
                raise HTTPException(
                    status_code=504,
                    detail=_ollama_error_detail("Inference timed out", "inference_timeout", request_id),
                ) from e
            except RuntimeError as e:
                logger.exception("Non-streaming chat inference failed", extra={"request_id": request_id})
                raise HTTPException(
                    status_code=503,
                    detail=_ollama_error_detail("Inference service unavailable", "inference_unavailable", request_id),
                ) from e
            except Exception as e:
                from npu_proxy.inference.engine import DeviceBusyError

                if isinstance(e, DeviceBusyError):
                    raise _device_busy_http_exception(e, request_id) from e
                logger.exception("Unexpected chat inference error", extra={"request_id": request_id})
                raise HTTPException(
                    status_code=500,
                    detail=_ollama_error_detail("Internal inference error", "inference_failed", request_id),
                ) from e
        else:
            execution_device = _get_execution_device(load_if_needed=False)
            fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
            _record_routing_execution(routing.device, execution_device, fallback_reason)
            add_execution_headers(
                response,
                routing,
                execution_device=execution_device,
                fallback_reason=fallback_reason,
            )
            response_text, finish_reason = _mock_text(
                "Hello! I'm running on Intel NPU via OpenVINO.",
                _effective_max_tokens(mapped_options),
            )

        if use_real:
            finish_reason = determine_finish_reason(
                completion_tokens=count_tokens_best_effort(response_text, model=request.model).count,
                max_new_tokens=_effective_max_tokens(mapped_options),
                native_finish_reason=getattr(locals().get("engine", None), "last_finish_reason", None),
            )

        return ChatResponse(
            model=request.model,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            message=ChatMessage(role="assistant", content=response_text),
            done=True,
            total_duration=1000000,
            eval_count=len(response_text.split()),
            done_reason=finish_reason,
        )
    except HTTPException as exc:
        if request.stream:
            raise
        return _ollama_error_response(exc, response_headers=dict(response.headers))


# =============================================================================
# Phase 2.5: Model Management Endpoints
# =============================================================================

# =============================================================================
# Phase 3.5: Ollama Native Embedding Endpoints
# =============================================================================

class OllamaEmbedRequest(BaseModel):
    """Request body for /api/embed endpoint (Ollama current format).

    The modern Ollama embedding format that supports batch inputs
    and additional options.

    Attributes:
        model: Name of the embedding model to use.
        input: Single string or list of strings to embed.
        truncate: If true, truncate inputs to max context length.
        options: Additional model options.
        keep_alive: How long to keep model loaded (e.g., "5m").
        dimensions: Optional output dimension (for models that support it).

    Example:
        >>> request = OllamaEmbedRequest(
        ...     model="all-minilm",
        ...     input=["Hello world", "Goodbye world"],
        ...     truncate=True
        ... )
    """
    model: str
    input: str | list[str]
    truncate: bool = True
    options: dict | None = None
    keep_alive: str = "5m"
    dimensions: int | None = None


class OllamaEmbedResponse(BaseModel):
    """Response body for /api/embed endpoint (Ollama current format).

    Returns embeddings as a nested list, one embedding per input.

    Attributes:
        model: Model name used for embedding.
        embeddings: List of embedding vectors (list of floats each).
        total_duration: Total processing time in nanoseconds.
        load_duration: Model loading time in nanoseconds.
        prompt_eval_count: Total tokens processed across all inputs.
    """
    model: str
    embeddings: list[list[float]]
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0


class OllamaEmbeddingsLegacyRequest(BaseModel):
    """Request body for /api/embeddings endpoint (Ollama legacy format).

    The legacy single-input embedding format for backward compatibility.

    Attributes:
        model: Name of the embedding model to use.
        prompt: Single string to embed.
        options: Additional model options.
        keep_alive: How long to keep model loaded.

    Example:
        >>> request = OllamaEmbeddingsLegacyRequest(
        ...     model="all-minilm",
        ...     prompt="Hello world"
        ... )
    """
    model: str
    prompt: str
    options: dict | None = None
    keep_alive: str = "5m"


class OllamaEmbeddingsLegacyResponse(BaseModel):
    """Response body for /api/embeddings endpoint (Ollama legacy format).

    Returns a single flat embedding vector.

    Attributes:
        embedding: Embedding vector as list of floats.
    """
    embedding: list[float]


@router.post("/embed", response_model=OllamaEmbedResponse)
async def ollama_embed(request: OllamaEmbedRequest, response: Response):
    """Generate embeddings for text inputs.

    Ollama-compatible endpoint using the current /api/embed format.
    Supports single or batch text inputs.

    Args:
        request: Embed request with model, input(s), and options.

    Returns:
        OllamaEmbedResponse: Embeddings as nested list of vectors.

    Raises:
        HTTPException: 500 if embedding engine initialization fails.
        HTTPException: 504 if embedding times out.

    Ollama Compatibility:
        - Matches POST /api/embed from Ollama API
        - Supports single string or list of strings as input
        - Returns embeddings as list of lists (one per input)
        - Includes timing statistics

    Example:
        >>> # Single input
        >>> response = client.post("/api/embed", json={
        ...     "model": "all-minilm",
        ...     "input": "Hello world"
        ... })
        >>> embedding = response.json()["embeddings"][0]

        >>> # Batch input
        >>> response = client.post("/api/embed", json={
        ...     "model": "all-minilm",
        ...     "input": ["Hello", "World", "Test"]
        ... })
        >>> embeddings = response.json()["embeddings"]  # 3 vectors
    """
    from npu_proxy.inference.embedding_engine import (
        EmbeddingInferenceError,
        EmbeddingTimeoutError,
        EmbeddingUnavailableError,
        get_embedding_engine,
    )
    from npu_proxy.inference.embedding_config import InvalidEmbeddingModelError
    from npu_proxy.inference.tokenizer import count_tokens

    request_id = _generate_request_id()
    add_request_headers(response, request_id)

    try:
        start = time.perf_counter_ns()
        texts = [request.input] if isinstance(request.input, str) else request.input
        _validate_ollama_embedding_inputs(texts, request_id)
        try:
            requested_device = _get_embedding_device_from_options(request.options)
            engine = get_embedding_engine(model_name=request.model, device=requested_device)
        except InvalidEmbeddingModelError as e:
            logger.warning(
                "Rejected Ollama embed model identifier",
                extra={"request_id": request_id, "model": request.model, "error": str(e)},
            )
            raise HTTPException(
                status_code=400,
                detail=_ollama_error_detail("Invalid embedding model", "invalid_embedding_model", request_id),
            )
        except EmbeddingUnavailableError as e:
            logger.warning(
                "Ollama embed unavailable",
                extra={"request_id": request_id, "model": request.model, "error": str(e)},
            )
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to get embedding engine", extra={"request_id": request_id})
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
            )

        engine_info = engine.get_engine_info()
        if request.dimensions is not None and request.dimensions != engine_info.get("dimensions"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested dimensions {request.dimensions} are not supported by "
                    f"{engine_info.get('resolved_model', request.model)} "
                    f"(expected {engine_info.get('dimensions')})"
                ),
            )
        _apply_embedding_engine_headers(response, engine_info)
        if engine_info.get("is_fallback") and not engine_info.get("fallback_allowed", False):
            reason = embedding_failure_reason(
                engine_info,
                default="Embedding fallback is disabled by default",
            )
            logger.error(
                "Embedding fallback rejected",
                extra={"request_id": request_id, "error": reason},
            )
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
                headers=dict(response.headers),
            )

        # Generate embeddings with error handling
        try:
            embeddings = await asyncio.to_thread(engine.embed_batch, texts)
            _validate_embedding_batch_result_count(texts, embeddings)
            _raise_for_runtime_embedding_fallback(engine, response, request_id=request_id)
        except EmbeddingTimeoutError as e:
            logger.exception("Embedding timeout", extra={"request_id": request_id})
            raise HTTPException(
                status_code=504,
                detail=_ollama_error_detail("Embedding timed out", "embedding_timeout", request_id),
            )
        except EmbeddingUnavailableError as e:
            logger.warning(
                "Ollama embed unavailable during inference",
                extra={"request_id": request_id, "model": request.model, "error": str(e)},
            )
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
            )
        except EmbeddingInferenceError as e:
            logger.exception("Embedding failed", extra={"request_id": request_id})
            raise HTTPException(
                status_code=500,
                detail=_ollama_error_detail("Embedding failed", "embedding_failed", request_id),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Embedding failed", extra={"request_id": request_id})
            raise HTTPException(
                status_code=500,
                detail=_ollama_error_detail("Embedding failed", "embedding_failed", request_id),
            )

        # Count tokens
        total_tokens = sum(count_tokens(text) for text in texts)

        total_duration = time.perf_counter_ns() - start

        return OllamaEmbedResponse(
            model=request.model,
            embeddings=embeddings,
            total_duration=total_duration,
            load_duration=0,
            prompt_eval_count=total_tokens,
        )
    except HTTPException as exc:
        return _ollama_error_response(exc, response_headers=dict(response.headers))


@router.post("/embeddings", response_model=OllamaEmbeddingsLegacyResponse)
async def ollama_embeddings_legacy(request: OllamaEmbeddingsLegacyRequest, response: Response):
    """Generate embedding for a single text input (legacy format).

    Ollama-compatible endpoint using the legacy /api/embeddings format.
    Accepts a single prompt and returns a flat embedding vector.

    Args:
        request: Embeddings request with model and prompt.

    Returns:
        OllamaEmbeddingsLegacyResponse: Single embedding as flat list.

    Raises:
        HTTPException: 500 if embedding engine initialization fails.
        HTTPException: 504 if embedding times out.

    Ollama Compatibility:
        - Matches POST /api/embeddings from Ollama API (legacy)
        - Single input only (use /api/embed for batch)
        - Returns flat embedding array (not nested)

    Note:
        This endpoint is deprecated in Ollama. Use /api/embed instead
        for new integrations.

    Example:
        >>> response = client.post("/api/embeddings", json={
        ...     "model": "all-minilm",
        ...     "prompt": "Hello world"
        ... })
        >>> embedding = response.json()["embedding"]  # Flat vector
    """
    from npu_proxy.inference.embedding_engine import (
        EmbeddingInferenceError,
        EmbeddingTimeoutError,
        EmbeddingUnavailableError,
        get_embedding_engine,
    )
    from npu_proxy.inference.embedding_config import InvalidEmbeddingModelError

    request_id = _generate_request_id()
    add_request_headers(response, request_id)

    try:
        _validate_ollama_embedding_inputs([request.prompt], request_id)
        try:
            requested_device = _get_embedding_device_from_options(request.options)
            engine = get_embedding_engine(model_name=request.model, device=requested_device)
        except InvalidEmbeddingModelError as e:
            logger.warning(
                "Rejected Ollama legacy embedding model identifier",
                extra={"request_id": request_id, "model": request.model, "error": str(e)},
            )
            raise HTTPException(
                status_code=400,
                detail=_ollama_error_detail("Invalid embedding model", "invalid_embedding_model", request_id),
            )
        except EmbeddingUnavailableError as e:
            logger.warning(
                "Ollama legacy embeddings unavailable",
                extra={"request_id": request_id, "model": request.model, "error": str(e)},
            )
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to get embedding engine", extra={"request_id": request_id})
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
            )

        engine_info = engine.get_engine_info()
        _apply_embedding_engine_headers(response, engine_info)
        if engine_info.get("is_fallback") and not engine_info.get("fallback_allowed", False):
            reason = embedding_failure_reason(
                engine_info,
                default="Embedding fallback is disabled by default",
            )
            logger.error(
                "Embedding fallback rejected",
                extra={"request_id": request_id, "error": reason},
            )
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
                headers=dict(response.headers),
            )

        # Generate single embedding with error handling
        try:
            embedding = await asyncio.to_thread(engine.embed, request.prompt)
            _raise_for_runtime_embedding_fallback(engine, response, request_id=request_id)
        except EmbeddingTimeoutError as e:
            logger.exception("Embedding timeout", extra={"request_id": request_id})
            raise HTTPException(
                status_code=504,
                detail=_ollama_error_detail("Embedding timed out", "embedding_timeout", request_id),
            )
        except EmbeddingUnavailableError as e:
            logger.warning(
                "Ollama legacy embeddings unavailable during inference",
                extra={"request_id": request_id, "model": request.model, "error": str(e)},
            )
            raise HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Embedding engine unavailable", "embedding_unavailable", request_id),
            )
        except EmbeddingInferenceError as e:
            logger.exception("Embedding failed", extra={"request_id": request_id})
            raise HTTPException(
                status_code=500,
                detail=_ollama_error_detail("Embedding failed", "embedding_failed", request_id),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Embedding failed", extra={"request_id": request_id})
            raise HTTPException(
                status_code=500,
                detail=_ollama_error_detail("Embedding failed", "embedding_failed", request_id),
            )

        return OllamaEmbeddingsLegacyResponse(
            embedding=embedding,
        )
    except HTTPException as exc:
        return _ollama_error_response(exc, response_headers=dict(response.headers))


# =============================================================================
# Phase 2.5: Model Management Endpoints
# =============================================================================

class PullRequest(BaseModel):
    """Request body for /api/pull endpoint.

    Attributes:
        name: Model name (Ollama-style short name or HuggingFace repo).
        stream: If true, stream download progress updates. Default: True.

    Example:
        >>> request = PullRequest(name="tinyllama", stream=True)
    """
    name: str = Field(..., description="Model name (Ollama-style or HuggingFace repo)")
    stream: bool = Field(default=True, description="Stream progress updates")
    huggingface_token: SecretStr | None = Field(
        default=None,
        description="Optional Hugging Face token for private repos",
        json_schema_extra={"writeOnly": True},
        repr=False,
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Parse an optional Bearer token from the Authorization header."""
    if authorization is None:
        return None

    scheme, _, value = authorization.partition(" ")
    token = value.strip()
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=400,
            detail="Authorization header must use Bearer <token> format.",
        )
    return token


def _resolve_pull_token(request: PullRequest, authorization: str | None) -> str | bool:
    """Resolve explicit pull credentials from body or header without ambiguity."""
    body_token = None
    if request.huggingface_token is not None:
        candidate = request.huggingface_token.get_secret_value().strip()
        body_token = candidate or None

    header_token = _extract_bearer_token(authorization)
    if body_token and header_token and body_token != header_token:
        raise HTTPException(
            status_code=400,
            detail="Provide the Hugging Face token in either the body or Authorization header, not both.",
        )

    return body_token or header_token or False


class KnownModelInfo(BaseModel):
    """Information about a pre-mapped model.

    Attributes:
        ollama_name: Short Ollama-style name (e.g., "tinyllama").
        huggingface_repo: Full HuggingFace repository path.
        quantization: Quantization format (INT4, INT8, FP16).
    """
    ollama_name: str
    huggingface_repo: str
    quantization: str


class KnownModelsResponse(BaseModel):
    """Response for /api/models/known endpoint.

    Attributes:
        models: List of pre-mapped model information.
    """
    models: list[KnownModelInfo]


class SearchResultModel(BaseModel):
    """A model in search results.

    Attributes:
        id: HuggingFace repository ID.
        name: Model display name.
        author: Model author/organization.
        downloads: Total download count.
        likes: Total like count.
        last_modified: ISO 8601 timestamp of last update.
        quantization: Quantization format.
        parameters: Parameter count string (e.g., "7B").
        architecture: Model architecture.
        pull_command: Command to pull this model.
    """
    id: str
    name: str
    author: str
    downloads: int
    likes: int
    last_modified: str
    quantization: str
    parameters: str
    architecture: str
    pull_command: str


class SearchResponse(BaseModel):
    """Response for /api/search endpoint.

    Attributes:
        models: List of matching models.
        total: Total number of matching models.
        offset: Current pagination offset.
        limit: Results per page.
        has_more: True if more results available.
    """
    models: list[SearchResultModel]
    total: int
    offset: int
    limit: int
    has_more: bool


def _sanitize_pull_progress(progress: dict, request_id: str) -> dict:
    """Return safe NDJSON progress frames for pull streams."""
    status = progress.get("status")
    if isinstance(status, str) and status.lower().startswith(("error", "failed")):
        logger.error(
            "Model pull progress error",
            extra={"request_id": request_id, "error": status},
        )
        return {
            "status": "error",
            "error": _safe_error_message("Model pull failed", request_id),
            "code": "model_pull_failed",
        }
    return progress


@router.post("/pull")
async def pull_model(
    request: PullRequest,
    response: Response,
    authorization: str | None = Header(default=None),
):
    """Download a model from HuggingFace.

    Ollama-compatible endpoint that maps model names to HuggingFace
    OpenVINO repositories and downloads them automatically.

    Args:
        request: Pull request with model name and streaming preference.

    Returns:
        JSONResponse with status for non-streaming.
        StreamingResponse with progress updates for streaming.

    Raises:
        HTTPException: 400 if model name is empty.
        HTTPException: 404 if model not found or unmapped.
        HTTPException: 500 if download fails.
        HTTPException: 503 if model management unavailable.

    Ollama Compatibility:
        - Matches POST /api/pull from Ollama API
        - Supports streaming progress updates
        - Returns same status messages as Ollama

    Streaming Response Format:
        Each line is a JSON object with download progress::

            {"status":"pulling manifest"}
            {"status":"downloading","digest":"abc123","total":1000000,"completed":500000}
            {"status":"success"}

    Example:
        >>> # Non-streaming
        >>> response = client.post("/api/pull", json={
        ...     "name": "tinyllama",
        ...     "stream": False
        ... })
        >>> print(response.json()["status"])

        >>> # Streaming progress
        >>> with client.post("/api/pull", json={
        ...     "name": "tinyllama",
        ...     "stream": True
        ... }, stream=True) as response:
        ...     for line in response.iter_lines():
        ...         progress = json.loads(line)
        ...         print(f"{progress['status']}: {progress.get('completed', 0)}")
    """
    request_id = _generate_request_id()
    add_request_headers(response, request_id)

    try:
        from npu_proxy.models.downloader import (
            DEFAULT_MODEL_DIR,
            download_model,
            is_model_downloaded,
            get_download_progress,
            resolve_download_target,
        )
    except ImportError as e:
        logger.exception("Model management import failed", extra={"request_id": request_id})
        return _ollama_error_response(
            HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Model management unavailable", "model_management_unavailable", request_id),
            ),
            response_headers=dict(response.headers),
        )

    model_name = request.name.strip()
    if not model_name:
        return _ollama_error_response(
            HTTPException(
                status_code=400,
                detail=_ollama_error_detail("Model name is required", "invalid_model_name", request_id),
            ),
            response_headers=dict(response.headers),
        )

    try:
        huggingface_token = _resolve_pull_token(request, authorization)
    except HTTPException as exc:
        logger.warning(
            "Ollama pull authentication validation failed",
            extra={"request_id": request_id, "error": _error_message(exc.detail)},
        )
        return _ollama_error_response(
            HTTPException(
                status_code=exc.status_code,
                detail=_ollama_error_detail("Invalid pull authentication", "invalid_pull_auth", request_id),
            ),
            response_headers=dict(response.headers),
        )

    logger.info(
        "Pull request for model: %s (auth=%s)",
        model_name,
        "explicit" if huggingface_token is not False else "anonymous",
    )

    # Resolve model name to HuggingFace repo and canonical cache target
    download_target = resolve_download_target(model_name, DEFAULT_MODEL_DIR)
    if download_target is None:
        return _ollama_error_response(
            HTTPException(
                status_code=404,
                detail=_ollama_error_detail("Model not found", "model_not_found", request_id),
            ),
            response_headers=dict(response.headers),
        )
    repo_id, runtime_model_name, local_dir, required_files = download_target

    # Check if already downloaded
    if is_model_downloaded(model_name, token=huggingface_token):
        already_downloaded = {
            "status": "success",
            "model": runtime_model_name,
            "message": "Model already downloaded",
        }
        if request.stream:
            async def already_downloaded_progress():
                yield json.dumps({"status": "pulling manifest"}) + "\n"
                yield json.dumps(already_downloaded) + "\n"

            return StreamingResponse(
                already_downloaded_progress(),
                media_type="application/x-ndjson",
                headers=dict(response.headers),
            )
        return JSONResponse(already_downloaded, headers=dict(response.headers))

    if request.stream:
        async def generate_progress():
            iterator = get_download_progress(
                repo_id,
                local_dir,
                token=huggingface_token,
                resolved_name=runtime_model_name,
                required_files=required_files,
            )
            while True:
                has_item, progress = await asyncio.to_thread(_next_sync_iterator_item, iterator)
                if not has_item:
                    break
                yield json.dumps(_sanitize_pull_progress(progress, request_id)) + "\n"

        return StreamingResponse(
            generate_progress(),
            media_type="application/x-ndjson",
            headers=dict(response.headers),
        )
    else:
        result = await asyncio.to_thread(download_model, model_name, token=huggingface_token)
        if "error" in result:
            logger.error(
                "Model pull failed",
                extra={"request_id": request_id, "error": result["error"]},
            )
            return _ollama_error_response(
                HTTPException(
                    status_code=int(result.get("status_code", "500")),
                    detail=_ollama_error_detail("Model pull failed", "model_pull_failed", request_id),
                ),
                response_headers=dict(response.headers),
            )
        return JSONResponse(result, headers=dict(response.headers))


@router.get("/search")
async def search_models(
    response: Response,
    q: str = Query(default="", description="Search query"),
    sort: str = Query(default="popular", description="Sort: popular, newest, downloads, likes"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    model_type: str = Query(default="all", alias="type", description="Filter: all, llm, embedding, vision"),
    quantization: str = Query(default="", description="Filter: int4, int8, fp16"),
    min_downloads: int = Query(default=0, ge=0, description="Minimum download count"),
):
    """Search HuggingFace for OpenVINO-compatible models.

    Search and filter models available for download from HuggingFace.
    Only returns models verified to be OpenVINO-compatible.

    Args:
        q: Search query string.
        sort: Sort order (popular, newest, downloads, likes).
        limit: Maximum results per page (1-100).
        offset: Pagination offset.
        model_type: Filter by model type (all, llm, embedding, vision).
        quantization: Filter by quantization format (int4, int8, fp16).
        min_downloads: Minimum download count filter.

    Returns:
        SearchResponse: Paginated list of matching models.

    Raises:
        HTTPException: 400 if invalid sort or type parameter.
        HTTPException: 503 if search service unavailable.

    Ollama Compatibility:
        - Extended endpoint (not in standard Ollama API)
        - Provides HuggingFace model discovery
        - Returns pull_command for easy download

    Example:
        >>> # Search for LLMs
        >>> response = client.get("/api/search", params={
        ...     "q": "llama",
        ...     "type": "llm",
        ...     "sort": "downloads"
        ... })
        >>> for model in response.json()["models"]:
        ...     print(f"{model['name']}: {model['pull_command']}")

        >>> # Find INT4 quantized models
        >>> response = client.get("/api/search", params={
        ...     "quantization": "int4",
        ...     "min_downloads": 1000
        ... })
    """
    request_id = _generate_request_id()
    add_request_headers(response, request_id)

    try:
        from npu_proxy.models.search import search_openvino_models
        from npu_proxy.models.mapper import get_ollama_name
    except ImportError as e:
        logger.exception("Model search import failed", extra={"request_id": request_id})
        return _ollama_error_response(
            HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Model search unavailable", "model_search_unavailable", request_id),
            ),
            response_headers=dict(response.headers),
        )

    # Validate sort parameter
    valid_sorts = ['popular', 'newest', 'downloads', 'likes']
    if sort not in valid_sorts:
        return _ollama_error_response(
            HTTPException(
                status_code=400,
                detail=_ollama_error_detail("Invalid sort value", "invalid_sort", request_id),
            ),
            response_headers=dict(response.headers),
        )

    # Validate type parameter
    valid_types = ['all', 'llm', 'embedding', 'vision']
    if model_type not in valid_types:
        return _ollama_error_response(
            HTTPException(
                status_code=400,
                detail=_ollama_error_detail("Invalid model type", "invalid_model_type", request_id),
            ),
            response_headers=dict(response.headers),
        )

    logger.info(f"Search request: q='{q}', sort={sort}, type={model_type}")

    try:
        results, total = await asyncio.to_thread(
            search_openvino_models,
            query=q,
            sort=sort,
            limit=limit,
            offset=offset,
            model_type=model_type,
            quantization=quantization,
            min_downloads=min_downloads,
        )

        # Convert SearchResult objects to response format
        models = []
        for r in results:
            ollama_name = get_ollama_name(r.id)
            models.append(SearchResultModel(
                id=r.id,
                name=r.name,
                author=r.author,
                downloads=r.downloads,
                likes=r.likes,
                last_modified=r.last_modified,
                quantization=r.quantization,
                parameters=r.parameters,
                architecture=r.architecture,
                pull_command=f"ollama pull {ollama_name}" if ollama_name else f"ollama pull {r.id}"
            ))

        return SearchResponse(
            models=models,
            total=total,
            offset=offset,
            limit=limit,
            has_more=offset + limit < total
        )

    except Exception as e:
        logger.exception("Search error", extra={"request_id": request_id})
        return _ollama_error_response(
            HTTPException(
                status_code=503,
                detail=_ollama_error_detail("Model search temporarily unavailable", "model_search_unavailable", request_id),
            ),
            response_headers=dict(response.headers),
        )


@router.get("/models/known")
async def list_known_models_endpoint():
    """List all pre-mapped Ollama model names.

    Returns models that have verified HuggingFace OpenVINO repositories
    and can be pulled using their short Ollama-style names.

    Returns:
        KnownModelsResponse: List of pre-mapped model information.

    Raises:
        HTTPException: 503 if model mapping unavailable.

    Ollama Compatibility:
        - Extended endpoint (not in standard Ollama API)
        - Provides discovery of NPU Proxy model mappings

    Example:
        >>> response = client.get("/api/models/known")
        >>> for model in response.json()["models"]:
        ...     print(f"{model['ollama_name']} -> {model['huggingface_repo']}")
        tinyllama -> OpenVINO/TinyLlama-1.1B-Chat-int4-ov
    """
    try:
        from npu_proxy.models.mapper import list_known_models
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Model mapping not available: {e}")

    models = list_known_models()
    return KnownModelsResponse(
        models=[KnownModelInfo(**m) for m in models]
    )
