"""Chat completions endpoint handlers.

This module implements the OpenAI-compatible Chat Completions API endpoint,
providing a drop-in replacement for the OpenAI API that routes requests to
Intel NPU hardware via OpenVINO.

OpenAI API Compatibility:
    This implementation follows the OpenAI Chat Completions API specification:
    https://platform.openai.com/docs/api-reference/chat/create

    Supported features:
        - POST /v1/chat/completions
        - Streaming (SSE) and non-streaming responses
        - System, user, and assistant message roles
        - temperature, max_tokens, top_p parameters
        - presence_penalty, frequency_penalty parameters
        - OpenAI-compatible error responses
        - Request ID tracking via X-Request-ID header

    Response format matches OpenAI's schema including:
        - chat.completion objects for non-streaming
        - chat.completion.chunk objects for streaming
        - Usage statistics (prompt_tokens, completion_tokens, total_tokens)

Example:
    >>> import httpx
    >>> response = httpx.post(
    ...     "http://localhost:8080/v1/chat/completions",
    ...     json={
    ...         "model": "llama-3.2-1b",
    ...         "messages": [{"role": "user", "content": "Hello!"}],
    ...         "max_tokens": 100
    ...     }
    ... )
    >>> response.json()["choices"][0]["message"]["content"]
    "Hello! How can I help you today?"
"""
import time
import uuid
import asyncio
import logging
import os
from typing import AsyncIterator, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse, Response, StreamingResponse
import orjson

from npu_proxy.api.header_utils import (
    build_openai_stream_error_chunk as shared_build_openai_stream_error_chunk,
    build_single_engine_execution_headers,
    openai_error_response as shared_openai_error_response,
    openai_error_response_from_http_exception as shared_openai_error_response_from_http_exception,
    render_api_chat_prompt,
    add_single_engine_execution_headers,
    generate_request_id as shared_generate_request_id,
    validate_registered_model,
)
from npu_proxy.inference.streaming import FinishReason, determine_finish_reason, stream_engine_tokens
from npu_proxy.inference.tokenizer import count_tokens, count_tokens_best_effort
from npu_proxy.models.ollama_defaults import merge_with_defaults
from npu_proxy.models.parameter_mapper import map_parameters
from npu_proxy.routing.context_router import get_context_router, RoutingResult
from npu_proxy.config import (
    DEFAULT_INFERENCE_TIMEOUT,
    load_device_queue_timeout,
    load_fallback_on_busy,
)
from npu_proxy.metrics import record_routing_decision, record_routing_execution, record_tokens

router = APIRouter(prefix="/v1", tags=["chat"])
logger = logging.getLogger(__name__)

def generate_request_id() -> str:
    """Generate a unique request ID for tracking.

    Creates a request ID in the format used by OpenAI's API for request
    tracking and debugging purposes.

    Returns:
        A unique request ID string in the format "req_<24-char-hex>".

    Example:
        >>> request_id = generate_request_id()
        >>> request_id.startswith("req_")
        True
        >>> len(request_id)
        28
    """
    return shared_generate_request_id(prefix="req_", hex_chars=24)


def create_error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """Create an OpenAI-compatible error response.

    Generates a JSON error response that matches the OpenAI API error format,
    ensuring clients expecting OpenAI responses can properly handle errors.

    Args:
        status_code: HTTP status code for the response (e.g., 400, 404, 500).
        message: Human-readable error description.
        error_type: Error category. Common values include:
            - "invalid_request_error": Client request validation failures
            - "authentication_error": API key issues
            - "rate_limit_error": Rate limiting triggered
            - "server_error": Internal server errors
        param: The request parameter that caused the error, if applicable.
        code: Machine-readable error code (e.g., "model_not_found").
        request_id: Optional request ID to include in response headers.

    Returns:
        JSONResponse with OpenAI-compatible error body and appropriate headers.

    Example:
        >>> response = create_error_response(
        ...     status_code=404,
        ...     message="Model 'unknown-model' not found",
        ...     error_type="invalid_request_error",
        ...     param="model",
        ...     code="model_not_found"
        ... )
        >>> response.status_code
        404

    OpenAI API Reference:
        https://platform.openai.com/docs/guides/error-codes
    """
    return shared_openai_error_response(
        status_code=status_code,
        message=message,
        error_type=error_type,
        param=param,
        code=code,
        request_id=request_id,
    )


def create_error_response_from_http_exception(
    exc: HTTPException,
    *,
    request_id: Optional[str] = None,
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    """Convert a handled HTTPException into an OpenAI-compatible error body."""
    return shared_openai_error_response_from_http_exception(
        exc,
        request_id=request_id,
        param=param,
        code=code,
    )


def get_engine(device: str | None = None):
    """Get an inference engine from the shared per-device engine pool."""
    from npu_proxy.inference.engine import get_llm_engine

    return get_llm_engine(device=device)


def _open_routed_engine_slot(device: str):
    """Acquire the routed device slot and return its engine plus release handle."""
    from npu_proxy.inference.engine import DeviceBusyError, acquire_device_slot, fallback_devices_after

    timeout = load_device_queue_timeout()
    fallback_on_busy = load_fallback_on_busy()
    normalized = str(device).strip().upper()
    candidates = [normalized]
    if fallback_on_busy:
        candidates.extend(fallback_devices_after(normalized))

    last_busy: DeviceBusyError | None = None
    for candidate in candidates:
        slot = acquire_device_slot(candidate, timeout=timeout)
        try:
            selected_device = slot.__enter__()
        except DeviceBusyError as exc:
            last_busy = exc
            if not fallback_on_busy:
                raise
            continue
        try:
            engine = get_engine(device=selected_device)
        except Exception:
            slot.__exit__(None, None, None)
            raise
        setattr(slot, "routed_device", normalized)
        setattr(slot, "selected_device", selected_device)
        setattr(slot, "fallback_reason", "busy" if selected_device != normalized else None)
        return engine, slot

    if last_busy is not None:
        raise last_busy
    raise DeviceBusyError(normalized)


def _close_engine_slot(slot: object) -> None:
    exit_method = getattr(slot, "__exit__")
    exit_method(None, None, None)


def _fallback_reason(
    *,
    routed_device: str,
    execution_device: str,
    engine_slot: object | None = None,
) -> str | None:
    routed = str(routed_device).strip().upper()
    executed = str(execution_device).strip().upper()
    if not routed or executed == routed:
        return None
    reason = getattr(engine_slot, "fallback_reason", None)
    return str(reason) if reason else "device_fallback"


def _record_routing_execution(routed_device: str, execution_device: str, fallback_reason: str | None) -> None:
    record_routing_execution(routed_device, execution_device, fallback_reason or "none")


def _execution_device_from_engine(engine: Any) -> str:
    """Return the actual device reported by an acquired engine."""
    get_device_info = getattr(engine, "get_device_info", None)
    if callable(get_device_info):
        try:
            info = get_device_info()
        except Exception:
            info = {}
        if isinstance(info, dict) and info.get("actual_device"):
            return str(info["actual_device"])
    return str(getattr(engine, "actual_device", None) or "unknown")


def validate_model_exists(model: str) -> None:
    """Validate that a model exists in the registry.

    Checks the model registry to ensure the requested model is available
    before attempting inference.

    Args:
        model: The model identifier to validate (e.g., "llama-3.2-1b").

    Raises:
        HTTPException: 404 error if the model is not found in the registry.

    Example:
        >>> validate_model_exists("llama-3.2-1b")  # OK if model exists
        >>> validate_model_exists("unknown-model")
        HTTPException: 404: Model 'unknown-model' not found
    """
    validate_registered_model(model)


def add_execution_headers(
    response: Response,
    routing_result: RoutingResult,
    execution_device: str,
    request_id: Optional[str] = None,
    fallback_reason: str | None = None,
) -> None:
    """Add truthful single-engine execution headers and request ID.

    Adds custom headers to track which device handled the request,
    why that routing decision was made, and the request identifier.

    Args:
        response: The FastAPI/Starlette Response object to modify.
        routing_result: The routing decision containing device and reason.
        request_id: Optional unique request identifier for tracking.

    Headers Added:
        - X-Request-ID: Unique request identifier for tracing
        - X-NPU-Proxy-Device: The configured or loaded singleton execution device
        - X-NPU-Proxy-Route-Reason: Why per-request routing is not claimed as execution fact
        - X-NPU-Proxy-Token-Count: Estimated token count for the request

    Example:
        >>> add_routing_headers(response, routing_result, "req_abc123")
        >>> response.headers["X-NPU-Proxy-Device"]
        "NPU"
    """
    add_single_engine_execution_headers(
        response,
        routing_result.token_count,
        execution_device=execution_device,
        routed_device=routing_result.device,
        fallback_reason=fallback_reason,
        request_id=request_id,
    )


class Message(BaseModel):
    """A single message in a chat conversation.

    Represents one turn in a multi-turn conversation, following the
    OpenAI message format for chat completions.

    Attributes:
        role: The role of the message author. Must be one of:
            - "system": Instructions that guide the assistant's behavior
            - "user": Messages from the end user
            - "assistant": Previous responses from the assistant
        content: The text content of the message.

    Example:
        >>> message = Message(role="user", content="What is 2+2?")
        >>> message.role
        "user"

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/create#messages
    """
    role: str
    content: str

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate that role is one of the allowed values."""
        if v not in ('system', 'user', 'assistant'):
            raise ValueError(f"Invalid role '{v}'. Must be 'system', 'user', or 'assistant'")
        return v


class ChatRequest(BaseModel):
    """Request body for chat completions endpoint.

    Defines the parameters for a chat completion request, compatible with
    the OpenAI Chat Completions API specification.

    Attributes:
        model: ID of the model to use (e.g., "llama-3.2-1b").
        messages: List of messages in the conversation.
        temperature: Sampling temperature (0.0-2.0). Higher values increase
            randomness. Default: 0.7.
        max_tokens: Maximum tokens to generate (1-4096). Default: 256.
        stream: If True, stream partial responses via SSE. Default: False.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to temperature.
        presence_penalty: Penalty for new topics (-2.0 to 2.0).
        frequency_penalty: Penalty for repetition (-2.0 to 2.0).

    Example:
        >>> request = ChatRequest(
        ...     model="llama-3.2-1b",
        ...     messages=[Message(role="user", content="Hello!")],
        ...     temperature=0.8,
        ...     max_tokens=100
        ... )

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/create
    """
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    @field_validator('messages')
    @classmethod
    def validate_messages_not_empty(cls, v: list[Message]) -> list[Message]:
        """Validate that at least one message is provided."""
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class ChatChoice(BaseModel):
    """A single completion choice in a chat response.

    Represents one possible completion generated by the model. For most
    requests, there will be exactly one choice.

    Attributes:
        index: Index of this choice in the choices array (usually 0).
        message: The generated message containing the assistant's response.
        finish_reason: Why generation stopped:
            - "stop": Natural completion or stop sequence
            - "length": max_tokens limit reached
            - "content_filter": Content was filtered

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/object#choices
    """
    index: int
    message: Message
    finish_reason: str


class ChatUsage(BaseModel):
    """Token usage statistics for a chat completion.

    Provides token counts for billing and monitoring purposes.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated response.
        total_tokens: Sum of prompt_tokens and completion_tokens.

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/object#usage
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Response body for non-streaming chat completions.

    The complete response object returned for non-streaming requests,
    following the OpenAI chat.completion object format.

    Attributes:
        id: Unique identifier for the completion (format: "chatcmpl-<id>").
        object: Object type, always "chat.completion".
        created: Unix timestamp of when the completion was created.
        model: The model used for generation.
        choices: List of completion choices (usually one).
        usage: Token usage statistics.

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/object
    """
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class ChatChunkDelta(BaseModel):
    """Incremental content in a streaming chunk.

    Contains the incremental update for a streaming response. The first
    chunk includes the role, subsequent chunks include content tokens.

    Attributes:
        role: The role of the message (only in first chunk).
        content: Token(s) of generated content (in subsequent chunks).

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/streaming
    """
    role: str | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    """A single choice in a streaming chunk.

    Represents incremental progress for one choice in a streaming response.

    Attributes:
        index: Index of this choice (usually 0).
        delta: Incremental content update.
        finish_reason: Set to "stop" in the final chunk, None otherwise.
    """
    index: int
    delta: ChatChunkDelta
    finish_reason: str | None = None


class ChatChunk(BaseModel):
    """A streaming chunk for chat completions.

    Sent as a Server-Sent Event (SSE) during streaming responses,
    following the OpenAI chat.completion.chunk format.

    Attributes:
        id: Unique identifier (same across all chunks in a response).
        object: Object type, always "chat.completion.chunk".
        created: Unix timestamp of when streaming started.
        model: The model used for generation.
        choices: List of choice deltas (usually one).

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/streaming
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]


def format_chat_prompt(messages: list[Message], model: str | None = None) -> str:
    """Format messages into a prompt string for the model."""

    return render_api_chat_prompt(messages, model=model)


def build_chat_routing_result(prompt: str, model: str | None) -> tuple[RoutingResult, int]:
    """Route chat requests using rendered-prompt accounting."""

    context_router = get_context_router()
    if model is None:
        prompt_tokens = count_tokens(prompt)
    else:
        prompt_tokens = count_tokens_best_effort(prompt, model=model).count

    if prompt_tokens > context_router.npu_limit:
        return (
            RoutingResult(
                device=context_router.fallback_device,
                reason="prompt_exceeds_npu_limit",
                token_count=prompt_tokens,
            ),
            prompt_tokens,
        )

    return (
        RoutingResult(
            device=context_router.preferred_device,
            reason="within_npu_limit",
            token_count=prompt_tokens,
        ),
        prompt_tokens,
    )


def _get_execution_device(*, load_if_needed: bool) -> str:
    """Return the actual singleton execution device for response reporting."""
    from npu_proxy.inference.execution_state import get_reportable_execution_device

    return get_reportable_execution_device(load_if_needed=load_if_needed)


def build_stream_headers(
    routing: RoutingResult,
    request_id: str,
    *,
    execution_device: str,
    fallback_reason: str | None = None,
) -> dict[str, str]:
    """Build streaming response headers for truthful single-engine reporting."""

    return build_single_engine_execution_headers(
        routing.token_count,
        execution_device=execution_device,
        routed_device=routing.device,
        fallback_reason=fallback_reason,
        request_id=request_id,
    )




def _effective_max_tokens(mapped_options: dict | None, fallback: int) -> int:
    value = (mapped_options or {}).get("max_new_tokens", fallback)
    try:
        token_limit = int(value)
    except (TypeError, ValueError):
        return fallback
    return token_limit if token_limit > 0 else fallback


def _effective_top_p(mapped_options: dict | None, fallback: float | None = None) -> float:
    value = (mapped_options or {}).get("top_p", fallback if fallback is not None else 0.9)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback if fallback is not None else 0.9


def _mock_completion_tokens(text: str, max_tokens: int) -> tuple[list[str], FinishReason]:
    words = text.split()
    if max_tokens > 0 and len(words) > max_tokens:
        words = words[:max_tokens]
        finish_reason: FinishReason = "length"
    else:
        finish_reason = "stop"
    return [word + (" " if i < len(words) - 1 else "") for i, word in enumerate(words)], finish_reason


def _mock_completion_text(text: str, max_tokens: int) -> tuple[str, FinishReason]:
    tokens, finish_reason = _mock_completion_tokens(text, max_tokens)
    return "".join(tokens), finish_reason

def _build_openai_stream_error_chunk(message: str) -> str:
    """Build an OpenAI-compatible SSE error payload for mid-stream failures."""
    return shared_build_openai_stream_error_chunk(message)


async def generate_stream_real(
    request: ChatRequest,
    chat_id: str,
    created: int,
    prompt: str,
    mapped_options: dict | None = None,
    request_id: str | None = None,
    prompt_tokens: int = 0,
    engine: Any | None = None,
    engine_slot: object | None = None,
) -> AsyncIterator[str]:
    """Generate streaming response with real-time token streaming.

    Uses AsyncTokenStream to yield tokens as they are generated by the
    inference engine, providing true real-time SSE streaming to clients.
    """
    if mapped_options is None:
        mapped_options = {}
    max_new_tokens = _effective_max_tokens(mapped_options, request.max_tokens)
    temperature = mapped_options.get("temperature", request.temperature)
    top_p = _effective_top_p(mapped_options, request.top_p)
    emitted_tokens: list[str] = []
    finish_reason: FinishReason = "stop"

    def set_finish_reason(reason: FinishReason) -> None:
        nonlocal finish_reason
        finish_reason = reason

    try:
        initial_chunk = ChatChunk(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                ChatChunkChoice(
                    index=0,
                    delta=ChatChunkDelta(role="assistant"),
                )
            ],
        )
        yield f"data: {orjson.dumps(initial_chunk.model_dump()).decode()}\n\n"

        stream_error_message: str | None = None
        try:
            if engine is None:
                engine = get_engine()
            async for token in stream_engine_tokens(
                engine_factory=lambda: engine,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                request_id=request_id,
                top_p=top_p,
                timeout=float(DEFAULT_INFERENCE_TIMEOUT),
                finish_reason_callback=set_finish_reason,
            ):
                emitted_tokens.append(token)
                chunk = ChatChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatChunkChoice(
                            index=0,
                            delta=ChatChunkDelta(content=token),
                        )
                    ],
                )
                yield f"data: {orjson.dumps(chunk.model_dump()).decode()}\n\n"
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            stream_error_message = "Inference timed out"
            logger.exception("Streaming inference timed out", extra={"request_id": request_id})
        except Exception:
            stream_error_message = "Streaming inference failed"
            logger.exception("Streaming inference failed", extra={"request_id": request_id})

        if stream_error_message is not None:
            yield _build_openai_stream_error_chunk(stream_error_message)
            return

        final_chunk = ChatChunk(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                ChatChunkChoice(
                    index=0,
                    delta=ChatChunkDelta(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {orjson.dumps(final_chunk.model_dump()).decode()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if engine_slot is not None:
            await asyncio.to_thread(_close_engine_slot, engine_slot)
        completion_tokens = count_tokens_best_effort(
            "".join(emitted_tokens), model=request.model
        ).count
        record_tokens(request.model, prompt_tokens, completion_tokens)


async def generate_stream(
    request: ChatRequest,
    chat_id: str,
    created: int,
    prompt_tokens: int = 0,
) -> AsyncIterator[str]:
    """Generate mock streaming response for testing.

    Produces a simulated streaming response without invoking the real
    inference engine. Used for integration tests and development.
    """
    mock_response = "Hello! I'm a helpful AI assistant running on Intel NPU via OpenVINO."
    mock_tokens, finish_reason = _mock_completion_tokens(mock_response, request.max_tokens)
    emitted_tokens: list[str] = []
    try:
        initial_chunk = ChatChunk(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                ChatChunkChoice(
                    index=0,
                    delta=ChatChunkDelta(role="assistant"),
                )
            ],
        )
        yield f"data: {orjson.dumps(initial_chunk.model_dump()).decode()}\n\n"

        for content in mock_tokens:
            emitted_tokens.append(content)
            chunk = ChatChunk(
                id=chat_id,
                created=created,
                model=request.model,
                choices=[
                    ChatChunkChoice(
                        index=0,
                        delta=ChatChunkDelta(content=content),
                    )
                ],
            )
            yield f"data: {orjson.dumps(chunk.model_dump()).decode()}\n\n"

        final_chunk = ChatChunk(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                ChatChunkChoice(
                    index=0,
                    delta=ChatChunkDelta(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {orjson.dumps(final_chunk.model_dump()).decode()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        completion_tokens = count_tokens("".join(emitted_tokens))
        record_tokens(request.model, prompt_tokens, completion_tokens)


@router.post("/chat/completions")
async def chat_completions(request: ChatRequest, response: Response):
    """Create a chat completion (OpenAI-compatible).

    Generates a model response for the given chat conversation. This endpoint
    is designed as a drop-in replacement for the OpenAI Chat Completions API,
    routing requests to Intel NPU hardware via OpenVINO.

    Args:
        request: The chat completion request containing model, messages,
            and generation parameters.
        response: FastAPI Response object for adding custom headers.

    Returns:
        ChatResponse: For non-streaming requests, returns a complete
            chat.completion object with the generated response.
        StreamingResponse: For streaming requests (stream=True), returns
            an SSE stream of chat.completion.chunk objects.

    Raises:
        HTTPException: 404 if model not found, 503 if inference fails,
            504 on timeout, 500 on unexpected errors.

    Response Headers:
        - X-Request-ID: Unique request identifier for tracing
        - X-NPU-Proxy-Device: Device that handled the request (NPU/CPU)
        - X-NPU-Proxy-Route-Reason: Why this device was selected
        - X-NPU-Proxy-Token-Count: Estimated input token count

    Example:
        >>> # Non-streaming request
        >>> response = client.post("/v1/chat/completions", json={
        ...     "model": "llama-3.2-1b",
        ...     "messages": [{"role": "user", "content": "Hello!"}]
        ... })
        >>> response.json()["choices"][0]["message"]["content"]
        "Hello! How can I help you?"

        >>> # Streaming request
        >>> with client.stream("POST", "/v1/chat/completions", json={
        ...     "model": "llama-3.2-1b",
        ...     "messages": [{"role": "user", "content": "Hello!"}],
        ...     "stream": True
        ... }) as response:
        ...     for line in response.iter_lines():
        ...         print(line)

    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/create
    """
    # Generate request ID for tracking
    request_id = generate_request_id()

    # Validate model exists
    try:
        validate_model_exists(request.model)
    except HTTPException as exc:
        return create_error_response_from_http_exception(
            exc,
            request_id=request_id,
            param="model" if exc.status_code == 404 else None,
            code="model_not_found" if exc.status_code == 404 else None,
        )

    # Use mock for tests, real inference otherwise
    use_real_inference = os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") == "1"

    prompt_model = request.model if use_real_inference else None
    prompt = format_chat_prompt(request.messages, model=prompt_model)

    # Route based on rendered prompt size
    routing, prompt_tokens = build_chat_routing_result(prompt, prompt_model)

    # Record metrics
    record_routing_decision(routing.device, routing.reason)

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Convert OpenAI parameters to Ollama-style options for the pipeline
    openai_options: dict = {
        "temperature": request.temperature,
        "num_predict": request.max_tokens,
    }
    if request.top_p is not None:
        openai_options["top_p"] = request.top_p
    if request.presence_penalty is not None:
        openai_options["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        openai_options["frequency_penalty"] = request.frequency_penalty

    # Apply parameter pipeline: merge defaults, then map to OpenVINO format
    full_options = merge_with_defaults(openai_options)
    mapped_options = map_parameters(full_options)

    if request.stream:
        engine = None
        if use_real_inference:
            try:
                engine, engine_slot = await asyncio.to_thread(
                    lambda: _open_routed_engine_slot(routing.device)
                )
            except Exception as exc:
                from npu_proxy.inference.engine import DeviceBusyError

                if isinstance(exc, DeviceBusyError):
                    return create_error_response(
                        status_code=503,
                        message=str(exc),
                        error_type="server_error",
                        code="device_busy",
                        request_id=request_id,
                    )
                raise
            execution_device = _execution_device_from_engine(engine)
            fallback_reason = _fallback_reason(
                routed_device=routing.device,
                execution_device=execution_device,
                engine_slot=engine_slot,
            )
            _record_routing_execution(routing.device, execution_device, fallback_reason)
            return StreamingResponse(
                generate_stream_real(
                    request,
                    chat_id,
                    created,
                    prompt,
                    mapped_options,
                    request_id,
                    prompt_tokens,
                    engine=engine,
                    engine_slot=engine_slot,
                ),
                media_type="text/event-stream",
                headers=build_stream_headers(
                    routing,
                    request_id,
                    execution_device=execution_device,
                    fallback_reason=fallback_reason,
                ),
            )
        execution_device = await asyncio.to_thread(_get_execution_device, load_if_needed=False)
        fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
        _record_routing_execution(routing.device, execution_device, fallback_reason)
        return StreamingResponse(
            generate_stream(request, chat_id, created, prompt_tokens),
            media_type="text/event-stream",
            headers=build_stream_headers(
                routing,
                request_id,
                execution_device=execution_device,
                fallback_reason=fallback_reason,
            ),
        )

    # Non-streaming response
    if use_real_inference:
        try:
            def run_generation():
                engine, engine_slot = _open_routed_engine_slot(routing.device)
                try:
                    text = engine.generate(
                        prompt,
                        max_new_tokens=mapped_options.get("max_new_tokens", request.max_tokens),
                        temperature=mapped_options.get("temperature", request.temperature),
                        top_p=_effective_top_p(mapped_options, request.top_p),
                        timeout=float(DEFAULT_INFERENCE_TIMEOUT),
                    )
                    return engine, text, getattr(engine_slot, "fallback_reason", None)
                finally:
                    _close_engine_slot(engine_slot)

            engine, response_text, slot_fallback_reason = await asyncio.wait_for(
                asyncio.to_thread(run_generation),
                timeout=float(DEFAULT_INFERENCE_TIMEOUT) + load_device_queue_timeout(),
            )
            execution_device = _execution_device_from_engine(engine)
            fallback_reason = _fallback_reason(
                routed_device=routing.device,
                execution_device=execution_device,
                engine_slot=None,
            )
            fallback_reason = slot_fallback_reason or fallback_reason
            _record_routing_execution(routing.device, execution_device, fallback_reason)
            add_execution_headers(
                response,
                routing,
                execution_device,
                request_id,
                fallback_reason,
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.exception("Inference timed out", extra={"request_id": request_id})
            return create_error_response(
                status_code=504,
                message="Inference timed out",
                error_type="server_error",
                code="timeout",
                request_id=request_id,
            )
        except RuntimeError:
            logger.exception("Inference failed", extra={"request_id": request_id})
            return create_error_response(
                status_code=503,
                message="Inference failed",
                error_type="server_error",
                code="inference_error",
                request_id=request_id,
            )
        except Exception as exc:
            from npu_proxy.inference.engine import DeviceBusyError

            if isinstance(exc, DeviceBusyError):
                return create_error_response(
                    status_code=503,
                    message=str(exc),
                    error_type="server_error",
                    code="device_busy",
                    request_id=request_id,
                )
            logger.exception("Unexpected inference error", extra={"request_id": request_id})
            return create_error_response(
                status_code=500,
                message="Internal inference error",
                error_type="server_error",
                code="internal_error",
                request_id=request_id,
            )
    else:
        execution_device = await asyncio.to_thread(_get_execution_device, load_if_needed=False)
        fallback_reason = _fallback_reason(routed_device=routing.device, execution_device=execution_device)
        _record_routing_execution(routing.device, execution_device, fallback_reason)
        add_execution_headers(
            response,
            routing,
            execution_device,
            request_id,
            fallback_reason,
        )
        response_text, finish_reason = _mock_completion_text(
            "Hello! I'm a helpful AI assistant running on Intel NPU via OpenVINO.",
            request.max_tokens,
        )

    if use_real_inference:
        completion_tokens = count_tokens_best_effort(response_text, model=request.model).count
    else:
        completion_tokens = count_tokens(response_text)
    record_tokens(request.model, prompt_tokens, completion_tokens)
    if use_real_inference:
        finish_reason = determine_finish_reason(
            completion_tokens=completion_tokens,
            max_new_tokens=_effective_max_tokens(mapped_options, request.max_tokens),
            native_finish_reason=getattr(locals().get("engine", None), "last_finish_reason", None),
        )

    return ChatResponse(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
