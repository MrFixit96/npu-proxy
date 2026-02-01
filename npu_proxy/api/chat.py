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
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse, Response
import orjson

from npu_proxy.models.ollama_defaults import merge_with_defaults
from npu_proxy.models.parameter_mapper import map_parameters
from npu_proxy.routing.context_router import get_context_router, RoutingResult
from npu_proxy.metrics import record_routing_decision, record_request, record_inference, record_tokens, track_request

router = APIRouter(prefix="/v1", tags=["chat"])
logger = logging.getLogger(__name__)

# Lazy import to avoid loading model at import time
_engine = None


@dataclass
class OpenAIError:
    """OpenAI-compatible error structure.
    
    This dataclass represents the error object format used by OpenAI's API,
    ensuring compatibility with clients expecting OpenAI error responses.
    
    Attributes:
        message: Human-readable error description.
        type: Error category (e.g., "invalid_request_error", "server_error").
        param: The parameter that caused the error, if applicable.
        code: Machine-readable error code, if applicable.
    
    Example:
        >>> error = OpenAIError(
        ...     message="Invalid model specified",
        ...     type="invalid_request_error",
        ...     param="model",
        ...     code="model_not_found"
        ... )
    """
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


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
    return f"req_{uuid.uuid4().hex[:24]}"


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
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers=headers if headers else None,
    )


def get_engine():
    """Lazy load the inference engine.
    
    Defers importing and initializing the LLM engine until first use,
    reducing startup time and memory usage when the engine isn't needed.
    
    Returns:
        The singleton LLMEngine instance configured for the current device.
    
    Note:
        This function is thread-safe for the initial load due to Python's
        GIL, but the engine itself may have its own threading constraints.
    """
    global _engine
    if _engine is None:
        from npu_proxy.inference.engine import get_llm_engine
        _engine = get_llm_engine()
    return _engine


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
    from npu_proxy.models.registry import get_model_info
    if not get_model_info(model):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")


def add_routing_headers(response: Response, routing_result: RoutingResult, request_id: Optional[str] = None) -> None:
    """Add routing information and request ID to response headers.
    
    Adds custom headers to track which device handled the request,
    why that routing decision was made, and the request identifier.
    
    Args:
        response: The FastAPI/Starlette Response object to modify.
        routing_result: The routing decision containing device and reason.
        request_id: Optional unique request identifier for tracking.
    
    Headers Added:
        - X-Request-ID: Unique request identifier for tracing
        - X-NPU-Proxy-Device: The device that handled the request (e.g., "NPU", "CPU")
        - X-NPU-Proxy-Route-Reason: Why this device was selected
        - X-NPU-Proxy-Token-Count: Estimated token count for the request
    
    Example:
        >>> add_routing_headers(response, routing_result, "req_abc123")
        >>> response.headers["X-NPU-Proxy-Device"]
        "NPU"
    """
    if request_id:
        response.headers["X-Request-ID"] = request_id
    response.headers["X-NPU-Proxy-Device"] = routing_result.device
    response.headers["X-NPU-Proxy-Route-Reason"] = routing_result.reason
    response.headers["X-NPU-Proxy-Token-Count"] = str(routing_result.token_count)


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


def format_chat_prompt(messages: list[Message]) -> str:
    """Format messages into a prompt string for the model.
    
    Converts the structured message list into a plain text prompt format
    that the underlying LLM can process.
    
    Args:
        messages: List of Message objects representing the conversation.
    
    Returns:
        A formatted prompt string with role prefixes and the final
        "Assistant:" prompt to trigger generation.
    
    Example:
        >>> messages = [
        ...     Message(role="system", content="You are helpful."),
        ...     Message(role="user", content="Hi!")
        ... ]
        >>> format_chat_prompt(messages)
        "System: You are helpful.\\nUser: Hi!\\nAssistant:"
    
    Note:
        This is a simple formatting scheme. Production systems may use
        model-specific chat templates (e.g., ChatML, Llama format).
    """
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


async def generate_stream_real(
    request: ChatRequest,
    chat_id: str,
    created: int,
    prompt: str,
    mapped_options: dict | None = None,
) -> AsyncIterator[str]:
    """Generate streaming response with real-time token streaming.
    
    Uses AsyncTokenStream to yield tokens as they are generated by the
    inference engine, providing true real-time SSE streaming to clients.
    
    Args:
        request: The validated chat completion request.
        chat_id: Unique identifier for this completion.
        created: Unix timestamp of request creation.
        prompt: The formatted prompt string to send to the model.
        mapped_options: Optional dict of inference parameters (max_new_tokens,
            temperature) already mapped to OpenVINO format.
    
    Yields:
        SSE-formatted strings containing chat.completion.chunk objects.
        The stream ends with "data: [DONE]\\n\\n".
    
    Note:
        Inference runs in a background thread via run_in_executor to avoid
        blocking the event loop. Tokens are pushed to an async queue and
        yielded as they arrive.
    
    OpenAI API Reference:
        https://platform.openai.com/docs/api-reference/chat/streaming
    """
    from npu_proxy.inference.streaming import AsyncTokenStream
    
    engine = get_engine()
    
    # Use mapped options if provided, otherwise fall back to request values
    if mapped_options is None:
        mapped_options = {}
    max_new_tokens = mapped_options.get("max_new_tokens", request.max_tokens)
    temperature = mapped_options.get("temperature", request.temperature)
    
    # Send initial chunk with role
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
    
    # Create async token stream for real-time streaming
    loop = asyncio.get_event_loop()
    stream = AsyncTokenStream(timeout=180.0)
    stream.set_loop(loop)
    
    def run_inference():
        """Run inference in thread, pushing tokens via callback."""
        try:
            # Consume the generator to trigger token callbacks
            for _ in engine.generate_stream(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                streamer_callback=stream.callback,
            ):
                pass  # Tokens are pushed via callback
        except Exception as e:
            stream.error(e)
        finally:
            stream.complete()
    
    # Start inference in background thread
    inference_task = loop.run_in_executor(None, run_inference)
    
    # Yield tokens as they arrive in real-time
    try:
        async for token in stream:
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
    except TimeoutError:
        logger.error("Streaming inference timed out")
    except Exception as e:
        logger.error(f"Streaming error: {e}")
    
    # Wait for inference to complete
    await inference_task
    
    # Send final chunk with finish_reason
    final_chunk = ChatChunk(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            ChatChunkChoice(
                index=0,
                delta=ChatChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {orjson.dumps(final_chunk.model_dump()).decode()}\n\n"
    yield "data: [DONE]\n\n"


async def generate_stream(
    request: ChatRequest,
    chat_id: str,
    created: int,
) -> AsyncIterator[str]:
    """Generate mock streaming response for testing.
    
    Produces a simulated streaming response without invoking the real
    inference engine. Used for integration tests and development.
    
    Args:
        request: The validated chat completion request.
        chat_id: Unique identifier for this completion.
        created: Unix timestamp of request creation.
    
    Yields:
        SSE-formatted strings containing chat.completion.chunk objects.
        Simulates word-by-word streaming of a fixed response.
    
    Note:
        This function is only used when NPU_PROXY_REAL_INFERENCE != "1".
    """
    mock_response = "Hello! I'm a helpful AI assistant running on Intel NPU via OpenVINO."
    
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
    
    words = mock_response.split()
    for i, word in enumerate(words):
        content = word + (" " if i < len(words) - 1 else "")
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
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {orjson.dumps(final_chunk.model_dump()).decode()}\n\n"
    yield "data: [DONE]\n\n"


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
        EventSourceResponse: For streaming requests (stream=True), returns
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
    validate_model_exists(request.model)
    
    # Route based on context size
    context_router = get_context_router()
    routing = context_router.select_device_for_messages(
        [{"role": m.role, "content": m.content} for m in request.messages]
    )
    add_routing_headers(response, routing, request_id)
    
    # Record metrics
    record_routing_decision(routing.device, routing.reason)
    
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    prompt = format_chat_prompt(request.messages)
    
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
    
    # Use mock for tests, real inference otherwise
    use_real_inference = os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") == "1"
    
    if request.stream:
        if use_real_inference:
            return EventSourceResponse(
                generate_stream_real(request, chat_id, created, prompt, mapped_options),
                media_type="text/event-stream",
                headers={"X-Request-ID": request_id},
            )
        return EventSourceResponse(
            generate_stream(request, chat_id, created),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )
    
    # Non-streaming response
    if use_real_inference:
        try:
            engine = get_engine()
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: engine.generate(
                    prompt,
                    max_new_tokens=mapped_options.get("max_new_tokens", request.max_tokens),
                    temperature=mapped_options.get("temperature", request.temperature),
                )
            )
        except TimeoutError as e:
            logger.error(f"Inference timeout: {e}", extra={"request_id": request_id})
            return create_error_response(
                status_code=504,
                message=str(e),
                error_type="server_error",
                code="timeout",
                request_id=request_id,
            )
        except RuntimeError as e:
            logger.error(f"Inference failed: {e}", extra={"request_id": request_id})
            return create_error_response(
                status_code=503,
                message=f"Inference failed: {e}",
                error_type="server_error",
                code="inference_error",
                request_id=request_id,
            )
        except Exception as e:
            logger.exception("Unexpected inference error", extra={"request_id": request_id})
            return create_error_response(
                status_code=500,
                message="Internal inference error",
                error_type="server_error",
                code="internal_error",
                request_id=request_id,
            )
    else:
        response_text = "Hello! I'm a helpful AI assistant running on Intel NPU via OpenVINO."
    
    return ChatResponse(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(prompt.split()) + len(response_text.split()),
        ),
    )
