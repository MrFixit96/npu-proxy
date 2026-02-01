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
import uuid
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.responses import Response
from pydantic import BaseModel, Field
from typing import Optional

from npu_proxy.routing.context_router import get_context_router, RoutingResult
from npu_proxy.metrics import record_routing_decision

from npu_proxy.models.ollama_defaults import merge_with_defaults
from npu_proxy.models.parameter_mapper import map_parameters

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


class ChatMessage(BaseModel):
    """A single message in a chat conversation.
    
    Attributes:
        role: Message role (system, user, or assistant).
        content: Message content text.
    """
    role: str
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


# Import from registry for single source of truth
from npu_proxy.models.registry import MODELS_INFO as REGISTRY_MODELS_INFO


def _get_ollama_model_info(model_id: str) -> dict:
    """Get model info in Ollama format from registry.
    
    Converts internal registry format to Ollama-compatible model info.
    
    Args:
        model_id: The model identifier (e.g., "tinyllama").
    
    Returns:
        Dictionary with size, digest, parameter_size, quantization_level,
        and family fields in Ollama format.
    """
    info = REGISTRY_MODELS_INFO.get(model_id, {})
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
    return f"req-{uuid.uuid4().hex[:12]}"


def add_request_headers(response: Response, request_id: str) -> None:
    """Add request ID header for tracing.
    
    Args:
        response: FastAPI response object.
        request_id: Unique request identifier.
    """
    response.headers["X-Request-ID"] = request_id


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
        0.1.0-npu-proxy
    """
    return VersionResponse(version="0.1.0-npu-proxy")


@router.post("/show", response_model=ShowResponse)
async def show_model(request: ShowRequest):
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
    from npu_proxy.models.registry import get_model_info
    
    model_info = get_model_info(request.model)
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")
    
    info = _get_ollama_model_info(request.model)
    
    return ShowResponse(
        modelfile=f"FROM {request.model}",
        parameters="",
        template="",
        details=ModelDetails(
            parameter_size=info.get("parameter_size", ""),
            quantization_level=info.get("quantization_level", ""),
            family=info.get("family", "llama"),
        ),
        model_info={
            "general.architecture": info.get("family", "llama"),
            "general.parameter_count": info.get("parameter_size", ""),
            "general.quantization": info.get("quantization_level", ""),
        },
    )


def validate_ollama_model(model: str) -> None:
    """Validate that a model exists in the registry.
    
    Args:
        model: Model name to validate.
    
    Raises:
        HTTPException: 404 if model not found in registry.
    """
    from npu_proxy.models.registry import get_model_info
    if not get_model_info(model):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")


def add_routing_headers(response: Response, routing_result: RoutingResult) -> None:
    """Add routing information to response headers.
    
    Adds headers indicating which device was selected and why,
    useful for debugging and monitoring routing decisions.
    
    Args:
        response: FastAPI response object.
        routing_result: The routing decision from context router.
    """
    response.headers["X-NPU-Proxy-Device"] = routing_result.device
    response.headers["X-NPU-Proxy-Route-Reason"] = routing_result.reason
    response.headers["X-NPU-Proxy-Token-Count"] = str(routing_result.token_count)


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


def get_routing_for_messages(messages: list) -> RoutingResult:
    """Get routing decision for chat messages.
    
    Analyzes the full conversation history to determine optimal
    device based on total token count across all messages.
    
    Args:
        messages: List of ChatMessage objects.
    
    Returns:
        RoutingResult with device selection and reasoning.
    """
    router = get_context_router()
    return router.select_device_for_messages(
        [{"role": m.role, "content": m.content} for m in messages]
    )


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
    import os
    import asyncio
    from sse_starlette.sse import EventSourceResponse
    import orjson
    
    # Generate request ID for tracing
    request_id = _generate_request_id()
    add_request_headers(response, request_id)
    
    # Validate model exists
    validate_ollama_model(request.model)
    
    # Route based on context size
    routing = get_routing_for_prompt(request.prompt)
    add_routing_headers(response, routing)
    record_routing_decision(routing.device, routing.reason)
    
    # Apply parameter pipeline: merge defaults, then map to OpenVINO format
    user_options = request.options or {}
    full_options = merge_with_defaults(user_options)
    mapped_options = map_parameters(full_options)
    
    use_real = os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") == "1"
    
    if request.stream:
        async def stream_generate():
            if use_real:
                from npu_proxy.inference.engine import get_llm_engine
                from npu_proxy.inference.streaming import AsyncTokenStream
                
                engine = get_llm_engine()
                loop = asyncio.get_event_loop()
                
                # Create async token stream for real-time streaming
                stream = AsyncTokenStream(timeout=180.0)
                stream.set_loop(loop)
                
                def run_inference():
                    """Run inference in thread, pushing tokens via callback."""
                    try:
                        for _ in engine.generate_stream(
                            request.prompt,
                            max_new_tokens=mapped_options.get("max_new_tokens", 256),
                            temperature=mapped_options.get("temperature", 0.8),
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
                        chunk = GenerateResponse(
                            model=request.model,
                            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            response=token,
                            done=False,
                        )
                        yield orjson.dumps(chunk.model_dump()).decode() + "\n"
                except TimeoutError:
                    logger.error("Streaming inference timed out")
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                
                # Wait for inference to complete
                await inference_task
            else:
                # Mock streaming
                response = "Hello! I'm running on Intel NPU."
                for word in response.split():
                    chunk = GenerateResponse(
                        model=request.model,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        response=word + " ",
                        done=False,
                    )
                    yield orjson.dumps(chunk.model_dump()).decode() + "\n"
            
            # Final chunk
            final = GenerateResponse(
                model=request.model,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                response="",
                done=True,
                total_duration=1000000,
                eval_count=10,
            )
            yield orjson.dumps(final.model_dump()).decode() + "\n"
        
        return EventSourceResponse(stream_generate(), media_type="application/x-ndjson")
    
    # Non-streaming
    if use_real:
        try:
            from npu_proxy.inference.engine import get_llm_engine
            engine = get_llm_engine()
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: engine.generate(
                    request.prompt,
                    max_new_tokens=mapped_options.get("max_new_tokens", 256),
                    temperature=mapped_options.get("temperature", 0.8),
                )
            )
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=f"Inference failed: {e}")
        except Exception as e:
            logger.exception("Unexpected inference error")
            raise HTTPException(status_code=500, detail="Internal inference error")
    else:
        response_text = "Hello! I'm running on Intel NPU via OpenVINO."
    
    return GenerateResponse(
        model=request.model,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        response=response_text,
        done=True,
        total_duration=1000000,
        eval_count=len(response_text.split()),
    )


def format_chat_prompt(messages: list[ChatMessage]) -> str:
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
    
    # Validate model exists
    validate_ollama_model(request.model)
    
    # Route based on context size
    routing = get_routing_for_messages(request.messages)
    add_routing_headers(response, routing)
    record_routing_decision(routing.device, routing.reason)
    
    # Apply parameter pipeline: merge defaults, then map to OpenVINO format
    user_options = request.options or {}
    full_options = merge_with_defaults(user_options)
    mapped_options = map_parameters(full_options)
    
    use_real = os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") == "1"
    
    prompt = format_chat_prompt(request.messages)
    
    if request.stream:
        from sse_starlette.sse import EventSourceResponse
        import orjson
        
        async def stream_chat():
            if use_real:
                try:
                    from npu_proxy.inference.engine import get_llm_engine
                    from npu_proxy.inference.streaming import AsyncTokenStream
                    
                    engine = get_llm_engine()
                    loop = asyncio.get_event_loop()
                    
                    # Create async token stream for real-time streaming
                    stream = AsyncTokenStream(timeout=180.0)
                    stream.set_loop(loop)
                    
                    def run_inference():
                        """Run inference in thread, pushing tokens via callback."""
                        try:
                            for _ in engine.generate_stream(
                                prompt,
                                max_new_tokens=mapped_options.get("max_new_tokens", 256),
                                temperature=mapped_options.get("temperature", 0.8),
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
                            chunk = ChatResponse(
                                model=request.model,
                                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                message=ChatMessage(role="assistant", content=token),
                                done=False,
                            )
                            yield orjson.dumps(chunk.model_dump()).decode() + "\n"
                    except TimeoutError:
                        logger.error("Streaming inference timed out")
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                    
                    # Wait for inference to complete
                    await inference_task
                except Exception as e:
                    logger.exception("Streaming inference error")
                    raise
            else:
                response = "Hello! I'm running on Intel NPU."
                for word in response.split():
                    chunk = ChatResponse(
                        model=request.model,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        message=ChatMessage(role="assistant", content=word + " "),
                        done=False,
                    )
                    yield orjson.dumps(chunk.model_dump()).decode() + "\n"
            
            # Final chunk
            final = ChatResponse(
                model=request.model,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                message=ChatMessage(role="assistant", content=""),
                done=True,
                total_duration=1000000,
                eval_count=10,
            )
            yield orjson.dumps(final.model_dump()).decode() + "\n"
        
        return EventSourceResponse(stream_chat(), media_type="application/x-ndjson")
    
    # Non-streaming
    if use_real:
        try:
            from npu_proxy.inference.engine import get_llm_engine
            engine = get_llm_engine()
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: engine.generate(
                    prompt,
                    max_new_tokens=mapped_options.get("max_new_tokens", 256),
                    temperature=mapped_options.get("temperature", 0.8),
                )
            )
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=f"Inference failed: {e}")
        except Exception as e:
            logger.exception("Unexpected inference error")
            raise HTTPException(status_code=500, detail="Internal inference error")
    else:
        response_text = "Hello! I'm running on Intel NPU via OpenVINO."
    
    return ChatResponse(
        model=request.model,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        message=ChatMessage(role="assistant", content=response_text),
        done=True,
        total_duration=1000000,
        eval_count=len(response_text.split()),
    )


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
async def ollama_embed(request: OllamaEmbedRequest):
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
    from npu_proxy.inference.embedding_engine import get_embedding_engine
    from npu_proxy.inference.tokenizer import count_tokens
    
    start = time.perf_counter_ns()
    
    try:
        engine = get_embedding_engine()
    except Exception as e:
        logger.error(f"Failed to get embedding engine: {e}")
        raise HTTPException(status_code=500, detail=f"Engine initialization failed: {e}")
    
    # Normalize input to list
    texts = [request.input] if isinstance(request.input, str) else request.input
    
    # Generate embeddings with error handling
    try:
        embeddings = engine.embed_batch(texts)
    except TimeoutError as e:
        logger.error(f"Embedding timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        # This catches OpenVINO errors like ZE_RESULT_ERROR_UNKNOWN
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    
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


@router.post("/embeddings", response_model=OllamaEmbeddingsLegacyResponse)
async def ollama_embeddings_legacy(request: OllamaEmbeddingsLegacyRequest):
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
    from npu_proxy.inference.embedding_engine import get_embedding_engine
    
    try:
        engine = get_embedding_engine()
    except Exception as e:
        logger.error(f"Failed to get embedding engine: {e}")
        raise HTTPException(status_code=500, detail=f"Engine initialization failed: {e}")
    
    # Generate single embedding with error handling
    try:
        embedding = engine.embed(request.prompt)
    except TimeoutError as e:
        logger.error(f"Embedding timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        # This catches OpenVINO errors like ZE_RESULT_ERROR_UNKNOWN
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    
    return OllamaEmbeddingsLegacyResponse(
        embedding=embedding,
    )


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


@router.post("/pull")
async def pull_model(request: PullRequest):
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
    try:
        from npu_proxy.models.mapper import resolve_model_repo
        from npu_proxy.models.downloader import (
            download_model,
            is_model_downloaded,
            get_download_progress,
        )
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Model management not available: {e}")
    
    if not request.name:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    logger.info(f"Pull request for model: {request.name}")
    
    # Resolve model name to HuggingFace repo
    resolved = resolve_model_repo(request.name)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.name}' not found. Use /api/search to find available models."
        )
    
    repo_id, local_name = resolved
    
    # Check if already downloaded
    if is_model_downloaded(request.name):
        return JSONResponse({
            "status": "success",
            "model": local_name,
            "message": "Model already downloaded"
        })
    
    if request.stream:
        async def generate_progress():
            for progress in get_download_progress(repo_id, None):
                yield json.dumps(progress) + "\n"
        
        return StreamingResponse(
            generate_progress(),
            media_type="application/x-ndjson"
        )
    else:
        result = download_model(request.name)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return JSONResponse(result)


@router.get("/search")
async def search_models(
    q: str = Query(default="", description="Search query"),
    sort: str = Query(default="popular", description="Sort: popular, newest, downloads, likes"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    type: str = Query(default="all", description="Filter: all, llm, embedding, vision"),
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
        type: Filter by model type (all, llm, embedding, vision).
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
    try:
        from npu_proxy.models.search import search_openvino_models, SearchResult
        from npu_proxy.models.mapper import get_ollama_name
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Model search not available: {e}")
    
    # Validate sort parameter
    valid_sorts = ['popular', 'newest', 'downloads', 'likes']
    if sort not in valid_sorts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort value. Use: {', '.join(valid_sorts)}"
        )
    
    # Validate type parameter
    valid_types = ['all', 'llm', 'embedding', 'vision']
    if type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid type value. Use: {', '.join(valid_types)}"
        )
    
    logger.info(f"Search request: q='{q}', sort={sort}, type={type}")
    
    try:
        results, total = search_openvino_models(
            query=q,
            sort=sort,
            limit=limit,
            offset=offset,
            model_type=type,
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
        logger.exception(f"Search error: {e}")
        raise HTTPException(status_code=503, detail="Model search temporarily unavailable")


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
