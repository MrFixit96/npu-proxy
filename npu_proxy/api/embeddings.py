"""Embeddings API endpoint handlers.

This module provides OpenAI-compatible embedding endpoints for generating
text embeddings using local NPU/CPU inference engines.

OpenAI API Compatibility:
    This module implements the OpenAI Embeddings API specification:
    https://platform.openai.com/docs/api-reference/embeddings
    
    Supported endpoints:
        - POST /v1/embeddings - Create embeddings for input text(s)
    
    Response format matches OpenAI's embedding response:
        {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.0023, -0.0094, ...],
                    "index": 0
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }

Headers:
    Request Headers:
        X-Request-ID: Optional client-provided request identifier
    
    Response Headers:
        X-Request-ID: Request identifier (echoed or generated)
        X-NPU-Proxy-Device: Device used for inference (CPU/NPU)
        X-NPU-Proxy-Model: Model used for embedding generation

Example:
    >>> import httpx
    >>> response = httpx.post(
    ...     "http://localhost:8000/v1/embeddings",
    ...     json={"model": "all-MiniLM-L6-v2", "input": "Hello world"},
    ...     headers={"X-Request-ID": "req-123"}
    ... )
    >>> response.json()["data"][0]["embedding"][:3]
    [0.0234, -0.0891, 0.0456]

Note:
    Embeddings run on CPU by default for reliability.
    NPU embedding support is experimental and may fall back to CPU
    if the model is too large for NPU memory.
"""

import logging
import uuid
from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from npu_proxy.api.header_utils import (
    add_request_id_header,
    apply_embedding_engine_headers as shared_apply_embedding_engine_headers,
    embedding_failure_reason,
    resolve_request_id,
    openai_error_content,
    validate_embedding_batch_result_count,
)
from npu_proxy.inference.embedding_config import InvalidEmbeddingModelError
from npu_proxy.inference.embedding_engine import (
    EmbeddingInferenceError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
    get_embedding_engine,
)
from npu_proxy.inference.tokenizer import count_tokens

router = APIRouter(prefix="/v1", tags=["embeddings"])
logger = logging.getLogger(__name__)
MAX_EMBEDDING_BATCH_SIZE = 128
MAX_EMBEDDING_TEXT_CHARS = 8192
MAX_EMBEDDING_TOTAL_CHARS = 65536


def _get_requested_device(http_request: Request) -> str | None:
    """Extract an optional per-request embedding device override."""
    device = http_request.headers.get("X-NPU-Proxy-Device")
    return device or None


def _apply_engine_headers(response: Response, engine_info: dict) -> None:
    """Expose resolved embedding routing details additively via sanitized headers."""
    public_info = dict(engine_info)
    if public_info.get("fallback_reason") or public_info.get("load_error"):
        public_info["fallback_reason"] = "Embedding engine unavailable"
        public_info.pop("load_error", None)
    shared_apply_embedding_engine_headers(response, public_info)


def _validate_embedding_batch_result_count(
    inputs: list[str],
    embeddings: list[list[float]],
) -> None:
    """Reject batch embedding responses that do not align 1:1 with inputs."""
    validate_embedding_batch_result_count(inputs, embeddings)


def _runtime_fallback_error_response(engine, response: Response, request_id: str) -> JSONResponse | None:
    """Reject runtime failover that would otherwise look like a successful embed."""
    engine_info = engine.get_engine_info()
    if not (engine_info.get("is_fallback") and engine_info.get("fallback_mode") == "runtime"):
        return None

    logger.error(
        "Request %s: embedding runtime fallback activated: %s",
        request_id,
        embedding_failure_reason(engine_info, default="unknown"),
    )
    _apply_engine_headers(response, engine_info)
    return _openai_error_response(
        500,
        "Embedding inference failed",
        error_type="inference_error",
        code="embedding_failed",
        headers=dict(response.headers),
    )


def _openai_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=openai_error_content(message, error_type=error_type, code=code),
        headers=headers,
    )


def _validate_inputs(inputs: list[str], response: Response) -> JSONResponse | None:
    """Validate input bounds before engine loading/inference."""
    if not inputs:
        return _openai_error_response(
            400,
            "Embedding input must contain at least one item",
            error_type="invalid_request_error",
            code="empty_input",
            headers=dict(response.headers),
        )
    if len(inputs) > MAX_EMBEDDING_BATCH_SIZE:
        return _openai_error_response(
            413,
            "Embedding input batch is too large",
            error_type="request_too_large",
            code="embedding_batch_too_large",
            headers=dict(response.headers),
        )
    total_chars = 0
    for text in inputs:
        if not text.strip():
            return _openai_error_response(
                400,
                "Embedding input text must not be empty",
                error_type="invalid_request_error",
                code="empty_input",
                headers=dict(response.headers),
            )
        text_length = len(text)
        if text_length > MAX_EMBEDDING_TEXT_CHARS:
            return _openai_error_response(
                413,
                "Embedding input text is too large",
                error_type="request_too_large",
                code="embedding_input_too_large",
                headers=dict(response.headers),
            )
        total_chars += text_length
    if total_chars > MAX_EMBEDDING_TOTAL_CHARS:
        return _openai_error_response(
            413,
            "Embedding request is too large",
            error_type="request_too_large",
            code="embedding_request_too_large",
            headers=dict(response.headers),
        )
    return None


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request.
    
    Attributes:
        model: Model identifier for embedding generation.
            Common values: "all-MiniLM-L6-v2", "text-embedding-ada-002"
        input: Text(s) to embed. Can be a single string or list of strings.
            Maximum tokens per input depends on model (typically 512).
        encoding_format: Output format for embeddings.
            Currently only "float" is supported. "base64" is not implemented.
    
    Example:
        >>> request = EmbeddingRequest(
        ...     model="all-MiniLM-L6-v2",
        ...     input=["Hello world", "Goodbye world"]
        ... )
    """
    
    model: str = Field(
        ...,
        description="Model identifier for embedding generation",
        examples=["all-MiniLM-L6-v2", "text-embedding-ada-002"]
    )
    input: str | list[str] = Field(
        ...,
        description="Text(s) to embed. Single string or list of strings.",
        examples=["Hello world", ["Hello", "World"]]
    )
    encoding_format: str = Field(
        default="float",
        description="Output encoding format. Only 'float' is currently supported."
    )


class EmbeddingData(BaseModel):
    """Single embedding result in OpenAI format.
    
    Attributes:
        object: Object type, always "embedding".
        embedding: Vector representation as list of floats.
            Dimension depends on model (e.g., 384 for MiniLM, 1536 for ada-002).
        index: Position in the input list (0-indexed).
    """
    
    object: str = Field(
        default="embedding",
        description="Object type identifier"
    )
    embedding: list[float] = Field(
        ...,
        description="Embedding vector as list of floats"
    )
    index: int = Field(
        ...,
        description="Index of this embedding in the input list"
    )


class EmbeddingUsage(BaseModel):
    """Token usage statistics for embedding request.
    
    Attributes:
        prompt_tokens: Number of tokens in the input text(s).
        total_tokens: Total tokens processed (equals prompt_tokens for embeddings).
    
    Note:
        Unlike chat completions, embeddings have no completion tokens.
        prompt_tokens always equals total_tokens.
    """
    
    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in input text(s)"
    )
    total_tokens: int = Field(
        ...,
        description="Total tokens processed"
    )


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response.
    
    Attributes:
        object: Response type, always "list".
        data: List of embedding results, one per input text.
        model: Model used for embedding generation.
        usage: Token usage statistics.
    
    OpenAI Compatibility:
        This response format matches the OpenAI Embeddings API exactly:
        https://platform.openai.com/docs/api-reference/embeddings/create
    """
    
    object: str = Field(
        default="list",
        description="Response object type"
    )
    data: list[EmbeddingData] = Field(
        ...,
        description="List of embedding results"
    )
    model: str = Field(
        ...,
        description="Model used for embeddings"
    )
    usage: EmbeddingUsage = Field(
        ...,
        description="Token usage statistics"
    )


class ErrorDetail(BaseModel):
    """Error detail following OpenAI error format.
    
    Attributes:
        message: Human-readable error description.
        type: Error category (e.g., "engine_error", "timeout_error").
        code: Machine-readable error code (e.g., "engine_init_failed").
    """
    
    message: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type category")
    code: str | None = Field(default=None, description="Machine-readable error code")


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response wrapper.
    
    Attributes:
        error: Error details containing message, type, and optional code.
    """
    
    error: ErrorDetail = Field(..., description="Error details")


@router.post("/embeddings", response_model=EmbeddingResponse, responses={
    503: {"model": ErrorResponse, "description": "Embedding engine unavailable"},
    500: {"model": ErrorResponse, "description": "Inference error"},
    504: {"model": ErrorResponse, "description": "Inference timeout"},
})
async def create_embeddings(
    request: EmbeddingRequest,
    http_request: Request,
    response: Response
):
    """Create embeddings for input text(s).
    
    OpenAI-compatible endpoint for generating text embeddings using local
    inference engines (NPU or CPU fallback).
    
    Args:
        request: Embedding request containing model and input text(s).
        http_request: FastAPI request object for accessing headers.
        response: FastAPI response object for setting headers.
    
    Returns:
        EmbeddingResponse with embedding vectors matching OpenAI format:
            - object: "list"
            - data: List of EmbeddingData with vectors and indices
            - model: Echo of requested model name
            - usage: Token counts for billing/tracking
    
    Error responses:
        Returns OpenAI-compatible {"error": {...}} payloads for invalid
        requests, unavailable engines, inference failures, and timeouts.
    
    OpenAI Compatibility:
        - Matches POST /v1/embeddings from OpenAI API
        - Supports single string or list of strings as input
        - Returns object="list" with embedding data array
        - Each embedding includes index for input correlation
        - Usage tracks prompt_tokens and total_tokens
    
    Headers:
        Request:
            X-Request-ID: Optional client-provided request identifier.
                If not provided, a UUID will be generated.
        
        Response:
            X-Request-ID: Request identifier (echoed or generated).
            X-NPU-Proxy-Device: Device used for inference (CPU/NPU).
            X-NPU-Proxy-Model: Actual model used for embedding.
            X-NPU-Proxy-Fallback: "true" if fallback was used due to errors.
    
    Example:
        >>> # Single text embedding
        >>> response = client.post("/v1/embeddings", json={
        ...     "model": "all-MiniLM-L6-v2",
        ...     "input": "Hello world"
        ... })
        >>> len(response.json()["data"][0]["embedding"])
        384
        
        >>> # Batch embedding
        >>> response = client.post("/v1/embeddings", json={
        ...     "model": "all-MiniLM-L6-v2",
        ...     "input": ["Hello", "World", "Test"]
        ... })
        >>> len(response.json()["data"])
        3
    
    Note:
        Embeddings run on CPU by default for reliability.
        NPU embedding support is experimental and may encounter
        ZE_RESULT_ERROR_UNKNOWN errors for large models.
    """
    # Generate or extract request ID for tracing
    request_id = resolve_request_id(
        http_request.headers.get("X-Request-ID"),
        prefix="",
        hex_chars=32,
        generator=lambda: str(uuid.uuid4()),
    )
    add_request_id_header(response, request_id)

    if request.encoding_format != "float":
        return _openai_error_response(
            400,
            "Only encoding_format='float' is currently supported",
            error_type="invalid_request_error",
            code="unsupported_encoding_format",
            headers=dict(response.headers),
        )

    logger.info("Embedding request %s: model=%s", request_id, request.model)

    # Normalize input to list
    inputs = request.input if isinstance(request.input, list) else [request.input]
    validation_error = _validate_inputs(inputs, response)
    if validation_error is not None:
        return validation_error
    requested_device = _get_requested_device(http_request)

    # Get embedding engine
    try:
        engine = await run_in_threadpool(
            get_embedding_engine,
            model_name=request.model,
            device=requested_device,
        )
    except InvalidEmbeddingModelError:
        logger.warning(
            "Request %s: rejected embedding model identifier",
            request_id,
            exc_info=True,
        )
        return _openai_error_response(
            400,
            "Invalid embedding model identifier",
            error_type="invalid_request_error",
            code="invalid_embedding_model",
            headers=dict(response.headers),
        )
    except EmbeddingUnavailableError:
        logger.warning(
            "Request %s: embedding engine unavailable",
            request_id,
            exc_info=True,
        )
        return _openai_error_response(
            503,
            "Embedding engine is unavailable",
            error_type="service_unavailable_error",
            code="embedding_unavailable",
            headers=dict(response.headers),
        )
    except Exception:
        logger.error("Request %s: failed to get embedding engine", request_id, exc_info=True)
        return _openai_error_response(
            503,
            "Embedding engine is unavailable",
            error_type="service_unavailable_error",
            code="embedding_unavailable",
            headers=dict(response.headers),
        )
    
    engine_info = engine.get_engine_info()
    _apply_engine_headers(response, engine_info)
    if engine_info.get("is_fallback"):
        logger.warning(
            "Request %s: using embedding fallback (%s)",
            request_id,
            embedding_failure_reason(engine_info, default="unknown"),
        )
        if not engine_info.get("fallback_allowed", False):
            return _openai_error_response(
                503,
                "Embedding engine is unavailable",
                error_type="service_unavailable_error",
                code="embedding_unavailable",
                headers=dict(response.headers),
            )
    
    # Generate embeddings for all inputs
    try:
        embeddings = await run_in_threadpool(engine.embed_batch, inputs)
        _validate_embedding_batch_result_count(inputs, embeddings)
        runtime_fallback_error = _runtime_fallback_error_response(engine, response, request_id)
        if runtime_fallback_error is not None:
            return runtime_fallback_error
        logger.debug("Request %s: Generated %s embeddings", request_id, len(embeddings))
    except EmbeddingTimeoutError:
        logger.error("Request %s: embedding timeout", request_id, exc_info=True)
        return _openai_error_response(
            504,
            "Embedding inference timed out",
            error_type="timeout_error",
            code="inference_timeout",
            headers=dict(response.headers),
        )
    except EmbeddingUnavailableError:
        logger.warning(
            "Request %s: embedding engine unavailable during inference",
            request_id,
            exc_info=True,
        )
        return _openai_error_response(
            503,
            "Embedding engine is unavailable",
            error_type="service_unavailable_error",
            code="embedding_unavailable",
            headers=dict(response.headers),
        )
    except EmbeddingInferenceError:
        logger.error("Request %s: embedding failed", request_id, exc_info=True)
        return _openai_error_response(
            500,
            "Embedding inference failed",
            error_type="inference_error",
            code="embedding_failed",
            headers=dict(response.headers),
        )
    except Exception:
        logger.error("Request %s: embedding failed", request_id, exc_info=True)
        return _openai_error_response(
            500,
            "Embedding inference failed",
            error_type="inference_error",
            code="embedding_failed",
            headers=dict(response.headers),
        )
    
    # Build response
    embeddings_data = []
    total_tokens = 0
    
    for i, (text, embedding) in enumerate(zip(inputs, embeddings)):
        embeddings_data.append(
            EmbeddingData(
                embedding=embedding,
                index=i,
            )
        )
        total_tokens += count_tokens(text)
    
    logger.info("Request %s: Completed with %s tokens", request_id, total_tokens)
    
    return EmbeddingResponse(
        data=embeddings_data,
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )
