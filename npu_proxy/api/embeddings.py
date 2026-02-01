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
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from npu_proxy.inference.embedding_engine import get_embedding_engine
from npu_proxy.inference.tokenizer import count_tokens

router = APIRouter(prefix="/v1", tags=["embeddings"])
logger = logging.getLogger(__name__)


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
    
    Raises:
        HTTPException(500): Engine initialization or inference failure.
        HTTPException(504): Inference timeout exceeded.
    
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
    request_id = http_request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response.headers["X-Request-ID"] = request_id
    
    logger.info(f"Embedding request {request_id}: model={request.model}")
    
    # Normalize input to list
    inputs = request.input if isinstance(request.input, list) else [request.input]
    
    # Get embedding engine
    try:
        engine = get_embedding_engine()
    except Exception as e:
        logger.error(f"Failed to get embedding engine: {e}")
        raise HTTPException(
            status_code=500,
            detail={"message": str(e), "type": "engine_error", "code": "engine_init_failed"},
        )
    
    # Check if engine had load errors (model too large for device, etc.)
    engine_info = engine.get_engine_info()
    if engine_info.get("load_error"):
        logger.warning(f"Request {request_id}: Using fallback due to load error: {engine_info['load_error']}")
        response.headers["X-NPU-Proxy-Fallback"] = "true"
    
    # Set routing headers for observability
    response.headers["X-NPU-Proxy-Device"] = engine_info.get("device", "unknown")
    response.headers["X-NPU-Proxy-Model"] = engine_info.get("model", request.model)
    
    # Generate embeddings for all inputs
    try:
        embeddings = engine.embed_batch(inputs)
        logger.debug(f"Request {request_id}: Generated {len(embeddings)} embeddings")
    except TimeoutError as e:
        logger.error(f"Request {request_id}: Embedding timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail={"message": str(e), "type": "timeout_error", "code": "inference_timeout"},
        )
    except Exception as e:
        # This catches OpenVINO errors like ZE_RESULT_ERROR_UNKNOWN
        error_msg = str(e)
        logger.error(f"Request {request_id}: Embedding failed: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail={"message": error_msg, "type": "inference_error", "code": "embedding_failed"},
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
    
    logger.info(f"Request {request_id}: Completed with {total_tokens} tokens")
    
    return EmbeddingResponse(
        data=embeddings_data,
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )
