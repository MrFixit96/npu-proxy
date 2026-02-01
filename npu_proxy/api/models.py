"""Models endpoint handlers.

This module provides OpenAI-compatible model listing endpoints for the NPU Proxy.
It allows clients to discover available OpenVINO-optimized models that can be
used for inference on Intel NPU hardware.

OpenAI API Compatibility:
    This module implements the GET /v1/models endpoint matching the OpenAI API
    specification. The response format is compatible with OpenAI client libraries.

Response Format:
    {
        "object": "list",
        "data": [
            {
                "id": "tinyllama-1.1b-chat-int4-ov",
                "object": "model",
                "created": 1677610602,
                "owned_by": "local"
            }
        ]
    }
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel
from npu_proxy.models.registry import list_all_models

router = APIRouter(prefix="/v1", tags=["models"])


class ModelInfo(BaseModel):
    """Information about a single model.
    
    Represents a model available for inference, following the OpenAI model
    object specification.
    
    Attributes:
        id: Unique identifier for the model (e.g., "tinyllama-1.1b-chat-int4-ov").
        object: Object type, always "model" per OpenAI spec.
        created: Unix timestamp of when the model info was generated.
        owned_by: Organization or source that owns/provides the model.
    
    OpenAI Compatibility:
        Maps directly to OpenAI's model object schema. The 'id' field can be
        used in chat completion requests to specify which model to use.
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """Response containing list of available models.
    
    OpenAI-compatible response format for the /v1/models endpoint.
    
    Attributes:
        object: Response type, always "list" per OpenAI spec.
        data: List of ModelInfo objects representing available models.
    
    OpenAI Compatibility:
        Matches the exact structure returned by OpenAI's GET /v1/models endpoint,
        enabling drop-in compatibility with OpenAI client libraries.
    """

    object: str = "list"
    data: list[ModelInfo]


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List available models.
    
    OpenAI-compatible endpoint for model discovery. Returns all locally
    available OpenVINO-optimized models that can be used for inference.
    
    Returns:
        ModelsResponse: Response with list of available models, each containing
            id, object type, creation timestamp, and owner information.
    
    OpenAI Compatibility:
        - Matches GET /v1/models from OpenAI API
        - Returns object="list" with model data array
        - Each model has id, object, created, owned_by fields
        - Model IDs can be used directly in /v1/chat/completions requests
    
    Note:
        Available models depend on which OpenVINO-optimized models have been
        downloaded to the local cache. The 'created' timestamp reflects when
        the listing was generated, not the model's actual creation date.
    
    Example:
        >>> response = client.get("/v1/models")
        >>> models = response.json()["data"]
        >>> [m["id"] for m in models]
        ['tinyllama-1.1b-chat-int4-ov', 'mistral-7b-instruct-int4-ov']
    """
    models = list_all_models()
    model_infos = [
        ModelInfo(
            id=m["id"],
            created=int(time.time()),
            owned_by=m.get("owned_by", "local"),
        )
        for m in models
    ]
    return ModelsResponse(data=model_infos)
