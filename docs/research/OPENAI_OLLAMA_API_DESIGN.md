# API Design Research: OpenAI/Ollama-Compatible Patterns

> Research for npu-proxy API implementation

## Executive Summary

This document captures best practices for building OpenAI and Ollama-compatible LLM APIs, based on research of:
- Ollama API documentation (https://github.com/ollama/ollama/blob/main/docs/api.md)
- vLLM OpenAI-compatible server patterns
- LocalAI implementation patterns
- npu-proxy existing implementation

---

## 1. Request Validation Patterns

### Pattern: Pydantic Model Validation with Field Constraints

**Problem:** Ensure incoming API requests have valid parameters with appropriate bounds.

**Implementation (from npu-proxy chat.py):**
```python
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
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
        if not v:
            raise ValueError("messages cannot be empty")
        return v
```

**Key Points:**
- Use `Field(ge=, le=)` for numeric bounds
- Use `field_validator` for complex validation logic
- Return 422 Unprocessable Entity for validation errors (FastAPI default)

**Applicability to npu-proxy:** ✅ Already implemented in `chat.py`

---

### Pattern: Role Validation for Messages

**Problem:** Ensure message roles match expected values.

**Implementation:**
```python
class Message(BaseModel):
    role: str
    content: str
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ('system', 'user', 'assistant'):
            raise ValueError(f"Invalid role '{v}'. Must be 'system', 'user', or 'assistant'")
        return v
```

**Applicability to npu-proxy:** ✅ Already implemented

---

### Pattern: Model Existence Validation

**Problem:** Return appropriate errors for non-existent models.

**Implementation:**
```python
def validate_model_exists(model: str) -> None:
    """Raise 404 if model doesn't exist in registry"""
    from npu_proxy.models.registry import get_model_info
    if not get_model_info(model):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
```

**Applicability to npu-proxy:** ✅ Already implemented in both chat.py and ollama.py

---

## 2. Response Formatting: Streaming vs Non-Streaming

### Pattern: OpenAI SSE Streaming Format

**Problem:** Stream tokens in real-time using Server-Sent Events format compatible with OpenAI clients.

**OpenAI Stream Format:**
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Implementation (npu-proxy pattern):**
```python
from sse_starlette.sse import EventSourceResponse
import orjson

class ChatChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None

class ChatChunkChoice(BaseModel):
    index: int
    delta: ChatChunkDelta
    finish_reason: str | None = None

class ChatChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]

async def generate_stream(request, chat_id, created, prompt):
    # Initial chunk with role
    initial_chunk = ChatChunk(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[ChatChunkChoice(
            index=0,
            delta=ChatChunkDelta(role="assistant"),
        )]
    )
    yield f"data: {orjson.dumps(initial_chunk.model_dump()).decode()}\n\n"
    
    # Token chunks
    async for token in stream:
        chunk = ChatChunk(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[ChatChunkChoice(
                index=0,
                delta=ChatChunkDelta(content=token),
            )]
        )
        yield f"data: {orjson.dumps(chunk.model_dump()).decode()}\n\n"
    
    # Final chunk
    final_chunk = ChatChunk(
        id=chat_id,
        created=created,
        model=request.model,
        choices=[ChatChunkChoice(
            index=0,
            delta=ChatChunkDelta(),
            finish_reason="stop",
        )]
    )
    yield f"data: {orjson.dumps(final_chunk.model_dump()).decode()}\n\n"
    yield "data: [DONE]\n\n"
```

**Applicability to npu-proxy:** ✅ Fully implemented

---

### Pattern: Ollama NDJSON Streaming

**Problem:** Ollama uses newline-delimited JSON (NDJSON) for streaming, not SSE.

**Ollama Stream Format:**
```json
{"model":"llama3.2","created_at":"2023-08-04T08:52:19Z","response":"The","done":false}
{"model":"llama3.2","created_at":"2023-08-04T08:52:19Z","response":" sky","done":false}
{"model":"llama3.2","created_at":"2023-08-04T19:22:45Z","response":"","done":true,"total_duration":4883583458}
```

**Implementation:**
```python
async def stream_generate():
    for token in tokens:
        chunk = GenerateResponse(
            model=request.model,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            response=token,
            done=False,
        )
        yield orjson.dumps(chunk.model_dump()).decode() + "\n"
    
    # Final chunk with stats
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
```

**Applicability to npu-proxy:** ✅ Implemented in ollama.py

---

### Pattern: Non-Streaming Response (OpenAI)

**OpenAI Format:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

**Applicability to npu-proxy:** ✅ Implemented

---

## 3. Error Response Format (OpenAI Schema)

### Pattern: OpenAI Error Schema

**Problem:** Return errors in a format compatible with OpenAI client libraries.

**OpenAI Error Format:**
```json
{
  "error": {
    "message": "The model 'gpt-5' does not exist",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

**Standard Error Types:**
- `invalid_request_error` - 400/422 validation errors
- `authentication_error` - 401 auth failures
- `rate_limit_error` - 429 rate limiting
- `server_error` - 500 internal errors
- `model_not_found` - 404 model not found

**Implementation Pattern:**
```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class OpenAIError(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None

class OpenAIErrorResponse(BaseModel):
    error: OpenAIError

@app.exception_handler(HTTPException)
async def openai_error_handler(request, exc):
    error_type = {
        400: "invalid_request_error",
        401: "authentication_error",
        404: "invalid_request_error",
        422: "invalid_request_error",
        429: "rate_limit_error",
        500: "server_error",
        503: "server_error",
        504: "server_error",
    }.get(exc.status_code, "server_error")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=OpenAIErrorResponse(
            error=OpenAIError(
                message=exc.detail,
                type=error_type,
                code=None
            )
        ).model_dump()
    )
```

**Applicability to npu-proxy:** ⚠️ Partially implemented - uses FastAPI default errors. Could add OpenAI error wrapper for better compatibility.

---

## 4. Header Conventions

### Pattern: Rate Limiting Headers

**Problem:** Communicate rate limit status to clients.

**Standard Headers (OpenAI-compatible):**
```
x-ratelimit-limit-requests: 10000
x-ratelimit-limit-tokens: 200000
x-ratelimit-remaining-requests: 9999
x-ratelimit-remaining-tokens: 199500
x-ratelimit-reset-requests: 1s
x-ratelimit-reset-tokens: 6m0s
```

**Implementation Pattern:**
```python
from fastapi import Response

@app.middleware("http")
async def rate_limit_headers(request, call_next):
    response = await call_next(request)
    
    # Add rate limiting headers
    response.headers["x-ratelimit-limit-requests"] = str(10000)
    response.headers["x-ratelimit-remaining-requests"] = str(9999)
    
    return response
```

**Applicability to npu-proxy:** 📋 Not implemented - could add for future rate limiting

---

### Pattern: Request ID Headers

**Problem:** Enable request tracing for debugging and logging.

**Standard Headers:**
```
x-request-id: req_abc123
```

**Implementation Pattern:**
```python
import uuid

@app.middleware("http")
async def request_id_middleware(request, call_next):
    request_id = request.headers.get("x-request-id", f"req_{uuid.uuid4().hex[:12]}")
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response
```

**Applicability to npu-proxy:** 📋 Not implemented - recommended for debugging

---

### Pattern: Custom Routing Headers (npu-proxy specific)

**Problem:** Expose internal routing decisions to clients.

**Implementation (existing in npu-proxy):**
```python
def add_routing_headers(response: Response, routing_result: RoutingResult) -> None:
    """Add routing information to response headers."""
    response.headers["X-NPU-Proxy-Device"] = routing_result.device
    response.headers["X-NPU-Proxy-Route-Reason"] = routing_result.reason
    response.headers["X-NPU-Proxy-Token-Count"] = str(routing_result.token_count)
```

**Headers Added:**
```
X-NPU-Proxy-Device: npu
X-NPU-Proxy-Route-Reason: within_context_limit
X-NPU-Proxy-Token-Count: 128
```

**Applicability to npu-proxy:** ✅ Already implemented

---

## 5. Model Aliasing Patterns

### Pattern: Ollama-style Model Name Resolution

**Problem:** Map user-friendly model names (e.g., "llama3.2") to HuggingFace repo IDs.

**Ollama Model Naming Convention:**
- `model:tag` format (e.g., `llama3.2:latest`, `mistral:7b-instruct`)
- Tag defaults to `latest` if not specified
- Namespace support: `example/model:tag`

**Implementation Pattern (from npu-proxy):**
```python
# npu_proxy/models/mapper.py
MODEL_ALIASES = {
    "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama"),
    "llama2": ("meta-llama/Llama-2-7b-chat-hf", "llama2"),
    "phi-2": ("microsoft/phi-2", "phi-2"),
    # OpenVINO-optimized repos
    "tinyllama-ov": ("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", "tinyllama"),
}

def resolve_model_repo(model_name: str) -> tuple[str, str] | None:
    """Resolve model name to (huggingface_repo, local_name)"""
    # Check direct alias
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    
    # Check if it's already a HuggingFace repo ID
    if "/" in model_name:
        return (model_name, model_name.split("/")[-1])
    
    return None
```

**Applicability to npu-proxy:** ✅ Already implemented in mapper.py

---

### Pattern: Tag Parsing

**Problem:** Handle model:tag format properly.

**Implementation:**
```python
def parse_model_name(model: str) -> tuple[str, str]:
    """Parse model:tag into (model, tag)"""
    if ":" in model:
        parts = model.rsplit(":", 1)
        return parts[0], parts[1]
    return model, "latest"
```

**Applicability to npu-proxy:** 📋 Could enhance current implementation

---

## 6. Parameter Mapping

### Pattern: OpenAI to Ollama Parameter Mapping

**Problem:** Map OpenAI API parameters to native inference parameters.

**Parameter Mapping Table:**

| OpenAI Parameter | Ollama Parameter | OpenVINO Parameter | Notes |
|------------------|------------------|-------------------|-------|
| `temperature` | `temperature` | `temperature` | 0.0-2.0 range |
| `max_tokens` | `num_predict` | `max_new_tokens` | |
| `top_p` | `top_p` | `top_p` | |
| `presence_penalty` | `presence_penalty` | N/A | May need custom impl |
| `frequency_penalty` | `frequency_penalty` | `repetition_penalty` | Different scale |
| `stop` | `stop` | `stop_strings` | Array of strings |
| `n` | N/A | N/A | Multiple completions |
| `stream` | `stream` | N/A | Handled at API layer |

**Implementation (from npu-proxy):**
```python
# npu_proxy/models/parameter_mapper.py
def map_parameters(options: dict) -> dict:
    """Map Ollama-style options to OpenVINO inference parameters"""
    mapped = {}
    
    if "num_predict" in options:
        mapped["max_new_tokens"] = options["num_predict"]
    if "temperature" in options:
        mapped["temperature"] = options["temperature"]
    if "top_p" in options:
        mapped["top_p"] = options["top_p"]
    if "repeat_penalty" in options:
        mapped["repetition_penalty"] = options["repeat_penalty"]
    
    return mapped
```

**Pipeline Pattern (npu-proxy):**
```python
# In endpoint handler:
user_options = request.options or {}
full_options = merge_with_defaults(user_options)  # Apply model defaults
mapped_options = map_parameters(full_options)      # Convert to native format
```

**Applicability to npu-proxy:** ✅ Fully implemented with pipeline pattern

---

### Pattern: Ollama Default Parameters

**Problem:** Apply sensible defaults when parameters aren't specified.

**Ollama Defaults:**
```python
DEFAULT_OPTIONS = {
    "num_keep": 5,
    "seed": -1,  # Random
    "num_predict": 128,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.0,
    "typical_p": 1.0,
    "temperature": 0.8,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "num_ctx": 2048,
}
```

**Implementation:**
```python
def merge_with_defaults(options: dict) -> dict:
    """Merge user options with sensible defaults"""
    return {**DEFAULT_OPTIONS, **options}
```

**Applicability to npu-proxy:** ✅ Implemented in ollama_defaults.py

---

## 7. Additional Patterns

### Pattern: Health Check Endpoints

**Problem:** Provide liveness/readiness probes for orchestration.

**Implementation (npu-proxy):**
```python
@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/health/ready")
async def readiness():
    # Check if model is loaded
    if is_model_loaded():
        return {"status": "ready", "model_loaded": True}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "model_loaded": False}
    )
```

**Applicability to npu-proxy:** ✅ Implemented in health.py

---

### Pattern: Model List Endpoint

**OpenAI Format:**
```json
{
  "object": "list",
  "data": [
    {"id": "gpt-4", "object": "model", "created": 1687882411, "owned_by": "openai"},
    {"id": "gpt-3.5-turbo", "object": "model", "created": 1677610602, "owned_by": "openai"}
  ]
}
```

**Ollama Format:**
```json
{
  "models": [
    {
      "name": "llama3.2:latest",
      "model": "llama3.2:latest",
      "size": 2019393189,
      "digest": "a80c4f17acd...",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "3B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

**Applicability to npu-proxy:** ✅ Both formats implemented

---

### Pattern: Metrics Endpoint

**Problem:** Expose Prometheus-compatible metrics.

**Implementation:**
```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['endpoint', 'status'])
INFERENCE_TIME = Histogram('inference_duration_seconds', 'Inference time')

@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

**Applicability to npu-proxy:** ✅ Implemented in metrics.py

---

## Summary: npu-proxy Implementation Status

| Pattern | Status | Location |
|---------|--------|----------|
| Request Validation | ✅ Implemented | chat.py, ollama.py |
| OpenAI SSE Streaming | ✅ Implemented | chat.py |
| Ollama NDJSON Streaming | ✅ Implemented | ollama.py |
| OpenAI Error Format | ⚠️ Partial | Could add wrapper |
| Rate Limit Headers | 📋 Not Implemented | Future work |
| Request ID Headers | 📋 Not Implemented | Recommended |
| Routing Headers | ✅ Implemented | Custom extension |
| Model Aliasing | ✅ Implemented | mapper.py |
| Parameter Mapping | ✅ Implemented | parameter_mapper.py |
| Health Checks | ✅ Implemented | health.py |
| Metrics | ✅ Implemented | metrics.py |

---

## Recommendations for npu-proxy

### High Priority
1. **Add OpenAI Error Wrapper** - Custom exception handler to return OpenAI-compatible error format
2. **Add Request ID Middleware** - For debugging and tracing

### Medium Priority
3. **Add Rate Limiting Headers** - Even if not enforcing limits, expose header structure
4. **Enhanced Tag Parsing** - Support full `model:tag` format

### Low Priority
5. **Tool/Function Calling Support** - Ollama supports this, may be useful for agents
6. **Structured Output (JSON mode)** - Ollama's `format: "json"` parameter

---

## References

1. Ollama API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
2. OpenAI API Reference: https://platform.openai.com/docs/api-reference
3. vLLM OpenAI Server: https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai
4. LocalAI: https://github.com/mudler/LocalAI
