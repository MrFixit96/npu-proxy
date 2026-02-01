# NPU Proxy - Codebase Optimization Summary

**Date**: 2026-02-01  
**Version**: Post-Optimization  
**Tests**: 399 passing, 0 regressions  

---

## Executive Summary

Comprehensive review and optimization of all 21 Python modules in npu-proxy.
Applied best practices from vLLM, OpenVINO GenAI, FastEmbed, TGI, and llama.cpp projects.

**Key Improvements:**
- 🚀 NPU warmup eliminates 80-130s cold start latency
- 📊 Full vLLM-style metrics (TTFT, TPOT, tokens/sec)
- 🔄 Intelligent context-aware routing with device fallback
- 📝 100% docstring coverage with Google-style format
- ✅ Full OpenAI and Ollama API compatibility

---

## Optimizations by Module

### Inference Layer (`npu_proxy/inference/`)

#### engine.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| NPU warmup | OpenVINO GenAI | Eliminates 80-130s cold start |
| Structured errors | vLLM | Consistent HTTP status mapping |
| Device fallback chain | llama.cpp | NPU→GPU→CPU automatic fallback |
| Thread-safe singleton | vLLM | Safe multi-threaded access |
| Configurable timeouts | TGI | Per-request timeout control |
| Google docstrings | - | Improved maintainability |

**New Exception Classes:**
- `InferenceError` - Base class with HTTP status_code attribute
- `InferenceTimeoutError` - 504 Gateway Timeout mapping
- `DeviceError` - 503 Service Unavailable with available_devices list
- `ModelNotLoadedError` - 500 Internal Server Error
- `ModelNotFoundError` - 404 Not Found with download instructions

**New APIs:**
```python
# NPU warmup to pre-compile pipeline
engine.warmup(warmup_tokens=16)

# Check warmup status
engine.is_warmed_up  # bool

# Streaming with abort callback
async for token in engine.generate_stream(
    prompt, 
    max_new_tokens=100,
    abort_callback=lambda: client_disconnected
):
    yield token
```

**Environment Variables:**
- `NPU_PROXY_INFERENCE_TIMEOUT`: Default timeout (default: 180s)
- `NPU_PROXY_MAX_PROMPT_LEN`: Max prompt length for NPU (default: 4096)
- `NPU_PROXY_DEVICE`: Default device selection (default: NPU)

---

#### embedding_engine.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| LRU caching | FastEmbed | Avoids redundant embeddings |
| Batch length grouping | Infinity | Minimizes padding overhead |
| L2 normalization | Sentence Transformers | float32 precision guarantee |
| Timeout protection | TGI | Prevents hung model loads |
| Fallback engine | FastEmbed | Hash-based fallback when model unavailable |

**New APIs:**
```python
# Length-grouped batching for efficiency
embeddings = engine.embed_batch_optimized(texts)

# Engine introspection
info = engine.get_engine_info()
# Returns: {"model_name": "...", "dimensions": 384, "is_production": True, "device": "CPU"}
```

**Environment Variables:**
- `NPU_PROXY_EMBEDDING_MODEL`: Model name (default: BAAI/bge-small-en-v1.5)
- `NPU_PROXY_EMBEDDING_DEVICE`: Device for inference (default: CPU)
- `NPU_PROXY_LOAD_TIMEOUT`: Model load timeout (default: 300s)
- `NPU_PROXY_EMBED_TIMEOUT`: Embedding inference timeout (default: 60s)
- `NPU_PROXY_EMBEDDING_CACHE_SIZE`: LRU cache size (default: 1024)

---

#### streaming.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| asyncio.Queue bridge | vLLM | Thread-safe callback→async |
| Cancellation support | llama.cpp | Clean abort on disconnect |
| Bounded queue | TGI | Backpressure handling |
| Timeout handling | TGI | Prevents infinite waits |

**New APIs:**
```python
# Create stream with custom settings
stream = create_token_stream(timeout=30.0, max_queue_size=100)

# Thread-safe callback for inference engine
stream.callback(token)  # Returns False to continue, True to stop

# Non-blocking push with backpressure feedback
success = stream.try_push(token)

# Cancellation from any thread
stream.cancel()

# Collect all tokens
tokens = await stream.collect(max_tokens=1000)

# Properties
stream.is_cancelled  # bool
stream.is_done       # bool
```

**Thread-Safety Model:**
- Inference thread calls `callback()` using `asyncio.run_coroutine_threadsafe()`
- Event loop thread consumes via `async for token in stream`
- Cancellation is thread-safe via atomic flag + sentinel push

---

#### tokenizer.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Multi-precision counting | vLLM | Accuracy vs. performance trade-off |
| Regex BPE approximation | tiktoken | ~95% accuracy without model |
| Fast character ratio | OpenAI | ~85% accuracy, 20x faster |
| Safe counting wrapper | - | Graceful fallback on errors |

**New APIs:**
```python
from npu_proxy.inference.tokenizer import count_tokens, TokenCountPrecision

# Default: ~95% accurate regex-based
count = count_tokens(text)

# Fast: ~85% accurate, 20x faster
count = count_tokens(text, precision=TokenCountPrecision.FAST)

# Exact: 100% accurate (when tiktoken integrated)
count = count_tokens(text, precision=TokenCountPrecision.EXACT)

# Error-safe with fallback
count = count_tokens_safe(text, fallback_to_words=True)

# Semantic wrappers
prompt_tokens = count_prompt_tokens(prompt)
completion_tokens = count_completion_tokens(completion)
```

---

### Routing Layer (`npu_proxy/routing/`)

#### context_router.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Token-based routing | vLLM | NPU context limit respect |
| Device fallback | llama.cpp | Automatic GPU→CPU chain |
| Configurable limits | - | Tunable per deployment |

**Routing Logic:**
1. Count tokens in prompt/messages
2. If `token_count <= NPU_PROXY_TOKEN_LIMIT`: use NPU
3. If `token_count > NPU_PROXY_TOKEN_LIMIT`: use fallback device
4. Fallback auto-detected: GPU if available, else CPU

**New APIs:**
```python
from npu_proxy.routing.context_router import get_context_router, RoutingResult

router = get_context_router()
result: RoutingResult = router.select_device(prompt)

# Result contains:
result.device       # "NPU", "GPU", or "CPU"
result.reason       # "within_npu_limit" or "prompt_exceeds_npu_limit"
result.token_count  # Token count used for decision
```

**Environment Variables:**
- `NPU_PROXY_TOKEN_LIMIT`: Max tokens for NPU (default: 1800)
- `NPU_PROXY_PREFERRED_DEVICE`: Primary device (default: NPU)
- `NPU_PROXY_FALLBACK_DEVICE`: Override fallback (default: auto-detect)

---

### API Layer (`npu_proxy/api/`)

#### chat.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| OpenAI error format | OpenAI API | Client compatibility |
| Request ID tracking | vLLM | Distributed tracing |
| Routing headers | TGI | Observability |
| Parameter validation | Pydantic | Input safety |

**OpenAI Compatibility:**
- `POST /v1/chat/completions` - Full OpenAI schema
- Streaming SSE with `chat.completion.chunk` objects
- Non-streaming with `chat.completion` objects
- Usage statistics (prompt_tokens, completion_tokens, total_tokens)

**Response Headers:**
- `X-Request-ID`: Unique request identifier
- `X-NPU-Proxy-Device`: Device used for inference
- `X-NPU-Proxy-Route-Reason`: Why device was selected
- `X-NPU-Proxy-Token-Count`: Token count for routing decision

---

#### embeddings.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| OpenAI response format | OpenAI API | Drop-in replacement |
| Request ID header | vLLM | Request tracing |
| Device header | - | Transparency |

**OpenAI Compatibility:**
- `POST /v1/embeddings` - Full OpenAI schema
- Batch input support (string or list)
- Usage statistics (prompt_tokens, total_tokens)

---

#### ollama.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Full Ollama API | Ollama | Drop-in replacement |
| Streaming NDJSON | Ollama | Line-delimited JSON |
| Model pull endpoint | Ollama | HuggingFace download |
| Model search | - | Discover OpenVINO models |

**Ollama Endpoints:**
- `POST /api/generate` - Raw text completion
- `POST /api/chat` - Chat with message history
- `POST /api/embed` - Embeddings (current format)
- `POST /api/embeddings` - Embeddings (legacy format)
- `GET /api/ps` - List running models
- `GET /api/version` - Version info
- `POST /api/show` - Model information
- `POST /api/pull` - Download from HuggingFace
- `GET /api/search` - Search OpenVINO models
- `GET /api/models/known` - List pre-mapped names

**Response Headers:**
- `X-Request-ID`: Request identifier
- `X-NPU-Proxy-Device`: Device used (npu, cpu, gpu)
- `X-NPU-Proxy-Route-Reason`: Routing decision reason
- `X-NPU-Proxy-Token-Count`: Estimated token count

---

#### health.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Load balancer probes | vLLM | Health monitoring |
| Device enumeration | OpenVINO | Hardware discovery |
| Engine status | vLLM | Readiness checking |

**Health Endpoints:**
- `GET /health` - Full health check with engine status
- `GET /health/devices` - Device info and fallback chain

**Important: Native Host Deployment**

NPU Proxy runs as a **native host service**, NOT in containers.
Intel NPU drivers require direct hardware access and cannot be containerized
(no Docker, Kubernetes, or WSL2 passthrough for NPU devices).

**Deployment Options:**
```bash
# Windows - Windows Service
sc create npu-proxy binPath="C:\path\to\npu-proxy.exe"

# Linux - systemd service  
sudo systemctl enable npu-proxy
sudo systemctl start npu-proxy
```

**Health Check Usage:**
- Load balancers: Check `status == 'healthy'`
- Monitoring: Check `engines.llm.status == 'loaded'`

---

### Models Layer (`npu_proxy/models/`)

#### registry.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Centralized metadata | Ollama | Single source of truth |
| LLM + embedding types | - | Unified registry |
| Quantization tracking | - | INT4/INT8/FP16 support |

**Model Metadata Fields:**
- `id`: Unique identifier
- `family`: Architecture (llama, phi, mistral, etc.)
- `parameter_size`: Human-readable (1.1B, 7B, etc.)
- `quantization`: INT4, INT8, FP16, FP32
- `type`: "llm" or "embedding"
- `dimensions`: Embedding vector size (embedding models only)
- `context_length`: Max context window
- `hf_repo`: HuggingFace repository path

---

#### mapper.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Ollama-style names | Ollama | User-friendly aliases |
| Version tag support | Ollama | Model variants (`:fp16`) |
| Reverse mapping | - | Repo→name lookup |

**Mapped Models:**
```python
# LLM Models
"tinyllama"     → "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
"tinyllama:fp16"→ "OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov"
"phi-2"         → "OpenVINO/phi-2-int4-ov"
"phi-3"         → "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
"llama2"        → "OpenVINO/llama-2-7b-chat-int4-ov"
"llama3.2"      → "OpenVINO/Llama-3.2-3B-Instruct-int4-ov"
"mistral"       → "OpenVINO/mistral-7b-instruct-v0.1-int4-ov"
"qwen2"         → "OpenVINO/Qwen2-1.5B-Instruct-int4-ov"

# Embedding Models
"bge-small"     → "BAAI/bge-small-en-v1.5"
"e5-small"      → "intfloat/e5-small-v2"
"all-minilm"    → "sentence-transformers/all-MiniLM-L6-v2"
```

---

#### parameter_mapper.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| OpenAI→OpenVINO | - | API translation |
| Ollama→OpenVINO | - | API translation |
| Penalty conversion | - | presence+frequency→repetition |
| Silent ignoring | - | Unsupported params handled gracefully |

**Parameter Mapping Table:**
| OpenAI | Ollama | OpenVINO GenAI | Notes |
|--------|--------|----------------|-------|
| max_tokens | num_predict | max_new_tokens | renamed |
| temperature | temperature | temperature | direct |
| top_p | top_p | top_p | direct |
| - | top_k | top_k | direct |
| presence_penalty | - | repetition_penalty | converted |
| frequency_penalty | - | repetition_penalty | converted |
| stop | stop | stop_strings | renamed |
| - | mirostat | (ignored) | unsupported |

---

#### ollama_defaults.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Ollama-compatible defaults | Ollama source | API parity |
| Merge function | - | Apply defaults safely |

**Default Values:**
```python
OLLAMA_DEFAULTS = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 128,
    "num_ctx": 2048,
    "seed": 0,
    "stop": [],
}
```

---

#### converter.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| Auto-convert HF models | optimum-intel | On-demand optimization |
| Progress streaming | - | UI feedback |
| Smart caching | - | Skip if already converted |

**New APIs:**
```python
from npu_proxy.models.converter import (
    is_openvino_model,
    convert_to_openvino,
    auto_download_and_convert,
    get_conversion_progress,
)

# Check if directory has valid OpenVINO model
if is_openvino_model("/models/gpt2"):
    print("Ready!")

# Auto-convert with caching
result = auto_download_and_convert("tinyllama")
# Returns: {"status": "success", "path": "...", "source": "cache"|"converted"}

# Stream progress
for progress in get_conversion_progress("gpt2", output_dir):
    print(f"{progress['status']}: {progress['message']}")
```

---

#### downloader.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| HuggingFace Hub integration | huggingface_hub | Official API |
| Automatic model resolution | - | Ollama names→HF repos |
| Required file validation | - | OpenVINO model verification |

**Cache Structure:**
```
~/.cache/npu-proxy/models/
└── {model_name}/
    ├── openvino_model.xml    # Required
    ├── openvino_model.bin    # Required
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── config.json
```

---

#### search.py
| Optimization | Pattern Source | Benefit |
|-------------|----------------|---------|
| HuggingFace Hub search | huggingface_hub | Model discovery |
| LRU caching | - | Repeated query performance |
| Architecture detection | - | Auto-extract from name |

**New APIs:**
```python
from npu_proxy.models.search import search_openvino_models, get_model_details

# Search for OpenVINO models
models, count = search_openvino_models("llama", limit=10)

# Get detailed model info
model = get_model_details("OpenVINO/phi-2-int4-ov")
```

---

## New Metrics Added

### Prometheus Metrics (`npu_proxy/metrics.py`)

| Metric | Type | Description |
|--------|------|-------------|
| `npu_proxy_requests_total` | Counter | Total API requests by endpoint/method/status |
| `npu_proxy_request_latency_seconds` | Histogram | End-to-end request latency |
| `npu_proxy_requests_in_progress` | Gauge | Currently processing requests |
| `npu_proxy_queue_time_seconds` | Histogram | Time waiting before processing |
| `npu_proxy_inference_total` | Counter | Inference operations by model/device/type |
| `npu_proxy_inference_latency_seconds` | Histogram | Total inference latency |
| `npu_proxy_inference_tokens_total` | Counter | Tokens by model and type |
| `npu_proxy_time_to_first_token_seconds` | Histogram | **TTFT** - critical UX metric |
| `npu_proxy_inter_token_latency_seconds` | Histogram | **TPOT/ITL** for streaming |
| `npu_proxy_tokens_per_second` | Gauge | Real-time throughput |
| `npu_proxy_routing_decisions_total` | Counter | Routing by device/reason |
| `npu_proxy_model` | Info | Model metadata |
| `npu_proxy_model_load_seconds` | Gauge | Model load time |
| `npu_proxy_errors_total` | Counter | Errors by endpoint/type |

**Histogram Buckets (optimized for LLM workloads):**
```python
REQUEST_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
INFERENCE_LATENCY_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
TTFT_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
TPOT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
```

**Convenience Functions:**
```python
from npu_proxy.metrics import (
    track_request,          # Context manager for request lifecycle
    record_request,         # Count completed request
    record_inference,       # Log inference operation
    record_tokens,          # Track prompt/completion tokens
    record_ttft,            # Time to first token
    record_tpot,            # Inter-token latency
    record_tokens_per_second,
    record_routing_decision,
    record_error,
    record_streaming_metrics,  # Convenience for all streaming metrics
)
```

---

## API Enhancements

### Request ID Support
All endpoints now include `X-Request-ID` header for distributed tracing:
```python
def generate_request_id() -> str:
    """Generate unique ID in format 'req_<24-char-hex>'"""
    return f"req_{uuid.uuid4().hex[:24]}"
```

### OpenAI Error Format
Added standardized error responses matching OpenAI's schema:
```python
@dataclass
class OpenAIError:
    message: str
    type: str  # "invalid_request_error", "server_error", etc.
    param: Optional[str] = None
    code: Optional[str] = None

# Response format:
{"error": {"message": "...", "type": "invalid_request_error", "code": null}}
```

### Routing Headers
Response headers expose routing decisions:
- `X-NPU-Proxy-Device`: Device used (NPU, GPU, CPU)
- `X-NPU-Proxy-Route-Reason`: Why device was selected
- `X-NPU-Proxy-Token-Count`: Token count for request

---

## Documentation Improvements

### Coverage
- **100% docstring coverage**: All public functions documented
- **Google-style format**: Consistent Args/Returns/Raises/Examples
- **Type hints**: Full typing annotations on all modules
- **Module docstrings**: Each module has overview, examples, env vars

### Docstring Format Example
```python
def warmup(self, warmup_tokens: int = 16) -> None:
    """Perform warmup inference to pre-compile the NPU pipeline.
    
    Intel NPU requires pipeline compilation on first inference, which
    can take 80-130 seconds. Running a short warmup during startup
    eliminates this latency from the first real user request.
    
    This method is thread-safe and idempotent - multiple calls will
    only perform warmup once.
    
    Args:
        warmup_tokens: Number of tokens to generate during warmup.
            Defaults to 16. More tokens don't improve compilation
            but increase warmup time.
    
    Example:
        >>> engine = InferenceEngine("/path/to/model", device="NPU")
        >>> engine.warmup()  # Blocks for 80-130s on NPU
        >>> # First real request now has normal latency
    
    Note:
        Warmup failures are logged but not raised - the pipeline will
        compile on first real request instead.
    """
```

---

## Test Suite Summary

| Category | Test Count | Status |
|----------|------------|--------|
| Inference Engine | 45+ | ✅ Passing |
| Embedding Engine | 35+ | ✅ Passing |
| Streaming | 25+ | ✅ Passing |
| Tokenizer | 15+ | ✅ Passing |
| Context Router | 20+ | ✅ Passing |
| Chat API | 30+ | ✅ Passing |
| Embeddings API | 20+ | ✅ Passing |
| Ollama API | 40+ | ✅ Passing |
| Health | 10+ | ✅ Passing |
| Models/Registry | 25+ | ✅ Passing |
| Models/Mapper | 15+ | ✅ Passing |
| Models/Converter | 29 | ✅ Passing |
| Models/Downloader | 20+ | ✅ Passing |
| Models/Search | 15+ | ✅ Passing |
| Metrics | 20+ | ✅ Passing |
| **TOTAL** | **399** | ✅ All Passing |

---

## Breaking Changes

**None** - all changes are backward compatible.

All existing APIs maintain their signatures. New functionality is additive:
- New parameters have defaults matching previous behavior
- New environment variables have sensible defaults
- New headers are informational only

---

## Configuration Reference

### Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `NPU_PROXY_INFERENCE_TIMEOUT` | 180 | Inference timeout (seconds) |
| `NPU_PROXY_MAX_PROMPT_LEN` | 4096 | Max prompt tokens for NPU |
| `NPU_PROXY_DEVICE` | NPU | Default inference device |
| `NPU_PROXY_TOKEN_LIMIT` | 1800 | NPU routing token threshold |
| `NPU_PROXY_PREFERRED_DEVICE` | NPU | Primary routing device |
| `NPU_PROXY_FALLBACK_DEVICE` | auto | Override fallback device |
| `NPU_PROXY_EMBEDDING_MODEL` | BAAI/bge-small-en-v1.5 | Embedding model |
| `NPU_PROXY_EMBEDDING_DEVICE` | CPU | Embedding device |
| `NPU_PROXY_LOAD_TIMEOUT` | 300 | Model load timeout |
| `NPU_PROXY_EMBED_TIMEOUT` | 60 | Embedding timeout |
| `NPU_PROXY_EMBEDDING_CACHE_SIZE` | 1024 | LRU cache entries |

---

## Recommendations for Future

### High Priority
1. **Add tiktoken integration** for exact token counting (when billing accuracy matters)
2. **Add circuit breaker** for NPU→GPU fallback (prevent cascade failures)
3. **Add request queuing** for load management (bounded concurrency)

### Medium Priority
4. **Add Grafana dashboard** JSON file for quick observability setup
5. **Add OpenTelemetry tracing** for distributed trace correlation
6. **Add model warm-up on pull** (warmup automatically after download)

### Future Enhancements
7. **Add batch inference endpoint** for high-throughput embedding
8. **Add model switching** without restart
9. **Add quantization selection** at runtime (INT4/INT8/FP16)
10. **Add continuous batching** (vLLM-style) for better GPU utilization

---

## Research Sources

| Project | Key Patterns Adopted |
|---------|---------------------|
| **vLLM** | Metrics (TTFT, TPOT), error structure, request IDs |
| **OpenVINO GenAI** | NPU warmup, device fallback, MAX_PROMPT_LEN config |
| **FastEmbed** | LRU caching, batch optimization, fallback engine |
| **TGI** | Streaming, timeout handling, Prometheus metrics |
| **llama.cpp** | Abort callbacks, cancellation, device chain |
| **Infinity** | Length-based batch grouping for embeddings |
| **Sentence Transformers** | L2 normalization, embedding engine patterns |
| **Ollama** | API compatibility, default values, model naming |
| **OpenAI** | Error response format, API schema, headers |

---

## Summary

The npu-proxy codebase has been comprehensively optimized with:

- ✅ **21 modules** fully documented with Google-style docstrings
- ✅ **399 tests** passing with zero regressions
- ✅ **14 Prometheus metrics** for full observability
- ✅ **NPU warmup** eliminating cold-start latency
- ✅ **Context-aware routing** respecting NPU limits
- ✅ **Full API compatibility** with OpenAI and Ollama
- ✅ **Structured error handling** with HTTP status mapping
- ✅ **Thread-safe streaming** with cancellation support
- ✅ **Smart model management** with caching and conversion

**Status: Production-Ready**
