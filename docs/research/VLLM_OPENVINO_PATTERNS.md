# LLM Inference Engine Best Practices Research

**Research Date:** January 2026  
**Target Project:** npu-proxy (OpenVINO-based NPU inference proxy)  
**Sources:** OpenVINO GenAI, vLLM, llama.cpp, Text Generation Inference (TGI)

---

## Executive Summary

This document captures best practices from leading LLM inference engines for application to the npu-proxy project. Key patterns include thread-safe singleton model management, structured generation configuration, memory lifecycle patterns, and timeout handling strategies.

---

## 1. Thread Safety Patterns

### Pattern: Double-Checked Locking with Global Engine

**Problem:** Multiple concurrent requests need access to a single model instance without race conditions during initialization or usage.

**Source:** llama.cpp, OpenVINO GenAI, npu-proxy current implementation

**Code Example (llama.cpp style):**
```c
// Thread safety achieved through context separation
// Model is loaded once, contexts are created per-request or reused
struct llama_model * model;       // Shared, immutable after load
struct llama_context * ctx;       // Per-session, mutable

// Initialize once
llama_backend_init();
model = llama_model_load_from_file(path, params);

// Per-request context (can be pooled)
ctx = llama_init_from_model(model, ctx_params);
```

**Code Example (Python - current npu-proxy pattern):**
```python
# Global engine instances (singleton pattern with thread safety)
_llm_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()

def get_llm_engine(model_path: str | None = None, device: str | None = None) -> InferenceEngine:
    global _llm_engine
    
    # Double-checked locking for thread safety
    if _llm_engine is None:
        with _engine_lock:
            if _llm_engine is None:
                _llm_engine = InferenceEngine(model_path, device)
    
    return _llm_engine
```

**Applicability to npu-proxy:**
- ✅ **Already implemented** in `engine.py` with `_engine_lock` and double-checked locking
- 🔧 **Enhancement:** Consider separating model loading from context/session management like llama.cpp
- 🔧 **Enhancement:** Add request-level sequence isolation (llama.cpp uses `llama_seq_id`)

### Pattern: Abort Callback for Cancellation

**Problem:** Need to cancel in-flight inference without corrupting state.

**Source:** llama.cpp

**Code Example:**
```c
struct llama_context_params {
    // Abort callback - if it returns true, execution of llama_decode() will be aborted
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
    // ...
};
```

**Applicability to npu-proxy:**
- 🔧 **Enhancement:** OpenVINO GenAI streamer callbacks can return `True` to stop generation
- 🔧 **Consider:** Exposing cancellation token pattern for HTTP request cancellation

---

## 2. Memory Management Patterns

### Pattern: Separate Model and Context Lifecycles

**Problem:** Models are expensive to load but contexts/sessions are cheap. Need efficient memory usage for concurrent requests.

**Source:** llama.cpp

**Code Example:**
```c
// Model: Load once, immutable, shared
struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);
void llama_model_free(struct llama_model * model);

// Context: Per-session, mutable, contains KV cache
struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);
void llama_free(struct llama_context * ctx);

// Memory operations (KV cache management)
void llama_memory_clear(llama_memory_t mem, bool data);
bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
void llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id);
```

**Applicability to npu-proxy:**
- 🔧 **Gap:** Current implementation uses single `LLMPipeline` per model
- 🔧 **Enhancement:** OpenVINO GenAI may support similar patterns - investigate `StatefulLLMPipeline` or continuous batching
- 📝 **Note:** NPU hardware may have limitations on concurrent contexts

### Pattern: Memory Mapped Model Loading

**Problem:** Large models shouldn't duplicate memory when loaded by multiple processes.

**Source:** llama.cpp

**Code Example:**
```c
struct llama_model_params {
    bool use_mmap;        // use mmap if possible
    bool use_mlock;       // force system to keep model in RAM
    bool use_direct_io;   // bypass OS cache
    bool no_alloc;        // only load metadata, simulate allocations
};
```

**Applicability to npu-proxy:**
- 📝 **Note:** OpenVINO handles model loading internally
- 🔧 **Consider:** Environment variable to control OpenVINO model caching behavior

### Pattern: State Serialization for Sessions

**Problem:** Need to save/restore conversation state for persistent sessions.

**Source:** llama.cpp

**Code Example:**
```c
// Full state save/load
size_t llama_state_get_size(struct llama_context * ctx);
size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size);
size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size);

// Session file operations
bool llama_state_save_file(struct llama_context * ctx, const char * path, ...);
bool llama_state_load_file(struct llama_context * ctx, const char * path, ...);

// Per-sequence state (for conversation isolation)
size_t llama_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id);
```

**Applicability to npu-proxy:**
- 🔧 **Future:** Consider session persistence for Claude Code long-running conversations
- 📝 **Note:** May require OpenVINO GenAI API investigation for KV cache serialization

---

## 3. Timeout Handling Patterns

### Pattern: ThreadPoolExecutor with Future Timeout

**Problem:** Inference can hang or take too long, need graceful timeout handling.

**Source:** npu-proxy current implementation

**Code Example:**
```python
def generate(self, prompt: str, timeout: int | None = None) -> str:
    if timeout is None:
        timeout = DEFAULT_INFERENCE_TIMEOUT  # 180s default
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self.pipeline.generate, prompt, config)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            logger.error(f"Inference timed out after {timeout}s")
            raise TimeoutError(f"Inference timed out after {timeout} seconds")
```

**Enhancement Pattern: Abort Callback Integration**
```python
def generate_with_abort(self, prompt: str, timeout: int, abort_event: threading.Event) -> str:
    """Generate with both timeout and abort capability"""
    
    def check_abort():
        return abort_event.is_set()
    
    # Configure OpenVINO with abort check in streamer
    def streamer(token: str) -> bool:
        return check_abort()  # Return True to abort
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self.pipeline.generate, prompt, config, streamer)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            abort_event.set()  # Signal abort to running inference
            raise
```

**Applicability to npu-proxy:**
- ✅ **Already implemented** basic timeout pattern
- 🔧 **Enhancement:** Add abort event for clean cancellation
- 🔧 **Enhancement:** Propagate timeout to OpenVINO's internal mechanisms

### Pattern: Progress Callback for Long Operations

**Problem:** Model loading can take a long time, need progress feedback.

**Source:** llama.cpp

**Code Example:**
```c
struct llama_model_params {
    // Called with progress value between 0.0 and 1.0
    // If returns true, model loading continues
    // If returns false, model loading is immediately aborted
    llama_progress_callback progress_callback;
    void * progress_callback_user_data;
};

typedef bool (*llama_progress_callback)(float progress, void * user_data);
```

**Applicability to npu-proxy:**
- 🔧 **Enhancement:** Add model loading progress callback for UI feedback
- 🔧 **Consider:** SSE endpoint for model loading status during `/api/pull`

---

## 4. Error Propagation Patterns

### Pattern: Structured Return Codes

**Problem:** Native inference functions need to communicate success, warnings, and errors to Python layer.

**Source:** llama.cpp

**Code Example:**
```c
// llama_decode return values:
//    0 - success
//    1 - could not find a KV slot for the batch (try reducing batch size)
//    2 - aborted (processed ubatches remain in context's memory)
//   -1 - invalid input batch
// < -1 - fatal error

int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch);

// Enum for explicit status
enum llama_params_fit_status {
    LLAMA_PARAMS_FIT_STATUS_SUCCESS = 0,
    LLAMA_PARAMS_FIT_STATUS_FAILURE = 1,
    LLAMA_PARAMS_FIT_STATUS_ERROR   = 2,
};
```

**Python Translation Pattern:**
```python
class InferenceResult:
    """Structured result from inference operations"""
    SUCCESS = 0
    PARTIAL = 1          # Some tokens generated before issue
    ABORTED = 2          # User-requested abort
    TIMEOUT = 3          # Timeout exceeded
    CONTEXT_FULL = 4     # KV cache exhausted
    INVALID_INPUT = -1
    FATAL_ERROR = -2
    
    def __init__(self, status: int, output: str = "", error: str = ""):
        self.status = status
        self.output = output
        self.error = error
        self.is_success = status == self.SUCCESS
        self.tokens_generated = 0
```

**Applicability to npu-proxy:**
- 🔧 **Enhancement:** Define structured error types for OpenVINO GenAI errors
- 🔧 **Enhancement:** Map OpenVINO exceptions to HTTP status codes consistently
- 🔧 **Consider:** Return partial output on timeout (currently all-or-nothing)

### Pattern: Device-Specific Error Handling

**Problem:** Different devices (NPU, GPU, CPU) fail in different ways.

**Source:** npu-proxy current implementation, OpenVINO

**Code Example:**
```python
def _load_model(self) -> None:
    """Load model with automatic fallback chain"""
    devices_to_try = [self.device]
    if self.fallback_device:
        devices_to_try.append(self.fallback_device)
    if "CPU" not in devices_to_try:
        devices_to_try.append("CPU")  # Always include CPU as last resort
    
    last_error = None
    for device in devices_to_try:
        try:
            if device == "NPU":
                config = {"MAX_PROMPT_LEN": DEFAULT_MAX_PROMPT_LEN}
                self.pipeline = ov_genai.LLMPipeline(str(self.model_path), device, config)
            else:
                self.pipeline = ov_genai.LLMPipeline(str(self.model_path), device)
            
            self.actual_device = device
            self.used_fallback = (device != self.device)
            return
        except Exception as e:
            logger.warning(f"Failed to load on {device}: {e}")
            last_error = e
            continue
    
    raise RuntimeError(f"Failed to load model on any device. Last error: {last_error}")
```

**Applicability to npu-proxy:**
- ✅ **Already implemented** device fallback chain
- 🔧 **Enhancement:** Categorize NPU-specific errors (driver issues, model incompatibility, memory)
- 🔧 **Enhancement:** Return device fallback info in API response headers

---

## 5. Warmup Strategies for NPU Compilation

### Pattern: First-Token Warmup

**Problem:** NPU compilation happens on first inference, causing cold start latency.

**Source:** OpenVINO GenAI, TGI, vLLM

**Code Example:**
```python
class InferenceEngine:
    def __init__(self, model_path: str, device: str = "NPU"):
        self._load_model()
        self._warmup()  # Trigger compilation
    
    def _warmup(self) -> None:
        """Trigger NPU graph compilation with minimal inference"""
        logger.info("Warming up NPU pipeline...")
        try:
            # Short generation to trigger compilation
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 1  # Minimal tokens
            config.do_sample = False   # Deterministic for consistent warmup
            
            warmup_prompt = "Hello"
            self.pipeline.generate(warmup_prompt, config)
            logger.info("NPU warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")
```

**Advanced Warmup Pattern (TGI-style):**
```python
def warmup_with_batch_sizes(self) -> dict:
    """Warmup with various sequence lengths to pre-compile all graph variants"""
    warmup_results = {}
    sequence_lengths = [32, 128, 512, 1024]  # Common sequence lengths
    
    for seq_len in sequence_lengths:
        try:
            prompt = "x " * (seq_len // 2)  # Approximate token count
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 1
            
            start = time.perf_counter()
            self.pipeline.generate(prompt, config)
            elapsed = time.perf_counter() - start
            
            warmup_results[seq_len] = {"status": "success", "time": elapsed}
        except Exception as e:
            warmup_results[seq_len] = {"status": "failed", "error": str(e)}
    
    return warmup_results
```

**Applicability to npu-proxy:**
- 🔧 **High Priority:** Add warmup call after model loading
- 🔧 **Enhancement:** Warmup on server startup, not first request
- 🔧 **Consider:** `/health` endpoint could trigger warmup if model not warmed
- 📝 **Note:** NPU MAX_PROMPT_LEN config already set (4096) for graph compilation

---

## 6. Generation Config Parameter Mapping

### Pattern: Structured Generation Configuration

**Problem:** Need consistent parameter naming between OpenAI, Ollama, and OpenVINO GenAI APIs.

**Source:** OpenVINO GenAI py_generation_config.cpp, llama.cpp

**OpenVINO GenAI GenerationConfig Fields:**
```python
# From OpenVINO GenAI source
GenerationConfig:
    # Sequence control
    max_length: int           # Total length (prompt + new tokens)
    max_new_tokens: int       # Max tokens to generate (takes priority)
    min_new_tokens: int       # Min before allowing EOS
    ignore_eos: bool          # Don't stop on EOS token
    eos_token_id: int         # EOS token ID
    stop_strings: set[str]    # Strings that stop generation
    stop_token_ids: set[int]  # Token IDs that stop generation
    echo: bool                # Include prompt in output
    
    # Penalties
    repetition_penalty: float # 1.0 = no penalty
    presence_penalty: float   # Reduces logprob if token appeared
    frequency_penalty: float  # Reduces logprob * occurrence count
    
    # Beam search
    num_beams: int            # 1 = greedy/sampling
    num_beam_groups: int      # For diverse beam search
    diversity_penalty: float
    length_penalty: float
    num_return_sequences: int
    no_repeat_ngram_size: int
    stop_criteria: StopCriteria  # EARLY, HEURISTIC, NEVER
    
    # Random sampling
    temperature: float        # > 0 enables sampling
    top_p: float             # Nucleus sampling
    top_k: int               # Top-k sampling
    do_sample: bool          # Enable multinomial sampling
    
    # Output control
    logprobs: int            # Top logprobs to return (0 = none)
    apply_chat_template: bool
```

**Parameter Mapping Table:**

| OpenAI | Ollama | OpenVINO GenAI | Notes |
|--------|--------|----------------|-------|
| `max_tokens` | `num_predict` | `max_new_tokens` | Direct map |
| `temperature` | `temperature` | `temperature` | Direct map |
| `top_p` | `top_p` | `top_p` | Direct map |
| `top_k` | `top_k` | `top_k` | Direct map (Ollama-specific) |
| `frequency_penalty` | - | `frequency_penalty` | OpenVINO supports |
| `presence_penalty` | - | `presence_penalty` | OpenVINO supports |
| - | `repeat_penalty` | `repetition_penalty` | Ollama-specific |
| `stop` | `stop` | `stop_strings` | Array of strings |
| `seed` | `seed` | `rng_seed` | For reproducibility |
| `logprobs` | - | `logprobs` | OpenVINO supports |
| `n` | - | `num_return_sequences` | Multiple sequences |
| - | `mirostat` | ❌ Not supported | Log warning |
| - | `mirostat_eta` | ❌ Not supported | Log warning |
| - | `mirostat_tau` | ❌ Not supported | Log warning |
| - | `min_p` | ❌ Not supported | Log warning |
| - | `typical_p` | ❌ Not supported | Log warning |
| - | `tfs_z` | ❌ Not supported | Log warning |

**Code Example (Parameter Mapper):**
```python
class GenerationConfigMapper:
    """Map API parameters to OpenVINO GenAI GenerationConfig"""
    
    # Parameters that are gracefully ignored (not supported)
    UNSUPPORTED_PARAMS = {"mirostat", "mirostat_eta", "mirostat_tau", "min_p", "typical_p", "tfs_z"}
    
    @classmethod
    def from_ollama(cls, params: dict) -> ov_genai.GenerationConfig:
        """Convert Ollama API parameters to GenerationConfig"""
        config = ov_genai.GenerationConfig()
        
        # Direct mappings
        if "num_predict" in params:
            config.max_new_tokens = params["num_predict"]
        if "temperature" in params:
            config.temperature = params["temperature"]
            config.do_sample = config.temperature > 0
        if "top_p" in params:
            config.top_p = params["top_p"]
        if "top_k" in params:
            config.top_k = params["top_k"]
        if "repeat_penalty" in params:
            config.repetition_penalty = params["repeat_penalty"]
        if "stop" in params:
            config.stop_strings = set(params["stop"]) if params["stop"] else set()
        if "seed" in params:
            config.rng_seed = params["seed"]
        
        # Warn about unsupported parameters
        unsupported_found = set(params.keys()) & cls.UNSUPPORTED_PARAMS
        if unsupported_found:
            logger.debug(f"Ignoring unsupported parameters: {unsupported_found}")
        
        return config
    
    @classmethod
    def from_openai(cls, params: dict) -> ov_genai.GenerationConfig:
        """Convert OpenAI API parameters to GenerationConfig"""
        config = ov_genai.GenerationConfig()
        
        if "max_tokens" in params:
            config.max_new_tokens = params["max_tokens"]
        if "temperature" in params:
            config.temperature = params["temperature"]
            config.do_sample = config.temperature > 0
        if "top_p" in params:
            config.top_p = params["top_p"]
        if "frequency_penalty" in params:
            config.frequency_penalty = params["frequency_penalty"]
        if "presence_penalty" in params:
            config.presence_penalty = params["presence_penalty"]
        if "stop" in params:
            config.stop_strings = set(params["stop"]) if params["stop"] else set()
        if "seed" in params:
            config.rng_seed = params["seed"]
        if "logprobs" in params:
            config.logprobs = params["logprobs"]
        if "n" in params:
            config.num_return_sequences = params["n"]
        
        return config
```

**Applicability to npu-proxy:**
- 🔧 **Enhancement:** Create centralized `GenerationConfigMapper` class
- 🔧 **Enhancement:** Log when unsupported parameters are ignored
- 🔧 **Enhancement:** Add OpenAI-to-OpenVINO penalty conversion (currently approximated)

---

## 7. Continuous Batching (Advanced)

### Pattern: Continuous Batching for Throughput

**Problem:** Maximize GPU/NPU utilization when serving multiple concurrent requests.

**Source:** vLLM, TGI, OpenVINO GenAI

**Key Concepts:**
- **PagedAttention (vLLM):** Manage KV cache like virtual memory pages
- **Continuous Batching:** Add new requests to ongoing batch without waiting for completion
- **Prefix Caching:** Cache common prefixes (system prompts) across requests

**vLLM Architecture:**
```
Input Requests → Scheduler → Model Executor → Output Manager
                    ↑              ↓
                    └─── KV Cache ←┘
                    (PagedAttention)
```

**OpenVINO GenAI Continuous Batching:**
```python
# OpenVINO GenAI supports continuous batching for serving
# Used in OpenVINO Model Server (OVMS) for LLM serving

# Features mentioned in OpenVINO GenAI:
# - Prefix caching
# - KVCache token eviction
# - Speculative decoding
# - Sparse attention (Tri-shape, XAttention)
```

**Applicability to npu-proxy:**
- 📝 **Note:** NPU typically supports 1 concurrent inference (hardware limitation)
- 🔧 **Future:** Consider request queuing with fair scheduling
- 🔧 **Future:** Investigate OpenVINO Model Server for production batching
- 📝 **Note:** vLLM patterns more applicable to GPU serving

---

## 8. Structured Output / Constrained Generation

### Pattern: Grammar-Constrained Generation

**Problem:** Need to ensure output follows specific format (JSON, XML, regex).

**Source:** OpenVINO GenAI, llama.cpp (grammar sampling)

**OpenVINO GenAI Structured Output:**
```python
# From py_generation_config.cpp
StructuredOutputConfig:
    json_schema: str      # JSON Schema constraint
    regex: str            # Regex constraint
    grammar: str          # EBNF grammar constraint
    structural_tags_config: StructuralTagsConfig  # Tag-based constraints
    compound_grammar: ...  # Complex grammar (Union/Concat)

# Example usage
config = ov_genai.GenerationConfig()
config.structured_output = StructuredOutputConfig(
    json_schema='{"type": "object", "properties": {"name": {"type": "string"}}}'
)
```

**Applicability to npu-proxy:**
- 🔧 **Future:** Add `response_format` support for OpenAI API compatibility
- 🔧 **Consider:** JSON mode for Claude Code tool use responses
- 📝 **Note:** Requires OpenVINO GenAI 2025.0+ for full structured output support

---

## 9. Speculative Decoding

### Pattern: Draft Model Acceleration

**Problem:** Improve inference latency using smaller draft model.

**Source:** OpenVINO GenAI, vLLM

**Concept:**
```
Draft Model (small, fast) → Generate N candidate tokens
                              ↓
Main Model (large, accurate) → Verify/Correct in parallel
                              ↓
                        Accept verified tokens
                        (~2x latency improvement)
```

**OpenVINO GenAI Support:**
```python
# OpenVINO GenAI mentions speculative decoding support
# Configuration via speculative decoding parameters
```

**Applicability to npu-proxy:**
- 📝 **Note:** May require model pair (draft + main)
- 🔧 **Future:** Investigate for NPU latency improvement
- 📝 **Note:** Memory constraints may limit on NPU

---

## Summary: Priority Recommendations for npu-proxy

### High Priority (Should Implement)

1. **NPU Warmup on Startup** - Add `_warmup()` call after model loading to eliminate cold start latency on first request

2. **Structured Error Types** - Define `InferenceError` hierarchy for consistent error handling and HTTP status mapping

3. **GenerationConfig Mapper** - Create centralized parameter mapping class for Ollama↔OpenVINO and OpenAI↔OpenVINO

4. **Abort Callback Integration** - Use streamer callback return value for clean inference cancellation on timeout/cancel

### Medium Priority (Nice to Have)

5. **Progress Callback for Model Loading** - Add progress reporting for `/api/pull` and model loading

6. **Unsupported Parameter Logging** - Log debug message when Ollama parameters are ignored

7. **Partial Output on Timeout** - Return generated tokens even if timeout occurs (currently all-or-nothing)

8. **Device Info in Headers** - Return actual device used in response headers (`X-NPU-Proxy-Device`)

### Future Considerations

9. **Session State Persistence** - For long-running Claude Code conversations

10. **Structured Output Support** - JSON mode for tool use responses

11. **Request Queuing** - Fair scheduling for concurrent requests

12. **Health Endpoint Warmup** - Trigger warmup if model not warmed on health check

---

## References

- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) - Intel's LLM inference library
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving with PagedAttention
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient C/C++ LLM inference
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - HuggingFace's serving toolkit
- [OpenVINO Model Server](https://docs.openvino.ai/2025/openvino-workflow/model-server/ovms_what_is_openvino_model_server.html) - Production LLM serving

---

*Document generated from web research of source repositories*
