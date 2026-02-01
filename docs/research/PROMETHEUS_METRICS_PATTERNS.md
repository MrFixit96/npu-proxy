# Prometheus Metrics Best Practices for LLM Inference Servers

## Research Summary

This document captures best practices for implementing Prometheus metrics in LLM inference servers, based on analysis of:
- **vLLM** (production-grade LLM serving engine)
- **Text Generation Inference (TGI)** (Hugging Face's LLM inference server)
- **prometheus_client** (official Python Prometheus client)
- **OpenTelemetry patterns** for LLM observability

---

## 1. Metric Naming Conventions

### Pattern: Namespaced Metric Names

**Problem it solves:** Avoids metric name collisions across different applications and makes metrics easily identifiable in Prometheus/Grafana.

**Best Practice:**
- Use a consistent prefix/namespace for all metrics
- Format: `<namespace>:<metric_name>` or `<namespace>_<metric_name>`
- Include units in name (e.g., `_seconds`, `_bytes`, `_total`)

**vLLM Example:**
```python
# vLLM uses "vllm:" prefix for all metrics
name="vllm:num_requests_running"
name="vllm:e2e_request_latency_seconds"
name="vllm:request_prompt_tokens"
name="vllm:generation_tokens"
name="vllm:time_to_first_token_seconds"
```

**TGI Example:**
```
tgi_request_duration          # seconds
tgi_request_generated_tokens  # count
tgi_batch_inference_duration  # seconds
tgi_queue_size               # count
```

**Applicability to npu-proxy:**
```python
# Current: Good namespace usage
'npu_proxy_requests_total'
'npu_proxy_request_latency_seconds'
'npu_proxy_inference_total'

# Recommended additions following vLLM patterns:
'npu_proxy_time_to_first_token_seconds'
'npu_proxy_inter_token_latency_seconds'
'npu_proxy_queue_time_seconds'
```

---

## 2. Label Cardinality Management

### Pattern: Fixed Low-Cardinality Labels

**Problem it solves:** High-cardinality labels (e.g., user_id, request_id) cause memory explosion in Prometheus and degrade query performance.

**Best Practice:**
- Limit labels to categorical data with bounded cardinality
- Use `model_name`, `device`, `endpoint`, `status` (not dynamic IDs)
- Maximum ~10-20 unique combinations per metric

**vLLM Example:**
```python
# Good: Fixed cardinality labels
labelnames = ["model_name", "engine"]
labelnames_with_reason = labelnames + ["finished_reason"]  # FinishReason is an enum

# Per-engine labeling pattern
per_engine_labelvalues: dict[int, list[object]] = {
    idx: [model_name, str(idx)] for idx in engine_indexes
}
```

**TGI Example:**
```
tgi_batch_decode_duration{method="prefill"}
tgi_batch_decode_duration{method="decode"}
# Only 2 values for 'method' label
```

**Applicability to npu-proxy:**
```python
# Current implementation - GOOD:
['endpoint', 'method', 'status']  # Bounded cardinality
['model', 'device', 'type']       # Limited values

# Avoid patterns like:
['request_id']        # HIGH CARDINALITY - DON'T DO
['user_id']           # HIGH CARDINALITY - DON'T DO
['prompt_hash']       # HIGH CARDINALITY - DON'T DO
```

---

## 3. Histogram Buckets for Latency Distributions

### Pattern: Exponential/Geometric Bucket Distributions

**Problem it solves:** Default Prometheus buckets are designed for HTTP requests, not LLM inference which has different latency characteristics (100ms to 10+ minutes).

**Best Practice:**
- Define custom buckets based on expected latency distribution
- Use exponential growth (1-2-5 pattern or powers of 2)
- Different bucket schemes for different metric types

**vLLM Examples:**

```python
# Time To First Token (TTFT) - ranges from ms to minutes
histogram_time_to_first_token = Histogram(
    name="vllm:time_to_first_token_seconds",
    buckets=[
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
        0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
    ],
)

# Inter-token latency - typically faster
histogram_inter_token_latency = Histogram(
    name="vllm:inter_token_latency_seconds",
    buckets=[
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
    ],
)

# End-to-end request latency - can span seconds to minutes
request_latency_buckets = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0,
    20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0,
    960.0, 1920.0, 7680.0,  # Up to ~2 hours
]

# Token count histograms (1-2-5 pattern)
def build_1_2_5_buckets(max_model_len):
    # Returns: [1, 2, 5, 10, 20, 50, 100, 200, 500, ...]
    buckets = []
    base = 1
    while base <= max_model_len:
        for multiplier in [1, 2, 5]:
            val = base * multiplier
            if val <= max_model_len:
                buckets.append(val)
        base *= 10
    return buckets

# Iteration token counts
buckets=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
```

**Applicability to npu-proxy:**
```python
# Current - reasonable but could be improved
REQUEST_LATENCY = Histogram(
    'npu_proxy_request_latency_seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

# Recommended: More granular for LLM inference patterns
INFERENCE_LATENCY = Histogram(
    'npu_proxy_inference_latency_seconds',
    buckets=[
        0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
        7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0,
        120.0, 180.0, 300.0, 600.0
    ]
)

# Add TTFT-specific histogram
TIME_TO_FIRST_TOKEN = Histogram(
    'npu_proxy_time_to_first_token_seconds',
    buckets=[
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0
    ]
)
```

---

## 4. Counter vs Gauge vs Histogram Selection

### Pattern: Metric Type Selection Guidelines

**Problem it solves:** Using wrong metric type leads to incorrect aggregations and misleading dashboards.

**Best Practice by Use Case:**

| Use Case | Metric Type | Why |
|----------|-------------|-----|
| Total requests served | Counter | Monotonically increasing |
| Requests in progress | Gauge | Can go up/down |
| Request latency | Histogram | Distribution analysis |
| Tokens processed | Counter | Monotonically increasing |
| Queue depth | Gauge | Point-in-time value |
| Cache hit rate | Counter (hits/queries) | Calculate rate from counters |
| Memory/KV cache usage | Gauge | Percentage, can fluctuate |
| Batch size | Histogram or Gauge | Distribution or current |

**vLLM Examples:**

```python
# GAUGE - Current state that fluctuates
gauge_scheduler_running = Gauge(
    name="vllm:num_requests_running",
    documentation="Number of requests in model execution batches.",
    multiprocess_mode="mostrecent",  # Important for multi-process!
)

gauge_kv_cache_usage = Gauge(
    name="vllm:kv_cache_usage_perc",
    documentation="KV-cache usage. 1 means 100 percent usage.",
    multiprocess_mode="mostrecent",
)

# COUNTER - Cumulative totals
counter_prompt_tokens = Counter(
    name="vllm:prompt_tokens",
    documentation="Number of prefill tokens processed.",
)

counter_request_success = Counter(
    name="vllm:request_success",
    documentation="Count of successfully processed requests.",
    labelnames=labelnames + ["finished_reason"],
)

# HISTOGRAM - Distribution analysis
histogram_e2e_time_request = Histogram(
    name="vllm:e2e_request_latency_seconds",
    documentation="Histogram of e2e request latency in seconds.",
)
```

**TGI Pattern:**

| Metric | Type | Rationale |
|--------|------|-----------|
| `tgi_request_count` | Counter | Total requests |
| `tgi_request_success` | Counter | Successful requests |
| `tgi_queue_size` | Gauge | Current queue depth |
| `tgi_batch_current_size` | Gauge | Current batch size |
| `tgi_request_duration` | Histogram | Latency distribution |
| `tgi_request_generated_tokens` | Histogram | Token count distribution |

**Applicability to npu-proxy:**
```python
# Current implementation follows best practices:
REQUEST_COUNT = Counter(...)      # Good: totals
REQUEST_IN_PROGRESS = Gauge(...)  # Good: current state
REQUEST_LATENCY = Histogram(...)  # Good: distribution
INFERENCE_TOKENS = Counter(...)   # Good: totals

# Consider adding:
QUEUE_SIZE = Gauge(
    'npu_proxy_queue_size',
    'Number of requests waiting in queue',
    multiprocess_mode='livesum'
)

BATCH_SIZE = Histogram(
    'npu_proxy_batch_size',
    'Distribution of batch sizes',
    buckets=[1, 2, 4, 8, 16, 32, 64]
)
```

---

## 5. Request Lifecycle Instrumentation

### Pattern: Comprehensive Request Phase Tracking

**Problem it solves:** Understanding where time is spent in the request lifecycle (queue, prefill, decode, etc.) for debugging and optimization.

**Best Practice:**
- Track each phase independently with histograms
- Use monotonic timestamps for interval calculations
- Record events: QUEUED → SCHEDULED → FIRST_TOKEN → COMPLETE

**vLLM Request Lifecycle Metrics:**

```python
# Phase-specific histograms
histogram_queue_time_request = Histogram(
    name="vllm:request_queue_time_seconds",
    documentation="Time spent in WAITING phase for request.",
)

histogram_prefill_time_request = Histogram(
    name="vllm:request_prefill_time_seconds",
    documentation="Time spent in PREFILL phase for request.",
)

histogram_decode_time_request = Histogram(
    name="vllm:request_decode_time_seconds",
    documentation="Time spent in DECODE phase for request.",
)

histogram_inference_time_request = Histogram(
    name="vllm:request_inference_time_seconds",
    documentation="Time spent in RUNNING phase (prefill + decode).",
)

histogram_time_to_first_token = Histogram(
    name="vllm:time_to_first_token_seconds",
    documentation="Time from request arrival to first token.",
)

histogram_e2e_time_request = Histogram(
    name="vllm:e2e_request_latency_seconds",
    documentation="End-to-end request latency.",
)

histogram_inter_token_latency = Histogram(
    name="vllm:inter_token_latency_seconds",
    documentation="Time between successive tokens (TPOT).",
)
```

**TGI Lifecycle Metrics:**

```
tgi_request_queue_duration       # Time in queue
tgi_request_validation_duration  # Request validation time
tgi_request_inference_duration   # Inference time
tgi_request_duration             # Total E2E latency
tgi_request_mean_time_per_token_duration  # TPOT
```

**Applicability to npu-proxy:**

```python
# Add lifecycle phase tracking:

QUEUE_TIME = Histogram(
    'npu_proxy_queue_time_seconds',
    'Time spent waiting in queue',
    ['model'],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

TIME_TO_FIRST_TOKEN = Histogram(
    'npu_proxy_time_to_first_token_seconds',
    'Time from request to first token (TTFT)',
    ['model', 'device'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

INTER_TOKEN_LATENCY = Histogram(
    'npu_proxy_inter_token_latency_seconds',
    'Time per output token (TPOT)',
    ['model', 'device'],
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# Context manager for lifecycle tracking
@contextmanager
def track_request_lifecycle(model: str, device: str):
    arrival_time = time.monotonic()
    first_token_time = None
    last_token_time = None
    
    class Tracker:
        def record_first_token(self):
            nonlocal first_token_time
            first_token_time = time.monotonic()
            TIME_TO_FIRST_TOKEN.labels(model=model, device=device).observe(
                first_token_time - arrival_time
            )
        
        def record_token(self):
            nonlocal last_token_time
            now = time.monotonic()
            if last_token_time:
                INTER_TOKEN_LATENCY.labels(model=model, device=device).observe(
                    now - last_token_time
                )
            last_token_time = now
    
    try:
        yield Tracker()
    finally:
        total_time = time.monotonic() - arrival_time
        INFERENCE_LATENCY.labels(model=model, device=device).observe(total_time)
```

---

## 6. Model-Specific Metrics (Tokens/sec, Batch Size)

### Pattern: LLM-Specific Performance Metrics

**Problem it solves:** Standard HTTP metrics don't capture LLM-specific performance characteristics like throughput in tokens/sec.

**Best Practice:**
- Track prompt tokens and generation tokens separately
- Record batch sizes for throughput analysis
- Calculate derived metrics (tokens/sec) in Grafana

**vLLM Token Metrics:**

```python
# Token counters (for calculating throughput)
counter_prompt_tokens = Counter(
    name="vllm:prompt_tokens",
    documentation="Number of prefill tokens processed.",
)

counter_generation_tokens = Counter(
    name="vllm:generation_tokens",
    documentation="Number of generation tokens processed.",
)

# Token distribution histograms
histogram_num_prompt_tokens_request = Histogram(
    name="vllm:request_prompt_tokens",
    documentation="Number of prefill tokens processed.",
    buckets=build_1_2_5_buckets(max_model_len),  # Dynamic based on model
)

histogram_num_generation_tokens_request = Histogram(
    name="vllm:request_generation_tokens",
    documentation="Number of generation tokens processed.",
    buckets=build_1_2_5_buckets(max_model_len),
)

# Iteration-level metrics
histogram_iteration_tokens = Histogram(
    name="vllm:iteration_tokens_total",
    documentation="Histogram of number of tokens per engine_step.",
    buckets=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
)
```

**TGI Token Metrics:**

```
tgi_request_input_length         # Input tokens per request
tgi_request_generated_tokens     # Output tokens per request
tgi_request_max_new_tokens       # Max tokens requested
tgi_batch_next_size             # Batch size distribution
```

**Throughput Calculation in Grafana:**

```promql
# Prompt tokens per second
rate(vllm:prompt_tokens_total[5m])

# Generation tokens per second  
rate(vllm:generation_tokens_total[5m])

# Total tokens per second
sum(rate(vllm:prompt_tokens_total[5m]) + rate(vllm:generation_tokens_total[5m]))
```

**Applicability to npu-proxy:**

```python
# Current token tracking - Good foundation
INFERENCE_TOKENS = Counter(
    'npu_proxy_inference_tokens_total',
    'Total tokens processed',
    ['model', 'type']  # type: prompt, completion
)

# Add token distribution histograms
PROMPT_TOKEN_DISTRIBUTION = Histogram(
    'npu_proxy_request_prompt_tokens',
    'Distribution of prompt token counts per request',
    ['model'],
    buckets=[1, 10, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000]
)

COMPLETION_TOKEN_DISTRIBUTION = Histogram(
    'npu_proxy_request_completion_tokens',
    'Distribution of completion token counts per request',
    ['model'],
    buckets=[1, 10, 25, 50, 100, 200, 500, 1000, 2000, 4000]
)

# Add batch size tracking if batching is supported
BATCH_SIZE = Histogram(
    'npu_proxy_batch_size',
    'Distribution of inference batch sizes',
    ['model', 'device'],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

# NPU-specific utilization
NPU_UTILIZATION = Gauge(
    'npu_proxy_npu_utilization_percent',
    'NPU utilization percentage',
    ['device_id'],
    multiprocess_mode='mostrecent'
)
```

---

## 7. Additional Patterns

### 7.1 Multiprocess Mode for Gauges

**Problem:** Gauge metrics don't aggregate correctly across worker processes.

**Solution from prometheus_client:**
```python
# Choose appropriate multiprocess_mode
Gauge('metric', 'help', multiprocess_mode='mostrecent')  # Latest value wins
Gauge('metric', 'help', multiprocess_mode='livesum')     # Sum across live processes
Gauge('metric', 'help', multiprocess_mode='max')         # Maximum value
```

### 7.2 Disable _created Metrics

**Problem:** Counter/Histogram _created metrics add cardinality without value.

**Solution:**
```python
from prometheus_client import disable_created_metrics
disable_created_metrics()
```

### 7.3 Info Metrics for Metadata

**Pattern:** Use Info metric for static metadata.

```python
from prometheus_client import Info

MODEL_INFO = Info('npu_proxy_model', 'Model information')
MODEL_INFO.info({
    'model_name': 'llama-3.2',
    'model_version': '1.0.0',
    'device_type': 'npu',
    'quantization': 'int8'
})
```

---

## 8. Recommended Metrics for npu-proxy

Based on vLLM, TGI, and current npu-proxy implementation:

### Core Metrics (Already Implemented)
- ✅ `npu_proxy_requests_total` - Request counter
- ✅ `npu_proxy_request_latency_seconds` - Request latency histogram
- ✅ `npu_proxy_requests_in_progress` - In-flight requests gauge
- ✅ `npu_proxy_inference_total` - Inference counter
- ✅ `npu_proxy_inference_latency_seconds` - Inference latency
- ✅ `npu_proxy_inference_tokens_total` - Token counter
- ✅ `npu_proxy_errors_total` - Error counter
- ✅ `npu_proxy_routing_decisions_total` - Routing decisions

### Recommended Additions

**LLM-Specific Latency:**
```python
npu_proxy_time_to_first_token_seconds   # TTFT histogram
npu_proxy_inter_token_latency_seconds   # TPOT histogram
npu_proxy_queue_time_seconds            # Queue wait time
```

**Token Distributions:**
```python
npu_proxy_request_prompt_tokens         # Prompt length histogram
npu_proxy_request_completion_tokens     # Output length histogram
```

**Resource Utilization:**
```python
npu_proxy_npu_utilization_percent       # NPU utilization gauge
npu_proxy_memory_used_bytes             # Memory usage gauge
npu_proxy_queue_size                    # Current queue depth gauge
```

**Streaming Metrics:**
```python
npu_proxy_streaming_requests_total      # Streaming vs non-streaming
npu_proxy_stream_tokens_sent_total      # Tokens sent in streams
```

---

## References

1. vLLM Metrics Design: https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md
2. vLLM Prometheus Logger: https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py
3. TGI Metrics Reference: https://huggingface.co/docs/text-generation-inference/reference/metrics
4. Prometheus Python Client: https://github.com/prometheus/client_python
5. Prometheus Best Practices: https://prometheus.io/docs/practices/naming/
