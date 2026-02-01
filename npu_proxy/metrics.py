"""Prometheus metrics for NPU Proxy.

This module provides comprehensive observability for LLM inference operations
running through the NPU Proxy. It implements industry-standard metrics inspired
by vLLM and TGI (Text Generation Inference) for monitoring ML workloads.

Key Metrics Categories:
    - Request metrics: Track API request counts, latency, and in-progress requests
    - Inference metrics: Monitor inference operations, latency, and token throughput
    - LLM-specific metrics: TTFT (Time To First Token), TPOT (Time Per Output Token)
    - Routing metrics: Track device routing decisions
    - Model metrics: Monitor model load times and information
    - Error metrics: Track error counts by type

The module uses lazy initialization to avoid import-time side effects and
gracefully degrades when prometheus_client is not installed.

Example:
    Basic usage with context manager::

        from npu_proxy.metrics import track_request, record_inference

        with track_request('/v1/chat/completions'):
            result = await process_inference(request)
            record_inference('phi-3', 'npu', 'chat', latency=1.5)

    Recording streaming metrics::

        from npu_proxy.metrics import record_ttft, record_tpot

        # Record time to first token
        record_ttft('phi-3', ttft_seconds=0.35)

        # Record inter-token latency for each token
        for token in stream:
            record_tpot('phi-3', tpot_seconds=0.025)

Attributes:
    PROMETHEUS_AVAILABLE (bool): Whether prometheus_client is installed.
    _metrics_initialized (bool): Whether metrics have been lazily initialized.
"""

import time
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to no-op if not installed
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics disabled")

# Metric definitions (created lazily)
_metrics_initialized = False

# Request metrics
REQUEST_COUNT = None
REQUEST_LATENCY = None
REQUEST_IN_PROGRESS = None
QUEUE_TIME = None

# Inference metrics
INFERENCE_COUNT = None
INFERENCE_LATENCY = None
INFERENCE_TOKENS = None

# LLM-specific streaming metrics (vLLM/TGI inspired)
TIME_TO_FIRST_TOKEN = None
INTER_TOKEN_LATENCY = None
TOKENS_PER_SECOND = None

# Routing metrics
ROUTING_DECISIONS = None

# Model metrics
MODEL_INFO = None
MODEL_LOAD_TIME = None

# Error metrics
ERROR_COUNT = None

# Histogram bucket definitions optimized for LLM workloads
# Based on observed latency patterns from vLLM and TGI
REQUEST_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
INFERENCE_LATENCY_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
TTFT_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
TPOT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
QUEUE_TIME_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0)


def _init_metrics() -> None:
    """Initialize Prometheus metrics using lazy initialization pattern.

    This function creates all Prometheus metric collectors on first call.
    Subsequent calls are no-ops. This pattern avoids import-time side effects
    and allows the module to be imported even when prometheus_client is not
    installed.

    The function initializes the following metric categories:
        - Request metrics (counts, latency, in-progress, queue time)
        - Inference metrics (counts, latency, tokens)
        - LLM streaming metrics (TTFT, TPOT, tokens/sec)
        - Routing metrics (device selection decisions)
        - Model metrics (info, load times)
        - Error metrics (counts by type)

    Note:
        All histogram buckets are tuned for LLM inference workloads based on
        observed patterns from vLLM and TGI production deployments.

    Returns:
        None
    """
    global _metrics_initialized
    global REQUEST_COUNT, REQUEST_LATENCY, REQUEST_IN_PROGRESS, QUEUE_TIME
    global INFERENCE_COUNT, INFERENCE_LATENCY, INFERENCE_TOKENS
    global TIME_TO_FIRST_TOKEN, INTER_TOKEN_LATENCY, TOKENS_PER_SECOND
    global ROUTING_DECISIONS, MODEL_INFO, MODEL_LOAD_TIME, ERROR_COUNT
    
    if _metrics_initialized or not PROMETHEUS_AVAILABLE:
        return
    
    from prometheus_client import Counter, Histogram, Gauge, Info
    
    # Request metrics
    REQUEST_COUNT = Counter(
        'npu_proxy_requests_total',
        'Total number of API requests',
        ['endpoint', 'method', 'status']
    )
    
    REQUEST_LATENCY = Histogram(
        'npu_proxy_request_latency_seconds',
        'Request latency in seconds (end-to-end)',
        ['endpoint'],
        buckets=REQUEST_LATENCY_BUCKETS
    )
    
    REQUEST_IN_PROGRESS = Gauge(
        'npu_proxy_requests_in_progress',
        'Number of requests currently being processed',
        ['endpoint']
    )
    
    QUEUE_TIME = Histogram(
        'npu_proxy_queue_time_seconds',
        'Time spent waiting in queue before processing begins',
        ['endpoint'],
        buckets=QUEUE_TIME_BUCKETS
    )
    
    # Inference metrics
    INFERENCE_COUNT = Counter(
        'npu_proxy_inference_total',
        'Total number of inference operations',
        ['model', 'device', 'type']  # type: chat, embeddings
    )
    
    INFERENCE_LATENCY = Histogram(
        'npu_proxy_inference_latency_seconds',
        'Total inference latency in seconds (includes all tokens)',
        ['model', 'device'],
        buckets=INFERENCE_LATENCY_BUCKETS
    )
    
    INFERENCE_TOKENS = Counter(
        'npu_proxy_inference_tokens_total',
        'Total tokens processed',
        ['model', 'type']  # type: prompt, completion
    )
    
    # LLM-specific streaming metrics (vLLM/TGI inspired)
    # These are critical SLOs for LLM inference quality
    TIME_TO_FIRST_TOKEN = Histogram(
        'npu_proxy_time_to_first_token_seconds',
        'Time from request receipt to first token generation (TTFT) - critical UX metric',
        ['model'],
        buckets=TTFT_BUCKETS
    )
    
    INTER_TOKEN_LATENCY = Histogram(
        'npu_proxy_inter_token_latency_seconds',
        'Time between consecutive tokens during streaming (TPOT/ITL)',
        ['model'],
        buckets=TPOT_BUCKETS
    )
    
    TOKENS_PER_SECOND = Gauge(
        'npu_proxy_tokens_per_second',
        'Current token generation throughput rate',
        ['model', 'device']
    )
    
    # Routing metrics
    ROUTING_DECISIONS = Counter(
        'npu_proxy_routing_decisions_total',
        'Routing decisions by device and reason',
        ['device', 'reason']
    )
    
    # Model metrics
    MODEL_INFO = Info(
        'npu_proxy_model',
        'Information about loaded models'
    )
    
    MODEL_LOAD_TIME = Gauge(
        'npu_proxy_model_load_seconds',
        'Time taken to load model in seconds',
        ['model']
    )
    
    # Error metrics
    ERROR_COUNT = Counter(
        'npu_proxy_errors_total',
        'Total number of errors by endpoint and type',
        ['endpoint', 'error_type']
    )
    
    _metrics_initialized = True
    logger.debug("Prometheus metrics initialized successfully")


def ensure_metrics() -> None:
    """Ensure metrics are initialized before use.

    This is a convenience wrapper around _init_metrics() that should be
    called before any metric recording operation. It is idempotent and
    safe to call multiple times.

    Example:
        >>> ensure_metrics()
        >>> REQUEST_COUNT.labels(endpoint='/health', method='GET', status='200').inc()
    """
    if not _metrics_initialized:
        _init_metrics()


@contextmanager
def track_request(endpoint: str):
    """Context manager to track request metrics with automatic timing.

    Tracks the lifecycle of an HTTP request including:
        - Incrementing in-progress counter on entry
        - Decrementing in-progress counter on exit
        - Recording request latency on exit

    This context manager is exception-safe and will record metrics even
    if the wrapped code raises an exception.

    Args:
        endpoint: The API endpoint being called (e.g., '/v1/chat/completions').

    Yields:
        None

    Example:
        >>> async def handle_chat(request):
        ...     with track_request('/v1/chat/completions'):
        ...         response = await process_chat(request)
        ...         return response
    """
    ensure_metrics()
    if not PROMETHEUS_AVAILABLE:
        yield
        return
    
    REQUEST_IN_PROGRESS.labels(endpoint=endpoint).inc()
    start_time = time.time()
    try:
        yield
    finally:
        REQUEST_IN_PROGRESS.labels(endpoint=endpoint).dec()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


def record_request(endpoint: str, method: str, status: int) -> None:
    """Record a completed HTTP request.

    Increments the request counter with labels for the endpoint, HTTP method,
    and response status code. This should be called once per completed request.

    Args:
        endpoint: The API endpoint (e.g., '/v1/chat/completions').
        method: The HTTP method (e.g., 'GET', 'POST').
        status: The HTTP response status code (e.g., 200, 500).

    Example:
        >>> record_request('/v1/chat/completions', 'POST', 200)
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and REQUEST_COUNT:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=str(status)).inc()


def record_queue_time(endpoint: str, queue_seconds: float) -> None:
    """Record time spent waiting in queue before processing.

    This metric is important for understanding request queuing behavior
    and identifying bottlenecks in request handling.

    Args:
        endpoint: The API endpoint being called.
        queue_seconds: Time in seconds the request waited in queue.

    Example:
        >>> start_queue = time.time()
        >>> # ... wait for worker to become available ...
        >>> record_queue_time('/v1/chat/completions', time.time() - start_queue)
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and QUEUE_TIME:
        QUEUE_TIME.labels(endpoint=endpoint).observe(queue_seconds)


def record_inference(model: str, device: str, inference_type: str, latency: float) -> None:
    """Record an inference operation with timing.

    Records both the inference count and latency. This is the primary metric
    for tracking inference operations.

    Args:
        model: The model identifier (e.g., 'phi-3-mini').
        device: The device used for inference ('npu', 'cpu', 'gpu').
        inference_type: The type of inference ('chat', 'embeddings').
        latency: Total inference latency in seconds.

    Example:
        >>> start = time.time()
        >>> result = run_inference(model, prompt)
        >>> record_inference('phi-3-mini', 'npu', 'chat', time.time() - start)
    """
    ensure_metrics()
    if not PROMETHEUS_AVAILABLE:
        return
    if INFERENCE_COUNT:
        INFERENCE_COUNT.labels(model=model, device=device, type=inference_type).inc()
    if INFERENCE_LATENCY:
        INFERENCE_LATENCY.labels(model=model, device=device).observe(latency)


def record_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts for an inference operation.

    Tracks both prompt (input) and completion (output) token counts separately.
    This is essential for understanding model utilization and costs.

    Args:
        model: The model identifier.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated in the response.

    Example:
        >>> record_tokens('phi-3-mini', prompt_tokens=150, completion_tokens=500)
    """
    ensure_metrics()
    if not PROMETHEUS_AVAILABLE or not INFERENCE_TOKENS:
        return
    INFERENCE_TOKENS.labels(model=model, type="prompt").inc(prompt_tokens)
    INFERENCE_TOKENS.labels(model=model, type="completion").inc(completion_tokens)


def record_ttft(model: str, ttft_seconds: float) -> None:
    """Record Time To First Token (TTFT) for an inference request.

    TTFT is a critical SLO metric for LLM inference that measures the time
    from when a request is received to when the first token is generated.
    This directly impacts user-perceived latency, especially for streaming
    responses.

    Industry benchmarks (from vLLM/TGI):
        - Excellent: < 0.5s
        - Good: 0.5s - 2s
        - Acceptable: 2s - 5s
        - Poor: > 5s

    Args:
        model: The model identifier.
        ttft_seconds: Time in seconds from request to first token.

    Example:
        >>> request_start = time.time()
        >>> first_token = await get_first_token(stream)
        >>> record_ttft('phi-3-mini', time.time() - request_start)
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and TIME_TO_FIRST_TOKEN:
        TIME_TO_FIRST_TOKEN.labels(model=model).observe(ttft_seconds)


def record_tpot(model: str, tpot_seconds: float) -> None:
    """Record Time Per Output Token (TPOT) / Inter-Token Latency (ITL).

    TPOT measures the time between consecutive tokens during streaming.
    This is a key metric for streaming quality and directly impacts
    the "smoothness" of text generation from the user's perspective.

    Industry benchmarks (from vLLM/TGI):
        - Excellent: < 0.05s (20+ tokens/sec)
        - Good: 0.05s - 0.1s (10-20 tokens/sec)
        - Acceptable: 0.1s - 0.25s (4-10 tokens/sec)
        - Poor: > 0.25s (< 4 tokens/sec)

    Args:
        model: The model identifier.
        tpot_seconds: Time in seconds between consecutive tokens.

    Example:
        >>> last_token_time = time.time()
        >>> async for token in stream:
        ...     now = time.time()
        ...     record_tpot('phi-3-mini', now - last_token_time)
        ...     last_token_time = now
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and INTER_TOKEN_LATENCY:
        INTER_TOKEN_LATENCY.labels(model=model).observe(tpot_seconds)


def record_tokens_per_second(model: str, device: str, tokens_per_sec: float) -> None:
    """Record current token generation throughput.

    This gauge metric represents the instantaneous token generation rate.
    Useful for monitoring real-time performance and capacity planning.

    Args:
        model: The model identifier.
        device: The device used for inference ('npu', 'cpu', 'gpu').
        tokens_per_sec: Current token generation rate (tokens/second).

    Example:
        >>> # Calculate tokens/sec over a window
        >>> tokens_generated = 100
        >>> generation_time = 2.5  # seconds
        >>> record_tokens_per_second('phi-3-mini', 'npu', tokens_generated / generation_time)
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and TOKENS_PER_SECOND:
        TOKENS_PER_SECOND.labels(model=model, device=device).set(tokens_per_sec)


def record_routing_decision(device: str, reason: str) -> None:
    """Record a routing decision for inference requests.

    Tracks why and where inference requests are routed. Useful for
    understanding device utilization and routing policy effectiveness.

    Args:
        device: The target device ('npu', 'cpu', 'gpu').
        reason: The reason for routing (e.g., 'model_affinity', 'load_balance',
                'device_unavailable', 'fallback').

    Example:
        >>> record_routing_decision('npu', 'model_affinity')
        >>> record_routing_decision('cpu', 'npu_unavailable')
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and ROUTING_DECISIONS:
        ROUTING_DECISIONS.labels(device=device, reason=reason).inc()


def record_error(endpoint: str, error_type: str) -> None:
    """Record an error occurrence.

    Tracks errors by endpoint and error type for alerting and debugging.

    Args:
        endpoint: The API endpoint where the error occurred.
        error_type: The type/category of error (e.g., 'timeout', 'oom',
                    'invalid_request', 'model_error').

    Example:
        >>> try:
        ...     result = await process_request(request)
        ... except TimeoutError:
        ...     record_error('/v1/chat/completions', 'timeout')
        ...     raise
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and ERROR_COUNT:
        ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()


def record_model_load_time(model: str, load_seconds: float) -> None:
    """Record the time taken to load a model.

    Tracks model loading performance, useful for cold start optimization
    and capacity planning.

    Args:
        model: The model identifier.
        load_seconds: Time in seconds to load the model.

    Example:
        >>> start = time.time()
        >>> model = load_model('phi-3-mini')
        >>> record_model_load_time('phi-3-mini', time.time() - start)
    """
    ensure_metrics()
    if PROMETHEUS_AVAILABLE and MODEL_LOAD_TIME:
        MODEL_LOAD_TIME.labels(model=model).set(load_seconds)


def get_metrics() -> bytes:
    """Get Prometheus metrics in exposition format.

    Returns all registered metrics in the standard Prometheus text-based
    exposition format, suitable for scraping by Prometheus or compatible
    collectors.

    Returns:
        bytes: Metrics in Prometheus exposition format, or a placeholder
               message if prometheus_client is not installed.

    Example:
        >>> from fastapi import Response
        >>> @app.get('/metrics')
        ... def metrics():
        ...     return Response(
        ...         content=get_metrics(),
        ...         media_type=get_content_type()
        ...     )
    """
    ensure_metrics()
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n"
    return generate_latest()


def get_content_type() -> str:
    """Get the Content-Type header value for the metrics endpoint.

    Returns the appropriate MIME type for Prometheus metrics exposition
    format, which includes version information.

    Returns:
        str: The Content-Type header value (e.g., 'text/plain; version=0.0.4').

    Example:
        >>> content_type = get_content_type()
        >>> # Returns: 'text/plain; version=0.0.4; charset=utf-8'
    """
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# Convenience aliases for common metric patterns
def record_streaming_metrics(
    model: str,
    device: str,
    ttft_seconds: float,
    total_tokens: int,
    total_duration_seconds: float
) -> None:
    """Record comprehensive metrics for a streaming inference operation.

    This is a convenience function that records multiple metrics from
    a single streaming inference operation.

    Args:
        model: The model identifier.
        device: The device used for inference.
        ttft_seconds: Time to first token in seconds.
        total_tokens: Total number of tokens generated.
        total_duration_seconds: Total inference duration in seconds.

    Example:
        >>> record_streaming_metrics(
        ...     model='phi-3-mini',
        ...     device='npu',
        ...     ttft_seconds=0.35,
        ...     total_tokens=150,
        ...     total_duration_seconds=3.5
        ... )
    """
    record_ttft(model, ttft_seconds)
    
    # Calculate average TPOT (excluding first token time)
    if total_tokens > 1 and total_duration_seconds > ttft_seconds:
        avg_tpot = (total_duration_seconds - ttft_seconds) / (total_tokens - 1)
        record_tpot(model, avg_tpot)
    
    # Calculate and record tokens per second
    if total_duration_seconds > 0:
        tps = total_tokens / total_duration_seconds
        record_tokens_per_second(model, device, tps)
