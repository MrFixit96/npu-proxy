"""Prometheus metrics endpoint for NPU Proxy observability.

This module provides the HTTP endpoint for Prometheus metric scraping,
exposing comprehensive observability data for LLM inference operations.
It serves metrics in the standard Prometheus exposition format.

The metrics exposed through this endpoint cover:
    - Request metrics (counts, latency, in-progress)
    - Inference metrics (operations, latency, token throughput)
    - LLM-specific metrics (TTFT, TPOT, tokens/sec)
    - Routing and error metrics

Example:
    The router is typically included in a FastAPI app::

        from fastapi import FastAPI
        from npu_proxy.api.metrics import router

        app = FastAPI()
        app.include_router(router)

Attributes:
    router (APIRouter): FastAPI router with the /metrics endpoint.
"""

from fastapi import APIRouter
from fastapi.responses import Response

from npu_proxy.metrics import get_metrics, get_content_type

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics for scraping.

    Returns metrics in Prometheus exposition format for monitoring.
    Configure Prometheus to scrape this endpoint for comprehensive
    observability of NPU Proxy operations.

    Returns:
        Response: FastAPI Response with Prometheus text format metrics.

    Content-Type:
        text/plain; version=0.0.4; charset=utf-8 (Prometheus exposition format)

    Available Metrics:
        Request Metrics:
            npu_proxy_requests_total: Request counter by endpoint/method/status
            npu_proxy_request_latency_seconds: Request latency histogram (end-to-end)
            npu_proxy_requests_in_progress: Gauge of concurrent requests
            npu_proxy_queue_time_seconds: Queue wait time histogram

        Inference Metrics:
            npu_proxy_inference_total: Inference counter by model/device/type
            npu_proxy_inference_latency_seconds: Inference latency histogram
            npu_proxy_inference_tokens_total: Token counter by model/type

        LLM Streaming Metrics (vLLM/TGI inspired):
            npu_proxy_time_to_first_token_seconds: TTFT histogram (critical SLO)
            npu_proxy_inter_token_latency_seconds: TPOT/ITL histogram
            npu_proxy_tokens_per_second: Token throughput gauge

        Routing Metrics:
            npu_proxy_routing_decisions_total: Routing decision counter by device/reason

        Model Metrics:
            npu_proxy_model_info: Model information
            npu_proxy_model_load_seconds: Model load time gauge

        Error Metrics:
            npu_proxy_errors_total: Error counter by endpoint/error_type

    Prometheus Configuration:
        Add to prometheus.yml::

            scrape_configs:
              - job_name: 'npu-proxy'
                static_configs:
                  - targets: ['localhost:8080']
                metrics_path: /metrics
                scrape_interval: 15s

    Example Queries:
        Request rate (5m window)::

            rate(npu_proxy_requests_total[5m])

        P95 TTFT latency::

            histogram_quantile(0.95,
              rate(npu_proxy_time_to_first_token_seconds_bucket[5m]))

        Error rate by endpoint::

            rate(npu_proxy_errors_total[5m])

    Grafana Dashboard:
        See docs/grafana-dashboard.json for a pre-built dashboard.

    Note:
        Metrics are lazily initialized on first access. If prometheus_client
        is not installed, returns a placeholder message.
    """
    return Response(
        content=get_metrics(),
        media_type=get_content_type(),
    )
