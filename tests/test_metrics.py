"""Tests for Prometheus metrics."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


class FakeRegistry:
    """Tiny Prometheus registry test double with sample lookup support."""

    def __init__(self):
        self.samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.help_names: set[str] = set()

    def set_sample(self, name: str, labels: dict[str, str], value: float) -> None:
        self.samples[(name, tuple(sorted(labels.items())))] = value
        self.help_names.add(name)

    def add_sample(self, name: str, labels: dict[str, str], value: float) -> None:
        self.set_sample(name, labels, self.get_sample_value(name, labels) + value)

    def get_sample_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        return self.samples.get((name, tuple(sorted((labels or {}).items()))), 0.0)

    def generate_latest(self) -> bytes:
        names = sorted(self.help_names or {"npu_proxy_placeholder"})
        return "".join(f"# HELP {name} fake metric\n" for name in names).encode()


class FakeMetric:
    metric_type = "gauge"

    def __init__(self, name, _description, labelnames=None, registry=None, **_kwargs):
        self.name = name
        self.registry = registry
        self.registry.help_names.add(name)

    def labels(self, **labels):
        return FakeMetricChild(self, {key: str(value) for key, value in labels.items()})


class FakeCounter(FakeMetric):
    metric_type = "counter"


class FakeHistogram(FakeMetric):
    metric_type = "histogram"


class FakeGauge(FakeMetric):
    metric_type = "gauge"


class FakeInfo(FakeMetric):
    pass


class FakeMetricChild:
    def __init__(self, metric: FakeMetric, labels: dict[str, str]):
        self.metric = metric
        self.labels = labels

    def inc(self, amount: float = 1.0) -> None:
        self.metric.registry.add_sample(self.metric.name, self.labels, amount)
        if self.metric.metric_type == "counter" and self.metric.name.endswith("_total"):
            created_name = f"{self.metric.name[:-6]}_created"
            self.metric.registry.set_sample(created_name, self.labels, 1.0)

    def dec(self, amount: float = 1.0) -> None:
        self.metric.registry.add_sample(self.metric.name, self.labels, -amount)

    def set(self, value: float) -> None:
        self.metric.registry.set_sample(self.metric.name, self.labels, float(value))

    def observe(self, value: float) -> None:
        self.metric.registry.add_sample(f"{self.metric.name}_count", self.labels, 1.0)
        self.metric.registry.add_sample(f"{self.metric.name}_sum", self.labels, float(value))


@pytest.fixture
def isolated_metrics_module():
    """Reload metrics with collectors bound to an isolated fake Prometheus registry."""
    import types
    from _pytest.monkeypatch import MonkeyPatch

    registry = FakeRegistry()
    patcher = MonkeyPatch()
    fake_prometheus = types.ModuleType("prometheus_client")
    fake_prometheus.Counter = lambda *args, **kwargs: FakeCounter(*args, registry=registry, **kwargs)
    fake_prometheus.Histogram = lambda *args, **kwargs: FakeHistogram(*args, registry=registry, **kwargs)
    fake_prometheus.Gauge = lambda *args, **kwargs: FakeGauge(*args, registry=registry, **kwargs)
    fake_prometheus.Info = lambda *args, **kwargs: FakeInfo(*args, registry=registry, **kwargs)
    fake_prometheus.generate_latest = lambda: registry.generate_latest()
    fake_prometheus.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    patcher.setitem(sys.modules, "prometheus_client", fake_prometheus)

    import npu_proxy.metrics as metrics

    metrics = importlib.reload(metrics)
    try:
        yield metrics, registry
    finally:
        patcher.undo()
        importlib.reload(metrics)


def sample(registry, name: str, labels: dict[str, str] | None = None) -> float | None:
    return registry.get_sample_value(name, labels or {})


class TestMetricsInfrastructure:
    """Tests for metrics module."""

    def test_get_metrics_returns_bytes(self, isolated_metrics_module):
        """get_metrics returns Prometheus exposition bytes from the isolated registry."""
        metrics, _registry = isolated_metrics_module

        result = metrics.get_metrics()

        assert isinstance(result, bytes)
        assert b"# HELP" in result

    def test_request_metrics_increment_with_concrete_labels(self, isolated_metrics_module):
        """record_request increments the expected counter sample."""
        metrics, registry = isolated_metrics_module

        metrics.record_request("/v1/chat", "post", 201)
        metrics.record_request("/v1/chat", "POST", 201)

        labels = {"endpoint": "/v1/chat", "method": "POST", "status": "201"}
        assert sample(registry, "npu_proxy_requests_total", labels) == 2.0
        assert sample(registry, "npu_proxy_requests_created", labels) is not None

    def test_inference_metrics_record_count_and_latency(self, isolated_metrics_module):
        """record_inference updates counter and histogram samples with bounded labels."""
        metrics, registry = isolated_metrics_module

        metrics.record_inference("tinyllama", "NPU", "chat", 1.5)
        metrics.record_inference("tinyllama", "TPU", "unknown-kind", 2.0)

        assert sample(
            registry,
            "npu_proxy_inference_total",
            {"model": "tinyllama", "device": "npu", "type": "chat"},
        ) == 1.0
        assert sample(
            registry,
            "npu_proxy_inference_latency_seconds_count",
            {"model": "tinyllama", "device": "npu"},
        ) == 1.0
        assert sample(
            registry,
            "npu_proxy_inference_latency_seconds_sum",
            {"model": "tinyllama", "device": "npu"},
        ) == 1.5
        assert sample(
            registry,
            "npu_proxy_inference_total",
            {"model": "tinyllama", "device": "unknown", "type": "other"},
        ) == 1.0

    def test_routing_runtime_feature_and_error_metrics(self, isolated_metrics_module):
        """Routing, runtime-feature, and error helpers expose concrete samples."""
        metrics, registry = isolated_metrics_module

        metrics.record_routing_decision("NPU", "within_npu_limit")
        metrics.record_routing_decision("ASIC", "surprise")
        metrics.record_runtime_feature_state("tinyllama", "npu", "compile_cache", True)
        metrics.record_runtime_feature_state("tinyllama", "npu", "prefix_cache", False)
        metrics.record_runtime_feature_degradation("tinyllama", "npu", "prefix_cache", "safe_retry")
        metrics.record_error("/v1/chat", "timeout")
        metrics.record_error("/v1/chat", "mystery")

        assert sample(
            registry,
            "npu_proxy_routing_decisions_total",
            {"device": "npu", "reason": "within_npu_limit"},
        ) == 1.0
        assert sample(
            registry,
            "npu_proxy_routing_decisions_total",
            {"device": "unknown", "reason": "other"},
        ) == 1.0
        assert sample(
            registry,
            "npu_proxy_runtime_feature_state",
            {"model": "tinyllama", "device": "npu", "feature": "compile_cache"},
        ) == 1.0
        assert sample(
            registry,
            "npu_proxy_runtime_feature_state",
            {"model": "tinyllama", "device": "npu", "feature": "prefix_cache"},
        ) == 0.0
        assert sample(
            registry,
            "npu_proxy_runtime_feature_degradations_total",
            {"model": "tinyllama", "device": "npu", "feature": "prefix_cache", "reason": "safe_retry"},
        ) == 1.0
        assert sample(registry, "npu_proxy_errors_total", {"endpoint": "/v1/chat", "error_type": "timeout"}) == 1.0
        assert sample(registry, "npu_proxy_errors_total", {"endpoint": "/v1/chat", "error_type": "other"}) == 1.0

    def test_routing_execution_metrics_use_bounded_labels(self, isolated_metrics_module):
        """Routed-vs-execution metrics keep device and reason cardinality bounded."""
        metrics, registry = isolated_metrics_module

        metrics.record_routing_execution("NPU", "CPU", "busy")
        metrics.record_routing_execution("ASIC", "TPU", "surprise")

        assert sample(
            registry,
            "npu_proxy_routing_executions_total",
            {"routed_device": "npu", "execution_device": "cpu", "fallback_reason": "busy"},
        ) == 1.0
        assert sample(
            registry,
            "npu_proxy_routing_executions_total",
            {"routed_device": "unknown", "execution_device": "unknown", "fallback_reason": "other"},
        ) == 1.0

    def test_routing_execution_collapses_multi_instance_gpu_labels(self, isolated_metrics_module):
        """Discrete GPUs (GPU.0/GPU.1) collapse to 'gpu' instead of 'unknown'."""
        metrics, registry = isolated_metrics_module

        metrics.record_routing_execution("GPU.0", "GPU.1", "device_fallback")

        assert sample(
            registry,
            "npu_proxy_routing_executions_total",
            {"routed_device": "gpu", "execution_device": "gpu", "fallback_reason": "device_fallback"},
        ) == 1.0

    def test_record_inference_collapses_multi_instance_gpu_label(self, isolated_metrics_module):
        """Per-inference counters collapse GPU.1 to 'gpu', not 'unknown'."""
        metrics, registry = isolated_metrics_module

        metrics.record_inference("tinyllama", "GPU.1", "chat", 1.0)

        assert sample(
            registry,
            "npu_proxy_inference_total",
            {"model": "tinyllama", "device": "gpu", "type": "chat"},
        ) == 1.0

    def test_record_tokens_per_second_collapses_multi_instance_gpu_label(self, isolated_metrics_module):
        """Tokens/sec gauge collapses GPU.0 to 'gpu', not 'unknown'."""
        metrics, registry = isolated_metrics_module

        metrics.record_tokens_per_second("tinyllama", "GPU.0", 42.0)

        assert sample(
            registry,
            "npu_proxy_tokens_per_second",
            {"model": "tinyllama", "device": "gpu"},
        ) == 42.0

    def test_track_request_context_manager_success(self, isolated_metrics_module, monkeypatch):
        """Successful request contexts decrement in-progress and record latency."""
        metrics, registry = isolated_metrics_module
        times = iter([10.0, 10.25])
        monkeypatch.setattr(metrics.time, "time", lambda: next(times))

        with metrics.track_request("/v1/chat"):
            assert sample(registry, "npu_proxy_requests_in_progress", {"endpoint": "/v1/chat"}) == 1.0

        assert sample(registry, "npu_proxy_requests_in_progress", {"endpoint": "/v1/chat"}) == 0.0
        assert sample(registry, "npu_proxy_request_latency_seconds_count", {"endpoint": "/v1/chat"}) == 1.0
        assert sample(registry, "npu_proxy_request_latency_seconds_sum", {"endpoint": "/v1/chat"}) == pytest.approx(0.25)

    def test_track_request_context_manager_failure(self, isolated_metrics_module, monkeypatch):
        """Failing request contexts still decrement in-progress and record latency."""
        metrics, registry = isolated_metrics_module
        times = iter([20.0, 20.5])
        monkeypatch.setattr(metrics.time, "time", lambda: next(times))

        with pytest.raises(RuntimeError, match="boom"):
            with metrics.track_request("/v1/chat"):
                raise RuntimeError("boom")

        assert sample(registry, "npu_proxy_requests_in_progress", {"endpoint": "/v1/chat"}) == 0.0
        assert sample(registry, "npu_proxy_request_latency_seconds_count", {"endpoint": "/v1/chat"}) == 1.0
        assert sample(registry, "npu_proxy_request_latency_seconds_sum", {"endpoint": "/v1/chat"}) == pytest.approx(0.5)

    def test_metrics_work_without_prometheus_in_subprocess(self):
        """Reloading with prometheus_client absent exercises the real no-op branch."""
        code = r'''
import builtins
import importlib

original_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "prometheus_client" or name.startswith("prometheus_client."):
        raise ImportError("blocked prometheus_client for test")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
metrics = importlib.import_module("npu_proxy.metrics")
assert metrics.PROMETHEUS_AVAILABLE is False
metrics.record_request("/v1/chat", "POST", 200)
metrics.record_inference("tinyllama", "NPU", "chat", 1.5)
metrics.record_routing_decision("CPU", "fallback")
metrics.record_runtime_feature_state("tinyllama", "npu", "compile_cache", True)
metrics.record_runtime_feature_degradation("tinyllama", "npu", "prefix_cache", "safe_retry")
metrics.record_error("/v1/chat", "timeout")
with metrics.track_request("/v1/chat"):
    pass
assert metrics.get_metrics() == b"# Prometheus client not installed\n"
assert metrics.get_content_type() == "text/plain"
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=".",
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

        assert result.returncode == 0, result.stderr


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_endpoint_returns_200(self, client):
        """GET /metrics should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_content_type(self, client):
        """GET /metrics should return a text content type."""
        response = client.get("/metrics")
        assert "text/" in response.headers.get("content-type", "")
