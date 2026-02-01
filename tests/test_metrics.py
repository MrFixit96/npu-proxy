"""Tests for Prometheus metrics."""
import pytest
from unittest.mock import patch, MagicMock


class TestMetricsInfrastructure:
    """Tests for metrics module."""
    
    def test_get_metrics_returns_bytes(self):
        """get_metrics should return bytes."""
        from npu_proxy.metrics import get_metrics
        result = get_metrics()
        assert isinstance(result, bytes)
    
    def test_metrics_work_without_prometheus(self):
        """Metrics should gracefully handle missing prometheus_client."""
        with patch.dict('sys.modules', {'prometheus_client': None}):
            from npu_proxy.metrics import record_request, record_inference
            # Should not raise
            record_request("/v1/chat", "POST", 200)
            record_inference("tinyllama", "NPU", "chat", 1.5)
    
    def test_track_request_context_manager(self):
        """track_request should work as context manager."""
        from npu_proxy.metrics import track_request
        with track_request("/v1/chat"):
            pass  # Should not raise
    
    def test_record_routing_decision(self):
        """Should record routing decisions."""
        from npu_proxy.metrics import record_routing_decision
        # Should not raise
        record_routing_decision("CPU", "prompt_exceeds_npu_limit")
        record_routing_decision("NPU", "within_npu_limit")


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from npu_proxy.main import app
        return TestClient(app)
    
    def test_metrics_endpoint_returns_200(self, client):
        """GET /metrics should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_endpoint_content_type(self, client):
        """GET /metrics should return correct content type."""
        response = client.get("/metrics")
        # Either prometheus format or plain text
        assert "text/" in response.headers.get("content-type", "")
