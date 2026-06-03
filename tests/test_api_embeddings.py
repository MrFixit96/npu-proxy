"""Integration tests for OpenAI-compatible /v1/embeddings endpoint."""
import math

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from npu_proxy.main import app
from npu_proxy.inference.embedding_engine import (
    EmbeddingInferenceError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)

client = TestClient(app)


def make_engine(
    *,
    embeddings,
    device="CPU",
    dimensions=384,
    is_fallback=False,
    fallback_allowed=False,
    fallback_reason=None,
):
    engine = Mock()
    engine.embed_batch.return_value = embeddings
    engine.get_engine_info.return_value = {
        "model_name": "bge-small",
        "resolved_model": "bge-small",
        "requested_model": "bge-small",
        "dimensions": dimensions,
        "device": device,
        "requested_device": device,
        "is_production": not is_fallback,
        "is_fallback": is_fallback,
        "fallback_allowed": fallback_allowed,
        "available": (not is_fallback) or fallback_allowed,
        "backend": "hash" if is_fallback else "openvino",
        **({"fallback_reason": fallback_reason} if fallback_reason else {}),
    }
    return engine


class TestEmbeddingsEndpoint:
    def test_embeddings_returns_correct_dimensions(self):
        """POST /v1/embeddings returns embeddings with 384 dimensions"""
        engine = make_engine(embeddings=[[0.0] * 384])
        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post("/v1/embeddings", json={
                "model": "bge-small",
                "input": "test text"
            })
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"][0]["embedding"]) == 384

    def test_embeddings_batch_input(self):
        """POST /v1/embeddings handles batch input"""
        engine = make_engine(embeddings=[[0.0] * 384, [0.1] * 384, [0.2] * 384])
        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post("/v1/embeddings", json={
                "model": "bge-small",
                "input": ["text1", "text2", "text3"]
            })
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        for item in data["data"]:
            assert len(item["embedding"]) == 384

    def test_embeddings_usage_reports_tokens(self):
        """POST /v1/embeddings reports token usage"""
        engine = make_engine(embeddings=[[0.0] * 384])
        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post("/v1/embeddings", json={
                "model": "test",
                "input": "hello world"
            })
        assert response.status_code == 200
        data = response.json()
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0

    def test_embeddings_uses_requested_model_and_device(self):
        """POST /v1/embeddings should route using requested model/device safely."""
        engine = make_engine(embeddings=[[0.1, 0.2, 0.3]], device="NPU", dimensions=3)

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine) as mock_get_engine:
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
                headers={"X-NPU-Proxy-Device": "NPU"},
            )

        assert response.status_code == 200
        mock_get_engine.assert_called_once_with(model_name="bge-small", device="NPU")
        assert response.headers["X-NPU-Proxy-Device"] == "NPU"
        assert response.headers["X-NPU-Proxy-Model"] == "bge-small"
        assert response.headers["X-NPU-Proxy-Fallback"] == "false"

    def test_embeddings_rejects_unsupported_encoding_format(self):
        """POST /v1/embeddings rejects non-float encoding formats."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-small",
                "input": "test text",
                "encoding_format": "base64",
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_encoding_format"

    def test_embeddings_rejects_empty_input_before_engine_load(self):
        """POST /v1/embeddings rejects empty inputs with a clean OpenAI error."""
        with patch("npu_proxy.api.embeddings.get_embedding_engine") as mock_get_engine:
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": []},
            )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "empty_input"
        mock_get_engine.assert_not_called()

    def test_embeddings_rejects_oversized_batch_before_engine_load(self):
        """POST /v1/embeddings rejects oversized batches before inference."""
        with patch("npu_proxy.api.embeddings.get_embedding_engine") as mock_get_engine:
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": ["x"] * 129},
            )

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "embedding_batch_too_large"
        mock_get_engine.assert_not_called()

    def test_embeddings_returns_503_when_engine_unavailable(self):
        """POST /v1/embeddings should hard-fail when embeddings are unavailable."""
        with patch(
            "npu_proxy.api.embeddings.get_embedding_engine",
            side_effect=EmbeddingUnavailableError("embedding engine unavailable"),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
            )

        assert response.status_code == 503
        detail = response.json()["error"]
        assert detail["code"] == "embedding_unavailable"
        assert detail["type"] == "service_unavailable_error"

    def test_embeddings_returns_504_on_timeout(self):
        """POST /v1/embeddings should preserve timeout status mapping."""
        engine = make_engine(embeddings=[[0.1, 0.2, 0.3]], dimensions=3)
        engine.embed_batch.side_effect = EmbeddingTimeoutError("Embedding timed out after 60s")

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
            )

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "inference_timeout"

    def test_embeddings_returns_500_on_runtime_failure(self):
        """POST /v1/embeddings should return 500 after a loaded engine fails at runtime."""
        engine = make_engine(embeddings=[[0.1, 0.2, 0.3]], dimensions=3)
        engine.embed_batch.side_effect = EmbeddingInferenceError("Batch embedding failed: boom")

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
            )

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "embedding_failed"

    def test_embeddings_reject_runtime_fallback_success_after_loaded_engine_failure(self):
        """POST /v1/embeddings should not silently succeed after runtime fallback activates."""
        engine = make_engine(embeddings=[[0.1, 0.2, 0.3]], dimensions=3)
        engine_info = dict(engine.get_engine_info.return_value)

        def runtime_fallback(_inputs):
            engine_info.update(
                {
                    "device": "CPU",
                    "is_production": False,
                    "is_fallback": True,
                    "backend": "hash",
                    "fallback_mode": "runtime",
                    "fallback_reason": "Batch embedding failed: boom",
                }
            )
            return [[0.1, 0.2, 0.3]]

        engine.embed_batch.side_effect = runtime_fallback
        engine.get_engine_info.side_effect = lambda: dict(engine_info)

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
            )

        assert response.status_code == 500
        detail = response.json()["error"]
        assert detail["code"] == "embedding_failed"
        assert detail["message"] == "Embedding inference failed"
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"
        assert response.headers["X-NPU-Proxy-Fallback-Reason"] == "Embedding engine unavailable"

    @pytest.mark.parametrize("embeddings", [[[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]] * 3])
    def test_embeddings_reject_batch_length_mismatches(self, embeddings):
        """POST /v1/embeddings should reject mismatched batch result counts."""
        engine = make_engine(embeddings=embeddings, dimensions=3)

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": ["text1", "text2"]},
            )

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "embedding_failed"
        assert response.json()["error"]["message"] == "Embedding inference failed"

    def test_embeddings_rejects_fallback_by_default(self):
        """POST /v1/embeddings should not return success-shaped fallback by default."""
        engine = make_engine(
            embeddings=[[0.1, 0.2, 0.3]],
            dimensions=3,
            is_fallback=True,
            fallback_reason="Model load timed out",
        )

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
            )

        assert response.status_code == 503
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"
        assert response.headers["X-NPU-Proxy-Fallback-Reason"] == "Embedding engine unavailable"
        assert response.json()["error"]["code"] == "embedding_unavailable"

    def test_embeddings_allows_explicit_fallback(self):
        """POST /v1/embeddings may still serve fallback when explicitly enabled."""
        engine = make_engine(
            embeddings=[[0.1, 0.2, 0.3]],
            dimensions=3,
            is_fallback=True,
            fallback_allowed=True,
            fallback_reason="Model load timed out",
        )

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
            )

        assert response.status_code == 200
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"

    @pytest.mark.parametrize(
        "model_name",
        [
            "../outside",
            "..\\outside",
            "C:\\models\\outside",
            "\\\\server\\share\\model",
            "tinyllama",
        ],
    )
    def test_embeddings_rejects_path_like_model_identifiers(self, model_name):
        """POST /v1/embeddings rejects filesystem-style model names."""
        response = client.post(
            "/v1/embeddings",
            json={"model": model_name, "input": "test text"},
        )

        assert response.status_code == 400
        detail = response.json()["error"]
        assert detail["code"] == "invalid_embedding_model"
        assert detail["type"] == "invalid_request_error"

    def test_embeddings_sanitizes_multiline_fallback_reason_header(self):
        """POST /v1/embeddings keeps fallback headers HTTP-safe."""
        engine = make_engine(
            embeddings=[[0.1, 0.2, 0.3]],
            dimensions=3,
            is_fallback=True,
            fallback_reason="NPU plugin rejected model\nFalling back to CPU hash",
        )

        with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
            response = client.post(
                "/v1/embeddings",
                json={"model": "bge-small", "input": "test text"},
            )

        assert response.status_code == 503
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"
        assert response.headers["X-NPU-Proxy-Fallback-Reason"] == "Embedding engine unavailable"


def _assert_no_embedding_engine_headers(response):
    assert "x-npu-proxy-device" not in response.headers
    assert "x-npu-proxy-model" not in response.headers
    assert "x-npu-proxy-fallback" not in response.headers


@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_body", "has_request_id"),
    [
        (
            {"model": "bge-small", "input": []},
            400,
            {"error": {"message": "Embedding input must contain at least one item", "type": "invalid_request_error", "param": None, "code": "empty_input"}},
            True,
        ),
        (
            {"model": "bge-small", "input": ["ok", 123]},
            422,
            {"detail": [
                {"type": "string_type", "loc": ["body", "input", "str"], "msg": "Input should be a valid string", "input": ["ok", 123]},
                {"type": "string_type", "loc": ["body", "input", "list[str]", 1], "msg": "Input should be a valid string", "input": 123},
            ]},
            False,
        ),
        (
            {"model": "bge-small", "input": ["x"] * 129},
            413,
            {"error": {"message": "Embedding input batch is too large", "type": "request_too_large", "param": None, "code": "embedding_batch_too_large"}},
            True,
        ),
        (
            {"model": "bge-small", "input": "   "},
            400,
            {"error": {"message": "Embedding input text must not be empty", "type": "invalid_request_error", "param": None, "code": "empty_input"}},
            True,
        ),
    ],
    ids=["empty-list", "mixed-batch", "oversized-batch", "whitespace-string"],
)
def test_embeddings_request_validation_matrix(client, payload, expected_status, expected_body, has_request_id):
    """OpenAI embeddings validation should return stable envelopes before engine inference."""
    with patch("npu_proxy.api.embeddings.get_embedding_engine") as mock_get_engine:
        response = client.post("/v1/embeddings", json=payload)

    assert response.status_code == expected_status
    assert response.json() == expected_body
    assert ("x-request-id" in response.headers) is has_request_id
    _assert_no_embedding_engine_headers(response)
    mock_get_engine.assert_not_called()


def test_embeddings_malformed_json_returns_stable_422_envelope(client):
    """Malformed OpenAI embedding JSON should not reach the engine."""
    with patch("npu_proxy.api.embeddings.get_embedding_engine") as mock_get_engine:
        response = client.post(
            "/v1/embeddings",
            data="{bad",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "json_invalid",
                "loc": ["body", 1],
                "msg": "JSON decode error",
                "input": {},
                "ctx": {"error": "Expecting property name enclosed in double quotes"},
            }
        ]
    }
    assert "x-request-id" not in response.headers
    _assert_no_embedding_engine_headers(response)
    mock_get_engine.assert_not_called()


def test_embeddings_multi_input_batch_returns_one_finite_vector_per_input(client):
    """Regression: batch embedding responses must not drop inputs or fill zero-vector corruption."""
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ]
    engine = make_engine(embeddings=embeddings, dimensions=4)

    with patch("npu_proxy.api.embeddings.get_embedding_engine", return_value=engine):
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-small", "input": ["alpha", "beta", "gamma"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 3
    assert [item["index"] for item in body["data"]] == [0, 1, 2]
    for item, expected in zip(body["data"], embeddings):
        vector = item["embedding"]
        assert vector == expected
        assert len(vector) == 4
        assert all(math.isfinite(value) for value in vector)
    assert response.headers["x-npu-proxy-device"] == "CPU"
    assert response.headers["x-npu-proxy-model"] == "bge-small"
    assert response.headers["x-npu-proxy-fallback"] == "false"
