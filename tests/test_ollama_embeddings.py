"""Tests for Ollama-native embedding endpoints (/api/embed and /api/embeddings)."""
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
    batch_embeddings=None,
    single_embedding=None,
    device="CPU",
    dimensions=384,
    is_fallback=False,
    fallback_allowed=False,
    fallback_reason=None,
):
    engine = Mock()
    if batch_embeddings is not None:
        engine.embed_batch.return_value = batch_embeddings
    if single_embedding is not None:
        engine.embed.return_value = single_embedding
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


class TestOllamaEmbedEndpoint:
    """Tests for POST /api/embed (Ollama current format)."""

    def test_embed_single_input(self):
        """POST /api/embed with single input returns nested embeddings array."""
        engine = make_engine(batch_embeddings=[[0.0] * 384])
        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post("/api/embed", json={
                "model": "bge-small",
                "input": "test text"
            })
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 384

    def test_embed_multiple_inputs(self):
        """POST /api/embed with multiple inputs returns multiple embeddings."""
        engine = make_engine(batch_embeddings=[[0.0] * 384, [0.1] * 384, [0.2] * 384])
        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post("/api/embed", json={
                "model": "bge-small",
                "input": ["text1", "text2", "text3"]
            })
        assert response.status_code == 200
        assert len(response.json()["embeddings"]) == 3

    def test_embed_returns_duration_stats(self):
        """POST /api/embed returns timing statistics."""
        engine = make_engine(batch_embeddings=[[0.0] * 384])
        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post("/api/embed", json={
                "model": "bge-small",
                "input": "test"
            })
        assert response.status_code == 200
        data = response.json()
        assert "total_duration" in data
        assert "prompt_eval_count" in data

    def test_embed_model_in_response(self):
        """POST /api/embed includes model name in response."""
        engine = make_engine(batch_embeddings=[[0.0] * 384])
        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post("/api/embed", json={
                "model": "bge-small",
                "input": "test"
            })
        assert response.status_code == 200
        assert response.json()["model"] == "bge-small"

    def test_embed_uses_requested_model_and_device(self):
        """POST /api/embed should honor requested model/device safely."""
        engine = make_engine(batch_embeddings=[[0.1, 0.2]], device="NPU", dimensions=2)
        engine.get_engine_info.return_value.update(
            {
                "model_name": "e5-large",
                "resolved_model": "e5-large",
                "requested_model": "e5-large",
            }
        )

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine) as mock_get_engine:
            response = client.post(
                "/api/embed",
                json={
                    "model": "e5-large",
                    "input": "test",
                    "options": {"device": "NPU"},
                },
            )

        assert response.status_code == 200
        mock_get_engine.assert_called_once_with(model_name="e5-large", device="NPU")
        assert response.headers["X-NPU-Proxy-Device"] == "NPU"
        assert response.headers["X-NPU-Proxy-Model"] == "e5-large"

    def test_embed_rejects_unsupported_dimensions(self):
        """POST /api/embed rejects mismatched output dimensions."""
        engine = make_engine(batch_embeddings=[[0.0] * 384])

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test", "dimensions": 768},
            )

        assert response.status_code == 400

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
    def test_embed_rejects_path_like_model_identifiers(self, model_name):
        """POST /api/embed rejects filesystem-style model names."""
        response = client.post(
            "/api/embed",
            json={"model": model_name, "input": "test"},
        )

        assert response.status_code == 400

    def test_embed_returns_503_when_engine_unavailable(self):
        """POST /api/embed should hard-fail when embeddings are unavailable."""
        with patch(
            "npu_proxy.inference.embedding_engine.get_embedding_engine",
            side_effect=EmbeddingUnavailableError("embedding engine unavailable"),
        ):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 503

    def test_embed_rejects_unknown_device_override(self):
        """POST /api/embed should reject unsupported device overrides before inference."""
        response = client.post(
            "/api/embed",
            json={"model": "bge-small", "input": "test", "options": {"device": "TPU"}},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_embedding_device"

    def test_embed_returns_504_on_timeout(self):
        """POST /api/embed should preserve timeout status mapping."""
        engine = make_engine(batch_embeddings=[[0.1, 0.2]], dimensions=2)
        engine.embed_batch.side_effect = EmbeddingTimeoutError("Embedding timed out after 60s")

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 504
        data = response.json()
        assert data["code"] == "embedding_timeout"
        assert data["error"].startswith("Embedding timed out (request id: req-")
        assert "60s" not in data["error"]

    def test_embed_returns_500_on_runtime_failure(self):
        """POST /api/embed should return 500 after a loaded engine fails at runtime."""
        engine = make_engine(batch_embeddings=[[0.1, 0.2]], dimensions=2)
        engine.embed_batch.side_effect = EmbeddingInferenceError("Batch embedding failed: boom")

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "embedding_failed"
        assert data["error"].startswith("Embedding failed (request id: req-")
        assert "boom" not in data["error"]

    def test_embed_rejects_runtime_fallback_success_after_loaded_engine_failure(self):
        """POST /api/embed should not silently succeed after runtime fallback activates."""
        engine = make_engine(batch_embeddings=[[0.1, 0.2]], dimensions=2)
        engine_info = dict(engine.get_engine_info.return_value)

        def runtime_fallback(_texts):
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
            return [[0.1, 0.2]]

        engine.embed_batch.side_effect = runtime_fallback
        engine.get_engine_info.side_effect = lambda: dict(engine_info)

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "embedding_failed"
        assert data["error"].startswith("Embedding failed (request id: req-")
        assert "boom" not in data["error"]
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"
        assert response.headers["X-NPU-Proxy-Fallback-Reason"] == "Batch embedding failed: boom"

    def test_embed_rejects_batch_length_mismatches(self):
        """POST /api/embed should reject over-returned batch embeddings."""
        engine = make_engine(batch_embeddings=[[0.0] * 2, [0.1] * 2], dimensions=2)

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "embedding_failed"
        assert data["error"].startswith("Embedding failed (request id: req-")
        assert "Embedding batch returned" not in data["error"]

    def test_embed_rejects_fallback_by_default(self):
        """POST /api/embed should not return success-shaped fallback by default."""
        engine = make_engine(
            batch_embeddings=[[0.1, 0.2]],
            dimensions=2,
            is_fallback=True,
            fallback_reason="NPU plugin rejected model",
        )

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 503
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"

    def test_embed_allows_explicit_fallback(self):
        """POST /api/embed may still serve fallback when explicitly enabled."""
        engine = make_engine(
            batch_embeddings=[[0.1, 0.2]],
            dimensions=2,
            is_fallback=True,
            fallback_allowed=True,
            fallback_reason="NPU plugin rejected model",
        )

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 200
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"

    def test_embed_sanitizes_multiline_fallback_reason_header(self):
        """POST /api/embed keeps fallback headers HTTP-safe."""
        engine = make_engine(
            batch_embeddings=[[0.1, 0.2]],
            dimensions=2,
            is_fallback=True,
            fallback_reason="NPU plugin rejected model\r\nUsing CPU hash fallback",
        )

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embed",
                json={"model": "bge-small", "input": "test"},
            )

        assert response.status_code == 503
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"
        assert response.headers["X-NPU-Proxy-Fallback-Reason"] == (
            "NPU plugin rejected model Using CPU hash fallback"
        )


class TestOllamaEmbeddingsLegacyEndpoint:
    """Tests for POST /api/embeddings (Ollama legacy format)."""

    def test_embeddings_legacy_single_prompt(self):
        """POST /api/embeddings with prompt returns single flat embedding."""
        engine = make_engine(single_embedding=[0.0] * 384)
        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post("/api/embeddings", json={
                "model": "bge-small",
                "prompt": "test text"
            })
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data  # NOT "embeddings"
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) == 384

    def test_embeddings_legacy_returns_single_array(self):
        """POST /api/embeddings returns flat array, not nested."""
        engine = make_engine(single_embedding=[0.0] * 384)
        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post("/api/embeddings", json={
                "model": "bge-small",
                "prompt": "single prompt only"
            })
        assert response.status_code == 200
        # Verify it's a flat array, not nested
        embedding = response.json()["embedding"]
        assert isinstance(embedding[0], float)  # First element is float, not list

    def test_embeddings_legacy_uses_requested_model_and_device(self):
        """POST /api/embeddings should honor requested model/device safely."""
        engine = make_engine(single_embedding=[0.1, 0.2], device="GPU", dimensions=2)

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine) as mock_get_engine:
            response = client.post(
                "/api/embeddings",
                json={
                    "model": "bge-small",
                    "prompt": "test text",
                    "options": {"device": "GPU"},
                },
            )

        assert response.status_code == 200
        mock_get_engine.assert_called_once_with(model_name="bge-small", device="GPU")
        assert response.headers["X-NPU-Proxy-Device"] == "GPU"

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
    def test_embeddings_legacy_rejects_path_like_model_identifiers(self, model_name):
        """POST /api/embeddings rejects filesystem-style model names."""
        response = client.post(
            "/api/embeddings",
            json={"model": model_name, "prompt": "test"},
        )

        assert response.status_code == 400

    def test_embeddings_legacy_rejects_fallback_by_default(self):
        """POST /api/embeddings should hard-fail instead of returning fallback vectors."""
        engine = make_engine(
            single_embedding=[0.1, 0.2],
            dimensions=2,
            is_fallback=True,
            fallback_reason="Model load timed out",
        )

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embeddings",
                json={"model": "bge-small", "prompt": "test"},
            )

        assert response.status_code == 503
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"

    def test_embeddings_legacy_returns_500_on_runtime_failure(self):
        """POST /api/embeddings should return 500 when the loaded engine fails at runtime."""
        engine = make_engine(single_embedding=[0.1, 0.2], dimensions=2)
        engine.embed.side_effect = EmbeddingInferenceError("Embedding generation failed: boom")

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embeddings",
                json={"model": "bge-small", "prompt": "test"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "embedding_failed"
        assert data["error"].startswith("Embedding failed (request id: req-")
        assert "boom" not in data["error"]

    def test_embeddings_legacy_returns_504_on_timeout(self):
        """POST /api/embeddings should preserve runtime timeout mapping."""
        engine = make_engine(single_embedding=[0.1, 0.2], dimensions=2)
        engine.embed.side_effect = EmbeddingTimeoutError("Embedding timed out after 60s")

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embeddings",
                json={"model": "bge-small", "prompt": "test"},
            )

        assert response.status_code == 504
        data = response.json()
        assert data["code"] == "embedding_timeout"
        assert data["error"].startswith("Embedding timed out (request id: req-")
        assert "60s" not in data["error"]

    def test_embeddings_legacy_rejects_runtime_fallback_success_after_loaded_engine_failure(self):
        """POST /api/embeddings should not silently succeed after runtime fallback activates."""
        engine = make_engine(single_embedding=[0.1, 0.2], dimensions=2)
        engine_info = dict(engine.get_engine_info.return_value)

        def runtime_fallback(_prompt):
            engine_info.update(
                {
                    "device": "CPU",
                    "is_production": False,
                    "is_fallback": True,
                    "backend": "hash",
                    "fallback_mode": "runtime",
                    "fallback_reason": "Embedding generation failed: boom",
                }
            )
            return [0.1, 0.2]

        engine.embed.side_effect = runtime_fallback
        engine.get_engine_info.side_effect = lambda: dict(engine_info)

        with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
            response = client.post(
                "/api/embeddings",
                json={"model": "bge-small", "prompt": "test"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "embedding_failed"
        assert data["error"].startswith("Embedding failed (request id: req-")
        assert "boom" not in data["error"]
        assert response.headers["X-NPU-Proxy-Fallback"] == "true"
        assert response.headers["X-NPU-Proxy-Fallback-Reason"] == "Embedding generation failed: boom"


def _assert_no_ollama_embedding_engine_headers(response):
    assert "x-npu-proxy-device" not in response.headers
    assert "x-npu-proxy-model" not in response.headers
    assert "x-npu-proxy-fallback" not in response.headers


@pytest.mark.parametrize(
    ("path", "payload", "expected_status", "expected_body"),
    [
        (
            "/api/embed",
            {"model": "bge-small", "input": ["ok", 123]},
            422,
            {"detail": [
                {"type": "string_type", "loc": ["body", "input", "str"], "msg": "Input should be a valid string", "input": ["ok", 123]},
                {"type": "string_type", "loc": ["body", "input", "list[str]", 1], "msg": "Input should be a valid string", "input": 123},
            ]},
        ),
        (
            "/api/embeddings",
            {"model": "bge-small", "prompt": ["ok", 123]},
            422,
            {"detail": [
                {"type": "string_type", "loc": ["body", "prompt"], "msg": "Input should be a valid string", "input": ["ok", 123]},
            ]},
        ),
    ],
    ids=["embed-mixed-batch", "legacy-mixed-prompt"],
)
def test_ollama_embedding_type_validation_matrix(client, path, payload, expected_status, expected_body):
    """Ollama embedding malformed types should return stable validation envelopes."""
    with patch("npu_proxy.inference.embedding_engine.get_embedding_engine") as mock_get_engine:
        response = client.post(path, json=payload)

    assert response.status_code == expected_status
    assert response.json() == expected_body
    assert "x-request-id" not in response.headers
    _assert_no_ollama_embedding_engine_headers(response)
    mock_get_engine.assert_not_called()


@pytest.mark.parametrize("path", ["/api/embed", "/api/embeddings"])
def test_ollama_embedding_malformed_json_returns_stable_422_envelope(client, path):
    """Malformed Ollama embedding JSON should not reach the engine."""
    with patch("npu_proxy.inference.embedding_engine.get_embedding_engine") as mock_get_engine:
        response = client.post(path, data="{bad", headers={"content-type": "application/json"})

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
    _assert_no_ollama_embedding_engine_headers(response)
    mock_get_engine.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_code", "expected_prefix"),
    [
        ({"model": "bge-small", "input": []}, 400, "empty_input", "Embedding input must contain at least one item"),
        ({"model": "bge-small", "input": ["x"] * 129}, 413, "embedding_batch_too_large", "Embedding input batch is too large"),
        ({"model": "bge-small", "input": "   "}, 400, "empty_input", "Embedding input text must not be empty"),
    ],
    ids=["empty-list", "oversized-batch", "whitespace-string"],
)
def test_ollama_embed_request_validation_matrix_expected_contract(client, payload, expected_status, expected_code, expected_prefix):
    """Expected Ollama /api/embed validation contract for malformed semantic inputs."""
    with patch("npu_proxy.inference.embedding_engine.get_embedding_engine") as mock_get_engine:
        response = client.post("/api/embed", json=payload)

    assert response.status_code == expected_status
    body = response.json()
    assert body["code"] == expected_code
    assert body["error"].startswith(f"{expected_prefix} (request id: req-")
    assert response.headers["x-request-id"].startswith("req-")
    _assert_no_ollama_embedding_engine_headers(response)
    mock_get_engine.assert_not_called()


def test_ollama_legacy_embeddings_rejects_whitespace_prompt_expected_contract(client):
    """Expected Ollama legacy validation contract for whitespace-only prompts."""
    with patch("npu_proxy.inference.embedding_engine.get_embedding_engine") as mock_get_engine:
        response = client.post("/api/embeddings", json={"model": "bge-small", "prompt": "   "})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "empty_input"
    assert body["error"].startswith("Embedding input text must not be empty (request id: req-")
    assert response.headers["x-request-id"].startswith("req-")
    _assert_no_ollama_embedding_engine_headers(response)
    mock_get_engine.assert_not_called()


def test_ollama_embed_multi_input_batch_returns_one_finite_vector_per_input(client):
    """Regression: /api/embed must return exactly one finite vector for each input."""
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    engine = make_engine(batch_embeddings=embeddings, dimensions=3)

    with patch("npu_proxy.inference.embedding_engine.get_embedding_engine", return_value=engine):
        response = client.post(
            "/api/embed",
            json={"model": "bge-small", "input": ["alpha", "beta", "gamma"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["embeddings"] == embeddings
    assert len(body["embeddings"]) == 3
    for vector in body["embeddings"]:
        assert len(vector) == 3
        assert all(math.isfinite(value) for value in vector)
    assert response.headers["x-npu-proxy-device"] == "CPU"
    assert response.headers["x-npu-proxy-model"] == "bge-small"
    assert response.headers["x-npu-proxy-fallback"] == "false"
