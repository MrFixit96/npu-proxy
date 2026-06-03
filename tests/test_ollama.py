"""Tests for Ollama-compatible endpoints - TDD Phase 1.7-2.0"""
import json
import os
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import Mock, patch
from npu_proxy import OLLAMA_VERSION
from npu_proxy.inference.engine import InferenceTimeoutError
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_ps_returns_200():
    """GET /api/ps should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/ps")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ps_returns_models_array():
    """GET /api/ps should return models array"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/ps")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_version_returns_200():
    """GET /api/version should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/version")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_version_returns_version_field():
    """GET /api/version should return version field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/version")
    data = response.json()
    assert "version" in data
    assert "npu-proxy" in data["version"]


@pytest.mark.asyncio
async def test_version_matches_shared_runtime_version():
    """GET /api/version should use the shared runtime version."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/version")
    data = response.json()
    assert data["version"] == OLLAMA_VERSION


@pytest.mark.asyncio
async def test_show_returns_200():
    """POST /api/show should return 200 for valid model"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/show",
            json={"model": "tinyllama-1.1b-chat-int4-ov"},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_show_returns_model_details():
    """POST /api/show should return model details"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/show",
            json={"model": "tinyllama-1.1b-chat-int4-ov"},
        )
    data = response.json()
    assert "details" in data
    assert "modelfile" in data
    assert "model_info" in data


@pytest.mark.asyncio
async def test_show_resolves_catalog_repo_id_metadata():
    """POST /api/show should preserve metadata when addressed by catalog repo ID."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/show",
            json={"model": "OpenVINO/phi-2-int4-ov"},
        )
    data = response.json()
    assert response.status_code == 200
    assert data["details"]["family"] == "phi"
    assert data["details"]["parameter_size"] == "2.7B"


@pytest.mark.asyncio
async def test_show_returns_404_for_invalid_model():
    """POST /api/show should return 404 for invalid model"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/show",
            json={"model": "nonexistent-model"},
        )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_generate_returns_200():
    """POST /api/generate should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": False,
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_generate_returns_response():
    """POST /api/generate should return response field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": False,
            },
        )
    data = response.json()
    assert "response" in data
    assert "done" in data
    assert data["done"] is True
    assert "model" in data


@pytest.mark.asyncio
async def test_generate_nonstream_inference_timeout_returns_504():
    """Non-streaming generate should preserve the timeout HTTP mapping."""
    engine = Mock()
    engine.generate.side_effect = InferenceTimeoutError(1)
    transport = ASGITransport(app=app)

    with patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}):
        with patch("npu_proxy.inference.engine.get_llm_engine", return_value=engine):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/generate",
                    json={
                        "model": "tinyllama-1.1b-chat-int4-ov",
                        "prompt": "Hello",
                        "stream": False,
                    },
                )

    assert response.status_code == 504
    assert "timed out" in response.json()["error"].lower()
    assert "detail" not in response.json()


@pytest.mark.asyncio
async def test_chat_returns_200():
    """POST /api/chat should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_returns_message():
    """POST /api/chat should return message field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
    data = response.json()
    assert "message" in data
    assert "done" in data
    assert data["done"] is True
    assert data["message"]["role"] == "assistant"
    assert len(data["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_chat_nonstream_invalid_model_returns_flat_error():
    """Non-streaming chat errors should use Ollama's flat error envelope."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 404
    assert response.json() == {"error": "Model 'nonexistent-model' not found"}


@pytest.mark.asyncio
async def test_chat_nonstream_uses_shared_render_chat_prompt():
    """POST /api/chat should render prompts through the shared chat template path."""
    engine = Mock()
    engine.generate.return_value = "Rendered reply"
    transport = ASGITransport(app=app)

    with patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}):
        with patch(
            "npu_proxy.api.ollama.render_chat_prompt",
            return_value=Mock(prompt="TOKENIZER PROMPT"),
        ) as mock_render, patch(
            "npu_proxy.inference.engine.get_llm_engine",
            return_value=engine,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": "tinyllama-1.1b-chat-int4-ov",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False,
                    },
                )

    assert response.status_code == 200
    mock_render.assert_called_once()
    assert engine.generate.call_args.args[0] == "TOKENIZER PROMPT"


@pytest.mark.asyncio
async def test_generate_stream_uses_raw_ndjson():
    """POST /api/generate streaming should return raw NDJSON, not SSE data: lines."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert "application/x-ndjson" in response.headers.get("content-type", "")

            lines = []
            async for line in response.aiter_lines():
                if line:
                    lines.append(line)
                if len(lines) >= 2:
                    break

    assert lines
    assert all(not line.startswith("data: ") for line in lines)
    parsed = json.loads(lines[0])
    assert parsed["model"] == "tinyllama-1.1b-chat-int4-ov"


@pytest.mark.asyncio
async def test_generate_stream_failure_does_not_emit_done_true():
    """Streaming errors should emit an error frame without a success terminator."""
    engine = Mock()
    seen_abort_callback = None

    def broken_generate_stream(*args, **kwargs):
        nonlocal seen_abort_callback
        seen_abort_callback = kwargs.get("abort_callback")
        raise RuntimeError("boom")

    engine.generate_stream.side_effect = broken_generate_stream
    transport = ASGITransport(app=app)

    with patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}):
        with patch(
            "npu_proxy.inference.engine.get_llm_execution_target",
            return_value={"device": "NPU"},
        ), patch(
            "npu_proxy.inference.engine.get_llm_engine",
            return_value=engine,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": "tinyllama-1.1b-chat-int4-ov",
                        "prompt": "Hello",
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    lines = [line async for line in response.aiter_lines() if line]

    assert any('"error"' in line for line in lines)
    assert not any('"done":true' in line or '"done": true' in line for line in lines)
    assert callable(seen_abort_callback)


@pytest.mark.asyncio
async def test_generate_stream_setup_failure_emits_terminal_error_chunk():
    """Setup failures should still surface as Ollama NDJSON error chunks."""
    transport = ASGITransport(app=app)

    with patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}):
        with patch(
            "npu_proxy.inference.engine.get_llm_engine",
            side_effect=RuntimeError("engine unavailable"),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": "tinyllama-1.1b-chat-int4-ov",
                        "prompt": "Hello",
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    lines = [line async for line in response.aiter_lines() if line]

    assert len(lines) == 1
    error = json.loads(lines[0])
    assert error["model"] == "tinyllama-1.1b-chat-int4-ov"
    assert error["code"] == "inference_failed"
    assert error["error"].startswith("Inference failed (request id: req-")
    assert "engine unavailable" not in error["error"]


@pytest.mark.asyncio
async def test_chat_stream_uses_raw_ndjson():
    """POST /api/chat streaming should return raw NDJSON, not SSE data: lines."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert "application/x-ndjson" in response.headers.get("content-type", "")

            lines = []
            async for line in response.aiter_lines():
                if line:
                    lines.append(line)
                if len(lines) >= 2:
                    break

    assert lines
    assert all(not line.startswith("data: ") for line in lines)
    parsed = json.loads(lines[0])
    assert parsed["model"] == "tinyllama-1.1b-chat-int4-ov"
    assert parsed["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_stream_failure_does_not_emit_done_true():
    """Chat streaming errors should emit an error frame without a success terminator."""
    engine = Mock()
    seen_abort_callback = None

    def broken_generate_stream(*args, **kwargs):
        nonlocal seen_abort_callback
        seen_abort_callback = kwargs.get("abort_callback")
        raise RuntimeError("boom")

    engine.generate_stream.side_effect = broken_generate_stream
    transport = ASGITransport(app=app)

    with patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}):
        with patch(
            "npu_proxy.inference.engine.get_llm_execution_target",
            return_value={"device": "NPU"},
        ), patch(
            "npu_proxy.inference.engine.get_llm_engine",
            return_value=engine,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST",
                    "/api/chat",
                    json={
                        "model": "tinyllama-1.1b-chat-int4-ov",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    lines = [line async for line in response.aiter_lines() if line]

    assert any('"error"' in line for line in lines)
    assert not any('"done":true' in line or '"done": true' in line for line in lines)
    assert callable(seen_abort_callback)


@pytest.mark.asyncio
async def test_chat_stream_setup_failure_emits_terminal_error_chunk():
    """Chat stream setup failures should still surface as NDJSON error chunks."""
    transport = ASGITransport(app=app)

    with patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}):
        with patch(
            "npu_proxy.inference.engine.get_llm_engine",
            side_effect=RuntimeError("engine unavailable"),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST",
                    "/api/chat",
                    json={
                        "model": "tinyllama-1.1b-chat-int4-ov",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    lines = [line async for line in response.aiter_lines() if line]

    assert len(lines) == 1
    error = json.loads(lines[0])
    assert error["model"] == "tinyllama-1.1b-chat-int4-ov"
    assert error["code"] == "inference_failed"
    assert error["error"].startswith("Inference failed (request id: req-")
    assert "engine unavailable" not in error["error"]


# Phase 2.5: Model Management Endpoint Tests


@pytest.mark.asyncio
async def test_pull_unknown_model_returns_404():
    """POST /api/pull with unknown model returns 404"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/pull", json={"name": "nonexistent-model-xyz"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_pull_requires_name():
    """POST /api/pull without name returns 422 (validation error)"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/pull", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
@patch("npu_proxy.models.downloader.download_model")
@patch("npu_proxy.models.downloader.is_model_downloaded")
async def test_pull_forwards_body_token_to_downloader(
    mock_is_model_downloaded,
    mock_download_model,
):
    """POST /api/pull should use an explicit body token for private pulls."""
    mock_is_model_downloaded.return_value = False
    mock_download_model.return_value = {
        "status": "success",
        "model": "tinyllama",
        "path": "/tmp/tinyllama",
        "source": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/pull",
            json={
                "name": "tinyllama",
                "stream": False,
                "huggingface_token": "hf_private_token",
            },
        )

    assert response.status_code == 200
    assert mock_is_model_downloaded.call_args.kwargs["token"] == "hf_private_token"
    assert mock_download_model.call_args.kwargs["token"] == "hf_private_token"


@pytest.mark.asyncio
@patch("npu_proxy.models.downloader.download_model")
@patch("npu_proxy.models.downloader.is_model_downloaded")
async def test_pull_forwards_bearer_token_to_downloader(
    mock_is_model_downloaded,
    mock_download_model,
):
    """POST /api/pull should accept Bearer auth for private pulls."""
    mock_is_model_downloaded.return_value = False
    mock_download_model.return_value = {
        "status": "success",
        "model": "tinyllama",
        "path": "/tmp/tinyllama",
        "source": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/pull",
            json={"name": "tinyllama", "stream": False},
            headers={"Authorization": "Bearer hf_private_token"},
        )

    assert response.status_code == 200
    assert mock_is_model_downloaded.call_args.kwargs["token"] == "hf_private_token"
    assert mock_download_model.call_args.kwargs["token"] == "hf_private_token"


@pytest.mark.asyncio
async def test_pull_rejects_conflicting_body_and_header_tokens():
    """POST /api/pull should reject ambiguous token sources."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/pull",
            json={
                "name": "tinyllama",
                "stream": False,
                "huggingface_token": "hf_body_token",
            },
            headers={"Authorization": "Bearer hf_header_token"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
@patch("npu_proxy.models.downloader.get_download_progress")
@patch("npu_proxy.models.downloader.is_model_downloaded")
async def test_pull_streaming_uses_explicit_anonymous_mode(
    mock_is_model_downloaded,
    mock_get_download_progress,
):
    """Streaming pulls should still pass anonymous auth, cache path, and target-specific requirements."""
    mock_is_model_downloaded.return_value = False
    mock_get_download_progress.return_value = iter([{"status": "success"}])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/pull", json={"name": "tinyllama", "stream": True})

    assert response.status_code == 200
    assert mock_is_model_downloaded.call_args.kwargs["token"] is False
    assert mock_get_download_progress.call_args.kwargs["token"] is False
    assert mock_get_download_progress.call_args.kwargs["required_files"] == (
        "openvino_model.xml",
        "openvino_model.bin",
    )
    assert str(mock_get_download_progress.call_args.args[1]).endswith("tinyllama-1.1b-chat-int4-ov")


@pytest.mark.asyncio
@patch("npu_proxy.models.downloader.is_model_downloaded")
async def test_pull_stream_already_downloaded_returns_ndjson_progress(mock_is_model_downloaded):
    """Streaming pulls should still stream progress when the model is cached."""
    mock_is_model_downloaded.return_value = True
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/pull", json={"name": "tinyllama", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.strip().splitlines()]
    assert lines[0] == {"status": "pulling manifest"}
    assert lines[1]["status"] == "success"
    assert lines[1]["message"] == "Model already downloaded"


@pytest.mark.asyncio
@patch("npu_proxy.models.registry.scan_available_models")
async def test_tags_lists_local_openvino_models(mock_scan_available_models):
    """GET /api/tags should list downloaded registry models in Ollama format."""
    mock_scan_available_models.return_value = [
        {
            "id": "tinyllama",
            "storage_key": "tinyllama-1.1b-chat-int4-ov",
            "format": "openvino-ir",
            "backend": "openvino",
            "family": "llama",
            "parameter_size": "1.1B",
            "quantization": "INT4",
            "size": 123,
            "digest": "sha256:test",
        }
    ]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/tags")

    assert response.status_code == 200
    data = response.json()
    assert data["models"][0]["name"] == "tinyllama"
    assert data["models"][0]["details"]["format"] == "openvino"
    assert data["models"][0]["details"]["quantization_level"] == "INT4"


@pytest.mark.asyncio
async def test_search_returns_200():
    """GET /api/search returns 200"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_search_returns_models_list():
    """GET /api/search returns models array"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_search_returns_pagination_fields():
    """GET /api/search returns pagination fields"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search")
    data = response.json()
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    assert "has_more" in data


@pytest.mark.asyncio
async def test_search_invalid_sort_returns_400():
    """GET /api/search with invalid sort returns 400"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search?sort=invalid")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_search_invalid_type_returns_400():
    """GET /api/search with invalid type returns 400"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/search?type=invalid")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_known_models_returns_200():
    """GET /api/models/known returns 200"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models/known")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_known_models_returns_list():
    """GET /api/models/known returns models list"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models/known")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


@pytest.mark.asyncio
async def test_known_models_have_required_fields():
    """Each known model has ollama_name, huggingface_repo, quantization"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/models/known")
    data = response.json()
    for model in data["models"]:
        assert "ollama_name" in model
        assert "huggingface_repo" in model
        assert "quantization" in model

@pytest.mark.asyncio
async def test_ollama_generate_nonstream_done_reason_length_when_truncated():
    """Ollama /api/generate non-streaming reports done_reason=length on token limit."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 1},
            },
        )

    data = response.json()
    assert response.status_code == 200
    assert data["done"] is True
    assert data["done_reason"] == "length"


@pytest.mark.asyncio
async def test_ollama_generate_nonstream_done_reason_stop_for_natural_completion():
    """Ollama /api/generate non-streaming reports done_reason=stop naturally."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": False,
            },
        )

    data = response.json()
    assert response.status_code == 200
    assert data["done_reason"] == "stop"


@pytest.mark.asyncio
async def test_ollama_generate_stream_done_reason_length_when_truncated():
    """Ollama /api/generate streaming final frame reports done_reason=length."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Hello",
                "stream": True,
                "options": {"num_predict": 1},
            },
        ) as response:
            lines = [json.loads(line) async for line in response.aiter_lines() if line]

    assert lines[-1]["done"] is True
    assert lines[-1]["done_reason"] == "length"


@pytest.mark.asyncio
async def test_ollama_chat_stream_done_reason_stop_for_natural_completion():
    """Ollama /api/chat streaming final frame reports done_reason=stop naturally."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        ) as response:
            lines = [json.loads(line) async for line in response.aiter_lines() if line]

    assert lines[-1]["done"] is True
    assert lines[-1]["done_reason"] == "stop"


@pytest.mark.asyncio
async def test_ollama_chat_nonstream_done_reason_length_when_truncated():
    """Ollama /api/chat non-streaming reports done_reason=length on token limit."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
        )

    data = response.json()
    assert response.status_code == 200
    assert data["done_reason"] == "length"


@pytest.mark.parametrize(
    "stream",
    [False, True],
    ids=["nonstream", "stream"],
)
@pytest.mark.asyncio
async def test_ollama_chat_real_engine_contract_receives_prompt_params_and_reports_done_reason(
    async_client,
    fake_llm_engine_factory,
    known_ollama_model,
    stream,
):
    """Real Ollama chat path should wire rendered prompts and params into the engine."""
    fake_engine = fake_llm_engine_factory(
        stream_tokens=["Alpha", " Beta", " Gamma"],
        finish_reason="length",
        model_name=known_ollama_model,
        actual_device="NPU",
        requested_device="NPU",
    )

    with (
        patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.inference.engine.get_llm_engine", return_value=fake_engine),
        patch("npu_proxy.api.ollama._get_execution_device", return_value="NPU"),
    ):
        response = await async_client.post(
            "/api/chat",
            json={
                "model": known_ollama_model,
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": "Explain NPU."},
                ],
                "stream": stream,
                "options": {"num_predict": 2, "temperature": 0.25, "top_p": 0.6},
            },
        )

    assert response.status_code == 200
    assert response.headers["x-npu-proxy-device"] == "NPU"
    assert response.headers["x-npu-proxy-route-reason"] == "single_engine_runtime"
    assert int(response.headers["x-npu-proxy-token-count"]) > 0
    assert response.headers["x-request-id"].startswith("req-")

    calls = fake_engine.generate_stream_calls if stream else fake_engine.generate_calls
    assert len(calls) == 1
    call = calls[0]
    assert "You are concise." in call["prompt"]
    assert "Explain NPU." in call["prompt"]
    assert call["max_new_tokens"] == 2
    assert call["temperature"] == 0.25

    if stream:
        lines = [json.loads(line) for line in response.text.splitlines() if line]
        assert len(lines) == 3
        assert [line["done"] for line in lines] == [False, False, True]
        assert all("error" not in line for line in lines)
        assert lines[0]["message"] == {"role": "assistant", "content": "Alpha"}
        assert lines[1]["message"] == {"role": "assistant", "content": " Beta"}
        assert lines[-1]["message"] == {"role": "assistant", "content": ""}
        assert lines[-1]["done_reason"] == "length"
        assert lines[-1]["eval_count"] == 2
    else:
        data = response.json()
        assert data["message"] == {"role": "assistant", "content": "Alpha Beta"}
        assert data["done"] is True
        assert data["done_reason"] == "length"
        assert data["eval_count"] == 2


@pytest.mark.parametrize(
    ("side_effect", "expected_status", "expected_code", "expected_prefix"),
    [
        (InferenceTimeoutError(3), 504, "inference_timeout", "Inference timed out"),
        (ValueError("backend path C:\\secret\\model.bin exploded"), 500, "inference_failed", "Internal inference error"),
    ],
    ids=["timeout", "non-runtime"],
)
@pytest.mark.asyncio
async def test_ollama_chat_real_engine_failures_return_sanitized_envelopes(
    async_client,
    fake_llm_engine_factory,
    known_ollama_model,
    side_effect,
    expected_status,
    expected_code,
    expected_prefix,
):
    """Real Ollama error paths should use flat sanitized error envelopes."""
    fake_engine = fake_llm_engine_factory(model_name=known_ollama_model)

    def fail_generate(*args, **kwargs):
        raise side_effect

    fake_engine.generate = fail_generate

    with (
        patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.inference.engine.get_llm_engine", return_value=fake_engine),
        patch("npu_proxy.api.ollama._get_execution_device", return_value="NPU"),
    ):
        response = await async_client.post(
            "/api/chat",
            json={
                "model": known_ollama_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

    assert response.status_code == expected_status
    assert response.headers["x-request-id"].startswith("req-")
    body = response.json()
    assert body["code"] == expected_code
    assert body["error"].startswith(f"{expected_prefix} (request id: req-")
    assert "secret" not in response.text


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.asyncio
async def test_ollama_chat_real_engine_contract_expected_top_p_and_timeout_wiring(
    async_client,
    fake_llm_engine_factory,
    known_ollama_model,
    stream,
):
    """Expected contract: mapped Ollama params and timeout should reach the engine."""
    fake_engine = fake_llm_engine_factory(stream_tokens=["ok"], model_name=known_ollama_model)

    with (
        patch.dict(os.environ, {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.inference.engine.get_llm_engine", return_value=fake_engine),
        patch("npu_proxy.api.ollama._get_execution_device", return_value="NPU"),
    ):
        response = await async_client.post(
            "/api/chat",
            json={
                "model": known_ollama_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": stream,
                "options": {"num_predict": 1, "temperature": 0.2, "top_p": 0.55},
            },
        )

    assert response.status_code == 200
    call = (fake_engine.generate_stream_calls if stream else fake_engine.generate_calls)[0]
    assert call["top_p"] == 0.55
    assert call["timeout"] == 180.0
