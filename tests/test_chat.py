"""Tests for chat completions endpoint - TDD Phase 1.4"""
import json
from unittest.mock import Mock, patch

import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy.main import app
from npu_proxy.api.chat import generate_stream_real, ChatRequest, Message
from npu_proxy.inference.engine import InferenceTimeoutError


@pytest.mark.asyncio
async def test_chat_completions_returns_200():
    """POST /v1/chat/completions should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_returns_openai_format():
    """Response should match OpenAI chat completion format"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
    data = response.json()

    assert "id" in data
    assert "object" in data
    assert data["object"] == "chat.completion"
    assert "created" in data
    assert "model" in data
    assert "choices" in data
    assert len(data["choices"]) > 0

    choice = data["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert "role" in choice["message"]
    assert "content" in choice["message"]
    assert "finish_reason" in choice


@pytest.mark.asyncio
async def test_chat_completions_streaming():
    """POST with stream=true should return SSE stream"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)

            assert len(chunks) > 0
            assert all(not line.startswith("data: data:") for line in chunks)
            assert all(line == "data: [DONE]" or line.startswith("data: {") for line in chunks)
            assert chunks[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_chat_unknown_model_returns_openai_error_envelope():
    """Unknown models should return an OpenAI-style error body."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 404
    body = response.json()
    assert "detail" not in body
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "model"
    assert body["error"]["code"] == "model_not_found"
    assert "not found" in body["error"]["message"].lower()


@pytest.mark.asyncio
async def test_chat_inference_failure_returns_openai_error_envelope():
    """Handled inference failures should keep the OpenAI error contract."""
    transport = ASGITransport(app=app)
    with (
        patch.dict("os.environ", {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.api.chat.get_engine", side_effect=RuntimeError("boom")),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama-1.1b-chat-int4-ov",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

    assert response.status_code == 503
    body = response.json()
    assert "detail" not in body
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == "inference_error"
    assert body["error"]["message"] == "Inference failed"
    assert "boom" not in body["error"]["message"]


@pytest.mark.asyncio
async def test_chat_nonstream_inference_timeout_returns_504():
    """Handled inference timeouts should keep the OpenAI timeout mapping."""
    engine = Mock()
    engine.generate.side_effect = InferenceTimeoutError(1)
    transport = ASGITransport(app=app)

    with (
        patch.dict("os.environ", {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.api.chat.get_engine", return_value=engine),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama-1.1b-chat-int4-ov",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

    assert response.status_code == 504
    body = response.json()
    assert body["error"]["code"] == "timeout"
    assert "timed out" in body["error"]["message"].lower()


@pytest.mark.asyncio
async def test_chat_stream_failure_does_not_emit_done():
    """OpenAI streaming failures should emit an error frame without a success terminator."""
    engine = Mock()
    seen_abort_callback = None

    def broken_generate_stream(*args, **kwargs):
        nonlocal seen_abort_callback
        seen_abort_callback = kwargs.get("abort_callback")
        raise RuntimeError("boom")

    engine.generate_stream.side_effect = broken_generate_stream
    request = ChatRequest(
        model="tinyllama-1.1b-chat-int4-ov",
        messages=[Message(role="user", content="Hi")],
        stream=True,
    )

    with patch("npu_proxy.api.chat.get_engine", return_value=engine):
        chunks = []
        async for chunk in generate_stream_real(
            request,
            "chatcmpl-test",
            123,
            "Hi",
            {},
            "req_test",
        ):
            chunks.append(chunk)

    assert any('"error"' in chunk for chunk in chunks)
    assert not any("[DONE]" in chunk for chunk in chunks)
    assert not any('"finish_reason":"stop"' in chunk for chunk in chunks)
    error_payload = next(chunk for chunk in chunks if '"error"' in chunk)
    data_line = error_payload.removeprefix("data: ").strip()
    parsed = json.loads(data_line)
    assert parsed["error"]["type"] == "server_error"
    assert parsed["error"]["message"] == "Streaming inference failed"
    assert "boom" not in parsed["error"]["message"]
    assert callable(seen_abort_callback)


@pytest.mark.asyncio
async def test_chat_stream_startup_failure_emits_openai_error():
    """Engine startup failures should still emit a terminal OpenAI error frame."""
    request = ChatRequest(
        model="tinyllama-1.1b-chat-int4-ov",
        messages=[Message(role="user", content="Hi")],
        stream=True,
    )

    with patch("npu_proxy.api.chat.get_engine", side_effect=RuntimeError("boom")):
        chunks = []
        async for chunk in generate_stream_real(
            request,
            "chatcmpl-test",
            123,
            "Hi",
            {},
            "req_test",
        ):
            chunks.append(chunk)

    assert any('"role":"assistant"' in chunk for chunk in chunks)
    assert any('"error"' in chunk for chunk in chunks)
    assert not any("[DONE]" in chunk for chunk in chunks)
    error_payload = next(chunk for chunk in chunks if '"error"' in chunk)
    parsed = json.loads(error_payload.removeprefix("data: ").strip())
    assert parsed["error"]["type"] == "server_error"
    assert parsed["error"]["code"] == "streaming_error"
    assert parsed["error"]["message"] == "Streaming inference failed"

@pytest.mark.asyncio
async def test_chat_nonstream_reports_length_when_mock_truncated():
    """OpenAI non-streaming mock responses report length when max_tokens truncates output."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            },
        )

    data = response.json()
    assert response.status_code == 200
    assert data["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_chat_nonstream_reports_stop_for_natural_mock_completion():
    """OpenAI non-streaming mock responses report stop for natural completion."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    data = response.json()
    assert response.status_code == 200
    assert data["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_stream_reports_length_when_mock_truncated():
    """OpenAI streaming final chunk reports length when max_tokens truncates output."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "max_tokens": 1,
            },
        ) as response:
            lines = [line async for line in response.aiter_lines() if line.startswith("data: {")]

    final = json.loads(lines[-1].removeprefix("data: "))
    assert final["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_chat_stream_reports_stop_for_natural_mock_completion():
    """OpenAI streaming final chunk reports stop for natural completion."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as response:
            lines = [line async for line in response.aiter_lines() if line.startswith("data: {")]

    final = json.loads(lines[-1].removeprefix("data: "))
    assert final["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "stream",
    [False, True],
    ids=["nonstream", "stream"],
)
@pytest.mark.asyncio
async def test_chat_real_engine_contract_receives_prompt_params_and_reports_finish(
    async_client,
    fake_llm_engine_factory,
    known_llm_model,
    stream,
):
    """Real OpenAI path should wire rendered prompts and generation params into the engine."""
    fake_engine = fake_llm_engine_factory(
        stream_tokens=["Alpha", " Beta", " Gamma"],
        finish_reason="length",
        model_name=known_llm_model,
        actual_device="NPU",
        requested_device="NPU",
    )

    with (
        patch.dict("os.environ", {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.api.chat.get_engine", return_value=fake_engine),
        patch("npu_proxy.api.chat._get_execution_device", return_value="NPU"),
    ):
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": known_llm_model,
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": "Explain NPU."},
                ],
                "temperature": 0.25,
                "top_p": 0.6,
                "max_tokens": 2,
                "stream": stream,
            },
        )

    assert response.status_code == 200
    assert response.headers["x-npu-proxy-device"] == "NPU"
    assert response.headers["x-npu-proxy-route-reason"] == "single_engine_runtime"
    assert int(response.headers["x-npu-proxy-token-count"]) > 0
    assert response.headers["x-request-id"].startswith("req_")

    calls = fake_engine.generate_stream_calls if stream else fake_engine.generate_calls
    assert len(calls) == 1
    call = calls[0]
    assert "You are concise." in call["prompt"]
    assert "Explain NPU." in call["prompt"]
    assert call["max_new_tokens"] == 2
    assert call["temperature"] == 0.25

    if stream:
        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        assert all(not line.startswith("data: data:") for line in lines)
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        token_chunks = [chunk for chunk in chunks if chunk["choices"][0]["delta"].get("content")]
        final = chunks[-1]
        assert "".join(chunk["choices"][0]["delta"]["content"] for chunk in token_chunks) == "Alpha Beta"
        assert final["choices"][0]["finish_reason"] == "length"
    else:
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Alpha Beta"
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["usage"]["prompt_tokens"] == int(response.headers["x-npu-proxy-token-count"])
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]


@pytest.mark.parametrize(
    ("side_effect", "expected_status", "expected_code", "expected_message"),
    [
        (InferenceTimeoutError(3), 504, "timeout", "Inference timed out"),
        (ValueError("backend path C:\\secret\\model.bin exploded"), 500, "internal_error", "Internal inference error"),
    ],
    ids=["timeout", "non-runtime"],
)
@pytest.mark.asyncio
async def test_chat_real_engine_failures_return_sanitized_openai_envelopes(
    async_client,
    fake_llm_engine_factory,
    known_llm_model,
    side_effect,
    expected_status,
    expected_code,
    expected_message,
):
    """Real OpenAI error paths should not leak raw backend exception details."""
    fake_engine = fake_llm_engine_factory(model_name=known_llm_model)

    def fail_generate(*args, **kwargs):
        raise side_effect

    fake_engine.generate = fail_generate

    with (
        patch.dict("os.environ", {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.api.chat.get_engine", return_value=fake_engine),
        patch("npu_proxy.api.chat._get_execution_device", return_value="NPU"),
    ):
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": known_llm_model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == expected_status
    assert response.headers["x-request-id"].startswith("req_")
    body = response.json()
    assert "detail" not in body
    assert body == {
        "error": {
            "message": expected_message,
            "type": "server_error",
            "param": None,
            "code": expected_code,
        }
    }
    assert "secret" not in response.text


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.asyncio
async def test_chat_real_engine_contract_expected_top_p_and_timeout_wiring(
    async_client,
    fake_llm_engine_factory,
    known_llm_model,
    stream,
):
    """Expected contract: mapped sampling params and timeout should reach the engine."""
    from npu_proxy.config import DEFAULT_INFERENCE_TIMEOUT

    fake_engine = fake_llm_engine_factory(stream_tokens=["ok"], model_name=known_llm_model)

    with (
        patch.dict("os.environ", {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.api.chat.get_engine", return_value=fake_engine),
        patch("npu_proxy.api.chat._get_execution_device", return_value="NPU"),
    ):
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": known_llm_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "temperature": 0.2,
                "top_p": 0.55,
                "stream": stream,
            },
        )

    assert response.status_code == 200
    call = (fake_engine.generate_stream_calls if stream else fake_engine.generate_calls)[0]
    assert call["top_p"] == 0.55
    assert call["timeout"] == DEFAULT_INFERENCE_TIMEOUT


@pytest.mark.asyncio
async def test_chat_streaming_releases_slot_when_bookkeeping_fails(async_client, known_llm_model):
    """Regression: if post-acquire bookkeeping raises in the streaming real path,
    the acquired device slot must be released so the device lock is not leaked
    (which would otherwise wedge the device into permanent 503 device_busy)."""
    closed = {"count": 0}

    class FakeSlot:
        fallback_reason = None
        selected_device = "NPU"
        routed_device = "NPU"

        def __exit__(self, *args):
            closed["count"] += 1
            return False

    slot = FakeSlot()
    engine = Mock()

    def boom(*args, **kwargs):
        raise RuntimeError("metrics backend exploded")

    with (
        patch.dict("os.environ", {"NPU_PROXY_REAL_INFERENCE": "1"}),
        patch("npu_proxy.api.chat._open_routed_engine_slot", return_value=(engine, slot)),
        patch("npu_proxy.api.chat._execution_device_from_engine", return_value="NPU"),
        patch("npu_proxy.api.chat._record_routing_execution", side_effect=boom),
    ):
        with pytest.raises(RuntimeError, match="metrics backend exploded"):
            await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": known_llm_model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )

    assert closed["count"] == 1, "device slot was not released after bookkeeping failure"
