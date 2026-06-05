from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


class FakeRoutedEngine:
    def __init__(self, device: str | None, calls: list[str | None]) -> None:
        self.requested_device = (device or "NPU").upper()
        self.actual_device = self.requested_device
        self.model_name = "tinyllama"
        self.last_finish_reason = "stop"
        calls.append(device)

    def get_device_info(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "used_fallback": False,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        return f"response from {self.actual_device}"

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        for token in ["stream ", self.actual_device]:
            callback = kwargs.get("streamer_callback")
            if callback is not None:
                callback(token)
            yield token


@pytest.fixture
def app_client(monkeypatch) -> TestClient:
    from npu_proxy.routing.context_router import reset_context_router

    monkeypatch.setenv("NPU_PROXY_REAL_INFERENCE", "1")
    monkeypatch.setenv("NPU_PROXY_FALLBACK_DEVICE", "CPU")
    reset_context_router()
    from npu_proxy.main import app

    return TestClient(app)


@pytest.fixture
def engine_calls(monkeypatch) -> list[str | None]:
    import npu_proxy.inference.engine as engine_module

    calls: list[str | None] = []

    def fake_get_llm_engine(*, device: str | None = None, **kwargs) -> FakeRoutedEngine:
        return FakeRoutedEngine(device, calls)

    monkeypatch.setattr(engine_module, "get_llm_engine", fake_get_llm_engine)
    return calls


def reset_router() -> None:
    from npu_proxy.routing.context_router import reset_context_router

    reset_context_router()


def test_openai_chat_non_stream_routes_short_prompt_to_preferred_device(
    app_client: TestClient,
    engine_calls: list[str | None],
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1000")
    reset_router()

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert engine_calls == ["NPU"]
    assert response.headers["x-npu-proxy-device"] == "NPU"
    assert response.headers["x-npu-proxy-routed-device"] == "NPU"
    assert response.headers["x-npu-proxy-execution-device"] == "NPU"
    assert "x-npu-proxy-fallback-reason" not in response.headers


def test_openai_chat_stream_routes_short_prompt_to_preferred_device(
    app_client: TestClient,
    engine_calls: list[str | None],
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1000")
    reset_router()

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert engine_calls == ["NPU"]
    assert response.headers["x-npu-proxy-device"] == "NPU"
    assert response.headers["x-npu-proxy-routed-device"] == "NPU"
    assert response.headers["x-npu-proxy-execution-device"] == "NPU"
    assert "stream " in response.text
    assert "NPU" in response.text


def test_ollama_generate_non_stream_routes_short_prompt_to_preferred_device(
    app_client: TestClient,
    engine_calls: list[str | None],
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1000")
    reset_router()

    response = app_client.post(
        "/api/generate",
        json={"model": "tinyllama", "prompt": "Hi", "stream": False},
    )

    assert response.status_code == 200
    assert engine_calls == ["NPU"]
    assert response.headers["x-npu-proxy-device"] == "NPU"
    assert response.headers["x-npu-proxy-routed-device"] == "NPU"
    assert response.headers["x-npu-proxy-execution-device"] == "NPU"


def test_ollama_generate_stream_routes_short_prompt_to_preferred_device(
    app_client: TestClient,
    engine_calls: list[str | None],
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1000")
    reset_router()

    response = app_client.post(
        "/api/generate",
        json={"model": "tinyllama", "prompt": "Hi", "stream": True},
    )

    assert response.status_code == 200
    assert engine_calls == ["NPU"]
    assert response.headers["x-npu-proxy-device"] == "NPU"
    assert response.headers["x-npu-proxy-routed-device"] == "NPU"
    assert response.headers["x-npu-proxy-execution-device"] == "NPU"
    assert "stream " in response.text
    assert "NPU" in response.text


def test_ollama_chat_real_paths_route_to_preferred_device(
    app_client: TestClient,
    engine_calls: list[str | None],
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1000")
    reset_router()

    non_stream = app_client.post(
        "/api/chat",
        json={
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    stream = app_client.post(
        "/api/chat",
        json={
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert non_stream.status_code == 200
    assert stream.status_code == 200
    assert engine_calls == ["NPU", "NPU"]
    assert non_stream.headers["x-npu-proxy-device"] == "NPU"
    assert non_stream.headers["x-npu-proxy-routed-device"] == "NPU"
    assert non_stream.headers["x-npu-proxy-execution-device"] == "NPU"
    assert stream.headers["x-npu-proxy-device"] == "NPU"
    assert stream.headers["x-npu-proxy-routed-device"] == "NPU"
    assert stream.headers["x-npu-proxy-execution-device"] == "NPU"


def test_openai_chat_long_prompt_routes_to_fallback_device(
    app_client: TestClient,
    engine_calls: list[str | None],
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1")
    monkeypatch.setenv("NPU_PROXY_FALLBACK_DEVICE", "CPU")
    reset_router()

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "this prompt is too long"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert engine_calls == ["CPU"]
    assert response.headers["x-npu-proxy-device"] == "CPU"
    assert response.headers["x-npu-proxy-routed-device"] == "CPU"
    assert response.headers["x-npu-proxy-execution-device"] == "CPU"


def test_ollama_generate_long_prompt_routes_to_fallback_device(
    app_client: TestClient,
    engine_calls: list[str | None],
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1")
    monkeypatch.setenv("NPU_PROXY_FALLBACK_DEVICE", "CPU")
    reset_router()

    response = app_client.post(
        "/api/generate",
        json={
            "model": "tinyllama",
            "prompt": "this prompt is too long",
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert engine_calls == ["CPU"]
    assert response.headers["x-npu-proxy-device"] == "CPU"


def test_mock_mode_request_still_succeeds_unchanged(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"].startswith("Hello!")
    assert response.headers["x-npu-proxy-device"] == "NPU"
