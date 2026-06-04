from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


class FakeEngine:
    def __init__(self, device: str, *, fail: bool = False) -> None:
        self.actual_device = device.upper()
        self.requested_device = self.actual_device
        self.model_name = "tinyllama"
        self.last_finish_reason = "stop"
        self.fail = fail

    def get_device_info(self) -> dict[str, object]:
        return {
            "actual_device": self.actual_device,
            "requested_device": self.requested_device,
            "used_fallback": False,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        if self.fail:
            raise RuntimeError("boom")
        return f"response from {self.actual_device}"

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        for token in ("stream ", self.actual_device):
            callback = kwargs.get("streamer_callback")
            if callback is not None:
                callback(token)
            yield token


@pytest.fixture
def real_client(monkeypatch) -> TestClient:
    from npu_proxy.routing.context_router import reset_context_router

    monkeypatch.setenv("NPU_PROXY_REAL_INFERENCE", "1")
    monkeypatch.setenv("NPU_PROXY_TOKEN_LIMIT", "1000")
    monkeypatch.setenv("NPU_PROXY_FALLBACK_DEVICE", "CPU")
    monkeypatch.setenv("NPU_PROXY_DEVICE_QUEUE_TIMEOUT", "0.001")
    reset_context_router()
    from npu_proxy.main import app

    return TestClient(app)


def _patch_fake_engines(monkeypatch, *, fail: bool = False) -> list[str]:
    import npu_proxy.inference.engine as engine_module

    calls: list[str] = []

    def fake_get_llm_engine(*, device: str | None = None, **kwargs) -> FakeEngine:
        selected = (device or "NPU").upper()
        calls.append(selected)
        return FakeEngine(selected, fail=fail)

    monkeypatch.setattr(engine_module, "get_llm_engine", fake_get_llm_engine)
    return calls


def test_same_device_slot_serializes_concurrent_requests() -> None:
    from npu_proxy.inference.engine import acquire_device_slot

    first_acquired = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()
    errors: list[BaseException] = []

    def first_worker() -> None:
        try:
            with acquire_device_slot("NPU", timeout=1.0):
                first_acquired.set()
                release_first.wait(timeout=2.0)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def second_worker() -> None:
        try:
            first_acquired.wait(timeout=2.0)
            with acquire_device_slot("NPU", timeout=1.0):
                second_acquired.set()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    second.start()

    assert first_acquired.wait(timeout=2.0)
    assert not second_acquired.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert second_acquired.is_set()


def test_busy_timeout_returns_device_busy() -> None:
    from npu_proxy.inference.engine import DeviceBusyError, acquire_device_slot

    with acquire_device_slot("NPU", timeout=1.0):
        with pytest.raises(DeviceBusyError) as exc_info:
            with acquire_device_slot("NPU", timeout=0.0):
                pass

    assert exc_info.value.reason == "device_busy"
    assert exc_info.value.status_code == 503


def test_fallback_on_busy_uses_next_available_device(monkeypatch) -> None:
    import npu_proxy.inference.engine as engine_module
    from npu_proxy.inference.engine import acquire_device_slot, open_routed_engine_slot

    calls = _patch_fake_engines(monkeypatch)
    monkeypatch.setattr(engine_module, "get_available_devices", lambda: ["NPU", "CPU"])

    with acquire_device_slot("NPU", timeout=1.0):
        engine, slot = open_routed_engine_slot("NPU", timeout=0.0, fallback_on_busy=True)
        try:
            assert engine.actual_device == "CPU"
        finally:
            slot.__exit__(None, None, None)

    assert calls == ["CPU"]


def test_openai_busy_returns_503_device_busy(real_client: TestClient) -> None:
    from npu_proxy.inference.engine import acquire_device_slot

    with acquire_device_slot("NPU", timeout=1.0):
        response = real_client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "device_busy"


def test_streaming_busy_returns_503_before_stream(real_client: TestClient) -> None:
    from npu_proxy.inference.engine import acquire_device_slot

    with acquire_device_slot("NPU", timeout=1.0):
        response = real_client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "device_busy"


def test_ollama_busy_returns_503_device_busy(real_client: TestClient) -> None:
    from npu_proxy.inference.engine import acquire_device_slot

    with acquire_device_slot("NPU", timeout=1.0):
        response = real_client.post(
            "/api/generate",
            json={"model": "tinyllama", "prompt": "Hi", "stream": False},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "device_busy"


def test_fallback_on_busy_header_reports_actual_device(
    real_client: TestClient,
    monkeypatch,
) -> None:
    import npu_proxy.inference.engine as engine_module
    from npu_proxy.inference.engine import acquire_device_slot

    monkeypatch.setenv("NPU_PROXY_FALLBACK_ON_BUSY", "1")
    calls = _patch_fake_engines(monkeypatch)
    monkeypatch.setattr(engine_module, "get_available_devices", lambda: ["NPU", "CPU"])

    with acquire_device_slot("NPU", timeout=1.0):
        response = real_client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert calls == ["CPU"]
    assert response.headers["x-npu-proxy-device"] == "CPU"


def test_slot_released_after_successful_request(
    real_client: TestClient,
    monkeypatch,
) -> None:
    from npu_proxy.inference.engine import acquire_device_slot

    _patch_fake_engines(monkeypatch)

    response = real_client.post(
        "/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    with acquire_device_slot("NPU", timeout=0.0):
        pass


def test_slot_released_after_erroring_request(
    real_client: TestClient,
    monkeypatch,
) -> None:
    from npu_proxy.inference.engine import acquire_device_slot

    _patch_fake_engines(monkeypatch, fail=True)

    response = real_client.post(
        "/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 503
    with acquire_device_slot("NPU", timeout=0.0):
        pass


def test_streaming_request_releases_slot_after_completion(
    real_client: TestClient,
    monkeypatch,
) -> None:
    from npu_proxy.inference.engine import acquire_device_slot

    _patch_fake_engines(monkeypatch)

    response = real_client.post(
        "/v1/chat/completions",
        json={
            "model": "tinyllama-1.1b-chat-int4-ov",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    with acquire_device_slot("NPU", timeout=0.0):
        pass
