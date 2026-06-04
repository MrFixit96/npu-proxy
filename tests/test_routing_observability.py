from __future__ import annotations

import threading

from fastapi.testclient import TestClient


class FakePoolEngine:
    model_name = "tinyllama"
    actual_device = "NPU"
    requested_device = "NPU"
    is_warmed_up = True

    def get_device_info(self) -> dict[str, object]:
        return {
            "actual_device": "NPU",
            "requested_device": "NPU",
            "is_warmed_up": True,
        }


def test_health_includes_empty_device_pool(client: TestClient, monkeypatch) -> None:
    import npu_proxy.api.health as health_module

    monkeypatch.setattr(health_module, "_get_available_devices", lambda: (["CPU", "NPU"], None))
    monkeypatch.setattr(health_module, "_get_openvino_version", lambda: "test-openvino")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["runtime"]["device_pool"] == []


def test_health_device_pool_reports_loaded_warmed_and_busy(client: TestClient, monkeypatch) -> None:
    import npu_proxy.api.health as health_module
    import npu_proxy.inference.engine as engine_module

    pool_key = ("C:\\models\\tinyllama", "NPU")
    lock = threading.Lock()
    lock.acquire()
    with engine_module._engine_lock:
        engine_module._engine_pool[pool_key] = FakePoolEngine()
        engine_module._device_locks[pool_key] = lock

    monkeypatch.setattr(health_module, "_get_available_devices", lambda: (["CPU", "NPU"], None))
    monkeypatch.setattr(health_module, "_get_openvino_version", lambda: "test-openvino")
    try:
        response = client.get("/health")
    finally:
        lock.release()

    assert response.status_code == 200
    pool = response.json()["runtime"]["device_pool"]
    assert pool == [
        {
            "model": "tinyllama",
            "model_path": "C:\\models\\tinyllama",
            "device": "NPU",
            "requested_device": "NPU",
            "loaded": True,
            "warmed": True,
            "busy": True,
        }
    ]
