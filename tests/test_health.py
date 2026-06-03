"""Tests for health endpoint - TDD Phase 1.1"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import AsyncClient, ASGITransport
from npu_proxy import __version__
from npu_proxy.api import health as health_api
from npu_proxy.main import app


@pytest.mark.asyncio
async def test_health_returns_200():
    """GET /health should return 200 OK"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_status():
    """GET /health should return status field"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] in ("ok", "healthy", "degraded")


@pytest.mark.asyncio
async def test_health_includes_npu_status():
    """GET /health should include NPU availability"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    data = response.json()
    assert "npu_available" in data
    assert isinstance(data["npu_available"], bool)


@pytest.mark.asyncio
async def test_health_reports_package_version():
    """GET /health should report the shared package version."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    data = response.json()
    assert data["version"] == __version__


@pytest.mark.asyncio
async def test_health_degrades_when_openvino_version_probe_fails(monkeypatch):
    """GET /health should stay observational when OpenVINO import/version probing fails."""
    monkeypatch.setattr(health_api, "_get_openvino_module", lambda: (_ for _ in ()).throw(RuntimeError("no openvino")))
    monkeypatch.setattr(health_api, "get_ov_core", lambda: (_ for _ in ()).throw(RuntimeError("no devices")))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["openvino_version"] is None
    assert data["device_probe_error"] == "device_enumeration_failed"


@pytest.mark.asyncio
async def test_health_reports_loaded_llm_device(monkeypatch):
    """GET /health should report the actual device for a loaded LLM engine."""

    class FakeCore:
        available_devices = ["CPU", "NPU"]

    class FakeLlmEngine:
        model_name = "tinyllama-1.1b-chat-int4-ov"

        def get_device_info(self):
            return {
                "actual_device": "NPU",
                "requested_device": "NPU",
                "fallback_device": "GPU",
                "used_fallback": False,
                "backend": "openvino",
                "compile_cache_dir": "build\\runtime-cache",
                "compile_cache_mode": "OPTIMIZE_SPEED",
                "prefix_cache_mode": "on",
                "runtime_features": {
                    "compile_cache_enabled": True,
                    "prefix_cache_enabled": True,
                    "degraded_features": [],
                },
                "model_load_seconds": 1.25,
                "load_diagnostics": [{"device": "NPU", "status": "loaded"}],
                "last_generation_stats": {
                    "ttft_seconds": 0.4,
                    "throughput_tokens_per_second": 25.0,
                },
            }

    class FakeEmbeddingEngine:
        def get_engine_info(self):
            return {
                "model_name": "BAAI/bge-small-en-v1.5",
                "resolved_model": "BAAI/bge-small-en-v1.5",
                "requested_model": "BAAI/bge-small-en-v1.5",
                "device": "CPU",
                "requested_device": "NPU",
                "dimensions": 384,
                "is_production": False,
                "is_fallback": True,
                "fallback_reason": "Embedding model missing on requested device",
                "fallback_mode": "missing_model",
                "model_path": "C:\\models\\embeddings\\BAAI_bge-small-en-v1.5",
            }

    class FakeRuntimeConfig:
        backend = SimpleNamespace(value="openvino")
        device = "NPU"
        compile_cache_dir = Path("build\\runtime-cache")
        compile_cache_mode = "OPTIMIZE_SPEED"
        prefix_cache_mode = "on"
        is_alpha_backend = False
        llama_cpp_model_path = None
        model_path = Path("C:\\models\\tinyllama-1.1b-chat-int4-ov")

        def backend_model_path(self):
            return self.model_path

    monkeypatch.setattr(health_api, "get_ov_core", lambda: FakeCore())
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: True)
    monkeypatch.setattr(
        health_api,
        "get_loaded_models",
        lambda: {"tinyllama-1.1b-chat-int4-ov": FakeLlmEngine()},
    )
    monkeypatch.setattr(health_api, "get_loaded_embedding_engine", lambda model_name=None, device=None: FakeEmbeddingEngine())
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: FakeRuntimeConfig())
    monkeypatch.setattr(
        health_api.embedding_config,
        "resolve_embedding_model_config",
        lambda: SimpleNamespace(
            requested_model="BAAI/bge-small-en-v1.5",
            requested_device="NPU",
            resolved_model="BAAI/bge-small-en-v1.5",
            dimensions=384,
            model_path=Path("C:\\models\\embeddings\\BAAI_bge-small-en-v1.5"),
            canonical_path=Path("C:\\models\\embeddings\\BAAI_bge-small-en-v1.5"),
            repo_id="BAAI/bge-small-en-v1.5",
            is_downloaded=True,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert data["engines"]["llm"]["status"] == "loaded"
    assert data["engines"]["llm"]["device"] == "NPU"
    assert data["engines"]["llm"]["model"] == "tinyllama-1.1b-chat-int4-ov"
    assert data["engines"]["llm"]["backend"] == "openvino"
    assert data["engines"]["llm"]["compile_cache_dir"] == "build\\runtime-cache"
    assert data["engines"]["llm"]["runtime_features"]["compile_cache_enabled"] is True
    assert data["engines"]["llm"]["last_generation_stats"]["ttft_seconds"] == 0.4
    assert data["engines"]["embedding"]["status"] == "loaded"
    assert data["engines"]["embedding"]["backend"] == "hash"
    assert data["engines"]["embedding"]["requested_device"] == "NPU"
    assert data["engines"]["embedding"]["cache"]["kind"] == "in_memory"


@pytest.mark.asyncio
async def test_health_devices_reports_engine_runtime_state(monkeypatch):
    """GET /health/devices should expose additive LLM and embedding runtime state."""

    monkeypatch.setattr(
        "npu_proxy.inference.engine.get_available_devices",
        lambda: ["CPU", "GPU", "NPU"],
    )

    class FakeLlmEngine:
        model_name = "tinyllama-1.1b-chat-int4-ov"

        def get_device_info(self):
            return {
                "actual_device": "NPU",
                "requested_device": "NPU",
                "fallback_device": "GPU",
                "used_fallback": False,
                "backend": "openvino",
                "runtime_features": {
                    "compile_cache_enabled": True,
                    "prefix_cache_enabled": True,
                    "degraded_features": [],
                },
                "compile_cache_dir": "build\\runtime-cache",
                "compile_cache_mode": "OPTIMIZE_SPEED",
                "prefix_cache_mode": "on",
                "last_generation_stats": {"throughput_tokens_per_second": 25.0},
            }

    class FakeEmbeddingEngine:
        def get_engine_info(self):
            return {
                "model_name": "BAAI/bge-small-en-v1.5",
                "resolved_model": "BAAI/bge-small-en-v1.5",
                "requested_model": "BAAI/bge-small-en-v1.5",
                "device": "CPU",
                "requested_device": "CPU",
                "dimensions": 384,
                "is_production": True,
                "is_fallback": False,
            }

    monkeypatch.setattr(health_api, "is_model_loaded", lambda: True)
    monkeypatch.setattr(
        health_api,
        "get_loaded_models",
        lambda: {"tinyllama-1.1b-chat-int4-ov": FakeLlmEngine()},
    )
    monkeypatch.setattr(
        health_api,
        "get_loaded_embedding_engine",
        lambda model_name=None, device=None: FakeEmbeddingEngine(),
    )
    monkeypatch.setattr(
        health_api.embedding_config,
        "resolve_embedding_model_config",
        lambda: SimpleNamespace(
            requested_model="BAAI/bge-small-en-v1.5",
            requested_device="CPU",
            resolved_model="BAAI/bge-small-en-v1.5",
            dimensions=384,
            model_path=Path("C:\\models\\embeddings\\BAAI_bge-small-en-v1.5"),
            canonical_path=Path("C:\\models\\embeddings\\BAAI_bge-small-en-v1.5"),
            repo_id="BAAI/bge-small-en-v1.5",
            is_downloaded=True,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/devices")

    assert response.status_code == 200
    data = response.json()
    assert data["active_device"] == "NPU"
    assert data["active_backend"] == "openvino"
    assert data["device_info"]["actual_device"] == "NPU"
    assert data["llm"]["runtime_state"]["compile_cache_dir"] == "build\\runtime-cache"
    assert data["llm"]["runtime_state"]["runtime_features"]["prefix_cache_enabled"] is True
    assert data["embedding"]["active_device"] == "CPU"
    assert data["embedding"]["backend"] == "openvino"
    assert data["embedding"]["runtime_state"]["cache"]["configured_max_entries"] == 1024


@pytest.mark.asyncio
async def test_health_liveness_returns_alive():
    """GET /health/liveness should return a cheap process-up signal."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/liveness")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert data["alive"] is True


@pytest.mark.asyncio
async def test_health_readiness_reports_not_loaded_models(monkeypatch):
    """GET /health/readiness should fail closed when models are not loaded."""

    class FakeCore:
        available_devices = ["CPU", "NPU"]

    class FakeRuntimeConfig:
        backend = SimpleNamespace(value="openvino")
        device = "NPU"
        compile_cache_dir = None
        compile_cache_mode = None
        prefix_cache_mode = "auto"
        is_alpha_backend = False
        llama_cpp_model_path = None
        model_path = Path("C:\\models\\tinyllama-1.1b-chat-int4-ov")

        def backend_model_path(self):
            return self.model_path

    monkeypatch.setattr(health_api, "get_ov_core", lambda: FakeCore())
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: False)
    monkeypatch.setattr(
        health_api,
        "get_loaded_embedding_engine",
        lambda model_name=None, device=None: None,
    )
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: FakeRuntimeConfig())
    monkeypatch.setattr(
        health_api.embedding_config,
        "resolve_embedding_model_config",
        lambda: SimpleNamespace(
            requested_model="all-minilm",
            requested_device="CPU",
            resolved_model="all-minilm-l6-v2",
            dimensions=384,
            model_path=Path("C:\\models\\embeddings\\all-minilm-l6-v2"),
            canonical_path=Path("C:\\models\\embeddings\\all-minilm-l6-v2"),
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            is_downloaded=True,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/readiness")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert any("LLM model is not loaded" in reason for reason in data["reasons"])
    assert any("Embedding model is not loaded" in reason for reason in data["reasons"])


class _FakeBackend:
    value = "openvino"


class _FakeRuntimeConfig:
    backend = _FakeBackend()
    device = "NPU"
    compile_cache_dir = Path("build\\runtime-cache")
    compile_cache_mode = "OPTIMIZE_SPEED"
    prefix_cache_mode = "auto"
    is_alpha_backend = False
    llama_cpp_model_path = None
    model_path = Path("C:\\models\\tinyllama")

    def __init__(self, *, backend_error=False, llama_cpp_model_path=None, model_path=None):
        self._backend_error = backend_error
        self.llama_cpp_model_path = llama_cpp_model_path
        self.model_path = model_path or self.model_path

    def backend_model_path(self):
        if self._backend_error:
            raise RuntimeError("backend-specific path failed")
        return self.model_path


def _embedding_config(*, downloaded=True):
    return SimpleNamespace(
        requested_model="all-minilm",
        requested_device="CPU",
        resolved_model="all-minilm-l6-v2",
        dimensions=384,
        model_path=Path("C:\\models\\embeddings\\all-minilm-l6-v2"),
        canonical_path=Path("C:\\models\\embeddings\\all-minilm-l6-v2"),
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        is_downloaded=downloaded,
    )


def test_safe_backend_model_path_returns_none_when_runtime_config_fails(monkeypatch):
    monkeypatch.setattr(
        health_api,
        "get_active_llm_runtime_config",
        lambda: (_ for _ in ()).throw(RuntimeError("bad config")),
    )

    assert health_api._safe_backend_model_path() is None


def test_safe_backend_model_path_falls_back_to_llama_cpp_path(monkeypatch):
    config = _FakeRuntimeConfig(
        backend_error=True,
        llama_cpp_model_path=Path("C:\\models\\model.gguf"),
    )
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: config)

    assert health_api._safe_backend_model_path() == "C:\\models\\model.gguf"


def test_safe_backend_model_path_falls_back_to_model_path(monkeypatch):
    config = _FakeRuntimeConfig(backend_error=True, model_path=Path("C:\\models\\openvino"))
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: config)

    assert health_api._safe_backend_model_path() == "C:\\models\\openvino"


def test_path_exists_handles_unknown_existing_missing_and_os_errors(monkeypatch):
    assert health_api._path_exists(None) is None
    assert health_api._path_exists(str(Path(__file__))) is True
    assert health_api._path_exists("Z:\\npu-proxy-definitely-missing\\model") is False

    monkeypatch.setattr(health_api.Path, "exists", lambda self: (_ for _ in ()).throw(OSError("bad path")))

    assert health_api._path_exists("C:\\bad\\path") is None


def test_get_llm_runtime_config_returns_error_when_config_unavailable(monkeypatch):
    monkeypatch.setattr(
        health_api,
        "get_active_llm_runtime_config",
        lambda: (_ for _ in ()).throw(RuntimeError("bad config")),
    )

    assert health_api._get_llm_runtime_config() == {"error": "runtime_config_unavailable"}


def test_gpu_and_npu_probe_exceptions_return_stable_error(monkeypatch):
    monkeypatch.setattr(health_api, "get_ov_core", lambda: (_ for _ in ()).throw(RuntimeError("no core")))

    assert health_api.check_npu_available() == (False, "device_probe_failed")
    assert health_api.check_gpu_available() == (False, "device_probe_failed")


def test_llm_state_message_distinguishes_missing_and_existing_model_paths(monkeypatch):
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: False)
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: _FakeRuntimeConfig(model_path=Path(__file__)))
    existing_state, _ = health_api._get_llm_engine_state()

    monkeypatch.setattr(
        health_api,
        "get_active_llm_runtime_config",
        lambda: _FakeRuntimeConfig(model_path=Path("Z:\\npu-proxy-definitely-missing\\model")),
    )
    missing_state, _ = health_api._get_llm_engine_state()

    assert "does not auto-load" in str(existing_state.message)
    assert "path is missing" in str(missing_state.message)


def test_llm_state_uses_first_loaded_model_when_multiple_are_present(monkeypatch):
    class FakeEngine:
        def __init__(self, model_name, actual_device):
            self.model_name = model_name
            self.actual_device = actual_device

        def get_device_info(self):
            return {"actual_device": self.actual_device, "backend": "openvino"}

    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: _FakeRuntimeConfig())
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: True)
    monkeypatch.setattr(
        health_api,
        "get_loaded_models",
        lambda: {
            "first": FakeEngine("first-model", "GPU"),
            "second": FakeEngine("second-model", "NPU"),
        },
    )

    state, device_info = health_api._get_llm_engine_state()

    assert state.model == "first-model"
    assert state.device == "GPU"
    assert device_info == {"actual_device": "GPU", "backend": "openvino"}


@pytest.mark.asyncio
async def test_health_endpoint_reports_runtime_config_errors(monkeypatch, async_client):
    monkeypatch.setattr(health_api, "get_ov_core", lambda: SimpleNamespace(available_devices=["CPU"]))
    monkeypatch.setattr(
        health_api,
        "get_active_llm_runtime_config",
        lambda: (_ for _ in ()).throw(RuntimeError("bad config")),
    )
    monkeypatch.setattr(health_api.embedding_config, "resolve_embedding_model_config", lambda: _embedding_config())
    monkeypatch.setattr(health_api, "get_loaded_embedding_engine", lambda model_name=None, device=None: None)

    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["engines"]["llm"]["error"] == "runtime_config_unavailable"


@pytest.mark.asyncio
async def test_health_endpoint_reports_unavailable_openvino_core(monkeypatch, async_client):
    monkeypatch.setattr(health_api, "get_ov_core", lambda: (_ for _ in ()).throw(RuntimeError("no core")))
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: _FakeRuntimeConfig())
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: False)
    monkeypatch.setattr(health_api.embedding_config, "resolve_embedding_model_config", lambda: _embedding_config())
    monkeypatch.setattr(health_api, "get_loaded_embedding_engine", lambda model_name=None, device=None: None)

    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["devices"] == []
    assert data["device_probe_error"] == "device_enumeration_failed"


@pytest.mark.asyncio
async def test_health_endpoint_reports_loaded_llm_without_embedding(monkeypatch, async_client):
    class FakeLlmEngine:
        model_name = "tinyllama"

        def get_device_info(self):
            return {"actual_device": "NPU", "requested_device": "NPU", "backend": "openvino"}

    monkeypatch.setattr(health_api, "get_ov_core", lambda: SimpleNamespace(available_devices=["CPU", "NPU"]))
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: _FakeRuntimeConfig())
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: True)
    monkeypatch.setattr(health_api, "get_loaded_models", lambda: {"tinyllama": FakeLlmEngine()})
    monkeypatch.setattr(health_api.embedding_config, "resolve_embedding_model_config", lambda: _embedding_config(downloaded=True))
    monkeypatch.setattr(health_api, "get_loaded_embedding_engine", lambda model_name=None, device=None: None)

    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["engines"]["llm"]["status"] == "loaded"
    assert data["engines"]["embedding"]["status"] == "not_loaded"
    assert "does not auto-load embedding engines" in data["engines"]["embedding"]["message"]


@pytest.mark.asyncio
async def test_health_endpoint_reports_embedding_config_resolution_failure(monkeypatch, async_client):
    monkeypatch.setattr(health_api, "get_ov_core", lambda: SimpleNamespace(available_devices=["CPU"]))
    monkeypatch.setattr(health_api, "get_active_llm_runtime_config", lambda: _FakeRuntimeConfig())
    monkeypatch.setattr(health_api, "is_model_loaded", lambda: False)
    monkeypatch.setattr(
        health_api.embedding_config,
        "resolve_embedding_model_config",
        lambda: (_ for _ in ()).throw(RuntimeError("bad embedding config")),
    )

    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["engines"]["embedding"]["error"] == "embedding_config_unavailable"
