"""Behavioral end-to-end tests against a live NPU Proxy server process."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


pytestmark = [pytest.mark.slow, pytest.mark.e2e]


def _find_free_port() -> int:
    """Reserve an available loopback TCP port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_log(log_path: Path) -> str:
    """Read captured server output for debugging failures."""
    if not log_path.exists():
        return "<no server log captured>"
    return log_path.read_text(encoding="utf-8", errors="replace")


def _tail_log(log_path: Path, max_lines: int = 80) -> str:
    """Return the last lines of the server log for compact failure output."""
    content = _read_log(log_path).splitlines()
    if len(content) <= max_lines:
        return "\n".join(content)
    return "\n".join(content[-max_lines:])


def _wait_for_server(base_url: str, process: subprocess.Popen, log_path: Path, timeout: float = 60.0) -> None:
    """Wait until the live server answers health checks or fail with logs."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                "Live server exited before becoming ready.\n"
                f"Exit code: {process.returncode}\n"
                f"Server log:\n{_tail_log(log_path)}"
            )

        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass

        time.sleep(0.25)

    pytest.fail(
        "Timed out waiting for the live server to become ready.\n"
        f"Server log:\n{_tail_log(log_path)}"
    )


def _start_server(repo_root: Path, log_path: Path) -> tuple[subprocess.Popen, str]:
    """Start the server subprocess, retrying if a chosen port is taken."""
    env = os.environ.copy()
    env.pop("NPU_PROXY_REAL_INFERENCE", None)
    env["PYTHONUNBUFFERED"] = "1"

    last_error = "server did not start"

    for _ in range(5):
        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "npu_proxy.cli",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "warning",
                ],
                cwd=repo_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        try:
            _wait_for_server(base_url, process, log_path)
            return process, base_url
        except BaseException:
            last_error = _tail_log(log_path)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    pytest.fail(
        "Failed to start live server after multiple attempts.\n"
        f"Last server log:\n{last_error}"
    )


@pytest.fixture(scope="session")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Start a real NPU Proxy subprocess in mock mode and return its base URL."""
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = tmp_path_factory.mktemp("e2e-server")
    log_path = log_dir / "server.log"
    process, base_url = _start_server(repo_root, log_path)

    try:
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_live_server_reports_health_and_versions(live_server: str) -> None:
    """The deployed server should expose health and version endpoints."""
    with httpx.Client(base_url=live_server, timeout=5.0) as client:
        health = client.get("/health")
        liveness = client.get("/health/liveness")
        readiness = client.get("/health/readiness")
        devices = client.get("/health/devices")
        version = client.get("/api/version")

    assert health.status_code == 200
    health_data = health.json()
    assert health_data["status"] in {"healthy", "degraded"}
    assert isinstance(health_data["version"], str)
    assert health_data["version"]
    assert isinstance(health_data["devices"], list)
    assert isinstance(health_data["openvino_version"], str)
    assert health_data["openvino_version"]
    assert isinstance(health_data["engines"]["llm"]["backend"], str)
    assert "compile_cache_mode" in health_data["engines"]["llm"]
    assert "runtime_features" in health_data["engines"]["llm"]
    assert "message" in health_data["engines"]["llm"]
    assert "message" in health_data["engines"]["embedding"]

    assert liveness.status_code == 200
    liveness_data = liveness.json()
    assert liveness_data["status"] == "alive"
    assert liveness_data["alive"] is True

    assert readiness.status_code in {200, 503}
    readiness_data = readiness.json()
    assert isinstance(readiness_data["ready"], bool)
    assert isinstance(readiness_data["reasons"], list)

    assert devices.status_code == 200
    devices_data = devices.json()
    assert "active_backend" in devices_data
    assert "llm" in devices_data
    assert "embedding" in devices_data
    assert "runtime_state" in devices_data["llm"]
    assert "runtime_state" in devices_data["embedding"]

    assert version.status_code == 200
    version_data = version.json()
    assert version_data["version"].endswith("-npu-proxy")


def test_live_server_serves_model_catalogs(live_server: str) -> None:
    """The deployed server should expose OpenAI and Ollama-compatible model discovery."""
    with httpx.Client(base_url=live_server, timeout=5.0) as client:
        openai_models = client.get("/v1/models")
        running_models = client.get("/api/ps")

    assert openai_models.status_code == 200
    models_data = openai_models.json()
    assert models_data["object"] == "list"
    assert isinstance(models_data["data"], list)
    assert len(models_data["data"]) > 0
    assert {"id", "object", "created", "owned_by"} <= set(models_data["data"][0])

    assert running_models.status_code == 200
    ps_data = running_models.json()
    assert "models" in ps_data
    assert isinstance(ps_data["models"], list)


def test_live_server_handles_ollama_chat_flow(live_server: str) -> None:
    """The deployed server should answer Ollama-compatible generate and chat requests."""
    with httpx.Client(base_url=live_server, timeout=5.0) as client:
        generate = client.post(
            "/api/generate",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "prompt": "Say hello from the Intel NPU proxy",
                "stream": False,
            },
        )
        chat = client.post(
            "/api/chat",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello there"}],
                "stream": False,
            },
        )

    assert generate.status_code == 200
    generate_data = generate.json()
    assert generate_data["done"] is True
    assert generate_data["model"] == "tinyllama-1.1b-chat-int4-ov"
    assert "Intel NPU" in generate_data["response"]

    assert chat.status_code == 200
    chat_data = chat.json()
    assert chat_data["done"] is True
    assert chat_data["model"] == "tinyllama-1.1b-chat-int4-ov"
    assert chat_data["message"]["role"] == "assistant"
    assert "Intel NPU" in chat_data["message"]["content"]


def test_live_server_enforces_request_validation(live_server: str) -> None:
    """The deployed server should reject invalid requests with FastAPI validation."""
    with httpx.Client(base_url=live_server, timeout=5.0) as client:
        invalid_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [],
            },
        )

    assert invalid_chat.status_code == 422
