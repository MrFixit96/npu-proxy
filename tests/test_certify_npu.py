from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest

from scripts import certify_npu


def _make_process(*, running: bool = True, returncode: int = 0):
    process = Mock()
    process.returncode = returncode
    process.poll.return_value = None if running else returncode
    process.wait.return_value = returncode
    return process


class _FakeClient:
    def __init__(self, status_code: int = 200):
        self._status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, path: str):
        assert path == "/health"
        return SimpleNamespace(status_code=self._status_code)


def test_wait_for_server_retries_until_health_check_succeeds(monkeypatch, tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text("booting\n", encoding="utf-8")
    process = _make_process()
    response_503 = SimpleNamespace(status_code=503)
    response_200 = SimpleNamespace(status_code=200)
    http_get = Mock(
        side_effect=[
            httpx.ConnectError("not ready"),
            response_503,
            response_200,
        ]
    )
    perf_counter = Mock(side_effect=[10.0, 10.0, 10.2, 10.4, 10.9])
    sleep = Mock()

    monkeypatch.setattr(certify_npu.httpx, "get", http_get)
    monkeypatch.setattr(certify_npu.time, "perf_counter", perf_counter)
    monkeypatch.setattr(certify_npu.time, "sleep", sleep)

    startup_seconds = certify_npu._wait_for_server(
        "http://127.0.0.1:9000",
        process,
        log_path,
        timeout=5.0,
    )

    assert startup_seconds == pytest.approx(0.9)
    assert http_get.call_count == 3
    sleep.assert_has_calls([call(0.25), call(0.25)])


def test_wait_for_server_raises_timeout_with_log_tail(monkeypatch, tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text("line1\nline2\n", encoding="utf-8")
    process = _make_process()
    perf_counter = Mock(side_effect=[0.0, 0.0, 0.3, 0.4, 0.7])
    sleep = Mock()

    monkeypatch.setattr(
        certify_npu.httpx,
        "get",
        Mock(side_effect=httpx.ConnectError("still starting")),
    )
    monkeypatch.setattr(certify_npu.time, "perf_counter", perf_counter)
    monkeypatch.setattr(certify_npu.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="Timed out waiting for certification server readiness") as exc_info:
        certify_npu._wait_for_server("http://127.0.0.1:9000", process, log_path, timeout=0.5)

    assert "line1\nline2" in str(exc_info.value)
    sleep.assert_has_calls([call(0.25), call(0.25)])


def test_start_server_retries_and_terminates_previous_process(monkeypatch, tmp_path: Path):
    repo_root = tmp_path
    log_path = tmp_path / "server.log"
    first_process = _make_process()
    second_process = _make_process()
    popen = Mock(side_effect=[first_process, second_process])
    wait_for_server = Mock(side_effect=[RuntimeError("first boot failed"), 0.75])
    ports = iter([9101, 9102])

    monkeypatch.setattr(certify_npu.subprocess, "Popen", popen)
    monkeypatch.setattr(certify_npu, "_wait_for_server", wait_for_server)
    monkeypatch.setattr(certify_npu, "_find_free_port", lambda: next(ports))

    process, base_url, startup_seconds = certify_npu._start_server(
        repo_root=repo_root,
        log_path=log_path,
        port=None,
        startup_timeout=3.0,
        requested_device="NPU",
    )

    assert process is second_process
    assert base_url == "http://127.0.0.1:9102"
    assert startup_seconds == 0.75
    first_process.terminate.assert_called_once_with()
    first_process.wait.assert_called_once_with(timeout=10)
    first_process.kill.assert_not_called()
    assert popen.call_count == 2


def test_start_server_kills_process_when_terminate_times_out(monkeypatch, tmp_path: Path):
    process = _make_process()
    process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), 0]

    monkeypatch.setattr(certify_npu.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(certify_npu, "_wait_for_server", Mock(side_effect=RuntimeError("boot failed")))

    with pytest.raises(RuntimeError, match="boot failed"):
        certify_npu._start_server(
            repo_root=tmp_path,
            log_path=tmp_path / "server.log",
            port=9200,
            startup_timeout=3.0,
            requested_device="NPU",
        )

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [call(timeout=10), call(timeout=10)]


def test_main_terminates_server_process_on_success(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "tiny").mkdir()
    log_path = tmp_path / "certify.log"
    process = _make_process()
    args = argparse.Namespace(
        port=None,
        startup_timeout=3.0,
        request_timeout=5.0,
        model="tiny",
        device="NPU",
        prompt="hello",
        output=None,
    )
    report = SimpleNamespace(certified=True)

    monkeypatch.setattr(certify_npu, "parse_args", lambda: args)
    monkeypatch.setattr(certify_npu, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr(
        certify_npu,
        "collect_openvino_runtime_details",
        lambda: {"npu_visible": True},
    )
    monkeypatch.setattr(certify_npu, "_get_certification_log_path", lambda repo_root, requested_device: log_path)
    monkeypatch.setattr(certify_npu, "_start_server", lambda **kwargs: (process, "http://127.0.0.1:9000", 0.5))
    monkeypatch.setattr(certify_npu.httpx, "Client", lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr(certify_npu, "_run_llm_workload", lambda client, args, log_path: ({"response": "ok"}, 1.2))
    monkeypatch.setattr(
        certify_npu,
        "_run_routing_checks",
        lambda client, args, log_path: {"short_headers": {}, "long_headers": {}, "fallback_device": "CPU"},
    )
    monkeypatch.setattr(
        certify_npu,
        "_collect_observability_snapshots",
        lambda client, log_path: ({"status": "healthy"}, {"active_device": "NPU"}),
    )
    monkeypatch.setattr(certify_npu, "evaluate_hardware_certification", lambda **kwargs: report)
    monkeypatch.setattr(certify_npu, "format_certification_report", lambda report: "certified")

    exit_code = certify_npu.main()

    assert exit_code == 0
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)
    process.kill.assert_not_called()


def test_main_kills_server_process_in_finally_after_failure(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "tiny").mkdir()
    log_path = tmp_path / "certify.log"
    process = _make_process()
    process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), 0]
    args = argparse.Namespace(
        port=None,
        startup_timeout=3.0,
        request_timeout=5.0,
        model="tiny",
        device="NPU",
        prompt="hello",
        output=None,
    )

    monkeypatch.setattr(certify_npu, "parse_args", lambda: args)
    monkeypatch.setattr(certify_npu, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr(
        certify_npu,
        "collect_openvino_runtime_details",
        lambda: {"npu_visible": True},
    )
    monkeypatch.setattr(certify_npu, "_get_certification_log_path", lambda repo_root, requested_device: log_path)
    monkeypatch.setattr(certify_npu, "_start_server", lambda **kwargs: (process, "http://127.0.0.1:9000", 0.5))
    monkeypatch.setattr(certify_npu.httpx, "Client", lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr(
        certify_npu,
        "_run_llm_workload",
        lambda client, args, log_path: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )

    exit_code = certify_npu.main()

    assert exit_code == 1
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [call(timeout=10), call(timeout=10)]
