"""Run a live hardware certification of NPU Proxy against a real Intel NPU."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from npu_proxy.hardware_certification import (
    collect_openvino_runtime_details,
    evaluate_hardware_certification,
    format_certification_report,
)
try:
    from npu_proxy.inference.engine import DEFAULT_LLM_MODEL, DEFAULT_MODEL_DIR
except Exception:
    DEFAULT_LLM_MODEL = "tinyllama"
    DEFAULT_MODEL_DIR = Path.home() / ".cache" / "npu-proxy" / "models"

_DEVICE_CHOICES = ["NPU", "GPU", "CPU", "AUTO"]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_log(log_path: Path) -> str:
    if not log_path.exists():
        return "<no server log captured>"
    return log_path.read_text(encoding="utf-8", errors="replace")


def _tail_log(log_path: Path, max_lines: int = 80) -> str:
    lines = _read_log(log_path).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _get_certification_log_path(repo_root: Path, requested_device: str) -> Path:
    log_dir = repo_root / "build" / "certification"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_device = "".join(
        char.lower() if char.isalnum() else "-"
        for char in requested_device
    ).strip("-") or "device"
    return log_dir / f"certify-{safe_device}.log"


def _wait_for_server(base_url: str, process: subprocess.Popen, log_path: Path, timeout: float) -> float:
    started = time.perf_counter()
    deadline = started + timeout

    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Certification server exited before readiness.\n"
                f"Exit code: {process.returncode}\n"
                f"Server log:\n{_tail_log(log_path)}"
            )

        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass

        time.sleep(0.25)

    raise RuntimeError(
        "Timed out waiting for certification server readiness.\n"
        f"Server log:\n{_tail_log(log_path)}"
    )


def _start_server(
    repo_root: Path,
    log_path: Path,
    port: int | None,
    startup_timeout: float,
    requested_device: str,
) -> tuple[subprocess.Popen, str, float]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    attempts = 1 if port is not None else 5
    last_error: Exception | None = None

    for _ in range(attempts):
        chosen_port = port or _find_free_port()
        base_url = f"http://127.0.0.1:{chosen_port}"

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "npu_proxy.cli",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(chosen_port),
                    "--device",
                    requested_device,
                    "--real-inference",
                    "--log-level",
                    "warning",
                ],
                cwd=repo_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        try:
            startup_seconds = _wait_for_server(base_url, process, log_path, startup_timeout)
            return process, base_url, startup_seconds
        except Exception as exc:
            last_error = exc
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    raise RuntimeError(str(last_error) if last_error else "Failed to start certification server.")


def _collect_observability_snapshots(
    client: httpx.Client,
    log_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    health_response = client.get("/health")
    devices_response = client.get("/health/devices")

    if health_response.status_code != 200:
        raise RuntimeError(
            f"Unexpected /health response after certification: {health_response.status_code}\n"
            f"Body: {health_response.text}\n"
            f"Server log:\n{_tail_log(log_path)}"
        )
    if devices_response.status_code != 200:
        raise RuntimeError(
            f"Unexpected /health/devices response after certification: {devices_response.status_code}\n"
            f"Body: {devices_response.text}\n"
            f"Server log:\n{_tail_log(log_path)}"
        )

    return health_response.json(), devices_response.json()


def _run_llm_workload(
    client: httpx.Client,
    args: argparse.Namespace,
    log_path: Path,
) -> tuple[dict[str, object], float]:
    start = time.perf_counter()
    generate_response = client.post(
        "/api/generate",
        json={
            "model": args.model,
            "prompt": args.prompt,
            "stream": False,
        },
    )
    inference_seconds = time.perf_counter() - start

    if generate_response.status_code != 200:
        raise RuntimeError(
            f"Generate request failed with {generate_response.status_code}.\n"
            f"Body: {generate_response.text}\n"
            f"Server log:\n{_tail_log(log_path)}"
        )

    return generate_response.json(), inference_seconds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Certify that NPU Proxy is performing real inference on the local Intel NPU.",
    )
    parser.add_argument("--port", type=int, help="Fixed port for the temporary certification server.")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the server to start.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=240.0,
        help="Seconds to allow the real inference request to complete.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help="Model name to certify. Defaults to the repo's default LLM model.",
    )
    parser.add_argument(
        "--device",
        type=str.upper,
        choices=_DEVICE_CHOICES,
        default="NPU",
        help="Requested LLM device for the temporary certification server (default: NPU).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with a short sentence confirming NPU certification.",
        help="Prompt used for the certification generation request.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON certification report.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_path = DEFAULT_MODEL_DIR / args.model
    runtime_details = collect_openvino_runtime_details()
    log_path = _get_certification_log_path(repo_root, args.device)

    if not runtime_details["npu_visible"]:
        print("Certification failed: OpenVINO does not see an NPU on this host.", file=sys.stderr)
        return 1

    if not model_path.exists():
        print(
            f"Certification failed: model not found at {model_path}. "
            "Download the default TinyLlama model before certifying.",
            file=sys.stderr,
        )
        return 1

    process = None
    try:
        process, base_url, startup_seconds = _start_server(
            repo_root=repo_root,
            log_path=log_path,
            port=args.port,
            startup_timeout=args.startup_timeout,
            requested_device=args.device,
        )

        with httpx.Client(base_url=base_url, timeout=args.request_timeout) as client:
            pre_health = client.get("/health")
            if pre_health.status_code != 200:
                raise RuntimeError(f"Unexpected /health response before certification: {pre_health.status_code}")

            generate_data, inference_seconds = _run_llm_workload(client, args, log_path)
            health_data, devices_data = _collect_observability_snapshots(client, log_path)

        report = evaluate_hardware_certification(
            runtime_details=runtime_details,
            health_data=health_data,
            devices_data=devices_data,
            generate_data=generate_data,
            requested_device=args.device,
            model=args.model,
            startup_seconds=startup_seconds,
            inference_seconds=inference_seconds,
        )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report.to_dict(), indent=2),
                encoding="utf-8",
            )

        print(format_certification_report(report))
        if args.output:
            print(f"JSON report: {Path(args.output).resolve()}")
        print(f"Server log: {log_path}")
        return 0 if report.certified else 1
    except Exception as exc:
        print(f"Certification failed: {exc}", file=sys.stderr)
        print(f"Server log: {log_path}", file=sys.stderr)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
