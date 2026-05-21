"""Helpers for certifying a real Intel NPU-backed NPU Proxy deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import openvino as ov


@dataclass
class CertificationReport:
    """Outcome of a hardware-backed certification run."""

    certified: bool
    timestamp_utc: str
    openvino_version: str
    available_devices: list[str]
    npu_visible: bool
    npu_device_name: str | None
    requested_device: str
    active_device: str | None
    used_fallback: bool | None
    llm_status: str | None
    model: str
    response_preview: str
    response_length: int
    startup_seconds: float
    inference_seconds: float
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serializable dictionary."""
        return asdict(self)


def collect_openvino_runtime_details() -> dict[str, Any]:
    """Collect OpenVINO runtime and NPU visibility details from the host."""
    core = ov.Core()
    available_devices = list(core.available_devices)
    npu_visible = "NPU" in available_devices
    npu_device_name = None

    if npu_visible:
        try:
            npu_device_name = str(core.get_property("NPU", "FULL_DEVICE_NAME"))
        except Exception:
            npu_device_name = "NPU"

    return {
        "openvino_version": ov.__version__,
        "available_devices": available_devices,
        "npu_visible": npu_visible,
        "npu_device_name": npu_device_name,
    }


def evaluate_hardware_certification(
    *,
    runtime_details: dict[str, Any],
    health_data: dict[str, Any],
    devices_data: dict[str, Any],
    generate_data: dict[str, Any],
    requested_device: str,
    model: str,
    startup_seconds: float,
    inference_seconds: float,
) -> CertificationReport:
    """Evaluate whether a live run qualifies as real NPU certification."""
    failures: list[str] = []
    llm_engine = health_data.get("engines", {}).get("llm", {})
    device_info = devices_data.get("device_info") or {}
    response_text = str(generate_data.get("response", "")).strip()
    active_device = devices_data.get("active_device")
    used_fallback = device_info.get("used_fallback")
    llm_status = llm_engine.get("status")

    if not runtime_details.get("npu_visible", False):
        failures.append("OpenVINO did not report an NPU device on this host.")

    if health_data.get("status") != "healthy":
        failures.append(f"Service health was {health_data.get('status')!r}, not 'healthy'.")

    if llm_status != "loaded":
        failures.append(f"LLM engine status was {llm_status!r}, not 'loaded'.")

    if llm_engine.get("device") != "NPU":
        failures.append(
            f"/health reported the LLM engine on {llm_engine.get('device')!r}, not 'NPU'."
        )

    if active_device != "NPU":
        failures.append(
            f"/health/devices reported active_device={active_device!r}, not 'NPU'."
        )

    if used_fallback is not False:
        failures.append("The live inference path used a fallback device instead of staying on NPU.")

    if not response_text:
        failures.append("The generate endpoint returned an empty response.")

    return CertificationReport(
        certified=not failures,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        openvino_version=str(runtime_details.get("openvino_version", "")),
        available_devices=list(runtime_details.get("available_devices", [])),
        npu_visible=bool(runtime_details.get("npu_visible", False)),
        npu_device_name=runtime_details.get("npu_device_name"),
        requested_device=requested_device,
        active_device=active_device,
        used_fallback=used_fallback,
        llm_status=llm_status,
        model=model,
        response_preview=response_text[:160],
        response_length=len(response_text),
        startup_seconds=round(startup_seconds, 3),
        inference_seconds=round(inference_seconds, 3),
        failures=failures,
    )


def format_certification_report(report: CertificationReport) -> str:
    """Render a human-readable certification summary."""
    lines = [
        f"Certified: {'YES' if report.certified else 'NO'}",
        f"Requested device: {report.requested_device}",
        f"Active device: {report.active_device or 'unknown'}",
        f"Fallback used: {report.used_fallback}",
        f"LLM status: {report.llm_status or 'unknown'}",
        f"NPU visible: {report.npu_visible}",
        f"NPU name: {report.npu_device_name or 'unknown'}",
        f"Model: {report.model}",
        f"OpenVINO: {report.openvino_version}",
        f"Startup seconds: {report.startup_seconds}",
        f"Inference seconds: {report.inference_seconds}",
        f"Response preview: {report.response_preview or '<empty>'}",
    ]

    if report.failures:
        lines.append("Failures:")
        lines.extend(f"- {failure}" for failure in report.failures)

    return "\n".join(lines)
