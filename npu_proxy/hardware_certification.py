"""Helpers for certifying a real Intel NPU-backed NPU Proxy deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import openvino as ov

from npu_proxy.inference.devices import device_class

_PRIMARY_DEVICES = frozenset({"NPU", "GPU", "CPU"})


def _coalesce(*values: Any) -> Any:
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class RuntimeDetails:
    """Typed OpenVINO runtime details collected from the live host."""

    openvino_version: str
    available_devices: tuple[str, ...]
    npu_visible: bool
    npu_device_name: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RuntimeDetails":
        return cls(
            openvino_version=str(payload.get("openvino_version", "")),
            available_devices=tuple(str(device) for device in payload.get("available_devices", [])),
            npu_visible=bool(payload.get("npu_visible", False)),
            npu_device_name=payload.get("npu_device_name"),
        )


@dataclass(frozen=True)
class EngineSnapshot:
    """Typed subset of health-engine payloads consumed by certification."""

    status: str | None = None
    device: str | None = None
    backend: str | None = None
    model: str | None = None
    requested_device: str | None = None
    compile_cache_dir: str | None = None
    compile_cache_mode: str | None = None
    prefix_cache_mode: str | None = None
    runtime_features: dict[str, Any] | None = None
    model_load_seconds: float | None = None
    load_diagnostics: tuple[dict[str, Any], ...] | None = None
    last_generation_stats: dict[str, Any] | None = None
    model_path: str | None = None
    model_format: str | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    dimensions: int | None = None
    is_production: bool | None = None
    is_fallback: bool | None = None
    fallback_reason: str | None = None
    fallback_mode: str | None = None
    load_error: str | None = None
    cache: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EngineSnapshot":
        return cls(
            status=payload.get("status"),
            device=payload.get("device"),
            backend=payload.get("backend"),
            model=payload.get("model"),
            requested_device=payload.get("requested_device"),
            compile_cache_dir=payload.get("compile_cache_dir"),
            compile_cache_mode=payload.get("compile_cache_mode"),
            prefix_cache_mode=payload.get("prefix_cache_mode"),
            runtime_features=dict(payload.get("runtime_features") or {})
            if payload.get("runtime_features") is not None
            else None,
            model_load_seconds=payload.get("model_load_seconds"),
            load_diagnostics=(
                tuple(dict(item) for item in payload.get("load_diagnostics") or ())
                if payload.get("load_diagnostics") is not None
                else None
            ),
            last_generation_stats=dict(payload.get("last_generation_stats") or {})
            if payload.get("last_generation_stats") is not None
            else None,
            model_path=payload.get("model_path"),
            model_format=payload.get("model_format"),
            requested_model=payload.get("requested_model"),
            resolved_model=payload.get("resolved_model"),
            dimensions=payload.get("dimensions"),
            is_production=payload.get("is_production"),
            is_fallback=payload.get("is_fallback"),
            fallback_reason=payload.get("fallback_reason"),
            fallback_mode=payload.get("fallback_mode"),
            load_error=payload.get("load_error"),
            cache=dict(payload.get("cache") or {}) if payload.get("cache") is not None else None,
        )


@dataclass(frozen=True)
class CertificationLLMRuntimeState:
    """Typed runtime-state payload recorded in certification reports."""

    backend: str | None
    requested_device: str | None
    fallback_device: str | None
    used_fallback: bool | None
    compile_cache_dir: str | None
    compile_cache_mode: str | None
    prefix_cache_mode: str | None
    runtime_features: dict[str, Any] | None
    model_load_seconds: float | None
    load_diagnostics: tuple[dict[str, Any], ...]
    last_generation_stats: dict[str, Any] | None
    model_path: str | None
    model_format: str | None

    @classmethod
    def from_sources(
        cls,
        engine: EngineSnapshot,
        device_info: dict[str, Any],
        used_fallback: bool | None,
    ) -> "CertificationLLMRuntimeState":
        return cls(
            backend=engine.backend or device_info.get("backend"),
            requested_device=_coalesce(engine.requested_device, device_info.get("requested_device")),
            fallback_device=_coalesce(device_info.get("fallback_device")),
            used_fallback=used_fallback,
            compile_cache_dir=_coalesce(engine.compile_cache_dir, device_info.get("compile_cache_dir")),
            compile_cache_mode=_coalesce(
                engine.compile_cache_mode,
                device_info.get("compile_cache_mode"),
            ),
            prefix_cache_mode=_coalesce(
                engine.prefix_cache_mode,
                device_info.get("prefix_cache_mode"),
            ),
            runtime_features=_coalesce(engine.runtime_features, device_info.get("runtime_features")),
            model_load_seconds=_coalesce(engine.model_load_seconds, device_info.get("model_load_seconds")),
            load_diagnostics=tuple(
                dict(item)
                for item in _coalesce(
                    engine.load_diagnostics,
                    device_info.get("load_diagnostics"),
                    (),
                )
            ),
            last_generation_stats=_coalesce(
                engine.last_generation_stats,
                device_info.get("last_generation_stats"),
            ),
            model_path=_coalesce(engine.model_path, device_info.get("model_path")),
            model_format=_coalesce(engine.model_format, device_info.get("model_format")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_device": self.requested_device,
            "fallback_device": self.fallback_device,
            "used_fallback": self.used_fallback,
            "compile_cache_dir": self.compile_cache_dir,
            "compile_cache_mode": self.compile_cache_mode,
            "prefix_cache_mode": self.prefix_cache_mode,
            "runtime_features": self.runtime_features,
            "model_load_seconds": self.model_load_seconds,
            "load_diagnostics": [dict(item) for item in self.load_diagnostics],
            "last_generation_stats": self.last_generation_stats,
            "model_path": self.model_path,
            "model_format": self.model_format,
        }


@dataclass(frozen=True)
class CertificationEmbeddingRuntimeState:
    """Typed embedding runtime-state payload recorded in certification reports."""

    device: str | None
    requested_device: str | None
    requested_model: str | None
    resolved_model: str | None
    dimensions: int | None
    is_production: bool | None
    is_fallback: bool | None
    fallback_reason: str | None
    fallback_mode: str | None
    load_error: str | None
    cache: dict[str, Any] | None

    @classmethod
    def from_sources(
        cls,
        engine: EngineSnapshot,
        embedding_device_info: dict[str, Any],
    ) -> "CertificationEmbeddingRuntimeState":
        return cls(
            device=embedding_device_info.get("device") or engine.device,
            requested_device=embedding_device_info.get("requested_device") or engine.requested_device,
            requested_model=engine.requested_model,
            resolved_model=engine.resolved_model,
            dimensions=engine.dimensions,
            is_production=engine.is_production,
            is_fallback=engine.is_fallback,
            fallback_reason=engine.fallback_reason,
            fallback_mode=engine.fallback_mode,
            load_error=engine.load_error,
            cache=engine.cache,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "requested_device": self.requested_device,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "dimensions": self.dimensions,
            "is_production": self.is_production,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "fallback_mode": self.fallback_mode,
            "load_error": self.load_error,
            "cache": self.cache,
        }


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
    llm_backend: str | None
    llm_runtime_state: dict[str, Any]
    embedding_status: str | None
    embedding_backend: str | None
    embedding_device: str | None
    embedding_model: str | None
    embedding_runtime_state: dict[str, Any] | None
    model: str
    response_preview: str
    response_length: int
    startup_seconds: float
    inference_seconds: float
    failures: list[str]
    routing_checks: dict[str, Any] = field(default_factory=dict)

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

    runtime = RuntimeDetails(
        openvino_version=str(ov.__version__),
        available_devices=tuple(available_devices),
        npu_visible=npu_visible,
        npu_device_name=npu_device_name,
    )
    return {
        "openvino_version": runtime.openvino_version,
        "available_devices": list(runtime.available_devices),
        "npu_visible": runtime.npu_visible,
        "npu_device_name": runtime.npu_device_name,
    }


def _header_value(headers: dict[str, Any], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value).strip().upper()
    return None


def evaluate_routing_certification(
    *,
    short_headers: dict[str, Any],
    long_headers: dict[str, Any],
    fallback_device: str,
    expected_primary_device: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate per-request routing headers captured from live certification calls.

    ``expected_primary_device`` is the device class a short prompt is expected to
    execute on (the requested/preferred device, e.g. ``NPU`` or ``GPU``). When
    ``None`` (e.g. an ``AUTO`` run), the specific-device assertion is skipped but
    routed/execution truthfulness is still enforced.
    """
    expected_fallback = fallback_device.strip().upper()
    short_routed = _header_value(short_headers, "X-NPU-Proxy-Routed-Device")
    short_execution = _header_value(short_headers, "X-NPU-Proxy-Execution-Device")
    long_routed = _header_value(long_headers, "X-NPU-Proxy-Routed-Device")
    long_execution = _header_value(long_headers, "X-NPU-Proxy-Execution-Device")
    long_reason = _header_value(long_headers, "X-NPU-Proxy-Fallback-Reason")

    failures: list[str] = []
    if not short_routed or not short_execution:
        failures.append("Short routing check did not return routed/execution device headers.")
    elif short_routed != short_execution:
        failures.append(
            f"Short routing check routed to {short_routed!r} but executed on {short_execution!r}."
        )
    elif expected_primary_device and short_execution != expected_primary_device:
        failures.append(
            f"Short routing check executed on {short_execution!r}, not {expected_primary_device!r}."
        )

    if not long_execution:
        failures.append("Long routing check did not return an execution device header.")
    elif long_execution != expected_fallback:
        failures.append(
            f"Long routing check executed on {long_execution!r}, not configured fallback {expected_fallback!r}."
        )

    return {
        "short": {
            "routed_device": short_routed,
            "execution_device": short_execution,
        },
        "long": {
            "routed_device": long_routed,
            "execution_device": long_execution,
            "fallback_reason": long_reason,
            "expected_fallback_device": expected_fallback,
        },
    }, failures


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
    routing_data: dict[str, Any] | None = None,
) -> CertificationReport:
    """Evaluate whether a live run qualifies as real certification on the requested device."""
    failures: list[str] = []
    expected_primary = device_class(requested_device)
    runtime = RuntimeDetails.from_payload(runtime_details)
    llm_engine = EngineSnapshot.from_payload(dict(health_data.get("engines", {}).get("llm", {})))
    embedding_engine = EngineSnapshot.from_payload(
        dict(health_data.get("engines", {}).get("embedding", {}))
    )
    device_info = dict(devices_data.get("device_info") or {})
    embedding_devices = dict(devices_data.get("embedding", {}) or {})
    embedding_device_info = dict(embedding_devices.get("device_info") or {})
    response_text = str(generate_data.get("response", "")).strip()
    active_device = devices_data.get("active_device")
    used_fallback = device_info.get("used_fallback")
    llm_status = llm_engine.status
    llm_runtime_state = CertificationLLMRuntimeState.from_sources(
        llm_engine,
        device_info,
        used_fallback,
    )
    llm_backend = llm_runtime_state.backend

    embedding_runtime_state: dict[str, Any] | None = None
    if embedding_engine.status is not None or embedding_device_info:
        embedding_runtime_state = CertificationEmbeddingRuntimeState.from_sources(
            embedding_engine,
            embedding_device_info,
        ).to_dict()

    available_classes = {device_class(device) for device in runtime.available_devices}
    if expected_primary in _PRIMARY_DEVICES and expected_primary not in available_classes:
        failures.append(f"OpenVINO did not report a {expected_primary} device on this host.")

    if health_data.get("status") != "healthy":
        failures.append(f"Service health was {health_data.get('status')!r}, not 'healthy'.")

    if llm_status != "loaded":
        failures.append(f"LLM engine status was {llm_status!r}, not 'loaded'.")

    if llm_backend and llm_backend != "openvino":
        failures.append(f"LLM backend was {llm_backend!r}, not 'openvino'.")

    if expected_primary in _PRIMARY_DEVICES and llm_engine.device != expected_primary:
        failures.append(
            f"/health reported the LLM engine on {llm_engine.device!r}, not {expected_primary!r}."
        )

    if expected_primary in _PRIMARY_DEVICES and active_device != expected_primary:
        failures.append(
            f"/health/devices reported active_device={active_device!r}, not {expected_primary!r}."
        )

    if used_fallback is not False:
        failures.append(
            "The live inference path used a fallback device instead of staying on the requested device."
        )

    if not response_text:
        failures.append("The generate endpoint returned an empty response.")

    routing_checks: dict[str, Any] = {}
    if routing_data is not None:
        routing_checks, routing_failures = evaluate_routing_certification(
            short_headers=dict(routing_data.get("short_headers") or {}),
            long_headers=dict(routing_data.get("long_headers") or {}),
            fallback_device=str(routing_data.get("fallback_device") or "CPU"),
            expected_primary_device=(
                expected_primary if expected_primary in _PRIMARY_DEVICES else None
            ),
        )
        failures.extend(routing_failures)

    return CertificationReport(
        certified=not failures,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        openvino_version=runtime.openvino_version,
        available_devices=list(runtime.available_devices),
        npu_visible=runtime.npu_visible,
        npu_device_name=runtime.npu_device_name,
        requested_device=requested_device,
        active_device=active_device,
        used_fallback=used_fallback,
        llm_status=llm_status,
        llm_backend=llm_backend,
        llm_runtime_state=llm_runtime_state.to_dict(),
        embedding_status=embedding_engine.status,
        embedding_backend=embedding_engine.backend,
        embedding_device=embedding_engine.device,
        embedding_model=embedding_engine.model,
        embedding_runtime_state=embedding_runtime_state,
        model=model,
        response_preview=response_text[:160],
        response_length=len(response_text),
        startup_seconds=round(startup_seconds, 3),
        inference_seconds=round(inference_seconds, 3),
        failures=failures,
        routing_checks=routing_checks,
    )


def format_certification_report(report: CertificationReport) -> str:
    """Render a human-readable certification summary."""
    runtime_features = report.llm_runtime_state.get("runtime_features") or {}
    degraded_features = runtime_features.get("degraded_features") or []
    compile_cache_summary = report.llm_runtime_state.get("compile_cache_dir") or "disabled"
    if report.llm_runtime_state.get("compile_cache_mode"):
        compile_cache_summary = (
            f"{compile_cache_summary} ({report.llm_runtime_state['compile_cache_mode']})"
        )
    prefix_cache_summary = runtime_features.get("prefix_cache_enabled")
    generation_stats = report.llm_runtime_state.get("last_generation_stats") or {}
    degraded_summary = ", ".join(str(feature) for feature in degraded_features) if degraded_features else "none"

    lines = [
        f"Certified: {'YES' if report.certified else 'NO'}",
        f"Requested device: {report.requested_device}",
        f"Active device: {report.active_device or 'unknown'}",
        f"Fallback used: {report.used_fallback}",
        f"LLM status: {report.llm_status or 'unknown'}",
        f"LLM backend: {report.llm_backend or 'unknown'}",
        f"Compile cache: {compile_cache_summary}",
        f"Prefix cache enabled: {prefix_cache_summary}",
        f"Runtime degraded features: {degraded_summary}",
        f"NPU visible: {report.npu_visible}",
        f"NPU name: {report.npu_device_name or 'unknown'}",
        (
            "Embedding engine: "
            f"{report.embedding_status or 'unknown'} via {report.embedding_backend or 'unknown'} "
            f"on {report.embedding_device or 'unknown'}"
        ),
        f"Model: {report.model}",
        f"OpenVINO: {report.openvino_version}",
        f"Startup seconds: {report.startup_seconds}",
        f"Inference seconds: {report.inference_seconds}",
        (
            "Last generation stats: "
            f"ttft={generation_stats.get('ttft_seconds')}, "
            f"throughput={generation_stats.get('throughput_tokens_per_second')}"
        ),
        f"Response preview: {report.response_preview or '<empty>'}",
    ]

    if report.routing_checks:
        short_check = report.routing_checks.get("short", {})
        long_check = report.routing_checks.get("long", {})
        lines.extend(
            [
                (
                    "Short routing: "
                    f"routed={short_check.get('routed_device')}, "
                    f"execution={short_check.get('execution_device')}"
                ),
                (
                    "Long routing: "
                    f"routed={long_check.get('routed_device')}, "
                    f"execution={long_check.get('execution_device')}, "
                    f"expected={long_check.get('expected_fallback_device')}"
                ),
            ]
        )

    if report.failures:
        lines.append("Failures:")
        lines.extend(f"- {failure}" for failure in report.failures)

    return "\n".join(lines)
