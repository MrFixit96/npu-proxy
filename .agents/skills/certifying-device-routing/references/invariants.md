# Routing invariants and code-edit rules

## Contents

- Stable wire strings (must never change)
- Class-aware device matching
- The test-patching rule
- `device_class` call sites
- Metric label collapsing
- Quick edit checklist

## Stable wire strings (must never change)

These values are consumed by headers, Prometheus, and external clients. Treat
them as a frozen contract.

| Kind | Value(s) | Defined in |
| --- | --- | --- |
| Fallback reasons | `busy`, `device_fallback` | `FallbackReason`, `devices.py` |
| Metric device label set | `npu`, `gpu`, `cpu`, `auto`, `mock`, `unknown` | `_ALLOWED_DEVICES`, `metrics.py` |
| Route reason | `single_engine_runtime` | `SINGLE_ENGINE_ROUTE_REASON`, `header_utils.py` |
| Fallback chain | `["NPU", "GPU", "CPU"]` | `DEVICE_FALLBACK_CHAIN`, `devices.py` |
| Metric names | `npu_proxy_inference_total`, `npu_proxy_inference_latency_seconds`, `npu_proxy_tokens_per_second`, `npu_proxy_routing_executions_total` | `metrics.py` |
| Routing headers | `X-NPU-Proxy-Device`, `X-NPU-Proxy-Routed-Device`, `X-NPU-Proxy-Execution-Device`, `X-NPU-Proxy-Fallback-Reason`, `X-NPU-Proxy-Route-Reason`, `X-NPU-Proxy-Token-Count` | `header_utils.py` |

Header semantics:

- `X-NPU-Proxy-Device` — backward-compatible alias; equals the execution device.
- `X-NPU-Proxy-Routed-Device` — device the context router classified for the request.
- `X-NPU-Proxy-Execution-Device` — device the request actually ran on.
- `X-NPU-Proxy-Fallback-Reason` — present **only** when execution differs from routed.

## Class-aware device matching

OpenVINO enumerates multiple accelerators of one kind with a numeric suffix
(`GPU.0`, `GPU.1`) and accepts the bare class name (`GPU`) as an alias for the
first instance at compile time.

Rule: availability and fallback decisions reason about the device **class**.

```python
from npu_proxy.inference.devices import device_class

available = get_available_devices()                      # caller's patchable symbol
classes = {device_class(d) for d in available if d}      # {"GPU", "NPU", "CPU"}
gpu_available = "GPU" in classes                          # matches GPU.0 / GPU.1
```

Anti-pattern that caused a silent NPU redirect:

```python
# WRONG: "GPU" is never literally in ["GPU.0", "GPU.1"], so this reports
# GPU unavailable and quietly selects the next chain device (NPU).
gpu_available = "GPU" in get_available_devices()
```

`select_best_device` matches a preferred device if it is exactly present
(preserving an explicit `GPU.1`) **or** if it is a chain class whose class is in
the available classes. An explicit suffixed request (`GPU.1`) returns
`("GPU.1", None)` with no chain fallback, because a specific instance is not a
chain class.

## The test-patching rule

Each module imports and calls its own `get_available_devices`. Tests patch that
per-module symbol (e.g. `npu_proxy.inference.engine.get_available_devices`).

- **Do**: derive classes from the caller's local `get_available_devices()`.
- **Do not**: add a shared helper that re-probes
  `devices.get_available_devices()` directly — it bypasses the patch and breaks
  tests while appearing correct in production.

## `device_class` call sites

When adding a new device-availability or fallback decision, apply `device_class`
consistently. Existing sites:

- `engine.select_best_device`
- `engine.fallback_devices_after` (index the chain by `device_class(device)` so a
  busy `GPU.1` descends to `CPU` and never offers a higher-priority `NPU`)
- `context_router.get_fallback_device`
- `main._warmup_configured_devices`
- `health.check_gpu_available` and the `/health` availability booleans

## Metric label collapsing

`record_inference`, `record_tokens_per_second`, and the routing-execution metric
wrap device values with `_normalize_device_label`, which collapses `GPU.0`/`GPU.1`
to `gpu`. Without this, a discrete-GPU execution records as `device="unknown"`.
Any new metric that takes a device label must do the same.

## Quick edit checklist

```
Before committing a routing change:
- [ ] No bare-class `in available_devices` comparisons remain
- [ ] New availability checks use device_class()
- [ ] No wire string renamed/re-cased
- [ ] Device classes derived from the caller's patchable get_available_devices()
- [ ] New device metric labels pass through _normalize_device_label
- [ ] Fast suite green; live cert green for every affected device
```
