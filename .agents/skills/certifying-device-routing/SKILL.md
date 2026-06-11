---
name: certifying-device-routing
description: >-
  Validates and safely modifies NPU Proxy's context-aware device routing and its
  live hardware certification. Use when changing device selection, fallback
  chains, routing/observability headers or metrics, or when running
  scripts/certify_npu.py for NPU, GPU, or CPU. Also use when the user mentions
  device routing, GPU.0/GPU.1 matching, fallback truthfulness, hardware
  certification, X-NPU-Proxy-* headers, or 503 device_busy backpressure.
license: Apache-2.0
compatibility: >-
  Requires the npu-proxy repo and its .venv (Python 3.10+, OpenVINO 2026.x).
  Live certification additionally requires real Intel NPU/GPU hardware and
  NPU_PROXY_REAL_INFERENCE=1.
metadata:
  author: npu-proxy
  version: "1.0"
---

# Certifying Device Routing

NPU Proxy routes each generation request to a compute device (`NPU`/`GPU`/`CPU`)
and must report **truthfully** which device actually executed it. This skill
covers the routing subsystem's hard invariants and the live-hardware
certification loop that proves them.

## Orientation: where the logic lives

| Concern | File | Symbols |
| --- | --- | --- |
| Device canonicalization & classes | `npu_proxy/inference/devices.py` | `normalize_device`, `device_class`, `DEVICE_FALLBACK_CHAIN`, `Device`, `FallbackReason` |
| Engine/device selection | `npu_proxy/inference/engine.py` | `select_best_device` (~L283), `fallback_devices_after` (~L1400) |
| Advisory routing | `npu_proxy/inference/context_router.py` | `get_fallback_device` |
| Truthful headers | `npu_proxy/api/header_utils.py` | `SINGLE_ENGINE_ROUTE_REASON`, `X-NPU-Proxy-*` |
| Routing/inference metrics | `npu_proxy/metrics.py` | `_normalize_device_label`, `_ALLOWED_DEVICES` |
| Cert evaluator | `npu_proxy/hardware_certification.py` | `evaluate_hardware_certification` |
| Live cert runner | `scripts/certify_npu.py` | `--device {NPU,GPU,CPU,AUTO}` |

## Non-negotiable invariants

Routing changes are **low-freedom**: the wire strings and matching rules below
are a public contract. Read [references/invariants.md](references/invariants.md)
before editing any file in the table above. The three that cause the most damage:

1. **Class-aware device matching.** OpenVINO enumerates accelerators as
   `GPU.0`/`GPU.1`. Never compare a bare class (`"GPU"`) against that list with
   `in`. Always reason about the class via `device_class(d)`; a `GPU` request
   must match `GPU.0`/`GPU.1` instead of silently falling through to the next
   device in `DEVICE_FALLBACK_CHAIN`.
2. **Stable wire strings.** Fallback reasons (`"busy"`, `"device_fallback"`),
   the metric device label set (`{"npu","gpu","cpu","auto","mock","unknown"}`),
   the route reason (`"single_engine_runtime"`), and the metric/series names are
   consumed downstream. Do not rename or re-case them.
3. **The test-patching rule.** Tests patch each module's *own*
   `get_available_devices` symbol. Derive device classes from the **caller's
   patchable** `get_available_devices()`, never by re-probing
   `devices.get_available_devices()` directly — that bypasses the patch and
   silently breaks the suite.

## Validation workflow

Copy this checklist and tick items as you go. Do not declare a routing change
done until every box is checked.

```
Routing change validation:
- [ ] 1. Fast suite green
- [ ] 2. Live cert: NPU passes
- [ ] 3. Live cert: GPU passes (real GPU present)
- [ ] 4. Live cert: CPU passes
- [ ] 5. Routing truthfulness verified in each report
```

**Step 1 — Fast suite** (no hardware needed):

```bash
.\.venv\Scripts\python.exe -m pytest -q -m "not slow and not e2e"
```

**Steps 2-4 — Live certification** (real hardware; one run per device):

```bash
.\.venv\Scripts\python.exe scripts\certify_npu.py --device NPU
.\.venv\Scripts\python.exe scripts\certify_npu.py --device GPU
.\.venv\Scripts\python.exe scripts\certify_npu.py --device CPU
```

Each run boots a real server, sends a live generation, and prints a PASS/FAIL
report. For flags, the PASS criteria, and how to read a report, see
[references/certification.md](references/certification.md).

**Step 5 — Truthfulness check.** A report PASSES only when the *requested*
device class matches the executed device and any fallback is reported honestly
(`used_fallback` + a real `fallback_reason`). A request "routed=GPU" that
"executed=NPU" with `used_fallback=False` is a **routing-truthfulness
violation**, even if the request succeeded.

## Feedback loop

Run cert → read the failure reason in the report (and
`build/certification/certify-<device>.log`) → fix the routing/header/metric site
→ re-run the same device. Only move to the next device once the current one
passes. Re-run the fast suite after any code change, because routing fixes
frequently touch metric labels and header construction that unit tests assert.

## Reference material

- **Invariant catalog & code-edit rules**: [references/invariants.md](references/invariants.md)
- **Running and interpreting certification**: [references/certification.md](references/certification.md)
