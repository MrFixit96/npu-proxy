# Running and interpreting hardware certification

## Contents

- Prerequisites
- Command and flags
- What a PASS means
- Reading a failure
- Logs and artifacts

## Prerequisites

- The repo `.venv` is active/usable (`.\.venv\Scripts\python.exe`).
- Real inference is enabled for the run: certification sets up a live server, so
  the target device must physically exist. NPU/GPU runs require the matching
  Intel hardware; CPU always works.
- A local OpenVINO model directory exists for the model under test.

## Command and flags

Run one certification per device from the repo root:

```bash
.\.venv\Scripts\python.exe scripts\certify_npu.py --device NPU
```

| Flag | Purpose |
| --- | --- |
| `--device {NPU,GPU,CPU,AUTO}` | Device to certify. Certifies against the **requested** device. |
| `--model MODEL` | Model name (defaults to the engine's default LLM model). |
| `--prompt PROMPT` | Override the certification prompt. |
| `--port PORT` | Fixed port (otherwise a free port is chosen). |
| `--startup-timeout` / `--request-timeout` | Server boot and request budgets in seconds. |
| `--output PATH` | Write the JSON certification report to a file. |

`AUTO` skips device-specific assertions (the executed device is not pinned) but
still enforces honest fallback reporting.

## What a PASS means

A run PASSES only when all hold:

1. The server boots and serves a live generation (no mock).
2. The **executed device class** equals the **requested device class**
   (`device_class(executed) == device_class(requested)`), for NPU/GPU/CPU.
3. Fallback is reported truthfully: if execution differs from routed, the report
   carries `used_fallback=True` and a real `fallback_reason`; if they match,
   `used_fallback` is `False`.

A successful response on the wrong device with `used_fallback=False` is a
**routing-truthfulness violation** and must FAIL — the headers and `/health`
would otherwise lie about where work ran.

## Reading a failure

The printed report includes the requested device, the resolved execution device,
the fallback flag/reason, and runtime details (including `available_devices`).
Typical failure shapes:

- **routed=GPU, executed=NPU, used_fallback=False** → a class-matching regression;
  check that the availability path uses `device_class` (see
  [invariants.md](invariants.md)).
- **device label `unknown` in metrics** → a metric label skipped
  `_normalize_device_label`.
- **server failed to start** → unrelated to routing; inspect the server log.

## Logs and artifacts

Per-device server logs are written under:

```
build/certification/certify-<device>.log
```

Tail this file when a run fails to start or to confirm which device the engine
actually compiled the model on. Use `--output report.json` to capture the
machine-readable report for diffing across runs.
