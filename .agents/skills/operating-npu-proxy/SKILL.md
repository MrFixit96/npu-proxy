---
name: operating-npu-proxy
description: >-
  Runs, configures, and tests NPU Proxy locally. Use when starting the server,
  switching between mock and real inference, setting NPU_PROXY_* environment
  variables, choosing devices, or running the test suite. Also use when the user
  mentions running the proxy, mock mode, real inference, env vars, the CLI, or
  which pytest markers to use.
license: Apache-2.0
compatibility: >-
  Requires the npu-proxy repo and its .venv. Real inference (and NPU/GPU) requires
  OpenVINO and the matching Intel hardware; mock mode needs neither.
metadata:
  author: npu-proxy
  version: "1.0"
---

# Operating NPU Proxy

NPU Proxy is a single-user, localhost tool. It runs in **mock mode by default**
and only touches real hardware when explicitly enabled.

## Running the server

Entry point is `npu_proxy/cli.py` (console script `npu-proxy`) which serves the
FastAPI app `npu_proxy.main:app`.

```bash
# Default: mock mode, no hardware needed
npu-proxy                                          # console script
.\.venv\Scripts\python.exe -m npu_proxy.cli        # equivalent
# Real inference on a chosen device
$env:NPU_PROXY_REAL_INFERENCE = "1"
.\.venv\Scripts\python.exe -m npu_proxy.cli --device NPU
```

`scripts/start-server.ps1` is a convenience launcher. CLI flags cover host, port,
device/preferred/fallback device, and log level (see `cli.py` argument groups).

## Mock vs real inference — the master switch

`NPU_PROXY_REAL_INFERENCE=1` is the gate. When unset/`0`:
- No OpenVINO runtime import; device probing returns `["CPU"]`.
- Generation/embeddings are mocked.

Enable real inference only when a model is present and the target hardware exists.

## Most-used environment variables

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_REAL_INFERENCE` | `1` enables real OpenVINO inference (else mock) |
| `NPU_PROXY_DEVICE` / `NPU_PROXY_PREFERRED_DEVICE` | Routed/default device |
| `NPU_PROXY_FALLBACK_DEVICE` | Device to fall back to |
| `NPU_PROXY_FALLBACK_ON_BUSY` | Opt-in busy fallback instead of `503 device_busy` |
| `NPU_PROXY_WARMUP_DEVICES` | Devices to compile/warm at startup |
| `NPU_PROXY_MODEL_DIR` | Model cache root |
| `NPU_PROXY_HOST` / `NPU_PROXY_PORT` | Bind address |
| `NPU_PROXY_MAX_PROMPT_LEN` / `NPU_PROXY_TOKEN_LIMIT` | Prompt/token caps |

The full catalog (timeouts, caches, embedding vars, backend selection) is in
[references/configuration.md](references/configuration.md). Config precedence is
resolved in `npu_proxy/config.py`.

## Testing

```bash
# Fast suite (no hardware) — the default loop
.\.venv\Scripts\python.exe -m pytest -q -m "not slow and not e2e"
# Live hardware end-to-end (real device required)
.\.venv\Scripts\python.exe scripts\certify_npu.py --device NPU
```

Markers: `slow` and `e2e` are deselected for the fast loop. Run the fast suite
after any change; use certification (see `certifying-device-routing`) for
hardware-dependent behavior.

## Reference

- **Full environment-variable catalog and precedence**:
  [references/configuration.md](references/configuration.md)
