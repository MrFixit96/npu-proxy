# Benchmark CLI

This guide documents the benchmark CLI that exists today in `scripts\benchmark.py` and the hardware certification helper in `scripts\certify_npu.py`.

## What the script currently does

The benchmark script can:

- run a benchmark against one requested device, or every device reported by the runtime when no device is requested
- print a simple results table
- optionally write a JSON results file
- compare two benchmark JSON files
- copy/export a results JSON file

The current end-to-end user workflow is the `run` command.

## Current CLI surface

```text
python scripts\benchmark.py run [--device CPU|GPU|NPU] [--model MODEL] [--iterations N] [--warmup N] [--output FILE]
python scripts\benchmark.py compare --baseline BASELINE CURRENT
python scripts\benchmark.py export --input INPUT --output OUTPUT
```

### Important truth note

`run` is the primary benchmark workflow. `compare` reads a baseline JSON file and a current JSON file, then compares matching `device` + `model` rows by `warm_inference_ms`. `export` currently copies one JSON file to another path with pretty formatting.

## `run` command

### Options

| Option | Default | Current behavior |
|---|---|---|
| `--device` | omitted | Runs each device returned by `get_available_devices()` |
| `--model` | `tinyllama` | Model name or path resolved to the runtime OpenVINO model path |
| `--iterations` | `5` | Number of measured warm iterations |
| `--warmup` | `1` | Number of warmup iterations before measurement |
| `--output` | none | Writes JSON results to the given path |

### Example

```powershell
python scripts\benchmark.py run --device NPU --model tinyllama --iterations 5 --warmup 1 --output .\results\npu.json
```

If `--device` is omitted:

```powershell
python scripts\benchmark.py run --model tinyllama
```

the script benchmarks each device reported by the runtime. In mock mode or on hosts without visible accelerators, that may only be `CPU`.

## What gets measured

Current JSON results may include:

- `device`
- `model`
- `cold_start_ms`
- `warm_inference_ms`
- `first_token_ms` (supported by the result object, only present if populated)
- `tokens_per_second`
- `embedding_ms` (only when a result includes that field)

The top-level JSON shape written by `--output` is:

```json
{
  "timestamp": "2026-05-21T00:00:00Z",
  "system": {
    "python": "3.12.10",
    "platform": "Windows",
    "openvino": "2026.1.0"
  },
  "results": [
    {
      "device": "NPU",
      "model": "tinyllama",
      "cold_start_ms": 8120.0,
      "warm_inference_ms": 4030.0,
      "tokens_per_second": 12.5
    }
  ]
}
```

The values above are illustrative and environment-dependent. Hardware, OpenVINO version, model files, compile cache state, and whether real inference is enabled can all change the numbers.

## Practical usage notes

- Use `--output` when you want a stable artifact to compare manually or with `compare` later.
- The benchmark script does **not** currently expose batch-size, prompt-length, or embedding-specific subcommands on the CLI.
- Device names are uppercase in the CLI: `CPU`, `GPU`, `NPU`.
- The benchmark runner calls the local engine directly; it does not require an HTTP server host/port.

## Recommended workflow

```powershell
# NPU
python scripts\benchmark.py run --device NPU --model tinyllama --output .\results\npu.json

# CPU
python scripts\benchmark.py run --device CPU --model tinyllama --output .\results\cpu.json

# GPU
python scripts\benchmark.py run --device GPU --model tinyllama --output .\results\gpu.json
```

Then compare JSON files:

```powershell
python scripts\benchmark.py compare --baseline .\results\cpu.json .\results\npu.json
```

Or copy/pretty-format a result file:

```powershell
python scripts\benchmark.py export --input .\results\npu.json --output .\results\npu.pretty.json
```

## Hardware certification helper

`scripts\certify_npu.py` is separate from the benchmark CLI. It starts a temporary local proxy server with real inference enabled, sends one `/api/generate` request, collects health/device observations, and prints a certification report.

Current CLI surface:

```text
python scripts\certify_npu.py [--port PORT] [--startup-timeout SECONDS] [--request-timeout SECONDS] [--model MODEL] [--device NPU|GPU|CPU|AUTO] [--prompt PROMPT] [--output FILE]
```

Defaults:

| Option | Default | Current behavior |
|---|---|---|
| `--port` | omitted | Chooses a free loopback port automatically |
| `--startup-timeout` | `60.0` | Seconds to wait for server readiness |
| `--request-timeout` | `240.0` | Seconds to allow the real inference request |
| `--model` | repo default LLM model | Defaults to `tinyllama-1.1b-chat-int4-ov` when package defaults are available |
| `--device` | `NPU` | Requested device for the temporary server |
| `--prompt` | short certification prompt | Prompt sent to `/api/generate` |
| `--output` | none | Optional JSON certification report path |

The temporary server binds to `127.0.0.1`. If you choose a fixed port, use the default project port unless you have a reason not to:

```powershell
python scripts\certify_npu.py --port 8080 --device NPU --output .\results\certification.json
```

Certification requires OpenVINO to see an NPU and requires the requested model to already exist under `~/.cache/npu-proxy/models/<model>`. Results are environment-dependent; a failed certification can mean missing hardware, missing model files, startup timeout, or a runtime inference failure.
