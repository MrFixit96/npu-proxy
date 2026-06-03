# NPU Proxy

Ollama-compatible and OpenAI-compatible API proxy for local Intel NPU inference via OpenVINO.

NPU Proxy is a **local, single-user developer-workstation tool**. It binds to loopback by default, has no authentication by design, and should not be treated as a shared production proxy.

This repository currently ships:

- FastAPI app with OpenAI-style chat, embeddings, and model listing endpoints
- Ollama-style generate, chat, embeddings, pull, tags, version, and model-discovery endpoints
- Windows start scripts plus Linux packaging assets
- Optional real inference mode backed by local OpenVINO model directories

## Current behavior at a glance

- **Default CLI bind:** `127.0.0.1:8080`
- **Default runtime device:** `NPU`
- **Default token limit:** `1800`
- **Default workers:** `1`
- **Default LLM timeout:** `180` seconds
- **Default mode:** mock responses unless `NPU_PROXY_REAL_INFERENCE=1` or `--real-inference` is set
- **OpenAI chat streaming:** SSE (`text/event-stream`)
- **Ollama streaming:** newline-delimited JSON chunks (`application/x-ndjson`)
- **Default LLM model path:** `~/.cache/npu-proxy/models/tinyllama-1.1b-chat-int4-ov`
- **Host-header allow-list:** loopback and test clients by default (`localhost`, `127.0.0.1`, `::1`, `[::1]`, `testserver`, `test`)

## Routing truth and roadmap

There are two different ideas that are easy to blur together:

1. **Current routing classification** — the router can classify a request as better suited for `NPU`, `GPU`, or `CPU` based on prompt size.
2. **Real per-request routing** — the runtime actually executes that request on the classified device.

### Current release truth

Today the service should be understood as **truthful single-engine execution**:

- the router can still make an advisory choice
- the API may report that choice for diagnostics
- the real inference path still runs through the currently loaded engine/runtime state
- the service should not promise that each request can switch devices independently

In other words, the router is currently more like a **dispatcher recommendation** than an automatic train-track switch.

### Roadmap

- **Current release**: harden the current behavior so headers, health, and documentation describe what the runtime really does today
- **Future release**: real per-request engine/device routing would need truthful execution binding, observability, certification, and concurrency rules

## Release-truth snapshot for this docs pass

- NPU Proxy is documented as a local developer-workstation process, not a production gateway.
- Mock mode is the default; real inference must be explicitly enabled.
- Host-header allow-listing, path hardening, and sanitized errors are implemented.
- `/api/tags` is implemented and `/api/show` returns registry-backed model metadata.
- OpenAI and Ollama generation responses now expose stop/length reasons.
- Embedding inputs are validated before engine execution, with a batch limit of 128.

## Quick start

### Prerequisites

- Python 3.10+
- Windows 11 host with current Intel NPU drivers for NPU-backed inference
- OpenVINO/OpenVINO GenAI packages from `requirements.txt`

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Download the default LLM model

Real LLM inference expects the default model directory to exist at:

```text
~/.cache/npu-proxy/models/tinyllama-1.1b-chat-int4-ov
```

One way to populate it is:

```powershell
pip install huggingface-hub
hf download OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov --local-dir ~/.cache/npu-proxy/models/tinyllama-1.1b-chat-int4-ov
```

### Start the server

#### CLI entry point

```powershell
# CLI defaults to 127.0.0.1:8080 in mock mode
npu-proxy

# Real inference on the default local bind
npu-proxy --real-inference

# Explicit host/port
npu-proxy --host 127.0.0.1 --port 8080 --real-inference
```

#### Recommended Windows launcher

```powershell
.\scripts\start-server.ps1
```

That script defaults to loopback (`127.0.0.1`) and port `11435` unless `NPU_PROXY_HOST` or `NPU_PROXY_PORT` is already set. Binding all interfaces requires explicit opt-in:

```powershell
.\scripts\start-server.ps1 -ListenAll
# or
$env:NPU_PROXY_LISTEN_ALL = "true"
.\scripts\start-server.ps1
```

If you bind beyond loopback, also set `NPU_PROXY_ALLOWED_HOSTS` or `--allowed-hosts` so legitimate remote Host headers are allowed.

#### Direct uvicorn

```powershell
python -m uvicorn npu_proxy.main:app --host 127.0.0.1 --port 8080

$env:NPU_PROXY_REAL_INFERENCE = "1"
python -m uvicorn npu_proxy.main:app --host 127.0.0.1 --port 8080
```

### Verify the service

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

Industry-standard probes are also available:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health/liveness
Invoke-RestMethod http://127.0.0.1:8080/health/readiness
```

Health contract:

- `/health` is an **observational summary** and does not auto-load models
- `/health/liveness` is the cheap process-up probe
- `/health/readiness` reports whether warmed runtime state is actually ready to serve traffic
- when models are not loaded, health surfaces say that explicitly instead of hiding it behind probe-triggered initialization

## Using with Ollama clients

Point Ollama-compatible clients at the local NPU Proxy server:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:8080"
ollama pull tinyllama
ollama run tinyllama "What is 2+2?"
ollama ps
```

If you use `scripts\start-server.ps1` without overriding its port, use `http://127.0.0.1:11435` instead.

WSL 2, when the Windows host intentionally listens beyond loopback:

```bash
WINDOWS_HOST=$(ip route show | grep default | awk '{print $3}')
export OLLAMA_HOST="http://${WINDOWS_HOST}:11435"
ollama pull tinyllama
ollama run tinyllama "Hello"
```

## Using with Claude Code

Windows:

```powershell
.\scripts\ollama-launch-claude.ps1
```

WSL 2:

```bash
./scripts/ollama-launch-claude.sh
```

Both scripts set `OLLAMA_HOST` and check `/health` before launching `claude --provider ollama`.

## Using with OpenAI SDKs

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="tinyllama-1.1b-chat-int4-ov",
    messages=[{"role": "user", "content": "Say hello"}],
)

print(response.choices[0].message.content)
print(response.choices[0].finish_reason)  # "stop" or "length"
```

### JavaScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: "not-needed",
});
```

## API surface

### OpenAI-compatible

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/models` | Lists built-in registry models plus scanned local models |
| POST | `/v1/chat/completions` | OpenAI-style chat; streaming uses SSE |
| POST | `/v1/embeddings` | OpenAI-style embeddings |

OpenAI chat responses set `choices[].finish_reason` to `"stop"` or `"length"`. Streaming responses set `finish_reason` on the final delta chunk before `data: [DONE]`.

### Ollama-compatible

| Method | Path | Notes |
|---|---|---|
| GET | `/api/tags` | Lists locally available models in Ollama format |
| POST | `/api/generate` | Non-streaming JSON or streaming NDJSON chunks |
| POST | `/api/chat` | Non-streaming JSON or streaming NDJSON chunks |
| POST | `/api/embed` | Current Ollama embeddings format |
| POST | `/api/embeddings` | Legacy Ollama embeddings format |
| GET | `/api/ps` | Running-model view |
| POST | `/api/show` | Registry-backed model metadata, parameters, template, and modelfile |
| GET | `/api/version` | Returns the NPU Proxy Ollama-compatible version string |
| POST | `/api/pull` | Downloads model files from Hugging Face |
| GET | `/api/search` | NPU Proxy extension for OpenVINO model search |
| GET | `/api/models/known` | NPU Proxy extension listing short-name mappings |

Ollama generate/chat final responses and final NDJSON frames include `done: true` and `done_reason` (`"stop"` or `"length"`).

### System

| Method | Path |
|---|---|
| GET | `/health` |
| GET | `/health/liveness` |
| GET | `/health/readiness` |
| GET | `/health/devices` |
| GET | `/metrics` |

## Model management

### Search

```bash
curl "http://127.0.0.1:8080/api/search?q=llama&sort=popular"
curl "http://127.0.0.1:8080/api/search?type=llm&quantization=int4"
```

`/api/search` accepts:

- `q`
- `sort` = `popular|newest|downloads|likes`
- `limit`
- `offset`
- `type` = `all|llm|embedding|vision`
- `quantization`
- `min_downloads`

The `vision` filter exists on the search endpoint, but this repository does **not** document vision serving support.

### Pull

```bash
curl -X POST http://127.0.0.1:8080/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "tinyllama", "stream": false}'
```

Known short-name mappings currently include:

- LLMs: `tinyllama`, `tinyllama:fp16`, `phi-2`, `phi-3`, `llama2`, `llama2:13b`, `llama3.2`, `mistral`, `qwen2`, `gemma`
- Embeddings: `bge-small`, `bge-base`, `bge-large`, `e5-small`, `e5-large`, `all-minilm`, `nomic-embed-text`

Model downloads enforce Hugging Face allow-patterns and size limits; full-snapshot download is opt-in. Conversion work runs in a timeout-safe child process and publishes with an atomic directory rename so partially written model directories are not exposed as complete.

## Embeddings

Current embedding behavior is request-aware:

- the service caches embedding engines by resolved model and device
- `NPU_PROXY_EMBEDDING_MODEL` and `NPU_PROXY_EMBEDDING_DEVICE` provide the default embedding model and device
- request `model` fields plus device overrides can select a different cached engine when a runtime-ready export exists
- if the requested runtime-ready model is missing or unusable, the request fails by default instead of returning success-shaped fallback embeddings
- set `NPU_PROXY_EMBEDDING_FALLBACK_MODE=hash` only when you intentionally want explicit hash fallback for operator testing or wiring checks

Embedding inputs are validated before engine execution:

- empty input list: HTTP 400, code `empty_input`
- more than 128 inputs: HTTP 413, code `embedding_batch_too_large`
- whitespace-only or empty text: HTTP 400, code `empty_input`
- oversized text: HTTP 413, code `embedding_input_too_large`
- successful batches return exactly one finite, correctly dimensioned vector per input

Current workstation truth:

- the documented default embedding device remains `CPU`
- `all-minilm` / `sentence-transformers/all-MiniLM-L6-v2` validated on `NPU` here via a static-shape profile
- `bge-small` on `NPU` did not validate here because the Intel NPU plugin failed with `check_sdpa_nodes(model)`
- download or registry support for an embedding model is **not** the same as validated NPU execution support

Recommended validated NPU setup:

```powershell
python scripts\download_model.py download all-minilm
$env:NPU_PROXY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
$env:NPU_PROXY_EMBEDDING_DEVICE = "NPU"
```

Then restart the server so it picks up the new embedding defaults.

## Security model

NPU Proxy is intentionally a local, single-user tool:

- it binds to `127.0.0.1` by default
- it has **no authentication or API key enforcement** by design
- do not expose it as a shared production proxy

Current hardening:

- Host-header allow-list middleware rejects disallowed `Host` headers with HTTP `421` and body `Host header not allowed`, mitigating DNS rebinding against the local service
- the default allow-list is `localhost`, `127.0.0.1`, `::1`, `[::1]`, `testserver`, and `test`
- configure the allow-list with `NPU_PROXY_ALLOWED_HOSTS` or `--allowed-hosts` (comma-separated)
- binding to a non-loopback address logs a warning; `scripts\start-server.ps1` requires explicit opt-in (`-ListenAll` or `NPU_PROXY_LISTEN_ALL=true`) before choosing `0.0.0.0`
- registry model names are slug-validated, resolved model paths are confined to the model directory, and tokenizer paths are guarded against traversal
- client-facing errors are sanitized so internal exception details and stack traces are not leaked

## Runtime extras that exist today

- OpenVINO remains the default LLM backend.
- OpenAI and Ollama chat prompt rendering attempts tokenizer chat templates when available and falls back to legacy formatting; set `NPU_PROXY_DISABLE_CHAT_TEMPLATES=1` to force legacy formatting.
- Compile cache controls exist through `NPU_PROXY_COMPILE_CACHE_DIR`, `NPU_PROXY_COMPILE_CACHE_MODE`, and `NPU_PROXY_PREFIX_CACHE_MODE`.
- An alpha `llama.cpp` GGUF path exists behind `NPU_PROXY_LLM_BACKEND=llama_cpp` and `NPU_PROXY_ENABLE_ALPHA_BACKENDS=1`, but it is currently CPU-only, feature-gated, and source-install only because `llama-cpp-python` is not part of the default packaged dependencies.

See:

- [docs/api/EMBEDDINGS.md](docs/api/EMBEDDINGS.md)
- [docs/guides/MODEL_DOWNLOAD.md](docs/guides/MODEL_DOWNLOAD.md)

## Configuration

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port`, `-p` | `8080` | Bind port |
| `--workers`, `-w` | `1` | Uvicorn worker processes |
| `--reload` | off | Enable development auto-reload |
| `--allowed-hosts` | loopback/test clients | Comma-separated Host-header allow-list |
| `--device`, `-d` | `AUTO` CLI parser; effective runtime default `NPU` | `NPU`, `GPU`, `CPU`, or `AUTO`; `AUTO` leaves the LLM runtime default in place |
| `--token-limit` | `1800` | Advisory context-routing threshold |
| `--compile-cache-dir` | unset | Optional OpenVINO compile cache directory |
| `--compile-cache-mode` | runtime default | `OPTIMIZE_SIZE` or `OPTIMIZE_SPEED` |
| `--prefix-cache-mode` | `auto` | `auto`, `on`, or `off` |
| `--real-inference` | off | Enable real model inference; otherwise mock mode |
| `--log-level`, `-l` | `info` | `debug`, `info`, `warning`, `error`, or `critical` |
| `--log-file` | unset | Log to file instead of stdout |
| `--version`, `-v` | n/a | Print version and exit |

### Bootstrap and LLM environment variables

| Variable | Default | Notes |
|---|---|---|
| `NPU_PROXY_HOST` | `127.0.0.1` | Bind address |
| `NPU_PROXY_PORT` | `8080` | Bind port |
| `NPU_PROXY_DEVICE` | `NPU` effective runtime default | LLM device; CLI also accepts `AUTO` |
| `NPU_PROXY_TOKEN_LIMIT` | `1800` | Advisory context-routing threshold |
| `NPU_PROXY_REAL_INFERENCE` | `0` | Set `1` to enable real inference |
| `NPU_PROXY_INFERENCE_TIMEOUT` | `180` | LLM inference timeout seconds |
| `NPU_PROXY_MAX_PROMPT_LEN` | `4096` | LLM prompt limit |
| `NPU_PROXY_COMPILE_CACHE_DIR` | unset | Optional OpenVINO compile cache directory |
| `NPU_PROXY_COMPILE_CACHE_MODE` | runtime default | `OPTIMIZE_SIZE` or `OPTIMIZE_SPEED` |
| `NPU_PROXY_PREFIX_CACHE_MODE` | `auto` | Prefix cache mode |
| `NPU_PROXY_LLM_BACKEND` | `openvino` | `openvino` or alpha `llama_cpp` |
| `NPU_PROXY_ENABLE_ALPHA_BACKENDS` | `0` | Required for alpha backends |
| `NPU_PROXY_LLAMACPP_MODEL_PATH` | unset | Local `.gguf` path for alpha `llama.cpp` backend |
| `NPU_PROXY_ALLOWED_HOSTS` | loopback/test clients | Comma-separated Host-header allow-list |
| `NPU_PROXY_PREFERRED_DEVICE` | `NPU` | Advisory context-router preference |
| `NPU_PROXY_FALLBACK_DEVICE` | auto | Advisory context-router fallback override |

Invalid `TOKEN_LIMIT`, `PREFERRED_DEVICE`, and `FALLBACK_DEVICE` values warn and fall back to defaults in request-time routing paths; explicit startup validation can still fail for invalid bootstrap settings.

### Other runtime environment variables

| Variable | Default | Notes |
|---|---|---|
| `NPU_PROXY_MODEL_DIR` | `~/.cache/npu-proxy/models` | Model/tokenizer lookup root |
| `NPU_PROXY_DISABLE_CHAT_TEMPLATES` | unset | Truthy value forces legacy chat formatting |
| `NPU_PROXY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Default embedding model |
| `NPU_PROXY_EMBEDDING_DEVICE` | `CPU` | Default embedding device |
| `NPU_PROXY_LOAD_TIMEOUT` | `300` | Embedding model load timeout seconds |
| `NPU_PROXY_EMBED_TIMEOUT` | `60` | Embedding inference timeout seconds |
| `NPU_PROXY_EMBEDDING_CACHE_SIZE` | `1024` | Embedding cache size |
| `NPU_PROXY_EMBEDDING_FALLBACK_MODE` | disabled | Set `hash` only for explicit operator fallback tests |
| `NPU_PROXY_EMBEDDING_UNAVAILABLE_COOLDOWN` | `30` | Cooldown seconds after embedding load failures |
| `NPU_PROXY_LISTEN_ALL` | unset | `scripts\start-server.ps1` opt-in for `0.0.0.0` |

## Benchmarking

The benchmark CLI currently documents one end-to-end workflow:

```powershell
python scripts\benchmark.py run --model tinyllama --device NPU --iterations 5 --warmup 1 --output results.json
```

See [docs/guides/BENCHMARKS.md](docs/guides/BENCHMARKS.md) for the current CLI surface.

## Development

### Tests

```powershell
python -m pytest -m "not slow and not e2e"
```

Fast tests do not require real NPU/model hardware. Slow/e2e tests do.

### Dev server

```powershell
uvicorn npu_proxy.main:app --reload --host 127.0.0.1 --port 8080
uvicorn npu_proxy.main:app --reload --host 127.0.0.1 --port 8080 --log-level debug
```

## Limitations

- Real LLM inference needs a local OpenVINO model directory.
- Mock mode is the default unless real inference is explicitly enabled.
- OpenAI chat streaming and Ollama streaming use different wire formats.
- Routing is advisory only in this release; per-request LLM device switching is not implemented.
- Validated NPU embedding support is currently limited to the static-shape `all-minilm` path; `bge-small` still failed workstation validation on NPU.
- The alpha GGUF backend is intentionally not documented as a packaged/default runtime path.
- NPU Proxy is documented for native host deployment; this repo does not document containerized NPU serving.

## More docs

- [SPEC.md](SPEC.md)
- [CHANGELOG.md](CHANGELOG.md)
- [docs/README.md](docs/README.md)
