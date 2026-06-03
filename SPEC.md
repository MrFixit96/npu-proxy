# NPU Proxy Specification

This file is a **released-truth snapshot** for the current repository state. It intentionally avoids planned or speculative features.

## Objective

Expose local OpenVINO-backed inference through:

- **OpenAI-style endpoints** for chat, embeddings, and model listing
- **Ollama-style endpoints** for generate, chat, embeddings, model download, tags, and model discovery

The primary documented deployment model is a native, local, single-user developer-workstation process. It binds to loopback by default and has no authentication by design. It is not specified as a shared production proxy.

## Validation snapshot behind this release-truth pass

- Mock mode is the default unless real inference is explicitly enabled.
- The authoritative default bind is `127.0.0.1:8080` for the CLI/config layer.
- Host-header allow-listing rejects disallowed Host headers with HTTP 421.
- `/api/tags` is implemented and `/api/show` returns real registry-backed metadata.
- Generation responses expose stop/length finish reasons.
- Embedding validation rejects invalid inputs before engine execution and limits batches to 128 inputs.
- `sentence-transformers/all-MiniLM-L6-v2` validated on NPU on the validation workstation via a static-shape embedding profile.
- `BAAI/bge-small-en-v1.5` on NPU did **not** validate on that workstation because the Intel NPU plugin failed with `check_sdpa_nodes(model)`.

## Current top-level architecture

```text
Client (Ollama/OpenAI SDK/Claude Code)
        |
        v
FastAPI app (npu_proxy.main:app)
        |
        +-- OpenAI routes: /v1/models, /v1/chat/completions, /v1/embeddings
        +-- Ollama routes: /api/tags, /api/generate, /api/chat, /api/embed, /api/pull, ...
        +-- Health/metrics routes: /health, /health/devices, /metrics
        |
        v
OpenVINO-backed engines, or mock responses when real inference is disabled
```

## Implemented HTTP surface

### OpenAI-compatible routes

| Method | Path | Current behavior |
|---|---|---|
| GET | `/v1/models` | Returns built-in registry models plus scanned local models |
| POST | `/v1/chat/completions` | OpenAI chat format; streaming uses SSE |
| POST | `/v1/embeddings` | OpenAI embeddings format |

`/v1/chat/completions` returns `choices[].finish_reason` in non-streaming responses and on the final streaming delta chunk. Values are `"stop"` for natural completion and `"length"` when the effective max output token limit is reached.

### Ollama-compatible routes

| Method | Path | Current behavior |
|---|---|---|
| GET | `/api/tags` | Lists locally available models in Ollama tags format |
| POST | `/api/generate` | Non-streaming JSON or streaming NDJSON chunks |
| POST | `/api/chat` | Non-streaming JSON or streaming NDJSON chunks |
| POST | `/api/embed` | Current Ollama embedding format |
| POST | `/api/embeddings` | Legacy single-embedding format |
| GET | `/api/ps` | Running-model status |
| POST | `/api/show` | Registry-backed model metadata, parameters, template, and modelfile |
| GET | `/api/version` | Returns the NPU Proxy Ollama-compatible version string |
| POST | `/api/pull` | Downloads OpenVINO model files from Hugging Face |
| GET | `/api/search` | Search/filter OpenVINO-compatible Hugging Face models |
| GET | `/api/models/known` | Lists static short-name mappings |

Ollama generate/chat final responses and final NDJSON frames include `done: true` and `done_reason` (`"stop"` or `"length"`).

### System routes

| Method | Path | Current behavior |
|---|---|---|
| GET | `/health` | Observational service summary with engine/device details |
| GET | `/health/liveness` | Cheap process-up probe |
| GET | `/health/readiness` | Observational warmed-runtime readiness gate |
| GET | `/health/devices` | Returns available devices, active device, fallback chain |
| GET | `/metrics` | Prometheus text format when metrics support is available |

## Startup and configuration

### Entry points

- `npu-proxy` → `npu_proxy.cli:main`
- `python -m uvicorn npu_proxy.main:app --host 127.0.0.1 --port 8080`
- `.\scripts\start-server.ps1`

### Current startup defaults

| Path | Host | Port | Notes |
|---|---|---|---|
| `npu-proxy` CLI/config | `127.0.0.1` | `8080` | Default local developer bind |
| `scripts\start-server.ps1` | `127.0.0.1` | `11435` | Launcher default; `0.0.0.0` requires `-ListenAll` or `NPU_PROXY_LISTEN_ALL=true` |

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port`, `-p` | `8080` | Bind port |
| `--workers`, `-w` | `1` | Uvicorn worker processes |
| `--reload` | off | Development auto-reload |
| `--allowed-hosts` | loopback/test clients | Comma-separated Host-header allow-list |
| `--device`, `-d` | `AUTO` parser default; effective runtime default `NPU` | `NPU`, `GPU`, `CPU`, or `AUTO`; routing remains advisory |
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
| `NPU_PROXY_INFERENCE_TIMEOUT` | `180` | LLM timeout seconds |
| `NPU_PROXY_MAX_PROMPT_LEN` | `4096` | LLM prompt limit |
| `NPU_PROXY_COMPILE_CACHE_DIR` | unset | Optional OpenVINO compile cache directory |
| `NPU_PROXY_COMPILE_CACHE_MODE` | runtime default | `OPTIMIZE_SIZE` or `OPTIMIZE_SPEED` |
| `NPU_PROXY_PREFIX_CACHE_MODE` | `auto` | Prefix cache mode |
| `NPU_PROXY_LLM_BACKEND` | `openvino` | `openvino` or alpha `llama_cpp` |
| `NPU_PROXY_ENABLE_ALPHA_BACKENDS` | `0` | Required to opt into alpha backends |
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
| `NPU_PROXY_LOAD_TIMEOUT` | `300` | Embedding load timeout seconds |
| `NPU_PROXY_EMBED_TIMEOUT` | `60` | Embedding inference timeout seconds |
| `NPU_PROXY_EMBEDDING_CACHE_SIZE` | `1024` | Embedding cache size |
| `NPU_PROXY_EMBEDDING_FALLBACK_MODE` | disabled | Set `hash` only for explicit operator fallback tests |
| `NPU_PROXY_EMBEDDING_UNAVAILABLE_COOLDOWN` | `30` | Cooldown seconds after embedding load failures |
| `NPU_PROXY_LISTEN_ALL` | unset | `scripts\start-server.ps1` opt-in for `0.0.0.0` |

## Security model

NPU Proxy is intentionally local-first:

- default host is `127.0.0.1`
- there is no authentication or API-key enforcement by design
- it should not be exposed as a shared production proxy

Implemented hardening:

- `npu_proxy.main` enforces a Host-header allow-list for every request
- disallowed Host headers receive HTTP `421` with body `Host header not allowed`
- the default allow-list covers loopback and test clients: `localhost`, `127.0.0.1`, `::1`, `[::1]`, `testserver`, `test`
- configure the allow-list with `NPU_PROXY_ALLOWED_HOSTS` or `--allowed-hosts`
- binding a non-loopback address logs a warning that remote clients must be represented in the allow-list
- `scripts\start-server.ps1` only chooses `0.0.0.0` with explicit `-ListenAll` or `NPU_PROXY_LISTEN_ALL=true`
- model registry names are slug-validated, resolved paths are confined to the configured model directory, and tokenizer paths are guarded against traversal
- API errors are sanitized so clients do not receive internal exception details or stack traces

## Runtime behavior

### Chat and generate

- `/v1/chat/completions` supports OpenAI-style non-streaming and SSE streaming responses
- `/v1/chat/completions` attempts tokenizer chat-template rendering when a model tokenizer exposes one, and falls back to the legacy role-prefixed formatter otherwise
- `/api/generate` and `/api/chat` return non-streaming JSON or stream newline-delimited JSON chunks
- `/api/chat` uses the same shared chat-template rendering path as `/v1/chat/completions`, with legacy formatting as the fallback when tokenizer templates are unavailable
- real streaming is backed by `npu_proxy.inference.streaming.AsyncTokenStream`
- cancellation is cooperative at token boundaries; a native OpenVINO call already in flight may finish its current token
- `top_p` and timeout settings are forwarded into engine generation paths
- mock mode yields canned responses unless real inference is enabled

### Finish reasons

- OpenAI non-streaming returns `choices[0].finish_reason`
- OpenAI streaming sets `finish_reason` on the final delta chunk
- Ollama final responses/final NDJSON frames include `done_reason` with `done: true`
- values are `"stop"` or `"length"`; native backend reasons are preferred when available, otherwise the service compares emitted completion tokens to the effective max output token limit

### Routing

`npu_proxy.routing.context_router.ContextRouter` selects:

- preferred device when token counts stay within `NPU_PROXY_TOKEN_LIMIT`
- fallback device when they exceed the limit

Default fallback chain reported by `/health/devices` is:

```text
NPU -> GPU -> CPU
```

Current released-truth note:

- the router computes a preferred or fallback device classification
- the real LLM execution path still uses the active runtime/engine state rather than guaranteed per-request device switching
- therefore current routing information should be interpreted as **advisory classification**, not proof of independently switched execution per request
- `devices.get_available_devices()` returns `['CPU']` in mock mode and degrades to `['CPU']` with a warning if OpenVINO import/Core initialization fails in real-inference mode

### LLM engine lifecycle

- `get_llm_engine()` is a singleton
- default model path is `~/.cache/npu-proxy/models/tinyllama-1.1b-chat-int4-ov`
- if that directory is missing in real inference mode, requests fail rather than auto-download
- default LLM device is `NPU`
- default LLM timeout is 180 seconds
- compile-cache and prefix-cache controls are implemented and surfaced through CLI/env configuration and health endpoints
- the repository includes an alpha `llama.cpp` backend seam for local `.gguf` files, but it is feature-gated, CPU-only, and source-install only because `llama-cpp-python` is not in the default project dependencies

### Embedding engine lifecycle and validation

Current truth that matters for users:

- the service caches embedding engines by resolved model and device
- `NPU_PROXY_EMBEDDING_MODEL` and `NPU_PROXY_EMBEDDING_DEVICE` define the default embedding model and device
- request `model` fields and device overrides can switch to another cached engine when a runtime-ready export exists
- unavailable real embeddings fail by default instead of returning success-shaped fallback vectors
- the documented default embedding device remains `CPU`
- `all-minilm-l6-v2` is validated on NPU here via a static-shape profile
- `bge-small` on NPU still failed inside the Intel NPU plugin with `check_sdpa_nodes(model)`

Embedding inputs are validated before hitting the engine:

| Condition | HTTP status | Code | Message |
|---|---:|---|---|
| Empty input list | 400 | `empty_input` | `Embedding input must contain at least one item` |
| Batch size greater than 128 | 413 | `embedding_batch_too_large` | `Embedding input batch is too large` |
| Empty or whitespace-only text | 400 | `empty_input` | `Embedding input text must not be empty` |
| Oversized text | 413 | `embedding_input_too_large` | `Embedding input text is too large` |

Successful embedding responses carry an `x-request-id` header and return exactly one finite, correctly dimensioned vector per input. OpenAI-compatible errors use the OpenAI error envelope; Ollama-compatible embedding routes use their own detail envelope.

## Model sources

### Built-in model registry

The current built-in registry includes:

- LLM IDs: `tinyllama-1.1b-chat-int4-ov`, `mistral-7b-int4-ov`, `granite-4-micro-ov`, `phi-2-int4-ov`
- Embedding IDs: `all-minilm-l6-v2`, `bge-small`, `e5-large`, `qwen3-embedding-0.6b-int4-ov`, `qwen3-embedding-8b-int4-ov`

`/v1/models` combines those entries with any additional locally scanned model directories. `/api/tags` exposes locally available models in Ollama's tags shape. `/api/show` returns metadata from the registry/local model scan rather than placeholder-only data.

### Short-name pull mappings

`/api/pull` and `/api/models/known` currently support static mappings such as:

- `tinyllama`, `tinyllama:fp16`, `phi-2`, `phi-3`, `llama2`, `llama2:13b`, `llama3.2`, `mistral`, `qwen2`, `gemma`
- `bge-small`, `bge-base`, `bge-large`, `e5-small`, `e5-large`, `all-minilm`, `nomic-embed-text`

Downloads enforce Hugging Face allow-patterns and a size cap; full-snapshot download is opt-in. Conversion runs in a timeout-safe child process and publishes via atomic temporary-directory rename.

## Packaging artifacts in this repo

### Windows

- `scripts\build_windows.ps1` builds `dist\npu-proxy.exe`
- `npu_proxy.pyinstaller.spec` defines the PyInstaller build
- `packaging\winget\*.yaml` contains WinGet manifest files for version `0.2.0`
- those packaging artifacts target the default OpenVINO runtime path, not the alpha GGUF experiment path

### Linux / Debian

- `packaging\npu-proxy.service` is the systemd unit
- `packaging\npu-proxy.environment` is the packaged environment template
- `packaging\debian\` contains Debian packaging files
- `packaging\debian\build.sh` copies resulting `.deb` files into `dist\`
- the packaged path documents the default OpenVINO runtime; the alpha GGUF backend is not packaged as a first-class install target

## Health and status semantics

### `/health`

Current top-level response includes:

- `status`
- `engines`
- `version`
- `npu_available`
- `gpu_available`
- `cpu_available`
- `devices`
- `openvino_version`
- `messages`

Current contract:

- `/health` is **observational only**
- it does **not** auto-load LLM or embedding models
- it reports not-loaded state explicitly in engine messages

### `/health/liveness`

Current response includes:

- `status = "alive"`
- `alive = true`
- `version`

### `/health/readiness`

Current behavior:

- returns readiness for warmed runtime state
- is observational only
- returns explicit reasons when the configured LLM or embedding model is not loaded
- uses HTTP `503` when the service is alive but not ready to serve warmed traffic

### `/health/devices`

Current response includes:

- `available_devices`
- `active_device`
- `device_info`
- `fallback_chain`

## Known operational constraints

- Native host deployment is the documented path
- Real inference requires local model directories to exist first
- Mock mode is the default
- OpenAI streaming and Ollama streaming use different wire formats
- Routing is advisory only in this release; per-request LLM device switching is not implemented
- The repository contains packaging assets and manifests, but this spec only claims what exists in-tree

## Related docs

- [README.md](README.md)
- [docs/api/OLLAMA_API.md](docs/api/OLLAMA_API.md)
- [docs/api/EMBEDDINGS.md](docs/api/EMBEDDINGS.md)
- [docs/api/STREAMING.md](docs/api/STREAMING.md)
