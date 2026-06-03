# Ollama API Compatibility

This file documents the Ollama-compatible surface that exists today.

## Supported endpoints

| Method | Path | Current behavior |
|---|---|---|
| POST | `/api/generate` | Ollama-style generate; streaming returns NDJSON chunks |
| POST | `/api/chat` | Ollama-style chat; streaming returns NDJSON chunks |
| POST | `/api/embed` | Current Ollama embedding format |
| POST | `/api/embeddings` | Legacy embedding format |
| GET | `/api/tags` | Lists locally available models in Ollama format |
| GET | `/api/ps` | Running models |
| POST | `/api/show` | Real registry-backed model metadata |
| GET | `/api/version` | Returns `0.2.0-npu-proxy` |
| POST | `/api/pull` | Download model files from Hugging Face |
| GET | `/api/search` | NPU Proxy extension |
| GET | `/api/models/known` | NPU Proxy extension |

Not implemented in this repository:

- `DELETE /api/delete`
- `POST /api/copy`
- `POST /api/create`

## Validation truth for this release pass

- The successful live NPU certification run in this repository exercised `POST /api/generate`.
- `/api/chat` remains implemented and now uses the shared chat-template rendering path, falling back to the legacy role-prefixed formatter only when tokenizer templates are unavailable or disabled.
- `GET /api/tags` returns locally available registry/scanned models with `models[].name`, `model`, `modified_at`, `size`, `digest`, and `details`.
- `POST /api/show` now returns real model metadata: `modelfile`, `parameters`, `template`, `details`, and `model_info` from the registry/tokenizer path.
- The Ollama embedding endpoints are implemented and `all-minilm` validated on NPU here via a static-shape profile, but `bge-small` on NPU still failed workstation validation when the Intel NPU plugin raised `check_sdpa_nodes(model)`.

## Streaming format

`/api/generate`, `/api/chat`, and `/api/pull` stream newline-delimited JSON chunks.

Current implementation detail:

- media type: `application/x-ndjson`
- the final success frame for `/api/generate` and `/api/chat` includes `done: true` and `done_reason`
- `done_reason` is either `stop` or `length`; `length` means generation reached the effective max output token limit
- stream failures emit a terminal NDJSON error object and stop without a final `done: true` success frame

This is different from the OpenAI chat endpoint, which uses SSE.

## Routing semantics

`/api/generate` and `/api/chat` currently expose **advisory routing semantics** that are narrower than multi-engine execution.

- the router can classify a request toward a preferred or fallback device
- the current runtime still behaves as a **single active engine** for real execution
- `X-NPU-Proxy-Device` reports the actual configured/loaded singleton runtime device
- `X-NPU-Proxy-Route-Reason` is currently `single_engine_runtime`, while `X-NPU-Proxy-Token-Count` still exposes the advisory token-count classification input

## Parameter handling

### Defaults currently applied

The server merges incoming Ollama options with these current defaults:

| Parameter | Default |
|---|---|
| `temperature` | `0.8` |
| `top_k` | `40` |
| `top_p` | `0.9` |
| `repeat_penalty` | `1.1` |
| `num_predict` | `128` |
| `num_ctx` | `2048` |
| `num_batch` | `512` |
| `seed` | `0` |
| `stop` | `[]` |
| `mirostat` | `0` |
| `mirostat_tau` | `5.0` |
| `mirostat_eta` | `0.1` |
| `min_p` | `0.0` |
| `typical_p` | `1.0` |
| `tfs_z` | `1.0` |

### Direct or renamed mappings

| Incoming parameter | Current handling |
|---|---|
| `temperature` | passed through |
| `top_k` | passed through |
| `top_p` | passed through |
| `seed` | passed through |
| `repeat_penalty` | renamed to `repetition_penalty` |
| `num_predict` | renamed to `max_new_tokens` |
| `stop` | renamed to `stop_strings` |

### Approximate mappings

`presence_penalty` and `frequency_penalty` are combined into:

```text
repetition_penalty = 1.0 + (presence_penalty + frequency_penalty) / 2
```

If `repeat_penalty` is explicitly provided, it wins.

### Accepted but ignored

Logged at debug level and dropped:

- `mirostat`
- `mirostat_tau`
- `mirostat_eta`
- `min_p`
- `typical_p`
- `tfs_z`

Silently ignored:

- `num_ctx`
- `num_batch`

Unknown parameters are warned about and ignored.

## Response and validation notes

### Generate/chat completion frames

Non-streaming `POST /api/generate` returns a single object with `done: true` and `done_reason`:

```json
{
  "model": "tinyllama-1.1b-chat-int4-ov",
  "created_at": "2025-01-01T00:00:00Z",
  "response": "Hello!",
  "done": true,
  "total_duration": 1000000,
  "eval_count": 1,
  "done_reason": "stop"
}
```

Non-streaming `POST /api/chat` has the same final fields, with generated content under `message`:

```json
{
  "model": "tinyllama-1.1b-chat-int4-ov",
  "created_at": "2025-01-01T00:00:00Z",
  "message": {"role": "assistant", "content": "Hello!"},
  "done": true,
  "total_duration": 1000000,
  "eval_count": 1,
  "done_reason": "stop"
}
```

For streaming, `done_reason` appears on the final NDJSON success frame alongside `done: true`.

### Embedding validation errors

`POST /api/embed` and `POST /api/embeddings` validate input before loading the engine:

| Condition | HTTP status | `code` | Message |
|---|---:|---|---|
| Empty input list | 400 | `empty_input` | `Embedding input must contain at least one item` |
| Batch larger than 128 | 413 | `embedding_batch_too_large` | `Embedding input batch is too large` |
| Empty or whitespace-only text | 400 | `empty_input` | `Embedding input text must not be empty` |
| Single text over 8192 characters | 413 | `embedding_input_too_large` | `Embedding input text is too large` |
| Request total over 65536 characters | 413 | `embedding_request_too_large` | `Embedding request is too large` |

Responses include `X-Request-ID`. Ollama error responses use the flat envelope returned by the implementation:

```json
{
  "error": "Embedding input text must not be empty (request id: req-abc123)",
  "code": "empty_input"
}
```

## Model naming

### Generate/chat/show/model listing

These paths work with built-in registry IDs and scanned local model IDs, such as:

- `tinyllama-1.1b-chat-int4-ov`
- `phi-2-int4-ov`
- `mistral-7b-int4-ov`

### Pull and known-model discovery

`/api/pull` and `/api/models/known` understand short aliases such as:

- `tinyllama`
- `phi-3`
- `llama3.2`
- `mistral`
- `bge-small`
- `all-minilm`

Alias resolution and download support do **not** imply that a model path is validated on Intel NPU hardware.
The currently validated NPU embedding alias on this workstation is `all-minilm`.

### Pull authentication

`POST /api/pull` now supports both public and private Hugging Face repos:

- **public repos**: no token required
- **private repos**: require an explicit Hugging Face token

The API does **not** silently inherit ambient server-side Hugging Face credentials for user-selected pulls.
Private pulls must provide a token in exactly one of these places:

- request body field: `huggingface_token`
- `Authorization: Bearer <token>` header

If a cached model was originally pulled with explicit private auth, anonymous callers do not get a cached success response for that repo.

## Client examples

### Ollama CLI

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:8080"
ollama pull tinyllama
ollama run tinyllama "Hello"
ollama ps
```

### Python `ollama` client

```python
import ollama

client = ollama.Client(host="http://127.0.0.1:8080")
response = client.chat(
    model="tinyllama",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### cURL

```bash
curl http://127.0.0.1:8080/api/chat -d '{
  "model": "tinyllama-1.1b-chat-int4-ov",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

### Private model pull

```bash
curl -X POST http://127.0.0.1:8080/api/pull \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer hf_xxx" \
  -d '{"name": "your-org/private-model", "stream": false}'
```

## Logging

Use the CLI log-level flag for debugging:

```powershell
npu-proxy --host 127.0.0.1 --port 8080 --log-level debug
```

## NPU Proxy extensions

These routes are repository-specific extensions, not standard Ollama API endpoints:

- `/api/search`
- `/api/models/known`

`/api/search` supports:

- `q`
- `sort=popular|newest|downloads|likes`
- `limit`
- `offset`
- `type=all|llm|embedding|vision`
- `quantization`
- `min_downloads`

The `vision` filter is part of the search API surface only; this document does not claim vision serving support.
