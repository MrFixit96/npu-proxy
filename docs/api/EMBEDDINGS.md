# Embeddings

This document covers the embedding behavior that exists in the current codebase.

## Endpoints

| Method | Path | Format |
|---|---|---|
| POST | `/v1/embeddings` | OpenAI-compatible |
| POST | `/api/embed` | Current Ollama embedding format |
| POST | `/api/embeddings` | Legacy Ollama embedding format |

## Current runtime model behavior

Today that means:

- the server caches embedding engines by resolved model and device
- `NPU_PROXY_EMBEDDING_MODEL` and `NPU_PROXY_EMBEDDING_DEVICE` provide the default model and device
- request `model` fields and device overrides can select another cached engine when a runtime-ready export exists
- if the requested model is missing or unusable in the runtime cache path, the request fails by default instead of returning success-shaped fallback embeddings

If you want a different real embedding model, either change the process defaults or provide a request model/device pair that already has a matching exported model on disk.

## Validation truth for this release pass

- The documented default embedding device remains `CPU`.
- On the validation workstation, `sentence-transformers/all-MiniLM-L6-v2` validated on `NPU` via a static-shape profile (`batch_size=1`, `max_length=256`, `pad_to_max_length=true`).
- On the validation workstation, `BAAI/bge-small-en-v1.5` on `NPU` did **not** validate because the Intel NPU plugin failed with `check_sdpa_nodes(model)`.
- If the configured model is missing or unusable, the service now fails the request unless explicit fallback mode is enabled.

## Current configuration defaults

| Variable | Default |
|---|---|
| `NPU_PROXY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `NPU_PROXY_EMBEDDING_DEVICE` | `CPU` |
| `NPU_PROXY_LOAD_TIMEOUT` | `300` |
| `NPU_PROXY_EMBED_TIMEOUT` | `60` |
| `NPU_PROXY_EMBEDDING_CACHE_SIZE` | `1024` |

## Runtime-ready model path

The runtime looks for exported models under:

```text
~/.cache/npu-proxy/models/embeddings/<canonical-id-or-sanitized-repo>
```

Examples:

```text
~/.cache/npu-proxy/models/embeddings/bge-small
~/.cache/npu-proxy/models/embeddings/all-minilm-l6-v2
```

Known registry-backed models use canonical runtime IDs such as `bge-small` and `all-minilm-l6-v2`. Sanitized repo-name paths such as `BAAI_bge-small-en-v1.5` are legacy compatibility paths.

See [../guides/MODEL_DOWNLOAD.md](../guides/MODEL_DOWNLOAD.md) for the current setup flow.

## Built-in registry IDs

Examples of built-in embedding registry IDs include:

- `all-minilm-l6-v2`
- `bge-small`
- `bge-base`
- `bge-large`
- `e5-small`
- `e5-large`
- `nomic-embed-text`
- `qwen3-embedding-0.6b-int4-ov`
- `qwen3-embedding-8b-int4-ov`

The pull/download helpers know additional short aliases such as `bge-base`, `bge-large`, `e5-small`, and `nomic-embed-text`.

## Input validation

`POST /v1/embeddings` accepts a single string or a list of strings in `input`. It validates input before loading the engine. `MAX_EMBEDDING_BATCH_SIZE` is `128`.

| Condition | HTTP status | `error.type` | `error.code` | Message |
|---|---:|---|---|---|
| Empty input list | 400 | `invalid_request_error` | `empty_input` | `Embedding input must contain at least one item` |
| Batch larger than 128 | 413 | `request_too_large` | `embedding_batch_too_large` | `Embedding input batch is too large` |
| Empty or whitespace-only text | 400 | `invalid_request_error` | `empty_input` | `Embedding input text must not be empty` |
| Single text over 8192 characters | 413 | `request_too_large` | `embedding_input_too_large` | `Embedding input text is too large` |
| Request total over 65536 characters | 413 | `request_too_large` | `embedding_request_too_large` | `Embedding request is too large` |

OpenAI errors use this envelope:

```json
{
  "error": {
    "message": "Embedding input text must not be empty",
    "type": "invalid_request_error",
    "param": null,
    "code": "empty_input"
  }
}
```

The response includes `X-Request-ID`; a provided `X-Request-ID` request header is echoed after sanitization, otherwise the server generates one.

## Result guarantee

For every successful embedding request, the response contains exactly one finite, correctly-dimensioned vector per input, in input order. The service rejects batch results that do not align 1:1 with the input list; it does not return zero/blank placeholder vectors as successful batch items.

## OpenAI example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

response = client.embeddings.create(
    model="all-minilm",
    input=["hello", "world"],
)

for item in response.data:
    print(item.index, len(item.embedding))
```

## Ollama examples

### `/api/embed`

```bash
curl -X POST http://127.0.0.1:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"all-minilm","input":["text1","text2"]}'
```

Response shape:

```json
{
  "model": "all-minilm",
  "embeddings": [[0.1, -0.2], [0.3, -0.4]],
  "total_duration": 14143917,
  "load_duration": 0,
  "prompt_eval_count": 8
}
```

### `/api/embeddings`

```bash
curl -X POST http://127.0.0.1:8080/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"all-minilm","prompt":"single text"}'
```

Response shape:

```json
{
  "embedding": [0.1, -0.2]
}
```

## OpenAI response notes

`POST /v1/embeddings` follows the OpenAI request/response shape implemented in `npu_proxy/api/embeddings.py`.

Request:

- `model` is required
- `input` is required and may be a string or list of strings
- `encoding_format` defaults to `float`; other values return `400` with code `unsupported_encoding_format`

Successful response:

- `object: "list"`
- `data[]` entries contain `object: "embedding"`, `embedding: [...]`, and `index`
- `model` echoes the request field
- `usage.prompt_tokens` and `usage.total_tokens` are populated

The `model` field in the response echoes the request field; use runtime health or engine info when you need to confirm which real engine/device handled a request.

## Fallback mode

The default runtime behavior is **no implicit fallback**. If no runtime-ready embedding model is found, the request fails with an embedding error instead of returning success-shaped placeholder vectors.

If you explicitly set `NPU_PROXY_EMBEDDING_FALLBACK_MODE=hash`, the service can return deterministic hash-based embeddings for operator testing or wiring checks. That mode is opt-in and should not be treated as semantic retrieval quality.

## Workstation-specific NPU note

On the validation workstation, `all-minilm-l6-v2` worked on `NPU` through a static-shape profile, while `bge-small` still failed inside the Intel NPU plugin with `check_sdpa_nodes(model)`.
