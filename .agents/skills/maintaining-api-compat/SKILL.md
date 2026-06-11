---
name: maintaining-api-compat
description: >-
  Maintains NPU Proxy's OpenAI- and Ollama-compatible HTTP API surface and its
  streaming behavior. Use when adding or changing endpoints, request/response
  schemas, SSE/streaming output, or cancellation, or when the user mentions
  OpenAI compatibility, Ollama API, /v1/chat/completions, /api/chat, /api/generate,
  /api/tags, streaming, SSE, or wire-format parity.
license: Apache-2.0
compatibility: Requires the npu-proxy repo and its .venv.
metadata:
  author: npu-proxy
  version: "1.0"
---

# Maintaining API Compatibility

NPU Proxy exposes two compatible surfaces — OpenAI (`/v1/...`) and Ollama
(`/api/...`) — over one routing/inference core. The value of the proxy is that
existing OpenAI and Ollama clients work unchanged, so **wire-format parity is the
product**. Changing a schema or stream shape can silently break real clients.

## Endpoint surface

| Surface | Endpoints (router-relative) | File |
| --- | --- | --- |
| OpenAI | `POST /v1/chat/completions` | `npu_proxy/api/chat.py` |
| OpenAI | embeddings, models | `api/embeddings.py`, `api/models.py` |
| Ollama | `/generate`, `/chat`, `/tags`, `/ps`, `/version`, `/show`, `/embed`, `/embeddings`, `/pull`, `/search`, `/models/known` | `npu_proxy/api/ollama.py` |
| Shared | routing headers | `api/header_utils.py` |
| Shared | health | `api/health.py` |

## Invariants

1. **Match the upstream wire format**, not a convenient internal shape. OpenAI
   and Ollama use different field names and different streaming framings — honor
   each independently.
2. **Two streaming formats.** OpenAI chat streams SSE `data:` chunks ending with
   `data: [DONE]`; Ollama streams newline-delimited JSON objects with a final
   object carrying `done: true`. Do not unify them.
3. **Routing headers are part of the contract.** Responses carry the
   `X-NPU-Proxy-*` headers (routed/execution/fallback/route-reason/token-count);
   see the `certifying-device-routing` skill for their exact semantics.
4. **Soft cancel.** Streaming cancellation is cooperative: a client disconnect or
   server shutdown stops generation between tokens rather than hard-killing the
   engine. Preserve this when touching the stream loop.

## Changing the API

```
API change checklist:
- [ ] Confirm the exact upstream (OpenAI/Ollama) schema you must match
- [ ] Update request and response models together
- [ ] Streaming: keep SSE (OpenAI) vs NDJSON (Ollama) framing distinct
- [ ] Preserve X-NPU-Proxy-* headers and soft-cancel behavior
- [ ] Add/adjust tests for BOTH non-streaming and streaming paths
- [ ] Fast suite green
```

Run: `.\.venv\Scripts\python.exe -m pytest -q -m "not slow and not e2e"`.

## Reference

- **Per-endpoint behavior and streaming framing details**:
  [references/endpoints.md](references/endpoints.md)
- Related: `docs/api/OLLAMA_API.md`, `docs/api/STREAMING.md`,
  `docs/research/OPENAI_OLLAMA_API_DESIGN.md`, `docs/research/SSE_STREAMING_PATTERNS.md`.
