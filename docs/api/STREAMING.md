# Streaming Behavior

This document describes the streaming formats that are implemented **today**.

## Two streaming formats exist

Release-truth note for this pass:

- the real NPU-backed `/api/generate` certification run succeeded on the validation workstation
- compile cache benefit was observed on that workstation
- these findings do **not** change the wire formats documented below

### 1. OpenAI chat streaming

`POST /v1/chat/completions` with `"stream": true` returns:

- a true FastAPI `StreamingResponse` with media type `text/event-stream`
- SSE frames shaped as `data: {json}\n\n`
- successful streams end with `data: [DONE]\n\n`
- failed streams emit a terminal `data: {"error": ...}` payload and stop without a success terminator

Prompt rendering for this route can use tokenizer chat templates when available, but the streaming transport stays SSE either way.

Example:

```text
data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 2. Ollama streaming

These endpoints stream newline-delimited JSON chunks:

- `POST /api/generate`
- `POST /api/chat`
- `POST /api/pull`

Current media type in the implementation is `application/x-ndjson`.

On success, the final `/api/generate` and `/api/chat` frame includes `done: true` and `done_reason`. `done_reason` is `stop` for natural completion and `length` when generation reaches the effective max output token limit.

On failure, `/api/generate` and `/api/chat` emit a terminal NDJSON error object and stop without sending a final success-shaped `done: true` chunk.

`/api/generate` example:

```text
{"model":"tinyllama","response":"Hello ","done":false}
{"model":"tinyllama","response":"world","done":false}
{"model":"tinyllama","response":"","done":true,"done_reason":"stop"}
```

`/api/chat` example:

```text
{"model":"tinyllama","message":{"role":"assistant","content":"Hello "},"done":false}
{"model":"tinyllama","message":{"role":"assistant","content":"world"},"done":false}
{"model":"tinyllama","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}
```

`/api/chat` now goes through the shared chat-template rendering path before streaming begins, falling back to the legacy role-prefixed formatter only when tokenizer templates are unavailable or disabled.

`/api/pull` example:

```text
{"status":"pulling manifest"}
{"status":"pulling openvino_model.xml","digest":"sha256:...","total":1234,"completed":0}
{"status":"pulling openvino_model.xml","digest":"sha256:...","total":1234,"completed":1234}
{"status":"verifying sha256 digest"}
{"status":"success"}
```

## Implementation notes

Real token streaming uses `npu_proxy.inference.streaming.AsyncTokenStream`.

That helper:

- receives token callbacks from the inference thread
- pushes them onto an `asyncio.Queue`
- exposes an async iterator to the HTTP layer
- normalizes the final reason to `stop` or `length`

In real inference mode, the OpenAI and Ollama chat/generate paths both use this bridge. In mock mode, they emit canned chunks.

The final OpenAI stream chunk carries the finish reason at `choices[0].finish_reason`, with an empty `choices[0].delta`. Ollama streaming carries the equivalent value as `done_reason` on the final success frame.

## Cancellation and timeouts

Cancellation is cooperative, or a **soft cancel**. If a client disconnects or the consumer stops, the stream is cancelled and the inference worker receives an abort callback. The worker observes that abort at token boundaries; a native OpenVINO call already in flight may still finish its current token and cannot be interrupted mid-token.

- `AsyncTokenStream` defaults to `60s`
- current real chat/generate streaming paths instantiate it with `180s`
- `top_p` and `timeout` are forwarded to `engine.generate` and `engine.generate_stream`

## Request/response headers

Streaming chat/generate responses can include:

- `X-Request-ID`
- `X-NPU-Proxy-Device` - the device that actually executed the request (kept for backward compatibility; equals the execution device)
- `X-NPU-Proxy-Routed-Device` - the device the context router classified the request for
- `X-NPU-Proxy-Execution-Device` - the device the request actually executed on
- `X-NPU-Proxy-Fallback-Reason` - present only when the execution device differs from the routed device (e.g. `busy`, `device_fallback`)
- `X-NPU-Proxy-Route-Reason` - routing provenance marker (`single_engine_runtime`)
- `X-NPU-Proxy-Token-Count`

## Practical guidance

- Use an OpenAI SDK or SSE client for `/v1/chat/completions`
- Use Ollama-compatible clients or an NDJSON reader for `/api/generate`, `/api/chat`, and `/api/pull`
