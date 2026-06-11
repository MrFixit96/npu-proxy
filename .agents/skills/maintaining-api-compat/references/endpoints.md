# Endpoint and streaming details

## Contents

- Ollama endpoints
- OpenAI endpoints
- Streaming framing
- Cancellation

## Ollama endpoints (`api/ollama.py`, mounted under the Ollama prefix)

| Method + path | Purpose |
| --- | --- |
| `POST /generate` | Single-prompt generation (streaming or not) |
| `POST /chat` | Multi-turn chat generation |
| `GET /tags` | List available models |
| `GET /ps` | List loaded/running models |
| `GET /version` | Server version (`<version>-npu-proxy`) |
| `POST /show` | Model details |
| `POST /embed` | Embeddings (newer field shape) |
| `POST /embeddings` | Embeddings (legacy field shape) |
| `POST /pull` | Download/convert a model |
| `GET /search` | Search known/available models |
| `GET /models/known` | Curated known-model list |

## OpenAI endpoints

| Method + path | File |
| --- | --- |
| `POST /v1/chat/completions` | `api/chat.py` |
| embeddings | `api/embeddings.py` |
| models | `api/models.py` |

## Streaming framing

- **OpenAI** (`/v1/chat/completions`, `stream=true`): Server-Sent Events. Each
  chunk is `data: {json}\n\n`; the stream terminates with `data: [DONE]\n\n`.
  Chunks use the `chat.completion.chunk` object with `choices[].delta`.
- **Ollama** (`/generate`, `/chat`, `stream=true`): newline-delimited JSON. Each
  line is a complete JSON object; the final object sets `done: true` and carries
  timing/eval fields. No `[DONE]` sentinel.

Both paths emit the routing headers and are produced through the shared streaming
machinery in `npu_proxy/inference/streaming.py` (`AsyncTokenStream`).

## Cancellation (soft cancel)

If a client disconnects or the server is shutting down, the stream loop stops
requesting further tokens and closes cleanly between tokens. It does not abort
the native engine mid-token. Keep cancellation checks inside the per-token loop
when modifying streaming.
