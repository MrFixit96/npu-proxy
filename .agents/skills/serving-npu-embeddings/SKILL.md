---
name: serving-npu-embeddings
description: >-
  Configures and debugs NPU Proxy's embedding inference and its OpenAI/Ollama
  embedding endpoints. Use when changing embedding models, devices, dimensions,
  caching, or fallback behavior, or when the user mentions embeddings, vectors,
  bge-small, all-minilm, /v1/embeddings, /api/embed, or RAG. Also use when
  embeddings fall back off NPU or return unexpected dimensions.
license: Apache-2.0
compatibility: >-
  Requires the npu-proxy repo and its .venv. NPU embedding requires a validated
  static-shape model and real Intel NPU hardware; otherwise embeddings run on CPU.
metadata:
  author: npu-proxy
  version: "1.0"
---

# Serving NPU Embeddings

Embeddings are a separate inference path from text generation, with their own
engine, config, device, and cache. They power `/v1/embeddings` (OpenAI) and
`/api/embed` + `/api/embeddings` (Ollama).

## Orientation

| Concern | File |
| --- | --- |
| Embedding engine (load, encode, cache, fallback) | `npu_proxy/inference/embedding_engine.py` |
| Config resolution + defaults | `npu_proxy/inference/embedding_config.py` |
| OpenAI embeddings endpoint | `npu_proxy/api/embeddings.py` |
| Ollama embed endpoints | `npu_proxy/api/ollama.py` (`/embed`, `/embeddings`) |

## Defaults and configuration

| Setting | Default | Env var |
| --- | --- | --- |
| Model | `BAAI/bge-small-en-v1.5` | `NPU_PROXY_EMBEDDING_MODEL` |
| Device | `CPU` | `NPU_PROXY_EMBEDDING_DEVICE` |
| Dimensions | `384` | (derived from model metadata) |
| Cache size | — | `NPU_PROXY_EMBEDDING_CACHE_SIZE` |
| Fallback mode | — | `NPU_PROXY_EMBEDDING_FALLBACK_MODE` |
| Unavailable cooldown | — | `NPU_PROXY_EMBEDDING_UNAVAILABLE_COOLDOWN` |
| Request timeout | — | `NPU_PROXY_EMBED_TIMEOUT` |

The embedding device defaults to **CPU**, independent of the generation device.
Device values are upper-cased and canonicalized the same way as generation
devices.

## Critical invariant: the validated NPU matrix

- **`all-minilm`** (static-shape) is the only validated NPU embedding path.
- **`bge-small`** (the default model) failed workstation validation on NPU and
  runs on CPU.

So the default config (`bge-small` on `CPU`) is deliberate. To serve embeddings
on NPU, switch the model to `all-minilm` **and** the device to `NPU`; do not set
`NPU_PROXY_EMBEDDING_DEVICE=NPU` while leaving the model on `bge-small`.

## Fallback and cooldown

When the embedding device is unavailable or compilation fails, the engine falls
back (per `NPU_PROXY_EMBEDDING_FALLBACK_MODE`) and applies an unavailable
cooldown before retrying the preferred device. Treat a fallback as expected
behavior, not an error, but verify the response dimensions still match the
requested model.

## Debugging checklist

```
Embedding issue:
- [ ] Confirm which model + device actually loaded (logs / health)
- [ ] Verify dimensions match the model (bge-small=384)
- [ ] If NPU was requested: is the model in the validated NPU matrix?
- [ ] Check cooldown isn't pinning it to CPU after a transient failure
- [ ] Reproduce via /v1/embeddings AND /api/embed (different wire shapes)
```

## Reference

- **Config resolution, cache, fallback details, endpoint shapes**:
  [references/embeddings.md](references/embeddings.md)
- Related: `docs/api/EMBEDDINGS.md`, `docs/research/FASTEMBED_OPTIMIZATION.md`.
