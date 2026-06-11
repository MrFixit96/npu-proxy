# Embedding internals

## Contents

- Config resolution order
- Model directory layout
- Endpoint shapes
- Fallback semantics

## Config resolution order

`embedding_config.py` resolves each setting as: explicit argument → environment
variable → built-in default.

- Model: arg → `NPU_PROXY_EMBEDDING_MODEL` → `BAAI/bge-small-en-v1.5`.
- Device: arg → `NPU_PROXY_EMBEDDING_DEVICE` → `CPU`; trimmed and upper-cased.
- Dimensions: from the model metadata when available, else
  `DEFAULT_EMBEDDING_DIMENSIONS = 384`.

## Model directory layout

Embedding models resolve under the shared model root (`DEFAULT_MODEL_DIR`,
overridable with `NPU_PROXY_MODEL_DIR`). The engine looks for an OpenVINO IR
export (the `feature-extraction` task; see the `converting-models-for-npu`
skill). A legacy flat layout is still consulted via `legacy_root`.

## Endpoint shapes

- **OpenAI** `POST /v1/embeddings` (`api/embeddings.py`): returns
  `{ "data": [{ "embedding": [...], "index": n }], "model": ..., "usage": ... }`.
- **Ollama** `POST /api/embed` (newer) and `POST /api/embeddings` (legacy) in
  `api/ollama.py` — the two use different request/response field names; keep both
  working and test both when changing the engine.

## Fallback semantics

- `NPU_PROXY_EMBEDDING_FALLBACK_MODE` controls whether/how the engine falls back
  off the preferred device.
- After an unavailable/compile failure, `NPU_PROXY_EMBEDDING_UNAVAILABLE_COOLDOWN`
  delays retrying the preferred device so every request doesn't pay the failed
  compile cost.
- A fallback must not change the contracted output dimensions for the model.
