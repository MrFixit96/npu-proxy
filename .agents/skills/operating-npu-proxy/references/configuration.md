# Configuration catalog

All settings are read in `npu_proxy/config.py` and the relevant subsystem modules.
Precedence is generally: explicit CLI flag / argument → environment variable →
built-in default.

## Contents

- Core / runtime
- Devices and routing
- Timeouts
- Caching
- Embeddings
- Backends and templates
- Networking

## Core / runtime

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_REAL_INFERENCE` | `1` enables real inference; else mock mode |
| `NPU_PROXY_MODEL_DIR` | Model cache root (default `~/.cache/npu-proxy/models`) |
| `NPU_PROXY_MAX_PROMPT_LEN` | Max prompt length |
| `NPU_PROXY_TOKEN_LIMIT` | Token cap |

## Devices and routing

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_DEVICE` | Device selection |
| `NPU_PROXY_PREFERRED_DEVICE` | Preferred/default routed device |
| `NPU_PROXY_FALLBACK_DEVICE` | Fallback device |
| `NPU_PROXY_FALLBACK_ON_BUSY` | Busy fallback instead of `503 device_busy` |
| `NPU_PROXY_WARMUP_DEVICES` | Devices warmed at startup |
| `NPU_PROXY_DEVICE_QUEUE_TIMEOUT` | Wait budget for a busy device |

## Timeouts

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_INFERENCE_TIMEOUT` | Generation timeout |
| `NPU_PROXY_LOAD_TIMEOUT` | Model load timeout |
| `NPU_PROXY_EMBED_TIMEOUT` | Embedding request timeout |

## Caching

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_COMPILE_CACHE_DIR` | OpenVINO compiled-model cache dir |
| `NPU_PROXY_COMPILE_CACHE_MODE` | Compile cache mode |
| `NPU_PROXY_PREFIX_CACHE_MODE` | Prompt-prefix cache mode |
| `NPU_PROXY_EMBEDDING_CACHE_SIZE` | Embedding result cache size |

## Embeddings

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_EMBEDDING_MODEL` | Embedding model (default `BAAI/bge-small-en-v1.5`) |
| `NPU_PROXY_EMBEDDING_DEVICE` | Embedding device (default `CPU`) |
| `NPU_PROXY_EMBEDDING_FALLBACK_MODE` | Embedding fallback behavior |
| `NPU_PROXY_EMBEDDING_UNAVAILABLE_COOLDOWN` | Cooldown before retrying preferred device |

See the `serving-npu-embeddings` skill for the validated NPU matrix.

## Backends and templates

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_LLM_BACKEND` | Select the LLM backend |
| `NPU_PROXY_ENABLE_ALPHA_BACKENDS` | Enable the alpha GGUF/llama.cpp backend |
| `NPU_PROXY_LLAMACPP_MODEL_PATH` | Model path for the llama.cpp backend |
| `NPU_PROXY_DISABLE_CHAT_TEMPLATES` | Disable chat-template application |

## Networking

| Var | Purpose |
| --- | --- |
| `NPU_PROXY_HOST` | Bind host (localhost by default) |
| `NPU_PROXY_PORT` | Bind port |
| `NPU_PROXY_ALLOWED_HOSTS` | Allowed Host header values |
