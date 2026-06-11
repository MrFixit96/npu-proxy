# Model conversion details

## Contents

- The export command
- Tasks and when to use them
- Quantization / weight compression
- Cache layout and metadata
- Name mapping
- Validated model notes

## The export command

`npu_proxy/models/converter.py` builds and runs (via `subprocess`):

```
optimum-cli export openvino --model <hf_repo> --task <task> <output_dir>
```

- Output is written to a `.partial` temp dir, then atomically renamed.
- Timeout: `CONVERSION_TIMEOUT_SECONDS = 3600` (1 hour).
- Missing tool surfaces: "optimum-cli not found. Install it with:
  pip install optimum-intel".
- Required output files: `openvino_model.xml`, `openvino_model.bin`
  (`REQUIRED_OPENVINO_FILES`).

## Tasks and when to use them

`VALID_EXPORT_TASKS`:

| Task | Use for |
| --- | --- |
| `text-generation-with-past` | Causal LLMs with KV cache (the normal generation path) |
| `text-generation` | Causal LLMs without past-key-values |
| `feature-extraction` | Embedding models |

## Quantization / weight compression

OpenVINO/optimum supports INT8 and INT4 weight compression for LLMs (smaller,
faster on NPU). If a conversion needs compression flags, add them in
`_build_command` rather than at call sites, and keep the command shape stable.
INT4 weight compression is the common choice for NPU LLMs; FP32 is auto-downcast
to FP16 on NPU.

## Cache layout and metadata

- Root: `~/.cache/npu-proxy/models` (`DEFAULT_CONVERSION_DIR`); override with
  `NPU_PROXY_MODEL_DIR`.
- Download metadata is recorded in `.npu_proxy_download.json`
  (`DOWNLOAD_METADATA_FILE`) by `downloader.py`.
- Storage keys / runtime names are resolved via `mapper.py`
  (`resolve_model_storage_key`, `resolve_runtime_model_name`).

## Name mapping

`OLLAMA_TO_HUGGINGFACE` (in `mapper.py`) maps friendly names to HF repos, e.g.:

```
all-minilm      -> sentence-transformers/all-MiniLM-L6-v2
bge-small       -> BAAI/bge-small-en-v1.5
bge-base        -> BAAI/bge-base-en-v1.5
bge-large       -> BAAI/bge-large-en-v1.5
all-mpnet       -> sentence-transformers/all-mpnet-base-v2
nomic-embed-text-> nomic-ai/nomic-embed-text-v1
```

Always resolve through `resolve_model_repo` so a new alias works everywhere.

## Validated model notes

- Generative NPU paths require **static shapes**. A model that loads on CPU/GPU
  may fail to compile on NPU.
- Embedding NPU support is validated only for the static-shape `all-minilm`
  path; `bge-small` failed workstation validation on NPU and runs on CPU.
- The alpha GGUF backend is intentionally not a packaged/default runtime path.
