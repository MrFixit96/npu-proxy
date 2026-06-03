# Model Download Guide

This guide documents the **current** model-download behavior in this repository.

## Scope

There are two separate model flows in-tree today:

1. **LLM model directories** used by real chat/generate inference
2. **Embedding model exports** used by the embedding engine

## LLM models

Real LLM inference expects a local OpenVINO model directory under:

```text
~/.cache/npu-proxy/models/<model-id>
```

The default model directory is:

```text
~/.cache/npu-proxy/models
```

The default LLM model ID is:

```text
tinyllama-1.1b-chat-int4-ov
```

So the default model path used by the LLM engine is:

```text
~/.cache/npu-proxy/models/tinyllama-1.1b-chat-int4-ov
```

Example direct Hugging Face download:

```powershell
pip install huggingface-hub
hf download OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov --local-dir "$HOME\.cache\npu-proxy\models\tinyllama-1.1b-chat-int4-ov"
```

You can also use the Ollama-compatible API while the proxy is running on its default loopback address:

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/pull `
  -H "Content-Type: application/json" `
  -d '{"name":"tinyllama","stream":false}'
```

### Download safety limits

The in-process downloader in `npu_proxy.models.downloader` uses `huggingface_hub` and does **not** pull arbitrary repository contents by default. It validates the Hugging Face manifest before downloading:

- default downloads use Hugging Face `allow_patterns` built from the required OpenVINO files plus safe config/tokenizer patterns (`openvino_*.xml`, `openvino_*.bin`, `config.json`, tokenizer files, and related metadata)
- each selected file is capped at 8 GiB
- the selected download set is capped at 16 GiB total
- the selected download set is capped at 64 files
- unexpected unsafe filenames are rejected

Full-snapshot download is opt-in in the Python API only:

```python
from npu_proxy.models.downloader import download_model

result = download_model("tinyllama", full_snapshot=True)
```

The parameter name is `full_snapshot`. It defaults to `False`. The `/api/pull` endpoint and `get_download_progress()` use the safe allow-pattern workflow and do not expose a full-snapshot request flag.

### Public vs private Hugging Face pulls

`/api/pull` supports both:

- **public repos** with no token
- **private repos** with an explicit Hugging Face token

For private repos, provide the token either in the request body or as a Bearer token header:

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/pull `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer hf_xxx" `
  -d '{"name":"your-org/private-model","stream":false}'
```

The service does **not** silently reuse ambient Hugging Face credentials for arbitrary user-selected pulls. That keeps the server from acting like a skeleton key for any repo the operator can access.

### Validation note

During this release pass, the real OpenVINO LLM path was certified successfully on Intel NPU for `/api/generate`. Compile cache benefit was also observed on the validation workstation, but cache settings do not change the on-disk model layout documented here.

## Embedding models

### Runtime expectation

The embedding runtime looks for exported models at:

```text
~/.cache/npu-proxy/models/embeddings/<canonical-id-or-sanitized-repo>
```

Examples:

```text
~/.cache/npu-proxy/models/embeddings/all-minilm-l6-v2
~/.cache/npu-proxy/models/embeddings/bge-small
```

Known registry-backed models use canonical runtime IDs such as `all-minilm-l6-v2` and `bge-small`. Direct or unknown repo IDs fall back to a sanitized repo-name directory such as `BAAI_bge-small-en-v1.5`, which is also still accepted as a legacy compatibility path.

Current workstation truth: this path documents how the runtime finds an embedding export. `all-minilm-l6-v2` validated on `NPU` here via a static-shape profile, while `bge-small` on `NPU` still failed because the Intel NPU plugin raised `check_sdpa_nodes(model)`.

### Reliable manual export path

If you want the runtime to auto-detect an embedding model today, export directly into the runtime path:

```powershell
optimum-cli export openvino `
  --task feature-extraction `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  "$HOME\.cache\npu-proxy\models\embeddings\all-minilm-l6-v2"
```

For `bge-small`, the canonical runtime path is:

```powershell
optimum-cli export openvino `
  --task feature-extraction `
  --model BAAI/bge-small-en-v1.5 `
  "$HOME\.cache\npu-proxy\models\embeddings\bge-small"
```

### Helper script

This repository also includes:

```powershell
python scripts\download_model.py download all-minilm
python scripts\download_model.py list
python scripts\download_model.py info all-minilm
```

Current helper-script behavior:

- supports `download`, `list`, and `info`
- `download` accepts `--force`, `--revision`, and `--timeout-seconds`
- resolves short aliases such as `bge-small`, `e5-large`, and `all-minilm`
- exports via `optimum-cli export openvino --task feature-extraction`
- uses `NPU_PROXY_DOWNLOAD_TIMEOUT` as the default timeout source when set, otherwise 3600 seconds
- prints the cache path it used
- writes Hugging Face source/revision metadata when available

### Important truth note

The helper script now resolves known embedding aliases into the same runtime cache path that the service probes, including canonical IDs such as `all-minilm-l6-v2`.

### Validated NPU recipe

On the validation workstation, the strongest first-class NPU embedding path is:

```powershell
python scripts\download_model.py download all-minilm
$env:NPU_PROXY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
$env:NPU_PROXY_EMBEDDING_DEVICE = "NPU"
```

Restart the service after changing the environment so the new defaults are loaded.

## Alpha GGUF path (source installs only)

This repository also contains an alpha-gated `llama.cpp` backend path for local `.gguf` files.

Current truth:

- it is **not** the default runtime path
- it is **feature-gated**
- it is currently **CPU-only**
- it is **source-install only** because `llama-cpp-python` is not part of the default package dependencies or packaging assets

If you are experimenting from a source checkout, the required opt-in variables are:

```powershell
$env:NPU_PROXY_LLM_BACKEND = "llama_cpp"
$env:NPU_PROXY_ENABLE_ALPHA_BACKENDS = "1"
$env:NPU_PROXY_DEVICE = "CPU"
$env:NPU_PROXY_LLAMACPP_MODEL_PATH = "C:\path\to\model.gguf"
```

This guide does **not** claim that the alpha GGUF path is release-validated on the workstation used for this docs pass.

## Supported helper aliases

Current alias resolution in `scripts\download_model.py` supports aliases from `npu_proxy.models.mapper`, including:

- `bge-small`
- `bge-base`
- `bge-large`
- `e5-small`
- `e5-large`
- `all-minilm`
- `nomic-embed-text`

Direct Hugging Face repo IDs also work.

## Embedding configuration

| Variable | Default | Notes |
|---|---|---|
| `NPU_PROXY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Configured embedding model repo |
| `NPU_PROXY_EMBEDDING_DEVICE` | `CPU` | Current default device |

## Troubleshooting

### Real embedding model not picked up

Check that the exported files exist in the runtime path the service expects:

```text
~/.cache/npu-proxy/models/embeddings/<canonical-id-or-sanitized-repo>/
  openvino_model.xml
  openvino_model.bin
```

### `optimum-cli` not found

Install the exporter:

```powershell
pip install "optimum-intel[openvino]"
```

### Embedding requests fail by default

The default behavior is now to fail embedding requests when the configured runtime-ready model is missing or unusable. Only set `NPU_PROXY_EMBEDDING_FALLBACK_MODE=hash` when you explicitly want deterministic hash fallback for operator testing or wiring checks. See [../api/EMBEDDINGS.md](../api/EMBEDDINGS.md) for the runtime behavior details.

### NPU embedding export still does not validate

If your export exists but NPU execution still fails, do not assume the export is release-ready. On the validation workstation for this docs pass, `all-minilm-l6-v2` worked on `NPU` via a static-shape profile, while `BAAI/bge-small-en-v1.5` on `NPU` still failed inside the Intel NPU plugin with `check_sdpa_nodes(model)`.
