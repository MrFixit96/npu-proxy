---
name: converting-models-for-npu
description: >-
  Converts and prepares models for NPU Proxy by exporting HuggingFace models to
  OpenVINO IR, managing the model cache, and mapping Ollama names to HuggingFace
  repos. Use when adding or downloading a model, converting to OpenVINO format,
  debugging "model not found" or missing openvino_model.xml/.bin, or when the
  user mentions optimum-cli, model export, IR conversion, NPU static shapes, or
  the model registry.
license: Apache-2.0
compatibility: >-
  Requires the npu-proxy repo, its .venv, and optimum-intel (optimum-cli).
  Conversion downloads from HuggingFace and may need significant disk and time.
metadata:
  author: npu-proxy
  version: "1.0"
---

# Converting Models for NPU

NPU Proxy serves OpenVINO IR models. A model is "ready" only when its directory
contains both `openvino_model.xml` and `openvino_model.bin` (see
`REQUIRED_OPENVINO_FILES` in `npu_proxy/models/converter.py`).

## Orientation

| Concern | File |
| --- | --- |
| HF → OpenVINO export (`optimum-cli`) | `npu_proxy/models/converter.py` |
| Download + cache + metadata | `npu_proxy/models/downloader.py`, `metadata.py` |
| Ollama name → HF repo mapping | `npu_proxy/models/mapper.py` (`OLLAMA_TO_HUGGINGFACE`) |
| Registry / discovery | `npu_proxy/models/registry.py`, `search.py` |
| Generation param mapping | `npu_proxy/models/parameter_mapper.py` |
| CLI download tool | `scripts/download_model.py` |

Cache root: `~/.cache/npu-proxy/models` (`DEFAULT_CONVERSION_DIR`), overridable
via `NPU_PROXY_MODEL_DIR`.

## Core invariants

These are **low-freedom** facts; do not guess around them.

1. **Conversion command shape** is fixed (`_build_command`):
   `optimum-cli export openvino --model <hf_repo> --task <task> <output_dir>`.
   Valid tasks: `text-generation-with-past`, `text-generation`,
   `feature-extraction` (`VALID_EXPORT_TASKS`).
2. **Atomic output.** Conversion writes to a `.partial` temp dir and renames on
   success. A model is valid only when both required IR files are non-empty
   regular files. Never point the server at a `.partial` dir.
3. **NPU needs static shapes.** Generative NPU paths require static-shaped
   models; dynamic-shape models run on CPU/GPU. Embedding NPU support is
   validated only for the static-shape `all-minilm` path — see the
   `serving-npu-embeddings` skill.
4. **Name mapping is the contract.** Ollama-style names (`all-minilm`,
   `bge-small`, …) resolve to HF repos via `OLLAMA_TO_HUGGINGFACE` /
   `resolve_model_repo`. Add new aliases there, not ad hoc in callers.

## Adding a model

```
Add-a-model checklist:
- [ ] 1. Pick the HF repo and the correct task
- [ ] 2. Download/convert into the cache
- [ ] 3. Verify both IR files exist and are non-empty
- [ ] 4. (NPU target) confirm static shapes / device certification
- [ ] 5. Register the Ollama alias if users will request it by name
```

**Convert** (CLI tool, from repo root):

```bash
.\.venv\Scripts\python.exe scripts\download_model.py download <model_name>
.\.venv\Scripts\python.exe scripts\download_model.py list
.\.venv\Scripts\python.exe scripts\download_model.py info <model_name>
```

If `optimum-cli` is missing, install it: `pip install optimum-intel`.

**Verify** the result is loadable with `is_openvino_model(path)` semantics (both
`openvino_model.xml` and `openvino_model.bin` present and non-empty).

## Feedback loop

Convert → verify IR files → load on the target device. If the target is NPU, run
the routing/certification loop (see `certifying-device-routing`) because a model
that loads on CPU may still fail NPU static-shape compilation.

## Reference

- **Conversion details, tasks, quantization, and the validated matrix**:
  [references/conversion.md](references/conversion.md)
