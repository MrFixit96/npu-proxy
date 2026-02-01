# Downloading and Setting Up Embedding Models

## Overview

The npu-proxy service requires embedding models to be downloaded and converted to OpenVINO IR format before use. This process prepares models for efficient inference on Intel NPU, GPU, or CPU devices. The conversion step is handled automatically by the download script, but manual conversion is also supported for advanced use cases.

## Quick Start

To download and convert the default embedding model (bge-small) with a single command:

```bash
python scripts/download_model.py download bge-small
```

This command will:
1. Download the model from HuggingFace
2. Convert it to OpenVINO IR format
3. Store it in the standard cache location

## Supported Models

The following embedding models are currently supported:

| Ollama Name | HuggingFace Repo | Dimensions |
|-------------|------------------|------------|
| bge-small | BAAI/bge-small-en-v1.5 | 384 |
| bge-base | BAAI/bge-base-en-v1.5 | 768 |
| bge-large | BAAI/bge-large-en-v1.5 | 1024 |
| all-minilm | sentence-transformers/all-MiniLM-L6-v2 | 384 |
| nomic-embed-text | nomic-ai/nomic-embed-text-v1.5 | 768 |

The model name provided to npu-proxy should match the "Ollama Name" column. Each model offers different trade-offs between dimensionality, performance, and accuracy.

## Manual Download

For advanced users or custom model conversions, you can use the optimum-cli tool directly:

```bash
optimum-cli export openvino --task feature-extraction --model BAAI/bge-small-en-v1.5 ~/.cache/npu-proxy/models/embeddings/BAAI_bge-small-en-v1.5
```

Replace the model repository path with your desired model from HuggingFace. The output directory structure should follow the pattern: `~/.cache/npu-proxy/models/embeddings/{MODEL_NAME}/`.

## Storage Location

Downloaded and converted embedding models are stored in:

```
~/.cache/npu-proxy/models/embeddings/
```

Each model is organized in its own subdirectory named after the HuggingFace repository path (with forward slashes converted to underscores). For example:
- `BAAI_bge-small-en-v1.5/`
- `BAAI_bge-base-en-v1.5/`
- `sentence-transformers_all-MiniLM-L6-v2/`

This location can be overridden by setting the `NPU_PROXY_MODEL_CACHE_DIR` environment variable.

## Environment Variables

The following environment variables control embedding model behavior:

### NPU_PROXY_EMBEDDING_MODEL

Sets the default embedding model name. Must correspond to one of the supported model names listed above.

```bash
export NPU_PROXY_EMBEDDING_MODEL=bge-base
```

### NPU_PROXY_EMBEDDING_DEVICE

Specifies the device for embedding model inference. Supported values are:

- `CPU` - Run on CPU (always available, slower inference)
- `GPU` - Run on discrete GPU (if available)
- `NPU` - Run on Intel NPU (if driver installed)

```bash
export NPU_PROXY_EMBEDDING_DEVICE=NPU
```

If not set, the service will attempt to use NPU, then GPU, then fall back to CPU.

## Troubleshooting

### "Model not found" Error

If you receive a "Model not found" error when starting the service:

1. Verify the model has been downloaded:
   ```bash
   python scripts/download_model.py download bge-small
   ```

2. Check that the model exists in the cache directory:
   ```bash
   ls ~/.cache/npu-proxy/models/embeddings/
   ```

3. Ensure the model name in your configuration matches one of the supported models listed above.

### "OpenVINO GenAI not available" Error

This indicates that the openvino-genai library is not installed. Install it with:

```bash
pip install openvino-genai
```

### NPU Device Not Detected

If the service cannot detect your Intel NPU:

1. Verify the Intel NPU driver is installed on your system.
2. Check driver installation status:
   ```bash
   # On Linux
   lsmod | grep vpu
   
   # On Windows
   # Check Device Manager for Intel AI Boost Neural Processing Unit
   ```

3. Set the device explicitly to CPU as a fallback:
   ```bash
   export NPU_PROXY_EMBEDDING_DEVICE=CPU
   ```

4. If the driver is installed but still not detected, refer to the Intel NPU driver documentation for your operating system.

### Out of Memory Errors

If you encounter memory errors during model conversion or inference:

1. Consider using a smaller model (bge-small or all-minilm) instead of larger variants
2. Ensure sufficient free disk space (at least 1GB) for model downloads
3. Run the download script on a machine with more available RAM if conversion fails
