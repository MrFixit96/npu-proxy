# Embeddings

NPU Proxy provides text embeddings via OpenVINO's TextEmbeddingPipeline, with full Ollama and OpenAI API compatibility.

## Endpoints

| Endpoint | Format | Description |
|----------|--------|-------------|
| `POST /v1/embeddings` | OpenAI | OpenAI SDK compatible |
| `POST /api/embed` | Ollama | Current Ollama format (batch support) |
| `POST /api/embeddings` | Ollama | Legacy Ollama format (single prompt) |

## Default Model

- **Model**: `BAAI/bge-small-en-v1.5`
- **Dimensions**: 384
- **Device**: CPU (most reliable for embeddings)

## Supported Embedding Models

| Model Name | Dimensions | Description |
|------------|------------|-------------|
| `bge-small` | 384 | Fast, high-quality English embeddings |
| `bge-base` | 768 | Balanced quality/speed |
| `bge-large` | 1024 | Highest quality BGE model |
| `e5-small` | 384 | Lightweight E5 model |
| `e5-large` | 1024 | Multilingual E5 embeddings |
| `all-minilm` | 384 | Sentence-transformers classic |
| `nomic-embed-text` | 768 | Nomic AI embeddings |

## Usage Examples

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11435/v1/",
    api_key="not-needed"
)

response = client.embeddings.create(
    model="bge-small",
    input=["Hello, world!", "How are you?"]
)

for embedding in response.data:
    print(f"Index {embedding.index}: {len(embedding.embedding)} dimensions")
```

### Ollama Client (Python)

```python
import requests

# Current format (/api/embed)
response = requests.post("http://localhost:11435/api/embed", json={
    "model": "bge-small",
    "input": ["Document 1", "Document 2"]
})
embeddings = response.json()["embeddings"]

# Legacy format (/api/embeddings)
response = requests.post("http://localhost:11435/api/embeddings", json={
    "model": "bge-small",
    "prompt": "Single document"
})
embedding = response.json()["embedding"]
```

### cURL

```bash
# OpenAI format
curl -X POST http://localhost:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-small", "input": "Hello world"}'

# Ollama current format
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-small", "input": ["text1", "text2"]}'

# Ollama legacy format
curl -X POST http://localhost:11435/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-small", "prompt": "Hello world"}'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NPU_PROXY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace model repo |
| `NPU_PROXY_EMBEDDING_DEVICE` | `CPU` | OpenVINO device (CPU, GPU, NPU) |

### Example

```powershell
$env:NPU_PROXY_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
$env:NPU_PROXY_EMBEDDING_DEVICE = "GPU"
python -m npu_proxy
```

## Model Download

Embedding models must be exported to OpenVINO format before use. See [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) for detailed instructions.

### Quick Start

```bash
python scripts/download_model.py download bge-small
```

### Using optimum-cli

```bash
pip install optimum[openvino]

# Export default model
optimum-cli export openvino \
  --task feature-extraction \
  --model BAAI/bge-small-en-v1.5 \
  ~/.cache/npu-proxy/models/embeddings/BAAI/bge-small-en-v1.5
```

### Automatic Fallback

If the configured model is not downloaded, NPU Proxy automatically falls back to a hash-based embedding generator. This produces deterministic embeddings that are useful for testing but lack semantic meaning.

To check which mode is active:

```python
from npu_proxy.inference.embedding_engine import get_embedding_engine

engine = get_embedding_engine()
info = engine.get_engine_info()
print(info)
# {'model_name': 'BAAI/bge-small-en-v1.5', 'dimensions': 384, 'is_production': False, 'device': 'CPU'}
```

The `is_production` field indicates whether real OpenVINO inference is being used (`True`) or hash-based fallback (`False`).

## API Reference

### POST /v1/embeddings (OpenAI format)

**Request:**
```json
{
  "model": "bge-small",
  "input": "text to embed",
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, -0.2, ...], "index": 0}
  ],
  "model": "bge-small",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

### POST /api/embed (Ollama current format)

**Request:**
```json
{
  "model": "bge-small",
  "input": ["text1", "text2"],
  "truncate": true,
  "keep_alive": "5m"
}
```

**Response:**
```json
{
  "model": "bge-small",
  "embeddings": [[0.1, -0.2, ...], [0.3, -0.4, ...]],
  "total_duration": 14143917,
  "load_duration": 1019500,
  "prompt_eval_count": 8
}
```

### POST /api/embeddings (Ollama legacy format)

**Request:**
```json
{
  "model": "bge-small",
  "prompt": "text to embed"
}
```

**Response:**
```json
{
  "embedding": [0.1, -0.2, ...]
}
```

## Performance

Benchmarked on Intel Core Ultra with NPU (Windows 11):

| Implementation | Semantic | Single Query | Batch (3 docs) | Use Case |
|----------------|----------|--------------|----------------|----------|
| Production (OpenVINO) | Yes | ~28ms | ~25ms | RAG, search, similarity |
| Fallback (hash-based) | No | <1ms | <1ms | Testing, development |

Model load time is approximately 2 seconds on first request (cached thereafter).

## Troubleshooting

### Model not found

```
WARNING: Embedding model not downloaded, using hash-based fallback
```

**Solution:** Download the model using `optimum-cli export openvino` as shown above.

### Wrong dimensions

If embeddings have unexpected dimensions, check which model is configured:

```python
from npu_proxy.inference.embedding_engine import get_embedding_engine
print(get_embedding_engine().dimensions)
```

### Device not available

If GPU or NPU device fails, the engine falls back to CPU automatically. Check logs for device selection messages.
