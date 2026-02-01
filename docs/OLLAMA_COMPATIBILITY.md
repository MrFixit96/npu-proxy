# Ollama API Compatibility

NPU Proxy implements the Ollama API to provide seamless compatibility with Ollama clients, including Claude Code, Open WebUI, and other tools that speak the Ollama protocol.

## Overview

This proxy accepts Ollama API requests and translates them to OpenVINO GenAI calls running on Intel NPU hardware. Most Ollama functionality works identically, with some parameters mapped or gracefully ignored due to OpenVINO limitations.

## Parameter Mapping

### Fully Supported Parameters

These parameters work exactly as in Ollama:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.8 | Sampling temperature (0.0-2.0) |
| `top_k` | 40 | Top-K sampling |
| `top_p` | 0.9 | Top-P (nucleus) sampling |
| `seed` | 0 | Random seed (0 = random) |
| `stop` | [] | Stop sequences |

### Renamed Parameters

These parameters are supported but use different names internally:

| Ollama Parameter | OpenVINO Equivalent | Notes |
|------------------|---------------------|-------|
| `repeat_penalty` | `repetition_penalty` | Direct 1:1 mapping |
| `num_predict` | `max_new_tokens` | Controls output length |
| `stop` | `stop_strings` | Same functionality |

### Approximate Mappings

These parameters are converted to the closest OpenVINO equivalent:

| Ollama Parameter | Behavior |
|------------------|----------|
| `presence_penalty` | Combined into `repetition_penalty` |
| `frequency_penalty` | Combined into `repetition_penalty` |

**Conversion Formula:**
```
repetition_penalty = 1.0 + (presence_penalty + frequency_penalty) / 2
```

If you specify `repeat_penalty` explicitly, it takes precedence over `presence_penalty` and `frequency_penalty`.

### Unsupported Parameters (Gracefully Ignored)

These parameters are accepted but have no effect. A DEBUG-level log is emitted when they are used:

| Parameter | Reason |
|-----------|--------|
| `mirostat` | Mirostat sampling not implemented in OpenVINO |
| `mirostat_tau` | Mirostat not available |
| `mirostat_eta` | Mirostat not available |
| `min_p` | MinP sampling not implemented in OpenVINO |
| `typical_p` | Locally typical sampling not implemented |
| `tfs_z` | Tail-free sampling not implemented |

### Silent Parameters (No Effect, No Log)

These parameters are accepted silently because they don't apply to our architecture:

| Parameter | Reason |
|-----------|--------|
| `num_ctx` | Context length is fixed at model export time |
| `num_batch` | Batch scheduling is internal to OpenVINO |

## Supported Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /api/chat` | ✅ Full | Streaming and non-streaming |
| `POST /api/generate` | ✅ Full | Streaming and non-streaming |
| `GET /api/tags` | ✅ Full | List available models |
| `POST /api/show` | ✅ Full | Model details |
| `GET /api/ps` | ✅ Full | Running models |
| `GET /api/version` | ✅ Full | Version info |
| `POST /api/pull` | ✅ Full | Download from HuggingFace |
| `GET /api/search` | ✅ NPU Proxy extension | Search OpenVINO models |
| `POST /api/embeddings` | ✅ Full | Generate embeddings |
| `DELETE /api/delete` | ❌ Not implemented | |
| `POST /api/copy` | ❌ Not implemented | |
| `POST /api/create` | ❌ Not implemented | |

## Client Compatibility

### Claude Code

NPU Proxy is fully compatible with Claude Code. Configure it with:

```bash
# In WSL2
export OLLAMA_HOST=http://<windows-ip>:11435

# Or in Claude Code settings
# Set Ollama endpoint to http://<windows-ip>:11435
```

### Open WebUI

Configure the Ollama API URL in Open WebUI settings:
```
http://<windows-ip>:11435
```

### Python (ollama library)

```python
import ollama

client = ollama.Client(host='http://localhost:11435')
response = client.chat(
    model='tinyllama',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
```

### curl

```bash
curl http://localhost:11435/api/chat -d '{
  "model": "tinyllama",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

## Troubleshooting

### "Parameter X doesn't seem to work"

Check if the parameter is in the "Unsupported Parameters" list above. These are accepted for compatibility but have no effect on generation.

### "Response is different from real Ollama"

The underlying model and inference engine are different. OpenVINO models may produce slightly different outputs than llama.cpp-based Ollama, even with identical parameters.

### "Getting truncated responses"

Increase `num_predict` (default: 128) to generate longer responses:

```json
{
  "model": "tinyllama",
  "prompt": "Tell me a story",
  "options": {"num_predict": 1000}
}
```

### Enable Debug Logging

To see which parameters are being ignored:

```bash
# Set log level before starting the server
export NPU_PROXY_LOG_LEVEL=DEBUG
python -m npu_proxy.main
```
