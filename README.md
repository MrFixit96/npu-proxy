# NPU Proxy

**Ollama-compatible API proxy for Intel NPU inference via OpenVINO.**

Enables Claude Code, Ollama clients, and any OpenAI-compatible application to use Intel NPU hardware acceleration for local LLM inference—including from WSL2 Linux applications.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Claude Code   │     │   Ollama CLI    │     │   OpenAI SDK    │
│   (WSL2/Win)    │     │   (any OS)      │     │   (Python/JS)   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │ HTTP (port 11435)
                                 ▼
                    ┌────────────────────────┐
                    │      NPU Proxy         │
                    │  FastAPI + OpenVINO    │
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼───────────┐
                    │     Intel NPU          │
                    │  (Meteor/Lunar/Arrow)  │
                    └────────────────────────┘
```

## Features

- **Ollama API Compatible**: Works with any Ollama client (`/api/generate`, `/api/ps`, `/api/show`)
- **OpenAI API Compatible**: Drop-in replacement for OpenAI SDK (`/v1/chat/completions`, `/v1/embeddings`)
- **Intel NPU Acceleration**: Uses OpenVINO GenAI for efficient NPU inference
- **Real-Time Streaming**: True token-by-token SSE streaming (not buffered)
- **WSL2 Bridge**: Run inference on Windows NPU from WSL2 Linux applications
- **NPU→CPU Fallback**: Automatic fallback if NPU unavailable
- **Benchmark CLI**: Compare NPU/GPU/CPU performance on your system

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.12 recommended |
| Windows | 11 23H2+ | With latest NPU drivers |
| Intel NPU | Meteor Lake, Lunar Lake, or Arrow Lake | Core Ultra series |
| OpenVINO | 2025.0+ | With NPU plugin |

### Installation

```powershell
# Clone/navigate to the project
cd npu-proxy

# Install dependencies
pip install -r requirements.txt

# Verify NPU is detected
python -c "import openvino as ov; print('NPU available:', 'NPU' in ov.Core().available_devices)"
# Expected: NPU available: True
```

### Download a Model

```powershell
# TinyLlama 1.1B INT4 (recommended for NPU - ~640MB)
pip install huggingface-hub
huggingface-cli download OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov --local-dir ~/.cache/npu-proxy/models/tinyllama-1.1b-chat-int4-ov
```

### Start the Server

```powershell
# Option 1: Using the startup script (recommended)
.\scripts\start-server.ps1

# Option 2: Direct uvicorn command
python -m uvicorn npu_proxy.main:app --host 0.0.0.0 --port 11435

# Option 3: With real NPU inference enabled
$env:NPU_PROXY_REAL_INFERENCE = "1"
python -m uvicorn npu_proxy.main:app --host 0.0.0.0 --port 11435
```

### Verify It's Working

```powershell
# Health check
Invoke-RestMethod http://localhost:11435/health

# Expected output:
# status        : ok
# npu_available : True
# devices       : {CPU, GPU.0, GPU.1, NPU}
```

---

## Using with Ollama CLI

NPU Proxy is fully compatible with the standard Ollama CLI. Simply point `OLLAMA_HOST` to the proxy:

### Windows PowerShell

```powershell
# Set the Ollama host to NPU Proxy
$env:OLLAMA_HOST = "http://localhost:11435"

# Now use Ollama normally
ollama list
ollama run tinyllama-1.1b-chat-int4-ov "What is 2+2?"
ollama ps
```

### WSL2 / Linux

```bash
# Get Windows host IP (WSL2 gateway)
WINDOWS_HOST=$(ip route show | grep default | awk '{print $3}')
export OLLAMA_HOST="http://${WINDOWS_HOST}:11435"

# Now use Ollama normally
ollama list
ollama run tinyllama-1.1b-chat-int4-ov "Hello!"
```

### Ollama API Examples

```bash
# List models
curl http://localhost:11435/v1/models

# Generate text (Ollama native format)
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-int4-ov",
    "prompt": "Why is the sky blue?",
    "stream": false
  }'

# Show model info
curl -X POST http://localhost:11435/api/show \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama-1.1b-chat-int4-ov"}'

# List running models
curl http://localhost:11435/api/ps

# Get version
curl http://localhost:11435/api/version
```

---

## Using with Claude Code

NPU Proxy can serve as the local LLM backend for Claude Code's Ollama provider.

### Windows: Launch Claude Code with NPU Backend

```powershell
# Option 1: Use the launcher script
.\scripts\ollama-launch-claude.ps1

# Option 2: Manual setup
$env:OLLAMA_HOST = "http://localhost:11435"
claude --provider ollama
```

### WSL2: Launch Claude Code with NPU Backend

```bash
# Option 1: Use the launcher script
./scripts/ollama-launch-claude.sh

# Option 2: Manual setup
WINDOWS_HOST=$(ip route show | grep default | awk '{print $3}')
export OLLAMA_HOST="http://${WINDOWS_HOST}:11435"
claude --provider ollama
```

### What the Launcher Scripts Do

The launcher scripts (`ollama-launch-claude.ps1` and `ollama-launch-claude.sh`):

1. **Set `OLLAMA_HOST`** to point to the NPU Proxy server
2. **Check connectivity** to verify the proxy is running
3. **Display NPU status** (available devices, health)
4. **Launch Claude CLI** with the Ollama provider configured

### Example Session

```
$ ./scripts/ollama-launch-claude.sh

Launching Claude CLI with NPU Proxy backend
  OLLAMA_HOST=http://<WINDOWS_HOST_IP>:11435

NPU Proxy Status: ok
  NPU Available: true

╭─────────────────────────────────────────────────╮
│ Claude Code (Ollama Provider)                   │
│ Model: tinyllama-1.1b-chat-int4-ov              │
│ Backend: Intel NPU via OpenVINO                 │
╰─────────────────────────────────────────────────╯

> What can you help me with?
```

---

## Using with OpenAI SDK

NPU Proxy implements the OpenAI API, making it compatible with any OpenAI SDK:

### Python

```python
from openai import OpenAI

# Point to NPU Proxy
client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="not-needed",  # No API key required
)

# Chat completion
response = client.chat.completions.create(
    model="tinyllama-1.1b-chat-int4-ov",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="tinyllama-1.1b-chat-int4-ov",
    messages=[{"role": "user", "content": "Write a haiku about coding."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Embeddings
embeddings = client.embeddings.create(
    model="all-minilm-l6-v2",
    input=["Hello world", "NPU acceleration is fast"],
)
print(f"Embedding dimensions: {len(embeddings.data[0].embedding)}")
```

### RAG Example

See [examples/rag_example.py](examples/rag_example.py) for a complete RAG (Retrieval-Augmented Generation) example using embeddings and chat:

```bash
python examples/rag_example.py
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:11435/v1',
  apiKey: 'not-needed',
});

const response = await client.chat.completions.create({
  model: 'tinyllama-1.1b-chat-int4-ov',
  messages: [{ role: 'user', content: 'Hello!' }],
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
# Chat completion
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-int4-ov",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'

# Streaming
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-int4-ov",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'

# Embeddings
curl -X POST http://localhost:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-minilm-l6-v2",
    "input": "Hello world"
  }'
```

---

## API Reference

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/embeddings` | POST | Generate embeddings |

### Ollama-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Raw text generation |
| `/api/chat` | POST | Chat completion (alias) |
| `/api/embed` | POST | Generate embeddings (batch) |
| `/api/embeddings` | POST | Generate embeddings (legacy) |
| `/api/ps` | GET | List running models |
| `/api/show` | POST | Show model details |
| `/api/version` | GET | Get version info |
| `/api/pull` | POST | Download model from HuggingFace |
| `/api/search` | GET | Search OpenVINO models |
| `/api/models/known` | GET | List pre-mapped model names |

### System Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with device status |
| `/health/devices` | GET | Detailed device information |

---

## Ollama Compatibility

NPU Proxy is designed for full Ollama API compatibility. It works with:
- ✅ Claude Code
- ✅ Open WebUI  
- ✅ ollama Python/JS libraries
- ✅ Any Ollama-compatible client

**Supported parameters:** `temperature`, `top_k`, `top_p`, `repeat_penalty`, `num_predict`, `seed`, `stop`

**Approximate mappings:** `presence_penalty` and `frequency_penalty` are converted to `repetition_penalty`

**Gracefully ignored:** `mirostat`, `min_p`, `typical_p`, `tfs_z` (not available in OpenVINO)

See [docs/api/OLLAMA_API.md](docs/api/OLLAMA_API.md) for complete details.

---

## Model Management

NPU Proxy can discover and download OpenVINO-optimized models from HuggingFace.

### Search for Models

```bash
# Search for LLaMA models
curl "http://localhost:11435/api/search?q=llama&sort=popular"

# Filter by quantization
curl "http://localhost:11435/api/search?quantization=int4&limit=10"

# Filter by type
curl "http://localhost:11435/api/search?type=llm&sort=newest"
```

**Query Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `q` | "" | Search query |
| `sort` | popular | Sort: popular, newest, downloads, likes |
| `limit` | 20 | Results per page (1-100) |
| `offset` | 0 | Pagination offset |
| `type` | all | Filter: all, llm, embedding, vision |
| `quantization` | "" | Filter: int4, int8, fp16 |
| `min_downloads` | 0 | Minimum download count |

### Download Models

```bash
# Using curl
curl -X POST http://localhost:11435/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "tinyllama"}'

# Using ollama CLI
OLLAMA_HOST=http://localhost:11435 ollama pull tinyllama
```

**Supported Pre-Mapped Models:**
| Ollama Name | HuggingFace Repo | Quantization |
|-------------|------------------|--------------|
| tinyllama | OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov | INT4 |
| tinyllama:fp16 | OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov | FP16 |
| phi-2 | OpenVINO/phi-2-int4-ov | INT4 |
| llama3.2 | OpenVINO/Llama-3.2-3B-Instruct-int4-ov | INT4 |
| mistral | OpenVINO/mistral-7b-instruct-v0.1-int4-ov | INT4 |
| qwen2 | OpenVINO/Qwen2-1.5B-Instruct-int4-ov | INT4 |

### List Known Models

```bash
curl http://localhost:11435/api/models/known
```

---

## Device Selection & Fallback

NPU Proxy automatically selects the best available device with a fallback chain:

```
NPU → GPU → CPU
```

### Automatic Fallback

If your preferred device is unavailable, the proxy automatically falls back:

1. **NPU** (default) - Best efficiency, fastest cold start
2. **GPU** (Intel integrated/discrete) - Fastest inference, slower cold start
3. **CPU** - Always available, slowest

### Manual Device Selection

```powershell
# Use GPU instead of NPU
$env:NPU_PROXY_DEVICE = "GPU"
python -m uvicorn npu_proxy.main:app --host 0.0.0.0 --port 11435

# Force CPU (useful for debugging)
$env:NPU_PROXY_DEVICE = "CPU"
python -m uvicorn npu_proxy.main:app --host 0.0.0.0 --port 11435
```

### Check Device Status

```powershell
# See all available devices
Invoke-RestMethod http://localhost:11435/health/devices

# Example output:
# available_devices : {CPU, GPU.0, GPU.1, NPU}
# active_device     : NPU
# fallback_chain    : {NPU, GPU, CPU}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NPU_PROXY_HOST` | `0.0.0.0` | Server bind address |
| `NPU_PROXY_PORT` | `11435` | Server port |
| `NPU_PROXY_REAL_INFERENCE` | `0` | Enable real NPU inference (`1`) or mock (`0`) |
| `NPU_PROXY_DEVICE` | `NPU` | Preferred device. Falls back: NPU→GPU→CPU |
| `NPU_PROXY_MODEL_PATH` | `~/.cache/npu-proxy/models` | Model cache directory |
| `NPU_PROXY_INFERENCE_TIMEOUT` | `180` | Inference timeout in seconds |

---

## Supported Models

### LLM Models (Chat/Generate)

| Model | Size | Quantization | NPU Optimized |
|-------|------|--------------|---------------|
| `tinyllama-1.1b-chat-int4-ov` | 640MB | INT4 | ✅ Yes |
| `phi-2-int4-ov` | 1.4GB | INT4 | ✅ Yes |
| `llama-2-7b-chat-int4-ov` | 3.8GB | INT4 | ✅ Yes |

### Embedding Models

| Model | Size | Dimensions | Status |
|-------|------|------------|--------|
| `bge-small` | 130MB | 384 | ✅ Production (OpenVINO) |
| `bge-base` | 420MB | 768 | ✅ Production (OpenVINO) |
| `bge-large` | 1.3GB | 1024 | ✅ Production (OpenVINO) |
| `all-minilm` | 91MB | 384 | ✅ Production (OpenVINO) |
| `nomic-embed-text` | 520MB | 768 | ✅ Production (OpenVINO) |

Download embedding models using:
```powershell
python scripts/download_model.py download bge-small
```

See [docs/api/EMBEDDINGS.md](docs/api/EMBEDDINGS.md) for full documentation.

---

## Limitations

- **Concurrency**: NPU supports 1 concurrent inference request. Additional requests queue.
- **Timeout**: Default 180s inference timeout (configurable via `NPU_PROXY_INFERENCE_TIMEOUT`)
- **Memory**: ~2-3GB peak RAM for TinyLlama INT4
- **Streaming**: Mock mode yields real-time; Real inference collects tokens before yielding (async queue pending)

---

## Development

### Running Tests

```powershell
# All tests (~200 tests, ~20s)
pytest tests/ -v

# Fast tests only (skip slow real-model tests)
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=npu_proxy --cov-report=term-missing

# Single test file
pytest tests/test_chat.py -v
```

### Development Server

```powershell
# Hot reload for development
uvicorn npu_proxy.main:app --reload --port 11435

# Debug mode with full logging
$env:NPU_PROXY_REAL_INFERENCE = "1"
uvicorn npu_proxy.main:app --reload --port 11435 --log-level debug
```

### Project Structure

```
npu-proxy/
├── npu_proxy/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   ├── health.py        # /health endpoint
│   │   ├── models.py        # /v1/models endpoint
│   │   ├── chat.py          # /v1/chat/completions endpoint
│   │   ├── embeddings.py    # /v1/embeddings endpoint
│   │   └── ollama.py        # /api/* Ollama endpoints
│   └── inference/
│       └── engine.py        # OpenVINO GenAI wrapper
├── tests/
│   ├── test_health.py
│   ├── test_models.py
│   ├── test_chat.py
│   ├── test_embeddings.py
│   └── test_ollama.py
├── scripts/
│   ├── start-server.ps1     # Windows server launcher
│   ├── ollama-launch-claude.ps1  # Windows Claude launcher
│   └── ollama-launch-claude.sh   # WSL2/Linux Claude launcher
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Troubleshooting

### NPU Not Detected

```powershell
# Check OpenVINO devices
python -c "import openvino as ov; print(ov.Core().available_devices)"

# If NPU not listed:
# 1. Update Intel NPU drivers from Intel Download Center
# 2. Ensure Windows 11 23H2 or later
# 3. Check Device Manager for "Intel AI Boost" or "NPU"
```

### WSL2 Cannot Connect

```bash
# Verify Windows IP is reachable
WINDOWS_HOST=$(ip route show | grep default | awk '{print $3}')
curl http://${WINDOWS_HOST}:11435/health

# If connection refused:
# 1. Ensure server is bound to 0.0.0.0 (not localhost)
# 2. Check Windows Firewall allows port 11435
# 3. Verify WSL2 networking mode (NAT vs mirrored)
```

### Slow First Request

The first inference request loads the model onto the NPU (~5-8 seconds for TinyLlama). Subsequent requests are fast (~1-2 seconds).

```powershell
# Pre-warm the model
curl http://localhost:11435/api/generate -d '{"model":"tinyllama-1.1b-chat-int4-ov","prompt":"hi","stream":false}'
```

### Out of Memory

NPU has limited memory. Use INT4 quantized models:
- TinyLlama INT4: ~640MB
- Phi-2 INT4: ~1.4GB
- Larger models may need CPU fallback

---

## Performance

Benchmarks on Intel Core Ultra 7 155H (Meteor Lake) with TinyLlama 1.1B INT4:

### NPU vs GPU Comparison

| Metric | NPU | GPU | Notes |
|--------|-----|-----|-------|
| **Cold Start** | 8.12s | 21.96s | NPU loads 2.7x faster |
| **Chat Avg** | 4.03s | 2.25s | GPU is 1.8x faster inference |
| **Generate Avg** | 4.23s | 2.31s | Similar pattern |

### When to Use Each Device

| Use Case | Recommended Device |
|----------|-------------------|
| Infrequent queries, fast startup | **NPU** |
| Sustained workloads, throughput | **GPU** |
| Low power consumption | **NPU** |
| Larger models (7B+) | **GPU** or **CPU** |

### Latency Breakdown (NPU, max_tokens=20)

| Phase | Time |
|-------|------|
| Cold start (model load) | ~8s |
| Warm inference | ~4s |
| First token latency | ~0.5s |

Set device with: `$env:NPU_PROXY_DEVICE = "GPU"` or `"NPU"` or `"CPU"`

### Benchmark CLI

Use the benchmark tool to compare devices on your system:

```powershell
# Benchmark NPU (default)
python scripts/benchmark.py run

# Benchmark specific device
python scripts/benchmark.py run --device CPU

# Compare multiple devices
python scripts/benchmark.py compare NPU CPU GPU

# Export results to JSON
python scripts/benchmark.py run --output results.json
```

See [docs/guides/BENCHMARKS.md](docs/guides/BENCHMARKS.md) for detailed documentation.

---

## Streaming

Real-time token streaming is enabled for all chat endpoints. Tokens are delivered as they are generated, not buffered.

### Streaming Architecture

```
┌──────────────────┐    callback()    ┌─────────────┐
│ Inference Thread │ ────────────────▶│ AsyncQueue  │
│ (OpenVINO)       │                  │             │
└──────────────────┘                  └──────┬──────┘
                                             │ async for
                                             ▼
                                      ┌─────────────┐
                                      │ HTTP/SSE    │
                                      │ Response    │
                                      └─────────────┘
```

### Client Integration

```python
# Using OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11435/v1", api_key="ignored")

stream = client.chat.completions.create(
    model="tinyllama",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

See [docs/api/STREAMING.md](docs/api/STREAMING.md) for architecture details and troubleshooting.

---

## Documentation

- **[SPEC.md](SPEC.md)** - Complete technical specification
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[docs/](docs/README.md)** - Full documentation index
  - [Guides](docs/guides/) - User guides and tutorials
  - [API Reference](docs/api/) - Endpoint documentation
  - [Research](docs/research/) - Design decisions and patterns

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This is a personal project - I check issues and PRs occasionally. Forks are encouraged!

---

## Acknowledgments

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's inference toolkit
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) - LLM inference library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Ollama](https://ollama.ai/) - API design inspiration
