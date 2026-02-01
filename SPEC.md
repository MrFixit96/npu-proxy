# SPEC: NPU Proxy - Intel NPU Inference Server

## 1. Objective

Enable **Ollama-compatible** and **OpenAI-compatible** local LLM inference using **Intel NPU** (Neural Processing Unit) hardware via OpenVINO GenAI. The solution provides a FastAPI-based proxy that bridges modern AI client interfaces with Intel's NPU, GPU, and CPU devices.

**Primary Use Case**: Allow WSL2 Linux applications (like Claude Code) to access Windows-hosted Intel NPU hardware for efficient on-device AI inference.

## 2. Environment Context

- **OS:** Windows 11 (23H2+) with NPU drivers, Linux (native NPU driver support)
- **CPU:** Intel Core Ultra (Meteor Lake, Lunar Lake, Arrow Lake) with integrated NPU
- **NPU:** Intel AI Boost (verified working)
- **GPU:** Intel Arc Graphics (optional, used as fallback)
- **Runtime:** Python 3.12+, OpenVINO GenAI 2025.x
- **Framework:** FastAPI + Uvicorn (async HTTP server)

## 3. Problem Statement

### The Challenge

1. **WSL2 NPU Gap**: Intel NPU hardware is not accessible from WSL2 Linux distributions
   - GPU-PV (paravirtualization) exists for GPUs but not for NPUs
   - Microsoft's `dxgkrnl` lacks NPU device passthrough
   - Intel has acknowledged the request (GitHub #56) but provided no timeline after 14+ months

2. **Client Compatibility**: Modern AI tools (Claude Code, Continue, Cursor) expect Ollama or OpenAI APIs
   - No direct OpenVINO integration in these clients
   - Need API translation layer

3. **NPU Complexity**: Intel NPU has unique constraints not present in GPU/CPU
   - Static tensor shapes required
   - Limited context length (~1800-2000 tokens practical max)
   - Long cold start (80-130s model compilation)

### The Solution

A user-space HTTP proxy running on Windows host that:
- Exposes Ollama-compatible and OpenAI-compatible REST APIs
- Routes inference requests to Intel NPU via OpenVINO GenAI
- Bridges WSL2 applications via TCP networking
- Provides automatic device fallback (NPU → GPU → CPU)

## 4. Architecture

### High-Level System Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                      │
├───────────────────┬──────────────────────┬────────────────────────────────────┤
│   Claude Code     │     Ollama CLI       │     OpenAI SDK (Python/JS/etc)     │
│   (WSL2/Win)      │     (any OS)         │                                    │
└─────────┬─────────┴──────────┬───────────┴───────────────┬────────────────────┘
          │                    │                           │
          └────────────────────┼───────────────────────────┘
                               │ HTTP (port 11435)
                               ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            NPU PROXY SERVER                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         FastAPI Application                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐    │  │
│  │  │ /v1/chat/   │  │ /v1/        │  │ /api/       │  │ /health      │    │  │
│  │  │ completions │  │ embeddings  │  │ generate    │  │ /metrics     │    │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────────┘    │  │
│  └─────────┼────────────────┼────────────────┼─────────────────────────────┘  │
│            │                │                │                                 │
│  ┌─────────▼────────────────▼────────────────▼─────────────────────────────┐  │
│  │                    Context-Aware Router                                  │  │
│  │            (Routes by token count to optimal device)                     │  │
│  └─────────────────────────────┬───────────────────────────────────────────┘  │
│                                │                                               │
│  ┌─────────────────────────────▼───────────────────────────────────────────┐  │
│  │                      Inference Layer                                     │  │
│  │  ┌──────────────────────┐    ┌──────────────────────────────────────┐   │  │
│  │  │   InferenceEngine    │    │        EmbeddingEngine               │   │  │
│  │  │   (LLMPipeline)      │    │     (TextEmbeddingPipeline)          │   │  │
│  │  └──────────┬───────────┘    └──────────────────────────────────────┘   │  │
│  └─────────────┼───────────────────────────────────────────────────────────┘  │
│                │                                                               │
│  ┌─────────────▼───────────────────────────────────────────────────────────┐  │
│  │                     OpenVINO GenAI Runtime                               │  │
│  └─────────────┬───────────────────────────────────────────────────────────┘  │
└────────────────┼───────────────────────────────────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────────────────────────────────┐
│                           HARDWARE LAYER                                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │      NPU        │  │      GPU        │  │      CPU        │                 │
│  │  (Primary)      │  │   (Fallback 1)  │  │   (Fallback 2)  │                 │
│  │ Meteor/Lunar/   │  │   Intel iGPU    │  │   x86-64        │                 │
│  │  Arrow Lake     │  │   or dGPU       │  │                 │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Device Fallback Chain

```
NPU (preferred) → GPU (fallback) → CPU (always available)
```

**Fallback Triggers**:
1. Device unavailable in OpenVINO Core
2. Model load failure on device
3. Context exceeds NPU token limit (via Context-Aware Routing)

### Context-Aware Routing

```
┌─────────────────────────────────────────────────────────────────┐
│                    Incoming Request                              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │  Count Message Tokens  │
                     │  (Fast regex ~95% acc) │
                     └────────────┬───────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
         tokens ≤ 1800    1800 < tokens    tokens > limit
                 │          ≤ limit               │
                 ▼                │                ▼
         ┌───────────┐    ┌──────▼──────┐   ┌──────────┐
         │    NPU    │    │    GPU      │   │ Reject   │
         │ (optimal) │    │ (fallback)  │   │ (error)  │
         └───────────┘    └─────────────┘   └──────────┘
```

## 5. Implementation Status

### Current Features (Implemented)

| Feature | Status | Files |
|---------|--------|-------|
| OpenAI Chat API (`/v1/chat/completions`) | ✅ Complete | `npu_proxy/api/chat.py` |
| OpenAI Embeddings (`/v1/embeddings`) | ✅ Complete | `npu_proxy/api/embeddings.py` |
| OpenAI Models (`/v1/models`) | ✅ Complete | `npu_proxy/api/models.py` |
| Ollama Generate (`/api/generate`) | ✅ Complete | `npu_proxy/api/ollama.py` |
| Ollama Chat (`/api/chat`) | ✅ Complete | `npu_proxy/api/ollama.py` |
| Ollama Embed (`/api/embed`) | ✅ Complete | `npu_proxy/api/ollama.py` |
| Ollama Pull (`/api/pull`) | ✅ Complete | `npu_proxy/api/ollama.py` |
| Ollama Tags (`/api/tags`) | ✅ Complete | `npu_proxy/api/ollama.py` |
| SSE Streaming | ✅ Complete | `npu_proxy/inference/streaming.py` |
| Context-Aware Routing | ✅ Complete | `npu_proxy/routing/context_router.py` |
| Device Fallback Chain | ✅ Complete | `npu_proxy/inference/engine.py` |
| Prometheus Metrics | ✅ Complete | `npu_proxy/metrics.py` |
| Health Checks | ✅ Complete | `npu_proxy/api/health.py` |
| Model Registry | ✅ Complete | `npu_proxy/models/registry.py` |
| HuggingFace Download | ✅ Complete | `npu_proxy/models/downloader.py` |
| Parameter Mapping | ✅ Complete | `npu_proxy/models/parameter_mapper.py` |

**Tests**: 300 passing | **Coverage**: ~95%

### Native OS Packaging (Implemented)

| Component | Platform | Status | Files |
|-----------|----------|--------|-------|
| systemd service | Linux | ✅ Complete | `packaging/npu-proxy.service` |
| Install script | Linux | ✅ Complete | `scripts/install_linux.sh` |
| Uninstall script | Linux | ✅ Complete | `scripts/uninstall_linux.sh` |
| PyInstaller build | Windows | ✅ Complete | `scripts/build_windows.ps1`, `npu_proxy.pyinstaller.spec` |
| CLI entry point | All | ✅ Complete | `npu_proxy/cli.py` |

### Planned Features

| Feature | Priority | Status |
|---------|----------|--------|
| WinGet Package | HIGH | ✅ Complete |
| Debian/apt Package | HIGH | ✅ Complete |
| Vision Model Support (VLMPipeline) | MEDIUM | 🔲 Planned |
| Multi-Model Concurrent Inference | LOW | 🔲 Research |

## 6. NPU Constraints and Limitations

### Context Length Limits

| Constraint | Value | Source |
|------------|-------|--------|
| **Default NPU context** | 1024 tokens | OpenVINO GenAI NPU defaults |
| **Extended context** | Up to 4096 tokens | Via `MAX_PROMPT_LEN` config |
| **Practical maximum** | ~1800-2000 tokens | Empirical testing (Issue #3161) |

**Why Context is Limited**:
- NPU memory is constrained (2-4GB depending on model)
- Static KV-cache shapes must be compiled at model load time
- Longer contexts require more memory for attention computation

### Memory Constraints

| Constraint | Value | Notes |
|------------|-------|-------|
| **NPU Memory** | 2-4 GB | Shared with system memory |
| **Concurrent Models** | 1 | Only ONE LLM model at a time |
| **Model Swap** | Not supported | Must restart server to change |

### Model Compilation Time

| Phase | Time | Notes |
|-------|------|-------|
| **Cold start (first load)** | 80-130 seconds | Model compilation to NPU kernels |
| **Warm start (cached)** | 5-8 seconds | Using cached compiled model |
| **Inference latency** | 1-4 seconds | After model is loaded |

**Mitigation**: NPU warmup on startup (`engine.warmup(warmup_tokens=16)`)

### NPU vs GPU vs CPU Comparison

| Characteristic | NPU | GPU | CPU |
|----------------|-----|-----|-----|
| **Cold Start** | 80-130s | 20-30s | 5-10s |
| **Inference Speed** | Moderate | Fast | Slow |
| **Power Efficiency** | Excellent | Moderate | Poor |
| **Memory Limit** | 2-4GB | 4-8GB+ | System RAM |
| **Concurrent Models** | 1 | 1-2 | Multiple |
| **Dynamic Shapes** | ❌ | ✅ | ✅ |
| **Long Context** | Limited | ✅ | ✅ |

## 7. API Reference

### OpenAI-Compatible Endpoints

| Method | Path | Description | Streaming |
|--------|------|-------------|-----------|
| POST | `/v1/chat/completions` | Chat completion | ✅ SSE |
| POST | `/v1/embeddings` | Generate embeddings | ❌ |
| GET | `/v1/models` | List available models | ❌ |

### Ollama-Compatible Endpoints

| Method | Path | Description | Streaming |
|--------|------|-------------|-----------|
| POST | `/api/generate` | Raw text generation | ✅ SSE |
| POST | `/api/chat` | Chat completion | ✅ SSE |
| POST | `/api/embed` | Batch embeddings | ❌ |
| POST | `/api/embeddings` | Single embedding (legacy) | ❌ |
| GET | `/api/ps` | List running models | ❌ |
| POST | `/api/show` | Show model details | ❌ |
| POST | `/api/pull` | Download model | ✅ Progress |
| GET | `/api/tags` | List models | ❌ |
| GET | `/api/version` | Version info | ❌ |

### System Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with NPU status |
| GET | `/health/devices` | Detailed device information |
| GET | `/metrics` | Prometheus metrics |

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier (req_<24-char-hex>) |
| `X-NPU-Proxy-Device` | Device used (NPU/GPU/CPU) |
| `X-NPU-Proxy-Route-Reason` | Why device was selected |
| `X-NPU-Proxy-Token-Count` | Token count for routing decision |

## 8. Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NPU_PROXY_HOST` | `0.0.0.0` | Server bind address |
| `NPU_PROXY_PORT` | `11435` | Server port (matches Ollama) |
| `NPU_PROXY_DEVICE` | `NPU` | Preferred device (NPU, GPU, CPU) |
| `NPU_PROXY_FALLBACK_DEVICE` | (auto) | Override fallback device selection |
| `NPU_PROXY_REAL_INFERENCE` | `0` | Enable real inference (`1`) or mock (`0`) |
| `NPU_PROXY_MODEL_PATH` | `~/.cache/npu-proxy/models` | Model cache directory |
| `NPU_PROXY_INFERENCE_TIMEOUT` | `180` | Inference timeout in seconds |
| `NPU_PROXY_MAX_PROMPT_LEN` | `4096` | Maximum prompt length for NPU |
| `NPU_PROXY_TOKEN_LIMIT` | `1800` | Token threshold for NPU routing |
| `NPU_PROXY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Default embedding model |
| `NPU_PROXY_EMBEDDING_DEVICE` | `CPU` | Device for embeddings |
| `NPU_PROXY_EMBEDDING_CACHE_SIZE` | `1000` | LRU cache size for embeddings |
| `NPU_PROXY_LOG_LEVEL` | `INFO` | Logging verbosity |

### Example Configuration

```powershell
# Windows - Production configuration
$env:NPU_PROXY_REAL_INFERENCE = "1"
$env:NPU_PROXY_DEVICE = "NPU"
$env:NPU_PROXY_INFERENCE_TIMEOUT = "300"

# Start server
npu-proxy --host 0.0.0.0 --port 11435
```

```bash
# WSL2 - Client configuration
WINDOWS_HOST=$(ip route show | grep default | awk '{print $3}')
export OLLAMA_HOST="http://${WINDOWS_HOST}:11435"

# Use with Claude Code or other Ollama clients
claude --chat
```

## 9. Deployment

### Critical Constraint: Host-Only Deployment

> ⚠️ **NPU Proxy MUST run as a native host service.**
> Intel NPU drivers cannot be containerized (no Docker, no Kubernetes).
> WSL2 workloads connect via HTTP bridge to Windows host.

### Windows Deployment

```powershell
# Install from source
pip install -e .

# Run as Windows Service (planned)
# winget install npu-proxy

# Start server
npu-proxy --host 0.0.0.0 --port 11435
```

### Linux Deployment (systemd)

```bash
# Install
sudo ./scripts/install_linux.sh

# Enable and start
sudo systemctl enable npu-proxy
sudo systemctl start npu-proxy

# Check status
sudo systemctl status npu-proxy
```

## 10. Performance Benchmarks

**Test System**: Intel Core Ultra 7 155H (Meteor Lake), 32GB RAM, Windows 11 23H2

### Inference Latency (TinyLlama 1.1B INT4)

| Device | Avg Latency | Tokens/sec |
|--------|-------------|------------|
| NPU | 4.03s | ~5 tok/s |
| GPU | 2.25s | ~9 tok/s |
| CPU | 8.5s | ~2.4 tok/s |

### Cold Start Performance

| Model | Device | Load Time |
|-------|--------|-----------|
| TinyLlama 1.1B INT4 | NPU | 8.12s (cached) |
| TinyLlama 1.1B INT4 | GPU | 21.96s |
| TinyLlama 1.1B INT4 | CPU | 5.2s |

### Embedding Performance (BGE-Small)

| Device | Single Query | Batch (3 docs) |
|--------|--------------|----------------|
| CPU | ~28ms | ~25ms |
| NPU | ~35ms | ~30ms |

## 11. Model Compatibility Matrix

| Model | Type | NPU | GPU | CPU | Notes |
|-------|------|-----|-----|-----|-------|
| TinyLlama 1.1B INT4 | LLM | ✅ | ✅ | ✅ | Recommended for NPU |
| Phi-2 2.7B INT4 | LLM | ✅ | ✅ | ✅ | Good balance |
| Mistral 7B INT4 | LLM | ⚠️ | ✅ | ✅ | May exceed NPU memory |
| LLaMA-2 7B INT4 | LLM | ⚠️ | ✅ | ✅ | May exceed NPU memory |
| Granite 4 Micro | LLM | ✅ | ✅ | ✅ | 1B FP32 model |
| BGE-Small | Embedding | ✅ | ✅ | ✅ | 384 dimensions |
| BGE-Base | Embedding | ✅ | ✅ | ✅ | 768 dimensions |
| All-MiniLM-L6-v2 | Embedding | ✅ | ✅ | ✅ | Lightweight |

**Legend**: ✅ Supported | ⚠️ May work with limitations | ❌ Not recommended

## 12. Prometheus Metrics

```
# Counter: Total requests by endpoint and status
npu_proxy_requests_total{endpoint="/v1/chat/completions", status="200"}

# Histogram: Inference latency
npu_proxy_inference_duration_seconds{model="tinyllama", device="NPU"}

# Histogram: Time to first token (critical SLO)
npu_proxy_time_to_first_token_seconds{model="tinyllama"}

# Histogram: Inter-token latency
npu_proxy_inter_token_latency_seconds{model="tinyllama"}

# Gauge: Tokens per second (real-time throughput)
npu_proxy_tokens_per_second{model="tinyllama"}

# Gauge: Currently loaded models
npu_proxy_loaded_models{model="tinyllama"}

# Counter: Tokens generated
npu_proxy_tokens_generated_total{model="tinyllama"}
```

## 13. References

### Official Documentation

| Resource | URL |
|----------|-----|
| OpenVINO GenAI NPU Guide | https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide-npu.html |
| OpenVINO GenAI GitHub | https://github.com/openvinotoolkit/openvino.genai |
| Intel NPU Driver (Linux) | https://github.com/intel/linux-npu-driver |
| OpenVINO Toolkit | https://github.com/openvinotoolkit/openvino |

### API Compatibility References

| Resource | URL |
|----------|-----|
| Ollama API Docs | https://github.com/ollama/ollama/blob/main/docs/api.md |
| OpenAI API Reference | https://platform.openai.com/docs/api-reference |

### Research & Implementation References

| Resource | URL | Usage |
|----------|-----|-------|
| vLLM | https://github.com/vllm-project/vllm | Metrics patterns, TTFT/TPOT |
| FastEmbed | https://github.com/qdrant/fastembed | Embedding optimization |
| TGI | https://github.com/huggingface/text-generation-inference | Streaming patterns |
| tiktoken | https://github.com/openai/tiktoken | Token counting accuracy |

## 14. Project Structure

```
npu-proxy/
├── npu_proxy/
│   ├── api/
│   │   ├── chat.py           # OpenAI chat endpoint
│   │   ├── embeddings.py     # OpenAI embeddings endpoint
│   │   ├── health.py         # Health checks
│   │   ├── metrics.py        # Prometheus endpoint
│   │   ├── models.py         # Model listing
│   │   └── ollama.py         # Ollama-compatible endpoints
│   ├── inference/
│   │   ├── engine.py         # LLM inference engine
│   │   ├── embedding_engine.py # Embedding engine
│   │   ├── streaming.py      # AsyncTokenStream
│   │   └── tokenizer.py      # Token counting
│   ├── models/
│   │   ├── registry.py       # Model catalog
│   │   ├── downloader.py     # HuggingFace download
│   │   ├── converter.py      # Model conversion
│   │   ├── mapper.py         # Name resolution
│   │   ├── parameter_mapper.py # Param translation
│   │   └── ollama_defaults.py # Default values
│   ├── routing/
│   │   └── context_router.py # Context-aware routing
│   ├── main.py               # FastAPI app
│   ├── metrics.py            # Prometheus metrics
│   └── cli.py                # CLI entry point
├── tests/                    # 300+ test files
├── scripts/                  # Build and launch scripts
├── packaging/                # systemd service files
├── docs/                     # Additional documentation
├── pyproject.toml            # Project metadata
├── requirements.txt          # Dependencies
└── LICENSE                   # MIT License
```

## 15. GitHub Repository

**Public Repository**: https://github.com/MrFixit96/npu-proxy

---

*Document Version*: 1.0.0  
*Last Updated*: February 2026  
*Status*: Production Ready
