# Changelog

All notable changes to NPU Proxy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Changed

- Centralized runtime version reporting across the CLI, FastAPI app, health endpoint, and Ollama compatibility surface.
- Upgraded the project dependency floor to the current FastAPI, OpenVINO, Hugging Face Hub, Optimum Intel, and test-tooling lines.
- Updated Hugging Face Hub integration to the latest compatible pre-1.0 Hub line and switched documentation from `huggingface-cli` to `hf`.
- Refreshed the documented Windows, WSL 2, and Intel NPU support baseline for modern OpenVINO releases.

---

## [0.1.0] - 2026-02-01

### Added

- **OpenAI-Compatible API**
  - `/v1/chat/completions` - Chat completion with SSE streaming
  - `/v1/embeddings` - Text embeddings
  - `/v1/models` - Model listing

- **Ollama-Compatible API**
  - `/api/generate` - Raw text generation
  - `/api/chat` - Chat completion
  - `/api/embed` - Batch embeddings
  - `/api/pull` - Model download from HuggingFace
  - `/api/tags` - List models
  - `/api/ps` - Running models
  - `/api/show` - Model details

- **Core Features**
  - Context-aware routing (NPU→GPU→CPU based on token count)
  - Device fallback chain with automatic failover
  - Real-time SSE streaming via AsyncTokenStream
  - NPU warmup to reduce cold start latency
  - LRU caching for embeddings

- **Observability**
  - Prometheus metrics endpoint (`/metrics`)
  - Time-to-first-token (TTFT) tracking
  - Inter-token latency (TPOT) tracking
  - Request tracing via X-Request-ID header

- **Documentation**
  - Comprehensive SPEC.md
  - API reference documentation
  - Streaming architecture docs
  - Benchmark tooling

### Infrastructure

- FastAPI + Uvicorn async server
- OpenVINO GenAI integration
- 300+ unit tests
- Native OS packaging (systemd, Windows scripts)

