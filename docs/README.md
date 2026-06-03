# Documentation Index

This index points to documentation for the current local developer-workstation version of NPU Proxy.

## Core docs

- [README.md](../README.md) - project overview, quick start, and local `127.0.0.1:8080` usage
- [SPEC.md](../SPEC.md) - current implementation snapshot and release-truth behavior
- [CHANGELOG.md](../CHANGELOG.md) - released changes

## API docs

- [api/OLLAMA_API.md](api/OLLAMA_API.md) - Ollama-compatible endpoints, request/response shapes, and local-client notes
- [api/EMBEDDINGS.md](api/EMBEDDINGS.md) - OpenAI/Ollama embedding endpoints, validation limits, and response behavior
- [api/STREAMING.md](api/STREAMING.md) - OpenAI SSE and Ollama NDJSON streaming formats and cancellation notes

## Guides

- [guides/MODEL_DOWNLOAD.md](guides/MODEL_DOWNLOAD.md) - Hugging Face/OpenVINO model download and conversion workflows
- [guides/BENCHMARKS.md](guides/BENCHMARKS.md) - benchmark helper usage for local performance checks

## Development / reference

- [development/CONVERTER_API.md](development/CONVERTER_API.md) - model converter API behavior and integration notes

## Research notes

- [research/](research/) - historical background material and exploratory notes; useful context, but not release commitments or current product behavior

For shipped behavior and current validation truth, prefer the root `README.md`, `SPEC.md`, and the API/guide docs above.
