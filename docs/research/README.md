# Research Notes

This directory contains research notes and analysis that informed the NPU Proxy implementation. These documents capture:

- Design decisions and alternatives considered
- Patterns learned from other projects (vLLM, FastEmbed, TGI)
- Performance optimization strategies
- API design rationale

## Context

These notes were created during the initial development phase (January-February 2026) to ensure NPU Proxy followed industry best practices. They're preserved here as reference for future contributors.

---

## Index

### Implementation Patterns

| Document | Topic | Key Sources |
|----------|-------|-------------|
| [VLLM_OPENVINO_PATTERNS.md](VLLM_OPENVINO_PATTERNS.md) | LLM inference engine design | vLLM, OpenVINO GenAI |
| [FASTEMBED_OPTIMIZATION.md](FASTEMBED_OPTIMIZATION.md) | Embedding engine optimization | FastEmbed, Infinity |
| [SSE_STREAMING_PATTERNS.md](SSE_STREAMING_PATTERNS.md) | Real-time streaming architecture | TGI, asyncio best practices |

### API & Integration

| Document | Topic | Key Sources |
|----------|-------|-------------|
| [OPENAI_OLLAMA_API_DESIGN.md](OPENAI_OLLAMA_API_DESIGN.md) | API compatibility patterns | OpenAI spec, Ollama API |
| [TOKEN_COUNTING_ACCURACY.md](TOKEN_COUNTING_ACCURACY.md) | Token counting trade-offs | tiktoken, sentencepiece |

### Observability

| Document | Topic | Key Sources |
|----------|-------|-------------|
| [PROMETHEUS_METRICS_PATTERNS.md](PROMETHEUS_METRICS_PATTERNS.md) | Metrics design | vLLM metrics, Prometheus best practices |

### Project Evolution

| Document | Topic | Description |
|----------|-------|-------------|
| [IMPROVEMENT_RECOMMENDATIONS.md](IMPROVEMENT_RECOMMENDATIONS.md) | Feature roadmap research | Analysis of 5 recommended improvements |
| [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) | Codebase audit results | Summary of all optimizations applied |

---

## How to Use These Notes

1. **Before implementing a new feature**: Check if there's relevant research here
2. **When debugging**: Reference patterns may explain why something was implemented a certain way
3. **For contributors**: Understand the reasoning behind design decisions

## Note

These are working documents, not polished documentation. Some may contain outdated information or exploratory notes that weren't fully implemented.
