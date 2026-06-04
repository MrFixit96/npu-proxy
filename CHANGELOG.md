# Changelog

All notable changes to NPU Proxy are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-06-03

This release is the result of a comprehensive security, correctness, and code-quality
review. NPU Proxy is intended for a developer's local workstation (single user, loopback),
not as a shared production proxy; the changes below harden that local model rather than
adding multi-tenant features.

### Security

- Added a Host-header allow-list middleware to mitigate DNS-rebinding. Requests whose
  `Host` header is not on the allow-list are rejected with `421 Misdirected Request`.
  The default allow-list covers loopback and test clients
  (`localhost`, `127.0.0.1`, `::1`, `[::1]`, `testserver`, `test`) and is configurable via
  the `NPU_PROXY_ALLOWED_HOSTS` environment variable or the `--allowed-hosts` CLI flag.
- The server now binds to loopback (`127.0.0.1`) by default. Binding to a non-loopback
  interface emits a warning, and `scripts/start-server.ps1` requires an explicit opt-in to
  bind `0.0.0.0`. No authentication is added: this is intentional for a single-user local tool.
- Hardened model/tokenizer path handling against path traversal. Registry model names are
  slug-validated and resolved paths are confined to the configured model directory.
- The model downloader now enforces Hugging Face allow-patterns and download size caps, and
  full-snapshot downloads are opt-in.
- Removed a misleading fabricated SHA-256 digest from model metadata responses.
- Error responses are sanitized so internal exception details are no longer leaked to clients.

### Added

- `finish_reason` on OpenAI chat responses (`choices[].finish_reason` is `"stop"` or
  `"length"`) and `done_reason` on Ollama responses, derived from emitted-token counts and
  native backend signals.
- Ollama `/api/tags` endpoint, and `/api/show` now returns real model metadata.
- Semantic validation for embedding requests (empty input, batch larger than 128,
  whitespace-only text, and oversized text) on both the OpenAI and Ollama embedding endpoints.
- New configuration knobs: `NPU_PROXY_ALLOWED_HOSTS`, `NPU_PROXY_PREFERRED_DEVICE`, and
  `NPU_PROXY_FALLBACK_DEVICE` environment variables, plus the `--allowed-hosts` CLI flag.
- Backend shutdown/close lifecycle so inference backends release resources cleanly.

### Changed

- Streaming responses now use a true `StreamingResponse`, fixing SSE double-encoding.
- Blocking engine and device probes are offloaded to worker threads, and non-streaming
  inference is wrapped with a timeout, keeping the event loop responsive.
- `top_p` and the inference timeout are now forwarded to the engine on both the streaming and
  non-streaming paths for OpenAI and Ollama requests.
- Routing and bootstrap configuration is centralized with forgiving parsing: an invalid token
  limit or device value now logs a warning and falls back to the default instead of failing a
  request; explicit startup validation still raises.
- Execution-device reporting in response headers and the health endpoint is now truthful
  (reports the device that actually ran the request).
- Thread-safety hardening: a llama.cpp backend lock, locked engine and metrics initialization,
  a `reset_engine(*, force=False)` race guard, bounded metrics label cardinality, and
  thread-safe bounded embedding caches with per-key load locks.
- Packaging and dev-workflow cleanup: development dependencies moved to `requirements-dev.txt`,
  the PyInstaller spec collects required submodules, and the OpenVINO version pin is documented.

### Fixed

- **Critical:** an embedding batch-handling bug that could return zero/blank vectors. Strict
  batch-count, finite-value, and dimension validation now guarantees exactly one finite vector
  per input.
- `devices.get_available_devices()` now degrades to `["CPU"]` with a logged warning when the
  OpenVINO runtime cannot be imported or initialized, instead of raising.
- Streaming same-loop token push and idempotent stream completion/error handling.
- The model converter now runs in a timeout-safe child process and publishes output via an
  atomic temporary-directory rename.
- Narrowed broad exception handlers, added missing logging, fixed a `KeyError` in Ollama
  defaults, and made default data structures immutable.

### Testing

- Expanded the fast test suite from 496 to 648 tests: engine-contract tests with a deterministic
  fake engine, a fake-OpenVINO device-discovery layer, and broader streaming, CLI, metrics, and
  config coverage, plus shared autouse fixtures that reset global state for order independence.

## [0.2.0] - 2026-05-21

### Changed

- Centralized runtime version reporting across the package, CLI, health endpoint, and Ollama compatibility surface.
- Refreshed dependency baselines for FastAPI, OpenVINO, Hugging Face Hub, Optimum Intel, and test tooling.
- Updated the Hugging Face Hub workflow and examples to use the `hf` CLI.
- Added backend-neutral LLM runtime configuration, compile-cache controls, and richer health/runtime reporting.
- Added tokenizer-backed OpenAI chat prompt rendering with legacy fallback behavior.
- Expanded embedding runtime reporting so health surfaces can distinguish OpenVINO vs fallback behavior.
- Added a validated static-shape NPU embedding profile for `sentence-transformers/all-MiniLM-L6-v2` and aligned the download helper with runtime embedding cache paths.
- Added an alpha-gated `llama.cpp` GGUF scaffold for CPU-only, source-install experimentation.
- Added live hardware-certification tooling for validating a real NPU-backed `/api/generate` request.
- Refreshed packaging assets and repository documentation for the current 0.2.0 runtime layout.
- Documented release-truth validation results: NPU LLM certification succeeded, compile cache benefit was observed, the focused suite plus baseline/full pytest runs passed, `all-MiniLM-L6-v2` validated on NPU through the static-shape path, and `bge-small` NPU embedding still did not validate because the Intel NPU plugin failed with `check_sdpa_nodes(model)`.

## [0.1.0] - 2026-02-01

### Added

- OpenAI-compatible endpoints:
  - `/v1/chat/completions`
  - `/v1/embeddings`
  - `/v1/models`
- Ollama-compatible endpoints:
  - `/api/generate`
  - `/api/chat`
  - `/api/embed`
  - `/api/embeddings`
  - `/api/pull`
  - `/api/ps`
  - `/api/show`
  - `/api/version`
- Context-aware device routing with NPU → GPU → CPU fallback.
- Streaming response support backed by `AsyncTokenStream`.
- Prometheus metrics and request tracing headers.
- Initial Windows and Linux packaging assets.

[Unreleased]: https://github.com/MrFixit96/npu-proxy/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/MrFixit96/npu-proxy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/MrFixit96/npu-proxy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MrFixit96/npu-proxy/releases/tag/v0.1.0
