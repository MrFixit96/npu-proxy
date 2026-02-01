"""NPU Proxy FastAPI Application.

This module defines the main FastAPI application instance and registers all
API routers. It serves as the ASGI entry point for uvicorn and other ASGI
servers.

The application provides an Ollama-compatible API for Intel NPU inference
via OpenVINO. It acts as a drop-in replacement for Ollama, routing inference
requests to Intel NPU hardware when available.

Application Structure:
    The FastAPI app is composed of modular routers, each handling a specific
    API domain:

    - health: Health check and readiness endpoints (/health, /ready)
    - models: Model listing and information (/api/tags, /api/show)
    - chat: Chat completion endpoints (/api/chat, /v1/chat/completions)
    - embeddings: Text embedding generation (/api/embeddings)
    - ollama: Ollama-specific compatibility endpoints
    - metrics: Prometheus metrics and observability (/metrics)

Router Registration Order:
    Routers are registered in dependency order - health and models first
    (no dependencies), then inference routers (chat, embeddings), and
    finally Ollama compatibility and metrics routers.

Usage:
    # Direct uvicorn invocation
    $ uvicorn npu_proxy.main:app --host 0.0.0.0 --port 8080

    # With the CLI (recommended)
    $ npu-proxy --port 8080

    # Programmatic access
    >>> from npu_proxy.main import app
    >>> # Use with TestClient for testing
    >>> from fastapi.testclient import TestClient
    >>> client = TestClient(app)
    >>> response = client.get("/health")

Example:
    >>> from npu_proxy.main import app
    >>> app.title
    'NPU Proxy'
    >>> len(app.routes) > 0
    True
"""
from fastapi import FastAPI
from npu_proxy.api import health, models, chat, embeddings, ollama, metrics

# Create the main FastAPI application instance.
# This is the ASGI app that uvicorn serves.
app = FastAPI(
    title="NPU Proxy",
    description="Ollama-compatible API proxy for Intel NPU inference via OpenVINO",
    version="0.1.0",
)

# Register API routers in dependency order.
# Each router handles a specific API domain with its own endpoints.

# Health and readiness endpoints for load balancers and orchestrators
app.include_router(health.router)

# Model listing and information (Ollama /api/tags compatibility)
app.include_router(models.router)

# Chat completion endpoints (Ollama and OpenAI compatible)
app.include_router(chat.router)

# Text embedding generation endpoints
app.include_router(embeddings.router)

# Ollama-specific compatibility endpoints (version, ps, etc.)
app.include_router(ollama.router)

# Prometheus metrics for observability
app.include_router(metrics.router)
