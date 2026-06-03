"""NPU Proxy FastAPI application bootstrap."""

from __future__ import annotations

from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse

from npu_proxy import __version__
from npu_proxy.api import chat, embeddings, health, metrics, models, ollama
from npu_proxy.config import (
    ProxyBootstrapConfig,
    activate_proxy_bootstrap_config,
    load_proxy_bootstrap_config,
    normalize_allowed_hosts,
    validate_port,
)


def _split_host_header(value: str) -> tuple[str, int | None]:
    host_header = value.strip()
    if not host_header:
        return "", None
    if host_header.startswith("["):
        closing = host_header.find("]")
        if closing == -1:
            return "", None
        host = host_header[1:closing].lower()
        suffix = host_header[closing + 1 :]
        if not suffix:
            return host, None
        if suffix.startswith(":") and suffix[1:].isdigit():
            try:
                return host, validate_port(int(suffix[1:]))
            except ValueError:
                return "", None
        return "", None
    if host_header.count(":") == 1:
        host, port = host_header.rsplit(":", 1)
        if port.isdigit():
            try:
                return host.lower().rstrip("."), validate_port(int(port))
            except ValueError:
                return "", None
    return host_header.lower().rstrip("."), None


def _allowed_host_entries(config: ProxyBootstrapConfig) -> set[tuple[str, int | None]]:
    entries: set[tuple[str, int | None]] = set()
    for allowed in normalize_allowed_hosts(config.allowed_hosts):
        host, port = _split_host_header(allowed)
        if host:
            entries.add((host, port))
            if port is None:
                entries.add((host, config.port))
    return entries


def _host_header_allowed(host_header: str, config: ProxyBootstrapConfig) -> bool:
    host, port = _split_host_header(host_header)
    if not host:
        return False
    return (host, port) in _allowed_host_entries(config)


def bootstrap_runtime(
    config: ProxyBootstrapConfig | None = None,
) -> ProxyBootstrapConfig:
    """Activate a new control-plane config and reset runtime singletons."""
    resolved = activate_proxy_bootstrap_config(config)

    from npu_proxy.inference.engine import reset_engine
    from npu_proxy.inference.llm_runtime import reset_llm_runtime

    reset_llm_runtime()
    reset_engine()
    return resolved


def create_app(config: ProxyBootstrapConfig | None = None) -> FastAPI:
    """Create the FastAPI application with an authoritative bootstrap config."""
    resolved = activate_proxy_bootstrap_config(config or load_proxy_bootstrap_config())

    application = FastAPI(
        title="NPU Proxy",
        description="Ollama-compatible API proxy for Intel NPU inference via OpenVINO",
        version=__version__,
    )
    application.state.proxy_config = resolved

    @application.middleware("http")
    async def enforce_host_allowlist(request: Request, call_next):
        host_header = request.headers.get("host", "")
        if not _host_header_allowed(host_header, resolved):
            return PlainTextResponse("Host header not allowed", status_code=421)
        return await call_next(request)

    application.include_router(health.router)
    application.include_router(models.router)
    application.include_router(chat.router)
    application.include_router(embeddings.router)
    application.include_router(ollama.router)
    application.include_router(metrics.router)
    return application


app = create_app()
