"""Context-aware routing for NPU inference"""
from .context_router import ContextRouter, RoutingResult, get_context_router, reset_context_router

__all__ = ["ContextRouter", "RoutingResult", "get_context_router", "reset_context_router"]
