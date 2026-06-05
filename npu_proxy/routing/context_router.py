"""Advisory context-aware routing recommendations.

This module estimates prompt size and returns a recommended device label. It
does not switch OpenVINO devices per request; the currently loaded runtime still
executes requests on the single configured device. The recommendation is used
for observability and future routing work only.

The token limit and preferred device come from the active
ProxyBootstrapConfig/LLMRuntimeConfig. Fallback recommendations follow the
central device fallback chain (NPU → GPU → CPU).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from npu_proxy.config import (
    DEFAULT_TOKEN_LIMIT,
    load_context_routing_config,
    normalize_context_device,
)
from npu_proxy.inference.devices import (
    DEVICE_FALLBACK_CHAIN,
    device_class,
    get_available_devices,
)
from npu_proxy.inference.tokenizer import count_tokens

# Default configuration values
DEFAULT_NPU_TOKEN_LIMIT: int = DEFAULT_TOKEN_LIMIT
"""Default maximum token count for NPU routing (conservative limit)."""

DEFAULT_PREFERRED_DEVICE: str = "NPU"
"""Default preferred device when token count is within limits."""

# Note: Actual fallback is determined dynamically based on available devices


@dataclass
class RoutingResult:
    """Result of an advisory device recommendation.

    Contains the recommended device label, the reason, and token count
    information used to make the recommendation.

    Attributes:
        device: The selected device identifier ("NPU", "GPU", or "CPU").
        reason: Human-readable reason for the recommendation.
            Values: "within_npu_limit", "prompt_exceeds_npu_limit".
        token_count: Number of tokens counted in the prompt/messages.

    Example:
        >>> result = RoutingResult(device="NPU", reason="within_npu_limit", token_count=150)
        >>> result.device
        'NPU'
    """

    device: str
    reason: str
    token_count: int


def _safe_device(device: str | None, *, default: str, field_name: str = "device") -> str:
    return normalize_context_device(device, default=default, field_name=field_name)


def get_npu_token_limit() -> int:
    """Return the advisory NPU token limit from centrally resolved config."""
    return load_context_routing_config().token_limit


def get_preferred_device() -> str:
    """Return the advisory preferred device from centrally resolved config."""
    return load_context_routing_config().preferred_device


def get_fallback_device(preferred_device: str | None = None) -> str:
    """Return the next available advisory fallback device.

    Explicit NPU_PROXY_FALLBACK_DEVICE takes precedence. This is a recommendation
    only; it does not change the active runtime device.
    """
    routing_config = load_context_routing_config()
    if routing_config.fallback_device:
        return routing_config.fallback_device

    preferred = _safe_device(
        preferred_device or routing_config.preferred_device,
        default=DEFAULT_PREFERRED_DEVICE,
        field_name="preferred_device",
    )
    available = {device_class(device) for device in get_available_devices() if device}

    try:
        chain_idx = DEVICE_FALLBACK_CHAIN.index(device_class(preferred))
    except ValueError:
        chain_idx = -1

    for device in DEVICE_FALLBACK_CHAIN[chain_idx + 1:]:
        if device in available:
            return device

    return "CPU"


class ContextRouter:
    """Recommends device labels based on context/token count thresholds.

    The ContextRouter implements advisory device selection based on prompt
    size. NPU devices have limited context windows, so prompts exceeding the
    configured limit are recommended for fallback devices (GPU or CPU).

    Device Fallback Chain:
        NPU → GPU → CPU

    Routing Logic:
        1. Count tokens in the input prompt or messages
        2. If count <= npu_limit: recommend preferred_device (default: NPU)
        3. If count > npu_limit: recommend fallback_device (default: GPU/CPU)

    Attributes:
        npu_limit: Maximum token count for recommending NPU.
        preferred_device: Device label to recommend when within token limit.
        fallback_device: Device label to recommend when token limit is exceeded.

    Example:
        >>> # Create router with default settings
        >>> router = ContextRouter()
        >>> router.npu_limit
        1800
        
        >>> # Route a short prompt to NPU
        >>> result = router.select_device("Hello, how are you?")
        >>> result.device
        'NPU'
        
        >>> # Long prompts recommend fallback
        >>> long_prompt = "word " * 2000
        >>> result = router.select_device(long_prompt)
        >>> result.device
        'GPU'  # or 'CPU' if GPU unavailable
    """

    def __init__(
        self,
        npu_limit: Optional[int] = None,
        preferred_device: Optional[str] = None,
        fallback_device: Optional[str] = None,
    ) -> None:
        """Initialize the ContextRouter with optional configuration overrides.

        Configuration values can be provided directly or read from the active
        centralized config. Direct parameters take precedence.

        Args:
            npu_limit: Maximum token count for NPU recommendation. If None,
                reads from the active ProxyBootstrapConfig.
            preferred_device: Recommended device when within token limit. If None,
                reads from the active LLMRuntimeConfig.
            fallback_device: Recommended device when limit is exceeded. If None,
                auto-detects from available hardware.

        Example:
            >>> # Use environment defaults
            >>> router = ContextRouter()
            
            >>> # Override specific settings
            >>> router = ContextRouter(npu_limit=2000, preferred_device="NPU")
            
            >>> # Force CPU fallback
            >>> router = ContextRouter(fallback_device="CPU")
        """
        routing_config = load_context_routing_config()
        self.npu_limit = npu_limit if npu_limit is not None else routing_config.token_limit
        self.preferred_device = _safe_device(
            preferred_device if preferred_device is not None else routing_config.preferred_device,
            default=DEFAULT_PREFERRED_DEVICE,
            field_name="preferred_device",
        )
        self.fallback_device = _safe_device(
            fallback_device
            if fallback_device is not None
            else (routing_config.fallback_device or get_fallback_device(self.preferred_device)),
            default="CPU",
            field_name="fallback_device",
        )
    
    def exceeds_npu_limit(self, prompt: str) -> bool:
        """Check if a prompt's token count exceeds the NPU limit.

        This is a convenience method for checking routing eligibility without
        performing the full device selection.

        Args:
            prompt: The text prompt to evaluate.

        Returns:
            True if the prompt token count exceeds npu_limit, False otherwise.
            Returns False for empty or whitespace-only prompts.

        Example:
            >>> router = ContextRouter(npu_limit=100)
            >>> router.exceeds_npu_limit("short prompt")
            False
            >>> router.exceeds_npu_limit("word " * 200)
            True
        """
        if not prompt or not prompt.strip():
            return False
        token_count = count_tokens(prompt)
        return token_count > self.npu_limit
    
    def select_device(self, prompt: str) -> RoutingResult:
        """Select the appropriate device for a single prompt.

        Counts tokens in the prompt and routes to the preferred device if
        within limits, or the fallback device if exceeded.

        Args:
            prompt: The text prompt to route.

        Returns:
            RoutingResult containing:
                - device: Selected device identifier ("NPU", "GPU", or "CPU")
                - reason: "within_npu_limit" or "prompt_exceeds_npu_limit"
                - token_count: Number of tokens in the prompt

        Example:
            >>> router = ContextRouter(npu_limit=100)
            >>> result = router.select_device("Hello world")
            >>> result.device, result.reason
            ('NPU', 'within_npu_limit')
            >>> result = router.select_device("word " * 200)
            >>> result.reason
            'prompt_exceeds_npu_limit'
        """
        token_count = count_tokens(prompt) if prompt and prompt.strip() else 0
        
        if token_count > self.npu_limit:
            return RoutingResult(
                device=self.fallback_device,
                reason="prompt_exceeds_npu_limit",
                token_count=token_count,
            )
        
        return RoutingResult(
            device=self.preferred_device,
            reason="within_npu_limit",
            token_count=token_count,
        )
    
    def select_device_for_messages(
        self, messages: List[Dict[str, str]]
    ) -> RoutingResult:
        """Select the appropriate device for a list of chat messages.

        Counts tokens across all messages in the conversation and routes
        based on the total. This is useful for chat-style APIs where context
        includes multiple messages (system, user, assistant turns).

        Args:
            messages: List of message dictionaries with 'content' key.
                Expected format: [{"role": "user", "content": "Hello"}, ...]

        Returns:
            RoutingResult containing:
                - device: Selected device identifier ("NPU", "GPU", or "CPU")
                - reason: "within_npu_limit" or "prompt_exceeds_npu_limit"
                - token_count: Total tokens across all messages

        Example:
            >>> router = ContextRouter(npu_limit=100)
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"},
            ... ]
            >>> result = router.select_device_for_messages(messages)
            >>> result.device
            'NPU'
        """
        total_tokens = sum(
            count_tokens(msg.get("content", ""))
            for msg in messages
        )
        
        if total_tokens > self.npu_limit:
            return RoutingResult(
                device=self.fallback_device,
                reason="prompt_exceeds_npu_limit",
                token_count=total_tokens,
            )
        
        return RoutingResult(
            device=self.preferred_device,
            reason="within_npu_limit",
            token_count=total_tokens,
        )


# Singleton pattern for shared router instance
_context_router: Optional[ContextRouter] = None


def get_context_router() -> ContextRouter:
    """Get or create the singleton ContextRouter instance.

    This function provides a shared router instance configured from the active
    centralized config. Use this for application-wide advisory recommendations.

    The singleton is created on first access and reused for subsequent calls.
    Use reset_context_router() to force re-creation (e.g., for testing).

    Returns:
        The shared ContextRouter instance.

    Example:
        >>> router = get_context_router()
        >>> same_router = get_context_router()
        >>> router is same_router
        True
    """
    global _context_router
    if _context_router is None:
        _context_router = ContextRouter()
    return _context_router


def reset_context_router() -> None:
    """Reset the router singleton to force re-creation.

    Clears the cached singleton instance so that the next call to
    get_context_router() creates a fresh instance. This is primarily useful for
    tests that activate a different ProxyBootstrapConfig.
    """
    global _context_router
    _context_router = None
