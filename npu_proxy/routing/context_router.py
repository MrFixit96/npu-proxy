"""Context-aware routing for NPU inference.

This module provides intelligent request routing based on prompt token count.
NPU devices have limited context windows (~1800 tokens practical max), so
prompts exceeding this limit are routed to fallback devices.

Device Fallback Chain:
    NPU → GPU → CPU

The router checks available devices and selects the next in chain when:
    1. Prompt token count exceeds NPU_PROXY_TOKEN_LIMIT
    2. Preferred device is unavailable

Environment Variables:
    NPU_PROXY_TOKEN_LIMIT: Maximum tokens for NPU routing (default: 1800).
        Prompts exceeding this count are routed to fallback devices.
    NPU_PROXY_PREFERRED_DEVICE: Primary device to use when within limits
        (default: "NPU"). Must be one of: NPU, GPU, CPU.
    NPU_PROXY_FALLBACK_DEVICE: Override automatic fallback selection
        (default: auto-detect based on available hardware).

Routing Logic:
    1. Count tokens in the prompt/messages
    2. If token_count <= NPU_PROXY_TOKEN_LIMIT: use preferred device
    3. If token_count > NPU_PROXY_TOKEN_LIMIT: use fallback device
    4. Fallback is auto-detected from available devices following chain order

Example:
    >>> router = ContextRouter(npu_limit=1800)
    >>> result = router.select_device("Hello world")
    >>> result.device
    'NPU'
    >>> result = router.select_device("word " * 2000)
    >>> result.device
    'GPU'  # or 'CPU' if GPU unavailable

Note:
    The 1800 token default is conservative. Intel NPUs can handle ~2048 tokens
    but performance degrades near the limit. The default provides headroom for
    system prompts and response generation.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

from npu_proxy.inference.tokenizer import count_tokens
from npu_proxy.inference.engine import DEVICE_FALLBACK_CHAIN, get_available_devices

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_NPU_TOKEN_LIMIT: int = 1800
"""Default maximum token count for NPU routing (conservative limit)."""

DEFAULT_PREFERRED_DEVICE: str = "NPU"
"""Default preferred device when token count is within limits."""

# Note: Actual fallback is determined dynamically based on available devices


@dataclass
class RoutingResult:
    """Result of a device routing decision.

    Contains the selected device, the reason for selection, and token count
    information used to make the routing decision.

    Attributes:
        device: The selected device identifier ("NPU", "GPU", or "CPU").
        reason: Human-readable reason for the routing decision.
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


def get_npu_token_limit() -> int:
    """Get the NPU token limit from environment variable or default.

    Reads the NPU_PROXY_TOKEN_LIMIT environment variable and returns its
    integer value. Falls back to DEFAULT_NPU_TOKEN_LIMIT (1800) if the
    variable is unset or contains an invalid (non-integer) value.

    Returns:
        The maximum token count allowed for NPU routing.

    Environment Variables:
        NPU_PROXY_TOKEN_LIMIT: Integer value for token limit.

    Example:
        >>> os.environ["NPU_PROXY_TOKEN_LIMIT"] = "2000"
        >>> get_npu_token_limit()
        2000
        >>> del os.environ["NPU_PROXY_TOKEN_LIMIT"]
        >>> get_npu_token_limit()
        1800
    """
    try:
        return int(os.environ.get("NPU_PROXY_TOKEN_LIMIT", DEFAULT_NPU_TOKEN_LIMIT))
    except ValueError:
        logger.warning(f"Invalid NPU_PROXY_TOKEN_LIMIT, using default: {DEFAULT_NPU_TOKEN_LIMIT}")
        return DEFAULT_NPU_TOKEN_LIMIT


def get_preferred_device() -> str:
    """Get the preferred device from environment variable or default.

    Reads the NPU_PROXY_PREFERRED_DEVICE environment variable. Falls back
    to DEFAULT_PREFERRED_DEVICE ("NPU") if the variable is unset.

    Returns:
        The preferred device identifier (e.g., "NPU", "GPU", "CPU").

    Environment Variables:
        NPU_PROXY_PREFERRED_DEVICE: Device identifier string.

    Example:
        >>> os.environ["NPU_PROXY_PREFERRED_DEVICE"] = "GPU"
        >>> get_preferred_device()
        'GPU'
    """
    return os.environ.get("NPU_PROXY_PREFERRED_DEVICE", DEFAULT_PREFERRED_DEVICE)


def get_fallback_device() -> str:
    """Get the fallback device from environment or auto-detect from available hardware.

    Determines the fallback device to use when prompts exceed the NPU token limit.
    First checks for an explicit override via environment variable, then
    auto-detects based on available devices following the fallback chain order.

    Device Fallback Chain:
        NPU → GPU → CPU

    The function finds the preferred device's position in the chain and returns
    the next available device. CPU is always available as the final fallback.

    Returns:
        The fallback device identifier (e.g., "GPU", "CPU").

    Environment Variables:
        NPU_PROXY_FALLBACK_DEVICE: Override automatic detection.
            If set, this value is returned (uppercased) regardless of
            device availability.

    Example:
        >>> # With NPU as preferred device and GPU available:
        >>> get_fallback_device()
        'GPU'
        >>> # With explicit override:
        >>> os.environ["NPU_PROXY_FALLBACK_DEVICE"] = "cpu"
        >>> get_fallback_device()
        'CPU'
    """
    # Check for explicit override
    env_fallback = os.environ.get("NPU_PROXY_FALLBACK_DEVICE")
    if env_fallback:
        return env_fallback.upper()
    
    # Determine fallback based on available devices and fallback chain
    preferred = get_preferred_device()
    available = get_available_devices()
    
    # Find position of preferred device in chain
    try:
        chain_idx = DEVICE_FALLBACK_CHAIN.index(preferred)
    except ValueError:
        chain_idx = -1
    
    # Return next available device in chain
    for device in DEVICE_FALLBACK_CHAIN[chain_idx + 1:]:
        if device in available:
            return device
    
    # Default to CPU (always available)
    return "CPU"


class ContextRouter:
    """Routes inference requests based on context/token count thresholds.

    The ContextRouter implements intelligent device selection based on prompt
    size. NPU devices have limited context windows, so prompts exceeding the
    configured limit are automatically routed to fallback devices (GPU or CPU).

    Device Fallback Chain:
        NPU → GPU → CPU

    Routing Logic:
        1. Count tokens in the input prompt or messages
        2. If count <= npu_limit: route to preferred_device (default: NPU)
        3. If count > npu_limit: route to fallback_device (default: GPU/CPU)

    Attributes:
        npu_limit: Maximum token count for NPU routing.
        preferred_device: Device to use when within token limit.
        fallback_device: Device to use when token limit is exceeded.

    Example:
        >>> # Create router with default settings
        >>> router = ContextRouter()
        >>> router.npu_limit
        1800
        
        >>> # Route a short prompt to NPU
        >>> result = router.select_device("Hello, how are you?")
        >>> result.device
        'NPU'
        
        >>> # Long prompts route to fallback
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

        Configuration values can be provided directly or read from environment
        variables. Direct parameters take precedence over environment variables.

        Args:
            npu_limit: Maximum token count for NPU routing. If None, reads from
                NPU_PROXY_TOKEN_LIMIT environment variable or uses default (1800).
            preferred_device: Device to use when within token limit. If None,
                reads from NPU_PROXY_PREFERRED_DEVICE or uses default ("NPU").
            fallback_device: Device to use when limit is exceeded. If None,
                reads from NPU_PROXY_FALLBACK_DEVICE or auto-detects from
                available hardware.

        Example:
            >>> # Use environment defaults
            >>> router = ContextRouter()
            
            >>> # Override specific settings
            >>> router = ContextRouter(npu_limit=2000, preferred_device="NPU")
            
            >>> # Force CPU fallback
            >>> router = ContextRouter(fallback_device="CPU")
        """
        self.npu_limit = npu_limit if npu_limit is not None else get_npu_token_limit()
        self.preferred_device = preferred_device if preferred_device is not None else get_preferred_device()
        self.fallback_device = fallback_device if fallback_device is not None else get_fallback_device()
    
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

    This function provides a shared router instance configured from environment
    variables. Use this for application-wide routing to ensure consistent
    configuration and avoid repeated environment variable lookups.

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
    get_context_router() creates a fresh instance. This is primarily
    useful for testing scenarios where environment variables are modified.

    Example:
        >>> os.environ["NPU_PROXY_TOKEN_LIMIT"] = "1000"
        >>> reset_context_router()
        >>> router = get_context_router()
        >>> router.npu_limit
        1000
    """
    global _context_router
    _context_router = None
