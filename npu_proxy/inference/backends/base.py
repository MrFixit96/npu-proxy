"""Shared abstractions for pluggable LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator

from npu_proxy.config import LLMBackend

TokenCallback = Callable[[str], bool]
AbortCallback = Callable[[], bool]


class LLMBackendError(RuntimeError):
    """Base error for backend selection and execution failures."""


class BackendConfigurationError(LLMBackendError, ValueError):
    """Raised when a backend is selected with invalid configuration."""


class BackendDependencyError(LLMBackendError):
    """Raised when an optional backend dependency is unavailable."""


class BaseLLMBackend(ABC):
    """Backend-neutral contract for LLM execution engines.

    Backend instances may wrap native model objects that are not safe for
    concurrent use. Implementations must either serialize generate and
    generate_stream access to shared native state or fail fast while busy.
    """

    backend: LLMBackend

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier."""

    @property
    @abstractmethod
    def requested_device(self) -> str:
        """Return the requested compute device."""

    @property
    @abstractmethod
    def actual_device(self) -> str:
        """Return the effective compute device."""

    @property
    def is_warmed_up(self) -> bool:
        """Return warmup state when the backend supports warmup."""
        return False

    def warmup(self, warmup_tokens: int = 16) -> None:
        """Warm up the backend when supported."""
        return None

    @abstractmethod
    def get_device_info(self) -> dict[str, object]:
        """Return backend/device metadata for health surfaces."""

    def shutdown(self) -> None:
        """Release native resources owned by this backend."""
        return None

    def close(self) -> None:
        """Alias for shutdown() for resource-management callers."""
        self.shutdown()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int | None = None,
    ) -> str:
        """Generate a full text response."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        streamer_callback: TokenCallback | None = None,
        abort_callback: AbortCallback | None = None,
        timeout: int | None = None,
    ) -> Iterator[str]:
        """Generate a streaming text response."""
