"""OpenVINO GenAI inference engine wrapper for NPU-accelerated text generation.

This module provides a thread-safe wrapper around the OpenVINO GenAI LLMPipeline
for running large language model inference on Intel NPU, GPU, or CPU devices.
It includes automatic device fallback, timeout handling, streaming support,
and NPU warmup capabilities.

Typical usage:
    >>> from npu_proxy.inference.engine import get_llm_engine
    >>> engine = get_llm_engine()
    >>> engine.warmup()  # Pre-compile NPU pipeline
    >>> response = engine.generate("Hello, world!")

Thread Safety:
    The module uses a singleton pattern with thread-safe initialization via
    double-checked locking. Multiple threads can safely call get_llm_engine()
    and share the same InferenceEngine instance. Individual inference calls
    are NOT thread-safe - callers should serialize access or use separate
    engine instances for concurrent inference.

Environment Variables:
    NPU_PROXY_INFERENCE_TIMEOUT: Default timeout in seconds (default: 180)
    NPU_PROXY_MAX_PROMPT_LEN: Maximum prompt length in tokens (default: 4096)
    NPU_PROXY_DEVICE: Default device selection (default: NPU)
"""

import importlib
import logging
import threading
import concurrent.futures
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Callable, Optional, Any, Literal

from npu_proxy.config import (
    DEFAULT_INFERENCE_TIMEOUT as CONFIG_DEFAULT_INFERENCE_TIMEOUT,
    DEFAULT_LLM_MODEL as CONFIG_DEFAULT_LLM_MODEL,
    DEFAULT_MAX_PROMPT_LEN as CONFIG_DEFAULT_MAX_PROMPT_LEN,
    DEFAULT_MODEL_DIR as CONFIG_DEFAULT_MODEL_DIR,
    LLMBackend,
    LLMRuntimeConfig,
    get_active_llm_runtime_config,
    normalize_compile_cache_mode,
    normalize_prefix_cache_mode,
)
from npu_proxy.inference.devices import (
    DEVICE_FALLBACK_CHAIN,
    FallbackReason,
    device_class,
    get_available_devices,
    normalize_device,
)

from npu_proxy.metrics import (
    record_error,
    record_inference,
    record_model_load_time,
    record_runtime_feature_degradation,
    record_runtime_feature_state,
    record_tokens_per_second,
    record_tpot,
    record_ttft,
)

logger = logging.getLogger(__name__)

FinishReason = Literal["stop", "length"]


def _normalize_finish_reason(reason: Any) -> FinishReason | None:
    if reason is None:
        return None
    normalized = str(reason).strip().lower()
    if not normalized:
        return None
    if any(marker in normalized for marker in ("length", "max", "limit")):
        return "length"
    if any(marker in normalized for marker in ("stop", "eos", "end")):
        return "stop"
    return None


class _OpenVINOGenAIProxy:
    """Lazily import openvino_genai only when the OpenVINO backend is used."""

    _module: Any | None = None
    _lock = threading.Lock()

    def _load(self) -> Any:
        if self._module is None:
            with self._lock:
                if self._module is None:
                    self._module = importlib.import_module("openvino_genai")
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


ov_genai = _OpenVINOGenAIProxy()


# =============================================================================
# Exception Classes
# =============================================================================


class InferenceError(Exception):
    """Base exception for inference-related errors.
    
    This exception serves as the base class for all inference errors,
    enabling consistent error handling and HTTP status code mapping.
    
    Attributes:
        message: Human-readable error description.
        status_code: Suggested HTTP status code for API responses.
    
    Example:
        >>> try:
        ...     engine.generate("test")
        ... except InferenceError as e:
        ...     return Response(str(e), status=e.status_code)
    """
    
    def __init__(self, message: str, status_code: int = 500) -> None:
        """Initialize InferenceError.
        
        Args:
            message: Human-readable error description.
            status_code: HTTP status code for API error responses.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InferenceTimeoutError(InferenceError, TimeoutError):
    """Raised when inference exceeds the configured timeout.
    
    NPU inference can take significant time, especially on first run
    when pipeline compilation occurs. This error indicates the operation
    exceeded the allowed time limit.
    
    Attributes:
        timeout_seconds: The timeout value that was exceeded.
        status_code: Always 504 (Gateway Timeout).
    
    Example:
        >>> try:
        ...     engine.generate("test", timeout=30)
        ... except InferenceTimeoutError as e:
        ...     logger.error(f"Timed out after {e.timeout_seconds}s")
    """
    
    def __init__(
        self,
        timeout_seconds: int,
        message: Optional[str] = None,
    ) -> None:
        """Initialize InferenceTimeoutError.
        
        Args:
            timeout_seconds: The timeout duration that was exceeded.
        """
        super().__init__(
            message or f"Inference timed out after {timeout_seconds} seconds",
            status_code=504
        )
        self.timeout_seconds = timeout_seconds


class DeviceBusyError(InferenceError):
    """Raised when a routed device cannot accept work before queue timeout."""

    def __init__(self, device: str) -> None:
        super().__init__(
            f"Device {device} is busy; retry shortly",
            status_code=503,
        )
        self.device = device
        self.reason = "device_busy"


class DeviceError(InferenceError):
    """Raised when the requested device is unavailable or fails.
    
    This error occurs when a specific accelerator device (NPU, GPU)
    cannot be used, either because it's not present or encountered
    an error during model loading.
    
    Attributes:
        device: The device that failed.
        available_devices: List of devices that are available.
        status_code: Always 503 (Service Unavailable).
    
    Example:
        >>> try:
        ...     engine = InferenceEngine(model_path, device="NPU")
        ... except DeviceError as e:
        ...     logger.warning(f"NPU unavailable, options: {e.available_devices}")
    """
    
    def __init__(
        self,
        device: str,
        available_devices: list[str],
        original_error: Optional[Exception] = None
    ) -> None:
        """Initialize DeviceError.
        
        Args:
            device: The device that was requested but failed.
            available_devices: List of currently available devices.
            original_error: The underlying exception, if any.
        """
        message = f"Device '{device}' unavailable or failed"
        if original_error:
            message += f": {original_error}"
        super().__init__(message, status_code=503)
        self.device = device
        self.available_devices = available_devices
        self.original_error = original_error


class ModelNotLoadedError(InferenceError):
    """Raised when inference is attempted before model loading.
    
    This error indicates that generate() or generate_stream() was called
    on an InferenceEngine instance before the model was successfully loaded.
    
    Attributes:
        status_code: Always 500 (Internal Server Error).
    """
    
    def __init__(self) -> None:
        """Initialize ModelNotLoadedError."""
        super().__init__("Model not loaded", status_code=500)


class ModelNotFoundError(InferenceError):
    """Raised when the model files cannot be found on disk.
    
    This error occurs during engine initialization when the specified
    model path does not exist or is not a valid model directory.
    
    Attributes:
        model_path: The path that was searched for the model.
        status_code: Always 404 (Not Found).
    """
    
    def __init__(self, model_path: Path) -> None:
        """Initialize ModelNotFoundError.
        
        Args:
            model_path: The filesystem path where the model was expected.
        """
        super().__init__(
            f"Model not found at {model_path}. Download with: "
            f"huggingface_hub.snapshot_download('OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov')",
            status_code=404
        )
        self.model_path = model_path


# =============================================================================
# Constants and Configuration
# =============================================================================


DEFAULT_MODEL_DIR: Path = CONFIG_DEFAULT_MODEL_DIR
DEFAULT_LLM_MODEL: str = CONFIG_DEFAULT_LLM_MODEL
DEFAULT_INFERENCE_TIMEOUT: int = CONFIG_DEFAULT_INFERENCE_TIMEOUT
DEFAULT_MAX_PROMPT_LEN: int = CONFIG_DEFAULT_MAX_PROMPT_LEN


# =============================================================================
# Device Selection Functions
# =============================================================================


def select_best_device(preferred: Optional[str] = None) -> tuple[str | None, Optional[str]]:
    """Select the best available compute device with automatic fallback.
    
    Implements a device selection strategy that respects user preference
    while providing automatic fallback to ensure inference can always run.
    The fallback chain is: NPU → GPU → CPU.
    
    Args:
        preferred: User-requested device (case-insensitive). If None or
            unavailable, the best available device is selected automatically.
    
    Returns:
        A tuple of (selected_device, fallback_device). selected_device is
        None when a custom preferred device is unavailable and has no known
        fallback chain. The fallback_device is None if no fallback is available.
    
    Example:
        >>> device, fallback = select_best_device("NPU")
        >>> print(f"Primary: {device}, Fallback: {fallback}")
        Primary: NPU, Fallback: GPU
        
        >>> device, fallback = select_best_device("GPU")
        >>> print(f"Primary: {device}, Fallback: {fallback}")
        Primary: GPU, Fallback: CPU
    
    Note:
        If the preferred device is not in DEVICE_FALLBACK_CHAIN, no
        fallback will be configured. This allows for custom device names
        while still supporting the standard fallback behavior.
    """
    available = get_available_devices()
    available_classes = {device_class(d) for d in available if d}
    logger.info(f"Available devices: {available}")
    
    # If user specified a device, try to use it
    if preferred:
        preferred = preferred.upper()
        preferred_is_chain_class = preferred in DEVICE_FALLBACK_CHAIN
        # A bare class like "GPU" must match OpenVINO's enumerated "GPU.0"/"GPU.1".
        if preferred in available or (preferred_is_chain_class and preferred in available_classes):
            # Find fallback from chain
            fallback: Optional[str] = None
            if preferred_is_chain_class:
                chain_idx = DEVICE_FALLBACK_CHAIN.index(preferred)
                for device in DEVICE_FALLBACK_CHAIN[chain_idx + 1:]:
                    if device in available_classes:
                        fallback = device
                        break
            return preferred, fallback
        if not preferred_is_chain_class:
            logger.warning(
                "Requested custom device %s is unavailable and has no fallback chain",
                preferred,
            )
            return None, None
        logger.warning(
            f"Requested device {preferred} not available, "
            "selecting best alternative"
        )
    
    # Select best available from fallback chain
    for device in DEVICE_FALLBACK_CHAIN:
        if device in available_classes:
            # Find fallback
            fallback = None
            chain_idx = DEVICE_FALLBACK_CHAIN.index(device)
            for fb_device in DEVICE_FALLBACK_CHAIN[chain_idx + 1:]:
                if fb_device in available_classes:
                    fallback = fb_device
                    break
            return device, fallback
    
    # Default to CPU (always available)
    return "CPU", None


# =============================================================================
# Inference Engine Class
# =============================================================================


class InferenceEngine:
    """Thread-safe wrapper for OpenVINO GenAI LLMPipeline.
    
    Provides a high-level interface for running LLM inference on Intel
    accelerators (NPU, GPU, CPU) with automatic device fallback, timeout
    handling, and streaming support.
    
    The engine handles device selection, model loading, and provides both
    blocking and streaming generation methods. It includes NPU-specific
    optimizations like warmup to eliminate cold-start latency.
    
    Attributes:
        model_path: Path to the OpenVINO model directory.
        requested_device: Originally requested device.
        device: Primary device selected for inference.
        fallback_device: Backup device if primary fails.
        actual_device: Device currently in use (may differ after fallback).
        model_name: Name of the loaded model (directory name).
        used_fallback: True if fallback device was activated.
        pipeline: The underlying OpenVINO GenAI LLMPipeline instance.
    
    Example:
        >>> engine = InferenceEngine("/path/to/model", device="NPU")
        >>> engine.warmup()  # Pre-compile NPU pipeline
        >>> 
        >>> # Blocking generation
        >>> response = engine.generate("What is 2+2?")
        >>> 
        >>> # Streaming generation
        >>> for token in engine.generate_stream("Tell me a story"):
        ...     print(token, end="", flush=True)
    
    Thread Safety:
        The engine instance is safe to share across threads, but individual
        inference calls should be serialized. The underlying LLMPipeline
        does not support concurrent inference on the same instance.
    
    Raises:
        DeviceError: If the model cannot be loaded on any device.
        ModelNotFoundError: If the model path does not exist.
    """
    
    def __init__(
        self,
        model_path: str | Path,
        device: str = "NPU",
        inference_timeout: int = DEFAULT_INFERENCE_TIMEOUT,
        max_prompt_len: int = DEFAULT_MAX_PROMPT_LEN,
        compile_cache_dir: Optional[str | Path] = None,
        compile_cache_mode: Optional[str] = None,
        prefix_cache_mode: str = "auto",
    ) -> None:
        """Initialize the inference engine and load the model.
        
        Args:
            model_path: Path to directory containing OpenVINO model files.
                Must contain openvino_model.xml and associated files.
            device: Target compute device. Case-insensitive.
                Options: "NPU", "GPU", "CPU". Defaults to "NPU".
        
        Raises:
            DeviceError: If model loading fails on all available devices.
            FileNotFoundError: If model_path does not exist.
        
        Example:
            >>> engine = InferenceEngine(
            ...     model_path="~/.cache/models/tinyllama",
            ...     device="NPU"
            ... )
        """
        self.model_path: Path = Path(model_path)
        self.requested_device: str = device.upper()
        self.inference_timeout: int = int(inference_timeout)
        if self.inference_timeout <= 0:
            raise ValueError("inference_timeout must be greater than zero")
        self.max_prompt_len: int = int(max_prompt_len)
        if self.max_prompt_len <= 0:
            raise ValueError("max_prompt_len must be greater than zero")
        self.compile_cache_dir: Optional[Path] = (
            Path(compile_cache_dir) if compile_cache_dir else None
        )
        self.compile_cache_mode: Optional[str] = normalize_compile_cache_mode(
            compile_cache_mode
        )
        self.prefix_cache_mode: str = normalize_prefix_cache_mode(prefix_cache_mode)
        self.device: str
        self.fallback_device: Optional[str]
        selected_device, self.fallback_device = select_best_device(self.requested_device)
        if selected_device is None:
            raise DeviceError(self.requested_device, get_available_devices())
        self.device = selected_device
        self.actual_device: str = self.device  # Updated if fallback occurs
        self.pipeline: Optional[Any] = None
        self.model_name: str = self.model_path.name
        self.used_fallback: bool = False
        self._is_warmed_up: bool = False
        self._warmup_lock: threading.Lock = threading.Lock()
        self._load_diagnostics: list[dict[str, Any]] = []
        self._runtime_features: dict[str, Any] = {
            "compile_cache_enabled": False,
            "prefix_cache_enabled": None,
            "degraded_features": [],
        }
        self._last_generation_stats: Optional[dict[str, Any]] = None
        self._last_finish_reason: FinishReason | None = None
        self._model_load_seconds: Optional[float] = None
        self._inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="npu-proxy-inference",
        )
        self._inference_lock = threading.Lock()
        self._active_future: Optional[concurrent.futures.Future[Any]] = None
        self._active_timeout_seconds: Optional[int] = None
        self._active_operation: Optional[str] = None
        self._load_model()

    def _ensure_compile_cache_dir(self) -> Optional[Path]:
        """Create compile cache directory if configured."""
        if self.compile_cache_dir is None:
            return None
        try:
            self.compile_cache_dir.mkdir(parents=True, exist_ok=True)
            return self.compile_cache_dir
        except OSError as exc:
            logger.warning(
                "Compile cache directory unavailable (%s): %s",
                self.compile_cache_dir,
                exc,
            )
            self._load_diagnostics.append(
                {
                    "device": self.actual_device,
                    "status": "compile_cache_dir_unavailable",
                    "compile_cache_dir": str(self.compile_cache_dir),
                    "error": str(exc),
                }
            )
            return None

    def _base_pipeline_config(self, device: str) -> dict[str, object]:
        """Build base LLMPipeline configuration for a device."""
        config: dict[str, object] = {}
        if device == "NPU":
            config["MAX_PROMPT_LEN"] = self.max_prompt_len
        return config

    def _build_pipeline_attempts(self, device: str) -> list[dict[str, Any]]:
        """Build runtime configuration attempts with safe degradation."""
        base_config = self._base_pipeline_config(device)
        compile_cache_config: dict[str, object] = {}
        cache_dir = self._ensure_compile_cache_dir()
        if cache_dir is not None:
            compile_cache_config["CACHE_DIR"] = str(cache_dir)
            if self.compile_cache_mode is not None:
                compile_cache_config["CACHE_MODE"] = self.compile_cache_mode

        prefix_cache_config: dict[str, object] = {}
        if self.prefix_cache_mode in {"on", "off"}:
            if device == "NPU":
                prefix_cache_config["NPUW_LLM_ENABLE_PREFIX_CACHING"] = (
                    "YES" if self.prefix_cache_mode == "on" else "NO"
                )
            else:
                logger.warning(
                    "Prefix cache mode %s requested on %s; skipping because it is only "
                    "configured for NPU",
                    self.prefix_cache_mode,
                    device,
                )
                record_runtime_feature_degradation(
                    self.model_name,
                    device.lower(),
                    "prefix_cache",
                    "device_not_npu",
                )

        attempts: list[dict[str, Any]] = []

        def add_attempt(
            config: dict[str, object],
            compile_cache_enabled: bool,
            prefix_cache_enabled: Optional[bool],
            degraded: list[str],
        ) -> None:
            signature = tuple(sorted((key, str(value)) for key, value in config.items()))
            if any(existing["signature"] == signature for existing in attempts):
                return
            attempts.append(
                {
                    "config": config,
                    "compile_cache_enabled": compile_cache_enabled,
                    "prefix_cache_enabled": prefix_cache_enabled,
                    "degraded_features": degraded,
                    "signature": signature,
                }
            )

        full_config = {
            **base_config,
            **compile_cache_config,
            **prefix_cache_config,
        }
        add_attempt(
            full_config,
            compile_cache_enabled=bool(compile_cache_config),
            prefix_cache_enabled=(
                True if prefix_cache_config.get("NPUW_LLM_ENABLE_PREFIX_CACHING") == "YES"
                else False if "NPUW_LLM_ENABLE_PREFIX_CACHING" in prefix_cache_config
                else None
            ),
            degraded=[],
        )

        if prefix_cache_config:
            add_attempt(
                {**base_config, **compile_cache_config},
                compile_cache_enabled=bool(compile_cache_config),
                prefix_cache_enabled=None,
                degraded=["prefix_cache"],
            )

        if compile_cache_config:
            add_attempt(
                {**base_config, **prefix_cache_config},
                compile_cache_enabled=False,
                prefix_cache_enabled=(
                    True if prefix_cache_config.get("NPUW_LLM_ENABLE_PREFIX_CACHING") == "YES"
                    else False if "NPUW_LLM_ENABLE_PREFIX_CACHING" in prefix_cache_config
                    else None
                ),
                degraded=["compile_cache"],
            )

        if prefix_cache_config and compile_cache_config:
            add_attempt(
                dict(base_config),
                compile_cache_enabled=False,
                prefix_cache_enabled=None,
                degraded=["prefix_cache", "compile_cache"],
            )

        return attempts
    
    def _load_model(self) -> None:
        """Load the model with automatic device fallback.
        
        Attempts to load the model on the primary device, falling back
        through the device chain (NPU → GPU → CPU) if loading fails.
        
        Raises:
            DeviceError: If model loading fails on all devices.
        
        Note:
            This method is called automatically during __init__.
            NPU device receives special configuration for MAX_PROMPT_LEN.
        """
        devices_to_try = [self.device]
        if self.fallback_device:
            devices_to_try.append(self.fallback_device)
        if "CPU" not in devices_to_try:
            devices_to_try.append("CPU")

        load_started_at = time.perf_counter()
        last_error: Optional[Exception] = None

        for device in devices_to_try:
            attempts = self._build_pipeline_attempts(device)
            for attempt_index, attempt in enumerate(attempts, start=1):
                config = dict(attempt["config"])
                try:
                    logger.info(
                        "Loading model from %s on %s (attempt %s/%s)",
                        self.model_path,
                        device,
                        attempt_index,
                        len(attempts),
                    )
                    if config:
                        logger.info("Runtime config for %s: %s", device, config)
                        self.pipeline = ov_genai.LLMPipeline(
                            str(self.model_path),
                            device,
                            config,
                        )
                    else:
                        self.pipeline = ov_genai.LLMPipeline(str(self.model_path), device)

                    self.actual_device = device
                    self.used_fallback = (device != self.device)
                    self._model_load_seconds = time.perf_counter() - load_started_at
                    degraded_features = list(attempt["degraded_features"])
                    prefix_cache_enabled = attempt["prefix_cache_enabled"]
                    if self.prefix_cache_mode in {"on", "off"} and device != "NPU":
                        if "prefix_cache" not in degraded_features:
                            degraded_features.append("prefix_cache")
                        prefix_cache_enabled = False
                    self._runtime_features = {
                        "compile_cache_enabled": attempt["compile_cache_enabled"],
                        "prefix_cache_enabled": prefix_cache_enabled,
                        "degraded_features": degraded_features,
                    }
                    self._load_diagnostics.append(
                        {
                            "device": device,
                            "status": "loaded",
                            "config": config,
                            "degraded_features": degraded_features,
                        }
                    )

                    if self.used_fallback:
                        logger.warning(
                            "Using fallback device %s instead of %s",
                            device,
                            self.device,
                        )
                    else:
                        logger.info("Model loaded successfully on %s", device)

                    record_model_load_time(self.model_name, self._model_load_seconds)
                    record_runtime_feature_state(
                        self.model_name,
                        device.lower(),
                        "compile_cache",
                        bool(attempt["compile_cache_enabled"]),
                    )
                    if prefix_cache_enabled is not None:
                        record_runtime_feature_state(
                            self.model_name,
                            device.lower(),
                            "prefix_cache",
                            bool(prefix_cache_enabled),
                        )
                    for feature in degraded_features:
                        record_runtime_feature_degradation(
                            self.model_name,
                            device.lower(),
                            feature,
                            "safe_retry",
                        )
                    return
                except Exception as exc:
                    last_error = exc
                    self._load_diagnostics.append(
                        {
                            "device": device,
                            "status": "failed",
                            "config": config,
                            "degraded_features": attempt["degraded_features"],
                            "error": str(exc),
                        }
                    )
                    if attempt_index < len(attempts):
                        next_degraded = attempts[attempt_index]["degraded_features"]
                        removed_features = sorted(
                            set(next_degraded) - set(attempt["degraded_features"])
                        )
                        logger.warning(
                            "Failed to load on %s with runtime settings %s: %s. "
                            "Retrying without %s.",
                            device,
                            config,
                            exc,
                            ", ".join(removed_features) or "optional runtime features",
                        )
                    else:
                        logger.warning("Failed to load on %s: %s", device, exc)

        record_error("engine.load_model", "device_error")
        raise DeviceError(
            device=self.requested_device,
            available_devices=get_available_devices(),
            original_error=last_error
        )

    @staticmethod
    def _metric_mean(value: Any) -> Optional[float]:
        """Convert OpenVINO perf metric values into floats."""
        if value is None:
            return None
        metric_value = getattr(value, "mean", value)
        try:
            return float(metric_value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _metric_seconds(cls, value: Any) -> Optional[float]:
        """Convert millisecond perf metric values into seconds."""
        mean_value = cls._metric_mean(value)
        if mean_value is None:
            return None
        return mean_value / 1000.0

    @staticmethod
    def _extract_native_finish_reason(result: Any) -> FinishReason | None:
        for attr in ("finish_reason", "stop_reason", "finish_status", "stop_reason_name"):
            reason = _normalize_finish_reason(getattr(result, attr, None))
            if reason is not None:
                return reason
        if isinstance(result, dict):
            for key in ("finish_reason", "stop_reason", "finish_status"):
                reason = _normalize_finish_reason(result.get(key))
                if reason is not None:
                    return reason
        return None

    def _set_last_finish_reason(
        self,
        result: Any,
        *,
        completion_tokens: int,
        max_new_tokens: int,
    ) -> None:
        native_reason = self._extract_native_finish_reason(result)
        if native_reason is not None:
            self._last_finish_reason = native_reason
        elif completion_tokens >= 0:
            if max_new_tokens > 0 and completion_tokens >= max_new_tokens:
                self._last_finish_reason = "length"
            else:
                self._last_finish_reason = "stop"
        else:
            self._last_finish_reason = None
        if self._last_generation_stats is not None:
            self._last_generation_stats["finish_reason"] = self._last_finish_reason

    @property
    def last_finish_reason(self) -> FinishReason | None:
        """Normalized reason for the most recent generation, when known."""
        return self._last_finish_reason

    @staticmethod
    def _decode_generation_text(result: Any) -> str:
        """Normalize LLMPipeline generate output into text."""
        texts = getattr(result, "texts", None)
        if isinstance(texts, list) and texts:
            return str(texts[0])
        return str(result)

    def _update_last_generation_stats(self, result: Any) -> None:
        """Capture additive runtime generation diagnostics."""
        perf_metrics = getattr(result, "perf_metrics", None)
        if perf_metrics is None:
            self._last_generation_stats = None
            return

        stats = {
            "device": self.actual_device,
            "load_time_seconds": self._metric_seconds(
                getattr(perf_metrics, "get_load_time", lambda: None)()
            ),
            "generate_duration_seconds": self._metric_seconds(
                getattr(perf_metrics, "get_generate_duration", lambda: None)()
            ),
            "inference_duration_seconds": self._metric_seconds(
                getattr(perf_metrics, "get_inference_duration", lambda: None)()
            ),
            "tokenization_duration_seconds": self._metric_seconds(
                getattr(perf_metrics, "get_tokenization_duration", lambda: None)()
            ),
            "detokenization_duration_seconds": self._metric_seconds(
                getattr(perf_metrics, "get_detokenization_duration", lambda: None)()
            ),
            "ttft_seconds": self._metric_seconds(
                getattr(perf_metrics, "get_ttft", lambda: None)()
            ),
            "tpot_seconds": self._metric_seconds(
                getattr(perf_metrics, "get_tpot", lambda: None)()
            ),
            "throughput_tokens_per_second": self._metric_mean(
                getattr(perf_metrics, "get_throughput", lambda: None)()
            ),
            "input_tokens": getattr(
                perf_metrics,
                "get_num_input_tokens",
                lambda: None,
            )(),
            "generated_tokens": getattr(
                perf_metrics,
                "get_num_generated_tokens",
                lambda: None,
            )(),
        }
        self._last_generation_stats = stats

        latency = stats["generate_duration_seconds"]
        if latency is not None:
            record_inference(self.model_name, self.actual_device.lower(), "chat", latency)
        if stats["ttft_seconds"] is not None:
            record_ttft(self.model_name, stats["ttft_seconds"])
        if stats["tpot_seconds"] is not None:
            record_tpot(self.model_name, stats["tpot_seconds"])
        if stats["throughput_tokens_per_second"] is not None:
            record_tokens_per_second(
                self.model_name,
                self.actual_device.lower(),
                stats["throughput_tokens_per_second"],
            )

    def _clear_active_call_locked(self) -> None:
        """Clear active inference bookkeeping."""
        self._active_future = None
        self._active_timeout_seconds = None
        self._active_operation = None

    def _reap_completed_call_locked(self) -> None:
        """Clear active inference state once the worker finishes."""
        if self._active_future is not None and self._active_future.done():
            self._clear_active_call_locked()

    def _submit_inference_call(
        self,
        operation: str,
        func: Callable[[], Any],
    ) -> concurrent.futures.Future[Any]:
        """Submit a single inference call while preventing hidden queueing."""
        with self._inference_lock:
            self._reap_completed_call_locked()
            if self._active_future is not None:
                if self._active_timeout_seconds is not None:
                    record_error(f"engine.{operation}", "busy_after_timeout")
                    raise InferenceTimeoutError(
                        self._active_timeout_seconds,
                        message=(
                            f"Previous {self._active_operation or 'inference'} timed out "
                            f"after {self._active_timeout_seconds} seconds and is still "
                            "running; refusing to queue another request"
                        ),
                    )

                record_error(f"engine.{operation}", "busy")
                raise InferenceError(
                    "Inference already in progress on this shared engine; "
                    "retry after the current request completes",
                    status_code=503,
                )

            future = self._inference_executor.submit(func)
            self._active_future = future
            self._active_timeout_seconds = None
            self._active_operation = operation
            return future

    def _release_inference_call(
        self,
        future: concurrent.futures.Future[Any],
    ) -> None:
        """Release the active inference slot once the future completes."""
        with self._inference_lock:
            if self._active_future is future and future.done():
                self._clear_active_call_locked()

    def _mark_inference_timeout(
        self,
        future: concurrent.futures.Future[Any],
        timeout: int,
    ) -> None:
        """Preserve timed-out in-flight work so follow-up calls fail fast."""
        with self._inference_lock:
            if self._active_future is not future:
                return
            if future.cancel():
                self._clear_active_call_locked()
                return
            self._active_timeout_seconds = timeout

    def _run_with_timeout(
        self,
        operation: str,
        func: Callable[[], Any],
        timeout: int,
    ) -> Any:
        """Run inference work on the long-lived executor with prompt timeout returns."""
        future = self._submit_inference_call(operation, func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self._mark_inference_timeout(future, timeout)
            logger.error("%s timed out after %ss", operation, timeout)
            record_error(f"engine.{operation}", "timeout")
            raise InferenceTimeoutError(timeout)
        except Exception as exc:
            record_error(f"engine.{operation}", exc.__class__.__name__.lower())
            raise
        finally:
            self._release_inference_call(future)

    def has_active_inference(self) -> bool:
        """Return whether native inference is still active on this engine."""
        with self._inference_lock:
            self._reap_completed_call_locked()
            return self._active_future is not None

    def shutdown(self, wait: bool = False) -> None:
        """Release executor resources associated with this engine.

        Timeouts are soft: shutdown can cancel queued work, but cannot forcibly
        stop a native OpenVINO call that is already running.
        """
        self._inference_executor.shutdown(wait=wait, cancel_futures=True)
    
    def warmup(self, warmup_tokens: int = 16) -> None:
        """Perform warmup inference to pre-compile the NPU pipeline.
        
        Intel NPU requires pipeline compilation on first inference, which
        can take 80-130 seconds. Running a short warmup during startup
        eliminates this latency from the first real user request.
        
        This method is thread-safe and idempotent - multiple calls will
        only perform warmup once.
        
        Args:
            warmup_tokens: Number of tokens to generate during warmup.
                Defaults to 16. More tokens don't improve compilation
                but increase warmup time.
        
        Example:
            >>> engine = InferenceEngine("/path/to/model", device="NPU")
            >>> engine.warmup()  # Blocks for 80-130s on NPU
            >>> # First real request now has normal latency
        
        Note:
            Warmup failures are logged but not raised - the pipeline will
            compile on first real request instead. This ensures the engine
            remains usable even if warmup fails for any reason.
        """
        with self._warmup_lock:
            if self._is_warmed_up:
                logger.debug("NPU pipeline already warmed up, skipping")
                return
            
            logger.info(
                f"Warming up {self.actual_device} pipeline with "
                f"{warmup_tokens} tokens..."
            )
            try:
                # Short generation to trigger compilation
                for _ in self.generate_stream("Hello", max_new_tokens=warmup_tokens):
                    pass
                self._is_warmed_up = True
                logger.info(f"{self.actual_device} warmup complete")
            except Exception as e:
                logger.warning(
                    f"Warmup failed (will compile on first request): {e}"
                )
    
    @property
    def is_warmed_up(self) -> bool:
        """Check if the pipeline has been warmed up.
        
        Returns:
            True if warmup() has completed successfully, False otherwise.
        """
        return self._is_warmed_up
    
    def get_device_info(self) -> dict[str, object]:
        """Get detailed information about device selection and status.
        
        Returns:
            Dictionary containing:
                - requested_device: Originally requested device
                - actual_device: Device currently in use
                - fallback_device: Configured fallback device
                - used_fallback: Whether fallback was activated
                - available_devices: All available devices
                - is_warmed_up: Whether warmup has completed
        
        Example:
            >>> info = engine.get_device_info()
            >>> print(f"Running on: {info['actual_device']}")
            Running on: NPU
        """
        return {
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "fallback_device": self.fallback_device,
            "used_fallback": self.used_fallback,
            "available_devices": get_available_devices(),
            "is_warmed_up": self._is_warmed_up,
            "inference_timeout": self.inference_timeout,
            "max_prompt_len": self.max_prompt_len,
            "compile_cache_dir": (
                str(self.compile_cache_dir) if self.compile_cache_dir is not None else None
            ),
            "compile_cache_mode": self.compile_cache_mode,
            "prefix_cache_mode": self.prefix_cache_mode,
            "runtime_features": dict(self._runtime_features),
            "model_load_seconds": self._model_load_seconds,
            "load_diagnostics": list(self._load_diagnostics),
            "last_generation_stats": self._last_generation_stats,
            "last_finish_reason": self._last_finish_reason,
        }
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: Optional[int] = None,
    ) -> str:
        """Generate text completion synchronously with timeout protection.
        
        Runs inference on the loaded model and returns the complete response.
        For long-form generation or real-time display, consider using
        generate_stream() instead.
        
        Args:
            prompt: Input text to generate continuation for.
            max_new_tokens: Maximum number of tokens to generate.
                Defaults to 256.
            temperature: Sampling temperature (0.0-2.0). Higher values
                increase randomness. Set to 0 for deterministic output.
                Defaults to 0.7.
            top_p: Nucleus sampling probability. Only tokens with cumulative
                probability <= top_p are considered. Defaults to 0.9.
            timeout: Maximum time in seconds to wait for completion.
                Defaults to DEFAULT_INFERENCE_TIMEOUT (180s). This is a soft
                timeout: native OpenVINO work may continue in the worker after
                the caller receives InferenceTimeoutError.
        
        Returns:
            Generated text completion as a string.
        
        Raises:
            ModelNotLoadedError: If generate() is called before model loading.
            InferenceTimeoutError: If generation exceeds the timeout.
            InferenceError: For other generation failures.
        
        Example:
            >>> response = engine.generate(
            ...     prompt="What is the capital of France?",
            ...     max_new_tokens=50,
            ...     temperature=0.3
            ... )
            >>> print(response)
            The capital of France is Paris.
        
        Note:
            First inference on NPU takes 80-130s for pipeline compilation.
            Use warmup() after loading to eliminate this cold-start latency.
        """
        if self.pipeline is None:
            raise ModelNotLoadedError()
        
        if timeout is None:
            timeout = self.inference_timeout
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        if temperature > 0:
            config.temperature = temperature
            config.top_p = top_p
            config.do_sample = True
        else:
            config.do_sample = False
        
        result = self._run_with_timeout(
            "generate",
            lambda: self.pipeline.generate(prompt, config),
            timeout,
        )
        self._update_last_generation_stats(result)
        response_text = self._decode_generation_text(result)
        generated_tokens = None
        if self._last_generation_stats is not None:
            generated_tokens = self._last_generation_stats.get("generated_tokens")
        try:
            completion_tokens = int(generated_tokens) if generated_tokens is not None else -1
        except (TypeError, ValueError):
            completion_tokens = -1
        self._set_last_finish_reason(
            result,
            completion_tokens=completion_tokens,
            max_new_tokens=max_new_tokens,
        )
        return response_text
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        streamer_callback: Optional[Callable[[str], bool]] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        timeout: Optional[int] = None,
    ) -> Iterator[str]:
        """Generate text completion with streaming token output.
        
        Runs inference and yields tokens as they are generated, enabling
        real-time display and early termination. Supports both custom
        callbacks and clean abort mechanisms.
        
        Args:
            prompt: Input text to generate continuation for.
            max_new_tokens: Maximum number of tokens to generate.
                Defaults to 256.
            temperature: Sampling temperature (0.0-2.0). Higher values
                increase randomness. Set to 0 for deterministic output.
                Defaults to 0.7.
            top_p: Nucleus sampling probability. Only tokens with cumulative
                probability <= top_p are considered. Defaults to 0.9.
            streamer_callback: Optional function called for each token.
                Receives the token string and should return True to abort
                generation or False to continue. Defaults to None.
            abort_callback: Optional function called to check for abort.
                Should return True to abort generation. This enables clean
                cancellation (e.g., when client disconnects). Defaults to None.
            timeout: Maximum time in seconds to wait for completion.
                Defaults to DEFAULT_INFERENCE_TIMEOUT (180s). This is a soft
                timeout: native OpenVINO work may continue in the worker after
                the caller receives InferenceTimeoutError.
        
        Yields:
            Individual tokens as strings.
        
        Raises:
            ModelNotLoadedError: If called before model loading.
            InferenceTimeoutError: If generation exceeds the timeout.
            InferenceError: For other generation failures.
        
        Example:
            >>> # Simple streaming
            >>> for token in engine.generate_stream("Tell me a story"):
            ...     print(token, end="", flush=True)
            
            >>> # With abort callback for clean cancellation
            >>> cancelled = False
            >>> def check_cancel():
            ...     return cancelled
            >>> for token in engine.generate_stream(
            ...     "Long story...",
            ...     abort_callback=check_cancel
            ... ):
            ...     print(token, end="")
            ...     if some_condition:
            ...         cancelled = True  # Will abort on next token
        
        Note:
            Tokens are collected internally and yielded after generation
            completes. For true real-time streaming, use streamer_callback.
        """
        if self.pipeline is None:
            raise ModelNotLoadedError()
        
        if timeout is None:
            timeout = self.inference_timeout
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        if temperature > 0:
            config.temperature = temperature
            config.top_p = top_p
            config.do_sample = True
        else:
            config.do_sample = False
        
        # Collect tokens for streaming
        tokens: list[str] = []
        
        def streamer(token: str) -> bool:
            """Internal streamer that collects tokens and handles callbacks.
            
            Args:
                token: The generated token string.
            
            Returns:
                True to abort generation, False to continue.
            """
            tokens.append(token)
            
            # Check abort callback first
            if abort_callback is not None and abort_callback():
                logger.info("Generation aborted via abort_callback")
                return True
            
            # Then check streamer callback
            if streamer_callback is not None:
                return streamer_callback(token)
            
            return False
        
        result = self._run_with_timeout(
            "generate_stream",
            lambda: self.pipeline.generate(prompt, config, streamer),
            timeout,
        )
        self._update_last_generation_stats(result)
        self._set_last_finish_reason(
            result,
            completion_tokens=len(tokens),
            max_new_tokens=max_new_tokens,
        )
        
        # Yield collected tokens
        for token in tokens:
            yield token


# =============================================================================
# Global Engine Management (Per-Model/Device Pool)
# =============================================================================


# Global engine instances (per-model/device pool with thread safety)
@dataclass(frozen=True)
class EnginePoolKey:
    """Identity of a pooled engine: the resolved model path and its device.

    Frozen so it can be used as a dict key. ``model_path`` is the resolved
    absolute path and ``device`` is the canonical upper-cased device string,
    so equivalent requests map to the same pooled engine and lock.
    """

    model_path: str
    device: str


_engine_pool: dict[EnginePoolKey, InferenceEngine] = {}
_device_locks: dict[EnginePoolKey, threading.Lock] = {}
_loaded_models: dict[str, InferenceEngine] = {}
_engine_lock: threading.Lock = threading.Lock()


def _resolve_engine_runtime_config(
    *,
    config: LLMRuntimeConfig | None = None,
    model_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    inference_timeout: Optional[int] = None,
    max_prompt_len: Optional[int] = None,
    compile_cache_dir: Optional[str | Path] = None,
    compile_cache_mode: Optional[str] = None,
    prefix_cache_mode: Optional[str] = None,
) -> LLMRuntimeConfig:
    """Resolve engine configuration from the active control plane plus overrides."""
    base_config = config or get_active_llm_runtime_config()
    overrides: dict[str, object] = {}
    if model_path is not None:
        overrides["model_path"] = Path(model_path)
    if device is not None:
        overrides["device"] = device
    if inference_timeout is not None:
        overrides["inference_timeout"] = inference_timeout
    if max_prompt_len is not None:
        overrides["max_prompt_len"] = max_prompt_len
    if compile_cache_dir is not None:
        overrides["compile_cache_dir"] = (
            Path(compile_cache_dir) if compile_cache_dir else None
        )
    if compile_cache_mode is not None:
        overrides["compile_cache_mode"] = compile_cache_mode
    if prefix_cache_mode is not None:
        overrides["prefix_cache_mode"] = prefix_cache_mode
    return replace(base_config, **overrides) if overrides else base_config


def _engine_pool_key(runtime_config: LLMRuntimeConfig) -> EnginePoolKey:
    """Return the normalized model/device cache key for an engine config."""
    return EnginePoolKey(
        model_path=str(Path(runtime_config.model_path).resolve()),
        device=normalize_device(runtime_config.device) or str(runtime_config.device).upper(),
    )


def _normalized_engine_config(
    *,
    config: LLMRuntimeConfig | None = None,
    model_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> LLMRuntimeConfig:
    runtime_config = _resolve_engine_runtime_config(
        config=config,
        model_path=model_path,
        device=device,
    )
    return replace(runtime_config, device=runtime_config.device.upper())


def _device_lock_for_key(pool_key: EnginePoolKey) -> threading.Lock:
    with _engine_lock:
        lock = _device_locks.get(pool_key)
        if lock is None:
            lock = threading.Lock()
            _device_locks[pool_key] = lock
        return lock


def _acquire_device_lock(
    device: str,
    *,
    model_path: Optional[str | Path] = None,
    config: LLMRuntimeConfig | None = None,
    timeout: float,
) -> tuple[str, threading.Lock]:
    """Acquire the per-(model, device) lock, returning (selected_device, lock).

    Raises ``DeviceBusyError`` if the lock is not acquired within ``timeout``.
    The caller owns the returned lock and is responsible for releasing it.
    """
    runtime_config = _normalized_engine_config(
        config=config,
        model_path=model_path,
        device=device,
    )
    pool_key = _engine_pool_key(runtime_config)
    lock = _device_lock_for_key(pool_key)
    acquired = lock.acquire(timeout=max(0.0, float(timeout)))
    if not acquired:
        raise DeviceBusyError(runtime_config.device)
    return runtime_config.device, lock


@contextmanager
def acquire_device_slot(
    device: str,
    *,
    model_path: Optional[str | Path] = None,
    config: LLMRuntimeConfig | None = None,
    timeout: float,
):
    """Acquire the per-(model, device) inference slot or raise DeviceBusyError."""
    selected_device, lock = _acquire_device_lock(
        device,
        model_path=model_path,
        config=config,
        timeout=timeout,
    )
    try:
        yield selected_device
    finally:
        lock.release()


def fallback_devices_after(device: str) -> list[str]:
    """Return available fallback devices after the requested device in priority order."""
    normalized = device_class(device)
    available = {device_class(candidate) for candidate in get_available_devices() if candidate}
    try:
        start_index = DEVICE_FALLBACK_CHAIN.index(normalized) + 1
    except ValueError:
        start_index = 0
    return [
        candidate
        for candidate in DEVICE_FALLBACK_CHAIN[start_index:]
        if candidate != normalized and candidate in available
    ]


def _slot_candidate_devices(device: str, *, fallback_on_busy: bool) -> list[str]:
    normalized = str(device).strip().upper()
    candidates = [normalized]
    if fallback_on_busy:
        candidates.extend(fallback_devices_after(normalized))
    return candidates


@dataclass
class RoutedEngineSlot:
    """A held inference slot describing the routed vs. selected device.

    Replaces the previous "contextmanager plus ad-hoc setattr" approach with a
    typed handle. Supports both explicit ``close()`` and the context-manager
    protocol so callers can release the underlying device lock exactly once.
    """

    routed_device: str
    selected_device: str
    fallback_reason: str | None
    lock: threading.Lock
    _released: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if not self._released:
            self._released = True
            self.lock.release()

    def __enter__(self) -> "RoutedEngineSlot":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


def open_routed_engine_slot(
    device: str,
    *,
    timeout: float,
    fallback_on_busy: bool = False,
    model_path: Optional[str | Path] = None,
    config: LLMRuntimeConfig | None = None,
    engine_factory: Optional[Callable[[str], InferenceEngine]] = None,
) -> tuple[InferenceEngine, RoutedEngineSlot]:
    """Acquire a device slot, optionally falling back on busy, and return its engine.

    ``engine_factory`` lets callers (e.g. the chat API) supply a patchable engine
    accessor while sharing the acquire/fallback/lock logic. It receives the
    selected device and must return the engine to use. When omitted, the pooled
    ``get_llm_engine`` is used.
    """
    requested = normalize_device(device) or str(device).strip().upper()
    last_busy: DeviceBusyError | None = None
    for candidate in _slot_candidate_devices(device, fallback_on_busy=fallback_on_busy):
        try:
            selected_device, lock = _acquire_device_lock(
                candidate,
                model_path=model_path,
                config=config,
                timeout=timeout,
            )
        except DeviceBusyError as exc:
            last_busy = exc
            if not fallback_on_busy:
                raise
            continue
        try:
            if engine_factory is not None:
                engine = engine_factory(selected_device)
            else:
                engine = get_llm_engine(model_path=model_path, device=selected_device, config=config)
        except Exception:
            lock.release()
            raise
        slot = RoutedEngineSlot(
            routed_device=requested,
            selected_device=selected_device,
            fallback_reason=(
                FallbackReason.BUSY.value if selected_device != requested else None
            ),
            lock=lock,
        )
        return engine, slot

    if last_busy is not None:
        raise last_busy
    raise DeviceBusyError(requested)


def _unique_engines() -> list[InferenceEngine]:
    """Return unique engines known to the pool and legacy loaded-model index."""
    engines: list[InferenceEngine] = []
    for engine in [*_engine_pool.values(), *_loaded_models.values()]:
        if engine not in engines:
            engines.append(engine)
    return engines


def get_engine_pool_snapshot() -> list[dict[str, object]]:
    """Return a non-mutating snapshot of loaded engine pool state."""
    with _engine_lock:
        items = list(_engine_pool.items())

    snapshot: list[dict[str, object]] = []
    for pool_key, engine in items:
        model_path = pool_key.model_path
        requested_device = pool_key.device
        lock = _device_lock_for_key(pool_key)
        get_device_info = getattr(engine, "get_device_info", None)
        try:
            device_info = dict(get_device_info()) if callable(get_device_info) else {}
        except Exception:
            logger.warning(
                "Failed to read device info for pooled engine on %s; reporting requested device",
                requested_device,
                exc_info=True,
            )
            device_info = {}
        model = getattr(engine, "model_name", None) or Path(model_path).name
        actual_device = str(device_info.get("actual_device") or getattr(engine, "actual_device", requested_device))
        snapshot.append(
            {
                "model": str(model),
                "model_path": Path(model_path).name,
                "device": actual_device,
                "requested_device": requested_device,
                "loaded": True,
                "warmed": bool(device_info.get("is_warmed_up", getattr(engine, "is_warmed_up", False))),
                "busy": lock.locked(),
            }
        )
    return snapshot


def reset_engine(*, force: bool = False) -> None:
    """Reset all global engines to allow reloading with different configuration.
    
    Clears the engine pool and loaded models dictionary, allowing the next
    call to get_llm_engine() to create fresh engines with potentially
    different device or model settings.
    
    This function is thread-safe.
    
    Example:
        >>> engine1 = get_llm_engine(device="NPU")
        >>> reset_engine()
        >>> engine2 = get_llm_engine(device="CPU")  # New instance
    
    Warning:
        Any references to previous engine instances become stale. Active
        inference prevents reset unless force=True is used. Forced reset is
        best-effort because already-running native calls cannot be killed.
    """
    global _loaded_models
    with _engine_lock:
        engines_to_shutdown = _unique_engines()

        active_engines = [
            engine
            for engine in engines_to_shutdown
            if callable(getattr(engine, "has_active_inference", None))
            and engine.has_active_inference() is True
        ]
        if active_engines and not force:
            raise InferenceError(
                "Cannot reset inference engine while inference is active; "
                "retry after completion or call reset_engine(force=True) for a best-effort reset",
                status_code=409,
            )
        if active_engines:
            logger.warning(
                "Forcing engine reset while %s inference call(s) are still active; "
                "native work may continue until it returns",
                len(active_engines),
            )
        _engine_pool.clear()
        # Preserve ALL device locks across reset. They serialize native inference
        # per (model, device); dropping a lock here races with acquire_device_slot,
        # which fetches the lock under _engine_lock but acquires it *after* releasing
        # _engine_lock. An "unlocked" lock may therefore have a thread about to
        # acquire it, so dropping it could let a fresh lock permit a second
        # concurrent native inference. Unlocked locks left behind are harmless and
        # the keyspace (model paths x devices) is finite.
        _loaded_models = {}
    for engine in engines_to_shutdown:
        engine.shutdown(wait=False)


def get_llm_engine(
    model_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    inference_timeout: Optional[int] = None,
    max_prompt_len: Optional[int] = None,
    compile_cache_dir: Optional[str | Path] = None,
    compile_cache_mode: Optional[str] = None,
    prefix_cache_mode: Optional[str] = None,
    config: LLMRuntimeConfig | None = None,
) -> InferenceEngine:
    """Get or create an LLM inference engine for a model/device pair.
    
    Implements a thread-safe per-(model, device) pool with double-checked
    locking. Calls for the same normalized model/device key return the same
    instance; calls for different devices create independent engines.
    """
    runtime_config = _resolve_engine_runtime_config(
        config=config,
        model_path=model_path,
        device=device,
        inference_timeout=inference_timeout,
        max_prompt_len=max_prompt_len,
        compile_cache_dir=compile_cache_dir,
        compile_cache_mode=compile_cache_mode,
        prefix_cache_mode=prefix_cache_mode,
    )
    runtime_config = replace(runtime_config, device=runtime_config.device.upper())
    pool_key = _engine_pool_key(runtime_config)

    engine = _engine_pool.get(pool_key)
    if engine is not None:
        return engine

    with _engine_lock:
        engine = _engine_pool.get(pool_key)
        if engine is None:
            if runtime_config.backend is not LLMBackend.OPENVINO:
                raise InferenceError(
                    "Legacy engine path only supports the openvino backend; "
                    "use npu_proxy.inference.llm_runtime.get_llm_runtime() "
                    "for backend-neutral access.",
                )
            if not runtime_config.model_path.exists():
                raise ModelNotFoundError(Path(runtime_config.model_path))

            engine = InferenceEngine(
                runtime_config.model_path,
                runtime_config.device,
                inference_timeout=runtime_config.inference_timeout,
                max_prompt_len=runtime_config.max_prompt_len,
                compile_cache_dir=runtime_config.compile_cache_dir,
                compile_cache_mode=runtime_config.compile_cache_mode,
                prefix_cache_mode=runtime_config.prefix_cache_mode,
            )
            _engine_pool[pool_key] = engine
            if pool_key not in _device_locks:
                _device_locks[pool_key] = threading.Lock()
            _loaded_models[engine.model_name] = engine

    return engine


def get_primary_loaded_engine() -> InferenceEngine | None:
    """Return the default-device engine if loaded, else any loaded engine.

    With the per-(model, device) pool a single model can have several engines
    loaded at once. Observability surfaces need a stable "primary" engine that
    reflects the configured/default device rather than whichever device served
    the most recent request, so they don't misreport the active device after a
    cross-device fallback.
    """
    runtime_config = get_active_llm_runtime_config()
    default_key = _engine_pool_key(runtime_config)
    with _engine_lock:
        engine = _engine_pool.get(default_key)
        if engine is None:
            engine = next(iter(_engine_pool.values()), None)
        if engine is None:
            engine = next(iter(_loaded_models.values()), None)
    return engine


def get_llm_execution_target(*, load_if_needed: bool = False) -> dict[str, object]:
    """Return the current/default LLM execution target."""
    runtime_config = get_active_llm_runtime_config()
    if load_if_needed:
        engine = get_llm_engine(config=runtime_config)
    else:
        engine = get_primary_loaded_engine()

    if engine is None:
        return {
            "model": runtime_config.model_path.name,
            "device": runtime_config.device,
            "requested_device": runtime_config.device,
            "used_fallback": False,
            "loaded": False,
        }

    device_info = engine.get_device_info()
    return {
        "model": engine.model_name,
        "device": str(device_info.get("actual_device") or engine.actual_device),
        "requested_device": str(device_info.get("requested_device") or engine.requested_device),
        "used_fallback": bool(device_info.get("used_fallback")),
        "loaded": True,
    }


def get_loaded_models() -> dict[str, InferenceEngine]:
    """Get dictionary of all currently loaded model engines.

    .. note::
        This is a legacy index keyed by model NAME only. With the
        per-(model, device) engine pool a single model can have several
        engines loaded at once (e.g. NPU and CPU), but this mapping retains
        only the most recently loaded engine per model name. For complete
        per-device state use :func:`get_engine_pool_snapshot`, and for the
        primary/default-device engine use :func:`get_primary_loaded_engine`.

    Returns:
        Dictionary mapping model names to their InferenceEngine instances.
        Empty if no models have been loaded.
    
    Example:
        >>> get_llm_engine()  # Load default model
        >>> models = get_loaded_models()
        >>> print(list(models.keys()))
        ['tinyllama-1.1b-chat-int4-ov']
    
    Note:
        This returns a shallow copy so callers cannot mutate module state.
    """
    with _engine_lock:
        return dict(_loaded_models)


def is_model_loaded() -> bool:
    """Check if any model is currently loaded.
    
    Returns:
        True if get_llm_engine() has been called and succeeded,
        False otherwise.
    
    Example:
        >>> is_model_loaded()
        False
        >>> get_llm_engine()
        >>> is_model_loaded()
        True
    """
    with _engine_lock:
        return bool(_engine_pool or _loaded_models)
