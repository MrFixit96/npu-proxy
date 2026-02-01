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

import logging
import os
import threading
import concurrent.futures
from pathlib import Path
from typing import Iterator, Callable, Optional
import openvino_genai as ov_genai

logger = logging.getLogger(__name__)


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


class InferenceTimeoutError(InferenceError):
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
    
    def __init__(self, timeout_seconds: int) -> None:
        """Initialize InferenceTimeoutError.
        
        Args:
            timeout_seconds: The timeout duration that was exceeded.
        """
        super().__init__(
            f"Inference timed out after {timeout_seconds} seconds",
            status_code=504
        )
        self.timeout_seconds = timeout_seconds


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


# Default model paths
DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"
DEFAULT_LLM_MODEL: str = "tinyllama-1.1b-chat-int4-ov"

# Default timeout for inference (seconds)
DEFAULT_INFERENCE_TIMEOUT: int = int(
    os.environ.get("NPU_PROXY_INFERENCE_TIMEOUT", "180")
)

# Maximum prompt length for NPU (tokens) - NPU default is 1024, increase for longer prompts
DEFAULT_MAX_PROMPT_LEN: int = int(
    os.environ.get("NPU_PROXY_MAX_PROMPT_LEN", "4096")
)

# Device priority for fallback (NPU → GPU → CPU)
DEVICE_FALLBACK_CHAIN: list[str] = ["NPU", "GPU", "CPU"]


# =============================================================================
# Device Selection Functions
# =============================================================================


def get_available_devices() -> list[str]:
    """Get list of available OpenVINO compute devices.
    
    Queries the OpenVINO runtime to discover available hardware accelerators
    including NPU, GPU, and CPU. Falls back to CPU-only if device enumeration
    fails.
    
    Returns:
        A list of device identifiers (e.g., ["NPU", "GPU", "CPU"]).
    
    Example:
        >>> devices = get_available_devices()
        >>> print(devices)
        ['NPU', 'GPU', 'CPU']
    
    Note:
        This function creates a new OpenVINO Core instance on each call.
        For performance-critical code, consider caching the result.
    """
    try:
        import openvino as ov
        core = ov.Core()
        return core.available_devices
    except Exception as e:
        logger.warning(f"Failed to enumerate devices: {e}")
        return ["CPU"]  # CPU is always available


def select_best_device(preferred: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Select the best available compute device with automatic fallback.
    
    Implements a device selection strategy that respects user preference
    while providing automatic fallback to ensure inference can always run.
    The fallback chain is: NPU → GPU → CPU.
    
    Args:
        preferred: User-requested device (case-insensitive). If None or
            unavailable, the best available device is selected automatically.
    
    Returns:
        A tuple of (selected_device, fallback_device). The fallback_device
        is None if no fallback is available (e.g., when CPU is selected).
    
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
    logger.info(f"Available devices: {available}")
    
    # If user specified a device, try to use it
    if preferred:
        preferred = preferred.upper()
        if preferred in available:
            # Find fallback from chain
            fallback: Optional[str] = None
            chain_idx = (
                DEVICE_FALLBACK_CHAIN.index(preferred) 
                if preferred in DEVICE_FALLBACK_CHAIN 
                else -1
            )
            for device in DEVICE_FALLBACK_CHAIN[chain_idx + 1:]:
                if device in available:
                    fallback = device
                    break
            return preferred, fallback
        else:
            logger.warning(
                f"Requested device {preferred} not available, "
                "selecting best alternative"
            )
    
    # Select best available from fallback chain
    for device in DEVICE_FALLBACK_CHAIN:
        if device in available:
            # Find fallback
            fallback = None
            chain_idx = DEVICE_FALLBACK_CHAIN.index(device)
            for fb_device in DEVICE_FALLBACK_CHAIN[chain_idx + 1:]:
                if fb_device in available:
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
        device: str = "NPU"
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
        self.device: str
        self.fallback_device: Optional[str]
        self.device, self.fallback_device = select_best_device(self.requested_device)
        self.actual_device: str = self.device  # Updated if fallback occurs
        self.pipeline: Optional[ov_genai.LLMPipeline] = None
        self.model_name: str = self.model_path.name
        self.used_fallback: bool = False
        self._is_warmed_up: bool = False
        self._warmup_lock: threading.Lock = threading.Lock()
        self._load_model()
    
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
        # Always include CPU as last resort
        if "CPU" not in devices_to_try:
            devices_to_try.append("CPU")
        
        last_error: Optional[Exception] = None
        for device in devices_to_try:
            try:
                logger.info(f"Loading model from {self.model_path} on {device}")
                
                # Configure device-specific settings
                if device == "NPU":
                    # NPU requires MAX_PROMPT_LEN config for longer prompts
                    config = {"MAX_PROMPT_LEN": DEFAULT_MAX_PROMPT_LEN}
                    logger.info(f"NPU config: MAX_PROMPT_LEN={DEFAULT_MAX_PROMPT_LEN}")
                    self.pipeline = ov_genai.LLMPipeline(
                        str(self.model_path), device, config
                    )
                else:
                    self.pipeline = ov_genai.LLMPipeline(
                        str(self.model_path), device
                    )
                
                self.actual_device = device
                self.used_fallback = (device != self.device)
                if self.used_fallback:
                    logger.warning(
                        f"Using fallback device {device} instead of {self.device}"
                    )
                else:
                    logger.info(f"Model loaded successfully on {device}")
                return
            except Exception as e:
                logger.warning(f"Failed to load on {device}: {e}")
                last_error = e
                continue
        
        # All devices failed
        raise DeviceError(
            device=self.requested_device,
            available_devices=get_available_devices(),
            original_error=last_error
        )
    
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
                Defaults to DEFAULT_INFERENCE_TIMEOUT (180s).
        
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
            timeout = DEFAULT_INFERENCE_TIMEOUT
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        if temperature > 0:
            config.temperature = temperature
            config.top_p = top_p
            config.do_sample = True
        else:
            config.do_sample = False
        
        # Run with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.pipeline.generate, prompt, config)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                logger.error(f"Inference timed out after {timeout}s")
                raise InferenceTimeoutError(timeout)
    
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
                Defaults to DEFAULT_INFERENCE_TIMEOUT (180s).
        
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
            timeout = DEFAULT_INFERENCE_TIMEOUT
        
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
        
        # Generate with streamer and timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.pipeline.generate, prompt, config, streamer
            )
            try:
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Streaming inference timed out after {timeout}s")
                raise InferenceTimeoutError(timeout)
        
        # Yield collected tokens
        for token in tokens:
            yield token


# =============================================================================
# Global Engine Management (Singleton Pattern)
# =============================================================================


# Global engine instances (singleton pattern with thread safety)
_llm_engine: Optional[InferenceEngine] = None
_loaded_models: dict[str, InferenceEngine] = {}
_engine_lock: threading.Lock = threading.Lock()


def reset_engine() -> None:
    """Reset the global engine to allow reloading with different configuration.
    
    Clears the singleton instance and loaded models dictionary, allowing
    the next call to get_llm_engine() to create a fresh engine with
    potentially different device or model settings.
    
    This function is thread-safe.
    
    Example:
        >>> engine1 = get_llm_engine(device="NPU")
        >>> reset_engine()
        >>> engine2 = get_llm_engine(device="CPU")  # New instance
    
    Warning:
        Any references to the previous engine instance become stale.
        Ensure no active inference is running before calling reset_engine().
    """
    global _llm_engine, _loaded_models
    with _engine_lock:
        _llm_engine = None
        _loaded_models = {}


def get_llm_engine(
    model_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> InferenceEngine:
    """Get or create the singleton LLM inference engine instance.
    
    Implements a thread-safe singleton pattern with double-checked locking.
    The first call creates the engine; subsequent calls return the same
    instance regardless of arguments.
    
    Args:
        model_path: Path to the OpenVINO model directory. Defaults to
            DEFAULT_MODEL_DIR/DEFAULT_LLM_MODEL. Only used on first call.
        device: Target compute device. Defaults to NPU_PROXY_DEVICE env var
            or "NPU". Only used on first call.
    
    Returns:
        The singleton InferenceEngine instance.
    
    Raises:
        ModelNotFoundError: If the model path does not exist.
        DeviceError: If model loading fails on all devices.
    
    Example:
        >>> # First call initializes the engine
        >>> engine = get_llm_engine(device="NPU")
        >>> engine.warmup()
        >>> 
        >>> # Subsequent calls return the same instance
        >>> same_engine = get_llm_engine()
        >>> assert engine is same_engine
    
    Thread Safety:
        This function is thread-safe. Multiple threads can call it
        concurrently, and all will receive the same engine instance.
    
    Note:
        To change devices or models, call reset_engine() first.
    """
    global _llm_engine
    
    # Use environment variable for device if not specified
    if device is None:
        device = os.environ.get("NPU_PROXY_DEVICE", "NPU")
    
    # Double-checked locking for thread safety
    if _llm_engine is None:
        with _engine_lock:
            if _llm_engine is None:
                if model_path is None:
                    model_path = DEFAULT_MODEL_DIR / DEFAULT_LLM_MODEL
                
                if not Path(model_path).exists():
                    raise ModelNotFoundError(Path(model_path))
                
                _llm_engine = InferenceEngine(model_path, device)
                _loaded_models[_llm_engine.model_name] = _llm_engine
    
    return _llm_engine


def get_loaded_models() -> dict[str, InferenceEngine]:
    """Get dictionary of all currently loaded model engines.
    
    Returns:
        Dictionary mapping model names to their InferenceEngine instances.
        Empty if no models have been loaded.
    
    Example:
        >>> get_llm_engine()  # Load default model
        >>> models = get_loaded_models()
        >>> print(list(models.keys()))
        ['tinyllama-1.1b-chat-int4-ov']
    
    Note:
        This returns a reference to the internal dictionary. Modifications
        will affect the module state - copy if mutation is needed.
    """
    return _loaded_models


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
    return _llm_engine is not None
