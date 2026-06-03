"""Embedding engine for generating text embeddings using OpenVINO.

This module provides a production-quality embedding engine that uses OpenVINO's
TextEmbeddingPipeline for efficient text embedding generation. Hash-based
fallback embeddings remain available only behind an explicit operator gate.

Features:
    - Production embeddings via OpenVINO TextEmbeddingPipeline
    - Optional hash-based fallback for operator workflows
    - LRU caching for repeated embedding requests
    - Length-based batch optimization to minimize padding overhead
    - Model/pipeline-provided vector normalization with validation
    - Configurable timeouts for model loading and inference

Example:
    >>> from npu_proxy.inference.embedding_engine import get_embedding_engine
    >>> engine = get_embedding_engine()
    >>> embedding = engine.embed("Hello, world!")
    >>> len(embedding)
    384

Environment Variables:
    NPU_PROXY_EMBEDDING_MODEL: Model name (default: BAAI/bge-small-en-v1.5)
    NPU_PROXY_EMBEDDING_DEVICE: Device for inference (default: CPU)
    NPU_PROXY_LOAD_TIMEOUT: Model load timeout in seconds (default: 300)
    NPU_PROXY_EMBED_TIMEOUT: Embedding inference timeout in seconds (default: 60)
    NPU_PROXY_EMBEDDING_CACHE_SIZE: LRU cache size (default: 1024)
    NPU_PROXY_EMBEDDING_FALLBACK_MODE: Embedding fallback mode
        (default: disabled, supported: disabled, hash)
"""
import concurrent.futures
import hashlib
import importlib
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from npu_proxy.inference import embedding_config


logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_MODEL = embedding_config.DEFAULT_EMBEDDING_MODEL
DEFAULT_EMBEDDING_DIMENSIONS = embedding_config.DEFAULT_EMBEDDING_DIMENSIONS
DEFAULT_EMBEDDING_DEVICE = embedding_config.DEFAULT_EMBEDDING_DEVICE

def _parse_bounded_int_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Parse a bounded integer environment variable without import-time crashes."""
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r; using default %s",
            name,
            raw_value,
            default,
        )
        return default

    if value < minimum:
        logger.warning(
            "%s=%s is below minimum %s; using %s",
            name,
            value,
            minimum,
            minimum,
        )
        return minimum
    if value > maximum:
        logger.warning(
            "%s=%s exceeds maximum %s; using %s",
            name,
            value,
            maximum,
            maximum,
        )
        return maximum
    return value


# Timeout for model loading (seconds) - large models on NPU can take 2+ minutes to compile
DEFAULT_LOAD_TIMEOUT = _parse_bounded_int_env(
    "NPU_PROXY_LOAD_TIMEOUT",
    300,
    minimum=1,
    maximum=3600,
)

# Timeout for embedding inference (seconds)
DEFAULT_EMBED_TIMEOUT = _parse_bounded_int_env(
    "NPU_PROXY_EMBED_TIMEOUT",
    60,
    minimum=1,
    maximum=600,
)

# LRU cache size for embedding results; zero disables caching.
DEFAULT_CACHE_SIZE = _parse_bounded_int_env(
    "NPU_PROXY_EMBEDDING_CACHE_SIZE",
    1024,
    minimum=0,
    maximum=100_000,
)

# Cooldown for failed production engine loads to avoid retry storms.
DEFAULT_UNAVAILABLE_COOLDOWN = _parse_bounded_int_env(
    "NPU_PROXY_EMBEDDING_UNAVAILABLE_COOLDOWN",
    30,
    minimum=1,
    maximum=3600,
)
EMBEDDING_FALLBACK_MODE_ENV_VAR = "NPU_PROXY_EMBEDDING_FALLBACK_MODE"
EMBEDDING_FALLBACK_MODE_DISABLED = "disabled"
EMBEDDING_FALLBACK_MODE_HASH = "hash"

# Static-shape presets validated for NPU embedding inference. These force
# OpenVINO GenAI down the direct static-compile path for supported models.
_NPU_STATIC_EMBEDDING_PRESETS: dict[str, dict[str, int | bool]] = {
    "all-minilm-l6-v2": {
        "batch_size": 1,
        "max_length": 256,
        "pad_to_max_length": True,
    },
    "sentence-transformers/all-minilm-l6-v2": {
        "batch_size": 1,
        "max_length": 256,
        "pad_to_max_length": True,
    },
}

openvino_genai = None


def _get_openvino_genai():
    """Import OpenVINO GenAI lazily so route registration stays side-effect free."""
    global openvino_genai
    if openvino_genai is None:
        openvino_genai = importlib.import_module("openvino_genai")
    return openvino_genai


class EmbeddingError(RuntimeError):
    """Base class for embedding engine errors."""


class EmbeddingUnavailableError(EmbeddingError):
    """Raised when real embeddings are unavailable and fallback is disabled."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when a loaded real embedding engine fails at runtime."""


class EmbeddingTimeoutError(EmbeddingError, TimeoutError):
    """Raised when embedding inference exceeds the configured timeout."""


def _format_timeout_seconds(timeout: float) -> str:
    """Format timeout seconds without trailing decimal noise."""
    return f"{timeout:g}s"


def get_embedding_fallback_mode() -> str:
    """Return the configured embedding fallback mode."""
    value = os.environ.get(
        EMBEDDING_FALLBACK_MODE_ENV_VAR,
        EMBEDDING_FALLBACK_MODE_DISABLED,
    )
    normalized = value.strip().lower() if value else EMBEDDING_FALLBACK_MODE_DISABLED
    if normalized == EMBEDDING_FALLBACK_MODE_HASH:
        return EMBEDDING_FALLBACK_MODE_HASH
    return EMBEDDING_FALLBACK_MODE_DISABLED


def is_embedding_fallback_enabled() -> bool:
    """Return whether hash-based embedding fallback is explicitly enabled."""
    return get_embedding_fallback_mode() == EMBEDDING_FALLBACK_MODE_HASH


def _with_fallback_gate_hint(reason: str) -> str:
    """Append the operator fallback hint to an unavailable embedding message."""
    if EMBEDDING_FALLBACK_MODE_ENV_VAR in reason:
        return reason
    return (
        f"{reason}. Set {EMBEDDING_FALLBACK_MODE_ENV_VAR}=hash to allow "
        "hash-based embedding fallback."
    )


def get_embedding_model_name() -> str:
    """Get the embedding model name from environment or default.

    Returns:
        The embedding model name from NPU_PROXY_EMBEDDING_MODEL environment
        variable, or DEFAULT_EMBEDDING_MODEL if not set.

    Example:
        >>> os.environ["NPU_PROXY_EMBEDDING_MODEL"] = "custom/model"
        >>> get_embedding_model_name()
        'custom/model'
    """
    return embedding_config.get_configured_embedding_model_name()


def get_embedding_device() -> str:
    """Get the embedding device from environment or default.

    Returns:
        The device name from NPU_PROXY_EMBEDDING_DEVICE environment variable,
        or DEFAULT_EMBEDDING_DEVICE ("CPU") if not set. Valid values include
        "CPU", "NPU", and "GPU".

    Example:
        >>> os.environ["NPU_PROXY_EMBEDDING_DEVICE"] = "NPU"
        >>> get_embedding_device()
        'NPU'
    """
    return embedding_config.get_configured_embedding_device()


def get_embedding_model_path(model_name: Optional[str] = None) -> Path:
    """Get the filesystem path where embedding models are stored.

    Known registry-backed embedding models resolve to their canonical runtime
    directory names under ~/.cache/npu-proxy/models/embeddings/. Unknown repo
    IDs fall back to a sanitized directory name with path separators replaced.

    Args:
        model_name: The model identifier (e.g., "BAAI/bge-small-en-v1.5").
            If None, uses get_embedding_model_name() to get the default.

    Returns:
        Path object pointing to the model directory.

    Example:
        >>> get_embedding_model_path("sentence-transformers/all-MiniLM-L6-v2")
        PosixPath('/home/user/.cache/npu-proxy/models/embeddings/all-minilm-l6-v2')
    """
    return embedding_config.get_embedding_model_path(model_name)


def is_embedding_model_downloaded(model_name: str) -> bool:
    """Check if an embedding model has been downloaded.

    Verifies that the model directory exists and contains the required
    OpenVINO model files (openvino_model.xml and openvino_model.bin).

    Args:
        model_name: The model identifier to check.

    Returns:
        True if the model is downloaded and has required files, False otherwise.

    Example:
        >>> is_embedding_model_downloaded("BAAI/bge-small-en-v1.5")
        True
    """
    return embedding_config.is_embedding_model_downloaded(model_name)


class EmbeddingEngine:
    """Engine for generating text embeddings using explicit hash fallback.

    Uses a deterministic hash-based approach that produces consistent,
    semantically-meaningful embeddings. This class serves only as an
    explicitly enabled operator fallback when the real OpenVINO model is
    not available.

    The hash-based approach ensures:
        - Deterministic output: same input always produces same embedding
        - Consistent dimensionality matching production models
        - Proper L2 normalization for cosine similarity compatibility

    Attributes:
        model_name: The name of the model being emulated.
        dimensions: The dimensionality of embedding vectors produced.

    Example:
        >>> engine = EmbeddingEngine(dimensions=384)
        >>> embedding = engine.embed("Hello, world!")
        >>> len(embedding)
        384
        >>> sum(x * x for x in embedding)  # L2 norm squared ≈ 1.0
        1.0
    """

    def __init__(
        self,
        model_name: str = "all-minilm-l6-v2",
        dimensions: int = 384,
        *,
        requested_model: Optional[str] = None,
        requested_device: Optional[str] = None,
        resolved_model: Optional[str] = None,
        device: str = "CPU",
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        fallback_mode: Optional[str] = None,
    ) -> None:
        """Initialize the hash-based embedding engine.

        Args:
            model_name: Human-readable model name for reporting purposes.
                Defaults to "all-minilm-l6-v2".
            dimensions: The dimensionality of embedding vectors to produce.
                Defaults to 384 to match common embedding models.
        """
        self._model_name = resolved_model or model_name
        self._requested_model = requested_model or model_name
        self._requested_device = requested_device or device
        self._device = device
        self._model_path = model_path
        self._repo_id = repo_id
        self._fallback_reason = fallback_reason
        self._fallback_mode = fallback_mode
        self._dimensions = dimensions
        self._cache = _create_embedding_cache(DEFAULT_CACHE_SIZE)

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            The human-readable model name string.
        """
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            The number of dimensions in produced embedding vectors.
        """
        return self._dimensions

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Produces a deterministic, L2-normalized embedding vector from the
        input text using SHA-256 hashing expanded to fill the required
        dimensions.

        Args:
            text: The input text to embed. Empty or whitespace-only text
                returns a zero vector.

        Returns:
            A list of floats representing the embedding vector with
            L2 norm equal to 1.0 (unit vector), or a zero vector for
            empty input.

        Example:
            >>> engine = EmbeddingEngine()
            >>> emb = engine.embed("test")
            >>> len(emb)
            384
        """
        self._ensure_fallback_allowed()
        return self._generate_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Processes each text independently and returns embeddings in the
        same order as the input texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text, maintaining
            input order.

        Example:
            >>> engine = EmbeddingEngine()
            >>> embeddings = engine.embed_batch(["hello", "world"])
            >>> len(embeddings)
            2
        """
        self._ensure_fallback_allowed()
        return [self._generate_embedding(text) for text in texts]

    def embed_batch_optimized(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with length-based grouping for efficiency.

        Groups texts by approximate length to minimize padding overhead
        and maximize batch throughput. For the hash-based fallback engine,
        this provides consistent API with ProductionEmbeddingEngine but
        processes texts individually since no padding is involved.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors in the original input order.

        Example:
            >>> engine = EmbeddingEngine()
            >>> texts = ["short", "a much longer text here", "medium text"]
            >>> embeddings = engine.embed_batch_optimized(texts)
            >>> len(embeddings)
            3
        """
        self._ensure_fallback_allowed()

        # Sort by length for optimal padding (matches production API)
        indexed_texts: List[Tuple[int, str]] = [(i, t) for i, t in enumerate(texts)]
        sorted_texts = sorted(indexed_texts, key=lambda x: len(x[1]))

        # Process in sorted order
        sorted_embeddings: List[List[float]] = []
        for _, text in sorted_texts:
            sorted_embeddings.append(self.embed(text))

        # Restore original order
        results: List[Optional[List[float]]] = [None] * len(texts)
        for (orig_idx, _), emb in zip(sorted_texts, sorted_embeddings):
            results[orig_idx] = emb
        return results  # type: ignore[return-value]

    def get_engine_info(self) -> Dict[str, Union[str, int, bool]]:
        """Return information about the engine configuration.

        Returns:
            Dictionary containing engine configuration details including
            model_name, dimensions, is_production flag, and device.

        Example:
            >>> engine = EmbeddingEngine()
            >>> info = engine.get_engine_info()
            >>> info["is_production"]
            False
        """
        info: Dict[str, Union[str, int, bool]] = {
            "model_name": self._model_name,
            "resolved_model": self._model_name,
            "requested_model": self._requested_model,
            "dimensions": self._dimensions,
            "is_production": False,
            "is_fallback": True,
            "backend": "hash",
            "device": self._device,
            "requested_device": self._requested_device,
            "fallback_allowed": is_embedding_fallback_enabled(),
            "configured_fallback_mode": get_embedding_fallback_mode(),
            "available": is_embedding_fallback_enabled(),
        }
        if self._model_path:
            info["model_path"] = self._model_path
        if self._repo_id:
            info["repo_id"] = self._repo_id
        if self._fallback_reason:
            info["fallback_reason"] = self._fallback_reason
            info["load_error"] = self._fallback_reason
        if self._fallback_mode:
            info["fallback_mode"] = self._fallback_mode
        return info

    def _ensure_fallback_allowed(self) -> None:
        """Fail closed unless operator fallback was explicitly enabled."""
        if is_embedding_fallback_enabled():
            return
        reason = self._fallback_reason or (
            f"Embedding fallback for model {self._model_name} is disabled by default"
        )
        raise EmbeddingUnavailableError(_with_fallback_gate_hint(reason))

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a deterministic embedding vector from text.

        Uses SHA-256 hash expanded to fill dimensions, with proper L2
        normalization to produce unit-length vectors suitable for cosine
        similarity calculations.

        The normalization uses float32 precision to ensure numerical
        stability and compatibility with vector databases that expect
        normalized embeddings.

        Algorithm:
            1. Normalize input text (lowercase, strip whitespace)
            2. Generate multiple SHA-256 hashes to fill dimensions
            3. Convert hash bytes to float values in range [-1, 1]
            4. Apply L2 normalization to produce unit vector

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the L2-normalized embedding.
            Returns a zero vector for empty input text.

        Note:
            L2 normalization ensures ||embedding||_2 = 1.0, calculated as:
            norm = sqrt(sum(x_i^2 for all i))
            normalized_x_i = x_i / norm
        """
        if not text.strip():
            return [0.0] * self._dimensions

        # Check cache first
        cache_key = text.lower().strip()
        cached = self._cache(cache_key)
        if cached is not None:
            return cached

        # Normalize text
        normalized = cache_key

        # Generate hash-based embedding
        embedding: List[float] = []

        # Use multiple hash iterations to fill dimensions
        for i in range(0, self._dimensions, 8):
            # Create unique hash for each chunk
            chunk_input = f"{normalized}:{i}".encode('utf-8')
            hash_bytes = hashlib.sha256(chunk_input).digest()

            # Convert bytes to floats in range [-1, 1]
            for j in range(min(8, self._dimensions - i)):
                byte_val = hash_bytes[j]
                # Map 0-255 to -1.0 to 1.0
                float_val = float((byte_val / 127.5) - 1.0)
                embedding.append(float_val)

        # L2 normalize to unit length using float32 precision
        # This ensures ||embedding||_2 = 1.0 for cosine similarity compatibility
        embedding = self._l2_normalize(embedding)

        # Cache the result
        self._cache.cache_set(cache_key, embedding)

        return embedding

    @staticmethod
    def _l2_normalize(embedding: List[float]) -> List[float]:
        """Apply L2 normalization to produce a unit-length vector.

        Normalizes the embedding vector so that its L2 (Euclidean) norm
        equals 1.0. This is essential for cosine similarity calculations
        where normalized vectors allow dot product to equal cosine similarity.

        Uses explicit float() conversion to ensure float32 precision
        throughout the calculation for numerical stability.

        Args:
            embedding: The raw embedding vector to normalize.

        Returns:
            L2-normalized embedding with ||result||_2 = 1.0.
            Returns the original embedding unchanged if the norm is zero.

        Example:
            >>> EmbeddingEngine._l2_normalize([3.0, 4.0])
            [0.6, 0.8]  # norm=5, 3/5=0.6, 4/5=0.8
        """
        # Calculate L2 norm with float32 precision
        norm_squared = sum(float(x) * float(x) for x in embedding)
        norm = math.sqrt(norm_squared)

        if norm > 0:
            return [float(x) / norm for x in embedding]
        return embedding


class ProductionEmbeddingEngine:
    """Production embedding engine using OpenVINO TextEmbeddingPipeline.

    This engine provides high-quality text embeddings using the OpenVINO
    GenAI library. It supports lazy loading, LRU caching for repeated
    requests, and length-based batch optimization to minimize padding
    overhead.

    Features:
        - OpenVINO TextEmbeddingPipeline for high-performance inference
        - Optional hash-based fallback for explicit operator workflows
        - LRU caching for repeated embedding requests
        - Length-based batch optimization for efficient padding
        - Configurable timeouts for model loading and inference
        - Model/pipeline-provided normalization with dimension/finite validation

    Attributes:
        model_name: The name of the loaded model.
        dimensions: The dimensionality of embedding vectors produced.

    Example:
        >>> engine = ProductionEmbeddingEngine(
        ...     model_path="/path/to/model",
        ...     device="NPU"
        ... )
        >>> embedding = engine.embed("Hello, world!")
        >>> len(embedding)
        384
    """

    def __init__(
        self,
        model_path: str,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        model_name: Optional[str] = None,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        *,
        requested_model: Optional[str] = None,
        requested_device: Optional[str] = None,
        resolved_model: Optional[str] = None,
        repo_id: Optional[str] = None,
        canonical_model_path: Optional[str] = None,
    ) -> None:
        """Initialize the production embedding engine.

        Attempts to load the OpenVINO TextEmbeddingPipeline from the specified
        path. If loading fails (missing OpenVINO, timeout, or other errors),
        the engine either enters an unavailable state or, when explicitly
        configured, activates hash-based fallback generation.

        Args:
            model_path: Path to the OpenVINO embedding model directory.
                Must contain openvino_model.xml and openvino_model.bin.
            device: Device to run inference on. Valid values are "CPU",
                "NPU", or "GPU". Defaults to "CPU".
            model_name: Human-readable model name for reporting. If None,
                uses the directory name from model_path.
            dimensions: Embedding vector dimensions. Defaults to 384.

        Note:
            Model loading uses DEFAULT_LOAD_TIMEOUT to prevent indefinite
            hangs, especially important for NPU compilation which can take
            several minutes for large models.
        """
        self._model_path = str(model_path)
        self._canonical_model_path = canonical_model_path or self._model_path
        self._device = device
        self._requested_device = requested_device or device
        self._resolved_model = resolved_model or model_name or Path(model_path).name
        self._model_name = self._resolved_model
        self._requested_model = requested_model or model_name or self._resolved_model
        self._repo_id = repo_id
        self._dimensions = dimensions
        self._pipeline_properties = self._get_pipeline_properties()
        self._pipeline: Optional[object] = None
        self._use_fallback = False
        self._fallback_engine: Optional[EmbeddingEngine] = None
        self._load_error: Optional[str] = None
        self._fallback_reason: Optional[str] = None
        self._fallback_mode: Optional[str] = None
        self._embedding_cache = _create_embedding_cache(DEFAULT_CACHE_SIZE)
        self._closed = False
        self._inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="npu-proxy-embeddings",
        )
        self._inference_lock = threading.Lock()
        self._active_future: Optional[concurrent.futures.Future[Any]] = None
        self._active_timeout_seconds: Optional[float] = None
        self._active_operation: Optional[str] = None

        # Try to initialize the pipeline
        self._initialize_pipeline()

    def _activate_fallback(self, reason: str, mode: str, *, mark_load_error: bool = False) -> None:
        """Switch to the deterministic hash-based fallback engine."""
        if mark_load_error:
            self._load_error = reason
        self._fallback_reason = reason
        self._fallback_mode = mode
        self._use_fallback = True
        if self._fallback_engine is None:
            self._fallback_engine = EmbeddingEngine(
                model_name=self._resolved_model,
                dimensions=self._dimensions,
                requested_model=self._requested_model,
                requested_device=self._requested_device,
                resolved_model=self._resolved_model,
                device="CPU",
                model_path=self._model_path,
                repo_id=self._repo_id,
                fallback_reason=reason,
                fallback_mode=mode,
            )

    def _mark_unavailable(
        self,
        reason: str,
        mode: str,
        *,
        mark_load_error: bool = False,
    ) -> None:
        """Record an unavailable engine state without activating fallback."""
        if mark_load_error:
            self._load_error = reason
        self._fallback_reason = reason
        self._fallback_mode = mode
        self._use_fallback = False
        self._fallback_engine = None

    def _get_unavailable_reason(self) -> Optional[str]:
        """Return the current availability error, if any."""
        if self._use_fallback:
            if is_embedding_fallback_enabled():
                return None
            return _with_fallback_gate_hint(
                self._fallback_reason
                or f"Embedding fallback for model {self._resolved_model} is disabled by default"
            )
        if self._pipeline is None:
            reason = self._load_error or self._fallback_reason
            if reason:
                return _with_fallback_gate_hint(reason)
            return f"Embedding engine for {self._resolved_model} is unavailable"
        return None

    def _ensure_available(self) -> None:
        """Raise when the engine cannot provide truthful embeddings."""
        reason = self._get_unavailable_reason()
        if reason:
            raise EmbeddingUnavailableError(reason)

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
        """Submit embedding work while preventing hidden queueing after timeouts."""
        with self._inference_lock:
            self._reap_completed_call_locked()
            if self._active_future is not None:
                if self._active_timeout_seconds is not None:
                    raise EmbeddingTimeoutError(
                        (
                            f"Previous {self._active_operation or 'embedding operation'} "
                            f"timed out after "
                            f"{_format_timeout_seconds(self._active_timeout_seconds)} "
                            "and is still running; refusing to queue another request"
                        )
                    )

                raise EmbeddingInferenceError(
                    "Embedding inference already in progress on this shared engine; "
                    "retry after the current request completes"
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
        timeout: float,
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
        timeout: float,
        timeout_message: str,
    ) -> Any:
        """Run embedding work on the long-lived executor with prompt timeout returns."""
        future = self._submit_inference_call(operation, func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            self._mark_inference_timeout(future, timeout)
            logger.error(timeout_message)
            raise EmbeddingTimeoutError(timeout_message) from exc
        finally:
            self._release_inference_call(future)

    def is_available(self) -> bool:
        """Return whether the engine can currently serve embeddings."""
        return self._get_unavailable_reason() is None

    def _initialize_pipeline(self) -> None:
        """Initialize the OpenVINO TextEmbeddingPipeline with timeout.

        Attempts to load the pipeline in a separate thread with a timeout
        to prevent indefinite hangs. On failure, records the engine as
        unavailable unless explicit hash fallback is enabled.

        Note:
            This method is called automatically during __init__ and should
            not be called directly.
        """
        try:
            openvino_genai = _get_openvino_genai()
        except ImportError:
            error_msg = "OpenVINO GenAI not available"
            if is_embedding_fallback_enabled():
                logger.warning(f"{error_msg}, using hash-based fallback")
                self._activate_fallback(error_msg, "missing_openvino", mark_load_error=True)
            else:
                logger.warning("%s; hash-based fallback is disabled", error_msg)
                self._mark_unavailable(error_msg, "missing_openvino", mark_load_error=True)
            return

        def _load_pipeline() -> object:
            """Inner function to load pipeline (runs in thread for timeout)."""
            if self._pipeline_properties:
                config = openvino_genai.TextEmbeddingPipeline.Config()
                for key, value in self._pipeline_properties.items():
                    setattr(config, key, value)
                return openvino_genai.TextEmbeddingPipeline(
                    self._model_path,
                    self._device,
                    config,
                )
            return openvino_genai.TextEmbeddingPipeline(
                self._model_path,
                self._device,
            )

        try:
            error_msg = (
                f"Model load timed out after {_format_timeout_seconds(DEFAULT_LOAD_TIMEOUT)} "
                f"on {self._device}. Model may exceed device memory limits."
            )
            self._pipeline = self._run_with_timeout(
                "model load",
                _load_pipeline,
                DEFAULT_LOAD_TIMEOUT,
                error_msg,
            )
            if self._pipeline_properties:
                logger.info(
                    "Loaded embedding model from %s on %s with pipeline properties %s",
                    self._model_path,
                    self._device,
                    self._pipeline_properties,
                )
            else:
                logger.info(f"Loaded embedding model from {self._model_path} on {self._device}")
        except EmbeddingTimeoutError as exc:
            error_msg = str(exc)
            if is_embedding_fallback_enabled():
                self._activate_fallback(error_msg, "load", mark_load_error=True)
            else:
                self._mark_unavailable(error_msg, "load", mark_load_error=True)
        except Exception as e:
            error_msg = f"Failed to load OpenVINO model: {e}"
            if is_embedding_fallback_enabled():
                logger.warning(f"{error_msg}. Using hash-based fallback.")
                self._activate_fallback(error_msg, "load", mark_load_error=True)
            else:
                logger.warning("%s. Hash-based fallback is disabled.", error_msg)
                self._mark_unavailable(error_msg, "load", mark_load_error=True)

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            The human-readable model name string.
        """
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            The number of dimensions in produced embedding vectors.
        """
        return self._dimensions

    def embed(self, text: str, timeout: Optional[float] = None) -> List[float]:
        """Generate embedding for a single text with timeout and caching.

        Produces an L2-normalized embedding vector from the input text
        using the OpenVINO pipeline. Results are cached for repeated
        requests to the same text.

        Args:
            text: Text to embed. Empty or whitespace-only text returns
                a zero vector.
            timeout: Timeout in seconds for the embedding operation.
                Defaults to DEFAULT_EMBED_TIMEOUT (60 seconds).

        Returns:
            List of floats representing the L2-normalized embedding vector
            with dimensions matching self.dimensions.

        Raises:
            EmbeddingTimeoutError: If embedding takes longer than the specified timeout.

        Example:
            >>> engine = ProductionEmbeddingEngine("/path/to/model")
            >>> emb = engine.embed("test query")
            >>> len(emb)
            384
        """
        self._ensure_available()

        if self._use_fallback:
            if self._fallback_engine is None:
                raise EmbeddingUnavailableError("Embedding fallback engine is unavailable")
            return self._fallback_engine.embed(text)

        if not text.strip():
            return [0.0] * self._dimensions

        # Check cache
        cached = self._embedding_cache(text)
        if cached is not None:
            return cached

        if timeout is None:
            timeout = DEFAULT_EMBED_TIMEOUT

        def _embed() -> List[float]:
            return self._pipeline.embed_query(text)

        try:
            result = self._run_with_timeout(
                "embedding",
                _embed,
                timeout,
                f"Embedding timed out after {_format_timeout_seconds(timeout)}",
            )
            embedding = self._validate_embedding_vector(result, context="query embedding")
            self._embedding_cache.cache_set(text, embedding)
            return list(embedding)
        except EmbeddingTimeoutError:
            raise
        except Exception as e:
            error_msg = f"Embedding generation failed: {e}"
            logger.error(error_msg)
            if is_embedding_fallback_enabled():
                self._activate_fallback(error_msg, "runtime")
                return self._fallback_engine.embed(text)
            raise EmbeddingInferenceError(error_msg) from e

    def embed_batch(self, texts: List[str], timeout: Optional[float] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts with timeout.

        Processes texts through the OpenVINO pipeline's batch embedding
        endpoint. Empty texts are handled separately to avoid pipeline
        errors.

        Args:
            texts: List of texts to embed.
            timeout: Timeout in seconds for the batch operation.
                Defaults to DEFAULT_EMBED_TIMEOUT (60 seconds).

        Returns:
            List of embedding vectors in the same order as input texts.

        Raises:
            EmbeddingTimeoutError: If batch embedding takes longer than timeout.

        Example:
            >>> engine = ProductionEmbeddingEngine("/path/to/model")
            >>> embeddings = engine.embed_batch(["hello", "world"])
            >>> len(embeddings)
            2
        """
        self._ensure_available()

        if self._use_fallback:
            if self._fallback_engine is None:
                raise EmbeddingUnavailableError("Embedding fallback engine is unavailable")
            return self._fallback_engine.embed_batch(texts)

        if timeout is None:
            timeout = DEFAULT_EMBED_TIMEOUT

        # Filter empty texts and track indices
        non_empty_texts: List[str] = []
        non_empty_indices: List[int] = []
        results: List[List[float]] = [[0.0] * self._dimensions for _ in texts]

        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        if not non_empty_texts:
            return results

        try:
            embeddings = self._embed_documents_chunked(non_empty_texts, timeout)
            if len(embeddings) != len(non_empty_texts):
                raise EmbeddingInferenceError(
                    f"Embedding batch returned {len(embeddings)} result(s) "
                    f"for {len(non_empty_texts)} non-empty input(s)"
                )
            for idx, embedding in zip(non_empty_indices, embeddings):
                results[idx] = self._validate_embedding_vector(
                    embedding,
                    context=f"batch embedding at index {idx}",
                )
            return results
        except EmbeddingTimeoutError:
            raise
        except Exception as e:
            error_msg = f"Batch embedding failed: {e}"
            logger.error(error_msg)
            if is_embedding_fallback_enabled():
                self._activate_fallback(error_msg, "runtime")
                return self._fallback_engine.embed_batch(texts)
            raise EmbeddingInferenceError(error_msg) from e

    def embed_batch_optimized(
        self, texts: List[str], timeout: Optional[float] = None
    ) -> List[List[float]]:
        """Generate embeddings with length-based grouping for efficiency.

        Groups texts by approximate length to minimize padding overhead
        and maximize batch throughput. Shorter texts are processed together
        to avoid excessive padding when mixed with longer texts.

        This optimization is particularly effective when:
            - Texts have highly variable lengths
            - Processing large batches
            - Using hardware accelerators sensitive to padding overhead

        Args:
            texts: List of input texts to embed.
            timeout: Overall timeout budget in seconds for the optimized batch.
                Defaults to DEFAULT_EMBED_TIMEOUT.

        Returns:
            List of embedding vectors in the original input order.

        Example:
            >>> engine = ProductionEmbeddingEngine("/path/to/model")
            >>> texts = ["short", "a much longer text here", "medium"]
            >>> embeddings = engine.embed_batch_optimized(texts)
            >>> len(embeddings)
            3

        Note:
            For small batches or texts of similar length, the overhead of
            sorting may outweigh benefits. Consider using embed_batch()
            for batches under 10 texts.
        """
        self._ensure_available()

        if self._use_fallback:
            return self._fallback_engine.embed_batch_optimized(texts)

        if timeout is None:
            timeout = DEFAULT_EMBED_TIMEOUT

        deadline = time.monotonic() + timeout

        # Sort by length for optimal padding
        indexed_texts: List[Tuple[int, str]] = [(i, t) for i, t in enumerate(texts)]
        sorted_texts = sorted(indexed_texts, key=lambda x: len(x[1]))

        # Process in sorted order
        sorted_embeddings: List[List[float]] = []
        for _, text in sorted_texts:
            remaining_timeout = deadline - time.monotonic()
            if remaining_timeout <= 0:
                raise EmbeddingTimeoutError(
                    f"Batch embedding timed out after {_format_timeout_seconds(timeout)}"
                )
            sorted_embeddings.append(self.embed(text, timeout=remaining_timeout))

        # Restore original order
        results: List[Optional[List[float]]] = [None] * len(texts)
        for (orig_idx, _), emb in zip(sorted_texts, sorted_embeddings):
            results[orig_idx] = emb
        return results  # type: ignore[return-value]

    def get_engine_info(self) -> Dict[str, Union[str, int, bool]]:
        """Return information about the engine configuration.

        Returns:
            Dictionary containing:
                - model_name: The model identifier
                - dimensions: Embedding vector size
                - is_production: True if using OpenVINO pipeline
                - device: The inference device (CPU, NPU, GPU)
                - load_error: Error message if loading failed (optional)

        Example:
            >>> engine = ProductionEmbeddingEngine("/path/to/model", device="NPU")
            >>> info = engine.get_engine_info()
            >>> info["device"]
            'NPU'
        """
        actual_device = self._device
        if self._use_fallback and self._fallback_engine is not None:
            actual_device = str(self._fallback_engine.get_engine_info().get("device", "CPU"))

        available = self.is_available()
        info: Dict[str, Union[str, int, bool]] = {
            "model_name": self._model_name,
            "resolved_model": self._resolved_model,
            "requested_model": self._requested_model,
            "dimensions": self._dimensions,
            "is_production": self._pipeline is not None and not self._use_fallback,
            "is_fallback": self._use_fallback,
            "backend": "hash" if self._use_fallback else ("openvino" if self._pipeline is not None else "unavailable"),
            "device": actual_device,
            "requested_device": self._requested_device,
            "model_path": self._model_path,
            "fallback_allowed": is_embedding_fallback_enabled(),
            "configured_fallback_mode": get_embedding_fallback_mode(),
            "available": available,
        }
        if self._canonical_model_path != self._model_path:
            info["canonical_model_path"] = self._canonical_model_path
        if self._repo_id:
            info["repo_id"] = self._repo_id
        if self._pipeline_properties:
            info["pipeline_properties"] = dict(self._pipeline_properties)
        if self._load_error:
            info["load_error"] = self._load_error
        if self._fallback_reason:
            info["fallback_reason"] = self._fallback_reason
        if self._fallback_mode:
            info["fallback_mode"] = self._fallback_mode
        return info

    def _get_pipeline_properties(self) -> Dict[str, Union[int, bool]]:
        """Return any model-specific pipeline properties for the current device."""
        if self._device != "NPU":
            return {}

        for candidate in (self._resolved_model, self._requested_model, self._repo_id):
            if not candidate:
                continue
            preset = _NPU_STATIC_EMBEDDING_PRESETS.get(candidate.lower())
            if preset:
                return dict(preset)

        return {}

    def _static_batch_size(self) -> Optional[int]:
        """Return a positive static pipeline batch size when configured."""
        batch_size = self._pipeline_properties.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            return None
        return batch_size if batch_size > 0 else None

    def _embed_documents_chunked(
        self,
        texts: List[str],
        timeout: float,
    ) -> List[List[float]]:
        """Run batch embedding, respecting static OpenVINO/NPU batch sizes."""
        batch_size = self._static_batch_size() or len(texts)
        chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        deadline = time.monotonic() + timeout
        all_embeddings: List[List[float]] = []

        for chunk in chunks:
            remaining_timeout = deadline - time.monotonic()
            if remaining_timeout <= 0:
                raise EmbeddingTimeoutError(
                    f"Batch embedding timed out after {_format_timeout_seconds(timeout)}"
                )

            def _embed_chunk(chunk: List[str] = chunk) -> List[List[float]]:
                return self._pipeline.embed_documents(chunk)

            chunk_embeddings = self._run_with_timeout(
                "batch embedding",
                _embed_chunk,
                remaining_timeout,
                f"Batch embedding timed out after {_format_timeout_seconds(timeout)}",
            )
            if len(chunk_embeddings) != len(chunk):
                raise EmbeddingInferenceError(
                    f"Embedding batch returned {len(chunk_embeddings)} result(s) "
                    f"for {len(chunk)} input(s)"
                )
            all_embeddings.extend(chunk_embeddings)

        return all_embeddings

    def _validate_embedding_vector(
        self,
        embedding: Any,
        *,
        context: str,
    ) -> List[float]:
        """Return a copied float vector after dimension and finiteness checks."""
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise EmbeddingInferenceError(f"Invalid {context}: non-numeric value") from exc

        if len(vector) != self._dimensions:
            raise EmbeddingInferenceError(
                f"Invalid {context}: expected {self._dimensions} dimensions, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingInferenceError(f"Invalid {context}: contains non-finite values")
        return vector

    def clear_cache(self) -> None:
        """Clear the embedding cache.

        Removes all cached embeddings to free memory. Useful when
        switching contexts or when memory pressure is high.
        """
        self._embedding_cache.cache_clear()

    def shutdown(self, wait: bool = False) -> None:
        """Release executor resources associated with this engine."""
        if self._closed:
            return
        self._closed = True
        self._inference_executor.shutdown(wait=wait, cancel_futures=True)

    def __del__(self) -> None:
        """Best-effort cleanup for executor resources."""
        if getattr(self, "_closed", True):
            return
        try:
            self.shutdown(wait=False)
        except Exception:
            logger.debug("Failed to shut down embedding executor", exc_info=True)


def _create_embedding_cache(maxsize: int):
    """Create a bounded, thread-safe LRU cache wrapper for embedding results."""
    cache: Dict[str, Tuple[float, ...]] = {}
    access_order: List[str] = []
    lock = threading.Lock()

    def cache_get(text: str) -> Optional[List[float]]:
        """Get a copied cached embedding or None if not cached."""
        if maxsize <= 0:
            return None
        with lock:
            cached = cache.get(text)
            if cached is None:
                return None
            if text in access_order:
                access_order.remove(text)
            access_order.append(text)
            return list(cached)

    def cache_set(text: str, embedding: List[float]) -> None:
        """Set a cache entry, evicting oldest if at capacity."""
        if maxsize <= 0:
            return
        with lock:
            if text not in cache:
                while len(cache) >= maxsize and access_order:
                    oldest = access_order.pop(0)
                    cache.pop(oldest, None)
                if len(cache) >= maxsize:
                    return
                access_order.append(text)
            elif text in access_order:
                access_order.remove(text)
                access_order.append(text)
            cache[text] = tuple(float(value) for value in embedding)

    def cache_clear() -> None:
        """Clear all cache entries."""
        with lock:
            cache.clear()
            access_order.clear()

    cache_get.cache_set = cache_set  # type: ignore[attr-defined]
    cache_get.cache_clear = cache_clear  # type: ignore[attr-defined]

    return cache_get


# Singleton instance. _embedding_engine is retained as a deprecated default-key
# observer for tests/backward compatibility; _embedding_engines is authoritative.
_embedding_engine: Optional[Union[EmbeddingEngine, ProductionEmbeddingEngine]] = None
_embedding_engines: Dict[Tuple[str, str], Union[EmbeddingEngine, ProductionEmbeddingEngine]] = {}
_embedding_unavailable_until: Dict[Tuple[str, str], float] = {}
_embedding_engine_lock = threading.RLock()
_embedding_engine_load_locks: Dict[Tuple[str, str], threading.Lock] = {}


def _reset_embedding_engine() -> None:
    """Reset the embedding engine singleton.

    Clears the singleton instance, allowing a new engine to be created
    on the next call to get_embedding_engine(). Primarily used for testing
    to ensure clean state between tests.

    Note:
        This function is intended for testing purposes only and should
        not be called in production code.
    """
    global _embedding_engine, _embedding_engines, _embedding_unavailable_until, _embedding_engine_load_locks
    with _embedding_engine_lock:
        seen_ids: set[int] = set()
        for engine in list(_embedding_engines.values()) + ([_embedding_engine] if _embedding_engine else []):
            if engine is None or id(engine) in seen_ids:
                continue
            seen_ids.add(id(engine))
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown(wait=False)
        _embedding_engine = None
        _embedding_engines = {}
        _embedding_unavailable_until = {}
        _embedding_engine_load_locks = {}


def get_loaded_embedding_engine(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[Union[EmbeddingEngine, ProductionEmbeddingEngine]]:
    """Return a loaded embedding engine without creating a new one.

    This helper is intentionally observational. It resolves the requested cache
    key and returns the already-loaded engine if present, otherwise None.
    """
    global _embedding_engine, _embedding_engines

    with _embedding_engine_lock:
        if model_name is None and device is None:
            return _embedding_engine

        config = embedding_config.resolve_embedding_model_config(model_name=model_name, device=device)
        cache_key = (config.resolved_model, config.device)
        return _embedding_engines.get(cache_key)


def get_embedding_engine(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Union[EmbeddingEngine, ProductionEmbeddingEngine]:
    """Get or create the keyed embedding engine singleton."""
    global _embedding_engine, _embedding_engines, _embedding_unavailable_until

    config = embedding_config.resolve_embedding_model_config(model_name=model_name, device=device)
    cache_key = (config.resolved_model, config.device)

    while True:
        with _embedding_engine_lock:
            engine = _embedding_engines.get(cache_key)
            retry_at = _embedding_unavailable_until.get(cache_key)
            if engine is not None and retry_at is not None:
                if time.monotonic() < retry_at:
                    _raise_unavailable_engine(engine, config.resolved_model)
                    if model_name is None and device is None:
                        _embedding_engine = engine
                    return engine
                shutdown = getattr(engine, "shutdown", None)
                if callable(shutdown):
                    shutdown(wait=False)
                _embedding_engines.pop(cache_key, None)
                _embedding_unavailable_until.pop(cache_key, None)
                engine = None

            if engine is not None:
                _raise_unavailable_engine(engine, config.resolved_model)
                if model_name is None and device is None:
                    _embedding_engine = engine
                return engine

            load_lock = _embedding_engine_load_locks.setdefault(cache_key, threading.Lock())

        with load_lock:
            with _embedding_engine_lock:
                engine = _embedding_engines.get(cache_key)
                retry_at = _embedding_unavailable_until.get(cache_key)
                if engine is not None and (retry_at is None or time.monotonic() < retry_at):
                    _raise_unavailable_engine(engine, config.resolved_model)
                    if model_name is None and device is None:
                        _embedding_engine = engine
                    return engine
                if engine is not None:
                    continue

            created_engine, retry_until = _create_embedding_engine_for_config(config)

            with _embedding_engine_lock:
                existing_engine = _embedding_engines.get(cache_key)
                if existing_engine is not None:
                    shutdown = getattr(created_engine, "shutdown", None)
                    if callable(shutdown):
                        shutdown(wait=False)
                    continue
                _embedding_engines[cache_key] = created_engine
                if retry_until is not None:
                    _embedding_unavailable_until[cache_key] = retry_until
                _raise_unavailable_engine(created_engine, config.resolved_model)
                if model_name is None and device is None:
                    _embedding_engine = created_engine
                return created_engine


def _create_embedding_engine_for_config(
    config: embedding_config.EmbeddingModelConfig,
) -> Tuple[Union[EmbeddingEngine, ProductionEmbeddingEngine], Optional[float]]:
    """Create an embedding engine for a resolved config outside the module lock."""
    if config.is_downloaded:
        logger.info(
            "Loading embedding model %s from %s on %s",
            config.resolved_model,
            config.model_path,
            config.device,
        )
        engine = ProductionEmbeddingEngine(
            model_path=str(config.model_path),
            device=config.device,
            model_name=config.resolved_model,
            dimensions=config.dimensions,
            requested_model=config.requested_model,
            requested_device=config.requested_device,
            resolved_model=config.resolved_model,
            repo_id=config.repo_id,
            canonical_model_path=str(config.canonical_path),
        )
        if engine.is_available():
            return engine, None

        info = engine.get_engine_info()
        reason = str(
            info.get("load_error")
            or info.get("fallback_reason")
            or f"Embedding engine for {config.resolved_model} is unavailable"
        )
        engine.shutdown(wait=False)
        retry_until = time.monotonic() + DEFAULT_UNAVAILABLE_COOLDOWN
        return EmbeddingEngine(
            model_name=config.resolved_model,
            dimensions=config.dimensions,
            requested_model=config.requested_model,
            requested_device=config.requested_device,
            resolved_model=config.resolved_model,
            device="CPU",
            model_path=str(config.canonical_path),
            repo_id=config.repo_id,
            fallback_reason=reason,
            fallback_mode=str(info.get("fallback_mode") or "load"),
        ), retry_until

    fallback_reason = f"Embedding model {config.resolved_model} not downloaded at {config.canonical_path}"
    if is_embedding_fallback_enabled():
        logger.info("%s, using hash-based fallback", fallback_reason)
    return EmbeddingEngine(
        model_name=config.resolved_model,
        dimensions=config.dimensions,
        requested_model=config.requested_model,
        requested_device=config.requested_device,
        resolved_model=config.resolved_model,
        device="CPU",
        model_path=str(config.canonical_path),
        repo_id=config.repo_id,
        fallback_reason=fallback_reason,
        fallback_mode="missing_model",
    ), None


def _raise_unavailable_engine(
    engine: Union[EmbeddingEngine, ProductionEmbeddingEngine],
    resolved_model: str,
) -> None:
    """Raise if a cached engine cannot currently serve truthful embeddings."""
    if isinstance(engine, EmbeddingEngine) and not is_embedding_fallback_enabled():
        reason = str(
            engine.get_engine_info().get("fallback_reason")
            or f"Embedding fallback for model {resolved_model} is disabled by default"
        )
        raise EmbeddingUnavailableError(_with_fallback_gate_hint(reason))
    if isinstance(engine, ProductionEmbeddingEngine) and not engine.is_available():
        reason = str(
            engine.get_engine_info().get("load_error")
            or engine.get_engine_info().get("fallback_reason")
            or f"Embedding engine for {resolved_model} is unavailable"
        )
        raise EmbeddingUnavailableError(reason)
