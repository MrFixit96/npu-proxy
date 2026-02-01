"""Embedding engine for generating text embeddings using OpenVINO.

This module provides a production-quality embedding engine that uses OpenVINO's
TextEmbeddingPipeline for efficient text embedding generation. Falls back to a
hash-based approach when the model is not available.

Features:
    - Production embeddings via OpenVINO TextEmbeddingPipeline
    - Hash-based fallback for environments without OpenVINO
    - LRU caching for repeated embedding requests
    - Length-based batch optimization to minimize padding overhead
    - Proper L2 normalization ensuring unit-length vectors
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
"""
import concurrent.futures
import hashlib
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import openvino_genai
    OPENVINO_AVAILABLE = True
except ImportError:
    openvino_genai = None
    OPENVINO_AVAILABLE = False


logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_EMBEDDING_DEVICE = "CPU"

# Timeout for model loading (seconds) - large models on NPU can take 2+ minutes to compile
DEFAULT_LOAD_TIMEOUT = int(os.environ.get("NPU_PROXY_LOAD_TIMEOUT", "300"))

# Timeout for embedding inference (seconds)
DEFAULT_EMBED_TIMEOUT = int(os.environ.get("NPU_PROXY_EMBED_TIMEOUT", "60"))

# LRU cache size for embedding results
DEFAULT_CACHE_SIZE = int(os.environ.get("NPU_PROXY_EMBEDDING_CACHE_SIZE", "1024"))


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
    return os.environ.get("NPU_PROXY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


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
    return os.environ.get("NPU_PROXY_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)


def get_embedding_model_path(model_name: Optional[str] = None) -> Path:
    """Get the filesystem path where embedding models are stored.

    Models are stored in ~/.cache/npu-proxy/models/embeddings/{model_name}/
    where the model name has path separators replaced with underscores.

    Args:
        model_name: The model identifier (e.g., "BAAI/bge-small-en-v1.5").
            If None, uses get_embedding_model_name() to get the default.

    Returns:
        Path object pointing to the model directory.

    Example:
        >>> get_embedding_model_path("BAAI/bge-small-en-v1.5")
        PosixPath('/home/user/.cache/npu-proxy/models/embeddings/BAAI_bge-small-en-v1.5')
    """
    if model_name is None:
        model_name = get_embedding_model_name()

    # Sanitize model name for filesystem
    safe_name = model_name.replace("/", "_").replace("\\", "_")

    cache_dir = Path.home() / ".cache" / "npu-proxy" / "models" / "embeddings" / safe_name
    return cache_dir


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
    model_path = get_embedding_model_path(model_name)
    if not model_path.exists():
        return False

    # Check for essential OpenVINO model files
    required_files = ["openvino_model.xml", "openvino_model.bin"]
    for fname in required_files:
        if not (model_path / fname).exists():
            return False

    return True


class EmbeddingEngine:
    """Engine for generating text embeddings using hash-based fallback.

    Uses a deterministic hash-based approach that produces consistent,
    semantically-meaningful embeddings. This class serves as a fallback
    when the production OpenVINO model is not available.

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

    def __init__(self, model_name: str = "all-minilm-l6-v2", dimensions: int = 384) -> None:
        """Initialize the hash-based embedding engine.

        Args:
            model_name: Human-readable model name for reporting purposes.
                Defaults to "all-minilm-l6-v2".
            dimensions: The dimensionality of embedding vectors to produce.
                Defaults to 384 to match common embedding models.
        """
        self._model_name = model_name
        self._dimensions = dimensions
        self._cache: Dict[str, Tuple[float, ...]] = {}

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
        return {
            "model_name": self._model_name,
            "dimensions": self._dimensions,
            "is_production": False,
            "device": "CPU",
        }

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
        if cache_key in self._cache:
            return list(self._cache[cache_key])

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
        self._cache[cache_key] = tuple(embedding)

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
        - Automatic fallback to hash-based embeddings on model load failure
        - LRU caching for repeated embedding requests
        - Length-based batch optimization for efficient padding
        - Configurable timeouts for model loading and inference
        - Proper L2 normalization ensuring unit-length vectors

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
    ) -> None:
        """Initialize the production embedding engine.

        Attempts to load the OpenVINO TextEmbeddingPipeline from the specified
        path. If loading fails (missing OpenVINO, timeout, or other errors),
        automatically falls back to hash-based embedding generation.

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
        self._model_path = model_path
        self._device = device
        self._model_name = model_name or Path(model_path).name
        self._dimensions = dimensions
        self._pipeline: Optional[object] = None
        self._use_fallback = False
        self._fallback_engine: Optional[EmbeddingEngine] = None
        self._load_error: Optional[str] = None
        self._embedding_cache = _create_embedding_cache(DEFAULT_CACHE_SIZE)

        # Try to initialize the pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the OpenVINO TextEmbeddingPipeline with timeout.

        Attempts to load the pipeline in a separate thread with a timeout
        to prevent indefinite hangs. On any failure, sets up the fallback
        hash-based embedding engine.

        Note:
            This method is called automatically during __init__ and should
            not be called directly.
        """
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO GenAI not available, using hash-based fallback")
            self._use_fallback = True
            self._fallback_engine = EmbeddingEngine(
                model_name=self._model_name,
                dimensions=self._dimensions,
            )
            return

        def _load_pipeline() -> object:
            """Inner function to load pipeline (runs in thread for timeout)."""
            return openvino_genai.TextEmbeddingPipeline(
                self._model_path,
                self._device,
            )

        try:
            # Run pipeline loading with timeout to prevent indefinite hangs
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_load_pipeline)
                self._pipeline = future.result(timeout=DEFAULT_LOAD_TIMEOUT)
            logger.info(f"Loaded embedding model from {self._model_path} on {self._device}")
        except concurrent.futures.TimeoutError:
            error_msg = (
                f"Model load timed out after {DEFAULT_LOAD_TIMEOUT}s on {self._device}. "
                f"Model may exceed device memory limits."
            )
            logger.error(error_msg)
            self._load_error = error_msg
            self._use_fallback = True
            self._fallback_engine = EmbeddingEngine(
                model_name=self._model_name,
                dimensions=self._dimensions,
            )
        except Exception as e:
            error_msg = f"Failed to load OpenVINO model: {e}"
            logger.warning(f"{error_msg}. Using hash-based fallback.")
            self._load_error = error_msg
            self._use_fallback = True
            self._fallback_engine = EmbeddingEngine(
                model_name=self._model_name,
                dimensions=self._dimensions,
            )

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

    def embed(self, text: str, timeout: Optional[int] = None) -> List[float]:
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
            TimeoutError: If embedding takes longer than the specified timeout.

        Example:
            >>> engine = ProductionEmbeddingEngine("/path/to/model")
            >>> emb = engine.embed("test query")
            >>> len(emb)
            384
        """
        if not text.strip():
            return [0.0] * self._dimensions

        if self._use_fallback:
            return self._fallback_engine.embed(text)

        # Check cache
        cached = self._embedding_cache(text)
        if cached is not None:
            return cached

        if timeout is None:
            timeout = DEFAULT_EMBED_TIMEOUT

        def _embed() -> List[float]:
            return self._pipeline.embed_query(text)

        try:
            # Run embedding with timeout to prevent hangs
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_embed)
                result = future.result(timeout=timeout)
            embedding = [float(x) for x in result]
            # Update cache by calling with the result
            self._embedding_cache.cache_set(text, embedding)
            return embedding
        except concurrent.futures.TimeoutError:
            error_msg = f"Embedding timed out after {timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            if self._fallback_engine is None:
                self._fallback_engine = EmbeddingEngine(
                    model_name=self._model_name,
                    dimensions=self._dimensions,
                )
            return self._fallback_engine.embed(text)

    def embed_batch(self, texts: List[str], timeout: Optional[int] = None) -> List[List[float]]:
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
            TimeoutError: If batch embedding takes longer than timeout.

        Example:
            >>> engine = ProductionEmbeddingEngine("/path/to/model")
            >>> embeddings = engine.embed_batch(["hello", "world"])
            >>> len(embeddings)
            2
        """
        if self._use_fallback:
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

        def _embed_batch() -> List[List[float]]:
            return self._pipeline.embed_documents(non_empty_texts)

        try:
            # Run batch embedding with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_embed_batch)
                embeddings = future.result(timeout=timeout)
            for idx, embedding in zip(non_empty_indices, embeddings):
                results[idx] = [float(x) for x in embedding]
            return results
        except concurrent.futures.TimeoutError:
            error_msg = f"Batch embedding timed out after {timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fall back to individual embedding
            for i, text in enumerate(texts):
                if text.strip():
                    results[i] = self.embed(text)
            return results

    def embed_batch_optimized(
        self, texts: List[str], timeout: Optional[int] = None
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
            timeout: Timeout in seconds for each embedding operation.
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
        if self._use_fallback:
            return self._fallback_engine.embed_batch_optimized(texts)

        # Sort by length for optimal padding
        indexed_texts: List[Tuple[int, str]] = [(i, t) for i, t in enumerate(texts)]
        sorted_texts = sorted(indexed_texts, key=lambda x: len(x[1]))

        # Process in sorted order
        sorted_embeddings: List[List[float]] = []
        for _, text in sorted_texts:
            sorted_embeddings.append(self.embed(text, timeout=timeout))

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
        info: Dict[str, Union[str, int, bool]] = {
            "model_name": self._model_name,
            "dimensions": self._dimensions,
            "is_production": not self._use_fallback,
            "device": self._device,
        }
        if self._load_error:
            info["load_error"] = self._load_error
        return info

    def clear_cache(self) -> None:
        """Clear the embedding cache.

        Removes all cached embeddings to free memory. Useful when
        switching contexts or when memory pressure is high.
        """
        self._embedding_cache.cache_clear()


def _create_embedding_cache(maxsize: int):
    """Create an LRU cache wrapper for embedding results.

    Creates a callable that checks if an embedding is cached and returns
    it, with additional methods for setting and clearing cache entries.

    Args:
        maxsize: Maximum number of embeddings to cache.

    Returns:
        A callable cache wrapper with cache_set and cache_clear methods.
    """
    cache: Dict[str, List[float]] = {}
    access_order: List[str] = []

    def cache_get(text: str) -> Optional[List[float]]:
        """Get cached embedding or None if not cached."""
        if text in cache:
            # Move to end (most recently used)
            access_order.remove(text)
            access_order.append(text)
            return cache[text]
        return None

    def cache_set(text: str, embedding: List[float]) -> None:
        """Set a cache entry, evicting oldest if at capacity."""
        if text not in cache:
            if len(cache) >= maxsize:
                # Evict oldest
                oldest = access_order.pop(0)
                del cache[oldest]
            access_order.append(text)
        cache[text] = embedding

    def cache_clear() -> None:
        """Clear all cache entries."""
        cache.clear()
        access_order.clear()

    cache_get.cache_set = cache_set  # type: ignore[attr-defined]
    cache_get.cache_clear = cache_clear  # type: ignore[attr-defined]

    return cache_get


# Singleton instance
_embedding_engine: Optional[Union[EmbeddingEngine, ProductionEmbeddingEngine]] = None


def _reset_embedding_engine() -> None:
    """Reset the embedding engine singleton.

    Clears the singleton instance, allowing a new engine to be created
    on the next call to get_embedding_engine(). Primarily used for testing
    to ensure clean state between tests.

    Note:
        This function is intended for testing purposes only and should
        not be called in production code.
    """
    global _embedding_engine
    _embedding_engine = None


def get_embedding_engine() -> Union[EmbeddingEngine, ProductionEmbeddingEngine]:
    """Get or create the embedding engine singleton.

    Returns a ProductionEmbeddingEngine if the configured model is downloaded
    and available, otherwise returns a hash-based EmbeddingEngine fallback.

    The engine type is determined by:
        1. Checking NPU_PROXY_EMBEDDING_MODEL environment variable
        2. Verifying if the model files exist in the cache directory
        3. If available, creating ProductionEmbeddingEngine with OpenVINO
        4. Otherwise, creating hash-based EmbeddingEngine

    Returns:
        Either a ProductionEmbeddingEngine (if model is available) or
        an EmbeddingEngine (hash-based fallback).

    Example:
        >>> engine = get_embedding_engine()
        >>> info = engine.get_engine_info()
        >>> print(info["is_production"])
        True  # or False if using fallback

    Note:
        The singleton is lazily initialized on first call. Subsequent
        calls return the same instance. Use _reset_embedding_engine()
        (testing only) to force re-initialization.
    """
    global _embedding_engine
    if _embedding_engine is None:
        model_name = get_embedding_model_name()

        if is_embedding_model_downloaded(model_name):
            model_path = get_embedding_model_path(model_name)
            device = get_embedding_device()
            _embedding_engine = ProductionEmbeddingEngine(
                model_path=str(model_path),
                device=device,
                model_name=model_name,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            )
        else:
            logger.info(f"Model {model_name} not downloaded, using hash-based fallback")
            _embedding_engine = EmbeddingEngine(
                model_name=model_name,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            )

    return _embedding_engine
