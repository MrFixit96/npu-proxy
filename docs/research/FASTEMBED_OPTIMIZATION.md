# Embedding Engine Best Practices Research

> Research compiled from OpenVINO GenAI, Sentence Transformers, FastEmbed, and Infinity

---

## 1. Batch Processing Optimization

### Pattern: Dynamic Batching with Priority Queue

**Problem it solves:** Efficiently process variable-sized embedding requests by batching them dynamically based on queue depth and optimal batch sizes.

**Source:** Infinity (`infinity_emb/inference/batch_handler.py`)

```python
class BatchHandler:
    def __init__(
        self,
        model_replicas: list["BaseTypeHint"],
        max_batch_size: int,
        max_queue_wait: int = 32_000,
        batch_delay: float = 5e-3,
        vector_disk_cache_path: str = "",
    ):
        self._queue_prio = CustomFIFOQueue()
        self._publish_to_model_queue: Queue = Queue(8)
        self._result_queue: Queue = Queue(8)
        self.max_batch_size = max_batch_size
        self.batch_delay = batch_delay

    def _publish_towards_model(self):
        """Worker that moves batches from priority_queue towards model."""
        max_n_batches = 8
        while not self._shutdown.is_set():
            if not self._publish_to_model_queue.empty() and (
                self._publish_to_model_queue.full()
                or (len(self._queue_prio) < self.max_batch_size * max_n_batches)
            ):
                # Patience: wait if queue still processing
                time.sleep(self.batch_delay)
                continue
            
            batches = self._queue_prio.pop_optimal_batches(
                self.max_batch_size, max_n_batches
            )
            for batch in batches:
                self._publish_to_model_queue.put(batch, timeout=QUEUE_TIMEOUT)
```

**How to apply to npu-proxy:**
- Implement a priority queue that groups requests by similar token lengths
- Use `batch_delay` parameter to allow accumulation of requests before processing
- Monitor queue depth to switch between latency-optimized and throughput-optimized modes

---

### Pattern: Generator-Based Batch Processing

**Problem it solves:** Memory-efficient processing of large datasets by yielding embeddings progressively.

**Source:** FastEmbed (`fastembed/text/text_embedding.py`)

```python
def embed(
    self,
    documents: str | Iterable[str],
    batch_size: int = 256,
    parallel: int | None = None,
    **kwargs: Any,
) -> Iterable[NumpyArray]:
    """
    Encode a list of documents into list of embeddings.
    
    Args:
        documents: Iterator of documents or single document
        batch_size: Batch size for encoding (higher = more memory, faster)
        parallel:
            If > 1, data-parallel encoding for offline large datasets
            If 0, use all available cores
            If None, use default onnxruntime threading
    
    Returns:
        Generator yielding embeddings, one per document
    """
    yield from self.model.embed(documents, batch_size, parallel, **kwargs)
```

**How to apply to npu-proxy:**
- Return generator/iterator instead of list for large batch requests
- Allow caller to control batch_size via API parameter
- Support `parallel` option for CPU-parallel inference when using ONNX backend

---

### Pattern: Three-Stage Pipeline (Pre/Core/Post)

**Problem it solves:** Maximize GPU utilization by overlapping CPU preprocessing, GPU inference, and CPU postprocessing.

**Source:** Infinity (`infinity_emb/transformer/embedder/sentence_transformer.py`)

```python
class SentenceTransformerPatched(SentenceTransformer, BaseEmbedder):
    
    def encode_pre(self, sentences) -> dict[str, "Tensor"]:
        """Stage 1: Tokenization on CPU"""
        features = self.tokenize(sentences)
        return features

    def encode_core(self, features: dict[str, "Tensor"]) -> "Tensor":
        """Stage 2: Forward pass on GPU/NPU"""
        with torch.no_grad():
            features = util.batch_to_device(features, self.device)
            out: dict[str, "Tensor"] = self.forward(features)
            out_features = out["sentence_embedding"].detach().cpu()
        return out_features

    def encode_post(self, out_features: "Tensor") -> "EmbeddingReturnType":
        """Stage 3: Normalization on CPU"""
        with torch.inference_mode():
            embeddings = out_features.to(torch.float32)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.numpy()
        return embeddings_np
```

**How to apply to npu-proxy:**
- Separate OpenVINO inference into distinct pre/core/post stages
- Run tokenization in separate thread from inference
- Use thread pools for pre/post to keep NPU fed continuously

---

## 2. Async Embedding Patterns

### Pattern: Async Engine with Context Manager

**Problem it solves:** Clean lifecycle management of embedding resources with async startup/shutdown.

**Source:** Infinity (`infinity_emb/engine.py`)

```python
class AsyncEmbeddingEngine:
    def __init__(self, engine_args: EngineArgs):
        self.running = False
        self._running_semaphore: Optional[Semaphore] = None
        self._model_replicas, self._min_inference_t, self._max_inference_t = select_model(engine_args)

    async def astart(self):
        """Startup engine with semaphore protection"""
        if self._running_semaphore is None:
            self._running_semaphore = Semaphore(1)
        async with self._running_semaphore:
            if not self.running:
                self.running = True
                self._batch_handler = BatchHandler(...)
                await self._batch_handler.spawn()

    async def astop(self):
        """Stop engine safely"""
        if self._running_semaphore is None:
            return
        async with self._running_semaphore:
            if self.running:
                self.running = False
                await self._batch_handler.shutdown()

    async def __aenter__(self):
        await self.astart()

    async def __aexit__(self, *args):
        await self.astop()

    async def embed(self, sentences: list[str]) -> tuple[list[EmbeddingReturnType], int]:
        self._assert_running()
        embeddings, usage = await self._batch_handler.embed(sentences=sentences)
        return embeddings, usage
```

**How to apply to npu-proxy:**
- Wrap OpenVINO model in async context manager
- Use semaphore to prevent concurrent startup/shutdown
- Provide both `async with` and explicit `astart()/astop()` patterns

---

### Pattern: Future-Based Result Waiting

**Problem it solves:** Allow multiple concurrent requests to wait for their individual results from a shared batch.

**Source:** Infinity (`infinity_emb/primitives.py`)

```python
@dataclass(order=True)
class EmbeddingInner(AbstractInner):
    content: EmbeddingSingle
    future: asyncio.Future
    embedding: Optional["EmbeddingReturnType"] = None

    async def complete(self, result: EmbeddingReturnType) -> None:
        """Mark future for completion - call from same thread as created"""
        self.embedding = result
        if self.embedding is None:
            raise ValueError("embedding is None")
        try:
            self.future.set_result(self.embedding)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> EmbeddingReturnType:
        """Wait for future to complete and return result"""
        await self.future
        assert self.embedding is not None
        return self.embedding

# Usage in BatchHandler:
async def _schedule(self, list_queueitem: Sequence[AbstractSingle]) -> tuple[list[Any], int]:
    new_prioqueue = []
    for re, p in zip(list_queueitem, prios):
        inner = inner_item(content=re, future=self.loop.create_future())
        item = PrioritizedQueueItem(priority=p, item=inner)
        new_prioqueue.append(item)
    
    self._queue_prio.extend(new_prioqueue)
    
    result = await asyncio.gather(
        *[self._result_store.wait_for_response(item.item) for item in new_prioqueue]
    )
    return result, usage
```

**How to apply to npu-proxy:**
- Create future for each incoming request
- Queue requests with their futures attached
- Resolve futures when batch completes to return results to correct callers

---

## 3. Caching Strategies

### Pattern: Disk-Based Vector Cache

**Problem it solves:** Avoid recomputing embeddings for repeated texts.

**Source:** Infinity (`infinity_emb/inference/batch_handler.py`)

```python
class BatchHandler:
    def __init__(self, ..., vector_disk_cache_path: str = ""):
        cache = (
            Cache(
                cache_name=str(vector_disk_cache_path),
                shutdown=self._shutdown,
            )
            if vector_disk_cache_path
            else None
        )
        self._result_store = ResultKVStoreFuture(cache)
```

**Cache Implementation Pattern:**

```python
class Cache:
    def __init__(self, cache_name: str, shutdown: threading.Event):
        self.cache_name = cache_name
        self._shutdown = shutdown
        # Use diskcache, sqlite, or similar
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding by text hash"""
        pass
        
    def set(self, key: str, value: np.ndarray) -> None:
        """Store embedding with text hash as key"""
        pass
        
    def compute_key(self, text: str) -> str:
        """Hash text for cache lookup"""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()
```

**How to apply to npu-proxy:**
- Add optional `cache_path` parameter to embedding engine
- Hash input texts before lookup (SHA256 or xxhash for speed)
- Store embeddings in memory-mapped file or SQLite for persistence
- Consider LRU eviction for memory-bounded cache

---

### Pattern: Model-Aware Cache Keys

**Problem it solves:** Cache invalidation when model changes.

**Source:** Common pattern across all libraries

```python
def compute_cache_key(
    text: str, 
    model_name: str, 
    normalize: bool,
    pooling_type: str
) -> str:
    """Create cache key that includes model configuration"""
    import hashlib
    key_parts = f"{model_name}:{normalize}:{pooling_type}:{text}"
    return hashlib.sha256(key_parts.encode()).hexdigest()
```

**How to apply to npu-proxy:**
- Include model name, pooling type, and normalization in cache key
- Allow cache prefix configuration per model instance
- Invalidate cache when model config changes

---

## 4. Normalization (L2 Norm) Best Practices

### Pattern: Optional L2 Normalization with Configuration

**Problem it solves:** Some use cases require normalized embeddings, others don't.

**Source:** OpenVINO GenAI TextEmbeddingPipeline

```python
import openvino_genai as ov_genai

pipeline = ov_genai.TextEmbeddingPipeline(
    models_path,
    "CPU",  # or "NPU"
    pooling_type=ov_genai.TextEmbeddingPipeline.PoolingType.MEAN,
    normalize=True,  # Enable L2 normalization
    query_instruction="Represent this sentence for searching: ",
    embed_instruction="Represent this passage for retrieval: "
)

documents_embeddings = pipeline.embed_documents(documents)
query_embeddings = pipeline.embed_query("What is the capital of France?")
```

**Source:** Infinity (`infinity_emb/transformer/embedder/sentence_transformer.py`)

```python
@quant_embedding_decorator()
def encode_post(self, out_features: "Tensor") -> "EmbeddingReturnType":
    with torch.inference_mode():
        embeddings = out_features.to(torch.float32)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.numpy()
    return embeddings_np
```

**NumPy L2 Normalization:**

```python
import numpy as np

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings along axis 1"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms
```

**How to apply to npu-proxy:**
- Add `normalize: bool = True` parameter to embedding API
- Perform normalization in post-processing stage (CPU)
- Use float32 for normalization even if model outputs float16
- Document that normalized embeddings enable cosine similarity via dot product

---

### Pattern: Matryoshka Dimension Truncation

**Problem it solves:** Allow variable-dimension embeddings from a single model.

**Source:** Infinity

```python
def matryososka_slice(
    embeddings: list[np.ndarray], 
    matryoshka_dim: Optional[int]
) -> list[np.ndarray]:
    if matryoshka_dim:
        if 1 > matryoshka_dim or matryoshka_dim > len(embeddings[0]):
            raise MatryoshkaDimError(
                f"matryoshka_dim={matryoshka_dim} is not in valid range. "
                f"Select between 1 and {len(embeddings[0])}."
            )
        return [e[:matryoshka_dim] for e in embeddings]
    return embeddings
```

**How to apply to npu-proxy:**
- Support `matryoshka_dim` parameter for compatible models
- Truncate AFTER normalization to maintain unit norm
- Re-normalize after truncation if needed: `embed = embed[:dim] / np.linalg.norm(embed[:dim])`

---

## 5. Timeout Handling for Batch Operations

### Pattern: Queue Timeout with Graceful Degradation

**Problem it solves:** Prevent requests from hanging indefinitely when system is overloaded.

**Source:** Infinity (`infinity_emb/inference/batch_handler.py`)

```python
QUEUE_TIMEOUT = 0.5  # 500ms per queue operation

class ModelWorker:
    def _preprocess_batch(self):
        while not self._shutdown.is_set():
            try:
                batch = self._input_q.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            
            # Process batch...
            
            # Put with timeout to avoid blocking forever
            while not self._shutdown.is_set():
                try:
                    self._feature_queue.put((feat, batch), timeout=QUEUE_TIMEOUT)
                    break
                except queue.Full:
                    continue
```

**Pattern: Overload Detection and Rejection:**

```python
class BatchHandler:
    def __init__(self, max_queue_wait: int = 32_000):
        self._max_queue_wait = max_queue_wait
    
    def is_overloaded(self) -> bool:
        """Check if more items can be queued."""
        return len(self._queue_prio) > self._max_queue_wait

    def overload_status(self) -> OverloadStatus:
        """Return queue status for monitoring."""
        return OverloadStatus(
            queue_fraction=len(self._queue_prio) / self._max_queue_wait,
            queue_absolute=len(self._queue_prio),
            results_absolute=len(self._result_store),
        )
```

**How to apply to npu-proxy:**
- Set configurable timeout for embedding requests (default 30s)
- Implement queue depth monitoring
- Return 503 Service Unavailable when queue exceeds threshold
- Add `/health` endpoint that includes queue status

---

### Pattern: Async Timeout Wrapper

**Problem it solves:** Enforce request-level timeouts in async context.

```python
import asyncio
from typing import TypeVar, Callable, Any

T = TypeVar('T')

async def with_timeout(
    coro: Callable[..., T],
    timeout_seconds: float,
    *args,
    **kwargs
) -> T:
    """Execute coroutine with timeout"""
    try:
        return await asyncio.wait_for(
            coro(*args, **kwargs),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

# Usage
embeddings = await with_timeout(
    engine.embed,
    timeout_seconds=30.0,
    sentences=["text1", "text2"]
)
```

**How to apply to npu-proxy:**
- Wrap embedding calls in `asyncio.wait_for`
- Make timeout configurable per-request via header or query param
- Log timeout occurrences for monitoring

---

## 6. Fallback Strategies When Model Fails

### Pattern: Multi-Backend Engine Selection

**Problem it solves:** Provide fallback inference engines when primary fails.

**Source:** Infinity supports multiple engines

```python
class InferenceEngine(EnumType):
    torch = "torch"
    ctranslate2 = "ctranslate2"
    optimum = "optimum"  # ONNX/TensorRT
    neuron = "neuron"    # AWS Inferentia
    debugengine = "debugengine"

    @staticmethod
    def default_value():
        return InferenceEngine.torch.value
```

**Fallback Pattern:**

```python
class EmbeddingEngineWithFallback:
    def __init__(
        self,
        primary_engine: AsyncEmbeddingEngine,
        fallback_engine: Optional[AsyncEmbeddingEngine] = None,
        max_retries: int = 3
    ):
        self.primary = primary_engine
        self.fallback = fallback_engine
        self.max_retries = max_retries
        self._primary_failures = 0
        self._use_fallback = False
    
    async def embed(self, sentences: list[str]) -> tuple[list[np.ndarray], int]:
        if self._use_fallback and self.fallback:
            return await self._embed_with_retry(self.fallback, sentences)
        
        try:
            result = await self._embed_with_retry(self.primary, sentences)
            self._primary_failures = 0
            return result
        except Exception as e:
            self._primary_failures += 1
            if self._primary_failures >= self.max_retries and self.fallback:
                logger.warning(f"Switching to fallback engine: {e}")
                self._use_fallback = True
                return await self._embed_with_retry(self.fallback, sentences)
            raise
    
    async def _embed_with_retry(
        self, 
        engine: AsyncEmbeddingEngine, 
        sentences: list[str]
    ) -> tuple[list[np.ndarray], int]:
        for attempt in range(self.max_retries):
            try:
                return await engine.embed(sentences)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
```

**How to apply to npu-proxy:**
- NPU primary -> CPU fallback pattern
- Monitor NPU health and switch automatically
- Use circuit breaker pattern to prevent cascade failures

---

### Pattern: Graceful Degradation with Health Checks

**Problem it solves:** Detect unhealthy models before they cause request failures.

```python
class HealthyEmbeddingEngine:
    def __init__(self, engine: AsyncEmbeddingEngine):
        self.engine = engine
        self._last_health_check = 0
        self._is_healthy = True
        self._health_check_interval = 30  # seconds
    
    async def health_check(self) -> bool:
        """Perform health check with a simple embedding"""
        try:
            test_embedding, _ = await asyncio.wait_for(
                self.engine.embed(["health check"]),
                timeout=5.0
            )
            # Validate embedding shape and values
            if len(test_embedding) != 1:
                return False
            if np.isnan(test_embedding[0]).any():
                return False
            return True
        except Exception:
            return False
    
    async def get_embeddings(self, sentences: list[str]) -> list[np.ndarray]:
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            self._is_healthy = await self.health_check()
            self._last_health_check = current_time
        
        if not self._is_healthy:
            raise RuntimeError("Engine is unhealthy")
        
        return await self.engine.embed(sentences)
```

**How to apply to npu-proxy:**
- Implement `/health` endpoint that runs test inference
- Check for NaN/Inf in embeddings as failure signal
- Periodic background health checks (every 30s)
- Expose health status in metrics

---

### Pattern: Input Validation and Sanitization

**Problem it solves:** Prevent model failures from malformed inputs.

```python
from typing import List
import unicodedata

def sanitize_texts(texts: List[str], max_length: int = 8192) -> List[str]:
    """Sanitize texts before embedding"""
    sanitized = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        # Remove null bytes
        text = text.replace("\x00", "")
        # Truncate to max length
        text = text[:max_length]
        # Handle empty strings
        if not text.strip():
            text = "[EMPTY]"
        sanitized.append(text)
    return sanitized

def validate_batch(texts: List[str], max_batch_size: int = 256) -> None:
    """Validate batch before processing"""
    if not texts:
        raise ValueError("Empty text list")
    if len(texts) > max_batch_size:
        raise ValueError(f"Batch size {len(texts)} exceeds max {max_batch_size}")
```

**How to apply to npu-proxy:**
- Sanitize all input texts before embedding
- Validate batch sizes against configured limits
- Return clear error messages for invalid inputs

---

## Summary: Key Recommendations for npu-proxy

1. **Batch Processing**
   - Implement 3-stage pipeline (pre/core/post) with thread pools
   - Use priority queue with token-length-based grouping
   - Support configurable batch sizes (8-256)

2. **Async Patterns**
   - Use async context manager for lifecycle
   - Attach futures to queued items for result routing
   - Protect startup/shutdown with semaphore

3. **Caching**
   - Optional disk cache with model-aware keys
   - LRU memory cache for hot texts
   - SHA256 hashing for cache keys

4. **Normalization**
   - L2 normalize in float32 after inference
   - Make normalization configurable
   - Support Matryoshka truncation with re-normalization

5. **Timeouts**
   - 500ms queue operation timeouts
   - 30s default request timeout
   - Overload detection and 503 responses

6. **Fallback**
   - NPU -> CPU fallback with circuit breaker
   - Health checks every 30s
   - Input sanitization to prevent model failures
