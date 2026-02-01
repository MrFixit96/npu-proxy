# Token Streaming Best Practices Research

Research on real-time token streaming patterns for async Python, compiled for npu-proxy `AsyncTokenStream`.

## Table of Contents
1. [Producer-Consumer Patterns](#1-producer-consumer-patterns)
2. [Backpressure Handling](#2-backpressure-handling)
3. [Graceful Cancellation](#3-graceful-cancellation)
4. [Error Propagation](#4-error-propagation)
5. [Memory Efficiency](#5-memory-efficiency)
6. [SSE Best Practices](#6-sse-best-practices)

---

## 1. Producer-Consumer Patterns

### Pattern: Thread-Safe AsyncIO Queue Bridge

**Problem it solves:**
Bridges synchronous inference callbacks (running in thread pools) to async HTTP response handlers. Required when inference engines like OpenVINO run in separate threads but HTTP responses need async iteration.

**Sources:** vLLM `RequestOutputCollector`, OpenVINO GenAI `StreamerBase`, npu-proxy `AsyncTokenStream`

**Key Design Elements:**
1. **Event-based signaling** instead of Queue polling (vLLM approach)
2. **asyncio.run_coroutine_threadsafe()** for cross-thread communication
3. **Sentinel values** (None) to signal completion
4. **Output aggregation** when producer outpaces consumer

**Code Example (vLLM pattern):**
```python
class RequestOutputCollector:
    """Collects streamed outputs for hand-off to consuming asyncio task.
    
    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """
    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: RequestOutput | Exception | None = None
        self.ready = asyncio.Event()  # Event-based, not Queue-based
        
    def put(self, output: RequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput):
            # Merge outputs when producer is faster than consumer
            self.output.add(output, aggregate=self.aggregate)
    
    async def get(self) -> RequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output
```

**Code Example (npu-proxy current pattern):**
```python
class AsyncTokenStream:
    def __init__(self, timeout: float = 60.0, max_queue_size: int = 1000):
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max_queue_size)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._timeout = timeout
    
    def callback(self, token: str) -> bool:
        """Thread-safe callback using run_coroutine_threadsafe."""
        if self._loop is None:
            return True  # Stop if no loop set
        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(token), self._loop)
            return False  # Continue generation
        except Exception:
            return True  # Stop on error
```

**Applicability to npu-proxy:**
- Current `AsyncTokenStream` uses Queue correctly but could benefit from vLLM's Event-based approach for lower latency
- Consider adding output aggregation for delta streaming when consumer is slow
- The `run_coroutine_threadsafe` pattern is correct for OpenVINO callback integration

---

## 2. Backpressure Handling

### Pattern: Bounded Queue with Producer Feedback

**Problem it solves:**
Prevents unbounded memory growth when the producer (inference engine) generates tokens faster than the consumer (HTTP client) can receive them.

**Sources:** vLLM, asyncio best practices, aiohttp streaming

**Key Design Elements:**
1. **Bounded queue** (`maxsize` parameter) to cap memory usage
2. **Producer return value** to signal when to slow down or stop
3. **Non-blocking put** with aggregation as alternative to blocking
4. **Timeout on consumer side** to detect stalls

**Code Example (Bounded Queue with Feedback):**
```python
class BackpressureAwareStream:
    def __init__(self, max_buffer: int = 1000):
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max_buffer)
        self._loop: asyncio.AbstractEventLoop | None = None
        
    def callback(self, token: str) -> bool:
        """Returns True to signal backpressure (stop generation)."""
        if self._loop is None:
            return True
            
        future = asyncio.run_coroutine_threadsafe(
            self._queue.put(token), 
            self._loop
        )
        
        try:
            # Wait with timeout - if queue is full, this blocks
            future.result(timeout=0.1)
            return False  # Continue
        except TimeoutError:
            # Queue is full - signal backpressure
            return True  # Stop generation temporarily
```

**Code Example (Aggregation-based Backpressure - vLLM style):**
```python
def put(self, output: RequestOutput) -> None:
    """Non-blocking put with aggregation instead of blocking."""
    if self.output is None:
        self.output = output
        self.ready.set()
    else:
        # Aggregate instead of blocking - handles backpressure gracefully
        self.output.add(output, aggregate=True)
        # Consumer will receive merged output next time
```

**Applicability to npu-proxy:**
- Current `max_queue_size=1000` is reasonable but consider making it configurable
- The callback return value pattern is already implemented correctly
- Consider adding metrics for queue high-water marks to monitor backpressure

---

## 3. Graceful Cancellation

### Pattern: Multi-Level Cancellation with Cleanup

**Problem it solves:**
Allows clients to cancel streaming requests mid-generation without leaving resources hanging. Must handle cancellation at HTTP layer, async layer, and inference thread.

**Sources:** vLLM `RequestOutputCollector.close()`, OpenVINO `StreamingStatus.CANCEL`, FastAPI

**Key Design Elements:**
1. **StreamingStatus enum** with RUNNING, STOP, CANCEL states
2. **Task tracking** for cleanup on cancellation
3. **Thread-safe cancellation** from any thread
4. **Cleanup in __del__** as safety net

**Code Example (OpenVINO StreamingStatus):**
```python
class StreamingStatus(Enum):
    RUNNING = 0   # Continue generation
    STOP = 1      # Stop normally (e.g., max tokens reached)
    CANCEL = 2    # Abort immediately (client disconnected)
```

**Code Example (vLLM Cancellation):**
```python
class RequestOutputCollector:
    def __init__(self, ...):
        self._input_stream_task: asyncio.Task | None = None
    
    def close(self):
        """Explicit cleanup."""
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None
    
    def __del__(self):
        """Safety net cleanup using call_soon_threadsafe."""
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
        self._input_stream_task = None
```

**Code Example (Enhanced AsyncTokenStream with cancellation):**
```python
class AsyncTokenStream:
    def __init__(self, ...):
        self._cancelled = False
        
    def cancel(self) -> None:
        """Signal cancellation - safe to call from any thread."""
        self._cancelled = True
        self.complete()  # Push sentinel to unblock consumer
    
    def callback(self, token: str) -> bool:
        """Check cancellation before pushing."""
        if self._cancelled:
            return True  # Stop generation immediately
        # ... rest of callback
    
    async def __aiter__(self) -> AsyncIterator[str]:
        while not self._done:
            try:
                token = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=self._timeout
                )
                if token is None or self._cancelled:
                    break
                yield token
            except asyncio.CancelledError:
                self._cancelled = True
                raise  # Re-raise for proper async cleanup
```

**Applicability to npu-proxy:**
- Add explicit `cancel()` method to `AsyncTokenStream`
- Handle `CancelledError` in `__aiter__` to properly propagate cancellation
- Consider returning `StreamingStatus` equivalent from callback for richer control

---

## 4. Error Propagation

### Pattern: Exception Tunneling Through Queues

**Problem it solves:**
Errors occurring in the inference thread must propagate to the async consumer so they can be handled appropriately (e.g., returning 500 to HTTP client).

**Sources:** vLLM `RequestOutputCollector`, npu-proxy `AsyncTokenStream.error()`

**Key Design Elements:**
1. **Store exception in collector** instead of separate error channel
2. **Raise on get()** not on put()
3. **Type union** for output: `RequestOutput | Exception | None`
4. **Preserve exception chain** for debugging

**Code Example (vLLM pattern):**
```python
class RequestOutputCollector:
    def put(self, output: RequestOutput | Exception) -> None:
        """Accept both outputs and exceptions."""
        if self.output is None or isinstance(output, Exception):
            self.output = output  # Exceptions override normal output
            self.ready.set()
    
    async def get(self) -> RequestOutput:
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output  # Propagate to consumer
        return output
```

**Code Example (Enhanced error propagation):**
```python
class AsyncTokenStream:
    def error(self, exc: Exception) -> None:
        """Signal an error occurred during generation."""
        self._error = exc
        self.complete()  # Unblock consumer
    
    async def __aiter__(self) -> AsyncIterator[str]:
        while not self._done:
            try:
                token = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=self._timeout
                )
                if token is None:
                    self._done = True
                    if self._error:
                        # Preserve exception chain
                        raise self._error from None
                    break
                yield token
            except asyncio.TimeoutError:
                self._done = True
                raise TimeoutError(f"Token stream timed out after {self._timeout}s")
```

**Applicability to npu-proxy:**
- Current implementation is correct but could add exception chaining
- Consider logging errors before propagation for debugging
- Ensure HTTP layer maps exceptions to appropriate status codes

---

## 5. Memory Efficiency

### Pattern: Incremental Detokenization with Buffer Limits

**Problem it solves:**
Streaming tokens individually is inefficient; batching is better for throughput but increases latency. Need balance between memory usage and responsiveness.

**Sources:** vLLM `IncrementalDetokenizer`, aiohttp chunked streaming

**Key Design Elements:**
1. **Stream interval** to batch tokens before sending
2. **Bounded buffer** with size limits
3. **Incremental processing** to avoid accumulating full response
4. **Chunk-based iteration** for large data

**Code Example (vLLM stream interval):**
```python
@dataclass
class RequestState:
    stream_interval: int  # How many tokens to accumulate before yielding
    
    # Only yield every N tokens for efficiency
    def should_stream(self, token_count: int) -> bool:
        return token_count % self.stream_interval == 0
```

**Code Example (Chunked streaming for efficiency):**
```python
async def iter_chunked(self, chunk_size: int = 1024):
    """Iterate over stream in chunks for efficiency."""
    buffer = []
    buffer_size = 0
    
    async for token in self._stream:
        buffer.append(token)
        buffer_size += len(token)
        
        if buffer_size >= chunk_size:
            yield ''.join(buffer)
            buffer = []
            buffer_size = 0
    
    # Yield remaining
    if buffer:
        yield ''.join(buffer)
```

**Code Example (Memory-bounded token buffer):**
```python
class TokenBuffer:
    def __init__(self, max_tokens: int = 1000, max_bytes: int = 1_000_000):
        self._tokens: deque[str] = deque(maxlen=max_tokens)
        self._byte_count = 0
        self._max_bytes = max_bytes
    
    def add(self, token: str) -> bool:
        """Returns False if buffer is at capacity."""
        token_bytes = len(token.encode('utf-8'))
        if self._byte_count + token_bytes > self._max_bytes:
            return False
        self._tokens.append(token)
        self._byte_count += token_bytes
        return True
```

**Applicability to npu-proxy:**
- Consider adding configurable `stream_interval` for batch efficiency
- Current `max_queue_size=1000` is reasonable; could add byte-based limit
- For very long generations, consider periodic buffer flush to prevent accumulation

---

## 6. SSE Best Practices

### Pattern: Proper SSE Formatting with Keep-Alive

**Problem it solves:**
Server-Sent Events (SSE) require specific formatting and connection management for reliable real-time streaming to web clients.

**Sources:** FastAPI `StreamingResponse`, MDN SSE documentation, FastAPI AI SDK

**Key Design Elements:**
1. **Correct media type**: `text/event-stream`
2. **Event format**: `data: {content}\n\n`
3. **Keep-alive comments** to prevent connection timeout
4. **Proper error events** for client notification
5. **ID and retry fields** for reconnection

**Code Example (FastAPI SSE endpoint):**
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def generate_sse_events(stream: AsyncTokenStream):
    """Convert token stream to SSE format."""
    try:
        async for token in stream:
            # Standard SSE data event
            yield f"data: {json.dumps({'token': token})}\n\n"
    except asyncio.CancelledError:
        yield f"data: {json.dumps({'event': 'cancelled'})}\n\n"
        raise
    except Exception as e:
        # Error event for client
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        raise
    finally:
        # End event
        yield f"data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def stream_completion(request: ChatRequest):
    stream = create_token_stream()
    # ... start inference ...
    return StreamingResponse(
        generate_sse_events(stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
```

**Code Example (OpenAI-compatible streaming format):**
```python
async def generate_openai_stream(stream: AsyncTokenStream, request_id: str):
    """Generate OpenAI-compatible SSE stream."""
    async for token in stream:
        chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "npu-model",
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Final chunk with finish_reason
    final_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

**Code Example (Keep-alive with timeout):**
```python
async def generate_with_keepalive(
    stream: AsyncTokenStream, 
    keepalive_interval: float = 15.0
):
    """SSE stream with keep-alive comments to prevent timeout."""
    last_event = time.monotonic()
    
    while True:
        try:
            token = await asyncio.wait_for(
                stream._queue.get(),
                timeout=keepalive_interval
            )
            if token is None:
                break
            yield f"data: {json.dumps({'token': token})}\n\n"
            last_event = time.monotonic()
        except asyncio.TimeoutError:
            # Send keep-alive comment (SSE spec)
            yield f": keep-alive {time.monotonic()}\n\n"
```

**SSE Headers for Production:**
```python
SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",      # nginx
    "X-Content-Type-Options": "nosniff",
    "Transfer-Encoding": "chunked",
}
```

**Applicability to npu-proxy:**
- Ensure SSE format is correct with proper `\n\n` terminators
- Add keep-alive mechanism for long inference times
- Include proper error events for client-side handling
- Consider adding event IDs for client reconnection support

---

## Summary: Recommendations for npu-proxy AsyncTokenStream

### Current Strengths
1. ✅ Correct use of `asyncio.Queue` with bounded size
2. ✅ Thread-safe `run_coroutine_threadsafe` for callbacks
3. ✅ Sentinel-based completion signaling
4. ✅ Error propagation via `error()` method
5. ✅ Configurable timeout

### Recommended Enhancements

| Priority | Enhancement | Rationale |
|----------|-------------|-----------|
| High | Add `cancel()` method | Enable graceful client disconnection handling |
| High | Handle `CancelledError` in `__aiter__` | Proper async cleanup on cancellation |
| Medium | Add keep-alive mechanism for SSE | Prevent proxy/client timeout on slow inference |
| Medium | Consider Event-based approach | Lower latency than Queue for single-consumer |
| Medium | Add metrics/logging for backpressure | Monitor queue depth, detect slow consumers |
| Low | Token aggregation option | Efficiency when consumer is slower than producer |
| Low | Stream interval batching | Reduce SSE overhead for high token rates |

### Architecture Pattern Summary

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Inference Thread│     │  AsyncTokenStream │     │ HTTP Handler    │
│ (OpenVINO NPU)  │     │  (Queue Bridge)   │     │ (FastAPI SSE)   │
├─────────────────┤     ├──────────────────┤     ├─────────────────┤
│                 │     │                  │     │                 │
│ callback(token) ├────►│ Queue.put(token) │     │ async for token │
│                 │     │                  ├────►│   yield SSE     │
│ return status   │◄────┤ backpressure     │     │                 │
│                 │     │                  │     │                 │
│ complete()      ├────►│ Queue.put(None)  │     │ break iteration │
│                 │     │                  │     │                 │
│ error(exc)      ├────►│ Store + complete │────►│ raise exception │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## References

- **vLLM**: `vllm/v1/engine/output_processor.py` - RequestOutputCollector pattern
- **OpenVINO GenAI**: `openvino_genai.StreamerBase` - StreamingStatus, write/end interface
- **FastAPI**: StreamingResponse, async generators for SSE
- **FastAPI AI SDK**: @streaming_endpoint decorator, event generation patterns
- **aiohttp**: Chunked streaming, async iterators for request/response bodies
