# Real-Time Streaming Architecture

## Architecture Overview

The NPU Proxy implements a real-time streaming architecture that enables server-sent events (SSE) for streaming inference results back to clients. The architecture bridges synchronous inference callbacks with asynchronous HTTP response streams, allowing the inference engine to push tokens to clients as they are generated without waiting for the entire response to be generated.

### Key Design Goals

- **Low Latency**: Tokens are delivered to clients immediately upon generation
- **Memory Efficient**: Queue-based streaming prevents buffering entire responses in memory
- **Non-blocking**: Async/await patterns ensure HTTP handlers remain responsive
- **Timeout Protection**: Built-in safeguards against hanging connections
- **Error Resilient**: Graceful handling of inference failures and client disconnections

## AsyncTokenStream Implementation

The `AsyncTokenStream` class in `npu_proxy/inference/streaming.py` is the core component that bridges synchronous callback-based token delivery with asynchronous streaming patterns required by HTTP clients.

### Queue-Based Design

AsyncTokenStream uses `asyncio.Queue` to decouple the producer (inference thread) from the consumer (HTTP response stream):

```python
class AsyncTokenStream:
    """
    Bridges synchronous token callbacks to async iteration.
    
    Maintains an asyncio.Queue that receives tokens from the inference
    thread's callback mechanism and exposes them through async iteration.
    """
    
    def __init__(self, timeout=180):
        self.queue = asyncio.Queue()
        self.timeout = timeout
        self.done = False
        self.error = None
```

### Callback Mechanism

The inference engine invokes a callback function for each generated token. AsyncTokenStream registers this callback and uses it to enqueue tokens:

```python
def token_callback(token: str):
    """
    Called from inference thread on each token generation.
    
    This runs in the inference thread context, so operations must be
    thread-safe. asyncio.Queue.put_nowait() is thread-safe and does
    not block.
    """
    try:
        self.queue.put_nowait(token)
    except asyncio.QueueFull:
        # Queue is full - inference is ahead of consumption
        logger.warning("Token queue full, dropping token")
```

### Async Iterator Protocol

AsyncTokenStream implements the async iterator protocol, allowing consumption through `async for`:

```python
async def __aiter__(self):
    """Return the async iterator object (self)."""
    return self

async def __anext__(self):
    """
    Retrieve the next token from the queue.
    
    Awaits with timeout protection. When the queue yields None (sentinel),
    iteration terminates. Raises StopAsyncIteration to end the loop.
    """
    try:
        # Wait for next token or timeout
        token = await asyncio.wait_for(
            self.queue.get(),
            timeout=self.timeout
        )
        
        # Sentinel value signals completion
        if token is None:
            raise StopAsyncIteration
            
        return token
        
    except asyncio.TimeoutError:
        # Inference took too long - terminate stream
        raise StopAsyncIteration
    
    except Exception as e:
        # Capture error state for client notification
        self.error = str(e)
        raise StopAsyncIteration
```

### Completion Signaling

The inference callback sends a sentinel value (None) to signal completion:

```python
def on_completion_callback():
    """Called when inference finishes successfully."""
    self.queue.put_nowait(None)  # Sentinel to end iteration

def on_error_callback(error: str):
    """Called when inference encounters an error."""
    self.error = error
    self.queue.put_nowait(None)  # Still send sentinel to unblock
```

## Data Flow Diagram

The following diagram shows how data flows from the inference engine to the client's HTTP response:

```
Inference Thread                    Main Event Loop
================                    ===============

Token Generated
    |
    v
token_callback(token)
    |
    +---> queue.put_nowait(token)    (Thread-safe, non-blocking)
                                             |
                                             v
                                   HTTP Handler (async)
                                       |
                                       v
                                   AsyncTokenStream.__anext__()
                                       |
                                       v
                                   queue.get() with timeout
                                       |
                                       v (if token)
                                   Format for SSE
                                       |
                                       v
                                   HTTP Response Writer
                                       |
                                       v
                                   Client Browser


Completion Signal
    |
    v
queue.put_nowait(None)              (Sentinel value)
                                       |
                                       v
                                   __anext__() returns None
                                       |
                                       v
                                   StopAsyncIteration
                                       |
                                       v
                                   Connection Closes
```

## Client Integration Examples

### Chat Completions Endpoint

The `/v1/chat/completions` endpoint supports streaming via the `stream` parameter:

```python
# Client request
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ],
  "stream": true
}
```

The response streams as Server-Sent Events (SSE):

```
data: {"choices":[{"delta":{"content":"4"}}]}

data: {"choices":[{"delta":{"content":""}}]}

data: [DONE]
```

### Server-Sent Events Consumer (JavaScript)

```javascript
const eventSource = new EventSource(
  '/v1/chat/completions',
  {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      model: 'gpt-3.5-turbo',
      messages: [{role: 'user', content: 'Hello'}],
      stream: true
    })
  }
);

let fullResponse = '';

eventSource.addEventListener('message', (event) => {
  if (event.data === '[DONE]') {
    console.log('Stream complete');
    return;
  }
  
  try {
    const chunk = JSON.parse(event.data);
    const token = chunk.choices[0]?.delta?.content || '';
    fullResponse += token;
    console.log('Token:', token);
  } catch (e) {
    console.error('Parse error:', e);
  }
});

eventSource.addEventListener('error', (error) => {
  console.error('Stream error:', error);
  eventSource.close();
});
```

### Python Streaming Consumer

```python
import httpx
import json

async def stream_chat_completion():
    """Stream chat completions from NPU Proxy."""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "model-name",
            "messages": [{"role": "user", "content": "Explain streaming"}],
            "stream": True
        }
        
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            timeout=180.0
        ) as response:
            full_text = ""
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                    
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        token = chunk['choices'][0]['delta']['content']
                        full_text += token
                        print(token, end='', flush=True)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            
            return full_text
```

### API Generate Endpoint

The `/api/generate` endpoint also supports streaming:

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "prompt": "The future of AI is",
    "stream": true
  }'
```

Response (streaming):

```
data: {"response": "bright"}
data: {"response": " and"}
data: {"response": " full"}
...
data: [DONE]
```

### API Chat Endpoint

The `/api/chat` endpoint supports streaming conversations:

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "messages": [
      {"role": "user", "content": "What is streaming?"}
    ],
    "stream": true
  }'
```

## Timeout and Error Handling

### Default Timeout Configuration

The AsyncTokenStream class enforces a 180-second (3-minute) default timeout for token delivery:

```python
# In streaming.py initialization
stream = AsyncTokenStream(timeout=180)
```

This timeout protects against:
- Inference processes that hang or deadlock
- Network connections stuck waiting for data
- Resource exhaustion from long-running requests

### Timeout Behavior

When the 180-second timeout is exceeded:

1. The `__anext__()` method raises `asyncio.TimeoutError`
2. Iteration terminates with `StopAsyncIteration`
3. HTTP connection closes with HTTP 200 (stream already started)
4. Client receives incomplete response
5. Error is logged for debugging

### Inference Error Handling

If the inference engine encounters an error:

```python
def handle_inference_error(error_message: str):
    """
    Called when inference fails (out of memory, model error, etc).
    """
    stream.error = error_message
    # Send sentinel to unblock async iterator
    stream.queue.put_nowait(None)
```

The client receives:
- Already-streamed tokens (up to failure point)
- Connection closes without explicit error (streaming HTTP limitation)
- Optional: Check response headers for error indicators

### Client Timeout Handling

Clients must also implement timeouts:

```python
# Python httpx - request timeout
async with client.stream(
    "POST",
    url,
    json=payload,
    timeout=180.0  # Must match or exceed server timeout
) as response:
    async for line in response.aiter_lines():
        # Process line
```

### Error Recovery Patterns

For non-streaming requests, implement retry logic:

```python
import asyncio

async def chat_with_retry(payload, max_retries=3):
    """Execute chat with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            async with client.stream(...) as response:
                # Process stream
                return result
        except asyncio.TimeoutError:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
                continue
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
```

## Troubleshooting Common Issues

### Issue: Connection Hangs for 180+ Seconds

**Symptom**: Client connection doesn't close, server appears stuck

**Root Causes**:
- Inference process deadlocked or infinite loop
- Model consuming excessive memory (OOM killer pending)
- Hardware thermal throttling causing extreme slowdown

**Diagnosis**:
```bash
# Check server logs for timeout messages
tail -f server.log | grep -i timeout

# Monitor inference process
ps aux | grep inference

# Check system resources
free -h
nvidia-smi  # GPU memory usage
```

**Resolution**:
1. Restart the inference service
2. Check model configuration and input validation
3. Verify hardware has sufficient VRAM
4. Reduce concurrent requests if resource-constrained

### Issue: Intermittent Empty Tokens in Response

**Symptom**: Response contains empty strings or malformed JSON

**Root Causes**:
- Token callback race conditions
- Queue overflow causing dropped tokens
- SSE formatting errors

**Diagnosis**:
```python
# Enable debug logging in streaming.py
import logging
logging.getLogger('npu_proxy.inference.streaming').setLevel(logging.DEBUG)
```

**Resolution**:
- Ensure synchronization around queue operations
- Increase queue max size if queue full warnings appear
- Validate token format before enqueuing

### Issue: Slow Token Delivery (Buffering)

**Symptom**: Tokens arrive in batches rather than individually

**Root Causes**:
- HTTP client buffering (use streaming/chunked transfer encoding)
- Callback running too fast, queuing tokens faster than HTTP sends
- Network buffer filling up

**Diagnosis**:
```python
# Monitor queue depth
logger.info(f"Queue size: {stream.queue.qsize()}")
```

**Resolution**:
- Ensure HTTP response uses chunked encoding (not buffered)
- Add small artificial delay in callback for smoother delivery
- Check client-side streaming configuration

### Issue: Memory Leak in Long-Running Services

**Symptom**: Memory usage grows over time, reaches OOM

**Root Causes**:
- AsyncTokenStream instances not being garbage collected
- Queues holding references to unreleased tokens
- Exception handlers not cleaning up resources

**Resolution**:
```python
# Ensure proper cleanup in HTTP handler
async def stream_response(request):
    stream = AsyncTokenStream(timeout=180)
    try:
        # Use stream
        async for token in stream:
            yield format_sse(token)
    finally:
        # Explicit cleanup
        while not stream.queue.empty():
            try:
                stream.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
```

### Issue: Clients Don't Receive First Token Immediately

**Symptom**: Initial delay before any response appears

**Root Causes**:
- Inference startup overhead (model loading)
- HTTP buffering waiting for minimum chunk size
- Client-side buffering

**Diagnosis**:
- Add logging at first token generation
- Verify SSE format includes flush headers

**Resolution**:
```python
# Ensure no buffering in HTTP response
response.headers['Cache-Control'] = 'no-cache'
response.headers['X-Accel-Buffering'] = 'no'
```

### Issue: Protocol Errors in SSE Stream

**Symptom**: Client reports parse errors, connection drops

**Root Causes**:
- Malformed SSE format (missing newlines)
- Non-UTF8 characters in token
- Incomplete JSON chunks

**Diagnosis**:
```bash
# Capture raw response
curl -v http://localhost:8000/v1/chat/completions \
  -d '{"model":"test","messages":[...],"stream":true}' \
  2>&1 | grep -A 50 'data:'
```

**Resolution**:
- Validate token format before SSE encoding
- Handle UTF-8 encoding/decoding explicitly
- Ensure SSE format compliance: `data: {json}\n\n`

### Issue: Concurrent Streams Interfering

**Symptom**: Tokens from different requests mixed in response

**Root Causes**:
- Shared state between stream instances
- Global queue instead of per-request queue
- Callback registration collision

**Resolution**:
- Ensure each request creates its own AsyncTokenStream
- Use request-scoped dependency injection
- Validate callback isolation in tests

## Performance Tuning

### Queue Size Configuration

```python
# Custom queue size (default: unlimited)
stream = AsyncTokenStream(timeout=180)
stream.queue = asyncio.Queue(maxsize=1000)  # Limit memory usage
```

### Token Batching

For high-throughput scenarios, batch tokens before sending:

```python
async def batched_tokens(stream, batch_size=5):
    """Yield batches of tokens to reduce HTTP overhead."""
    batch = []
    async for token in stream:
        batch.append(token)
        if len(batch) >= batch_size:
            yield ''.join(batch)
            batch = []
    
    if batch:
        yield ''.join(batch)
```

### Monitoring and Metrics

Recommended metrics to track:
- Token generation latency (ms per token)
- Queue depth over time (max, average)
- Stream duration (total time to completion)
- Timeout occurrences (count, frequency)
- Client error rates (partial responses)

## References

- AsyncIO Documentation: https://docs.python.org/3/library/asyncio.html
- Server-Sent Events Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html
- NPU Proxy Streaming Module: `npu_proxy/inference/streaming.py`
- HTTP Chunked Transfer Encoding: https://tools.ietf.org/html/rfc7230#section-4.1
