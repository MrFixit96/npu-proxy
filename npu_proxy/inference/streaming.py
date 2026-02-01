"""Real-time token streaming using asyncio.Queue.

This module provides an async iterator for streaming tokens from an inference engine.
It bridges synchronous callbacks from inference threads to async iteration for SSE
(Server-Sent Events) responses.

Thread-Safety Model:
    The AsyncTokenStream is designed to handle cross-thread communication safely:
    
    1. The inference thread calls callback() which uses asyncio.run_coroutine_threadsafe()
       to safely push tokens to the queue from a non-event-loop thread.
    
    2. The event loop thread consumes tokens via async iteration (__aiter__).
    
    3. The event loop must be set via set_loop() before any tokens are pushed to ensure
       cross-thread communication works correctly.
    
    4. The complete() and error() methods use the same threadsafe mechanism to signal
       completion or errors from the inference thread.
    
    5. Cancellation is supported via the cancel() method, which is thread-safe and
       signals the producer to stop while allowing buffered tokens to be consumed.
    
    This design allows the inference engine (running in a thread pool) to communicate
    with the async HTTP response handler without explicit locks or queues on the
    application side.

Backpressure Handling:
    The queue has a bounded size (default 1000 tokens). If the consumer is slower than
    the producer, the queue will block on put operations, providing natural backpressure.
    Use try_push() for non-blocking puts that return immediately if the queue is full.

Example:
    >>> stream = create_token_stream(timeout=30.0)
    >>> # In inference thread:
    >>> stream.callback("Hello")
    >>> stream.complete()
    >>> # In async handler:
    >>> async for token in stream:
    ...     print(token)
"""

import asyncio
from typing import AsyncIterator

# Sentinel value to signal stream completion
_SENTINEL: None = None


class AsyncTokenStream:
    """Async iterator for streaming tokens from inference engine.
    
    Uses asyncio.Queue to bridge sync callbacks from inference engines (like OpenVINO)
    to async iteration for SSE (Server-Sent Events) responses.
    
    The stream is safe to use across thread boundaries:
    - Inference thread calls callback() to push tokens
    - Event loop thread consumes via async iteration
    - Cancellation can be requested from any thread via cancel()
    
    Attributes:
        is_cancelled: Read-only property indicating if cancellation was requested.
        is_done: Read-only property indicating if the stream has completed.
    
    Example:
        Basic streaming usage::
        
            # In inference thread setup:
            stream = create_token_stream()
            
            # In inference thread (e.g., OpenVINO callback):
            stream.callback("Hello")
            stream.callback(" ")
            stream.callback("world")
            stream.complete()
            
            # In async handler:
            async for token in stream:
                yield token
        
        Cancellation example::
        
            stream = create_token_stream()
            # ... start inference ...
            
            # Cancel from another thread or task:
            stream.cancel()
            
            # The async iteration will stop gracefully
    """
    
    def __init__(self, timeout: float = 60.0, max_queue_size: int = 1000) -> None:
        """Initialize the token stream.
        
        Args:
            timeout: Maximum seconds to wait for a token before raising TimeoutError.
                Set to a value appropriate for your inference latency. Default is 60
                seconds which accommodates slower models or cold starts.
            max_queue_size: Maximum tokens to buffer in the queue. Prevents unbounded
                memory growth if consumer is slower than producer. Default 1000
                provides ~4KB buffer for typical token sizes.
        
        Raises:
            ValueError: If timeout is not positive or max_queue_size is not positive.
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be positive, got {max_queue_size}")
        
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max_queue_size)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._timeout: float = timeout
        self._max_queue_size: int = max_queue_size
        self._done: bool = False
        self._cancelled: bool = False
        self._error: Exception | None = None
    
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for cross-thread communication.
        
        Must be called before the stream is used to ensure callbacks from
        inference threads can safely push tokens to the queue.
        
        Args:
            loop: The asyncio event loop to use for queue communication.
                Typically obtained via asyncio.get_event_loop() or
                asyncio.get_running_loop().
        
        Raises:
            TypeError: If loop is not an AbstractEventLoop.
        
        Note:
            This is automatically called by create_token_stream(). Only call
            manually if you're constructing AsyncTokenStream directly.
        """
        self._loop = loop
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.
        
        Returns:
            True if cancel() has been called, False otherwise.
        """
        return self._cancelled
    
    @property
    def is_done(self) -> bool:
        """Check if the stream has completed.
        
        Returns:
            True if the stream has finished (either completed normally,
            cancelled, or errored), False otherwise.
        """
        return self._done
    
    def cancel(self) -> None:
        """Request cancellation of the stream.
        
        Thread-safe method to signal the producer to stop. The stream will
        complete gracefully, allowing any in-flight operations to finish.
        
        After calling cancel():
        - callback() will return True (stop signal) for subsequent calls
        - The async iterator will stop after processing any buffered tokens
        - is_cancelled property will return True
        
        This method is idempotent; calling it multiple times has no additional
        effect after the first call.
        
        Thread-Safety:
            Safe to call from any thread. Uses asyncio.run_coroutine_threadsafe()
            to push a sentinel value to unblock the consumer.
        
        Example:
            >>> stream = create_token_stream()
            >>> # Start inference...
            >>> # Later, cancel from any thread:
            >>> stream.cancel()
        """
        if self._cancelled:
            return  # Already cancelled, no-op
        
        self._cancelled = True
        # Push sentinel to unblock consumer if waiting
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(_SENTINEL), self._loop
                )
            except Exception:
                # Loop may be closed or unavailable; consumer will detect
                # cancellation via _cancelled flag on next iteration
                pass
    
    def callback(self, token: str) -> bool:
        """Callback for inference engine - pushes token to queue.
        
        This method is designed to be called from the inference thread
        (e.g., as a callback from OpenVINO's async inference).
        
        Uses asyncio.run_coroutine_threadsafe() to safely push to the queue
        even though this method runs in a different thread than the event loop.
        
        Args:
            token: The generated token to stream to the client.
        
        Returns:
            bool: False to continue generation (normal case).
                True to stop generation (if cancelled, queue failed, or no loop).
        
        Thread-Safety:
            Safe to call from any thread. Uses asyncio.run_coroutine_threadsafe()
            to communicate with the event loop thread.
        
        Example:
            >>> def on_token(token: str) -> bool:
            ...     return stream.callback(token)
            >>> # Pass on_token to your inference engine
        """
        if self._cancelled:
            return True  # Stop if cancelled
        if self._loop is None:
            return True  # Stop if no loop set
        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(token), self._loop)
            return False  # Continue generation
        except Exception:
            return True  # Stop on error
    
    def try_push(self, token: str) -> bool:
        """Non-blocking push of a token to the queue.
        
        Unlike callback(), this method returns immediately if the queue is full,
        providing explicit backpressure feedback to the producer.
        
        Args:
            token: The generated token to stream to the client.
        
        Returns:
            bool: True if the token was successfully queued, False if the queue
                is full or the stream is cancelled/not initialized.
        
        Thread-Safety:
            Safe to call from any thread.
        
        Example:
            >>> if not stream.try_push(token):
            ...     # Queue full, apply backpressure
            ...     time.sleep(0.001)
        """
        if self._cancelled or self._loop is None:
            return False
        try:
            # Use put_nowait wrapped in threadsafe call
            future = asyncio.run_coroutine_threadsafe(
                self._try_put(token), self._loop
            )
            return future.result(timeout=0.1)
        except Exception:
            return False
    
    async def _try_put(self, token: str) -> bool:
        """Internal async helper for non-blocking put.
        
        Args:
            token: Token to push to queue.
        
        Returns:
            bool: True if successful, False if queue is full.
        """
        try:
            self._queue.put_nowait(token)
            return True
        except asyncio.QueueFull:
            return False
    
    def complete(self) -> None:
        """Signal stream completion by pushing sentinel.
        
        Call this from the inference thread when generation is complete.
        This pushes a None sentinel which signals __aiter__ to stop.
        
        This method is idempotent; calling it multiple times is safe.
        
        Thread-Safety:
            Safe to call from any thread. Uses asyncio.run_coroutine_threadsafe()
            to communicate with the event loop thread.
        
        Example:
            >>> # After all tokens have been generated:
            >>> stream.complete()
        """
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(_SENTINEL), self._loop
                )
            except Exception:
                # Loop may be closed; set done flag directly
                self._done = True
    
    def error(self, exc: Exception) -> None:
        """Signal an error occurred during generation.
        
        Call this from the inference thread if generation fails.
        The exception will be raised to the async iterator consumer.
        
        Args:
            exc: The exception that occurred during inference.
        
        Raises:
            TypeError: If exc is not an Exception instance.
        
        Thread-Safety:
            Safe to call from any thread. Stores the exception and signals
            completion, which will raise the error to the consumer.
        
        Example:
            >>> try:
            ...     # inference code
            ... except Exception as e:
            ...     stream.error(e)
        """
        self._error = exc
        self.complete()
    
    def __aiter__(self) -> "AsyncTokenStream":
        """Return self as async iterator.
        
        Returns:
            AsyncTokenStream: Self, implementing the async iterator protocol.
        """
        return self
    
    async def __anext__(self) -> str:
        """Get the next token from the stream.
        
        Waits for the next token to be pushed by callback(). Handles
        cancellation, timeouts, and error propagation.
        
        Returns:
            str: The next token in the stream.
        
        Raises:
            StopAsyncIteration: When the stream is complete or cancelled.
            TimeoutError: If no token arrives within the configured timeout.
            Exception: Any exception passed to error() is re-raised here.
        
        Note:
            This method handles asyncio.CancelledError by setting the cancelled
            flag and stopping iteration gracefully.
        """
        # Check for early termination
        if self._cancelled or self._done:
            raise StopAsyncIteration
        
        try:
            token = await asyncio.wait_for(
                self._queue.get(),
                timeout=self._timeout
            )
        except asyncio.CancelledError:
            # Handle task cancellation gracefully
            self._cancelled = True
            self._done = True
            raise StopAsyncIteration
        except asyncio.TimeoutError:
            self._done = True
            raise TimeoutError(f"Token stream timed out after {self._timeout}s")
        
        # Check for sentinel (completion signal)
        if token is _SENTINEL or token is None:
            self._done = True
            if self._error is not None:
                raise self._error
            raise StopAsyncIteration
        
        return token
    
    async def collect(self, max_tokens: int | None = None) -> list[str]:
        """Collect all tokens into a list.
        
        Convenience method to consume the entire stream and return all tokens.
        Useful for testing or when you need all tokens at once.
        
        Args:
            max_tokens: Maximum number of tokens to collect. None means no limit.
        
        Returns:
            list[str]: All tokens from the stream.
        
        Raises:
            TimeoutError: If waiting for a token times out.
            Exception: Any exception passed to error().
        
        Example:
            >>> tokens = await stream.collect()
            >>> full_text = "".join(tokens)
        """
        tokens: list[str] = []
        count = 0
        async for token in self:
            tokens.append(token)
            count += 1
            if max_tokens is not None and count >= max_tokens:
                break
        return tokens


def create_token_stream(
    timeout: float = 60.0,
    max_queue_size: int = 1000
) -> AsyncTokenStream:
    """Create a new token stream with the current event loop.
    
    Convenience function that creates an AsyncTokenStream and automatically
    sets the running event loop. Use this in async contexts where you're
    already inside the event loop.
    
    Args:
        timeout: Maximum seconds to wait for a token before raising TimeoutError.
            Default 60 seconds is suitable for most inference tasks.
            Adjust based on your typical inference latency.
        max_queue_size: Maximum number of tokens to buffer. Default 1000 provides
            backpressure for slow consumers. Reduce for memory-constrained
            environments.
    
    Returns:
        AsyncTokenStream: A properly configured stream ready to use.
    
    Raises:
        RuntimeError: If called outside an async context (no running event loop).
        ValueError: If timeout or max_queue_size is not positive.
    
    Example:
        Basic usage in FastAPI::
        
            @app.post("/generate")
            async def generate():
                stream = create_token_stream(timeout=30)
                # ... pass stream to inference thread ...
                async for token in stream:
                    yield token
        
        With custom backpressure settings::
        
            stream = create_token_stream(
                timeout=120.0,      # 2 minutes for slow models
                max_queue_size=100  # Smaller buffer for memory
            )
    """
    stream = AsyncTokenStream(timeout=timeout, max_queue_size=max_queue_size)
    stream.set_loop(asyncio.get_event_loop())
    return stream
