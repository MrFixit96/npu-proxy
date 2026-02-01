"""Unit tests for the token streaming module."""

import asyncio
import pytest
from npu_proxy.inference.streaming import AsyncTokenStream, create_token_stream


class TestAsyncTokenStream:
    """Test cases for AsyncTokenStream class."""
    
    def test_async_token_stream_creation(self):
        """Test creation of AsyncTokenStream with default and custom parameters."""
        # Default parameters
        stream = AsyncTokenStream()
        assert stream._timeout == 60.0
        assert stream._queue.maxsize == 1000
        assert stream._done is False
        assert stream._error is None
        assert stream._loop is None
        
        # Custom parameters
        stream = AsyncTokenStream(timeout=30.0, max_queue_size=500)
        assert stream._timeout == 30.0
        assert stream._queue.maxsize == 500
        assert stream._done is False
        assert stream._error is None
    
    def test_stream_callback_pushes_token(self):
        """Test that callback pushes tokens to the queue from another thread."""
        async def run_test():
            stream = AsyncTokenStream()
            loop = asyncio.get_event_loop()
            stream.set_loop(loop)
            
            # Simulate token from inference thread
            result = stream.callback("Hello")
            assert result is False  # Should continue generation
            
            # Token should be in queue
            token = await asyncio.wait_for(stream._queue.get(), timeout=1.0)
            assert token == "Hello"
        
        asyncio.run(run_test())
    
    def test_stream_callback_without_loop_stops(self):
        """Test that callback returns True (stop) when no loop is set."""
        stream = AsyncTokenStream()
        # Don't set loop
        result = stream.callback("Hello")
        assert result is True  # Should stop generation
    
    def test_stream_complete_sends_sentinel(self):
        """Test that complete() sends None sentinel to signal end of stream."""
        async def run_test():
            stream = AsyncTokenStream()
            loop = asyncio.get_event_loop()
            stream.set_loop(loop)
            
            # Push a token and then complete
            stream.callback("token1")
            stream.complete()
            
            # Consume tokens
            tokens = []
            async for token in stream:
                tokens.append(token)
            
            assert tokens == ["token1"]
            assert stream._done is True
        
        asyncio.run(run_test())
    
    def test_stream_iteration(self):
        """Test complete iteration over streamed tokens."""
        async def run_test():
            stream = AsyncTokenStream()
            loop = asyncio.get_event_loop()
            stream.set_loop(loop)
            
            # Simulate tokens arriving
            test_tokens = ["Hello", " ", "world", "!"]
            for token in test_tokens:
                stream.callback(token)
            stream.complete()
            
            # Consume all tokens
            collected = []
            async for token in stream:
                collected.append(token)
            
            assert collected == test_tokens
        
        asyncio.run(run_test())
    
    def test_stream_timeout_raises(self):
        """Test that timeout raises TimeoutError when no token arrives."""
        async def run_test():
            stream = AsyncTokenStream(timeout=0.1)
            loop = asyncio.get_event_loop()
            stream.set_loop(loop)
            
            # Try to iterate without sending anything
            with pytest.raises(TimeoutError) as exc_info:
                async for token in stream:
                    pass
            
            assert "timed out" in str(exc_info.value).lower()
            assert stream._done is True
        
        asyncio.run(run_test())
    
    def test_stream_error_propagates(self):
        """Test that errors are properly propagated to the async iterator."""
        async def run_test():
            stream = AsyncTokenStream()
            loop = asyncio.get_event_loop()
            stream.set_loop(loop)
            
            # Push a token, then signal error
            stream.callback("partial")
            test_error = ValueError("Inference failed!")
            stream.error(test_error)
            
            # Consume tokens
            tokens = []
            with pytest.raises(ValueError) as exc_info:
                async for token in stream:
                    tokens.append(token)
            
            assert tokens == ["partial"]
            assert exc_info.value is test_error
        
        asyncio.run(run_test())
    
    def test_stream_set_loop(self):
        """Test setting the event loop."""
        stream = AsyncTokenStream()
        assert stream._loop is None
        
        loop = asyncio.new_event_loop()
        stream.set_loop(loop)
        assert stream._loop is loop
        loop.close()
    
    def test_multiple_tokens_before_complete(self):
        """Test queuing multiple tokens before completion."""
        async def run_test():
            stream = AsyncTokenStream()
            loop = asyncio.get_event_loop()
            stream.set_loop(loop)
            
            # Queue multiple tokens quickly
            for i in range(10):
                stream.callback(f"token{i}")
            stream.complete()
            
            # Verify all tokens are returned
            tokens = []
            async for token in stream:
                tokens.append(token)
            
            assert len(tokens) == 10
            assert tokens[0] == "token0"
            assert tokens[9] == "token9"
        
        asyncio.run(run_test())
    
    def test_callback_with_exception_stops(self):
        """Test that callback stops generation on exception."""
        stream = AsyncTokenStream()
        
        # Mock a broken loop
        class BrokenLoop:
            pass
        
        stream._loop = BrokenLoop()  # type: ignore
        
        # Should return True (stop) due to exception
        result = stream.callback("token")
        assert result is True


class TestCreateTokenStream:
    """Test cases for create_token_stream helper function."""
    
    def test_create_token_stream_with_running_loop(self):
        """Test creating stream with create_token_stream helper."""
        async def run_test():
            stream = create_token_stream(timeout=45.0)
            
            assert isinstance(stream, AsyncTokenStream)
            assert stream._timeout == 45.0
            assert stream._loop is not None
            assert stream._loop == asyncio.get_event_loop()
        
        asyncio.run(run_test())
    
    def test_create_token_stream_default_timeout(self):
        """Test create_token_stream uses default timeout."""
        async def run_test():
            stream = create_token_stream()
            assert stream._timeout == 60.0
        
        asyncio.run(run_test())


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""
    
    def test_stream_with_simulated_inference_thread(self):
        """Test streaming with simulated concurrent inference thread."""
        async def run_test():
            stream = AsyncTokenStream(timeout=5.0)
            loop = asyncio.get_event_loop()
            stream.set_loop(loop)
            
            # Simulate inference in background task
            async def simulated_inference():
                await asyncio.sleep(0.01)
                stream.callback("The")
                await asyncio.sleep(0.01)
                stream.callback(" ")
                await asyncio.sleep(0.01)
                stream.callback("quick")
                await asyncio.sleep(0.01)
                stream.callback(" ")
                await asyncio.sleep(0.01)
                stream.callback("brown")
                stream.complete()
            
            # Start inference task and consume stream concurrently
            inference_task = asyncio.create_task(simulated_inference())
            
            tokens = []
            async for token in stream:
                tokens.append(token)
            
            await inference_task
            assert "".join(tokens) == "The quick brown"
        
        asyncio.run(run_test())
    
    def test_stream_graceful_stop_on_no_loop(self):
        """Test that stream gracefully handles missing event loop."""
        stream = AsyncTokenStream()
        # Don't set loop
        
        # Callback should return True (stop)
        stop = stream.callback("token")
        assert stop is True
        
        # Complete should not raise
        stream.complete()  # Should be no-op
