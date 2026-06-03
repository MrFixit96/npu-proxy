"""Unit tests for the token streaming module."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from unittest.mock import Mock, patch

import pytest

from npu_proxy.inference.streaming import (
    AsyncTokenStream,
    _await_stream_shutdown,
    create_token_stream,
    stream_engine_tokens,
)


async def collect_stream(stream: AsyncTokenStream) -> list[str]:
    tokens = []
    async for token in stream:
        tokens.append(token)
    return tokens


class TestAsyncTokenStream:
    """Test cases for AsyncTokenStream class."""

    def test_async_token_stream_creation(self):
        """Test creation of AsyncTokenStream with default and custom parameters."""
        stream = AsyncTokenStream()
        assert stream.is_done is False
        assert stream.is_cancelled is False
        assert stream.finish_reason is None
        assert stream.completion_token_count == 0

        stream = AsyncTokenStream(timeout=30.0, max_queue_size=500)
        assert stream.is_done is False
        assert stream.is_cancelled is False

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"timeout": 0}, "timeout must be positive"),
            ({"timeout": -1}, "timeout must be positive"),
            ({"max_queue_size": 0}, "max_queue_size must be positive"),
            ({"max_queue_size": -1}, "max_queue_size must be positive"),
        ],
    )
    def test_constructor_rejects_invalid_arguments(self, kwargs, message):
        """Invalid stream bounds should fail before any async work starts."""
        with pytest.raises(ValueError, match=message):
            AsyncTokenStream(**kwargs)

    def test_set_loop_rejects_non_event_loop(self):
        """set_loop only accepts real asyncio event loops."""
        stream = AsyncTokenStream()
        with pytest.raises(TypeError, match="loop must be"):
            stream.set_loop(object())  # type: ignore[arg-type]

    def test_error_rejects_non_exception(self):
        """error() requires an Exception instance so consumers get a real error."""
        stream = AsyncTokenStream()
        with pytest.raises(TypeError, match="exc must be"):
            stream.error("broken")  # type: ignore[arg-type]

    async def test_stream_callback_pushes_token(self):
        """callback pushes tokens to the queue from the event-loop thread."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())

        assert stream.callback("Hello") is False

        token = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert token == "Hello"
        assert stream.completion_token_count == 1

    def test_stream_callback_without_loop_stops(self):
        """callback returns True (stop) when no loop is set."""
        stream = AsyncTokenStream()

        result = stream.callback("Hello")

        assert result is True

    async def test_stream_complete_sends_sentinel(self):
        """complete() terminates async iteration after queued tokens."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())

        stream.callback("token1")
        stream.complete("stop")

        tokens = await asyncio.wait_for(collect_stream(stream), timeout=1.0)
        assert tokens == ["token1"]
        assert stream.is_done is True
        assert stream.finish_reason == "stop"

    async def test_stream_iteration(self):
        """Test complete iteration over streamed tokens."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())

        test_tokens = ["Hello", " ", "world", "!"]
        for token in test_tokens:
            stream.callback(token)
        stream.complete()

        collected = await asyncio.wait_for(collect_stream(stream), timeout=1.0)
        assert collected == test_tokens

    async def test_stream_timeout_raises(self):
        """TimeoutError is raised when no token arrives before the stream timeout."""
        stream = AsyncTokenStream(timeout=0.01)
        stream.set_loop(asyncio.get_running_loop())

        with pytest.raises(TimeoutError) as exc_info:
            await asyncio.wait_for(stream.__anext__(), timeout=1.0)

        assert "timed out" in str(exc_info.value).lower()
        assert stream.is_done is True

    async def test_stream_error_propagates(self):
        """Errors are propagated to the async iterator after buffered tokens."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())

        stream.callback("partial")
        test_error = ValueError("Inference failed!")
        stream.error(test_error)

        tokens = []
        with pytest.raises(ValueError) as exc_info:
            async with asyncio.timeout(1.0):
                async for token in stream:
                    tokens.append(token)

        assert tokens == ["partial"]
        assert exc_info.value is test_error

    async def test_multiple_tokens_before_complete(self):
        """Queuing multiple tokens before completion preserves order."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())

        for i in range(10):
            stream.callback(f"token{i}")
        stream.complete()

        tokens = await asyncio.wait_for(collect_stream(stream), timeout=1.0)
        assert tokens == [f"token{i}" for i in range(10)]

    def test_callback_with_exception_stops(self):
        """callback stops generation and records enqueue scheduling failure."""
        stream = AsyncTokenStream()
        loop = asyncio.new_event_loop()
        stream.set_loop(loop)

        with patch.object(loop, "is_running", return_value=True), patch(
            "npu_proxy.inference.streaming.asyncio.run_coroutine_threadsafe",
            side_effect=RuntimeError("queue broken"),
        ):
            result = stream.callback("token")

        assert result is True
        loop.close()

    def test_cancelled_callback_does_not_surface_error(self):
        """Consumer cancellation stops generation without fabricating a stream error."""
        stream = AsyncTokenStream()
        stream.cancel()

        result = stream.callback("token")

        assert result is True
        assert stream.is_cancelled is True

    def test_callback_waits_for_enqueue_completion(self):
        """callback waits on the enqueue future so max_queue_size backpressures."""
        stream = AsyncTokenStream()
        loop = asyncio.new_event_loop()
        stream.set_loop(loop)
        put_future = Mock()

        def fake_run_coroutine_threadsafe(coro, _loop):
            coro.close()
            return put_future

        with patch.object(loop, "is_running", return_value=True), patch(
            "npu_proxy.inference.streaming.asyncio.run_coroutine_threadsafe",
            side_effect=fake_run_coroutine_threadsafe,
        ), patch(
            "npu_proxy.inference.streaming.asyncio.get_running_loop",
            side_effect=RuntimeError,
        ):
            result = stream.callback("token")

        assert result is False
        put_future.result.assert_called()
        loop.close()

    def test_callback_cancelled_enqueue_stops_without_error(self):
        """A cancelled cross-thread enqueue tells the producer to stop cleanly."""
        stream = AsyncTokenStream()
        loop = asyncio.new_event_loop()
        stream.set_loop(loop)
        put_future = Mock()
        put_future.result.side_effect = concurrent.futures.CancelledError

        def fake_run_coroutine_threadsafe(coro, _loop):
            coro.close()
            return put_future

        with patch.object(loop, "is_running", return_value=True), patch(
            "npu_proxy.inference.streaming.asyncio.run_coroutine_threadsafe",
            side_effect=fake_run_coroutine_threadsafe,
        ), patch(
            "npu_proxy.inference.streaming.asyncio.get_running_loop",
            side_effect=RuntimeError,
        ):
            result = stream.callback("token")

        assert result is True
        loop.close()

    async def test_done_with_stored_error_raises_on_next(self):
        """A stored producer error must not be hidden by the done flag."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())
        stream.error(RuntimeError("Producer error"))

        with pytest.raises(RuntimeError, match="Producer error"):
            await asyncio.wait_for(stream.__anext__(), timeout=1.0)

    async def test_consumer_cancellation_propagates(self):
        """Cancelling the consumer surfaces CancelledError, not EOF."""
        stream = AsyncTokenStream(timeout=5.0)
        stream.set_loop(asyncio.get_running_loop())

        consumer_task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        consumer_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await consumer_task

        assert stream.is_cancelled is True
        assert stream.is_done is True

    async def test_try_push_on_event_loop_thread_uses_nowait(self):
        """try_push() should not deadlock when called on its own event loop."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())

        assert stream.try_push("token") is True
        assert await asyncio.wait_for(stream.__anext__(), timeout=1.0) == "token"

    async def test_try_push_reports_queue_full_without_ending_stream(self):
        """try_push returns False on a full queue without adding a termination frame."""
        stream = AsyncTokenStream(max_queue_size=1)
        stream.set_loop(asyncio.get_running_loop())

        assert stream.try_push("first") is True
        assert stream.try_push("second") is False
        stream.complete("stop")

        tokens = await asyncio.wait_for(collect_stream(stream), timeout=1.0)
        assert tokens == ["first"]
        assert stream.finish_reason == "stop"

    async def test_try_push_returns_false_when_stream_cannot_accept_tokens(self):
        """try_push fails fast when the stream is not initialized or is terminal."""
        stream = AsyncTokenStream()
        assert stream.try_push("missing-loop") is False

        stream.set_loop(asyncio.get_running_loop())
        stream.complete()
        assert stream.try_push("late") is False

        cancelled_stream = AsyncTokenStream()
        cancelled_stream.set_loop(asyncio.get_running_loop())
        cancelled_stream.cancel()
        assert cancelled_stream.try_push("cancelled") is False

    async def test_callback_queue_full_surfaces_error_after_buffered_token(self):
        """callback reports queue-full as a stream error after draining queued tokens."""
        stream = AsyncTokenStream(max_queue_size=1)
        stream.set_loop(asyncio.get_running_loop())

        assert stream.callback("first") is False
        assert stream.callback("second") is True
        assert await asyncio.wait_for(stream.__anext__(), timeout=1.0) == "first"

        with pytest.raises(RuntimeError, match="queue is full"):
            await asyncio.wait_for(stream.__anext__(), timeout=1.0)

    async def test_collect_honors_max_tokens(self):
        """collect(max_tokens) returns early without losing remaining stream tokens."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())
        for token in ["a", "b", "c"]:
            stream.callback(token)
        stream.complete("stop")

        first_two = await asyncio.wait_for(stream.collect(max_tokens=2), timeout=1.0)
        remainder = await asyncio.wait_for(stream.collect(), timeout=1.0)

        assert first_two == ["a", "b"]
        assert remainder == ["c"]
        assert stream.finish_reason == "stop"

    async def test_complete_rejects_late_tokens_and_remains_idempotent(self):
        """Completion is single-shot and prevents tokens after done."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())
        stream.callback("token1")

        stream.complete("stop")
        stream.complete("length")
        assert stream.callback("late") is True

        tokens = await asyncio.wait_for(collect_stream(stream), timeout=1.0)
        assert tokens == ["token1"]
        assert stream.finish_reason == "length"

    async def test_error_is_idempotent_and_preserves_first_error(self):
        """Only the first error is delivered to consumers."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())
        first_error = RuntimeError("first")

        stream.error(first_error)
        stream.error(RuntimeError("second"))

        with pytest.raises(RuntimeError) as exc_info:
            await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert exc_info.value is first_error

    async def test_same_loop_callback_does_not_use_threadsafe_enqueue(self):
        """callback uses same-loop put_nowait instead of a deadlocking threadsafe future."""
        stream = AsyncTokenStream()
        stream.set_loop(asyncio.get_running_loop())

        with patch(
            "npu_proxy.inference.streaming.asyncio.run_coroutine_threadsafe",
            side_effect=AssertionError("should not be used on the event-loop thread"),
        ):
            assert stream.callback("same-loop") is False

        assert await asyncio.wait_for(stream.__anext__(), timeout=1.0) == "same-loop"


class TestCreateTokenStream:
    """Test cases for create_token_stream helper function."""

    async def test_create_token_stream_with_running_loop(self):
        """create_token_stream sets the running loop and custom timeout."""
        stream = create_token_stream(timeout=45.0)

        assert isinstance(stream, AsyncTokenStream)
        assert stream.try_push("token") is True
        assert await asyncio.wait_for(stream.__anext__(), timeout=1.0) == "token"

    async def test_create_token_stream_default_timeout(self):
        """create_token_stream uses default timeout and can stream immediately."""
        stream = create_token_stream()
        stream.callback("ok")
        stream.complete()

        assert await asyncio.wait_for(stream.collect(), timeout=1.0) == ["ok"]


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    async def test_stream_with_simulated_inference_task(self):
        """Streaming works with a concurrent producer task without asyncio.run wrappers."""
        stream = AsyncTokenStream(timeout=5.0)
        stream.set_loop(asyncio.get_running_loop())
        allow_next_token = asyncio.Event()
        produced = 0

        async def simulated_inference():
            nonlocal produced
            for token in ["The", " ", "quick", " ", "brown"]:
                await asyncio.wait_for(allow_next_token.wait(), timeout=1.0)
                allow_next_token.clear()
                produced += 1
                stream.callback(token)
            stream.complete()

        inference_task = asyncio.create_task(simulated_inference())
        tokens = []
        try:
            for _ in range(5):
                allow_next_token.set()
                tokens.append(await asyncio.wait_for(stream.__anext__(), timeout=1.0))
            await asyncio.wait_for(inference_task, timeout=1.0)
        finally:
            if not inference_task.done():
                inference_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await inference_task

        assert "".join(tokens) == "The quick brown"
        assert produced == 5

    def test_stream_graceful_stop_on_no_loop(self):
        """Stream gracefully handles a missing event loop."""
        stream = AsyncTokenStream()

        assert stream.callback("token") is True
        stream.complete()
        assert stream.is_done is True


async def test_await_stream_shutdown_logs_timeout(monkeypatch, caplog):
    """Shutdown timeout logging is deterministic without waiting for wall clock."""
    pending = asyncio.get_running_loop().create_future()

    async def fake_wait_for(awaitable, timeout):
        assert timeout == 0.25
        raise asyncio.TimeoutError

    monkeypatch.setattr("npu_proxy.inference.streaming.asyncio.wait_for", fake_wait_for)

    with caplog.at_level("WARNING", logger="npu_proxy.inference.streaming"):
        await _await_stream_shutdown(pending, request_id="req-timeout", timeout=0.25)

    assert "did not stop before shutdown timeout" in caplog.text
    pending.cancel()


async def test_stream_engine_tokens_reports_length_finish_reason():
    """Streaming orchestration exposes length when emitted tokens hit max_new_tokens."""

    class FakeEngine:
        def generate_stream(self, prompt, max_new_tokens, temperature, streamer_callback, abort_callback):
            for token in ["A", "B"][:max_new_tokens]:
                streamer_callback(token)
            return iter(())

    finish_reasons = []
    tokens = []
    async for token in stream_engine_tokens(
        engine_factory=FakeEngine,
        prompt="hi",
        max_new_tokens=2,
        temperature=0.0,
        request_id="req-test",
        timeout=1.0,
        finish_reason_callback=finish_reasons.append,
    ):
        tokens.append(token)

    assert tokens == ["A", "B"]
    assert finish_reasons == ["length"]


async def test_stream_engine_tokens_cancelled_consumer_stops_worker():
    """Breaking out of the stream cancels the producer and waits for worker shutdown."""
    worker_started = threading.Event()
    worker_saw_abort = threading.Event()

    class CooperativeEngine:
        def generate_stream(self, prompt, max_new_tokens, temperature, streamer_callback, abort_callback):
            worker_started.set()
            streamer_callback("first")
            while not abort_callback():
                pass
            worker_saw_abort.set()
            return iter(())

    tokens = []
    async for token in stream_engine_tokens(
        engine_factory=CooperativeEngine,
        prompt="hi",
        max_new_tokens=5,
        temperature=0.0,
        request_id="req-cancel",
        timeout=1.0,
    ):
        tokens.append(token)
        break

    assert tokens == ["first"]
    assert worker_started.is_set()
    assert await asyncio.to_thread(worker_saw_abort.wait, 1.0)
