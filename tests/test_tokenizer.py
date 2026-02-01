"""Tests for accurate token counting - TDD P2-2"""
import pytest


# 🔴 RED: These tests should FAIL initially

def test_count_tokens_more_accurate_than_word_split():
    """Token count should be more accurate than simple word split"""
    from npu_proxy.inference.tokenizer import count_tokens
    
    # "Hello, world!" is typically 4 tokens but 2 words
    text = "Hello, world!"
    
    token_count = count_tokens(text)
    word_count = len(text.split())
    
    # Token count should be >= word count (punctuation = extra tokens)
    assert token_count >= word_count
    assert token_count == 4  # Expected: "Hello", ",", "world", "!"


def test_count_tokens_handles_special_chars():
    """Token count should handle special characters correctly"""
    from npu_proxy.inference.tokenizer import count_tokens
    
    text = "What's the time? It's 3:30pm."
    
    token_count = count_tokens(text)
    
    # Should be more than word count due to contractions and punctuation
    assert token_count > len(text.split())


def test_count_tokens_handles_empty_string():
    """Token count should return 0 for empty string"""
    from npu_proxy.inference.tokenizer import count_tokens
    
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0


def test_count_tokens_fallback_on_error():
    """Token count should fallback to word split if tokenizer fails"""
    from npu_proxy.inference.tokenizer import count_tokens_safe
    
    text = "Hello world"
    
    # Should work even without a proper tokenizer loaded
    count = count_tokens_safe(text, fallback_to_words=True)
    
    assert count >= 2  # At least word count


@pytest.mark.asyncio
async def test_chat_response_uses_accurate_tokens():
    """Chat response usage should use accurate token counting"""
    from httpx import AsyncClient, ASGITransport
    from npu_proxy.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat-int4-ov",
                "messages": [{"role": "user", "content": "Hello, world!"}],
            },
        )
    
    data = response.json()
    usage = data["usage"]
    
    # prompt_tokens should reflect actual tokenization
    # "Hello, world!" = 4 tokens, plus template overhead
    assert usage["prompt_tokens"] >= 4
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
