"""Tests for context-aware routing."""
import os
import pytest
from unittest.mock import patch

# Reset router before each test module
@pytest.fixture(autouse=True)
def reset_router():
    """Reset context router singleton before each test."""
    from npu_proxy.routing.context_router import reset_context_router
    reset_context_router()
    yield
    reset_context_router()


class TestTokenThresholdDetection:
    """Tests for token threshold detection."""
    
    def test_detects_prompt_exceeding_npu_limit(self):
        """Prompts over 1800 tokens should be flagged as exceeding NPU limit."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        # ~2000 tokens (words ≈ tokens for approximation)
        long_prompt = "word " * 2000
        assert router.exceeds_npu_limit(long_prompt) == True
    
    def test_allows_prompt_within_npu_limit(self):
        """Short prompts should not exceed NPU limit."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        short_prompt = "Hello world"
        assert router.exceeds_npu_limit(short_prompt) == False
    
    def test_edge_case_at_exact_limit(self):
        """Prompts at exact token limit should be allowed."""
        from npu_proxy.routing.context_router import ContextRouter
        from npu_proxy.inference.tokenizer import count_tokens
        
        # Create prompt and measure its actual token count
        prompt = "hello " * 100
        token_count = count_tokens(prompt)
        
        # Create router with limit exactly at token count
        router_at_limit = ContextRouter(npu_limit=token_count)
        
        # At limit should NOT exceed (uses >)
        assert router_at_limit.exceeds_npu_limit(prompt) == False
    
    def test_edge_case_one_over_limit(self):
        """Prompts one token over limit should exceed."""
        from npu_proxy.routing.context_router import ContextRouter
        from npu_proxy.inference.tokenizer import count_tokens
        
        prompt = "hello " * 100
        token_count = count_tokens(prompt)
        
        # Create router with limit one below token count
        router_under_limit = ContextRouter(npu_limit=token_count - 1)
        
        # One over limit should exceed
        assert router_under_limit.exceeds_npu_limit(prompt) == True
    
    def test_empty_prompt_within_limit(self):
        """Empty prompts should always be within limit."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        assert router.exceeds_npu_limit("") == False
        assert router.exceeds_npu_limit("   ") == False


class TestDeviceRouting:
    """Tests for device routing logic."""
    
    def test_routes_long_prompt_to_cpu(self):
        """Prompts exceeding NPU limit should route to CPU."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        result = router.select_device("word " * 2000)
        assert result.device == "CPU"
        assert result.reason == "prompt_exceeds_npu_limit"
    
    def test_routes_short_prompt_to_npu(self):
        """Short prompts should route to preferred device (NPU)."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        result = router.select_device("Hello world")
        assert result.device == "NPU"
        assert result.reason == "within_npu_limit"
    
    def test_routes_with_custom_fallback_device(self):
        """Long prompts should use custom fallback device."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800, fallback_device="GPU")
        result = router.select_device("word " * 2000)
        assert result.device == "GPU"
    
    def test_routes_with_custom_preferred_device(self):
        """Short prompts should use custom preferred device."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800, preferred_device="GPU")
        result = router.select_device("Hello world")
        assert result.device == "GPU"
    
    def test_routing_result_includes_token_count(self):
        """Routing result should include actual token count."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        result = router.select_device("Hello world, this is a test.")
        assert result.token_count > 0
        assert isinstance(result.token_count, int)
    
    def test_routes_multi_message_conversation(self):
        """Router should handle multi-message chat format."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=100)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "word " * 150},  # Exceeds
        ]
        result = router.select_device_for_messages(messages)
        assert result.device == "CPU"
    
    def test_routes_short_multi_message_to_npu(self):
        """Short conversations should route to NPU."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        result = router.select_device_for_messages(messages)
        assert result.device == "NPU"
        assert result.reason == "within_npu_limit"
    
    def test_multi_message_token_count_is_sum(self):
        """Multi-message token count should be sum of all messages."""
        from npu_proxy.routing.context_router import ContextRouter
        from npu_proxy.inference.tokenizer import count_tokens
        
        router = ContextRouter(npu_limit=1800)
        messages = [
            {"role": "system", "content": "System prompt here."},
            {"role": "user", "content": "User message here."},
        ]
        result = router.select_device_for_messages(messages)
        
        expected_count = (
            count_tokens("System prompt here.") +
            count_tokens("User message here.")
        )
        assert result.token_count == expected_count
    
    def test_empty_message_content_handled(self):
        """Messages with empty content should be handled."""
        from npu_proxy.routing.context_router import ContextRouter
        router = ContextRouter(npu_limit=1800)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        result = router.select_device_for_messages(messages)
        assert result.device == "NPU"


class TestContextRouterConfiguration:
    """Tests for environment-based configuration."""
    
    def test_npu_limit_configurable_via_env(self):
        """NPU token limit should be configurable via environment variable."""
        from npu_proxy.routing.context_router import reset_context_router, get_context_router
        reset_context_router()
        with patch.dict(os.environ, {"NPU_PROXY_TOKEN_LIMIT": "1000"}):
            router = get_context_router()
            assert router.npu_limit == 1000
    
    def test_default_npu_limit_is_1800(self):
        """Default NPU limit should be 1800 tokens."""
        from npu_proxy.routing.context_router import ContextRouter
        
        # Clear env var if set and test default
        with patch.dict(os.environ, {}, clear=True):
            # Remove specific var if it exists
            os.environ.pop("NPU_PROXY_TOKEN_LIMIT", None)
            router = ContextRouter()
            assert router.npu_limit == 1800
    
    def test_fallback_device_configurable(self):
        """Fallback device should be configurable."""
        from npu_proxy.routing.context_router import reset_context_router, get_context_router
        reset_context_router()
        with patch.dict(os.environ, {"NPU_PROXY_FALLBACK_DEVICE": "GPU"}):
            router = get_context_router()
            assert router.fallback_device == "GPU"
    
    def test_preferred_device_configurable(self):
        """Preferred device should be configurable."""
        from npu_proxy.routing.context_router import reset_context_router, get_context_router
        reset_context_router()
        with patch.dict(os.environ, {"NPU_PROXY_PREFERRED_DEVICE": "GPU"}):
            router = get_context_router()
            assert router.preferred_device == "GPU"
    
    def test_invalid_limit_uses_default(self):
        """Invalid token limit should fall back to default."""
        from npu_proxy.routing.context_router import reset_context_router, get_context_router
        reset_context_router()
        with patch.dict(os.environ, {"NPU_PROXY_TOKEN_LIMIT": "not_a_number"}):
            router = get_context_router()
            assert router.npu_limit == 1800  # Default
    
    def test_singleton_pattern(self):
        """get_context_router should return same instance."""
        from npu_proxy.routing.context_router import reset_context_router, get_context_router
        reset_context_router()
        router1 = get_context_router()
        router2 = get_context_router()
        assert router1 is router2
    
    def test_reset_creates_new_instance(self):
        """reset_context_router should force new instance creation."""
        from npu_proxy.routing.context_router import reset_context_router, get_context_router
        reset_context_router()
        router1 = get_context_router()
        reset_context_router()
        router2 = get_context_router()
        assert router1 is not router2
    
    def test_constructor_overrides_env(self):
        """Constructor arguments should override environment variables."""
        from npu_proxy.routing.context_router import ContextRouter
        
        with patch.dict(os.environ, {"NPU_PROXY_TOKEN_LIMIT": "1000"}):
            router = ContextRouter(npu_limit=500)
            assert router.npu_limit == 500
    
    def test_default_devices(self):
        """Default devices should be NPU (preferred) and CPU (fallback)."""
        from npu_proxy.routing.context_router import ContextRouter
        
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NPU_PROXY_PREFERRED_DEVICE", None)
            os.environ.pop("NPU_PROXY_FALLBACK_DEVICE", None)
            router = ContextRouter()
            assert router.preferred_device == "NPU"
            assert router.fallback_device == "CPU"


class TestDeviceFallbackChain:
    """Tests for device fallback chain (NPU → GPU → CPU)."""
    
    def test_fallback_uses_gpu_when_available(self):
        """When GPU is available, it should be used as fallback before CPU."""
        from npu_proxy.routing.context_router import get_fallback_device, reset_context_router
        
        reset_context_router()
        # Mock available devices to include GPU
        with patch('npu_proxy.routing.context_router.get_available_devices', return_value=["NPU", "GPU", "CPU"]):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("NPU_PROXY_FALLBACK_DEVICE", None)
                os.environ.pop("NPU_PROXY_PREFERRED_DEVICE", None)
                fallback = get_fallback_device()
                assert fallback == "GPU", f"Expected GPU as fallback when available, got {fallback}"
    
    def test_fallback_uses_cpu_when_no_gpu(self):
        """When GPU is not available, CPU should be fallback."""
        from npu_proxy.routing.context_router import get_fallback_device, reset_context_router
        
        reset_context_router()
        # Mock available devices without GPU
        with patch('npu_proxy.routing.context_router.get_available_devices', return_value=["NPU", "CPU"]):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("NPU_PROXY_FALLBACK_DEVICE", None)
                fallback = get_fallback_device()
                assert fallback == "CPU"
    
    def test_routing_uses_gpu_fallback_for_long_prompt(self):
        """Long prompts should route to GPU when available, not CPU."""
        from npu_proxy.routing.context_router import ContextRouter, reset_context_router
        
        reset_context_router()
        # Mock GPU available
        with patch('npu_proxy.routing.context_router.get_available_devices', return_value=["NPU", "GPU", "CPU"]):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("NPU_PROXY_FALLBACK_DEVICE", None)
                router = ContextRouter(npu_limit=100)
                result = router.select_device("word " * 200)  # Exceeds limit
                assert result.device == "GPU", f"Expected GPU fallback, got {result.device}"
                assert result.reason == "prompt_exceeds_npu_limit"
    
    def test_env_override_takes_precedence(self):
        """Explicit env var should override dynamic fallback detection."""
        from npu_proxy.routing.context_router import get_fallback_device, reset_context_router
        
        reset_context_router()
        # Even with GPU available, explicit CPU override should win
        with patch('npu_proxy.routing.context_router.get_available_devices', return_value=["NPU", "GPU", "CPU"]):
            with patch.dict(os.environ, {"NPU_PROXY_FALLBACK_DEVICE": "CPU"}):
                fallback = get_fallback_device()
                assert fallback == "CPU"
