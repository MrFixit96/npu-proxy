# tests/test_ollama_defaults.py
import pytest
from npu_proxy.models.ollama_defaults import OLLAMA_DEFAULTS, get_default, merge_with_defaults


class TestOllamaDefaultsValues:
    def test_temperature_default(self):
        assert OLLAMA_DEFAULTS["temperature"] == 0.8

    def test_top_k_default(self):
        assert OLLAMA_DEFAULTS["top_k"] == 40

    def test_top_p_default(self):
        assert OLLAMA_DEFAULTS["top_p"] == 0.9

    def test_num_predict_default(self):
        assert OLLAMA_DEFAULTS["num_predict"] == 128

    def test_repeat_penalty_default(self):
        assert OLLAMA_DEFAULTS["repeat_penalty"] == 1.1

    def test_seed_default(self):
        assert OLLAMA_DEFAULTS["seed"] == 0

    def test_stop_default(self):
        assert OLLAMA_DEFAULTS["stop"] == []

    def test_num_ctx_default(self):
        assert OLLAMA_DEFAULTS["num_ctx"] == 2048

    def test_mirostat_default(self):
        assert OLLAMA_DEFAULTS["mirostat"] == 0


class TestGetDefault:
    def test_get_default_returns_value(self):
        assert get_default("temperature") == 0.8

    def test_get_default_unknown_raises(self):
        with pytest.raises(KeyError):
            get_default("unknown_param")


class TestMergeWithDefaults:
    def test_merge_fills_missing(self):
        result = merge_with_defaults({})
        assert result["temperature"] == 0.8
        assert result["top_k"] == 40

    def test_merge_preserves_user_values(self):
        result = merge_with_defaults({"temperature": 0.5})
        assert result["temperature"] == 0.5
        assert result["top_k"] == 40  # default still applied
