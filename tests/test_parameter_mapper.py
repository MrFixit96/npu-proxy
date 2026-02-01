# tests/test_parameter_mapper.py
import pytest
import logging
from npu_proxy.models.parameter_mapper import map_parameters


class TestDirectMappings:
    def test_map_temperature(self):
        result = map_parameters({"temperature": 0.5})
        assert result["temperature"] == 0.5

    def test_map_top_k(self):
        result = map_parameters({"top_k": 50})
        assert result["top_k"] == 50

    def test_map_top_p(self):
        result = map_parameters({"top_p": 0.95})
        assert result["top_p"] == 0.95

    def test_map_repeat_penalty_renamed(self):
        result = map_parameters({"repeat_penalty": 1.2})
        assert result["repetition_penalty"] == 1.2
        assert "repeat_penalty" not in result

    def test_map_num_predict_renamed(self):
        result = map_parameters({"num_predict": 256})
        assert result["max_new_tokens"] == 256
        assert "num_predict" not in result

    def test_map_stop_renamed(self):
        result = map_parameters({"stop": ["END", "STOP"]})
        assert result["stop_strings"] == ["END", "STOP"]
        assert "stop" not in result

    def test_map_seed(self):
        result = map_parameters({"seed": 42})
        assert result["seed"] == 42


class TestApproximateMappings:
    def test_presence_penalty_maps_to_repetition(self):
        result = map_parameters({"presence_penalty": 0.5})
        assert "repetition_penalty" in result
        assert result["repetition_penalty"] > 1.0

    def test_frequency_penalty_maps_to_repetition(self):
        result = map_parameters({"frequency_penalty": 0.5})
        assert "repetition_penalty" in result

    def test_combined_penalties(self):
        result = map_parameters({"presence_penalty": 0.3, "frequency_penalty": 0.3})
        assert "repetition_penalty" in result

    def test_penalty_conversion_formula(self):
        # Formula: repetition_penalty = 1.0 + (presence + frequency) / 2
        result = map_parameters({"presence_penalty": 0.4, "frequency_penalty": 0.6})
        assert result["repetition_penalty"] == pytest.approx(1.5, rel=0.01)

    def test_explicit_repeat_penalty_not_overridden(self):
        result = map_parameters({"repeat_penalty": 1.3, "presence_penalty": 0.5})
        assert result["repetition_penalty"] == 1.3


class TestIgnoredParams:
    @pytest.mark.parametrize("param", [
        "mirostat", "mirostat_tau", "mirostat_eta",
        "min_p", "typical_p", "tfs_z"
    ])
    def test_unsupported_param_not_in_output(self, param):
        result = map_parameters({param: 1.0})
        assert param not in result

    def test_num_ctx_silently_ignored(self):
        result = map_parameters({"num_ctx": 4096})
        assert "num_ctx" not in result

    def test_num_batch_silently_ignored(self):
        result = map_parameters({"num_batch": 1024})
        assert "num_batch" not in result


class TestLogging:
    def test_ignored_param_logs_debug(self, caplog):
        with caplog.at_level(logging.DEBUG):
            map_parameters({"mirostat": 1})
        assert any("mirostat" in record.message.lower() for record in caplog.records)

    def test_ignored_param_not_logged_at_info(self, caplog):
        with caplog.at_level(logging.INFO):
            caplog.clear()
            map_parameters({"mirostat": 1})
        debug_or_below = [r for r in caplog.records if r.levelno <= logging.DEBUG]
        info_or_above = [r for r in caplog.records if r.levelno >= logging.INFO]
        assert not any("mirostat" in r.message.lower() for r in info_or_above)

    def test_silent_param_not_logged(self, caplog):
        with caplog.at_level(logging.DEBUG):
            map_parameters({"num_ctx": 4096})
        assert not any("num_ctx" in record.message.lower() for record in caplog.records)

    def test_unknown_param_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            map_parameters({"totally_fake_param": 999})
        assert any("totally_fake_param" in record.message.lower() for record in caplog.records)

    def test_unknown_param_not_in_output(self):
        result = map_parameters({"unknown_xyz": 123})
        assert "unknown_xyz" not in result
