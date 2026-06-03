"""Focused runtime cache and diagnostics tests for the inference engine."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from npu_proxy.config import (
    LLMBackend,
    load_proxy_bootstrap_config,
    reset_active_proxy_bootstrap_config,
)
from npu_proxy.inference.engine import InferenceEngine, get_llm_engine, reset_engine


class BlockingPipeline:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.result = MagicMock()
        self.result.texts = ["unblocked"]

    def generate(self, prompt, config, streamer=None):
        self.release.wait(timeout=5)
        if streamer is not None:
            streamer("unblocked")
        return self.result


def test_engine_applies_runtime_cache_config_on_npu():
    """Compile cache and prefix cache config should flow into the NPU pipeline."""
    with patch("npu_proxy.inference.engine.select_best_device", return_value=("NPU", None)):
        with patch.object(Path, "mkdir") as mock_mkdir:
            with patch("npu_proxy.inference.engine.ov_genai.LLMPipeline") as mock_pipeline:
                InferenceEngine(
                    "model-dir",
                    device="NPU",
                    compile_cache_dir="build\\runtime-cache",
                    compile_cache_mode="optimize_speed",
                    prefix_cache_mode="on",
                )

    mock_mkdir.assert_called_once()
    config = mock_pipeline.call_args.args[2]
    assert config["MAX_PROMPT_LEN"] > 0
    assert config["CACHE_DIR"] == "build\\runtime-cache"
    assert config["CACHE_MODE"] == "OPTIMIZE_SPEED"
    assert config["NPUW_LLM_ENABLE_PREFIX_CACHING"] == "YES"


def test_engine_safely_degrades_prefix_cache_when_pipeline_rejects_it():
    """Prefix cache failures should retry safely without breaking model load."""
    with patch("npu_proxy.inference.engine.select_best_device", return_value=("NPU", None)):
        with patch(
            "npu_proxy.inference.engine.ov_genai.LLMPipeline",
            side_effect=[RuntimeError("prefix cache unsupported"), MagicMock()],
        ) as mock_pipeline:
            engine = InferenceEngine(
                "model-dir",
                device="NPU",
                prefix_cache_mode="on",
            )

    assert mock_pipeline.call_count == 2
    first_config = mock_pipeline.call_args_list[0].args[2]
    second_config = mock_pipeline.call_args_list[1].args[2]
    assert first_config["NPUW_LLM_ENABLE_PREFIX_CACHING"] == "YES"
    assert "NPUW_LLM_ENABLE_PREFIX_CACHING" not in second_config

    info = engine.get_device_info()
    assert info["runtime_features"]["degraded_features"] == ["prefix_cache"]
    assert info["runtime_features"]["prefix_cache_enabled"] is None


def test_engine_retries_without_compile_cache_when_pipeline_rejects_it():
    """Compile-cache failures should retry without cache settings and surface diagnostics."""
    with patch("npu_proxy.inference.engine.select_best_device", return_value=("NPU", None)):
        with patch.object(Path, "mkdir") as mock_mkdir:
            with patch(
                "npu_proxy.inference.engine.ov_genai.LLMPipeline",
                side_effect=[RuntimeError("compile cache unsupported"), MagicMock()],
            ) as mock_pipeline:
                engine = InferenceEngine(
                    "model-dir",
                    device="NPU",
                    compile_cache_dir="build\\runtime-cache",
                    compile_cache_mode="optimize_speed",
                )

    mock_mkdir.assert_called_once()
    assert mock_pipeline.call_count == 2
    first_config = mock_pipeline.call_args_list[0].args[2]
    second_config = mock_pipeline.call_args_list[1].args[2]
    assert first_config["CACHE_DIR"] == "build\\runtime-cache"
    assert first_config["CACHE_MODE"] == "OPTIMIZE_SPEED"
    assert "CACHE_DIR" not in second_config
    assert "CACHE_MODE" not in second_config

    info = engine.get_device_info()
    assert info["runtime_features"]["compile_cache_enabled"] is False
    assert info["runtime_features"]["prefix_cache_enabled"] is None
    assert info["runtime_features"]["degraded_features"] == ["compile_cache"]
    assert info["load_diagnostics"][0]["status"] == "failed"
    assert info["load_diagnostics"][0]["degraded_features"] == []
    assert info["load_diagnostics"][1]["status"] == "loaded"
    assert info["load_diagnostics"][1]["degraded_features"] == ["compile_cache"]
    assert "CACHE_DIR" not in info["load_diagnostics"][1]["config"]
    assert "CACHE_MODE" not in info["load_diagnostics"][1]["config"]


def test_engine_reports_compile_cache_dir_failures_and_loads_without_cache():
    """Unavailable compile-cache directories should degrade cleanly into uncached loads."""
    with patch("npu_proxy.inference.engine.select_best_device", return_value=("NPU", None)):
        with patch.object(Path, "mkdir", side_effect=OSError("access denied")):
            with patch(
                "npu_proxy.inference.engine.ov_genai.LLMPipeline",
                return_value=MagicMock(),
            ) as mock_pipeline:
                engine = InferenceEngine(
                    "model-dir",
                    device="NPU",
                    compile_cache_dir="build\\runtime-cache",
                    compile_cache_mode="optimize_speed",
                )

    config = mock_pipeline.call_args.args[2]
    assert "CACHE_DIR" not in config
    assert "CACHE_MODE" not in config

    info = engine.get_device_info()
    assert info["runtime_features"]["compile_cache_enabled"] is False
    assert info["runtime_features"]["degraded_features"] == []
    assert info["load_diagnostics"][0]["status"] == "compile_cache_dir_unavailable"
    assert info["load_diagnostics"][0]["compile_cache_dir"] == "build\\runtime-cache"
    assert info["load_diagnostics"][1]["status"] == "loaded"
    assert "CACHE_DIR" not in info["load_diagnostics"][1]["config"]


def test_engine_walks_full_runtime_cache_retry_ladder_when_both_caches_enabled():
    """Load retries should walk the full cache-degradation ladder before succeeding."""
    with patch("npu_proxy.inference.engine.select_best_device", return_value=("NPU", None)):
        with patch.object(Path, "mkdir"):
            with patch(
                "npu_proxy.inference.engine.ov_genai.LLMPipeline",
                side_effect=[
                    RuntimeError("full config unsupported"),
                    RuntimeError("prefix cache unsupported"),
                    RuntimeError("compile cache unsupported"),
                    MagicMock(),
                ],
            ) as mock_pipeline:
                engine = InferenceEngine(
                    "model-dir",
                    device="NPU",
                    compile_cache_dir="build\\runtime-cache",
                    compile_cache_mode="optimize_speed",
                    prefix_cache_mode="on",
                )

    assert mock_pipeline.call_count == 4
    configs = [call.args[2] for call in mock_pipeline.call_args_list]
    assert configs[0]["CACHE_DIR"] == "build\\runtime-cache"
    assert configs[0]["CACHE_MODE"] == "OPTIMIZE_SPEED"
    assert configs[0]["NPUW_LLM_ENABLE_PREFIX_CACHING"] == "YES"
    assert configs[1]["CACHE_DIR"] == "build\\runtime-cache"
    assert configs[1]["CACHE_MODE"] == "OPTIMIZE_SPEED"
    assert "NPUW_LLM_ENABLE_PREFIX_CACHING" not in configs[1]
    assert "CACHE_DIR" not in configs[2]
    assert "CACHE_MODE" not in configs[2]
    assert configs[2]["NPUW_LLM_ENABLE_PREFIX_CACHING"] == "YES"
    assert "CACHE_DIR" not in configs[3]
    assert "CACHE_MODE" not in configs[3]
    assert "NPUW_LLM_ENABLE_PREFIX_CACHING" not in configs[3]

    info = engine.get_device_info()
    assert info["runtime_features"]["compile_cache_enabled"] is False
    assert info["runtime_features"]["prefix_cache_enabled"] is None
    assert info["runtime_features"]["degraded_features"] == [
        "prefix_cache",
        "compile_cache",
    ]
    assert [entry["status"] for entry in info["load_diagnostics"]] == [
        "failed",
        "failed",
        "failed",
        "loaded",
    ]
    assert [entry["degraded_features"] for entry in info["load_diagnostics"]] == [
        [],
        ["prefix_cache"],
        ["compile_cache"],
        ["prefix_cache", "compile_cache"],
    ]


def test_engine_falls_back_after_exhausting_npu_runtime_retry_ladder():
    """Engine should continue to fallback devices after NPU cache retries are exhausted."""
    with patch("npu_proxy.inference.engine.select_best_device", return_value=("NPU", "CPU")):
        with patch.object(Path, "mkdir"):
            with patch(
                "npu_proxy.inference.engine.ov_genai.LLMPipeline",
                side_effect=[
                    RuntimeError("full config unsupported"),
                    RuntimeError("prefix cache unsupported"),
                    RuntimeError("compile cache unsupported"),
                    RuntimeError("npu base unsupported"),
                    MagicMock(),
                ],
            ) as mock_pipeline:
                engine = InferenceEngine(
                    "model-dir",
                    device="NPU",
                    compile_cache_dir="build\\runtime-cache",
                    compile_cache_mode="optimize_speed",
                    prefix_cache_mode="on",
                )

    assert mock_pipeline.call_count == 5
    assert engine.used_fallback is True
    assert engine.actual_device == "CPU"

    info = engine.get_device_info()
    assert info["runtime_features"]["compile_cache_enabled"] is True
    assert info["runtime_features"]["prefix_cache_enabled"] is False
    assert info["runtime_features"]["degraded_features"] == ["prefix_cache"]
    assert [entry["device"] for entry in info["load_diagnostics"]] == [
        "NPU",
        "NPU",
        "NPU",
        "NPU",
        "CPU",
    ]
    assert [entry["status"] for entry in info["load_diagnostics"]] == [
        "failed",
        "failed",
        "failed",
        "failed",
        "loaded",
    ]
    assert info["load_diagnostics"][-1]["degraded_features"] == ["prefix_cache"]


def test_generate_collects_runtime_diagnostics_and_metrics():
    """Successful generation should update additive runtime diagnostics."""
    perf_metrics = MagicMock()
    perf_metrics.get_load_time.return_value = 250.0
    perf_metrics.get_generate_duration.return_value = SimpleNamespace(mean=1500.0)
    perf_metrics.get_inference_duration.return_value = SimpleNamespace(mean=1200.0)
    perf_metrics.get_tokenization_duration.return_value = SimpleNamespace(mean=30.0)
    perf_metrics.get_detokenization_duration.return_value = SimpleNamespace(mean=20.0)
    perf_metrics.get_ttft.return_value = SimpleNamespace(mean=400.0)
    perf_metrics.get_tpot.return_value = SimpleNamespace(mean=40.0)
    perf_metrics.get_throughput.return_value = SimpleNamespace(mean=25.0)
    perf_metrics.get_num_input_tokens.return_value = 12
    perf_metrics.get_num_generated_tokens.return_value = 24

    result = MagicMock()
    result.texts = ["hello world"]
    result.perf_metrics = perf_metrics

    pipeline = MagicMock()
    pipeline.generate.return_value = result

    with patch("npu_proxy.inference.engine.select_best_device", return_value=("CPU", None)):
        with patch("npu_proxy.inference.engine.ov_genai.LLMPipeline", return_value=pipeline):
            with patch("npu_proxy.inference.engine.record_inference") as record_inference:
                with patch("npu_proxy.inference.engine.record_ttft") as record_ttft:
                    with patch("npu_proxy.inference.engine.record_tpot") as record_tpot:
                        with patch(
                            "npu_proxy.inference.engine.record_tokens_per_second"
                        ) as record_tps:
                            engine = InferenceEngine("model-dir", device="CPU")
                            output = engine.generate("hello")

    assert output == "hello world"
    stats = engine.get_device_info()["last_generation_stats"]
    assert stats["generate_duration_seconds"] == 1.5
    assert stats["ttft_seconds"] == 0.4
    assert stats["throughput_tokens_per_second"] == 25.0
    record_inference.assert_called_once_with(engine.model_name, "cpu", "chat", 1.5)
    record_ttft.assert_called_once_with(engine.model_name, 0.4)
    record_tpot.assert_called_once_with(engine.model_name, 0.04)
    record_tps.assert_called_once_with(engine.model_name, "cpu", 25.0)


def test_engine_reports_prefix_cache_degradation_on_non_npu():
    """Runtime diagnostics should show explicit prefix-cache drops on non-NPU devices."""
    with patch("npu_proxy.inference.engine.select_best_device", return_value=("CPU", None)):
        with patch("npu_proxy.inference.engine.ov_genai.LLMPipeline", return_value=MagicMock()):
            engine = InferenceEngine("model-dir", device="CPU", prefix_cache_mode="on")

    info = engine.get_device_info()
    assert info["runtime_features"]["prefix_cache_enabled"] is False
    assert "prefix_cache" in info["runtime_features"]["degraded_features"]


def test_get_llm_engine_reads_runtime_cache_env():
    """Singleton construction should plumb runtime cache env vars into the engine."""
    reset_engine()
    reset_active_proxy_bootstrap_config()
    try:
        model_path = str(Path(__file__).resolve().parent)
        with patch.dict(
            "os.environ",
            {
                "NPU_PROXY_COMPILE_CACHE_DIR": "build\\cache",
                "NPU_PROXY_COMPILE_CACHE_MODE": "OPTIMIZE_SIZE",
                "NPU_PROXY_PREFIX_CACHE_MODE": "off",
            },
            clear=False,
        ):
            with patch("npu_proxy.inference.engine.InferenceEngine") as mock_engine:
                instance = MagicMock()
                instance.model_name = "tests"
                mock_engine.return_value = instance
                engine = get_llm_engine(model_path=model_path, device="CPU")

        assert engine is instance
        mock_engine.assert_called_once_with(
            Path(model_path),
            "CPU",
            inference_timeout=180,
            max_prompt_len=4096,
            compile_cache_dir=Path("build\\cache"),
            compile_cache_mode="OPTIMIZE_SIZE",
            prefix_cache_mode="off",
        )
    finally:
        reset_engine()
        reset_active_proxy_bootstrap_config()


def test_get_llm_engine_rejects_non_openvino_authoritative_backend(tmp_path):
    """The legacy engine path should fail closed for non-OpenVINO backends."""
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_text("gguf")

    with pytest.raises(Exception, match="Legacy engine path only supports the openvino backend"):
        get_llm_engine(
            config=load_proxy_bootstrap_config(
                env={},
                backend=LLMBackend.LLAMA_CPP,
                device="CPU",
                enable_alpha_backends=True,
                llama_cpp_model_path=gguf_path,
            ).llm
        )


def test_generate_timeout_returns_promptly():
    """Timed-out generation should return without waiting for worker drain."""
    pipeline = BlockingPipeline()

    with patch("npu_proxy.inference.engine.select_best_device", return_value=("CPU", None)):
        with patch("npu_proxy.inference.engine.ov_genai.LLMPipeline", return_value=pipeline):
            engine = InferenceEngine("model-dir", device="CPU")

    try:
        started_at = time.perf_counter()
        with pytest.raises(TimeoutError):
            engine.generate("hello", timeout=0.05)
        assert time.perf_counter() - started_at < 0.5
    finally:
        pipeline.release.set()
        if engine._active_future is not None:
            engine._active_future.result(timeout=1)
        engine.shutdown(wait=False)


def test_generate_rejects_new_call_while_timed_out_worker_drains():
    """Timed-out in-flight work should cause later calls to fail fast."""
    pipeline = BlockingPipeline()

    with patch("npu_proxy.inference.engine.select_best_device", return_value=("CPU", None)):
        with patch("npu_proxy.inference.engine.ov_genai.LLMPipeline", return_value=pipeline):
            engine = InferenceEngine("model-dir", device="CPU")

    try:
        with pytest.raises(TimeoutError):
            engine.generate("hello", timeout=0.05)

        started_at = time.perf_counter()
        with pytest.raises(TimeoutError, match="still running"):
            engine.generate("again", timeout=0.05)
        assert time.perf_counter() - started_at < 0.2

        pipeline.release.set()
        engine._active_future.result(timeout=1)

        assert engine.generate("recovered", timeout=0.2) == "unblocked"
    finally:
        pipeline.release.set()
        engine.shutdown(wait=False)


def test_generate_stream_timeout_returns_promptly():
    """Streaming timeouts should surface promptly without waiting for worker shutdown."""
    pipeline = BlockingPipeline()

    with patch("npu_proxy.inference.engine.select_best_device", return_value=("CPU", None)):
        with patch("npu_proxy.inference.engine.ov_genai.LLMPipeline", return_value=pipeline):
            engine = InferenceEngine("model-dir", device="CPU")

    try:
        started_at = time.perf_counter()
        with pytest.raises(TimeoutError):
            list(engine.generate_stream("hello", timeout=0.05))
        assert time.perf_counter() - started_at < 0.5
    finally:
        pipeline.release.set()
        if engine._active_future is not None:
            engine._active_future.result(timeout=1)
        engine.shutdown(wait=False)
