"""
Tests for the NPU Proxy Benchmark CLI Tool.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.benchmark import NPUProxyBenchmark, BenchmarkResult, main


class TestBenchmarkCLI:
    """Test CLI argument parsing and help."""

    def test_benchmark_cli_help(self, capsys):
        """Test that --help works correctly."""
        with patch("sys.argv", ["benchmark.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "NPU Proxy Inference Benchmark" in captured.out
            assert "run" in captured.out
            assert "compare" in captured.out

    def test_benchmark_run_help(self, capsys):
        """Test that run subcommand --help works."""
        with patch("sys.argv", ["benchmark.py", "run", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "--device" in captured.out
            assert "--model" in captured.out
            assert "--iterations" in captured.out

    def test_benchmark_compare_help(self, capsys):
        """Test that compare subcommand --help works."""
        with patch("sys.argv", ["benchmark.py", "compare", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "--baseline" in captured.out

    def test_benchmark_compare_command_passes_both_json_paths(self):
        """Compare mode should pass baseline and current JSON inputs through the CLI."""
        with (
            patch.object(NPUProxyBenchmark, "compare_results", autospec=True) as mock_compare_results,
            patch("sys.argv", ["benchmark.py", "compare", "--baseline", "baseline.json", "current.json"]),
        ):
            result = main()

        assert result == 0
        _, baseline_path, current_path = mock_compare_results.call_args.args
        assert baseline_path == "baseline.json"
        assert current_path == "current.json"

    def test_benchmark_run_without_device_enumerates_available_devices(self):
        """Run mode should benchmark each discovered device instead of falling back to a None placeholder."""
        with (
            patch("scripts.benchmark.get_available_devices", return_value=["CPU", "GPU"]) as mock_get_available_devices,
            patch.object(NPUProxyBenchmark, "run_benchmark", autospec=True, return_value=None) as mock_run_benchmark,
            patch.object(NPUProxyBenchmark, "print_results", autospec=True),
            patch("sys.argv", [
                "benchmark.py",
                "run",
                "--model",
                "tinyllama",
                "--iterations",
                "1",
                "--warmup",
                "0",
            ]),
        ):
            result = main()

        assert result == 1
        assert mock_get_available_devices.call_count == 1
        assert [call.kwargs["device"] for call in mock_run_benchmark.call_args_list] == ["CPU", "GPU"]
        assert all(call.kwargs["device"] is not None for call in mock_run_benchmark.call_args_list)


class TestBenchmarkResult:
    """Test BenchmarkResult data structure."""

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = BenchmarkResult(
            device="NPU",
            model="tinyllama",
            cold_start_ms=1000.5,
            warm_inference_ms=50.25,
            tokens_per_second=20.0,
        )
        result_dict = result.to_dict()

        assert result_dict["device"] == "NPU"
        assert result_dict["model"] == "tinyllama"
        assert result_dict["cold_start_ms"] == 1000.5
        assert result_dict["warm_inference_ms"] == 50.25
        assert result_dict["tokens_per_second"] == 20.0

    def test_result_dict_excludes_none(self):
        """Test that None values are excluded from dict."""
        result = BenchmarkResult(
            device="CPU",
            model="tinyllama",
            cold_start_ms=100.0,
            warm_inference_ms=50.0,
            embedding_ms=None,
        )
        result_dict = result.to_dict()

        assert "embedding_ms" not in result_dict
        assert "first_token_ms" not in result_dict


class TestBenchmarkJSONOutput:
    """Test JSON output formatting and validation."""

    def test_benchmark_json_output_format(self):
        """Test that JSON output format is valid."""
        benchmark = NPUProxyBenchmark()

        # Add mock results
        result1 = BenchmarkResult(
            device="NPU",
            model="tinyllama",
            cold_start_ms=8120.0,
            warm_inference_ms=4030.0,
            tokens_per_second=12.5,
        )
        result2 = BenchmarkResult(
            device="CPU",
            model="tinyllama",
            cold_start_ms=15000.0,
            warm_inference_ms=8500.0,
            tokens_per_second=5.2,
        )
        benchmark.results = [result1, result2]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            benchmark.export_json(str(output_path))

            # Load and validate JSON
            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)

            # Check structure
            assert "timestamp" in data
            assert "system" in data
            assert "results" in data

            # Validate timestamp format
            assert data["timestamp"].endswith("Z")

            # Check system info
            assert "python" in data["system"]
            assert "platform" in data["system"]
            assert "openvino" in data["system"]

            # Check results array
            assert len(data["results"]) == 2

            # Validate first result
            assert data["results"][0]["device"] == "NPU"
            assert data["results"][0]["model"] == "tinyllama"
            assert data["results"][0]["cold_start_ms"] == 8120.0
            assert data["results"][0]["warm_inference_ms"] == 4030.0
            assert data["results"][0]["tokens_per_second"] == 12.5

    def test_json_output_creates_parent_directories(self):
        """Test that parent directories are created if needed."""
        benchmark = NPUProxyBenchmark()
        result = BenchmarkResult(
            device="CPU",
            model="tinyllama",
            cold_start_ms=100.0,
            warm_inference_ms=50.0,
        )
        benchmark.results = [result]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "deep" / "results.json"
            benchmark.export_json(str(output_path))

            assert output_path.exists()

    def test_json_output_is_valid_json(self):
        """Test that output can be parsed as valid JSON."""
        benchmark = NPUProxyBenchmark()
        result = BenchmarkResult(
            device="GPU",
            model="bert",
            cold_start_ms=5000.0,
            warm_inference_ms=100.0,
            embedding_ms=50.0,
        )
        benchmark.results = [result]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            benchmark.export_json(str(output_path))

            # Should not raise
            with open(output_path) as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert isinstance(data["results"], list)


class TestBenchmarkOnCPU:
    """Test benchmark execution on CPU device."""

    @patch("scripts.benchmark.get_llm_engine")
    def test_benchmark_runs_on_cpu(self, mock_get_llm_engine):
        """Test running a minimal benchmark on CPU."""
        # Mock the engine with a generate method
        mock_engine = MagicMock()
        mock_engine.device = "CPU"
        mock_engine.generate.return_value = "Artificial intelligence is..." * 10
        mock_get_llm_engine.return_value = mock_engine

        benchmark = NPUProxyBenchmark()

        # Run benchmark with minimal iterations
        result = benchmark.run_benchmark(
            device="CPU",
            model="tinyllama",
            iterations=2,
            warmup=0,
        )

        assert result is not None
        assert result.device == "CPU"
        assert result.model == "tinyllama"
        assert result.cold_start_ms > 0
        assert result.warm_inference_ms > 0

    @patch("scripts.benchmark.reset_engine")
    @patch("scripts.benchmark.get_llm_engine")
    def test_benchmark_uses_runtime_model_path_and_device_api(
        self,
        mock_get_llm_engine,
        mock_reset_engine,
    ):
        """Benchmarks should call the runtime with model_path/device and reset the singleton."""
        mock_engine = MagicMock()
        mock_engine.device = "CPU"
        mock_engine.generate.return_value = "test output"
        mock_get_llm_engine.return_value = mock_engine

        benchmark = NPUProxyBenchmark()
        result = benchmark.run_benchmark(device="CPU", model="tinyllama", iterations=1, warmup=0)

        assert result is not None
        call_kwargs = mock_get_llm_engine.call_args.kwargs
        assert call_kwargs["device"] == "CPU"
        assert str(call_kwargs["model_path"]).endswith("tinyllama-1.1b-chat-int4-ov")
        assert mock_engine.generate.call_args.kwargs["max_new_tokens"] == 100
        assert mock_reset_engine.call_count == 2

    @patch("scripts.benchmark.get_llm_engine")
    def test_benchmark_handles_unavailable_device(self, mock_get_llm_engine):
        """Test benchmark gracefully handles unavailable devices."""
        mock_get_llm_engine.return_value = None

        benchmark = NPUProxyBenchmark()

        with pytest.warns(UserWarning):
            result = benchmark.run_benchmark(device="NPU")

        assert result is None
        assert len(benchmark.results) == 0

    @patch("scripts.benchmark.get_llm_engine")
    def test_benchmark_handles_engine_errors(self, mock_get_llm_engine):
        """Test benchmark handles engine initialization errors."""
        mock_get_llm_engine.side_effect = RuntimeError("Device error")

        benchmark = NPUProxyBenchmark()

        with pytest.warns(UserWarning):
            result = benchmark.run_benchmark(device="GPU")

        assert result is None

    @patch("scripts.benchmark.get_llm_engine")
    def test_benchmark_calculates_tokens_per_second(self, mock_get_llm_engine):
        """Test tokens per second calculation."""
        mock_engine = MagicMock()
        mock_engine.device = "CPU"
        # Return output with known word count
        output = " ".join(["word"] * 100)
        mock_engine.generate.return_value = output
        mock_get_llm_engine.return_value = mock_engine

        benchmark = NPUProxyBenchmark()
        result = benchmark.run_benchmark(
            device="CPU", iterations=2, warmup=0
        )

        assert result is not None
        # Should have some tokens per second (words / total_time_in_seconds)
        assert result.tokens_per_second > 0

    @patch("scripts.benchmark.get_llm_engine")
    def test_benchmark_prefers_engine_reported_tokens_per_second(self, mock_get_llm_engine):
        """Benchmark TPS should use engine-reported metrics when the runtime exposes them."""
        mock_engine = MagicMock()
        mock_engine.device = "CPU"
        mock_engine.generate.return_value = "irrelevant output"
        mock_engine.get_engine_info.return_value = {
            "last_generation_stats": {
                "generated_tokens": 24,
                "throughput_tokens_per_second": 42.5,
            }
        }
        mock_get_llm_engine.return_value = mock_engine

        benchmark = NPUProxyBenchmark()
        result = benchmark.run_benchmark(device="CPU", iterations=2, warmup=0)

        assert result is not None
        assert result.tokens_per_second == 42.5

    def test_benchmark_prints_results(self, capsys):
        """Test that benchmark results print correctly."""
        benchmark = NPUProxyBenchmark()
        result1 = BenchmarkResult(
            device="CPU",
            model="tinyllama",
            cold_start_ms=1000.0,
            warm_inference_ms=50.0,
            tokens_per_second=20.0,
        )
        result2 = BenchmarkResult(
            device="NPU",
            model="tinyllama",
            cold_start_ms=500.0,
            warm_inference_ms=25.0,
            tokens_per_second=40.0,
        )
        benchmark.results = [result1, result2]

        benchmark.print_results()

        captured = capsys.readouterr()
        assert "CPU" in captured.out
        assert "NPU" in captured.out
        assert "tinyllama" in captured.out

    def test_benchmark_handles_empty_results(self, capsys):
        """Test benchmark with no results."""
        benchmark = NPUProxyBenchmark()
        benchmark.print_results()

        captured = capsys.readouterr()
        assert "No results" in captured.out

    @patch("scripts.benchmark.get_available_devices")
    @patch("scripts.benchmark.get_llm_engine")
    def test_run_command_without_device_arg(
        self, mock_get_llm_engine, mock_get_available_devices
    ):
        """Test run command benchmarks each enumerated device."""
        cpu_engine = MagicMock()
        cpu_engine.device = "CPU"
        cpu_engine.generate.return_value = "Test output " * 20
        gpu_engine = MagicMock()
        gpu_engine.device = "GPU"
        gpu_engine.generate.return_value = "Test output " * 20
        mock_get_available_devices.return_value = ["CPU", "GPU"]
        mock_get_llm_engine.side_effect = [cpu_engine, gpu_engine]

        with patch("sys.argv", [
            "benchmark.py",
            "run",
            "--model",
            "tinyllama",
            "--iterations",
            "1",
            "--warmup",
            "0",
        ]):
            # Should not raise
            result = main()
            # May return 0 or handle gracefully
            assert result in (0, None)
        assert mock_get_available_devices.called
        assert [call.kwargs["device"] for call in mock_get_llm_engine.call_args_list] == ["CPU", "GPU"]


class TestBenchmarkComparison:
    """Test benchmark comparison functionality."""

    def test_compare_with_baseline(self, capsys):
        """Test comparing results with baseline."""
        benchmark = NPUProxyBenchmark()
        result = BenchmarkResult(
            device="CPU",
            model="tinyllama",
            cold_start_ms=2000.0,
            warm_inference_ms=100.0,
        )
        benchmark.results = [result]

        baseline_data = {
            "timestamp": "2025-01-01T00:00:00Z",
            "system": {"python": "3.11", "platform": "Linux"},
            "results": [
                {
                    "device": "CPU",
                    "model": "tinyllama",
                    "cold_start_ms": 1000.0,
                    "warm_inference_ms": 50.0,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            with open(baseline_path, "w") as f:
                json.dump(baseline_data, f)

            benchmark.compare_results(str(baseline_path))
            captured = capsys.readouterr()

            assert "COMPARISON" in captured.out
            assert "CPU" in captured.out

    def test_compare_uses_current_results_file_when_provided(self, capsys):
        """CLI comparison should read both baseline and current result files."""
        benchmark = NPUProxyBenchmark()
        benchmark.results = [
            BenchmarkResult(
                device="CPU",
                model="tinyllama",
                cold_start_ms=9999.0,
                warm_inference_ms=999.0,
            )
        ]

        baseline_data = {
            "results": [
                {
                    "device": "CPU",
                    "model": "tinyllama",
                    "warm_inference_ms": 50.0,
                }
            ]
        }
        current_data = {
            "results": [
                {
                    "device": "CPU",
                    "model": "tinyllama",
                    "warm_inference_ms": 100.0,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            current_path = Path(tmpdir) / "current.json"
            baseline_path.write_text(json.dumps(baseline_data), encoding="utf-8")
            current_path.write_text(json.dumps(current_data), encoding="utf-8")

            benchmark.compare_results(str(baseline_path), str(current_path))
            captured = capsys.readouterr()

            assert "+50.00 (+100.0%)" in captured.out
            assert "999.00" not in captured.out

    def test_compare_missing_baseline(self, capsys):
        """Test comparison with missing baseline file."""
        benchmark = NPUProxyBenchmark()
        benchmark.compare_results("/nonexistent/baseline.json")

        captured = capsys.readouterr()
        assert "not found" in captured.err


class TestBenchmarkIntegration:
    """Integration tests for benchmark workflow."""

    @patch("scripts.benchmark.get_llm_engine")
    @patch("scripts.benchmark.InferenceEngine.generate")
    def test_full_benchmark_run_with_export(
        self, mock_generate, mock_get_llm_engine
    ):
        """Test full benchmark run from CLI with JSON export."""
        mock_engine = MagicMock()
        mock_engine.device = "CPU"
        mock_get_llm_engine.return_value = mock_engine
        mock_generate.return_value = "Test " * 50

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.json"

            with patch("sys.argv", [
                "benchmark.py",
                "run",
                "--device",
                "CPU",
                "--model",
                "tinyllama",
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--output",
                str(output_file),
            ]):
                result = main()
                assert result in (0, None)

            # Verify output file was created
            assert output_file.exists()

            # Verify JSON is valid
            with open(output_file) as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "results" in data
            assert len(data["results"]) > 0


class TestEmbeddingBenchmark:
    """Test embedding benchmark execution."""

    @patch("scripts.benchmark.get_embedding_engine")
    def test_benchmark_embedding_passes_model_and_device(self, mock_get_embedding_engine):
        """Embedding benchmark should honor the requested model/device arguments."""
        mock_engine = MagicMock()
        mock_engine.embed.return_value = [0.1, 0.2, 0.3]
        mock_engine.get_engine_info.return_value = {"device": "CPU"}
        mock_get_embedding_engine.return_value = mock_engine

        benchmark = NPUProxyBenchmark()
        result = benchmark.benchmark_embedding(model="bge-small", device="CPU")

        assert result is not None
        assert result.model == "bge-small"
        assert result.device == "CPU"
        mock_get_embedding_engine.assert_called_once_with(model_name="bge-small", device="CPU")
