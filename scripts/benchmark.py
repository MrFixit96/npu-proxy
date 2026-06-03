#!/usr/bin/env python3
"""
NPU Proxy Benchmark CLI Tool

Benchmarks NPU Proxy inference performance across different devices
(CPU, GPU, NPU) and generates performance reports.
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None



def get_available_devices() -> list[str]:
    from npu_proxy.inference.devices import get_available_devices as impl

    return impl()


def get_llm_engine(*args, **kwargs):
    from npu_proxy.inference.engine import get_llm_engine as impl

    return impl(*args, **kwargs)


def reset_engine() -> None:
    try:
        from npu_proxy.inference.engine import reset_engine as impl
    except Exception:
        return

    impl()


def get_embedding_engine(*args, **kwargs):
    from npu_proxy.inference.embedding_engine import get_embedding_engine as impl

    return impl(*args, **kwargs)


def resolve_model_storage_key(model: str) -> Optional[str]:
    from npu_proxy.models.mapper import resolve_model_storage_key as impl

    return impl(model)


def _default_model_dir() -> Path:
    from npu_proxy.models.registry import DEFAULT_MODEL_DIR

    return DEFAULT_MODEL_DIR


def count_completion_tokens(output: str, *, precision: object = None, model: Optional[str] = None) -> int:
    try:
        from npu_proxy.inference.tokenizer import (
            TokenCountPrecision,
            count_completion_tokens as impl,
        )
    except Exception:
        return len(output.split())

    return impl(
        output,
        precision=precision or TokenCountPrecision.APPROXIMATE,
        model=model,
    )


class TokenCountPrecision:
    APPROXIMATE = "approximate"


class InferenceEngine:
    def generate(self, *args, **kwargs):
        raise NotImplementedError


class ProductionEmbeddingEngine:
    pass


class BenchmarkResult:
    """Represents a single benchmark result."""

    def __init__(
        self,
        device: str,
        model: str,
        cold_start_ms: float,
        warm_inference_ms: float,
        first_token_ms: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
        embedding_ms: Optional[float] = None,
    ):
        self.device = device
        self.model = model
        self.cold_start_ms = cold_start_ms
        self.warm_inference_ms = warm_inference_ms
        self.first_token_ms = first_token_ms
        self.tokens_per_second = tokens_per_second
        self.embedding_ms = embedding_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {
            "device": self.device,
            "model": self.model,
            "cold_start_ms": round(self.cold_start_ms, 2),
            "warm_inference_ms": round(self.warm_inference_ms, 2),
        }
        if self.first_token_ms is not None:
            result["first_token_ms"] = round(self.first_token_ms, 2)
        if self.tokens_per_second is not None:
            result["tokens_per_second"] = round(self.tokens_per_second, 2)
        if self.embedding_ms is not None:
            result["embedding_ms"] = round(self.embedding_ms, 2)
        return result


class NPUProxyBenchmark:
    """Main benchmark runner class."""

    # Test prompts
    INFERENCE_PROMPT = "Tell me about artificial intelligence in 50 words."
    EMBEDDING_TEXT = "The quick brown fox jumps over the lazy dog."

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.failures: List[str] = []

    def run_benchmark(
        self,
        device: Optional[str] = None,
        model: str = "tinyllama",
        iterations: int = 5,
        warmup: int = 1,
    ) -> Optional[BenchmarkResult]:
        """
        Run benchmark on specified device.

        Args:
            device: Target device (CPU, GPU, NPU, or None for auto)
            model: Model name to benchmark
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations

        Returns:
            BenchmarkResult or None if device unavailable
        """
        try:
            print(f"\nBenchmarking {model} on {device or 'auto'}...")

            # Initialize engine
            model_path = self._resolve_model_path(model)
            reset_engine()
            engine = get_llm_engine(model_path=model_path, device=device)
            if engine is None:
                actual_device = device or "available device"
                message = f"Device {actual_device} unavailable, skipping..."
                self.failures.append(f"{model} on {device or 'auto'}: {message}")
                warnings.warn(message)
                return None

            # Determine actual device
            actual_device = self._get_device_name(engine, requested_device=device)

            # Cold start - first inference
            print("  Measuring cold start...")
            start = time.perf_counter()
            self._run_inference(engine, self.INFERENCE_PROMPT)
            cold_start_ms = (time.perf_counter() - start) * 1000

            # Warmup iterations
            print(f"  Running {warmup} warmup iteration(s)...")
            for _ in range(warmup):
                self._run_inference(engine, self.INFERENCE_PROMPT)

            # Warm iterations for averaging
            print(f"  Running {iterations} benchmark iteration(s)...")
            warm_times = []
            token_counts = []
            reported_throughputs = []

            for i in range(iterations):
                start = time.perf_counter()
                output, token_count, reported_throughput = self._run_inference(
                    engine,
                    self.INFERENCE_PROMPT,
                    model=model,
                )
                elapsed = (time.perf_counter() - start) * 1000
                warm_times.append(elapsed)
                token_counts.append(token_count)
                if reported_throughput is not None:
                    reported_throughputs.append(reported_throughput)

            warm_inference_ms = sum(warm_times) / len(warm_times)

            # Calculate tokens per second
            if reported_throughputs:
                tokens_per_second = sum(reported_throughputs) / len(reported_throughputs)
            else:
                total_tokens = sum(token_counts)
                total_time_s = sum(warm_times) / 1000
                tokens_per_second = (
                    total_tokens / total_time_s if total_time_s > 0 else 0
                )

            result = BenchmarkResult(
                device=actual_device,
                model=model,
                cold_start_ms=cold_start_ms,
                warm_inference_ms=warm_inference_ms,
                tokens_per_second=tokens_per_second,
            )

            self.results.append(result)
            print(f"  ✓ Complete")

            return result

        except Exception as e:
            message = f"Benchmark on {device or 'auto'} failed: {str(e)}"
            self.failures.append(f"{model} on {device or 'auto'}: {str(e)}")
            warnings.warn(message)
            return None
        finally:
            reset_engine()

    def benchmark_embedding(
        self, model: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ) -> Optional[BenchmarkResult]:
        """
        Run embedding model benchmark.

        Args:
            model: Embedding model name
            device: Target device

        Returns:
            BenchmarkResult or None if model unavailable
        """
        try:
            print(f"\nBenchmarking embedding model {model}...")

            engine = get_embedding_engine(model_name=model, device=device)
            engine_info = engine.get_engine_info()
            actual_device = (
                engine_info.get("actual_device")
                or engine_info.get("device")
                or device
                or "CPU"
            )

            # Cold start
            print("  Measuring cold start...")
            start = time.perf_counter()
            engine.embed(self.EMBEDDING_TEXT)
            cold_start_ms = (time.perf_counter() - start) * 1000

            # Warmup
            for _ in range(1):
                engine.embed(self.EMBEDDING_TEXT)

            # Benchmark iterations
            print("  Running 5 benchmark iteration(s)...")
            times = []
            for _ in range(5):
                start = time.perf_counter()
                engine.embed(self.EMBEDDING_TEXT)
                times.append((time.perf_counter() - start) * 1000)

            embedding_ms = sum(times) / len(times)

            result = BenchmarkResult(
                device=actual_device,
                model=model,
                cold_start_ms=cold_start_ms,
                warm_inference_ms=embedding_ms,
                embedding_ms=embedding_ms,
            )

            self.results.append(result)
            print(f"  ✓ Complete")

            return result

        except Exception as e:
            message = f"Embedding benchmark failed: {str(e)}"
            self.failures.append(f"embedding {model} on {device or 'auto'}: {str(e)}")
            warnings.warn(message)
            return None

    @staticmethod
    def _run_inference(
        engine: InferenceEngine,
        prompt: str,
        *,
        model: Optional[str] = None,
    ) -> tuple[str, int, Optional[float]]:
        """
        Run single inference and return output with token accounting.

        Returns:
            Tuple of (output_text, token_count, reported_tokens_per_second).

        Notes:
            Prefers runtime-reported generation metrics when available. Falls
            back to tokenizer-backed approximate completion counting instead of
            using a raw word-count heuristic.
        """
        output = str(engine.generate(prompt, max_new_tokens=100))
        stats = NPUProxyBenchmark._get_last_generation_stats(engine)

        generated_tokens = stats.get("generated_tokens")
        token_count = generated_tokens if isinstance(generated_tokens, int) else count_completion_tokens(
            output,
            precision=TokenCountPrecision.APPROXIMATE,
            model=model,
        )

        throughput = stats.get("throughput_tokens_per_second")
        reported_tps = float(throughput) if isinstance(throughput, (int, float)) else None
        return output, token_count, reported_tps

    @staticmethod
    def _get_last_generation_stats(engine: InferenceEngine) -> dict[str, Any]:
        """Return engine-reported generation metrics when available."""
        get_engine_info = getattr(engine, "get_engine_info", None)
        if not callable(get_engine_info):
            return {}

        try:
            info = get_engine_info()
        except Exception:
            return {}

        if not isinstance(info, dict):
            return {}

        stats = info.get("last_generation_stats")
        return stats if isinstance(stats, dict) else {}

    @staticmethod
    def _get_device_name(
        engine: InferenceEngine,
        requested_device: Optional[str] = None,
    ) -> str:
        """Extract device name from engine."""
        for attr in ("actual_device", "device", "requested_device"):
            value = getattr(engine, attr, None)
            if isinstance(value, str) and value:
                return value.upper()

        get_engine_info = getattr(engine, "get_engine_info", None)
        if callable(get_engine_info):
            try:
                info = get_engine_info()
            except Exception:
                info = {}
            if isinstance(info, dict):
                for key in ("actual_device", "device", "requested_device"):
                    value = info.get(key)
                    if isinstance(value, str) and value:
                        return value.upper()

        return (requested_device or "UNKNOWN").upper()

    @staticmethod
    def _resolve_model_path(model: str) -> Path:
        """Resolve a benchmark model argument to the runtime's OpenVINO model path."""
        candidate = Path(model)
        if candidate.exists():
            return candidate

        storage_key = resolve_model_storage_key(model)
        if storage_key is not None:
            return _default_model_dir() / storage_key

        return _default_model_dir() / model

    def print_results(self):
        """Print formatted results table."""
        if not self.results:
            print("No results to display")
            return

        headers = [
            "Device",
            "Model",
            "Cold Start (ms)",
            "Warm Inference (ms)",
            "Tokens/sec",
        ]

        rows = []
        for result in self.results:
            rows.append(
                [
                    result.device,
                    result.model,
                    f"{result.cold_start_ms:.2f}",
                    f"{result.warm_inference_ms:.2f}",
                    f"{result.tokens_per_second:.2f}"
                    if result.tokens_per_second
                    else "N/A",
                ]
            )

        if tabulate:
            print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            # Fallback to simple format
            print("\n" + " | ".join(headers))
            print("-" * 80)
            for row in rows:
                print(" | ".join(str(cell) for cell in row))

    def export_json(self, output_path: str):
        """Export results to JSON file."""
        from datetime import timezone
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "system": self._get_system_info(),
            "results": [result.to_dict() for result in self.results],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results exported to {output_path}")

    @staticmethod
    def _get_system_info() -> Dict[str, str]:
        """Get system information."""
        import platform

        try:
            import openvino
            openvino_version = openvino.__version__
        except ImportError:
            openvino_version = "not installed"

        return {
            "python": platform.python_version(),
            "platform": platform.system(),
            "openvino": openvino_version,
        }

    @staticmethod
    def _load_results_file(path: str) -> dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def compare_results(self, baseline_json: str, current_json: Optional[str] = None) -> bool:
        """
        Compare current results with baseline.

        Args:
            baseline_json: Path to baseline JSON file
            current_json: Optional path to current results JSON file
        """
        try:
            baseline = self._load_results_file(baseline_json)
        except FileNotFoundError:
            print(f"Baseline file not found: {baseline_json}", file=sys.stderr)
            return False

        if current_json is not None:
            try:
                current_results = self._load_results_file(current_json).get("results", [])
            except FileNotFoundError:
                print(f"Current results file not found: {current_json}", file=sys.stderr)
                return False
        else:
            current_results = [result.to_dict() for result in self.results]

        if not current_results:
            print("No current results to compare", file=sys.stderr)
            return False

        print("\n" + "=" * 80)
        print("COMPARISON WITH BASELINE")
        print("=" * 80)

        headers = [
            "Device",
            "Model",
            "Current (ms)",
            "Baseline (ms)",
            "Difference",
        ]
        rows = []

        baseline_results = {
            f"{r['device']}_{r['model']}": r
            for r in baseline.get("results", [])
        }

        for result in current_results:
            key = f"{result['device']}_{result['model']}"
            if key in baseline_results:
                baseline_result = baseline_results[key]
                current = result.get("warm_inference_ms", 0)
                baseline_val = baseline_result.get("warm_inference_ms", 0)
                diff = current - baseline_val
                diff_pct = (
                    (diff / baseline_val * 100)
                    if baseline_val > 0
                    else 0
                )

                rows.append(
                    [
                        result["device"],
                        result["model"],
                        f"{current:.2f}",
                        f"{baseline_val:.2f}",
                        f"{diff:+.2f} ({diff_pct:+.1f}%)",
                    ]
                )

        if not rows:
            print("No matching benchmark rows found to compare", file=sys.stderr)
            return False

        if tabulate:
            print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            print("\n" + " | ".join(headers))
            print("-" * 80)
            for row in rows:
                print(" | ".join(str(cell) for cell in row))
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NPU Proxy Inference Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all available devices
  python benchmark.py run --model tinyllama

  # Benchmark specific device
  python benchmark.py run --device NPU --model tinyllama --iterations 10

  # Export results to JSON
  python benchmark.py run --output results.json

  # Compare with baseline
  python benchmark.py compare --baseline baseline.json current.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark"
    )
    run_parser.add_argument(
        "--device",
        choices=["CPU", "GPU", "NPU"],
        default=None,
        help="Target device (default: all available)",
    )
    run_parser.add_argument(
        "--model",
        default="tinyllama",
        help="Model name (default: tinyllama)",
    )
    run_parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Benchmark iterations (default: 5)",
    )
    run_parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations (default: 1)",
    )
    run_parser.add_argument(
        "--output",
        help="Output JSON file path",
    )

    # Compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="Compare benchmark results"
    )
    compare_parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline JSON file",
    )
    compare_parser.add_argument(
        "current",
        help="Current results JSON file",
    )

    # Export subcommand
    export_parser = subparsers.add_parser(
        "export", help="Export results to JSON"
    )
    export_parser.add_argument(
        "--input",
        required=True,
        help="Input results JSON file",
    )
    export_parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    benchmark = NPUProxyBenchmark()

    if args.command == "run":
        # Determine devices to benchmark
        if args.device:
            devices = [args.device]
        else:
            devices = list(dict.fromkeys(get_available_devices()))
            if not devices:
                warnings.warn("No available devices found for benchmarking.")
                return 1

        for device in devices:
            benchmark.run_benchmark(
                device=device,
                model=args.model,
                iterations=args.iterations,
                warmup=args.warmup,
            )

        # Display results
        benchmark.print_results()

        # Export if requested
        if args.output:
            benchmark.export_json(args.output)

        if benchmark.failures:
            for failure in benchmark.failures:
                print(f"Benchmark failed: {failure}", file=sys.stderr)
            return 1
        if not benchmark.results:
            print("No benchmark results produced", file=sys.stderr)
            return 1

    elif args.command == "compare":
        return 0 if benchmark.compare_results(args.baseline, args.current) else 1

    elif args.command == "export":
        # Load from input and export
        try:
            with open(args.input, "r") as f:
                data = json.load(f)
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Exported to {args.output}")
        except Exception as e:
            print(f"Error exporting: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
