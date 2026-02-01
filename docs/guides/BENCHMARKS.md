# NPU Proxy Benchmark CLI Documentation

## Overview

The NPU Proxy benchmark CLI tool (`scripts/benchmark.py`) provides comprehensive performance measurement capabilities for evaluating model execution across different hardware devices (NPU, CPU, GPU). It enables performance comparison, tracking, and analysis of inference workloads with a focus on cold-start latency, warm inference throughput, and embedding query performance.

### Capabilities

The benchmark tool supports:

- **Cold Start Benchmarks**: Measure model initialization and first inference latency
- **Warm Inference Benchmarks**: Measure sustained throughput in tokens per second
- **Embedding Benchmarks**: Measure embedding generation performance in queries per second
- **Device Comparison**: Run and compare identical workloads across NPU, CPU, and GPU backends
- **Result Export**: Export benchmark data to JSON for further analysis or reporting
- **Result Comparison**: Compare results across different benchmark runs to identify performance changes

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

The benchmark tool requires:
- Python 3.8 or higher
- Model weights and configuration files accessible
- Target devices (NPU, CPU, GPU) properly configured

## Available Commands

### 1. `run` Command

Execute a benchmark against a specified device or model configuration.

#### Syntax

```bash
python scripts/benchmark.py run [OPTIONS]
```

#### Options

- `--device {npu,cpu,gpu}`: Target device for benchmarking (default: npu)
- `--model MODEL_NAME`: Model identifier or path (required)
- `--iterations N`: Number of benchmark iterations (default: 5)
- `--batch-size N`: Batch size for inference (default: 1)
- `--input-length N`: Input sequence length in tokens (default: 512)
- `--output-length N`: Output sequence length in tokens (default: 128)
- `--warmup N`: Number of warmup iterations before measurement (default: 1)
- `--output FILE`: Output file path for results (JSON format)
- `--verbose`: Enable detailed output logging

#### Output

Results are written to a JSON file containing latency, throughput, and device metrics. Standard output provides a summary of benchmark progress and final results.

### 2. `compare` Command

Compare benchmark results from multiple runs to identify performance differences.

#### Syntax

```bash
python scripts/benchmark.py compare [OPTIONS] BASELINE CURRENT
```

#### Parameters

- `BASELINE`: Path to baseline benchmark results (JSON)
- `CURRENT`: Path to current benchmark results (JSON)

#### Options

- `--format {text,json,csv}`: Output format (default: text)
- `--threshold PERCENT`: Percentage threshold for highlighting differences (default: 5)
- `--output FILE`: Write comparison results to file
- `--detailed`: Show detailed metric-by-metric breakdown

#### Output

Comparison shows:
- Percentage differences for key metrics
- Flagged regressions (if current is slower than baseline)
- Improvement highlights (if current is faster than baseline)
- Statistical analysis when available

### 3. `export` Command

Export benchmark results to various formats for reporting and documentation.

#### Syntax

```bash
python scripts/benchmark.py export [OPTIONS] INPUT_FILE
```

#### Parameters

- `INPUT_FILE`: Benchmark results file (JSON)

#### Options

- `--format {json,csv,markdown,html}`: Export format (default: json)
- `--output FILE`: Output file path (required)
- `--include-metadata`: Include system and environment metadata
- `--normalize`: Normalize metrics to reference device
- `--filter METRIC`: Export only specific metrics (repeatable)

#### Output

Exports benchmark data in the specified format suitable for:
- CSV: Spreadsheet and analysis tools
- Markdown: Documentation and reports
- HTML: Web viewing and sharing
- JSON: Programmatic consumption

## Usage Examples

### Example 1: Basic NPU Benchmark

Run a standard benchmark on the NPU device with default settings:

```bash
python scripts/benchmark.py run --device npu --model llama-7b --output results/npu_baseline.json
```

This executes:
- 5 iterations of inference with default settings
- 1 warmup iteration
- Single batch, 512 token input, 128 token output
- Results saved to `results/npu_baseline.json`

### Example 2: Cross-Device Comparison

Run the same benchmark across all three devices:

```bash
python scripts/benchmark.py run --device npu --model llama-7b --output results/npu.json
python scripts/benchmark.py run --device cpu --model llama-7b --output results/cpu.json
python scripts/benchmark.py run --device gpu --model llama-7b --output results/gpu.json
```

Then compare NPU against CPU:

```bash
python scripts/benchmark.py compare results/cpu.json results/npu.json --detailed
```

### Example 3: Warm Inference Throughput

Measure sustained inference throughput with more iterations:

```bash
python scripts/benchmark.py run --device npu --model llama-7b \
  --iterations 20 --warmup 3 --output results/warm_inference.json
```

This provides better statistical significance for throughput measurements by:
- Running 3 warmup iterations for cache warming
- Taking 20 measurement iterations
- Better isolation of steady-state performance

### Example 4: Embedding Benchmarks

Benchmark embedding generation performance:

```bash
python scripts/benchmark.py run --device npu --model bge-small \
  --batch-size 32 --input-length 256 --output-length 1 \
  --iterations 10 --output results/embedding_benchmark.json
```

Parameters optimized for embedding workloads:
- Larger batch size for throughput
- Shorter output (embeddings are usually 1 vector)
- Reasonable input length for text encoding

### Example 5: Export Results for Documentation

Generate a markdown report from benchmark results:

```bash
python scripts/benchmark.py export results/npu_baseline.json \
  --format markdown --output reports/npu_performance.md \
  --include-metadata
```

### Example 6: Batch Performance Analysis

Analyze performance across different batch sizes:

```bash
for batch in 1 2 4 8 16; do
  python scripts/benchmark.py run --device npu --model llama-7b \
    --batch-size $batch --output results/batch_${batch}.json
done

# Compare results across batch sizes
python scripts/benchmark.py export results/batch_1.json --format csv --output analysis/batch_comparison.csv
```

## Output Format

### JSON Schema

The benchmark tool outputs results in the following JSON structure:

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:45Z",
    "device": "npu",
    "device_info": {
      "name": "Qualcomm Hexagon NPU",
      "compute_capability": "8.1"
    },
    "model": {
      "name": "llama-7b",
      "parameters": 7000000000,
      "quantization": "int8"
    },
    "benchmark_config": {
      "iterations": 5,
      "batch_size": 1,
      "input_length": 512,
      "output_length": 128,
      "warmup_iterations": 1
    },
    "system_info": {
      "os": "Linux",
      "cpu": "ARM Cortex-X1",
      "memory_gb": 8,
      "npu_memory_gb": 4
    }
  },
  "results": {
    "latency": {
      "cold_start_ms": {
        "mean": 245.3,
        "std": 12.5,
        "min": 232.1,
        "max": 265.8,
        "median": 243.2
      },
      "first_token_ms": {
        "mean": 125.4,
        "std": 8.3,
        "min": 115.2,
        "max": 140.6,
        "median": 123.1
      },
      "subsequent_token_ms": {
        "mean": 18.5,
        "std": 1.2,
        "min": 16.8,
        "max": 21.3,
        "median": 18.2
      }
    },
    "throughput": {
      "tokens_per_second": {
        "mean": 54.1,
        "std": 2.3,
        "min": 50.8,
        "max": 57.2,
        "median": 54.8
      },
      "tokens_per_second_after_first": {
        "mean": 53.9,
        "std": 2.1,
        "min": 50.5,
        "max": 57.0,
        "median": 54.7
      }
    },
    "memory": {
      "peak_memory_usage_mb": {
        "mean": 1845.3,
        "std": 23.1,
        "min": 1820.1,
        "max": 1875.4,
        "median": 1850.2
      },
      "average_memory_usage_mb": {
        "mean": 1654.2,
        "std": 18.5,
        "min": 1635.0,
        "max": 1680.3,
        "median": 1658.1
      }
    },
    "power": {
      "average_power_w": {
        "mean": 4.2,
        "std": 0.3,
        "min": 3.8,
        "max": 4.8,
        "median": 4.1
      },
      "peak_power_w": {
        "mean": 8.5,
        "std": 0.5,
        "min": 7.8,
        "max": 9.2,
        "median": 8.4
      }
    }
  },
  "iterations": [
    {
      "iteration": 1,
      "cold_start_ms": 247.2,
      "first_token_ms": 126.1,
      "subsequent_token_ms": 18.3,
      "tokens_per_second": 54.6,
      "peak_memory_mb": 1850.5,
      "average_memory_mb": 1660.2,
      "average_power_w": 4.1,
      "peak_power_w": 8.3
    }
  ]
}
```

### Key Metrics Explained

**Latency Metrics**:
- `cold_start_ms`: Time from process start to first token output, including model loading
- `first_token_ms`: Time to generate the first output token (does not include cold start)
- `subsequent_token_ms`: Average time to generate tokens after the first

**Throughput Metrics**:
- `tokens_per_second`: Overall throughput including first token latency
- `tokens_per_second_after_first`: Throughput for tokens after the first (steady-state)

**Memory Metrics**:
- `peak_memory_usage_mb`: Maximum memory consumed during benchmark
- `average_memory_usage_mb`: Average memory usage across iterations

**Power Metrics**:
- `average_power_w`: Average power consumption during inference
- `peak_power_w`: Maximum power consumption during benchmark

## Methodology

### What is Measured

The benchmark tool measures the following aspects of model execution:

#### 1. Cold Start Performance

Cold start benchmarks measure the complete initialization and first inference cycle:

- **Model Loading**: Time to deserialize model weights and initialize weights on device
- **First Inference**: Time to execute first forward pass and obtain first output token
- **Total Cold Start**: Sum of loading and first inference time

This metric is critical for user-facing applications where latency from application startup matters.

#### 2. Warm Inference Performance

After initial model loading and cache warming, warm inference measures sustained performance:

- **Steady-State Throughput**: Tokens generated per second after model is loaded and caches are warm
- **Per-Token Latency**: Time required to generate each token in steady state
- **Batch Processing**: How throughput changes with batch size

This metric represents typical production performance once models are deployed.

#### 3. Device Overhead

Measurements capture device-specific overhead:

- **Device Transfer Overhead**: Time to move data to/from device
- **Device Context Switching**: Time for device management operations
- **Device Synchronization**: Synchronization points between CPU and device

#### 4. Resource Utilization

Resource usage is monitored throughout execution:

- **Memory Consumption**: Peak and average memory usage on device
- **Power Consumption**: Average and peak power draw (when available)
- **Device Utilization**: Percentage of device resources used

### Warmup and Stability

The benchmark employs several techniques to ensure stable, representative measurements:

1. **Warmup Iterations**: Initial iterations are discarded to allow:
   - Cache population
   - Device frequency scaling stabilization
   - Memory allocation patterns to settle

2. **Repeated Measurements**: Multiple iterations are collected to calculate:
   - Mean performance (average)
   - Standard deviation (variability)
   - Min/max (range)
   - Median (robust center)

3. **Statistical Analysis**: Results include standard deviation to quantify variability and confidence in measurements

### Device-Specific Considerations

#### NPU Benchmarking

- Warmup iterations are essential for NPU frequency scaling
- Memory allocation may differ from CPU/GPU patterns
- Quantization effects on accuracy are noted separately

#### CPU Benchmarking

- CPU frequency scaling and thermal throttling can affect results
- Multi-threaded execution may vary
- Results often show higher variability than NPU/GPU

#### GPU Benchmarking

- CUDA/ROCm graph recording can affect first iteration
- Memory transfer overhead is more significant
- Batch size has larger impact on throughput

## Example Baseline Results

### LLaMA 7B Model Across Devices

Baseline measurements for LLaMA 7B model (int8 quantization) on reference hardware:

```
Device Configuration: LLaMA 7B (int8)
Input: 512 tokens, Output: 128 tokens
Batch Size: 1, Iterations: 5, Warmup: 1

NPU (Qualcomm Hexagon):
  Cold Start:         245.3 ms
  First Token:        125.4 ms
  Subsequent Token:   18.5 ms
  Throughput:         54.1 tokens/sec
  Peak Memory:        1845.3 MB
  Avg Power:          4.2 W

CPU (ARM Cortex-X1):
  Cold Start:         1850.2 ms
  First Token:        945.1 ms
  Subsequent Token:   187.2 ms
  Throughput:         5.3 tokens/sec
  Peak Memory:        2450.5 MB
  Avg Power:          8.5 W

GPU (Adreno 8cx):
  Cold Start:         512.3 ms
  First Token:        287.4 ms
  Subsequent Token:   42.3 ms
  Throughput:         23.6 tokens/sec
  Peak Memory:        3200.1 MB
  Avg Power:          6.8 W
```

### Performance Ratios (vs CPU)

```
Metric                  NPU         GPU
Cold Start Speedup      7.5x        3.6x
First Token Speedup     7.5x        3.3x
Throughput Speedup      10.2x       4.5x
Power Efficiency        2.0x        1.3x
Memory per Token        0.82x       1.24x
```

### Scaling with Batch Size (NPU LLaMA 7B)

```
Batch Size    Throughput      Power       Memory
1             54.1 tok/sec    4.2 W       1.8 GB
2             105.3 tok/sec   7.1 W       2.1 GB
4             187.5 tok/sec   11.3 W      2.8 GB
8             285.2 tok/sec   14.8 W      4.2 GB
16            312.1 tok/sec   15.2 W      5.9 GB
```

## Best Practices for Running Benchmarks

### Before Benchmarking

1. **System Preparation**
   - Close unnecessary applications to reduce system noise
   - Disable background processes and services
   - Ensure stable power supply (plug in device for mobile)
   - Allow device to reach thermal equilibrium (5-10 minute idle)

2. **Environment Setup**
   - Use a quiet environment for power measurements
   - Ensure ambient temperature is stable
   - Document hardware configuration and driver versions
   - Verify model weights are properly loaded

3. **Baseline Establishment**
   - Create baseline benchmarks under controlled conditions
   - Document all configuration details
   - Run on known, stable hardware
   - Keep baseline data for future comparisons

### During Benchmarking

4. **Run Configuration**
   - Use adequate warmup iterations (minimum 1-3)
   - Run sufficient iterations for statistical significance (5-10 minimum)
   - Keep batch size appropriate for the use case
   - Avoid changing configuration mid-benchmark series

5. **Monitoring**
   - Monitor for thermal throttling (check peak power output)
   - Watch for memory pressure (OOM errors)
   - Check for system interference (background processes)
   - Log any anomalies or interruptions

6. **Multiple Runs**
   - Run benchmarks multiple times (at least 3 separate runs)
   - Verify consistency across runs
   - Report average and standard deviation
   - Investigate outliers

### After Benchmarking

7. **Result Analysis**
   - Verify measurements are physically reasonable
   - Check for outliers (>3 standard deviations)
   - Review device resource utilization
   - Compare against known baselines

8. **Documentation**
   - Document exact configuration and parameters
   - Include timestamp and hardware information
   - Note any system conditions or anomalies
   - Include raw iteration data for reproducibility

9. **Comparison and Reporting**
   - Always compare against same configuration
   - Report changes as percentages, not absolute times
   - Include confidence intervals (standard deviation)
   - Highlight regressions (>5% degradation)
   - Acknowledge limitations and sources of variance

### Common Pitfalls to Avoid

- **Insufficient Warmup**: Results will show inflated latencies for first device operations
- **Too Few Iterations**: High variance obscures true performance
- **Thermal Throttling**: Hot device performs worse; allow cooldown between runs
- **System Noise**: Background processes cause variability; clean system required
- **Incorrect Input Sizes**: Changing sequence length changes inference time; keep consistent
- **Single-Run Comparison**: One run is not statistically valid; require multiple runs
- **Mixed Configurations**: Never mix batch sizes or input lengths in same benchmark series
- **Ignoring Memory Pressure**: Slow memory access dominates when model doesn't fit in fast memory

### Performance Optimization Workflow

Use benchmarking to guide optimization:

1. Establish baseline with current configuration
2. Change one variable at a time (batch size, quantization, etc.)
3. Benchmark after each change
4. Keep changes that improve performance
5. Revert changes that hurt performance or stability
6. Document final configuration and results

## Troubleshooting

### High Variance in Results

**Cause**: System noise, thermal throttling, or insufficient warmup

**Solutions**:
- Increase warmup iterations
- Close background applications
- Ensure stable thermal conditions
- Disable dynamic frequency scaling if possible
- Increase number of measurement iterations

### Out of Memory Errors

**Cause**: Batch size or model size exceeds available device memory

**Solutions**:
- Reduce batch size
- Use quantization (int8 or int4)
- Reduce input sequence length
- Check available device memory

### Inconsistent Results Across Devices

**Cause**: Different device drivers, CUDA/ROCm versions, or hardware variations

**Solutions**:
- Update device drivers
- Use identical software versions across devices
- Account for hardware differences in documentation
- Run on identical hardware when possible

### Power Measurements Unavailable

**Cause**: Device or kernel does not support power monitoring

**Solutions**:
- Use alternative power measurement tools
- Document that power measurements unavailable
- Check device firmware/driver support
- Substitute other metrics (thermal output if available)

## Integration with CI/CD

To integrate benchmarking into continuous integration:

```bash
# Run benchmarks on each commit
if git diff HEAD~1 HEAD --name-only | grep -q "model\|inference"; then
  python scripts/benchmark.py run --device npu --model llama-7b \
    --output results/current_build.json
  
  # Compare against baseline
  python scripts/benchmark.py compare results/baseline.json \
    results/current_build.json --threshold 5
  
  # Export results
  python scripts/benchmark.py export results/current_build.json \
    --format markdown --output reports/build_benchmarks.md
fi
```

See CI/CD configuration documentation for specific implementation details.
