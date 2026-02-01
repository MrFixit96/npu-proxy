# Converter Module Summary

## Overview
Created a complete model conversion module (`npu_proxy/models/converter.py`) for converting HuggingFace models to OpenVINO format, along with comprehensive test coverage.

## Files Created

### 1. `npu_proxy/models/converter.py` (8,810 bytes)
Complete module for converting HuggingFace models to OpenVINO format.

#### Functions Implemented:

##### `is_openvino_model(path: Path) -> bool`
- Validates if a directory contains a valid OpenVINO model
- Checks for both `openvino_model.xml` and `openvino_model.bin` files
- Accepts both `Path` and string paths
- Returns `True` only if both required files exist

##### `convert_to_openvino(hf_repo: str, output_dir: Path, task: str = "text-generation", progress_callback: Callable[[str], None] | None = None) -> dict`
- Converts HuggingFace models to OpenVINO format using `optimum-cli`
- Uses `subprocess.run()` to execute: `optimum-cli export openvino --model {hf_repo} --task {task} {output_dir}`
- Supports task types:
  - `"text-generation"` (LLMs)
  - `"feature-extraction"` (embedding models)
- Optional progress callback for real-time updates
- Returns dict with:
  - Success: `{"status": "success", "path": "...", "model": "..."}`
  - Error: `{"error": "error message"}`
- Comprehensive error handling:
  - Missing `optimum-cli` (helpful installation message)
  - Conversion failures
  - Timeout after 1 hour
  - Missing output files validation

##### `auto_download_and_convert(model_name: str, task: str = "text-generation", cache_dir: Path | None = None) -> dict`
- Smart caching: Checks if OpenVINO model already exists in cache
- Returns cached model immediately if available (skips conversion)
- Resolves Ollama-style or HuggingFace repo names using `resolve_model_repo()`
- Converts model if not in cache
- Returns dict with:
  - `status`: "success"
  - `path`: Path to converted/cached model
  - `model`: Local model name
  - `source`: "cache" (if cached) or "converted" (if newly converted)
- Defaults to `~/.cache/npu-proxy/models/` for caching

##### `get_conversion_progress(hf_repo: str, output_dir: Path, task: str = "text-generation") -> Generator[dict, None, None]`
- Generator yielding progress updates during conversion for streaming
- Live stdout streaming from `optimum-cli`
- Yields dicts with:
  - `status`: "starting", "running", "converting", "success", or "error"
  - `message`: Progress or error message
  - `path`: (optional) Path to converted model on success
- Comprehensive error handling:
  - Missing `optimum-cli`
  - Timeout handling
  - Process failures

## Constants
- `DEFAULT_CONVERSION_DIR`: `~/.cache/npu-proxy/models/`
- `REQUIRED_OPENVINO_FILES`: `["openvino_model.xml", "openvino_model.bin"]`

## Code Style & Integration
- Follows existing codebase patterns from `downloader.py`
- Uses `from __future__ import annotations` for Python 3.9+ compatibility
- Consistent with type hints, docstrings, and error handling style
- Integrates with `npu_proxy.models.mapper.resolve_model_repo()`

---

### 2. `tests/test_converter.py` (16,697 bytes)
Comprehensive test suite with 29 tests covering all functions.

#### Test Classes:

##### `TestIsOpenVINOModel` (6 tests)
- `test_is_openvino_model_with_valid_model`: Valid model detection ✓
- `test_is_openvino_model_missing_xml`: Missing .xml file ✓
- `test_is_openvino_model_missing_bin`: Missing .bin file ✓
- `test_is_openvino_model_nonexistent_directory`: Non-existent path ✓
- `test_is_openvino_model_with_string_path`: String path handling ✓
- `test_is_openvino_model_with_file_path`: File vs directory check ✓

##### `TestConvertToOpenVINO` (11 tests)
- Invalid task validation ✓
- Missing `optimum-cli` error handling ✓
- Conversion failures ✓
- Successful conversion ✓
- Return dict structure validation ✓
- Progress callback invocation ✓
- Timeout handling ✓
- Missing output files detection ✓
- String path handling ✓
- Feature extraction task support ✓
- Task type validation in subprocess call ✓

##### `TestAutoDownloadAndConvert` (6 tests)
- Unknown model error handling ✓
- **Cache skipping (key requirement)**: Existing models not re-converted ✓
- Conversion when not in cache ✓
- Conversion error propagation ✓
- String cache_dir handling ✓
- Feature extraction task passthrough ✓

##### `TestGetConversionProgress` (6 tests)
- Invalid task error handling ✓
- Missing `optimum-cli` error ✓
- Progress update streaming ✓
- Non-zero exit code handling ✓
- Timeout handling ✓
- String output_dir handling ✓

##### `TestConversionIntegration` (1 test)
- Full conversion flow with caching ✓

## Test Results
```
29 passed in 0.57s
```

### Test Coverage Highlights:
- ✓ All required functions implemented and tested
- ✓ Error handling: missing tools, network errors, timeouts
- ✓ Both text-generation and feature-extraction tasks
- ✓ Caching logic (models not re-converted if already cached)
- ✓ Progress callbacks and generators for streaming
- ✓ Type flexibility (string vs Path, with/without kwargs)
- ✓ Mock-based testing (no external dependencies needed)

## Key Implementation Details

### Error Handling
1. **Missing optimum-cli**: Returns helpful error with installation instructions
2. **Invalid tasks**: Validates against allowed task types
3. **Conversion failures**: Captures and returns subprocess stderr
4. **Timeout**: 1-hour limit prevents infinite conversion attempts
5. **Network/Permission errors**: Caught and returned as error dicts

### Subprocess Integration
- Uses `subprocess.run()` for blocking conversion (with 1-hour timeout)
- Uses `subprocess.Popen()` with pipe output for streaming progress
- Properly handles both stdout and stderr
- Non-zero exit codes trigger error reporting

### Caching Strategy
- Automatic detection of existing OpenVINO models
- Prevents redundant conversions
- Uses consistent naming from `resolve_model_repo()`
- Returns source indicator ("cache" vs "converted")

### Type Flexibility
All path parameters accept:
- `pathlib.Path` objects
- String paths (automatically converted to `Path`)

## Usage Examples

```python
from pathlib import Path
from npu_proxy.models.converter import (
    is_openvino_model,
    auto_download_and_convert,
    get_conversion_progress,
)

# Check if a model is already OpenVINO format
if is_openvino_model("/path/to/model"):
    print("Ready to use!")

# Auto-convert (skips if already cached)
result = auto_download_and_convert("tinyllama")
if "error" not in result:
    print(f"Model at: {result['path']}")
    print(f"From: {result['source']}")  # "cache" or "converted"

# Stream conversion progress
for progress in get_conversion_progress("gpt2", Path("/models/gpt2")):
    print(f"{progress['status']}: {progress['message']}")
```

## Integration with Existing Code
- Uses `resolve_model_repo()` from `npu_proxy.models.mapper` for model name resolution
- Follows same patterns as `download_model()` from `npu_proxy.models.downloader`
- Returns dicts compatible with existing error handling patterns
- Uses same cache directory structure conventions

## Dependencies
- `pathlib` (standard library)
- `subprocess` (standard library)
- `optimum-intel` (external, for actual conversions)
- `npu_proxy.models.mapper.resolve_model_repo`

---

## Summary
✓ Module fully implemented with 4 main functions
✓ All 29 tests passing
✓ Comprehensive error handling
✓ Smart caching to avoid redundant conversions
✓ Progress streaming support
✓ Both text-generation and embedding models supported
✓ Production-ready with proper documentation and type hints
