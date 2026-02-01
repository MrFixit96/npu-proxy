# Converter Module - API Reference

## Module: `npu_proxy.models.converter`

Quick conversion and validation of HuggingFace models to OpenVINO format with smart caching.

---

## Functions

### `is_openvino_model(path: Path) -> bool`

**Check if a directory contains a valid OpenVINO model.**

Validates that a directory contains both required OpenVINO files.

#### Parameters
- `path` (Path | str): Directory path to validate
  - Can be `pathlib.Path` or string path
  - Non-existent paths return `False`
  - File paths return `False` (directory check fails)

#### Returns
- `bool`: `True` if both `openvino_model.xml` and `openvino_model.bin` exist, `False` otherwise

#### Examples
```python
from pathlib import Path
from npu_proxy.models.converter import is_openvino_model

# Check a model directory
if is_openvino_model("/models/gpt2"):
    print("Ready for inference!")

# String paths work too
is_openvino_model("/models/gpt2")  # True or False

# Non-existent paths
is_openvino_model("/does/not/exist")  # False
```

---

### `convert_to_openvino(hf_repo: str, output_dir: Path, task: str = "text-generation", progress_callback: Callable[[str], None] | None = None) -> dict`

**Convert a HuggingFace model to OpenVINO format.**

Uses `optimum-cli` to convert models. Validates output and returns status.

#### Parameters
- `hf_repo` (str): HuggingFace model repository ID
  - Examples: `"gpt2"`, `"meta-llama/Llama-2-7b"`, `"sentence-transformers/all-MiniLM-L6-v2"`
- `output_dir` (Path | str): Directory where converted model will be saved
  - Directory is created if it doesn't exist
  - Must be writable
- `task` (str, optional): Task type for conversion (default: `"text-generation"`)
  - `"text-generation"` for LLM models
  - `"feature-extraction"` for embedding models
- `progress_callback` (Callable, optional): Function called with progress messages
  - Called with string messages during conversion
  - Useful for logging or UI updates

#### Returns
- **Success (dict):**
  ```python
  {
      "status": "success",
      "path": "/path/to/converted/model",
      "model": "gpt2"
  }
  ```
- **Error (dict):**
  ```python
  {
      "error": "Error message describing what went wrong"
  }
  ```

#### Possible Errors
- `"optimum-cli not found..."` - Install with `pip install optimum-intel`
- `"Invalid task type..."` - Use only "text-generation" or "feature-extraction"
- `"Conversion failed: ..."` - Model not found, network error, etc.
- `"Conversion timed out..."` - Took longer than 1 hour
- `"output files not found"` - Conversion succeeded but files missing

#### Examples
```python
from pathlib import Path
from npu_proxy.models.converter import convert_to_openvino

# Basic conversion
result = convert_to_openvino("gpt2", Path("/models/gpt2"))
if "error" not in result:
    print(f"Converted to: {result['path']}")

# With progress callback
def log_progress(msg):
    print(f"[Conversion] {msg}")

result = convert_to_openvino(
    "meta-llama/Llama-2-7b",
    Path("/models/llama2"),
    progress_callback=log_progress
)

# Feature extraction task
result = convert_to_openvino(
    "sentence-transformers/all-MiniLM-L6-v2",
    Path("/models/embedding"),
    task="feature-extraction"
)

# Using string path
result = convert_to_openvino("gpt2", "/models/gpt2")
```

---

### `auto_download_and_convert(model_name: str, task: str = "text-generation", cache_dir: Path | None = None) -> dict`

**Download and convert a model if needed, with smart caching.**

Checks cache first, skips conversion if already cached, converts if needed.

#### Parameters
- `model_name` (str): Model name or HuggingFace repo ID
  - Ollama-style names: `"tinyllama"`, `"llama2"`
  - HuggingFace repos: `"gpt2"`, `"meta-llama/Llama-2-7b"`
  - Names are resolved via `npu_proxy.models.mapper.resolve_model_repo()`
- `task` (str, optional): Task type (default: `"text-generation"`)
  - `"text-generation"` for LLM models
  - `"feature-extraction"` for embedding models
- `cache_dir` (Path | str, optional): Directory for cached models
  - Defaults to `~/.cache/npu-proxy/models/`
  - Allows custom caching location

#### Returns
- **Success (dict):**
  ```python
  {
      "status": "success",
      "path": "/path/to/model",
      "model": "tinyllama",
      "source": "cache"  # or "converted"
  }
  ```
- **Error (dict):**
  ```python
  {
      "error": "Error message"
  }
  ```

#### Key Features
- **Caching**: If model is already in cache, returns immediately without conversion
- **Source tracking**: Returns `"source"` to indicate if from cache or newly converted
- **Automatic resolution**: Converts Ollama names to HuggingFace repos

#### Examples
```python
from pathlib import Path
from npu_proxy.models.converter import auto_download_and_convert

# Auto-convert with default cache
result = auto_download_and_convert("tinyllama")
if "error" not in result:
    print(f"Model path: {result['path']}")
    print(f"From: {result['source']}")  # "cache" or "converted"

# Custom cache directory
result = auto_download_and_convert(
    "tinyllama",
    cache_dir="/custom/cache"
)

# Feature extraction model
result = auto_download_and_convert(
    "sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

# Caching demo
result1 = auto_download_and_convert("gpt2")  # Converts (slow)
result2 = auto_download_and_convert("gpt2")  # Uses cache (instant)
assert result1['source'] == "converted"
assert result2['source'] == "cache"
```

---

### `get_conversion_progress(hf_repo: str, output_dir: Path, task: str = "text-generation") -> Generator[dict, None, None]`

**Yield progress updates during model conversion for streaming.**

Generator that yields progress updates as the conversion runs.

#### Parameters
- `hf_repo` (str): HuggingFace model repository ID
- `output_dir` (Path | str): Directory where converted model will be saved
- `task` (str, optional): Task type (default: `"text-generation"`)
  - `"text-generation"` for LLM models
  - `"feature-extraction"` for embedding models

#### Yields
Progress dictionaries with these fields:
- `"status"` (str): Current status
  - `"starting"` - Conversion initialization
  - `"running"` - Subprocess started
  - `"converting"` - Model conversion in progress
  - `"success"` - Conversion completed
  - `"error"` - An error occurred
- `"message"` (str): Status message or output line
- `"path"` (str, optional): Path to converted model (only on success)

#### Examples
```python
from pathlib import Path
from npu_proxy.models.converter import get_conversion_progress

# Stream conversion progress
for progress in get_conversion_progress("gpt2", Path("/models/gpt2")):
    print(f"{progress['status']}: {progress['message']}")
    if progress['status'] == 'success':
        print(f"Converted to: {progress['path']}")

# For UI updates
for progress in get_conversion_progress("llama2", "/models/llama2"):
    if progress['status'] == 'error':
        display_error(progress['message'])
    else:
        update_progress_bar(progress['message'])

# Feature extraction conversion with progress
for p in get_conversion_progress(
    "sentence-transformers/all-MiniLM-L6-v2",
    "/models/embedding",
    task="feature-extraction"
):
    if p['status'] in ['converting']:
        # Log conversion progress
        logger.info(p['message'])
```

---

## Constants

### `DEFAULT_CONVERSION_DIR`
```python
Path.home() / ".cache" / "npu-proxy" / "models"
```
Default cache directory for converted models. Typically: `~/.cache/npu-proxy/models/`

### `REQUIRED_OPENVINO_FILES`
```python
["openvino_model.xml", "openvino_model.bin"]
```
Files required for a valid OpenVINO model.

---

## Error Handling

### Common Error Scenarios

#### Missing optimum-cli
```python
result = convert_to_openvino("gpt2", "/models/gpt2")
# Returns: {"error": "optimum-cli not found. Install it with: pip install optimum-intel"}
```
**Solution:** `pip install optimum-intel`

#### Invalid Task Type
```python
result = convert_to_openvino("gpt2", "/models/gpt2", task="invalid-task")
# Returns: {"error": "Invalid task type. Must be one of ['text-generation', 'feature-extraction']"}
```
**Solution:** Use only `"text-generation"` or `"feature-extraction"`

#### Model Not Found
```python
result = convert_to_openvino("nonexistent-model-xyz", "/models/xyz")
# Returns: {"error": "Conversion failed: ...HF model not found..."}
```
**Solution:** Verify the model name/repo exists on HuggingFace

#### Timeout
```python
result = convert_to_openvino("large-model", "/models/large")
# If conversion takes >1 hour: {"error": "Conversion timed out after 1 hour"}
```
**Solution:** Models requiring >1 hour may need custom infrastructure

#### Unknown Model Name
```python
result = auto_download_and_convert("unknown-xyz")
# Returns: {"error": "Model 'unknown-xyz' not found"}
```
**Solution:** Use HuggingFace repo ID or Ollama-registered name

---

## Return Value Examples

### Successful is_openvino_model
```python
is_openvino_model("/models/gpt2")  # True
```

### Successful convert_to_openvino
```python
{
    "status": "success",
    "path": "/home/user/.cache/npu-proxy/models/gpt2",
    "model": "gpt2"
}
```

### Successful auto_download_and_convert (from cache)
```python
{
    "status": "success",
    "path": "/home/user/.cache/npu-proxy/models/tinyllama",
    "model": "tinyllama",
    "source": "cache"
}
```

### Successful auto_download_and_convert (newly converted)
```python
{
    "status": "success",
    "path": "/home/user/.cache/npu-proxy/models/tinyllama",
    "model": "tinyllama",
    "source": "converted"
}
```

### get_conversion_progress yields
```python
{"status": "starting", "message": "Starting conversion of gpt2"}
{"status": "running", "message": "Running: optimum-cli export openvino ..."}
{"status": "converting", "message": "Exporting GPT2 model to OpenVINO format..."}
{"status": "converting", "message": "Converting encoder model..."}
{"status": "success", "message": "Conversion complete: /models/gpt2", "path": "/models/gpt2"}
```

---

## Integration with Existing Code

### With npu_proxy.models.mapper
```python
from npu_proxy.models.mapper import resolve_model_repo
from npu_proxy.models.converter import auto_download_and_convert

# auto_download_and_convert uses resolve_model_repo internally
result = auto_download_and_convert("tinyllama")
# Internally resolves "tinyllama" to HuggingFace repo ID
```

### With Error Patterns
```python
# Consistent error handling with rest of codebase
result = auto_download_and_convert("gpt2")
if "error" in result:
    logger.error(result["error"])
    return None
else:
    return result["path"]
```

---

## Performance Notes

- **Caching**: Second call to same model returns instantly (from cache)
- **Timeout**: 1-hour limit prevents long-running conversions
- **Streaming**: `get_conversion_progress()` yields live output (suitable for UI)
- **Validation**: Models validated after conversion before returning

---

## Dependencies

- **Required**: Python 3.9+
- **For actual conversion**: `optimum-intel` package
- **Internal**: `npu_proxy.models.mapper.resolve_model_repo()`

---

## See Also

- `npu_proxy.models.downloader` - For downloading pre-converted models
- `npu_proxy.models.mapper` - For model name resolution
- HuggingFace documentation: https://huggingface.co/docs
- Optimum documentation: https://huggingface.co/docs/optimum
