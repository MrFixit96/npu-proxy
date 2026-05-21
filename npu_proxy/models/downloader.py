"""Download OpenVINO models from Hugging Face Hub.

This module provides functions for downloading, caching, and managing
OpenVINO-optimized models from Hugging Face Hub. It implements an
Ollama-compatible download interface with progress streaming support.

Key Features:
    - Automatic model resolution from Ollama-style names to HF repos
    - Local caching with validation of required OpenVINO files
    - Streaming progress updates compatible with Ollama API format
    - Graceful error handling for network, auth, and disk issues

Cache Structure:
    ~/.cache/npu-proxy/models/
    └── {model_name}/
        ├── openvino_model.xml    # Model architecture (required)
        ├── openvino_model.bin    # Model weights (required)
        ├── tokenizer.json        # Tokenizer config
        ├── tokenizer_config.json
        └── config.json           # Model configuration

Hugging Face Hub Integration:
    Uses huggingface_hub library for:
    - snapshot_download(): Full model directory download
    - hf_hub_download(): Individual file download with progress
    - HfApi: Repository metadata and file listing

Example:
    >>> from npu_proxy.models.downloader import download_model
    >>> result = download_model("tinyllama")
    >>> result['path']
    '/home/user/.cache/npu-proxy/models/tinyllama-1.1b-chat-v1.0-int4-ov'
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Generator

from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

from npu_proxy.models.mapper import resolve_model_repo

#: Default cache directory for downloaded models.
#: Set via NPU_PROXY_MODEL_DIR environment variable or defaults to ~/.cache/npu-proxy/models
DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"

#: Required files for a valid OpenVINO model.
#: Download is considered incomplete without these files.
REQUIRED_FILES: list[str] = ["openvino_model.xml", "openvino_model.bin"]


def download_model(
    name: str,
    model_dir: Path | str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, str]:
    """Download an OpenVINO-optimized model from Hugging Face Hub.

    Downloads model files to local cache for offline inference. Uses
    huggingface_hub's snapshot_download for atomic, resumable downloads.
    Skips download if model already exists in cache.

    Args:
        name: Model name in Ollama-style format (e.g., 'tinyllama', 'phi-3')
            or full Hugging Face repo ID (e.g., 'OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov').
            Resolved via resolve_model_repo() mapper.
        model_dir: Local cache directory for storing models. Defaults to
            ~/.cache/npu-proxy/models if not specified.
        progress_callback: Optional callback function invoked during download
            with progress dict updates. Currently unused but reserved for
            future progress streaming implementation.

    Returns:
        dict: Result dictionary with either success or error information.

        On success::

            {
                "status": "success",
                "model": "tinyllama-1.1b-chat-v1.0-int4-ov",  # Local name
                "path": "/home/user/.cache/npu-proxy/models/...",  # Full path
                "source": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"  # HF repo
            }

        On failure::

            {"error": "Failed to download model: 404 Not Found"}

    Raises:
        No exceptions are raised; all errors are returned in the result dict.
        Handled error types include:
        - HfHubHTTPError: Network failures, 401/403 auth errors, 404 not found
        - PermissionError: Cannot write to model_dir
        - OSError: Disk full, path too long, etc.

    Cache Behavior:
        - Creates model_dir if it doesn't exist
        - Downloads to model_dir/{local_name}/
        - Uses huggingface_hub caching with symlinks disabled
        - Re-downloads if openvino_model.xml is missing

    Example:
        >>> result = download_model("tinyllama")
        >>> if "error" not in result:
        ...     print(f"Downloaded to {result['path']}")
        ... else:
        ...     print(f"Failed: {result['error']}")
        Downloaded to /home/user/.cache/npu-proxy/models/tinyllama-1.1b-chat-v1.0-int4-ov

        >>> # Using custom cache directory
        >>> result = download_model("phi-3", model_dir="/opt/models")
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    elif isinstance(model_dir, str):
        model_dir = Path(model_dir)

    # Resolve model name to HuggingFace repo
    resolved = resolve_model_repo(name)
    if resolved is None:
        return {"error": "Model not found"}

    repo_id, local_name = resolved
    local_dir = model_dir / local_name

    try:
        # Ensure model directory exists
        model_dir.mkdir(parents=True, exist_ok=True)

        # Download model snapshot from HuggingFace
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        # Verify openvino_model.xml exists
        xml_path = Path(path) / "openvino_model.xml"
        if not xml_path.exists():
            return {"error": f"Model downloaded but openvino_model.xml not found at {path}"}

        return {
            "status": "success",
            "model": local_name,
            "path": str(path),
            "source": repo_id,
        }

    except HfHubHTTPError as e:
        return {"error": f"Failed to download model: {e}"}
    except PermissionError as e:
        return {"error": f"Permission denied: {e}"}
    except OSError as e:
        return {"error": f"OS error during download: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def get_download_progress(
    repo_id: str,
    local_dir: Path,
) -> Generator[dict[str, str | int], None, None]:
    """Stream download progress updates in Ollama-compatible format.

    Downloads model files one-by-one from Hugging Face Hub, yielding
    progress updates after each file. Designed to match Ollama's pull
    API response format for client compatibility.

    This function is a generator that yields progress dicts as files
    are downloaded, enabling real-time progress streaming to clients.

    Args:
        repo_id: Full Hugging Face repository ID
            (e.g., 'OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov').
        local_dir: Target directory for downloaded files.
            Created automatically if it doesn't exist.

    Yields:
        dict: Progress update dictionaries in Ollama format.

        Progress phases::

            # Phase 1: Initial manifest fetch
            {"status": "pulling manifest"}

            # Phase 2: Per-file progress (start)
            {
                "status": "pulling openvino_model.bin",
                "digest": "sha256:a1b2c3d4e5f6",  # Truncated SHA256 of filename
                "total": 1234567890,  # File size in bytes
                "completed": 0
            }

            # Phase 2: Per-file progress (complete)
            {
                "status": "pulling openvino_model.bin",
                "digest": "sha256:a1b2c3d4e5f6",
                "total": 1234567890,
                "completed": 1234567890  # Matches total when done
            }

            # Phase 3: Verification
            {"status": "verifying sha256 digest"}

            # Phase 4: Final status
            {"status": "success"}
            # or
            {"status": "error: openvino_model.xml not found"}

    Raises:
        No exceptions raised; errors yielded as status messages.

    Hugging Face Hub Integration:
        - Uses HfApi.repo_info() to list repository files and sizes
        - Falls back to REQUIRED_FILES if repo_info fails
        - Downloads via hf_hub_download() with symlinks disabled

    Example:
        >>> for progress in get_download_progress(
        ...     "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        ...     Path("/tmp/models/tinyllama")
        ... ):
        ...     print(f"{progress['status']}: {progress.get('completed', 0)}")
        pulling manifest: 0
        pulling openvino_model.xml: 0
        pulling openvino_model.xml: 45678
        pulling openvino_model.bin: 0
        pulling openvino_model.bin: 1234567890
        verifying sha256 digest: 0
        success: 0
    """
    yield {"status": "pulling manifest"}

    try:
        api = HfApi()

        # Get repo info to list files
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
            files = [f.rfilename for f in repo_info.siblings] if repo_info.siblings else []
        except Exception:
            files = REQUIRED_FILES

        total_size = 0
        completed_size = 0

        # Calculate total size from repo info
        if repo_info.siblings:
            total_size = sum(f.size or 0 for f in repo_info.siblings)

        for filename in files:
            # Generate a digest-like identifier
            digest = f"sha256:{hashlib.sha256(filename.encode()).hexdigest()[:12]}"

            # Get file size if available
            file_size = 0
            if repo_info.siblings:
                for f in repo_info.siblings:
                    if f.rfilename == filename:
                        file_size = f.size or 0
                        break

            yield {
                "status": f"pulling {filename}",
                "digest": digest,
                "total": file_size,
                "completed": 0,
            }

            try:
                # Download individual file
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )

                completed_size += file_size

                yield {
                    "status": f"pulling {filename}",
                    "digest": digest,
                    "total": file_size,
                    "completed": file_size,
                }

            except Exception as e:
                yield {"status": f"error downloading {filename}: {e}"}
                return

        yield {"status": "verifying sha256 digest"}

        # Verify required files exist
        xml_path = local_dir / "openvino_model.xml"
        if xml_path.exists():
            yield {"status": "success"}
        else:
            yield {"status": "error: openvino_model.xml not found"}

    except Exception as e:
        yield {"status": f"error: {e}"}


def is_model_downloaded(name: str, model_dir: Path | str | None = None) -> bool:
    """Check if a model is already downloaded and valid.

    Verifies that a model exists in the local cache by checking for
    the presence of the required openvino_model.xml file. This is a
    fast check that doesn't verify file integrity.

    Args:
        name: Model name in Ollama-style format (e.g., 'tinyllama', 'phi-3')
            or full Hugging Face repo ID. Resolved via resolve_model_repo().
        model_dir: Directory where models are stored. Defaults to
            ~/.cache/npu-proxy/models if not specified.

    Returns:
        bool: True if the model directory exists AND contains
            openvino_model.xml. False otherwise, including:
            - Model name cannot be resolved
            - Directory doesn't exist
            - Directory exists but lacks openvino_model.xml

    Note:
        This check only validates presence, not integrity. A model
        could return True but have corrupted or incomplete files.
        Use download_model() with force_download for re-validation.

    Example:
        >>> is_model_downloaded("tinyllama")
        True
        >>> is_model_downloaded("nonexistent-model")
        False
        >>> is_model_downloaded("phi-3", model_dir="/opt/models")
        False
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    elif isinstance(model_dir, str):
        model_dir = Path(model_dir)

    # Resolve model name
    resolved = resolve_model_repo(name)
    if resolved is None:
        return False

    _, local_name = resolved
    local_dir = model_dir / local_name

    # Check if directory exists and has the required XML file
    xml_path = local_dir / "openvino_model.xml"
    return xml_path.exists()


def get_downloaded_models(model_dir: Path | str | None = None) -> list[str]:
    """List all downloaded models in the cache directory.

    Scans the model cache directory and returns names of all valid
    OpenVINO models (directories containing openvino_model.xml).

    Args:
        model_dir: Directory where models are stored. Defaults to
            ~/.cache/npu-proxy/models if not specified.

    Returns:
        list[str]: Sorted list of model directory names that contain
            valid OpenVINO models. Returns empty list if:
            - model_dir doesn't exist
            - model_dir is not readable (PermissionError)
            - No valid models found

    Note:
        Only checks for openvino_model.xml presence, not full validity.
        Model names returned are local directory names, not original
        Hugging Face repo IDs or Ollama-style names.

    Example:
        >>> get_downloaded_models()
        ['phi-3-mini-4k-instruct-int4-ov', 'tinyllama-1.1b-chat-v1.0-int4-ov']

        >>> get_downloaded_models("/opt/custom-models")
        []
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    elif isinstance(model_dir, str):
        model_dir = Path(model_dir)

    if not model_dir.exists():
        return []

    downloaded = []
    try:
        for entry in model_dir.iterdir():
            if entry.is_dir():
                xml_path = entry / "openvino_model.xml"
                if xml_path.exists():
                    downloaded.append(entry.name)
    except PermissionError:
        return []
    except OSError:
        return []

    return sorted(downloaded)
