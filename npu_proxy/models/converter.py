"""Model conversion utilities for OpenVINO optimization.

This module provides tools for converting HuggingFace models to OpenVINO
Intermediate Representation (IR) format using the optimum-intel library.
It supports both synchronous conversion and streaming progress updates.

The conversion process uses the `optimum-cli` command-line tool, which must
be installed separately via `pip install optimum-intel[openvino]`.

External Tool Integration:
    - optimum-intel: https://huggingface.co/docs/optimum/intel/index
    - Uses `optimum-cli export openvino` for model conversion
    - Supports text-generation and feature-extraction tasks

Conversion Flow:
    1. Resolve model name to HuggingFace repository (via mapper module)
    2. Check if model already exists in cache
    3. Run optimum-cli to export model to OpenVINO format
    4. Verify output files (openvino_model.xml, openvino_model.bin)

OpenVINO Model Structure:
    A valid OpenVINO model directory contains:
    - openvino_model.xml: Model architecture definition
    - openvino_model.bin: Model weights in binary format

Example:
    >>> from npu_proxy.models.converter import auto_download_and_convert
    >>> result = auto_download_and_convert("tinyllama")
    >>> result["status"]
    'success'
    >>> result["path"]
    '/home/user/.cache/npu-proxy/models/tinyllama'

    >>> from npu_proxy.models.converter import is_openvino_model
    >>> from pathlib import Path
    >>> is_openvino_model(Path("/path/to/model"))
    True

Note:
    Most models are pre-converted and hosted on HuggingFace under the
    OpenVINO organization. Direct conversion is rarely needed for end users.
    Use the search module to find pre-converted models first.

Warning:
    Model conversion can be time-consuming (10+ minutes for large models)
    and requires significant disk space for downloading source models.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Generator

from npu_proxy.models.mapper import resolve_model_repo

# Default cache directory for converted models
DEFAULT_CONVERSION_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"

# Required files for a valid OpenVINO model
REQUIRED_OPENVINO_FILES: list[str] = ["openvino_model.xml", "openvino_model.bin"]


def is_openvino_model(path: Path) -> bool:
    """Check if a directory contains a valid OpenVINO model.

    Returns True if openvino_model.xml and openvino_model.bin exist.

    Args:
        path: Directory path to check.

    Returns:
        True if both required OpenVINO files exist, False otherwise.
    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.is_dir():
        return False

    xml_file = path / "openvino_model.xml"
    bin_file = path / "openvino_model.bin"

    return xml_file.exists() and bin_file.exists()


def convert_to_openvino(
    hf_repo: str,
    output_dir: Path,
    task: str = "text-generation",
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Convert a HuggingFace model to OpenVINO format.

    Uses optimum-intel via subprocess to run:
    optimum-cli export openvino --model {hf_repo} --task {task} {output_dir}

    Args:
        hf_repo: HuggingFace model repository ID (e.g., "meta-llama/Llama-2-7b-hf").
        output_dir: Directory where to save the converted model.
        task: Task type ("text-generation" or "feature-extraction").
        progress_callback: Optional callback function for progress updates.

    Returns:
        Dict with status and path on success, or error message on failure.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    # Validate task type
    valid_tasks = ["text-generation", "feature-extraction"]
    if task not in valid_tasks:
        return {"error": f"Invalid task type. Must be one of {valid_tasks}"}

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(f"Starting conversion of {hf_repo}")

        # Build optimum-cli command
        cmd = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            hf_repo,
            "--task",
            task,
            str(output_dir),
        ]

        if progress_callback:
            progress_callback(f"Running: {' '.join(cmd)}")

        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            if progress_callback:
                progress_callback(f"Conversion failed: {error_msg}")
            return {"error": f"Conversion failed: {error_msg}"}

        # Verify output files exist
        if not is_openvino_model(output_dir):
            error_msg = "Conversion completed but output files not found"
            if progress_callback:
                progress_callback(error_msg)
            return {"error": error_msg}

        if progress_callback:
            progress_callback(f"Conversion successful: {output_dir}")

        return {
            "status": "success",
            "path": str(output_dir),
            "model": hf_repo,
        }

    except FileNotFoundError:
        return {
            "error": "optimum-cli not found. Install it with: pip install optimum-intel"
        }
    except subprocess.TimeoutExpired:
        return {"error": "Conversion timed out after 1 hour"}
    except Exception as e:
        return {"error": f"Conversion failed: {e}"}


def auto_download_and_convert(
    model_name: str,
    task: str = "text-generation",
    cache_dir: Path | None = None,
) -> dict:
    """Download and convert a model if needed.

    Flow:
    1. Check if OpenVINO model already in cache -> return path
    2. Resolve model_name to HuggingFace repo
    3. Convert using optimum-cli
    4. Return path to converted model

    Args:
        model_name: Model name (Ollama-style or HuggingFace repo format).
        task: Task type ("text-generation" or "feature-extraction").
        cache_dir: Directory to cache models. Defaults to DEFAULT_CONVERSION_DIR.

    Returns:
        Dict with status and path on success, or error message on failure.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CONVERSION_DIR
    elif isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    # Resolve model name to HuggingFace repo
    resolved = resolve_model_repo(model_name)
    if resolved is None:
        return {"error": f"Model '{model_name}' not found"}

    hf_repo, local_name = resolved
    output_dir = cache_dir / local_name

    # Check if already converted
    if is_openvino_model(output_dir):
        return {
            "status": "success",
            "path": str(output_dir),
            "model": local_name,
            "source": "cache",
        }

    # Convert model
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        result = convert_to_openvino(hf_repo, output_dir, task)

        if "error" in result:
            return result

        return {
            "status": "success",
            "path": result["path"],
            "model": local_name,
            "source": "converted",
        }

    except Exception as e:
        return {"error": f"Failed to convert model: {e}"}


def get_conversion_progress(
    hf_repo: str,
    output_dir: Path,
    task: str = "text-generation",
) -> Generator[dict, None, None]:
    """Yield progress updates during model conversion for streaming.

    Args:
        hf_repo: HuggingFace model repository ID.
        output_dir: Directory where to save the converted model.
        task: Task type ("text-generation" or "feature-extraction").

    Yields:
        Progress dicts with status information.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    # Validate task type
    valid_tasks = ["text-generation", "feature-extraction"]
    if task not in valid_tasks:
        yield {
            "status": "error",
            "message": f"Invalid task type. Must be one of {valid_tasks}",
        }
        return

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        yield {"status": "starting", "message": f"Starting conversion of {hf_repo}"}

        # Build optimum-cli command
        cmd = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            hf_repo,
            "--task",
            task,
            str(output_dir),
        ]

        yield {"status": "running", "message": f"Running: {' '.join(cmd)}"}

        # Run conversion with live output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output
        for line in process.stdout:
            line = line.rstrip("\n")
            if line:
                yield {"status": "converting", "message": line}

        # Wait for completion
        return_code = process.wait(timeout=3600)

        if return_code != 0:
            yield {
                "status": "error",
                "message": "Conversion failed with non-zero exit code",
            }
            return

        # Verify output
        if is_openvino_model(output_dir):
            yield {
                "status": "success",
                "message": f"Conversion complete: {output_dir}",
                "path": str(output_dir),
            }
        else:
            yield {
                "status": "error",
                "message": "Conversion completed but output files not found",
            }

    except FileNotFoundError:
        yield {
            "status": "error",
            "message": "optimum-cli not found. Install it with: pip install optimum-intel",
        }
    except subprocess.TimeoutExpired:
        yield {"status": "error", "message": "Conversion timed out after 1 hour"}
    except Exception as e:
        yield {"status": "error", "message": f"Conversion failed: {e}"}
