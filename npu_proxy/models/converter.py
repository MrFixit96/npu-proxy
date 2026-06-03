"""Model conversion utilities for OpenVINO optimization."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Generator

from npu_proxy.models.mapper import (
    resolve_model_repo,
    resolve_model_storage_key,
    resolve_runtime_model_name,
)

logger = logging.getLogger(__name__)
DEFAULT_CONVERSION_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"
REQUIRED_OPENVINO_FILES: list[str] = ["openvino_model.xml", "openvino_model.bin"]
VALID_EXPORT_TASKS: tuple[str, ...] = (
    "text-generation-with-past",
    "text-generation",
    "feature-extraction",
)
CONVERSION_TIMEOUT_SECONDS = 3600


def _is_regular_nonempty_file(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        return path.stat().st_size > 0
    except OSError:
        return False


def is_openvino_model(path: Path) -> bool:
    """Check if a directory contains a valid OpenVINO model."""
    if not isinstance(path, Path):
        path = Path(path)
    return path.is_dir() and all(_is_regular_nonempty_file(path / name) for name in REQUIRED_OPENVINO_FILES)


def _temp_output_dir(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.partial"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _commit_output(temp_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temp_dir.rename(output_dir)


def _cleanup_dir(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        logger.warning("Failed to clean up partial conversion directory %s", path, exc_info=True)


def _build_command(hf_repo: str, task: str, output_dir: Path) -> list[str]:
    return [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        hf_repo,
        "--task",
        task,
        str(output_dir),
    ]


def convert_to_openvino(
    hf_repo: str,
    output_dir: Path,
    task: str = "text-generation-with-past",
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Convert a HuggingFace model to OpenVINO format."""
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if task not in VALID_EXPORT_TASKS:
        return {"error": f"Invalid task type. Must be one of {list(VALID_EXPORT_TASKS)}"}

    temp_dir = _temp_output_dir(output_dir)
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        _reset_dir(temp_dir)

        if progress_callback:
            progress_callback(f"Starting conversion of {hf_repo}")

        cmd = _build_command(hf_repo, task, temp_dir)
        if progress_callback:
            progress_callback(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CONVERSION_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            if progress_callback:
                progress_callback(f"Conversion failed: {error_msg}")
            _cleanup_dir(temp_dir)
            return {"error": f"Conversion failed: {error_msg}"}

        if not is_openvino_model(temp_dir):
            error_msg = "Conversion completed but output files not found"
            if progress_callback:
                progress_callback(error_msg)
            _cleanup_dir(temp_dir)
            return {"error": error_msg}

        _commit_output(temp_dir, output_dir)
        if progress_callback:
            progress_callback(f"Conversion successful: {output_dir}")

        return {"status": "success", "path": str(output_dir), "model": hf_repo}

    except FileNotFoundError:
        _cleanup_dir(temp_dir)
        return {"error": "optimum-cli not found. Install it with: pip install optimum-intel"}
    except subprocess.TimeoutExpired:
        _cleanup_dir(temp_dir)
        return {"error": "Conversion timed out after 1 hour"}
    except (PermissionError, OSError, subprocess.SubprocessError) as e:
        _cleanup_dir(temp_dir)
        logger.warning("Conversion failed for %s", hf_repo, exc_info=True)
        return {"error": f"Conversion failed: {e}"}
    except Exception as e:
        _cleanup_dir(temp_dir)
        logger.exception("Unexpected conversion failure for %s", hf_repo)
        return {"error": f"Conversion failed: {e}"}


def auto_download_and_convert(
    model_name: str,
    task: str = "text-generation-with-past",
    cache_dir: Path | None = None,
) -> dict:
    """Download and convert a model if needed."""
    if cache_dir is None:
        cache_dir = DEFAULT_CONVERSION_DIR
    elif isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    resolved = resolve_model_repo(model_name)
    if resolved is None:
        return {"error": f"Model '{model_name}' not found"}

    hf_repo, local_name = resolved
    runtime_name = resolve_runtime_model_name(model_name) or local_name
    storage_key = resolve_model_storage_key(model_name) or local_name
    output_dir = cache_dir / storage_key

    if is_openvino_model(output_dir):
        return {"status": "success", "path": str(output_dir), "model": runtime_name, "source": "cache"}

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        result = convert_to_openvino(hf_repo, output_dir, task)
        if "error" in result:
            return result
        return {"status": "success", "path": result["path"], "model": runtime_name, "source": "converted"}
    except (PermissionError, OSError) as e:
        logger.warning("Failed to convert model %s", model_name, exc_info=True)
        return {"error": f"Failed to convert model: {e}"}
    except Exception as e:
        logger.exception("Unexpected auto conversion failure for %s", model_name)
        return {"error": f"Failed to convert model: {e}"}


def get_conversion_progress(
    hf_repo: str,
    output_dir: Path,
    task: str = "text-generation-with-past",
) -> Generator[dict, None, None]:
    """Yield progress updates during model conversion for streaming."""
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if task not in VALID_EXPORT_TASKS:
        yield {"status": "error", "message": f"Invalid task type. Must be one of {list(VALID_EXPORT_TASKS)}"}
        return

    temp_dir = _temp_output_dir(output_dir)
    process: subprocess.Popen[str] | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        _reset_dir(temp_dir)
        yield {"status": "starting", "message": f"Starting conversion of {hf_repo}"}

        cmd = _build_command(hf_repo, task, temp_dir)
        yield {"status": "running", "message": f"Running: {' '.join(cmd)}"}

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            output, _ = process.communicate(timeout=CONVERSION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
            _cleanup_dir(temp_dir)
            for line in (output or "").splitlines():
                if line:
                    yield {"status": "converting", "message": line}
            yield {"status": "error", "message": "Conversion timed out after 1 hour"}
            return

        for line in (output or "").splitlines():
            if line:
                yield {"status": "converting", "message": line}

        if process.returncode != 0:
            _cleanup_dir(temp_dir)
            yield {"status": "error", "message": "Conversion failed with non-zero exit code"}
            return

        if is_openvino_model(temp_dir):
            _commit_output(temp_dir, output_dir)
            yield {"status": "success", "message": f"Conversion complete: {output_dir}", "path": str(output_dir)}
        else:
            _cleanup_dir(temp_dir)
            yield {"status": "error", "message": "Conversion completed but output files not found"}

    except FileNotFoundError:
        _cleanup_dir(temp_dir)
        yield {"status": "error", "message": "optimum-cli not found. Install it with: pip install optimum-intel"}
    except (PermissionError, OSError, subprocess.SubprocessError) as e:
        _cleanup_dir(temp_dir)
        logger.warning("Conversion progress failed for %s", hf_repo, exc_info=True)
        yield {"status": "error", "message": f"Conversion failed: {e}"}
    except Exception as e:
        _cleanup_dir(temp_dir)
        logger.exception("Unexpected conversion progress failure for %s", hf_repo)
        yield {"status": "error", "message": f"Conversion failed: {e}"}
