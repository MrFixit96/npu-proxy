"""Download OpenVINO models from Hugging Face Hub."""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Callable, Generator, Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

try:
    from huggingface_hub import HfHubHTTPError
except ImportError:  # pragma: no cover
    from huggingface_hub.utils import HfHubHTTPError

from npu_proxy.models.mapper import (
    resolve_model_repo,
    resolve_model_storage_key,
    resolve_runtime_model_name,
)
from npu_proxy.models.registry import find_catalog_entry

logger = logging.getLogger(__name__)
DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "npu-proxy" / "models"
REQUIRED_FILES: list[str] = ["openvino_model.xml", "openvino_model.bin"]
DOWNLOAD_METADATA_FILE = ".npu_proxy_download.json"
MAX_DOWNLOAD_FILE_BYTES = 8 * 1024**3
MAX_DOWNLOAD_TOTAL_BYTES = 16 * 1024**3
MAX_DOWNLOAD_FILES = 64
SAFE_DOWNLOAD_PATTERNS: tuple[str, ...] = (
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_*.xml",
    "openvino_*.bin",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
)


def _normalize_huggingface_token(token: str | bool | None) -> str | bool:
    if isinstance(token, str):
        normalized = token.strip()
        if normalized:
            return normalized
    return False


def _download_metadata_path(local_dir: Path) -> Path:
    return local_dir / DOWNLOAD_METADATA_FILE


def _repo_bool_flag(repo_info: object, attribute: str) -> bool:
    value = getattr(repo_info, attribute, False)
    return value if isinstance(value, bool) else False


def _write_download_metadata(
    local_dir: Path,
    *,
    source_repo: str,
    resolved_name: str,
    requires_token: bool,
    private: bool | None = None,
) -> None:
    metadata: dict[str, str | bool] = {
        "source_repo": source_repo,
        "resolved_name": resolved_name,
        "requires_token": requires_token,
    }
    if private is not None:
        metadata["private"] = private
    _download_metadata_path(local_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _read_download_metadata(local_dir: Path) -> dict[str, str | bool]:
    metadata_path = _download_metadata_path(local_dir)
    if not metadata_path.exists():
        return {}
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug("Unable to read download metadata from %s", metadata_path, exc_info=True)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if isinstance(key, str) and isinstance(value, (str, bool))}


def _huggingface_error_status_code(error: HfHubHTTPError) -> int:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return 401
    if status_code == 404:
        return 404
    return 502


def _huggingface_error_message(repo_id: str, error: HfHubHTTPError, *, attempted_auth: bool) -> str:
    status_code = _huggingface_error_status_code(error)
    if status_code == 401:
        if attempted_auth:
            return f"Failed to authenticate for Hugging Face repo '{repo_id}'."
        return (
            f"Hugging Face repo '{repo_id}' requires an explicit token. "
            "Provide a token in the request body or Authorization header."
        )
    if status_code == 404:
        return f"Model repository '{repo_id}' was not found or is not accessible."
    return f"Failed to download model: {error}"


def _safe_siblings(repo_info: object) -> list[object]:
    siblings = getattr(repo_info, "siblings", None)
    return list(siblings) if isinstance(siblings, (list, tuple)) else []


def _safe_file_size(sibling: object) -> int | None:
    size = getattr(sibling, "size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        return size
    return None


def _is_safe_repo_filename(filename: str, *, allow_nested: bool = False) -> bool:
    if not isinstance(filename, str) or not filename:
        return False
    if "\\" in filename or filename.startswith("/"):
        return False
    path = PurePosixPath(filename)
    if not allow_nested and len(path.parts) != 1:
        return False
    return all(part not in {"", ".", ".."} for part in path.parts)


def _allowed_patterns(required_files: Iterable[str]) -> list[str]:
    return sorted(set(SAFE_DOWNLOAD_PATTERNS) | set(required_files))


def _is_allowed_download_file(filename: str, required_files: Iterable[str]) -> bool:
    if filename in set(required_files):
        return True
    return any(fnmatch.fnmatchcase(filename, pattern) for pattern in SAFE_DOWNLOAD_PATTERNS)


def _select_download_files(siblings: Iterable[object], required_files: tuple[str, ...]) -> list[str]:
    selected: list[str] = []
    for sibling in siblings:
        filename = getattr(sibling, "rfilename", "")
        if isinstance(filename, str) and _is_allowed_download_file(filename, required_files):
            if not _is_safe_repo_filename(filename):
                raise ValueError(f"Unexpected unsafe filename in repository: {filename}")
            selected.append(filename)
    return sorted(dict.fromkeys(selected))


def _validate_download_manifest(
    repo_info: object,
    required_files: tuple[str, ...],
    *,
    full_snapshot: bool = False,
) -> list[str] | None:
    for filename in required_files:
        if not _is_safe_repo_filename(filename):
            raise ValueError(f"Unexpected unsafe required filename: {filename}")

    siblings = _safe_siblings(repo_info)
    if not siblings:
        return None if full_snapshot else _allowed_patterns(required_files)

    selected = siblings if full_snapshot else [
        sibling for sibling in siblings
        if isinstance(getattr(sibling, "rfilename", ""), str)
        and _is_allowed_download_file(getattr(sibling, "rfilename"), required_files)
    ]
    if len(selected) > MAX_DOWNLOAD_FILES:
        raise ValueError(f"Repository has too many downloadable files ({len(selected)} > {MAX_DOWNLOAD_FILES})")

    total_size = 0
    for sibling in selected:
        filename = getattr(sibling, "rfilename", "")
        if not _is_safe_repo_filename(filename, allow_nested=full_snapshot):
            raise ValueError(f"Unexpected unsafe filename in repository: {filename}")
        size = _safe_file_size(sibling)
        if size is None:
            continue
        if size > MAX_DOWNLOAD_FILE_BYTES:
            raise ValueError(f"File {filename} exceeds download size cap")
        total_size += size
    if total_size > MAX_DOWNLOAD_TOTAL_BYTES:
        raise ValueError("Repository exceeds total download size cap")

    return None if full_snapshot else _allowed_patterns(required_files)


def _is_regular_nonempty_file(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        return path.stat().st_size > 0
    except OSError:
        return False


def resolve_download_target(name: str, model_dir: Path) -> tuple[str, str, Path, tuple[str, ...]] | None:
    resolved = resolve_model_repo(name)
    if resolved is None:
        return None

    repo_id, _ = resolved
    from npu_proxy.inference.embedding_config import (
        EMBEDDING_REQUIRED_FILES,
        is_known_embedding_model,
        resolve_embedding_model_config,
    )

    if is_known_embedding_model(name):
        embedding_model = resolve_embedding_model_config(name, model_dir=model_dir)
        return (repo_id, embedding_model.requested_model, embedding_model.canonical_path, tuple(EMBEDDING_REQUIRED_FILES))

    runtime_name = resolve_runtime_model_name(name)
    storage_key = resolve_model_storage_key(name)
    if runtime_name is None or storage_key is None:
        return None
    return repo_id, runtime_name, model_dir / storage_key, tuple(REQUIRED_FILES)


def _has_required_files(local_dir: Path, required_files: tuple[str, ...]) -> bool:
    return all(_is_regular_nonempty_file(local_dir / file_name) for file_name in required_files)


def _has_repo_access(repo_id: str, token: str | bool) -> bool:
    if token is False:
        return False
    try:
        HfApi(token=token).repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        logger.warning("Failed to validate Hugging Face access for %s", repo_id, exc_info=True)
        return False
    return True


def download_model(
    name: str,
    model_dir: Path | str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    *,
    token: str | bool | None = False,
    full_snapshot: bool = False,
) -> dict[str, str]:
    """Download an OpenVINO-optimized model from Hugging Face Hub."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    elif isinstance(model_dir, str):
        model_dir = Path(model_dir)

    normalized_token = _normalize_huggingface_token(token)
    target = resolve_download_target(name, model_dir)
    if target is None:
        return {"error": "Model not found"}

    repo_id, local_name, local_dir, required_files = target

    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            progress_callback({"status": "pulling manifest"})

        repo_info = HfApi(token=normalized_token).repo_info(repo_id=repo_id, repo_type="model")
        allow_patterns = _validate_download_manifest(repo_info, required_files, full_snapshot=full_snapshot)

        snapshot_kwargs = {
            "repo_id": repo_id,
            "local_dir": local_dir,
            "token": normalized_token,
        }
        if allow_patterns is not None:
            snapshot_kwargs["allow_patterns"] = allow_patterns
        path = snapshot_download(**snapshot_kwargs)
        if progress_callback:
            progress_callback({"status": "downloaded"})

        missing_files = [file_name for file_name in required_files if not _is_regular_nonempty_file(Path(path) / file_name)]
        if missing_files:
            return {
                "error": f"Model downloaded but missing required files ({', '.join(missing_files)}) at {path}",
                "status_code": "422",
            }

        _write_download_metadata(
            Path(path),
            source_repo=repo_id,
            resolved_name=name,
            requires_token=(_repo_bool_flag(repo_info, "private") or _repo_bool_flag(repo_info, "gated")),
            private=_repo_bool_flag(repo_info, "private"),
        )
        return {"status": "success", "model": local_name, "path": str(path), "source": repo_id}

    except HfHubHTTPError as e:
        return {
            "error": _huggingface_error_message(repo_id, e, attempted_auth=normalized_token is not False),
            "status_code": str(_huggingface_error_status_code(e)),
        }
    except ValueError as e:
        return {"error": str(e), "status_code": "422"}
    except PermissionError as e:
        return {"error": f"Permission denied: {e}"}
    except OSError as e:
        return {"error": f"OS error during download: {e}"}
    except Exception as e:
        logger.exception("Unexpected error downloading %s", repo_id)
        return {"error": f"Unexpected error: {e}"}


def get_download_progress(
    repo_id: str,
    local_dir: Path | str,
    *,
    token: str | bool | None = False,
    resolved_name: str | None = None,
    required_files: tuple[str, ...] | None = None,
) -> Generator[dict[str, str | int], None, None]:
    """Stream download progress updates in Ollama-compatible format."""
    normalized_token = _normalize_huggingface_token(token)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    required_files = tuple(required_files or REQUIRED_FILES)

    yield {"status": "pulling manifest"}

    try:
        api = HfApi(token=normalized_token)
        siblings: list[object] = []
        private_repo: bool | None = None
        repo_requires_token: bool | None = None

        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
            siblings = _safe_siblings(repo_info)
            _validate_download_manifest(repo_info, required_files)
            files = _select_download_files(siblings, required_files) or list(required_files)
            private_repo = _repo_bool_flag(repo_info, "private")
            repo_requires_token = _repo_bool_flag(repo_info, "private") or _repo_bool_flag(repo_info, "gated")
        except HfHubHTTPError as e:
            logger.warning("Failed to fetch Hugging Face manifest for %s", repo_id, exc_info=True)
            yield {"status": _huggingface_error_message(repo_id, e, attempted_auth=normalized_token is not False)}
            return
        except ValueError as e:
            yield {"status": f"error: {e}"}
            return
        except Exception:
            logger.warning("Falling back to required-file manifest for %s", repo_id, exc_info=True)
            files = list(required_files)

        for filename in files:
            if not _is_safe_repo_filename(filename):
                yield {"status": f"error: unsafe filename {filename}"}
                return
            file_size = 0
            for sibling in siblings:
                if getattr(sibling, "rfilename", None) == filename:
                    file_size = _safe_file_size(sibling) or 0
                    break

            yield {"status": f"pulling {filename}", "total": file_size, "completed": 0}
            try:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, token=normalized_token)
                yield {"status": f"pulling {filename}", "total": file_size, "completed": file_size}
            except HfHubHTTPError as e:
                yield {"status": _huggingface_error_message(repo_id, e, attempted_auth=normalized_token is not False)}
                return
            except (OSError, ValueError) as e:
                yield {"status": f"error downloading {filename}: {e}"}
                return
            except Exception as e:
                logger.exception("Unexpected error downloading %s from %s", filename, repo_id)
                yield {"status": f"error downloading {filename}: {e}"}
                return

        yield {"status": "verifying required files"}
        if _has_required_files(local_dir, required_files):
            _write_download_metadata(
                local_dir,
                source_repo=repo_id,
                resolved_name=resolved_name or repo_id,
                requires_token=(repo_requires_token if repo_requires_token is not None else normalized_token is not False),
                private=private_repo,
            )
            yield {"status": "success"}
        else:
            missing_files = [file_name for file_name in required_files if not _is_regular_nonempty_file(local_dir / file_name)]
            yield {"status": "error: missing required files " + ", ".join(missing_files)}
    except Exception as e:
        logger.exception("Unexpected streaming download failure for %s", repo_id)
        yield {"status": f"error: {e}"}


def is_model_downloaded(
    name: str,
    model_dir: Path | str | None = None,
    *,
    token: str | bool | None = False,
) -> bool:
    """Check if a model is already downloaded and valid."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    elif isinstance(model_dir, str):
        model_dir = Path(model_dir)

    normalized_token = _normalize_huggingface_token(token)
    target = resolve_download_target(name, model_dir)
    if target is None:
        return False

    repo_id, _, local_dir, required_files = target
    if not _has_required_files(local_dir, required_files):
        return False

    metadata = _read_download_metadata(local_dir)
    catalog_entry = find_catalog_entry(name)
    if catalog_entry:
        expected_repo = catalog_entry["repo_id"]
        if not metadata or metadata.get("source_repo") != expected_repo:
            return False
    if not metadata and "/" in name:
        return False
    if metadata:
        source_repo = metadata.get("source_repo")
        if isinstance(source_repo, str) and source_repo and source_repo != repo_id:
            return False

    if metadata.get("requires_token") is True:
        if normalized_token is False:
            return False
        source_repo = metadata.get("source_repo")
        if isinstance(source_repo, str) and source_repo:
            return _has_repo_access(source_repo, normalized_token)
        return _has_repo_access(repo_id, normalized_token)

    return True


def get_downloaded_models(model_dir: Path | str | None = None) -> list[str]:
    """List all downloaded models in the cache directory."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    elif isinstance(model_dir, str):
        model_dir = Path(model_dir)

    if not model_dir.exists():
        return []

    downloaded = []
    try:
        for entry in model_dir.iterdir():
            if entry.is_dir() and _has_required_files(entry, tuple(REQUIRED_FILES)):
                downloaded.append(entry.name)
    except PermissionError:
        logger.warning("Permission denied while listing downloaded models in %s", model_dir, exc_info=True)
        return []
    except OSError:
        logger.warning("OS error while listing downloaded models in %s", model_dir, exc_info=True)
        return []

    return sorted(downloaded)
