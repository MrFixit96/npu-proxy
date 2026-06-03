"""Model search utilities for finding OpenVINO-compatible models."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, TypeVar

from npu_proxy.models.metadata import (
    MODEL_TYPE_KEYWORDS,
    detect_architecture,
    detect_format,
    detect_model_metadata,
    detect_model_type,
    detect_quantization,
    detect_parameters,
)

logger = logging.getLogger(__name__)
HF_TIMEOUT_SECONDS = 10.0
MAX_QUERY_LENGTH = 200
MAX_SEARCH_LIMIT = 100
T = TypeVar("T")

try:
    from huggingface_hub import HfApi, list_models
    from huggingface_hub.utils import HFValidationError, HfHubHTTPError, RepositoryNotFoundError

    try:
        from requests import RequestException
    except ImportError:  # pragma: no cover - requests is a HF dependency in normal installs
        RequestException = OSError  # type: ignore[assignment]

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

    class HfHubHTTPError(Exception):
        pass

    class HFValidationError(ValueError):
        pass

    class RepositoryNotFoundError(HfHubHTTPError):
        pass

    class RequestException(OSError):
        pass


_EXPECTED_HF_ERRORS = (
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
    RequestException,
    OSError,
    TimeoutError,
    FutureTimeoutError,
)


@dataclass
class SearchResult:
    """Represents a search result for an OpenVINO model."""

    id: str
    name: str
    author: str
    downloads: int
    likes: int
    last_modified: str
    quantization: str
    parameters: str
    architecture: str


def _run_with_timeout(func: Callable[[], T]) -> T:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=HF_TIMEOUT_SECONDS)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def extract_quantization(text: str) -> str:
    """Extract quantization type from a model name or ID string."""
    return detect_quantization(text)


def extract_parameters(text: str) -> str:
    """Extract parameter count from a model name or ID string."""
    return detect_parameters(text)


def extract_architecture(text: str) -> str:
    """Extract model architecture family from a model name or ID string."""
    return detect_architecture(text)


def extract_model_metadata(model_id: str) -> dict:
    """Extract legacy search metadata fields from a model ID/name."""
    metadata = detect_model_metadata(model_id)
    return {
        "quantization": metadata["quantization"],
        "parameters": metadata["parameters"],
        "architecture": metadata["architecture"],
    }


def is_openvino_compatible(repo_id: str) -> bool:
    """Check if a HuggingFace repository is OpenVINO compatible."""
    if not HF_AVAILABLE:
        return False

    try:
        api = HfApi()
        model_info = api.model_info(repo_id, timeout=HF_TIMEOUT_SECONDS)

        if model_info.author and model_info.author.lower() == "openvino":
            return True

        if detect_format(repo_id) == "openvino-ir":
            return True

        if model_info.tags:
            for tag in model_info.tags:
                if "openvino" in tag.lower():
                    return True

        return False
    except _EXPECTED_HF_ERRORS:
        logger.warning("Failed to check OpenVINO compatibility for %r", repo_id, exc_info=True)
        return False


def _matches_model_type(model_id: str, model_type: str) -> bool:
    if model_type == "all" or not model_type:
        return True

    detected_type = detect_model_type(model_id)
    if detected_type:
        return detected_type == model_type.lower()

    keywords = MODEL_TYPE_KEYWORDS.get(model_type.lower(), ())
    lowered = model_id.lower()
    return any(keyword in lowered for keyword in keywords)


def _matches_quantization_filter(quantization: str, filter_quantization: str) -> bool:
    if not filter_quantization:
        return True

    actual = (quantization or "").upper()
    expected = filter_quantization.upper()
    if actual == expected:
        return True

    if expected.startswith("INT") and actual.startswith("Q"):
        return actual[1:2] == expected[-1:]

    return False


def _normalize_search_query(query: str) -> str:
    return str(query or "").strip()[:MAX_QUERY_LENGTH]


@lru_cache(maxsize=128)
def _cached_search(
    query: str,
    sort: str,
    model_type: str,
    quantization: str,
    min_downloads: int,
) -> tuple[list[dict], int]:
    if not HF_AVAILABLE:
        return [], 0

    try:
        sort_mapping = {
            "popular": "downloads",
            "newest": "lastModified",
            "downloads": "downloads",
            "likes": "likes",
        }
        hf_sort = sort_mapping.get(sort.lower(), "downloads")

        search_kwargs = {
            "filter": "openvino",
            "sort": hf_sort,
        }
        normalized_query = _normalize_search_query(query)
        if normalized_query:
            search_kwargs["search"] = normalized_query

        models = _run_with_timeout(lambda: list(list_models(**search_kwargs, limit=500)))
        results: list[dict] = []

        for model in models:
            model_id = model.id or ""
            metadata = detect_model_metadata(model_id)

            if not _matches_model_type(model_id, model_type):
                continue
            if not _matches_quantization_filter(metadata["quantization"], quantization):
                continue

            downloads = model.downloads or 0
            if downloads < min_downloads:
                continue

            last_modified = ""
            if model.last_modified:
                try:
                    last_modified = model.last_modified.isoformat()
                except (AttributeError, ValueError):
                    last_modified = str(model.last_modified)

            name = model_id.split("/")[-1] if "/" in model_id else model_id
            results.append(
                {
                    "id": model_id,
                    "name": name,
                    "author": model.author or "",
                    "downloads": downloads,
                    "likes": model.likes or 0,
                    "last_modified": last_modified,
                    "quantization": metadata["quantization"],
                    "parameters": metadata["parameters"],
                    "architecture": metadata["architecture"],
                }
            )

        return results, len(results)
    except _EXPECTED_HF_ERRORS:
        logger.warning("Hugging Face OpenVINO search failed", exc_info=True)
        return [], 0


def _clamp_pagination(limit: int, offset: int) -> tuple[int, int]:
    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        limit_int = 20
    try:
        offset_int = int(offset)
    except (TypeError, ValueError):
        offset_int = 0
    return max(1, min(limit_int, MAX_SEARCH_LIMIT)), max(0, offset_int)


def search_openvino_models(
    query: str = "",
    sort: str = "popular",
    limit: int = 20,
    offset: int = 0,
    model_type: str = "all",
    quantization: str = "",
    min_downloads: int = 0,
) -> tuple[list[SearchResult], int]:
    """Search HuggingFace for OpenVINO-compatible models."""
    limit, offset = _clamp_pagination(limit, offset)
    min_downloads = max(0, int(min_downloads or 0))
    cached_results, total = _cached_search(
        _normalize_search_query(query), sort, model_type, quantization, min_downloads
    )
    paginated = cached_results[offset : offset + limit]
    return [SearchResult(**result) for result in paginated], total


def get_model_details(repo_id: str) -> Optional[SearchResult]:
    """Get detailed information for a specific model."""
    if not HF_AVAILABLE:
        return None

    try:
        api = HfApi()
        model = api.model_info(repo_id, timeout=HF_TIMEOUT_SECONDS)
        metadata = detect_model_metadata(repo_id)

        last_modified = ""
        if model.last_modified:
            try:
                last_modified = model.last_modified.isoformat()
            except (AttributeError, ValueError):
                last_modified = str(model.last_modified)

        name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        return SearchResult(
            id=repo_id,
            name=name,
            author=model.author or "",
            downloads=model.downloads or 0,
            likes=model.likes or 0,
            last_modified=last_modified,
            quantization=metadata["quantization"],
            parameters=metadata["parameters"],
            architecture=metadata["architecture"],
        )
    except _EXPECTED_HF_ERRORS:
        logger.warning("Failed to fetch model details for %r", repo_id, exc_info=True)
        return None
