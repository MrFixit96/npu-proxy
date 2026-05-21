"""Model registry package"""
from npu_proxy.models.registry import (
    get_model_info,
    list_all_models,
    scan_available_models,
    get_openai_model_list,
    MODELS_INFO,
)
from npu_proxy.models.mapper import (
    resolve_model_repo,
    get_ollama_name,
    list_known_models,
)
from npu_proxy.models.downloader import (
    download_model,
    get_download_progress,
    is_model_downloaded,
    get_downloaded_models,
)

__all__ = [
    "get_model_info",
    "list_all_models",
    "scan_available_models",
    "get_openai_model_list",
    "MODELS_INFO",
    "resolve_model_repo",
    "get_ollama_name",
    "list_known_models",
    "download_model",
    "get_download_progress",
    "is_model_downloaded",
    "get_downloaded_models",
]
