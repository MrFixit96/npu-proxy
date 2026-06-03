"""Default parameter values for Ollama-compatible API."""
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping

_OLLAMA_DEFAULTS: dict[str, Any] = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "num_predict": 128,
    "num_ctx": 2048,
    "num_batch": 512,
    "seed": 0,
    "stop": (),
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "min_p": 0.0,
    "typical_p": 1.0,
    "tfs_z": 1.0,
}

OLLAMA_DEFAULTS: Mapping[str, Any] = MappingProxyType(_OLLAMA_DEFAULTS)


def _copy_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return deepcopy(value)


def get_default(param: str) -> Any:
    """Get a defensive copy of the default value for an Ollama parameter."""
    return _copy_default(OLLAMA_DEFAULTS[param])


def merge_with_defaults(options: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user options with defensive copies of Ollama defaults."""
    merged = {key: _copy_default(value) for key, value in OLLAMA_DEFAULTS.items()}
    if options:
        for key, value in options.items():
            merged[key] = deepcopy(value)
    return merged
