"""Parameter mapping between API formats and OpenVINO GenAI."""
import logging
from numbers import Integral, Real
from typing import Any

logger = logging.getLogger(__name__)

DIRECT_PARAMS: set[str] = {"temperature", "top_k", "top_p", "seed"}
RENAMED_PARAMS: dict[str, str] = {
    "repeat_penalty": "repetition_penalty",
    "num_predict": "max_new_tokens",
    "stop": "stop_strings",
}
IGNORED_PARAMS: set[str] = {
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "min_p",
    "typical_p",
    "tfs_z",
}
SILENT_PARAMS: set[str] = {"num_ctx", "num_batch"}
PENALTY_PARAMS: set[str] = {"presence_penalty", "frequency_penalty"}
ALL_KNOWN_PARAMS: set[str] = (
    DIRECT_PARAMS
    | set(RENAMED_PARAMS.keys())
    | IGNORED_PARAMS
    | SILENT_PARAMS
    | PENALTY_PARAMS
)

MAX_NEW_TOKENS_LIMIT = 8192
MAX_STOP_STRINGS = 16
MAX_STOP_STRING_LENGTH = 1024


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _validate_int(name: str, value: Any, *, minimum: int, maximum: int) -> int:
    if _is_bool(value) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _validate_float(name: str, value: Any, *, minimum: float, maximum: float) -> float:
    if _is_bool(value) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")
    normalized = float(value)
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _normalize_stop(value: Any) -> list[str]:
    if isinstance(value, str):
        stop_values = [value]
    elif isinstance(value, (list, tuple)):
        stop_values = list(value)
    else:
        raise ValueError("stop must be a string or list of strings")

    if len(stop_values) > MAX_STOP_STRINGS:
        raise ValueError(f"stop may contain at most {MAX_STOP_STRINGS} strings")
    normalized: list[str] = []
    for item in stop_values:
        if not isinstance(item, str):
            raise ValueError("stop entries must be strings")
        if len(item) > MAX_STOP_STRING_LENGTH:
            raise ValueError(f"stop entries must be at most {MAX_STOP_STRING_LENGTH} characters")
        normalized.append(item)
    return normalized


def _validate_value(target_key: str, value: Any) -> Any:
    if target_key == "max_new_tokens":
        token_count = _validate_int(target_key, value, minimum=-2, maximum=MAX_NEW_TOKENS_LIMIT)
        if token_count in {-1, -2} or token_count >= 1:
            return token_count
        raise ValueError(
            f"{target_key} must be -1, -2, or between 1 and {MAX_NEW_TOKENS_LIMIT}"
        )
    if target_key == "temperature":
        return _validate_float(target_key, value, minimum=0.0, maximum=2.0)
    if target_key == "top_p":
        return _validate_float(target_key, value, minimum=0.0, maximum=1.0)
    if target_key == "top_k":
        return _validate_int(target_key, value, minimum=0, maximum=1000)
    if target_key == "seed":
        return _validate_int(target_key, value, minimum=0, maximum=2**32 - 1)
    if target_key == "repetition_penalty":
        return _validate_float(target_key, value, minimum=0.0, maximum=3.0)
    if target_key == "stop_strings":
        return _normalize_stop(value)
    return value


def map_parameters(options: dict[str, Any] | None) -> dict[str, Any]:
    """Map Ollama/OpenAI parameters to OpenVINO GenAI equivalents.

    Invalid types or out-of-range values raise ValueError with a clear message.
    """
    if options is None:
        return {}

    result: dict[str, Any] = {}
    has_explicit_repeat = "repeat_penalty" in options

    for key, value in options.items():
        if key in DIRECT_PARAMS:
            result[key] = _validate_value(key, value)
        elif key in RENAMED_PARAMS:
            target_key = RENAMED_PARAMS[key]
            result[target_key] = _validate_value(target_key, value)
        elif key in IGNORED_PARAMS:
            logger.debug("Parameter %r is not supported by OpenVINO, ignoring", key)
        elif key in SILENT_PARAMS:
            pass
        elif key in PENALTY_PARAMS:
            _validate_float(key, value, minimum=-2.0, maximum=2.0)
        else:
            logger.warning("Unknown parameter %r, ignoring", key)

    if not has_explicit_repeat:
        presence = _validate_float(
            "presence_penalty",
            options.get("presence_penalty", 0.0),
            minimum=-2.0,
            maximum=2.0,
        )
        frequency = _validate_float(
            "frequency_penalty",
            options.get("frequency_penalty", 0.0),
            minimum=-2.0,
            maximum=2.0,
        )
        if presence or frequency:
            derived_penalty = 1.0 + (presence + frequency) / 2
            result["repetition_penalty"] = max(0.0, min(derived_penalty, 3.0))

    return result
