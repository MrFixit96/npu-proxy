"""Parameter mapping between API formats and OpenVINO GenAI.

This module translates parameters from OpenAI and Ollama API formats to
the internal format used by OpenVINO GenAI LLMPipeline. It handles direct
mappings, parameter renaming, value transformations, and graceful handling
of unsupported parameters.

Parameter Mapping Table:
    +-------------------+-----------------+--------------------+-------------+
    | OpenAI            | Ollama          | OpenVINO GenAI     | Notes       |
    +===================+=================+====================+=============+
    | max_tokens        | num_predict     | max_new_tokens     | renamed     |
    +-------------------+-----------------+--------------------+-------------+
    | temperature       | temperature     | temperature        | direct      |
    +-------------------+-----------------+--------------------+-------------+
    | top_p             | top_p           | top_p              | direct      |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | top_k           | top_k              | direct      |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | seed            | seed               | direct      |
    +-------------------+-----------------+--------------------+-------------+
    | presence_penalty  | (n/a)           | repetition_penalty | converted   |
    +-------------------+-----------------+--------------------+-------------+
    | frequency_penalty | (n/a)           | repetition_penalty | converted   |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | repeat_penalty  | repetition_penalty | renamed     |
    +-------------------+-----------------+--------------------+-------------+
    | stop              | stop            | stop_strings       | renamed     |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | num_ctx         | (ignored)          | silent      |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | num_batch       | (ignored)          | silent      |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | mirostat        | (ignored)          | unsupported |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | mirostat_tau    | (ignored)          | unsupported |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | mirostat_eta    | (ignored)          | unsupported |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | min_p           | (ignored)          | unsupported |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | typical_p       | (ignored)          | unsupported |
    +-------------------+-----------------+--------------------+-------------+
    | (n/a)             | tfs_z           | (ignored)          | unsupported |
    +-------------------+-----------------+--------------------+-------------+

Value Transformations:
    - presence_penalty + frequency_penalty:
        OpenAI uses [-2, 2] for each. These are combined and converted to
        OpenVINO's repetition_penalty using: 1.0 + (presence + frequency) / 2
        This maps the OpenAI range to approximately [0, 3] for repetition_penalty.

    - repeat_penalty (Ollama):
        Maps directly to repetition_penalty. If explicitly provided, it takes
        precedence over presence_penalty/frequency_penalty conversion.

    - temperature:
        OpenAI allows [0, 2]. Values of 0 may cause issues; recommend using
        a small positive value like 0.01 as minimum.

Parameter Categories:
    - DIRECT_PARAMS: Pass through unchanged (temperature, top_k, top_p, seed)
    - RENAMED_PARAMS: Same meaning, different name (num_predict -> max_new_tokens)
    - PENALTY_PARAMS: Require value transformation (presence/frequency_penalty)
    - IGNORED_PARAMS: Unsupported, logged at DEBUG level (mirostat, etc.)
    - SILENT_PARAMS: Not applicable, silently ignored (num_ctx, num_batch)

Example:
    >>> from npu_proxy.models.parameter_mapper import map_parameters
    >>> # Ollama-style parameters
    >>> params = {"num_predict": 100, "temperature": 0.7, "top_p": 0.9}
    >>> mapped = map_parameters(params)
    >>> mapped["max_new_tokens"]
    100
    >>> mapped["temperature"]
    0.7

    >>> # OpenAI-style penalty conversion
    >>> params = {"presence_penalty": 0.5, "frequency_penalty": 0.3}
    >>> mapped = map_parameters(params)
    >>> mapped["repetition_penalty"]  # 1.0 + (0.5 + 0.3) / 2 = 1.4
    1.4
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Parameter Category Sets
# -----------------------------------------------------------------------------

DIRECT_PARAMS: set[str] = {"temperature", "top_k", "top_p", "seed"}
"""Parameters that map directly with the same name.

These parameters have identical names and semantics in both Ollama/OpenAI
and OpenVINO GenAI, requiring no transformation.
"""

RENAMED_PARAMS: dict[str, str] = {
    "repeat_penalty": "repetition_penalty",
    "num_predict": "max_new_tokens",
    "stop": "stop_strings",
}
"""Parameters that need renaming from source to OpenVINO format.

Keys are the source parameter names (Ollama/OpenAI), values are the
corresponding OpenVINO GenAI parameter names. The values themselves
do not require transformation.
"""

IGNORED_PARAMS: set[str] = {
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "min_p",
    "typical_p",
    "tfs_z",
}
"""Parameters not supported by OpenVINO GenAI (logged at DEBUG level).

These are advanced sampling parameters from Ollama that have no equivalent
in OpenVINO GenAI. Their use is logged for debugging but they are otherwise
silently dropped.
"""

SILENT_PARAMS: set[str] = {"num_ctx", "num_batch"}
"""Parameters silently ignored (not applicable to OpenVINO GenAI).

These context/batch size parameters are handled internally by OpenVINO
and cannot be configured via the generation API.
"""

PENALTY_PARAMS: set[str] = {"presence_penalty", "frequency_penalty"}
"""OpenAI-style penalty parameters that convert to repetition_penalty.

These are combined using the formula:
    repetition_penalty = 1.0 + (presence_penalty + frequency_penalty) / 2

This conversion only occurs when repeat_penalty is not explicitly set.
"""

ALL_KNOWN_PARAMS: set[str] = (
    DIRECT_PARAMS
    | set(RENAMED_PARAMS.keys())
    | IGNORED_PARAMS
    | SILENT_PARAMS
    | PENALTY_PARAMS
)
"""Union of all recognized parameter names for unknown parameter detection."""


def map_parameters(options: dict[str, Any] | None) -> dict[str, Any]:
    """Map Ollama/OpenAI parameters to OpenVINO GenAI equivalents.

    Translates API request parameters from Ollama or OpenAI format to the
    parameter names and values expected by OpenVINO GenAI's LLMPipeline.

    The mapping process follows this order of precedence:
        1. Direct parameters (temperature, top_k, top_p, seed) pass through
        2. Renamed parameters are translated to OpenVINO names
        3. If repeat_penalty is set, it's used directly
        4. Otherwise, presence_penalty + frequency_penalty are combined
        5. Unsupported parameters are logged and dropped

    Args:
        options: Dictionary of generation parameters in Ollama or OpenAI
            format. Can contain any combination of:
            - Direct: temperature, top_k, top_p, seed
            - Renamed: num_predict, repeat_penalty, stop
            - Penalty: presence_penalty, frequency_penalty
            - Ignored: mirostat, mirostat_tau, mirostat_eta, min_p,
              typical_p, tfs_z, num_ctx, num_batch
            If None, returns an empty dictionary.

    Returns:
        Dictionary of OpenVINO GenAI compatible parameters. Keys may include:
        - max_new_tokens (int): Maximum tokens to generate
        - temperature (float): Sampling temperature [0, 2]
        - top_k (int): Top-k sampling parameter
        - top_p (float): Nucleus sampling probability [0, 1]
        - seed (int): Random seed for reproducibility
        - repetition_penalty (float): Penalty for repeated tokens [0, ~3]
        - stop_strings (list[str]): Sequences that stop generation

    Examples:
        Basic parameter mapping:

        >>> map_parameters({"temperature": 0.7, "num_predict": 100})
        {'temperature': 0.7, 'max_new_tokens': 100}

        Penalty conversion (when no explicit repeat_penalty):

        >>> map_parameters({"presence_penalty": 0.6, "frequency_penalty": 0.4})
        {'repetition_penalty': 1.5}

        Explicit repeat_penalty takes precedence:

        >>> map_parameters({
        ...     "repeat_penalty": 1.2,
        ...     "presence_penalty": 0.6  # ignored
        ... })
        {'repetition_penalty': 1.2}

        None input returns empty dict:

        >>> map_parameters(None)
        {}

    Note:
        Unknown parameters not in any category will trigger a warning log
        and be ignored. This allows forward compatibility with new API
        parameters while making debugging easier.
    """
    if options is None:
        return {}

    result: dict[str, Any] = {}

    # Track if repeat_penalty was explicitly provided (takes precedence)
    has_explicit_repeat: bool = "repeat_penalty" in options

    for key, value in options.items():
        if key in DIRECT_PARAMS:
            # Pass through unchanged: temperature, top_k, top_p, seed
            result[key] = value
        elif key in RENAMED_PARAMS:
            # Translate parameter name: e.g., num_predict -> max_new_tokens
            result[RENAMED_PARAMS[key]] = value
        elif key in IGNORED_PARAMS:
            # Unsupported by OpenVINO - log for debugging
            logger.debug(f"Parameter '{key}' is not supported by OpenVINO, ignoring")
        elif key in SILENT_PARAMS:
            # Not applicable - silently ignore (num_ctx, num_batch)
            pass
        elif key in PENALTY_PARAMS:
            # Handled in penalty conversion below
            pass
        else:
            # Unknown parameter - warn for debugging
            logger.warning(f"Unknown parameter '{key}', ignoring")

    # Convert OpenAI-style penalties to repetition_penalty
    # Only if no explicit repeat_penalty was provided
    if not has_explicit_repeat:
        presence: float = options.get("presence_penalty", 0.0)
        frequency: float = options.get("frequency_penalty", 0.0)
        if presence or frequency:
            # Conversion formula: maps OpenAI [-2,2] range to repetition [0,~3]
            # When both are 0, no repetition_penalty is added (use model default)
            result["repetition_penalty"] = 1.0 + (presence + frequency) / 2

    return result
