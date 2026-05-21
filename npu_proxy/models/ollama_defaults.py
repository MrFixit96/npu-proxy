"""Default parameter values for Ollama-compatible API.

This module provides sensible defaults for LLM generation parameters,
mirroring Ollama's default behavior for compatibility.

Default Values:
    Sampling Parameters:
        temperature: 0.8 - Balanced creativity vs coherence (0=deterministic, 1+=creative)
        top_p: 0.9 - Nucleus sampling threshold (cumulative probability cutoff)
        top_k: 40 - Limits vocabulary to top K tokens per step
        repeat_penalty: 1.1 - Slight penalty to discourage repetition
        presence_penalty: 0.0 - No penalty for token presence (OpenAI-style)
        frequency_penalty: 0.0 - No penalty based on frequency (OpenAI-style)

    Generation Control:
        num_predict: 128 - Default max tokens to generate
        num_ctx: 2048 - Context window size in tokens
        num_batch: 512 - Batch size for prompt processing
        seed: 0 - Random seed (0 = non-deterministic)
        stop: [] - Empty list of stop sequences

    Mirostat (Adaptive Sampling):
        mirostat: 0 - Disabled by default (0=off, 1=v1, 2=v2)
        mirostat_tau: 5.0 - Target entropy for mirostat
        mirostat_eta: 0.1 - Learning rate for mirostat

    Advanced Sampling:
        min_p: 0.0 - Minimum probability threshold (disabled)
        typical_p: 1.0 - Typical sampling disabled (1.0 = no effect)
        tfs_z: 1.0 - Tail-free sampling disabled (1.0 = no effect)

Rationale:
    These defaults are chosen to match Ollama's behavior, ensuring
    that applications migrating from Ollama see consistent results.
    Values are sourced from the Ollama source code to guarantee
    API compatibility.

Example:
    >>> from npu_proxy.models.ollama_defaults import merge_with_defaults
    >>> options = {"temperature": 0.5}  # User provides only temp
    >>> full_options = merge_with_defaults(options)
    >>> full_options["top_p"]  # Gets default
    0.9

See Also:
    - Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
    - Ollama modelfile options: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
"""
from typing import Any

# All Ollama default values (from Ollama source code)
# See: https://github.com/ollama/ollama/blob/main/api/types.go
OLLAMA_DEFAULTS: dict[str, Any] = {
    # Sampling parameters - control randomness and token selection
    "temperature": 0.8,      # Higher = more random, lower = more focused
    "top_k": 40,             # Consider only top K tokens (0 = disabled)
    "top_p": 0.9,            # Consider tokens with cumulative prob <= top_p
    "repeat_penalty": 1.1,   # Penalize repeated tokens (1.0 = no penalty)
    "presence_penalty": 0.0, # OpenAI-style: penalize any token presence
    "frequency_penalty": 0.0,# OpenAI-style: penalize by frequency
    
    # Generation control - limits and behavior
    "num_predict": 128,      # Max tokens to generate (-1 = infinite, -2 = fill context)
    "num_ctx": 2048,         # Context window size
    "num_batch": 512,        # Prompt processing batch size
    "seed": 0,               # RNG seed (0 = random each time)
    "stop": [],              # Stop sequences (empty = none)
    
    # Mirostat - adaptive perplexity sampling (disabled by default)
    "mirostat": 0,           # 0=disabled, 1=mirostat v1, 2=mirostat v2
    "mirostat_tau": 5.0,     # Target entropy (perplexity) for mirostat
    "mirostat_eta": 0.1,     # Learning rate for mirostat adjustment
    
    # Advanced sampling techniques (disabled/neutral by default)
    "min_p": 0.0,            # Min probability relative to top token (0 = disabled)
    "typical_p": 1.0,        # Locally typical sampling (1.0 = disabled)
    "tfs_z": 1.0,            # Tail-free sampling z parameter (1.0 = disabled)
}


def get_default(param: str) -> Any:
    """Get the default value for an Ollama parameter.

    Retrieves the standard Ollama default for a given parameter name.
    Use this when you need a single default value without merging.

    Args:
        param: Parameter name (e.g., "temperature", "top_p", "num_predict").
            Must be a valid key in OLLAMA_DEFAULTS.

    Returns:
        Any: The default value for the parameter. Type depends on parameter:
            - float: temperature, top_p, repeat_penalty, etc.
            - int: top_k, num_predict, num_ctx, seed, mirostat, etc.
            - list: stop (empty list)

    Raises:
        KeyError: If param is not a known Ollama parameter.
            Valid parameters are keys in OLLAMA_DEFAULTS.

    Example:
        >>> get_default("temperature")
        0.8
        >>> get_default("num_predict")
        128
        >>> get_default("invalid_param")
        Traceback (most recent call last):
            ...
        KeyError: 'invalid_param'

    See Also:
        merge_with_defaults: For merging user options with all defaults.
    """
    return OLLAMA_DEFAULTS[param]


def merge_with_defaults(options: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user options with Ollama defaults.

    Creates a complete options dictionary by starting with all Ollama
    defaults and overlaying any user-provided values. User values take
    precedence over defaults.

    This ensures that NPU proxy requests always have all required
    parameters, even when the user only specifies a subset.

    Args:
        options: User-provided options dictionary, or None.
            Any keys present will override the corresponding defaults.
            Keys not in OLLAMA_DEFAULTS are preserved (pass-through).

    Returns:
        dict[str, Any]: Complete options dictionary containing:
            - All keys from OLLAMA_DEFAULTS with their default values
            - Any user-provided values overriding the defaults
            - Any extra user keys not in defaults (preserved as-is)

    Example:
        >>> merge_with_defaults(None)["temperature"]
        0.8
        >>> merge_with_defaults({})["top_p"]
        0.9
        >>> merge_with_defaults({"temperature": 0.5})["temperature"]
        0.5
        >>> merge_with_defaults({"temperature": 0.5})["top_p"]
        0.9
        >>> # Extra keys are preserved
        >>> merge_with_defaults({"custom_key": "value"})["custom_key"]
        'value'

    Note:
        This function does not validate parameter values. Invalid values
        (e.g., temperature > 2.0) are passed through and may cause errors
        during generation.

    See Also:
        get_default: For retrieving a single default value.
        OLLAMA_DEFAULTS: The source dictionary of all defaults.
    """
    if options is None:
        options = {}
    return {**OLLAMA_DEFAULTS, **options}
