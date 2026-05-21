"""Model search utilities for finding OpenVINO-compatible models.

This module provides functionality to search, filter, and retrieve metadata
for OpenVINO-optimized models from the HuggingFace Hub. It supports searching
by model name, filtering by quantization type and model family, and sorting
by popularity or recency.

The module integrates with the HuggingFace Hub API via the `huggingface_hub`
package. Results are cached using LRU caching for improved performance on
repeated queries.

External API Integration:
    - HuggingFace Hub API: https://huggingface.co/docs/huggingface_hub
    - Uses `huggingface_hub.list_models()` for search queries
    - Uses `huggingface_hub.HfApi.model_info()` for detailed model metadata
    - Filters by the 'openvino' tag to find compatible models

Supported Model Types:
    - LLM: Language models (Llama, Phi, Qwen, Mistral, etc.)
    - Embedding: Text embedding models (BGE, E5, MiniLM, etc.)
    - Vision: Vision models (ViT, CLIP, LLaVA, etc.)

Quantization Formats:
    - INT4: 4-bit integer quantization (best for NPU)
    - INT8: 8-bit integer quantization
    - FP16: Half-precision floating point
    - FP32: Full-precision floating point
    - BF16: Brain floating point 16

Example:
    >>> from npu_proxy.models.search import search_openvino_models
    >>> models, count = search_openvino_models("llama", limit=5)
    >>> models[0].id
    'OpenVINO/Llama-2-7B-chat-int4-ov'
    >>> models[0].quantization
    'INT4'

    >>> from npu_proxy.models.search import get_model_details
    >>> model = get_model_details("OpenVINO/phi-2-int4-ov")
    >>> model.parameters
    '2B'

Note:
    The `huggingface_hub` package is optional. If not installed, all search
    functions will return empty results and `HF_AVAILABLE` will be False.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

try:
    from huggingface_hub import HfApi, list_models
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Architecture patterns for extraction
ARCHITECTURE_PATTERNS = {
    r'\bllama\b': 'LLaMA',
    r'\bphi\b': 'Phi',
    r'\bqwen\b': 'Qwen',
    r'\bmistral\b': 'Mistral',
    r'\bgemma\b': 'Gemma',
    r'\bfalcon\b': 'Falcon',
    r'\bmpt\b': 'MPT',
    r'\bopt\b': 'OPT',
    r'\bgpt-?neo': 'GPT-Neo',
    r'\bgpt-?j\b': 'GPT-J',
    r'\bbloom\b': 'BLOOM',
    r'\bstarcoder\b': 'StarCoder',
    r'\bcodellama\b': 'CodeLLaMA',
    r'\btinyllama\b': 'TinyLLaMA',
    r'\bvicuna\b': 'Vicuna',
    r'\balpaca\b': 'Alpaca',
    r'\bzephyr\b': 'Zephyr',
    r'\borca\b': 'Orca',
    r'\bneuralchat\b': 'NeuralChat',
    r'\bdolly\b': 'Dolly',
    r'\bred.?pajama\b': 'RedPajama',
    r'\byi\b': 'Yi',
    r'\bdeepseek\b': 'DeepSeek',
    r'\bchat.?glm\b': 'ChatGLM',
    r'\bbaichuan\b': 'Baichuan',
    r'\binternlm\b': 'InternLM',
}

# Model type keywords for filtering
MODEL_TYPE_KEYWORDS = {
    'llm': ['llama', 'phi', 'qwen', 'mistral', 'gemma', 'falcon', 'gpt', 'chat', 
            'instruct', 'coder', 'bloom', 'opt', 'mpt', 'yi', 'deepseek', 'zephyr'],
    'embedding': ['embedding', 'bge', 'e5', 'gte', 'sentence', 'bert', 'minilm'],
    'vision': ['vision', 'vit', 'clip', 'image', 'llava', 'blip', 'sam'],
}


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


def extract_quantization(text: str) -> str:
    """Extract quantization type from a model name or ID string.

    Parses the model name to identify the quantization format used.
    Supports common notations like 'int4', 'INT8', 'fp16', 'F32', etc.

    Args:
        text: Model name or ID string to parse (e.g., "llama-2-7b-int4-ov").

    Returns:
        Uppercase quantization string (e.g., "INT4", "FP16") or empty string
        if no quantization format is detected.

    Example:
        >>> extract_quantization("OpenVINO/phi-2-int4-ov")
        'INT4'
        >>> extract_quantization("model-fp16")
        'FP16'
        >>> extract_quantization("some-model")
        ''
    """
    text_lower = text.lower()
    
    # Check for INT4 variants
    if re.search(r'\bint4\b', text_lower) or re.search(r'\bi4\b', text_lower):
        return 'INT4'
    
    # Check for INT8 variants
    if re.search(r'\bint8\b', text_lower) or re.search(r'\bi8\b', text_lower):
        return 'INT8'
    
    # Check for FP16 variants
    if re.search(r'\bfp16\b', text_lower) or re.search(r'\bf16\b', text_lower):
        return 'FP16'
    
    # Check for FP32 variants
    if re.search(r'\bfp32\b', text_lower) or re.search(r'\bf32\b', text_lower):
        return 'FP32'
    
    # Check for BF16 variants
    if re.search(r'\bbf16\b', text_lower):
        return 'BF16'
    
    return ''


def extract_parameters(text: str) -> str:
    """Extract parameter count from a model name or ID string.

    Parses the model name to identify the parameter count. Supports
    patterns like '7B', '1.1B', '350M', '125M', etc.

    Args:
        text: Model name or ID string to parse (e.g., "llama-2-7b-chat").

    Returns:
        Parameter count string with unit (e.g., "7B", "350M") or empty
        string if no parameter count is detected.

    Example:
        >>> extract_parameters("OpenVINO/Llama-2-7B-chat-int4-ov")
        '7B'
        >>> extract_parameters("TinyLlama-1.1B-Chat")
        '1.1B'
        >>> extract_parameters("all-MiniLM-L6-v2")
        ''
    """
    # Match patterns like: 7B, 7b, 1.1B, 0.5B, 70B, 1B, 2B, 13B, etc.
    match = re.search(r'(\d+\.?\d*)\s*[Bb]\b', text)
    if match:
        param_num = match.group(1)
        return f'{param_num}B'
    
    # Match patterns like: 350M, 125M, etc.
    match = re.search(r'(\d+)\s*[Mm]\b', text)
    if match:
        return f'{match.group(1)}M'
    
    return ''


def extract_architecture(text: str) -> str:
    """Extract model architecture family from a model name or ID string.

    Parses the model name to identify the base architecture. Uses pattern
    matching against known architecture names defined in ARCHITECTURE_PATTERNS.

    Args:
        text: Model name or ID string to parse (e.g., "llama-2-7b-chat").

    Returns:
        Architecture name (e.g., "LLaMA", "Phi", "Mistral") or empty string
        if no known architecture is detected.

    Example:
        >>> extract_architecture("OpenVINO/Llama-2-7B-chat-int4-ov")
        'LLaMA'
        >>> extract_architecture("phi-2-int4-ov")
        'Phi'
        >>> extract_architecture("unknown-model")
        ''
    """
    text_lower = text.lower()
    
    for pattern, arch in ARCHITECTURE_PATTERNS.items():
        if re.search(pattern, text_lower):
            return arch
    
    return ''


def extract_model_metadata(model_id: str) -> dict:
    """
    Extract metadata from a model ID/name.
    
    Args:
        model_id: The HuggingFace model ID or name
        
    Returns:
        Dictionary with quantization, parameters, and architecture fields
    """
    return {
        'quantization': extract_quantization(model_id),
        'parameters': extract_parameters(model_id),
        'architecture': extract_architecture(model_id),
    }


def is_openvino_compatible(repo_id: str) -> bool:
    """
    Check if a HuggingFace repository is OpenVINO compatible.
    
    Args:
        repo_id: The HuggingFace repository ID (e.g., "OpenVINO/phi-2-int4-ov")
        
    Returns:
        True if the model is OpenVINO compatible, False otherwise
    """
    if not HF_AVAILABLE:
        return False
    
    try:
        api = HfApi()
        model_info = api.model_info(repo_id)
        
        # Check if author is OpenVINO
        if model_info.author and model_info.author.lower() == 'openvino':
            return True
        
        # Check if model has openvino tag
        if model_info.tags:
            for tag in model_info.tags:
                if 'openvino' in tag.lower():
                    return True
        
        return False
        
    except Exception:
        return False


def _matches_model_type(model_id: str, model_type: str) -> bool:
    """Check if a model matches the specified type filter.

    Uses keyword matching against MODEL_TYPE_KEYWORDS to determine if a
    model belongs to the specified category (llm, embedding, or vision).

    Args:
        model_id: HuggingFace model ID to check.
        model_type: Type filter - "all", "llm", "embedding", or "vision".

    Returns:
        True if the model matches the type filter or if filter is "all".
    """
    if model_type == 'all' or not model_type:
        return True
    
    model_id_lower = model_id.lower()
    keywords = MODEL_TYPE_KEYWORDS.get(model_type.lower(), [])
    
    return any(kw in model_id_lower for kw in keywords)


def _matches_quantization_filter(quant: str, filter_quant: str) -> bool:
    """Check if a model's quantization matches the filter.

    Performs case-insensitive comparison of quantization strings.

    Args:
        quant: Model's quantization type (e.g., "INT4").
        filter_quant: Filter value to match against.

    Returns:
        True if quantization matches filter or if filter is empty.
    """
    if not filter_quant:
        return True
    
    return quant.lower() == filter_quant.lower()


@lru_cache(maxsize=128)
def _cached_search(
    query: str,
    sort: str,
    model_type: str,
    quantization: str,
    min_downloads: int,
) -> tuple[list[dict], int]:
    """
    Cached search implementation.
    
    Returns raw dicts to allow lru_cache (dataclasses aren't hashable by default).
    """
    if not HF_AVAILABLE:
        return [], 0
    
    try:
        # Map sort parameter to HuggingFace API sort options
        sort_mapping = {
            'popular': 'downloads',
            'newest': 'lastModified',
            'downloads': 'downloads',
            'likes': 'likes',
        }
        hf_sort = sort_mapping.get(sort.lower(), 'downloads')
        
        # Build search parameters
        search_kwargs = {
            'filter': 'openvino',
            'sort': hf_sort,
        }
        
        if query:
            search_kwargs['search'] = query
        
        # Fetch enough to account for filtering
        fetch_limit = 500
        
        # Get models from HuggingFace
        models = list(list_models(**search_kwargs, limit=fetch_limit))
        
        # Process and filter results
        results: list[dict] = []
        
        for model in models:
            model_id = model.id or ''
            
            # Extract metadata
            metadata = extract_model_metadata(model_id)
            
            # Apply filters
            if not _matches_model_type(model_id, model_type):
                continue
            
            if not _matches_quantization_filter(metadata['quantization'], quantization):
                continue
            
            downloads = model.downloads or 0
            if downloads < min_downloads:
                continue
            
            # Parse last modified date
            last_modified = ''
            if model.last_modified:
                try:
                    last_modified = model.last_modified.isoformat()
                except (AttributeError, ValueError):
                    last_modified = str(model.last_modified)
            
            # Extract display name from model ID
            name = model_id.split('/')[-1] if '/' in model_id else model_id
            
            result = {
                'id': model_id,
                'name': name,
                'author': model.author or '',
                'downloads': downloads,
                'likes': model.likes or 0,
                'last_modified': last_modified,
                'quantization': metadata['quantization'],
                'parameters': metadata['parameters'],
                'architecture': metadata['architecture'],
            }
            results.append(result)
        
        return results, len(results)
        
    except Exception:
        return [], 0


def search_openvino_models(
    query: str = "",
    sort: str = "popular",
    limit: int = 20,
    offset: int = 0,
    model_type: str = "all",
    quantization: str = "",
    min_downloads: int = 0,
) -> tuple[list[SearchResult], int]:
    """
    Search HuggingFace for OpenVINO-compatible models.
    
    Args:
        query: Search query string
        sort: Sort order - "popular", "newest", "downloads", "likes"
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        model_type: Filter by type - "all", "llm", "embedding", "vision"
        quantization: Filter by quantization - "int4", "int8", "fp16"
        min_downloads: Minimum number of downloads required
        
    Returns:
        Tuple of (list of SearchResult, total count matching filters)
    """
    # Use cached search for the heavy lifting
    cached_results, total = _cached_search(
        query, sort, model_type, quantization, min_downloads
    )
    
    # Apply pagination
    paginated = cached_results[offset:offset + limit]
    
    # Convert dicts to SearchResult objects
    results = [SearchResult(**r) for r in paginated]
    
    return results, total


def get_model_details(repo_id: str) -> Optional[SearchResult]:
    """
    Get detailed information for a specific model.
    
    Args:
        repo_id: The HuggingFace repository ID
        
    Returns:
        SearchResult with model details, or None if not found
    """
    if not HF_AVAILABLE:
        return None
    
    try:
        api = HfApi()
        model = api.model_info(repo_id)
        
        metadata = extract_model_metadata(repo_id)
        
        last_modified = ''
        if model.last_modified:
            try:
                last_modified = model.last_modified.isoformat()
            except (AttributeError, ValueError):
                last_modified = str(model.last_modified)
        
        name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
        
        return SearchResult(
            id=repo_id,
            name=name,
            author=model.author or '',
            downloads=model.downloads or 0,
            likes=model.likes or 0,
            last_modified=last_modified,
            quantization=metadata['quantization'],
            parameters=metadata['parameters'],
            architecture=metadata['architecture'],
        )
        
    except Exception:
        return None
