# Tokenizer Best Practices Research

## Research Sources
- **tiktoken** (OpenAI): Fast BPE tokenizer for OpenAI models
- **Hugging Face tokenizers**: State-of-the-art tokenizer library (Rust core)
- **SentencePiece** (Google): Unsupervised text tokenizer supporting BPE and Unigram

---

## 1. Accurate Token Counting vs Approximation Trade-offs

### Pattern: Exact vs Approximate Token Counting

**Problem it solves:** Balancing accuracy requirements against performance costs for token counting.

**Trade-off Analysis:**

| Approach | Accuracy | Performance | Use Case |
|----------|----------|-------------|----------|
| Full tokenization (tiktoken/HF) | 100% | Slower | Billing, context limits |
| Regex approximation | 80-90% | Fast | Quick estimates, UI |
| Character/word ratio | 70-80% | Fastest | Rough estimates |

**Code Example - tiktoken (Exact):**
```python
import tiktoken

def count_tokens_exact(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken - 100% accurate for OpenAI models."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# For chat messages with overhead
def count_tokens_for_messages(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Message overhead
        for key, value in message.items():
            num_tokens += len(enc.encode(str(value)))
    num_tokens += 2  # Reply priming
    return num_tokens
```

**Code Example - Fast Approximation:**
```python
import re

# Current npu-proxy approach - regex approximation
TOKEN_PATTERN = re.compile(r"""
    '[a-zA-Z]+        # Contractions
    |[a-zA-Z]+        # Words
    |[0-9]+           # Numbers
    |[^\s\w]          # Punctuation
""", re.VERBOSE)

def count_tokens_approximate(text: str) -> int:
    """Fast approximation - good for estimates."""
    if not text or not text.strip():
        return 0
    return len(TOKEN_PATTERN.findall(text))
```

**Applicability to npu-proxy tokenizer.py:**
- Current implementation uses regex approximation (fast, ~85% accurate)
- Consider optional tiktoken integration for exact counting when needed
- Add accuracy parameter: `count_tokens(text, exact=False)`

---

## 2. Tokenizer Caching Patterns

### Pattern: Singleton Tokenizer Cache

**Problem it solves:** Avoid expensive tokenizer initialization on every call. Tokenizer loading can take 50-200ms.

**Code Example - tiktoken (Built-in caching):**
```python
import tiktoken

# tiktoken caches encodings automatically
enc1 = tiktoken.get_encoding("cl100k_base")  # First call: loads
enc2 = tiktoken.get_encoding("cl100k_base")  # Cached: instant

# Model-based encoding also cached
enc = tiktoken.encoding_for_model("gpt-4o")  # Returns o200k_base encoding
```

**Code Example - HuggingFace Tokenizers with LRU Cache:**
```python
from functools import lru_cache
from tokenizers import Tokenizer

@lru_cache(maxsize=8)
def get_tokenizer(model_name: str) -> Tokenizer:
    """Cache tokenizers by model name."""
    return Tokenizer.from_pretrained(model_name)

# Thread-safe usage
class TokenizerCache:
    _instance = None
    _lock = threading.Lock()
    _tokenizers: Dict[str, Tokenizer] = {}
    
    @classmethod
    def get(cls, model_name: str) -> Tokenizer:
        with cls._lock:
            if model_name not in cls._tokenizers:
                cls._tokenizers[model_name] = Tokenizer.from_pretrained(model_name)
            return cls._tokenizers[model_name]
```

**Code Example - SentencePiece Caching:**
```python
import sentencepiece as spm

class SentencePieceCache:
    _processors: Dict[str, spm.SentencePieceProcessor] = {}
    
    @classmethod
    def get(cls, model_path: str) -> spm.SentencePieceProcessor:
        if model_path not in cls._processors:
            sp = spm.SentencePieceProcessor(model_file=model_path)
            cls._processors[model_path] = sp
        return cls._processors[model_path]
```

**Applicability to npu-proxy tokenizer.py:**
- Current implementation has no caching (regex pattern is compiled at module load)
- If adding tiktoken/HF tokenizers, implement singleton cache
- Consider warm-up on startup for common model encodings

---

## 3. Fast Token Estimation Without Full Tokenization

### Pattern: Heuristic Token Estimation

**Problem it solves:** Get token count estimates quickly when exact counting is too slow.

**Code Example - Character Ratio Estimation:**
```python
# Average characters per token varies by language/encoding
# English: ~4 chars/token, Code: ~3.5 chars/token
CHARS_PER_TOKEN = {
    "gpt-4": 4.0,
    "gpt-3.5": 4.0,
    "claude": 3.8,
    "code": 3.5,
}

def estimate_tokens_fast(text: str, model_type: str = "gpt-4") -> int:
    """Ultra-fast estimation based on character count."""
    ratio = CHARS_PER_TOKEN.get(model_type, 4.0)
    return int(len(text) / ratio)
```

**Code Example - Word + Punctuation Estimation:**
```python
import re

def estimate_tokens_medium(text: str) -> int:
    """Medium accuracy estimation - faster than full tokenization."""
    if not text:
        return 0
    
    # Count words (each word ≈ 1.3 tokens on average)
    words = len(text.split())
    
    # Count punctuation (each is typically 1 token)
    punctuation = len(re.findall(r'[^\w\s]', text))
    
    # Count numbers (each digit group ≈ 1 token)
    numbers = len(re.findall(r'\d+', text))
    
    return int(words * 1.3) + punctuation
```

**Code Example - Tiered Estimation Strategy:**
```python
def count_tokens_smart(text: str, precision: str = "medium") -> int:
    """
    Tiered token counting with configurable precision.
    
    precision levels:
    - "fast": Character ratio (~100x faster, ±20% accuracy)
    - "medium": Regex pattern (~10x faster, ±10% accuracy)  
    - "exact": Full tokenization (baseline, 100% accuracy)
    """
    if precision == "fast":
        return int(len(text) / 4.0)
    elif precision == "medium":
        return len(TOKEN_PATTERN.findall(text))
    else:  # exact
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
```

**Applicability to npu-proxy tokenizer.py:**
- Add tiered estimation with precision parameter
- Use fast estimation for real-time UI feedback
- Use exact counting for billing/context limit enforcement

---

## 4. Multi-Model Tokenizer Handling

### Pattern: Model-Aware Tokenizer Registry

**Problem it solves:** Different models use different tokenization schemes (BPE, Unigram, WordPiece).

**Code Example - tiktoken Model Registry:**
```python
import tiktoken

# tiktoken handles model -> encoding mapping automatically
ENCODING_MAP = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}

def get_encoding_for_model(model_name: str):
    """Get correct encoding for any model."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for unknown models
        return tiktoken.get_encoding("cl100k_base")

# Fine-tuned models supported
enc = tiktoken.encoding_for_model("ft:gpt-4o:org:model:id")  # Works!
```

**Code Example - Multi-Tokenizer Factory:**
```python
from abc import ABC, abstractmethod
from typing import List

class TokenizerInterface(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]: ...
    @abstractmethod
    def decode(self, tokens: List[int]) -> str: ...
    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

class TiktokenAdapter(TokenizerInterface):
    def __init__(self, model: str):
        self.enc = tiktoken.encoding_for_model(model)
    
    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

class HuggingFaceAdapter(TokenizerInterface):
    def __init__(self, model_name: str):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_pretrained(model_name)
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

class TokenizerFactory:
    @staticmethod
    def create(model: str) -> TokenizerInterface:
        if model.startswith("gpt-") or model.startswith("ft:gpt"):
            return TiktokenAdapter(model)
        else:
            return HuggingFaceAdapter(model)
```

**Applicability to npu-proxy tokenizer.py:**
- Current implementation is model-agnostic (regex pattern)
- Consider adapter pattern for NPU models that need specific tokenization
- Add model parameter to `count_tokens()` for future multi-model support

---

## 5. Special Tokens Handling

### Pattern: Special Token Management

**Problem it solves:** Correctly handle special tokens ([CLS], [SEP], <|endoftext|>, etc.) during encoding/decoding.

**Code Example - tiktoken Special Tokens:**
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Default: raises error for special tokens
try:
    enc.encode("<|endoftext|>")  # ValueError!
except ValueError as e:
    print(f"Error: {e}")

# Allow specific special tokens
tokens = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

# Allow all special tokens
tokens = enc.encode("<|endoftext|>", allowed_special="all")

# Treat special tokens as regular text
tokens = enc.encode("<|endoftext|>", disallowed_special=())

# Check available special tokens
print(enc.special_tokens_set)  # {'<|endoftext|>', '<|fim_prefix|>', ...}
```

**Code Example - HuggingFace Special Tokens:**
```python
from tokenizers import Tokenizer, AddedToken
from tokenizers.processors import TemplateProcessing

# Add custom special tokens
tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

# Add tokens with options
tokens = [
    AddedToken("<mask>", lstrip=True, rstrip=True),
    AddedToken("<pad>", single_word=True),
]
tokenizer.add_tokens(tokens)

# Configure post-processor for BERT-style
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]"))
    ]
)
```

**Code Example - SentencePiece Special Tokens:**
```python
import sentencepiece as spm

# Train with user-defined symbols
spm.SentencePieceTrainer.train(
    '--input=data.txt '
    '--model_prefix=model '
    '--user_defined_symbols=<sep>,<cls> '
    '--vocab_size=32000'
)

sp = spm.SentencePieceProcessor(model_file='model.model')

# Reserved IDs: <unk>=0, <s>=1, </s>=2, <sep>=3, <cls>=4
print(sp.piece_to_id('<sep>'))  # 3
print(sp.piece_to_id('<cls>'))  # 4

# Special tokens appear in encoded output
print(sp.encode_as_pieces('text<sep>more text'))
```

**Applicability to npu-proxy tokenizer.py:**
- Current implementation doesn't handle special tokens
- Add special token detection before counting
- Consider stripping special tokens from count or counting separately

---

## 6. Truncation Strategies

### Pattern: Smart Truncation with Stride

**Problem it solves:** Handle texts longer than model context limits while preserving important content.

**Code Example - HuggingFace Truncation:**
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# Enable truncation with stride for overlapping chunks
tokenizer.enable_truncation(
    max_length=512,
    stride=128,  # Overlap between chunks
    strategy="longest_first",  # or "only_first", "only_second"
    direction="right"  # or "left"
)

# Access overflowing tokens
output = tokenizer.encode("very long text" * 100)
for overflow in output.overflowing:
    print(f"Overflow chunk: {len(overflow.tokens)} tokens")
```

**Code Example - Manual Truncation Strategies:**
```python
from typing import List, Tuple

def truncate_to_tokens(
    text: str,
    max_tokens: int,
    strategy: str = "tail",
    enc = None
) -> str:
    """
    Truncate text to fit within token limit.
    
    Strategies:
    - "tail": Keep last N tokens (good for context)
    - "head": Keep first N tokens (good for prompts)
    - "middle": Keep head and tail, remove middle
    """
    if enc is None:
        enc = tiktoken.get_encoding("cl100k_base")
    
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    if strategy == "tail":
        return enc.decode(tokens[-max_tokens:])
    elif strategy == "head":
        return enc.decode(tokens[:max_tokens])
    elif strategy == "middle":
        half = max_tokens // 2
        return enc.decode(tokens[:half] + tokens[-half:])
    
    return enc.decode(tokens[:max_tokens])

def chunk_by_tokens(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    enc = None
) -> List[str]:
    """Split text into overlapping token chunks."""
    if enc is None:
        enc = tiktoken.get_encoding("cl100k_base")
    
    tokens = enc.encode(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        
        if end >= len(tokens):
            break
        start = end - overlap
    
    return chunks
```

**Code Example - Sentence-Aware Truncation:**
```python
import re
from typing import List

def truncate_by_sentences(
    text: str,
    max_tokens: int,
    enc = None
) -> Tuple[str, int]:
    """
    Truncate at sentence boundaries to preserve coherence.
    Returns (truncated_text, token_count).
    """
    if enc is None:
        enc = tiktoken.get_encoding("cl100k_base")
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    result = []
    total_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence + " "))
        if total_tokens + sentence_tokens > max_tokens:
            break
        result.append(sentence)
        total_tokens += sentence_tokens
    
    return " ".join(result), total_tokens
```

**Applicability to npu-proxy tokenizer.py:**
- Add truncation utilities for context management
- Implement stride-based chunking for long documents
- Consider sentence-aware truncation for better coherence

---

## 7. Performance Benchmarks

### Pattern: Batch Processing for Performance

**Problem it solves:** Tokenizing many texts efficiently.

**Code Example - tiktoken Batch Processing:**
```python
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

texts = [
    "hello world",
    "goodbye world",
    "python programming",
    "machine learning"
]

# Batch encode with multithreading
encoded_batch = enc.encode_batch(texts, num_threads=4)

# Batch encode ignoring special tokens (faster)
encoded_ordinary = enc.encode_ordinary_batch(texts, num_threads=4)

# Batch decode
decoded_batch = enc.decode_batch(encoded_batch, num_threads=4)

# NumPy integration for ML pipelines
tokens_array = enc.encode_to_numpy("large text")
```

**Code Example - HuggingFace Batch Processing:**
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# Enable padding for batch processing
tokenizer.enable_padding(
    direction="right",
    pad_id=0,
    pad_token="[PAD]",
    length=512
)

# Batch encode
outputs = tokenizer.encode_batch(["Hello world!", "How are you?"])
for output in outputs:
    print(output.tokens, output.attention_mask)
```

**Performance Tips:**
1. **Use batch APIs** - 3-5x faster than individual calls
2. **Use `encode_ordinary`** - Skip special token handling for speed
3. **Set appropriate thread count** - Match CPU cores
4. **Pre-compile regex patterns** - Module-level compilation

---

## 8. Recommendations for npu-proxy

### Current State Analysis

The current `tokenizer.py` implementation:
- ✅ Fast regex-based approximation
- ✅ No external dependencies
- ✅ Safe fallback on error
- ❌ Not model-specific
- ❌ No caching mechanism (beyond compiled regex)
- ❌ No truncation utilities
- ❌ No special token handling

### Recommended Improvements

1. **Add precision parameter:**
```python
def count_tokens(text: str, precision: str = "approximate") -> int:
    """
    Count tokens with configurable precision.
    
    Args:
        text: Input text to count
        precision: "fast" | "approximate" | "exact"
    """
```

2. **Add tokenizer caching (if adding tiktoken):**
```python
from functools import lru_cache

@lru_cache(maxsize=4)
def get_encoding(model: str):
    return tiktoken.encoding_for_model(model)
```

3. **Add truncation utility:**
```python
def truncate_to_limit(text: str, max_tokens: int, strategy: str = "tail") -> str:
    """Truncate text to fit token limit."""
```

4. **Add batch counting:**
```python
def count_tokens_batch(texts: List[str]) -> List[int]:
    """Count tokens for multiple texts efficiently."""
```

### Dependency Consideration

| Option | Pros | Cons |
|--------|------|------|
| Keep regex-only | Zero deps, fast | Not accurate |
| Add tiktoken (optional) | Exact for OpenAI | 5MB dependency |
| Add tokenizers | Fast, multi-model | Larger dependency |

**Recommendation:** Keep regex as default, add optional tiktoken integration with lazy import.

---

## References

- [tiktoken GitHub](https://github.com/openai/tiktoken)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [SentencePiece](https://github.com/google/sentencepiece)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
