"""
Centralized Constants for IngestForge.

This module defines all constants used throughout IngestForge.
Import from here to ensure consistency across the codebase.

Usage
-----
    from ingestforge.core.constants import (
        DEFAULT_CHUNK_SIZE,
        SUPPORTED_EXTENSIONS,
        MODEL_CONTEXT_LENGTHS,
    )

Organization
------------
Constants are grouped by domain:
- Document Processing
- Chunking
- Embeddings
- LLM Configuration
- Storage
- Retrieval
- CLI
- Retry Configuration
- Timeouts
"""

from typing import FrozenSet, Dict, Tuple

# ============================================================================
# Document Processing
# ============================================================================

#: Maximum document size in megabytes before warning
MAX_DOCUMENT_SIZE_MB: int = 100

#: Supported file extensions for ingestion
SUPPORTED_EXTENSIONS: FrozenSet[str] = frozenset(
    {
        # Documents
        ".pdf",
        ".epub",
        ".txt",
        ".md",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".rtf",
        # Web
        ".html",
        ".htm",
        ".mhtml",
        # Images (for OCR)
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".gif",
        ".webp",
        # Code (Salesforce)
        ".cls",
        ".trigger",
        ".apex",
        # Data
        ".json",
        ".jsonl",
        ".csv",
    }
)

#: File extensions that require OCR processing
OCR_EXTENSIONS: FrozenSet[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".gif",
        ".webp",
    }
)

#: File extensions for code files
CODE_EXTENSIONS: FrozenSet[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cls",
        ".trigger",
        ".apex",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
    }
)


# ============================================================================
# Chunking
# ============================================================================

#: Default target chunk size in tokens
DEFAULT_CHUNK_SIZE: int = 500

#: Default chunk overlap in tokens
DEFAULT_CHUNK_OVERLAP: int = 50

#: Minimum chunk size to keep (chunks smaller are merged)
MIN_CHUNK_SIZE: int = 100

#: Maximum chunk size before forced split
MAX_CHUNK_SIZE: int = 2000

#: Quality score threshold for chunk acceptance
MIN_CHUNK_QUALITY_SCORE: float = 0.3


# ============================================================================
# Embeddings
# ============================================================================

#: Default embedding model
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

#: Embedding dimension by model
EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

#: Batch sizes by available VRAM (GB)
VRAM_BATCH_SIZES: Dict[int, int] = {
    4: 16,
    6: 24,
    8: 32,
    12: 48,
    16: 64,
    24: 96,
}

#: Default batch size for CPU processing
CPU_BATCH_SIZE: int = 32


# ============================================================================
# LLM Configuration
# ============================================================================

#: Context window sizes by model
MODEL_CONTEXT_LENGTHS: Dict[str, int] = {
    # Anthropic Claude
    "claude-3-op": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-5-sonnet-20240620": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-op5-20251101": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    # OpenAI GPT
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # Google Gemini
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.0-pro": 32_000,
    "gemini-2.0-flash": 1_000_000,
    # Local models (Ollama)
    "llama3": 8_192,
    "llama3:70b": 8_192,
    "mistral": 32_000,
    "mixtral": 32_000,
    "codellama": 16_000,
}

#: Default LLM provider priority order
#: Local models first; cloud providers are stretch goals for later development
LLM_PROVIDER_PRIORITY: Tuple[str, ...] = (
    "llamacpp",
    "ollama",
    # Stretch goals - cloud providers for future integration
    # "gemini",
    # "claude",
    # "openai",
)


# ============================================================================
# Storage
# ============================================================================

#: Default storage backend
DEFAULT_STORAGE_BACKEND: str = "chromadb"

#: ChromaDB collection name prefix
CHROMADB_COLLECTION_PREFIX: str = "ingestforge_"

#: JSONL compression level (0-9, 0=none, 9=max)
JSONL_COMPRESSION_LEVEL: int = 6


# ============================================================================
# Retrieval
# ============================================================================

#: Default number of results to return
DEFAULT_TOP_K: int = 10

#: Default BM25 weight in hybrid search
DEFAULT_BM25_WEIGHT: float = 0.3

#: Default semantic weight in hybrid search
DEFAULT_SEMANTIC_WEIGHT: float = 0.7

#: RRF constant for rank fusion
RRF_K: int = 60


# ============================================================================
# CLI Symbols
# ============================================================================

#: Unicode symbols for CLI output (with ASCII fallbacks)
SYMBOLS: Dict[str, str] = {
    "check": "\u2713",  # checkmark
    "cross": "\u2717",  # X mark
    "arrow": "\u2192",  # right arrow
    "bullet": "\u2022",  # bullet point
    "warning": "\u26a0",  # warning sign
    "info": "\u2139",  # info sign
    "star": "\u2605",  # star
    "circle": "\u25cf",  # filled circle
    "square": "\u25a0",  # filled square
    "triangle": "\u25b2",  # triangle
}

#: ASCII fallback symbols for terminals without Unicode
SYMBOLS_ASCII: Dict[str, str] = {
    "check": "[OK]",
    "cross": "[X]",
    "arrow": "->",
    "bullet": "*",
    "warning": "[!]",
    "info": "[i]",
    "star": "*",
    "circle": "o",
    "square": "#",
    "triangle": "^",
}


# ============================================================================
# Retry Configuration
# ============================================================================

#: Default retry attempts for transient failures
DEFAULT_RETRY_ATTEMPTS: int = 3

#: Default base delay between retries (seconds)
DEFAULT_RETRY_DELAY: float = 1.0

#: Maximum delay between retries (seconds)
MAX_RETRY_DELAY: float = 60.0

#: Jitter factor for retry delays (0.0-1.0)
RETRY_JITTER: float = 0.1


# ============================================================================
# Timeouts (seconds)
# ============================================================================

#: Default HTTP request timeout
HTTP_TIMEOUT: int = 30

#: Default LLM API timeout
LLM_TIMEOUT: int = 120

#: Default embedding generation timeout
EMBEDDING_TIMEOUT: int = 60

#: Default OCR processing timeout
OCR_TIMEOUT: int = 300
