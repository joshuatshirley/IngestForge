"""
Text Processing Utilities.

This module provides centralized text cleaning and normalization functions
used throughout the ingestion pipeline. By consolidating these operations here,
we avoid duplication and ensure consistent text handling.

Architecture Context
--------------------
Text utilities are used at multiple pipeline stages:

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Extraction    │────→│    Chunking     │────→│   Enrichment    │
    │  clean_text()   │     │ normalize_ws()  │     │  clean_text()   │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

Common use cases:
- PDFProcessor: Clean extracted PDF text (extra whitespace, artifacts)
- HTMLProcessor: Clean HTML-to-text output (normalize newlines)
- SemanticChunker: Normalize text before boundary detection
- EntityExtractor: Clean text before NER analysis

Functions
---------
**clean_text(text, remove_urls=False)**
    General-purpose cleaning: reduce multiple newlines/spaces to reasonable amounts.
    Preserves paragraph structure (keeps double newlines).

**normalize_whitespace(text)**
    Aggressive cleaning: all whitespace becomes single spaces.
    Good for comparison, embedding, or when structure doesn't matter.

**read_text_with_fallback(file_path)**
    Read text file trying multiple encodings (UTF-8, Latin-1, etc.).
    Handles messy real-world files that may have encoding issues.

**split_into_sentences(text)**
    Basic sentence splitting using regex heuristics.
    For production NLP, consider spaCy or NLTK instead.

**truncate_text(text, max_length)**
    Truncate text with suffix ("...") for previews and logging.

Design Decisions
----------------
1. **Regex-based**: No external dependencies (no NLTK/spaCy required).
2. **Non-destructive defaults**: clean_text preserves paragraph structure.
3. **Encoding fallbacks**: Graceful degradation on encoding errors.
"""

import re
from pathlib import Path
from typing import List


def clean_text(text: str, remove_urls: bool = False) -> str:
    """Clean and normalize text by removing excessive whitespace.

    Args:
        text: The text to clean
        remove_urls: If True, remove HTTP(S) URLs from the text

    Returns:
        Cleaned text with normalized whitespace

    Examples:
        >>> clean_text("Hello\\n\\n\\nWorld")
        'Hello\\n\\nWorld'
        >>> clean_text("Multiple    spaces")
        'Multiple spaces'
        >>> clean_text("Link: https://example.com here", remove_urls=True)
        'Link:  here'
    """
    # Reduce multiple newlines to maximum of 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Reduce multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)

    # Optionally remove URLs
    if remove_urls:
        text = re.sub(r"https?://\S+", "", text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces.

    This is more aggressive than clean_text - it replaces ALL whitespace
    (including newlines, tabs) with single spaces.

    Args:
        text: The text to normalize

    Returns:
        Text with all whitespace normalized to single spaces

    Examples:
        >>> normalize_whitespace("Hello\\n\\tWorld")
        'Hello World'
        >>> normalize_whitespace("Multiple    spaces\\n\\nand lines")
        'Multiple spaces and lines'
    """
    return re.sub(r"\s+", " ", text).strip()


def read_text_with_fallback(file_path: Path) -> str:
    """Read text file trying multiple encodings.

    Attempts to read the file with multiple common encodings, falling back
    to ignore errors if all encodings fail.

    Args:
        file_path: Path to the text file

    Returns:
        The file contents as a string

    Raises:
        FileNotFoundError: If the file does not exist

    Examples:
        >>> from pathlib import Path
        >>> content = read_text_with_fallback(Path("document.txt"))
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: read as bytes and decode with error handling
    return file_path.read_bytes().decode("utf-8", errors="ignore")


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics.

    This is a basic sentence splitter that handles common cases.
    For more sophisticated splitting, consider using spaCy or NLTK.

    Args:
        text: The text to split

    Returns:
        List of sentences

    Examples:
        >>> split_into_sentences("Hello world. How are you?")
        ['Hello world.', 'How are you?']
    """
    # Basic sentence splitting on .!? followed by space and capital letter
    # This is intentionally simple - more complex rules can be added as needed
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length, adding suffix if truncated.

    Args:
        text: The text to truncate
        max_length: Maximum length including suffix
        suffix: String to append if truncated (default: "...")

    Returns:
        Truncated text

    Examples:
        >>> truncate_text("This is a long sentence", 10)
        'This is...'
        >>> truncate_text("Short", 10)
        'Short'
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix
