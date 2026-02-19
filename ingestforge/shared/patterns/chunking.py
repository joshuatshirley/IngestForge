"""
Base Interface for Chunking Strategies.

This module defines the IChunkingStrategy interface - the contract that all
text chunking implementations must follow. Chunking splits extracted text
into semantically meaningful units suitable for retrieval.

Architecture Context
--------------------
Chunking is Stage 3 in the pipeline (Split → Extract → **Chunk** → Enrich → Index).
Chunkers receive raw text and produce ChunkRecords:

    ┌───────────────────────────────────────────────────────────────┐
    │                    Extracted Text                              │
    │  "Chapter 1: Introduction\\n\\nThis book explores..."          │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │     IChunkingStrategy         │
                    │  (semantic/legal/code/fixed)  │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────────┐
        ↓                           ↓                               ↓
    ┌─────────┐               ┌─────────┐                     ┌─────────┐
    │ Chunk 1 │               │ Chunk 2 │         ...         │ Chunk N │
    │ 300 wds │               │ 300 wds │                     │ 250 wds │
    └─────────┘               └─────────┘                     └─────────┘

Available Implementations
-------------------------
- SemanticChunker: Splits on semantic boundaries (paragraphs, sections)
- FixedSizeChunker: Splits at fixed word/character counts
- LegalChunker: Splits legal documents by section numbering
- CodeChunker: Splits code by function/class definitions

Why Chunking Matters
--------------------
1. **Retrieval precision**: Smaller chunks = more precise matches
2. **LLM context limits**: Chunks must fit in model context window
3. **Citation granularity**: Smaller chunks = more precise citations
4. **Semantic coherence**: Chunks should be self-contained units

Trade-offs:
- Too small: Lose context, fragments ideas
- Too large: Dilute relevance, waste context window
- Target: ~300 words with 50-word overlap

Interface Contract
------------------
Implementors must provide:

    chunk(text, document_id)  - Split text into ChunkRecords (required)
    get_strategy_name()       - Return strategy identifier (required)

The base class provides:

    validate_text(text)  - Check if text is suitable for chunking
    estimate_chunks(text) - Estimate output count for progress tracking
    get_config()         - Return config dict for logging

Implementing a Custom Chunker
-----------------------------
    class ParagraphChunker(IChunkingStrategy):
        def chunk(self, text, document_id, source_file="", metadata=None) -> None:
            paragraphs = text.split("\\n\\n")
            return [
                ChunkRecord(
                    chunk_id=f"{document_id}_{i}",
                    content=para,
                    document_id=document_id,
                )
                for i, para in enumerate(paragraphs)
                if para.strip()
            ]

        def get_strategy_name(self) -> None:
            return "paragraph"
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IChunkingStrategy(ABC):
    """Interface for text chunking strategies.

    All chunkers should implement this interface to ensure consistent behavior.
    Different strategies (semantic, legal, code, fixed-size) can be swapped
    without changing the calling code.

    Examples:
        >>> class MyChunker(IChunkingStrategy):
        ...     def chunk(self, text, document_id, source_file="", metadata=None):
        ...         # Split into fixed 1000 char chunks
        ...         chunks = []
        ...         for i in range(0, len(text), 1000):
        ...             chunk_text = text[i:i+1000]
        ...             chunks.append(ChunkRecord(...))
        ...         return chunks
        ...
        ...     def get_strategy_name(self):
        ...         return "fixed-1000"
        ...
        >>> chunker = MyChunker()
        >>> chunks = chunker.chunk("Long text...", "doc-001")
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Split text into chunks.

        Args:
            text: Text to chunk
            document_id: Unique identifier for the source document
            source_file: Path or name of source file
            metadata: Optional metadata to attach to chunks

        Returns:
            List of ChunkRecord objects

        Examples:
            >>> chunker = SemanticChunker(config)
            >>> chunks = chunker.chunk(
            ...     text="Chapter 1\\n\\nThis is content...",
            ...     document_id="book-001",
            ...     source_file="book.pdf",
            ...     metadata={"author": "Smith", "year": 2024}
            ... )
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy.

        Returns:
            Strategy name (e.g., "semantic", "legal", "code", "fixed")

        Examples:
            >>> chunker = SemanticChunker(config)
            >>> chunker.get_strategy_name()
            'semantic'
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get chunker configuration.

        Returns configuration parameters for logging and debugging.

        Returns:
            Dictionary with configuration

        Examples:
            >>> chunker = SemanticChunker(config)
            >>> config = chunker.get_config()
            >>> print(f"Target size: {config['target_size']}")
        """
        return {
            "strategy": self.get_strategy_name(),
            "class": self.__class__.__name__,
        }

    def validate_text(self, text: str) -> bool:
        """Validate that text is suitable for chunking.

        Default implementation checks for non-empty text.
        Subclasses can override for strategy-specific validation.

        Args:
            text: Text to validate

        Returns:
            True if text is valid

        Examples:
            >>> chunker = SemanticChunker(config)
            >>> chunker.validate_text("Some content")
            True
            >>> chunker.validate_text("")
            False
        """
        return bool(text and text.strip())

    def estimate_chunks(self, text: str) -> int:
        """Estimate number of chunks that will be created.

        This is useful for progress tracking and resource allocation.
        Default implementation returns 1. Subclasses should override
        with strategy-specific estimation.

        Args:
            text: Text to estimate

        Returns:
            Estimated number of chunks

        Examples:
            >>> chunker = SemanticChunker(config)
            >>> count = chunker.estimate_chunks(long_text)
            >>> print(f"Will create ~{count} chunks")
        """
        return 1 if text else 0

    def __repr__(self) -> str:
        """String representation of chunker."""
        return f"{self.__class__.__name__}(strategy={self.get_strategy_name()})"


class ChunkingError(Exception):
    """Raised when chunking fails."""

    pass


class ChunkValidator:
    """Validator for chunk quality and consistency.

    Validates chunks to ensure they meet quality standards:
    - Minimum/maximum size requirements
    - Content quality (not just whitespace)
    - Metadata completeness
    """

    def __init__(
        self,
        min_size: int = 50,
        max_size: int = 10000,
        allow_empty: bool = False,
    ):
        """Initialize validator.

        Args:
            min_size: Minimum chunk size in characters
            max_size: Maximum chunk size in characters
            allow_empty: Whether to allow empty chunks
        """
        self.min_size = min_size
        self.max_size = max_size
        self.allow_empty = allow_empty

    def validate(self, chunk: Any) -> bool:
        """Validate a chunk.

        Args:
            chunk: ChunkRecord to validate

        Returns:
            True if chunk is valid

        Examples:
            >>> validator = ChunkValidator(min_size=100, max_size=5000)
            >>> is_valid = validator.validate(chunk)
        """
        # Check content exists
        if not hasattr(chunk, "content"):
            return False

        content = chunk.content
        if not content and not self.allow_empty:
            return False

        # Check size
        content_length = len(content)
        if content_length < self.min_size or content_length > self.max_size:
            return False

        # Check quality (not just whitespace)
        if content.strip() == "":
            return False

        return True

    def filter_valid(self, chunks: List[Any]) -> List[Any]:
        """Filter chunks to only valid ones.

        Args:
            chunks: List of ChunkRecords

        Returns:
            List of valid ChunkRecords

        Examples:
            >>> validator = ChunkValidator()
            >>> valid_chunks = validator.filter_valid(all_chunks)
        """
        return [chunk for chunk in chunks if self.validate(chunk)]

    def get_stats(self, chunks: List[Any]) -> Dict[str, Any]:
        """Get validation statistics.

        Args:
            chunks: List of ChunkRecords

        Returns:
            Dictionary with validation stats

        Examples:
            >>> validator = ChunkValidator()
            >>> stats = validator.get_stats(chunks)
            >>> print(f"Valid: {stats['valid_count']}/{stats['total_count']}")
        """
        total = len(chunks)
        valid = sum(1 for chunk in chunks if self.validate(chunk))

        return {
            "total_count": total,
            "valid_count": valid,
            "invalid_count": total - valid,
            "validation_rate": valid / total if total > 0 else 0,
        }
