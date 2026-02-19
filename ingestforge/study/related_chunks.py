"""Semantic context linker for study materials (QUIZ-002.2).

This module provides "Related" button functionality for quizzes,
retrieving similar source passages from the vector store.

NASA JPL Commandments compliance:
- Rule #1: No deep nesting, early returns
- Rule #2: Fixed iteration bounds
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.study.related_chunks import RelatedChunksLinker

    linker = RelatedChunksLinker(storage)
    related = linker.find_related(
        question="What is the capital of France?",
        answer="Paris",
        max_results=3,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from ingestforge.core.logging import get_logger

if TYPE_CHECKING:
    from ingestforge.storage.base import BaseStorage

logger = get_logger(__name__)

# Default limits (Rule #2: Fixed bounds)
DEFAULT_MAX_RESULTS: int = 3
MAX_ALLOWED_RESULTS: int = 10

# Minimum similarity threshold
DEFAULT_SIMILARITY_THRESHOLD: float = 0.3


@dataclass
class RelatedChunk:
    """A related source passage.

    Attributes:
        text: The chunk text content
        source: Source document name/path
        similarity: Similarity score (0-1)
        chunk_id: Unique chunk identifier
        metadata: Additional chunk metadata
    """

    text: str
    source: str
    similarity: float
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def source_display(self) -> str:
        """Get display-friendly source name."""
        if not self.source:
            return "Unknown Source"

        # Extract filename from path
        parts = self.source.replace("\\", "/").split("/")
        return parts[-1] if parts else self.source

    def truncate(self, max_length: int = 200) -> str:
        """Get truncated text with ellipsis.

        Args:
            max_length: Maximum text length

        Returns:
            Truncated text
        """
        if len(self.text) <= max_length:
            return self.text

        return self.text[: max_length - 3].rstrip() + "..."


@dataclass
class RelatedChunksResult:
    """Result of finding related chunks.

    Attributes:
        chunks: List of related chunks
        query_text: The query that was used
        total_searched: Number of chunks searched
    """

    chunks: List[RelatedChunk] = field(default_factory=list)
    query_text: str = ""
    total_searched: int = 0

    @property
    def count(self) -> int:
        """Number of related chunks found."""
        return len(self.chunks)

    @property
    def has_results(self) -> bool:
        """Whether any related chunks were found."""
        return len(self.chunks) > 0

    @property
    def avg_similarity(self) -> float:
        """Average similarity score."""
        if not self.chunks:
            return 0.0
        return sum(c.similarity for c in self.chunks) / len(self.chunks)


class RelatedChunksLinker:
    """Finds related source passages for study questions.

    This class provides the "Related" button functionality for quizzes,
    retrieving similar passages from the vector store based on semantic
    similarity to the question and answer.

    Args:
        storage: Storage backend with vector search capability
        similarity_threshold: Minimum similarity score (0-1)
        max_results: Maximum results to return (Rule #2 bound)

    Example:
        linker = RelatedChunksLinker(storage)

        result = linker.find_related(
            question="What causes photosynthesis?",
            answer="Light energy converts CO2 and water",
        )

        for chunk in result.chunks:
            print(f"[{chunk.similarity:.0%}] {chunk.truncate(100)}")
            print(f"  From: {chunk.source_display}")
    """

    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        """Initialize the related chunks linker."""
        if max_results > MAX_ALLOWED_RESULTS:
            raise ValueError(
                f"max_results cannot exceed {MAX_ALLOWED_RESULTS} (Rule #2)"
            )
        if max_results <= 0:
            raise ValueError("max_results must be positive")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self._storage = storage
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results

    @property
    def storage(self) -> Optional[BaseStorage]:
        """Get the storage backend."""
        return self._storage

    @storage.setter
    def storage(self, value: BaseStorage) -> None:
        """Set the storage backend."""
        self._storage = value

    @property
    def is_available(self) -> bool:
        """Check if linker can search."""
        return self._storage is not None

    def find_related(
        self,
        question: str,
        answer: Optional[str] = None,
        max_results: Optional[int] = None,
        include_answer_in_query: bool = True,
    ) -> RelatedChunksResult:
        """Find related source passages.

        Args:
            question: The question text
            answer: Optional answer to include in search
            max_results: Override default max results
            include_answer_in_query: Whether to include answer in search

        Returns:
            RelatedChunksResult with matching chunks
        """
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return RelatedChunksResult(query_text="")
        if not self.is_available:
            logger.warning("No storage configured for related chunks")
            return RelatedChunksResult(query_text=question)

        # Build query text
        query = self._build_query(question, answer, include_answer_in_query)

        # Apply result limit (Rule #2)
        limit = min(
            max_results or self.max_results,
            MAX_ALLOWED_RESULTS,
        )

        # Perform search
        try:
            return self._search_chunks(query, limit)
        except Exception as e:
            logger.error(f"Failed to search related chunks: {e}")
            return RelatedChunksResult(query_text=query)

    def find_related_to_text(
        self,
        text: str,
        max_results: Optional[int] = None,
    ) -> RelatedChunksResult:
        """Find chunks related to arbitrary text.

        Args:
            text: Text to find related content for
            max_results: Override default max results

        Returns:
            RelatedChunksResult with matching chunks
        """
        return self.find_related(
            question=text,
            answer=None,
            max_results=max_results,
            include_answer_in_query=False,
        )

    def _build_query(
        self,
        question: str,
        answer: Optional[str],
        include_answer: bool,
    ) -> str:
        """Build the search query text.

        Args:
            question: The question text
            answer: Optional answer text
            include_answer: Whether to include answer

        Returns:
            Combined query string
        """
        query = question.strip()

        if include_answer and answer and answer.strip():
            query = f"{query} {answer.strip()}"

        return query

    def _search_chunks(
        self,
        query: str,
        limit: int,
    ) -> RelatedChunksResult:
        """Search for related chunks in storage.

        Args:
            query: Search query text
            limit: Maximum results

        Returns:
            RelatedChunksResult with found chunks
        """
        assert self._storage is not None, "Storage should be set"

        # Perform vector search
        # Request more than needed to allow filtering by threshold
        search_limit = min(limit * 2, MAX_ALLOWED_RESULTS * 2)

        results = self._storage.search(
            query=query,
            k=search_limit,
        )

        # Convert to RelatedChunk objects with filtering
        chunks: List[RelatedChunk] = []

        for result in results:
            if len(chunks) >= limit:
                break

            # Extract similarity (default to 0 if not provided)
            similarity = self._extract_similarity(result)

            # Filter by threshold
            if similarity < self.similarity_threshold:
                continue

            chunk = self._convert_result(result, similarity)
            chunks.append(chunk)

        return RelatedChunksResult(
            chunks=chunks,
            query_text=query,
            total_searched=len(results),
        )

    def _extract_similarity(self, result: dict) -> float:
        """Extract similarity score from search result.

        Args:
            result: Search result dict

        Returns:
            Similarity score (0-1)
        """
        # Try different field names used by various backends
        for field_name in ["similarity", "score", "distance"]:
            if field_name in result:
                value = result[field_name]

                # Distance needs to be inverted
                if field_name == "distance":
                    # Assume cosine distance (0-2 range)
                    return max(0.0, 1.0 - (value / 2.0))

                return float(value)

        # Default similarity if not provided
        return 0.5

    def _convert_result(
        self,
        result: dict,
        similarity: float,
    ) -> RelatedChunk:
        """Convert search result to RelatedChunk.

        Args:
            result: Raw search result
            similarity: Computed similarity score

        Returns:
            RelatedChunk object
        """
        # Extract text content
        text = result.get("text", result.get("content", ""))

        # Extract source
        source = result.get("source", result.get("metadata", {}).get("source", ""))

        # Extract chunk ID
        chunk_id = result.get("id", result.get("chunk_id", ""))

        # Extract metadata
        metadata = result.get("metadata", {})

        return RelatedChunk(
            text=text,
            source=source,
            similarity=similarity,
            chunk_id=str(chunk_id),
            metadata=metadata,
        )


def find_related_chunks(
    question: str,
    answer: Optional[str] = None,
    storage: Optional[BaseStorage] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[RelatedChunk]:
    """Convenience function to find related chunks.

    Args:
        question: Question text
        answer: Optional answer text
        storage: Storage backend
        max_results: Maximum results

    Returns:
        List of RelatedChunk objects
    """
    linker = RelatedChunksLinker(
        storage=storage,
        max_results=max_results,
    )
    result = linker.find_related(question, answer)
    return result.chunks
