"""Citation-boundary chunking strategy.

Chunks text at citation boundaries to preserve citation context."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import re


class CitationChunker:
    """Chunk text at citation boundaries."""

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100) -> None:
        """Initialize citation chunker.

        Args:
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text at citation boundaries.

        Args:
            text: Text to chunk
            metadata: Optional metadata

        Returns:
            List of citation-bounded chunks
        """
        # Find citation boundaries
        boundaries = self._find_citation_boundaries(text)

        # Create chunks
        chunks = []
        start = 0

        for boundary in boundaries:
            chunk_text = text[start:boundary].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    {
                        "text": chunk_text,
                        "start_index": start,
                        "end_index": boundary,
                        "chunking_strategy": "citation_boundary",
                        "metadata": metadata or {},
                    }
                )

            start = boundary

        # Add final chunk
        final_text = text[start:].strip()
        if len(final_text) >= self.min_chunk_size:
            chunks.append(
                {
                    "text": final_text,
                    "start_index": start,
                    "end_index": len(text),
                    "chunking_strategy": "citation_boundary",
                    "metadata": metadata or {},
                }
            )

        return chunks if chunks else [{"text": text, "metadata": metadata or {}}]

    def _find_citation_boundaries(self, text: str) -> List[int]:
        """Find citation boundaries in text.

        Args:
            text: Input text

        Returns:
            List of boundary positions
        """
        boundaries = []

        # Pattern: [Author, Year] or (Author, Year)
        citation_pattern = r"\[([A-Z][a-z]+,?\s+\d{4})\]|\(([A-Z][a-z]+,?\s+\d{4})\)"

        for match in re.finditer(citation_pattern, text):
            boundaries.append(match.end())

        return sorted(boundaries)


def chunk_text(
    text: str, max_size: int = 1000, min_size: int = 100
) -> List[Dict[str, Any]]:
    """Chunk text at citation boundaries.

    Args:
        text: Text to chunk
        max_size: Maximum chunk size
        min_size: Minimum chunk size

    Returns:
        List of chunks
    """
    chunker = CitationChunker(max_size, min_size)
    return chunker.chunk(text)
