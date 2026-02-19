"""Base class for research commands.

Provides common functionality for research and verification tools.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class ResearchCommand(IngestForgeCommand):
    """Base class for research commands."""

    def analyze_chunks_metadata(self, chunks: list) -> Dict[str, Any]:
        """Analyze metadata distribution in chunks.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with metadata statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "unique_sources": 0,
                "avg_chunk_size": 0,
                "sources": [],
            }

        # Calculate statistics (Commandment #4: Small function)
        total_chunks = len(chunks)
        sources = self._extract_unique_sources(chunks)
        avg_size = self._calculate_average_chunk_size(chunks)

        return {
            "total_chunks": total_chunks,
            "unique_sources": len(sources),
            "avg_chunk_size": avg_size,
            "sources": sources,
        }

    def _extract_unique_sources(self, chunks: list) -> List[str]:
        """Extract unique source documents from chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of unique source identifiers
        """
        sources: set[str] = set()

        for chunk in chunks:
            # Handle different chunk formats (Commandment #7: Check parameters)
            if isinstance(chunk, dict):
                source = chunk.get("metadata", {}).get("source", "unknown")
            elif hasattr(chunk, "metadata"):
                source = getattr(chunk.metadata, "source", "unknown")
            else:
                source = "unknown"

            sources.add(source)

        return sorted(sources)

    def _calculate_average_chunk_size(self, chunks: list) -> float:
        """Calculate average chunk size.

        Args:
            chunks: List of chunks

        Returns:
            Average chunk size in characters
        """
        if not chunks:
            return 0.0

        total_size = 0

        for chunk in chunks:
            # Handle different chunk formats
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            elif hasattr(chunk, "text"):
                text = chunk.text
            else:
                text = str(chunk)

            total_size += len(text)

        return total_size / len(chunks)

    def get_all_chunks_from_storage(self, storage: Any) -> list[Any]:
        """Retrieve all chunks from storage.

        Args:
            storage: ChunkRepository instance

        Returns:
            List of all chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: self._retrieve_all_chunks(storage),
            "Retrieving all chunks from storage...",
            "Chunks retrieved",
        )

    def _retrieve_all_chunks(self, storage: Any) -> list[Any]:
        """Retrieve all chunks (internal helper).

        Args:
            storage: ChunkRepository instance

        Returns:
            List of chunks
        """
        # Different storage backends have different APIs
        if hasattr(storage, "get_all_chunks"):
            return storage.get_all_chunks()
        elif hasattr(storage, "list_all"):
            return storage.list_all()
        else:
            # Fallback: search with empty query to get all
            return storage.search("", k=10000)

    def validate_citation_format(self, citation: str) -> bool:
        """Validate citation string format.

        Args:
            citation: Citation string to validate

        Returns:
            True if valid format
        """
        if not citation or not isinstance(citation, str):
            return False

        # Basic validation: citation should have minimum structure
        citation = citation.strip()

        # Must be non-empty and have reasonable length
        if len(citation) < 10 or len(citation) > 1000:
            return False

        # Should contain at least some alphanumeric content
        if not any(c.isalnum() for c in citation):
            return False

        return True

    def extract_citations_from_chunks(self, chunks: list[Any]) -> Dict[str, List[str]]:
        """Extract citations from chunk metadata.

        Args:
            chunks: List of chunks with metadata

        Returns:
            Dictionary mapping source to citations
        """
        citations: Dict[str, List[str]] = {}

        for chunk in chunks:
            source = self._get_chunk_source(chunk)
            citation = self._get_chunk_citation(chunk)

            if citation and self.validate_citation_format(citation):
                if source not in citations:
                    citations[source] = []
                if citation not in citations[source]:
                    citations[source].append(citation)

        return citations

    def _get_chunk_source(self, chunk: Any) -> str:
        """Get source from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Source identifier
        """
        if isinstance(chunk, dict):
            return chunk.get("metadata", {}).get("source", "unknown")
        elif hasattr(chunk, "metadata"):
            return getattr(chunk.metadata, "source", "unknown")
        else:
            return "unknown"

    def _get_chunk_citation(self, chunk: Any) -> Optional[str]:
        """Get citation from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Citation string or None
        """
        if isinstance(chunk, dict):
            return chunk.get("metadata", {}).get("citation")
        elif hasattr(chunk, "metadata"):
            return getattr(chunk.metadata, "citation", None)
        else:
            return None
