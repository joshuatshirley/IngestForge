"""Base class for citation commands.

Provides common functionality for citation extraction and formatting.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class CitationCommand(IngestForgeCommand):
    """Base class for citation commands."""

    def get_all_chunks_from_storage(self, storage: Any) -> list[Any]:
        """Retrieve all chunks from storage.

        Args:
            storage: ChunkRepository instance

        Returns:
            List of all chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: self._retrieve_all_chunks(storage),
            "Retrieving chunks from storage...",
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
            # Fallback: search with empty query
            return storage.search("", k=10000)

    def extract_citations_from_chunks(self, chunks: list[Any]) -> List[Dict[str, Any]]:
        """Extract citations from chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of citation dictionaries
        """
        citations = []
        seen = set()

        for chunk in chunks:
            citation_data = self._extract_citation_from_chunk(chunk)

            if citation_data and citation_data["text"]:
                # Deduplicate by citation text
                citation_key = citation_data["text"]

                if citation_key not in seen:
                    seen.add(citation_key)
                    citations.append(citation_data)

        return citations

    def _extract_citation_from_chunk(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """Extract citation from single chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Citation dictionary or None
        """
        metadata = self._get_chunk_metadata(chunk)

        # Try different metadata fields for citation
        citation_text = (
            metadata.get("citation")
            or metadata.get("reference")
            or metadata.get("source")
        )

        if not citation_text:
            return None

        return {
            "text": citation_text,
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page"),
            "chapter": metadata.get("chapter"),
            "metadata": metadata,
        }

    def _get_chunk_metadata(self, chunk: Any) -> Dict[str, Any]:
        """
        Get metadata from chunk.

        Rule #1: Early return pattern eliminates if/elif chain

        Args:
            chunk: Chunk object or dict

        Returns:
            Metadata dictionary
        """
        if isinstance(chunk, dict):
            return chunk.get("metadata", {})
        if hasattr(chunk, "metadata"):
            metadata = chunk.metadata
            if isinstance(metadata, dict):
                return metadata
            return vars(metadata) if metadata else {}

        # Default: empty dict
        return {}

    def group_citations_by_source(
        self, citations: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group citations by source.

        Args:
            citations: List of citations

        Returns:
            Dictionary mapping source to citations
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for citation in citations:
            source = citation["source"]

            if source not in grouped:
                grouped[source] = []

            grouped[source].append(citation)

        return grouped

    def format_citation(self, citation: Dict[str, Any], style: str) -> str:
        """
        Format citation in specified style.

        Rule #1: Dictionary dispatch eliminates if/elif chain
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            citation: Citation dictionary
            style: Citation style (apa/mla/chicago/bibtex)

        Returns:
            Formatted citation string
        """
        assert citation is not None, "Citation cannot be None"
        assert isinstance(citation, dict), "Citation must be dictionary"
        assert "text" in citation, "Citation must have 'text' field"
        assert style is not None, "Style cannot be None"
        assert isinstance(style, str), "Style must be string"
        style_formatters = {
            "apa": self._format_apa,
            "mla": self._format_mla,
            "chicago": self._format_chicago,
            "bibtex": self._format_bibtex,
        }
        style_lower = style.lower()
        formatter = style_formatters.get(style_lower)
        if formatter is None:
            return citation["text"]
        result = formatter(citation)
        assert isinstance(result, str), "Formatter must return string"

        return result

    def _format_apa(self, citation: Dict[str, Any]) -> str:
        """Format citation in APA style.

        Args:
            citation: Citation dictionary

        Returns:
            APA formatted citation
        """
        # Simplified APA format
        parts = [citation["text"]]

        if citation.get("page"):
            parts.append(f"(p. {citation['page']})")

        return " ".join(parts)

    def _format_mla(self, citation: Dict[str, Any]) -> str:
        """Format citation in MLA style.

        Args:
            citation: Citation dictionary

        Returns:
            MLA formatted citation
        """
        # Simplified MLA format
        parts = [citation["text"]]

        if citation.get("page"):
            parts.append(f"({citation['page']})")

        return " ".join(parts)

    def _format_chicago(self, citation: Dict[str, Any]) -> str:
        """Format citation in Chicago style.

        Args:
            citation: Citation dictionary

        Returns:
            Chicago formatted citation
        """
        # Simplified Chicago format
        return citation["text"]

    def _format_bibtex(self, citation: Dict[str, Any]) -> str:
        """Format citation in BibTeX format.

        Args:
            citation: Citation dictionary

        Returns:
            BibTeX formatted citation
        """
        # Simplified BibTeX format
        source = citation["source"].replace(" ", "_")

        lines = [
            f"@article{{{source},",
            f"  title = {{{citation['text']}}},",
        ]

        if citation.get("page"):
            lines.append(f"  pages = {{{citation['page']}}},")

        lines.append("}")

        return "\n".join(lines)
