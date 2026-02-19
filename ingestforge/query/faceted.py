"""Faceted search functionality.

Enables filtering search results by metadata facets (date, author, type, topic, etc.)."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime


class FacetedSearch:
    """Faceted search with metadata filtering."""

    def __init__(self, storage: Any) -> None:
        """Initialize faceted search.

        Args:
            storage: ChunkRepository instance
        """
        self.storage = storage

    def search(
        self,
        query: str,
        facets: Optional[Dict[str, Any]] = None,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Perform faceted search.

        Args:
            query: Search query
            facets: Facet filters (e.g., {'type': 'pdf', 'author': 'Smith'})
            k: Number of results

        Returns:
            Search results with facet information
        """
        # Get initial results
        all_chunks = self.storage.search(query, k=k * 5)  # Get more for filtering

        # Apply facet filters
        if facets:
            filtered_chunks = self._apply_facet_filters(all_chunks, facets)
        else:
            filtered_chunks = all_chunks

        # Limit to k results
        results = filtered_chunks[:k]

        # Extract facets from all results
        available_facets = self._extract_facets(all_chunks)

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "facets": available_facets,
            "applied_filters": facets or {},
        }

    def _apply_facet_filters(
        self, chunks: List[Any], facets: Dict[str, Any]
    ) -> List[Any]:
        """Apply facet filters to chunks.

        Args:
            chunks: List of chunks
            facets: Facet filters

        Returns:
            Filtered chunks
        """
        filtered = chunks

        for facet_name, facet_value in facets.items():
            filtered = self._filter_by_facet(filtered, facet_name, facet_value)

        return filtered

    def _filter_by_facet(
        self, chunks: List[Any], facet_name: str, facet_value: Any
    ) -> List[Any]:
        """Filter chunks by single facet.

        Args:
            chunks: List of chunks
            facet_name: Facet name
            facet_value: Facet value

        Returns:
            Filtered chunks
        """
        filtered = []

        for chunk in chunks:
            if self._chunk_matches_facet(chunk, facet_name, facet_value):
                filtered.append(chunk)

        return filtered

    def _chunk_matches_facet(
        self, chunk: Any, facet_name: str, facet_value: Any
    ) -> bool:
        """Check if chunk matches facet filter.

        Args:
            chunk: Chunk to check
            facet_name: Facet name
            facet_value: Facet value

        Returns:
            True if matches
        """
        metadata = self._get_metadata(chunk)

        if facet_name == "date_range":
            return self._matches_date_range(metadata, facet_value)

        if facet_name not in metadata:
            return False

        chunk_value = metadata[facet_name]

        # Handle list values
        if isinstance(chunk_value, list):
            return bool(facet_value in chunk_value)

        return bool(chunk_value == facet_value)

    def _matches_date_range(
        self, metadata: Dict[str, Any], date_range: Dict[str, str]
    ) -> bool:
        """Check if metadata matches date range.

        Args:
            metadata: Chunk metadata
            date_range: Date range dict with 'start' and 'end'

        Returns:
            True if matches
        """
        chunk_date = metadata.get("date")
        if not chunk_date:
            return False

        try:
            # Parse dates
            chunk_dt = datetime.fromisoformat(str(chunk_date))
            start = datetime.fromisoformat(date_range.get("start", "1900-01-01"))
            end = datetime.fromisoformat(date_range.get("end", "2100-12-31"))

            return start <= chunk_dt <= end

        except (ValueError, TypeError):
            return False

    def _extract_facets(self, chunks: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract available facets from chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary of facets with counts
        """
        facets: Dict[str, Dict[Any, int]] = {}

        for chunk in chunks:
            metadata = self._get_metadata(chunk)
            self._count_metadata_facets(metadata, facets)

        return self._format_facet_counts(facets)

    def _count_metadata_facets(
        self, metadata: Dict[str, Any], facets: Dict[str, Dict[Any, int]]
    ) -> None:
        """Count facets from metadata.

        Args:
            metadata: Chunk metadata
            facets: Facets dictionary to update
        """
        for key, value in metadata.items():
            if key not in facets:
                facets[key] = {}

            # Handle list values
            if isinstance(value, list):
                for item in value:
                    facets[key][item] = facets[key].get(item, 0) + 1
            else:
                facets[key][value] = facets[key].get(value, 0) + 1

    def _format_facet_counts(
        self, facets: Dict[str, Dict[Any, int]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Format facet counts for output.

        Args:
            facets: Raw facet counts

        Returns:
            Formatted facets
        """
        formatted = {}
        for facet_name, value_counts in facets.items():
            formatted[facet_name] = [
                {"value": value, "count": count}
                for value, count in sorted(
                    value_counts.items(), key=lambda x: x[1], reverse=True
                )[:20]  # Top 20 values per facet
            ]
        return formatted

    def _get_metadata(self, chunk: Any) -> Dict[str, Any]:
        """Extract metadata from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Metadata dictionary
        """
        if isinstance(chunk, dict):
            result: dict[str, Any] = chunk.get("metadata", {})
            return result
        elif hasattr(chunk, "metadata"):
            metadata: dict[str, Any] = chunk.metadata
            return metadata
        else:
            return {}


def faceted_search(
    storage: Any,
    query: str,
    facets: Optional[Dict[str, Any]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """Perform faceted search.

    Args:
        storage: ChunkRepository instance
        query: Search query
        facets: Facet filters
        k: Number of results

    Returns:
        Search results with facets
    """
    searcher = FacetedSearch(storage)
    return searcher.search(query, facets, k)
