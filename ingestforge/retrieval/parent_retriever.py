"""
Parent document retriever.

Fetches parent chunks when child chunks are matched for fuller context.
Uses the parent-child mapping from storage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Set

from ingestforge.storage.base import SearchResult, ChunkRecord
from ingestforge.storage.parent_mapping import ParentMappingStore
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParentExpandedResult:
    """Search result with parent context expansion."""

    # Original child chunk match
    child_result: SearchResult
    # Parent chunk content (if available)
    parent_chunk: Optional[ChunkRecord]
    # Position of child within parent
    child_position: int
    # Total siblings (children of same parent)
    total_siblings: int
    # Truncated parent content (if context window applied)
    truncated_content: Optional[str] = None

    @property
    def expanded_content(self) -> str:
        """Get expanded content (parent if available, else child)."""
        if self.truncated_content:
            return self.truncated_content
        if self.parent_chunk:
            return self.parent_chunk.content
        return self.child_result.content

    @property
    def has_parent(self) -> bool:
        """Check if parent chunk is available."""
        return self.parent_chunk is not None

    @property
    def is_truncated(self) -> bool:
        """Check if content was truncated to fit context window."""
        return self.truncated_content is not None


class ParentRetriever:
    """
    Retriever that expands child chunks to parent chunks.

    This implements the parent document retriever pattern:
    1. Index small chunks for precise matching
    2. Return larger parent chunks for fuller context

    Usage:
        retriever = ParentRetriever(storage, mapping_store)
        results = retriever.search(query, top_k=5)
        expanded = retriever.expand_results(results)

    Alias: ParentDocumentRetriever (for backwards compatibility)
    """

    def __init__(
        self,
        storage: Any,  # ChunkRepository
        mapping_store: ParentMappingStore,
        deduplicate_parents: bool = True,
        context_window_tokens: int = 2048,
        context_window_chars: int = 8000,
    ):
        """
        Initialize parent retriever.

        Args:
            storage: Chunk storage backend
            mapping_store: Parent-child mapping store
            deduplicate_parents: Remove duplicate parents from results
            context_window_tokens: Max tokens for parent context (if tokenizer available)
            context_window_chars: Max characters for parent context (fallback)
        """
        self.storage = storage
        self.mapping_store = mapping_store
        self.deduplicate_parents = deduplicate_parents
        self.context_window_tokens = context_window_tokens
        self.context_window_chars = context_window_chars
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy load tokenizer for token counting."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except ImportError:
                logger.debug("transformers not available, using char-based limits")
        return self._tokenizer

    def _truncate_to_window(self, content: str) -> str:
        """
        Truncate content to fit within context window.

        Args:
            content: Content to truncate

        Returns:
            Truncated content within window limits
        """
        tokenizer = self._get_tokenizer()

        if tokenizer:
            # Token-based truncation
            tokens = tokenizer.encode(content, add_special_tokens=False)
            if len(tokens) > self.context_window_tokens:
                tokens = tokens[: self.context_window_tokens]
                content = tokenizer.decode(tokens)
        else:
            # Character-based fallback
            if len(content) > self.context_window_chars:
                content = content[: self.context_window_chars]
                # Try to break at word boundary
                last_space = content.rfind(" ")
                if last_space > self.context_window_chars * 0.8:
                    content = content[:last_space]

        return content

    def _apply_truncation(self, parent_chunk: Any, apply_window: bool) -> Optional[str]:
        """
        Apply context window truncation to parent chunk.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            parent_chunk: Parent chunk to truncate
            apply_window: Whether to apply truncation

        Returns:
            Truncated content or None if no truncation
        """
        if not apply_window:
            return None

        original_len = len(parent_chunk.content)
        truncated = self._truncate_to_window(parent_chunk.content)
        if len(truncated) >= original_len:
            return None

        logger.debug(f"Truncated parent from {original_len} to {len(truncated)} chars")
        return truncated

    def _fetch_parent_chunk(
        self, mapping: Any, result_chunk_id: str, apply_window: bool
    ) -> tuple:
        """
        Fetch parent chunk and apply truncation.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            mapping: Parent mapping
            result_chunk_id: Child chunk ID
            apply_window: Apply context window truncation

        Returns:
            Tuple of (parent_chunk, truncated_content)
        """
        parent_chunk = self.storage.get_chunk(mapping.parent_chunk_id)
        if not parent_chunk:
            return (None, None)

        logger.debug(f"Expanded {result_chunk_id} to parent {mapping.parent_chunk_id}")

        # Apply truncation
        truncated_content = self._apply_truncation(parent_chunk, apply_window)

        return (parent_chunk, truncated_content)

    def _process_result_mapping(
        self,
        result: SearchResult,
        mapping: Any,
        seen_parents: Set[str],
        apply_window: bool,
    ) -> Optional[tuple]:
        """
        Process result with mapping.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            result: Search result
            mapping: Parent mapping
            seen_parents: Set of seen parent IDs (mutated)
            apply_window: Apply truncation

        Returns:
            Tuple of (parent_chunk, child_position, total_siblings, truncated_content)
            or None if duplicate parent
        """
        if self.deduplicate_parents and mapping.parent_chunk_id in seen_parents:
            return None

        seen_parents.add(mapping.parent_chunk_id)

        # Fetch parent chunk
        parent_chunk, truncated_content = self._fetch_parent_chunk(
            mapping, result.chunk_id, apply_window
        )

        return (
            parent_chunk,
            mapping.child_position,
            mapping.total_children,
            truncated_content,
        )

    def expand_results(
        self,
        results: List[SearchResult],
        apply_window: bool = True,
    ) -> List[ParentExpandedResult]:
        """
        Expand search results to include parent chunks.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            results: Original search results (child chunks)
            apply_window: Apply context window truncation

        Returns:
            List of expanded results with parent context
        """
        if not results:
            return []

        expanded = []
        seen_parents: Set[str] = set()

        for result in results:
            # Get parent mapping
            mapping = self.mapping_store.get_mapping(result.chunk_id)

            # Process mapping if exists
            if mapping:
                result_tuple = self._process_result_mapping(
                    result, mapping, seen_parents, apply_window
                )
                if result_tuple is None:
                    continue

                (
                    parent_chunk,
                    child_position,
                    total_siblings,
                    truncated_content,
                ) = result_tuple
            else:
                parent_chunk = None
                child_position = 0
                total_siblings = 1
                truncated_content = None

            expanded.append(
                ParentExpandedResult(
                    child_result=result,
                    parent_chunk=parent_chunk,
                    child_position=child_position,
                    total_siblings=total_siblings,
                    truncated_content=truncated_content,
                )
            )

        return expanded

    def search_with_expansion(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> List[ParentExpandedResult]:
        """
        Search and automatically expand to parent chunks.

        Args:
            query: Search query
            top_k: Number of results (before deduplication)
            **search_kwargs: Additional arguments for storage.search()

        Returns:
            List of expanded results
        """
        # Search for child chunks
        child_results = self.storage.search(query, top_k=top_k, **search_kwargs)

        # Expand to parents
        return self.expand_results(child_results)


def create_parent_retriever(
    storage: Any,
    data_path: Path,
    config: Any = None,
) -> ParentRetriever:
    """
    Create a parent retriever with default mapping store.

    Args:
        storage: Chunk storage backend
        data_path: Base data directory
        config: Optional Config object for settings

    Returns:
        Configured ParentRetriever
    """
    from ingestforge.storage.parent_mapping import create_parent_mapping_store

    mapping_store = create_parent_mapping_store(data_path)

    # Get settings from config if provided
    deduplicate = True
    context_window_tokens = 2048
    context_window_chars = 8000

    if (
        config
        and hasattr(config, "retrieval")
        and hasattr(config.retrieval, "parent_doc")
    ):
        parent_config = config.retrieval.parent_doc
        deduplicate = getattr(parent_config, "deduplicate_parents", True)
        context_window_tokens = getattr(parent_config, "context_window_tokens", 2048)
        context_window_chars = getattr(parent_config, "context_window_chars", 8000)

    return ParentRetriever(
        storage=storage,
        mapping_store=mapping_store,
        deduplicate_parents=deduplicate,
        context_window_tokens=context_window_tokens,
        context_window_chars=context_window_chars,
    )


# Alias for backwards compatibility
ParentDocumentRetriever = ParentRetriever
