"""Context Aggregator for Multi-Source Synthesis.

Multi-Source Context Aggregation.
Follows NASA JPL Power of Ten rules.

Combines retrieved chunks from multiple documents into a coherent
context window for LLM synthesis while maintaining provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestforge.core.pipeline.artifacts import IFChunkArtifact

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CONTEXT_CHUNKS = 100
MAX_TOKEN_BUDGET = 128000
DEFAULT_TOKEN_BUDGET = 8000
MIN_TOKEN_BUDGET = 100
TOKENS_PER_WORD_ESTIMATE = 1.3
MAX_CONTENT_LENGTH = 50000


@dataclass
class ContextChunk:
    """Individual chunk with citation information.

    GWT-2: Provenance preservation.
    Rule #9: Complete type hints.
    """

    content: str
    artifact_id: str
    document_id: str
    chunk_index: int
    relevance_score: float
    page_number: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        """Generate content hash for deduplication."""
        if not self.content_hash:
            self.content_hash = self._compute_hash(self.content)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute content hash for deduplication."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    @property
    def estimated_tokens(self) -> int:
        """Estimate token count for this chunk."""
        word_count = len(self.content.split())
        return int(word_count * TOKENS_PER_WORD_ESTIMATE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "artifact_id": self.artifact_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "relevance_score": self.relevance_score,
            "page_number": self.page_number,
            "estimated_tokens": self.estimated_tokens,
        }

    def to_citation(self) -> str:
        """Generate citation string for this chunk."""
        citation = f"[{self.document_id}"
        if self.page_number is not None:
            citation += f", p.{self.page_number}"
        citation += f", chunk {self.chunk_index}]"
        return citation


@dataclass
class ContextWindow:
    """Aggregated context window for LLM synthesis.

    GWT-1: Multi-document context assembly.
    GWT-5: Context metadata generation.
    Rule #9: Complete type hints.
    """

    chunks: List[ContextChunk] = field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = DEFAULT_TOKEN_BUDGET
    source_documents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """Number of chunks in context."""
        return len(self.chunks)

    @property
    def document_count(self) -> int:
        """Number of unique source documents."""
        return len(self.source_documents)

    @property
    def budget_utilization(self) -> float:
        """Percentage of token budget used."""
        if self.token_budget <= 0:
            return 0.0
        return min(self.total_tokens / self.token_budget, 1.0)

    @property
    def is_within_budget(self) -> bool:
        """Check if context is within token budget."""
        return self.total_tokens <= self.token_budget

    def to_text(self, include_citations: bool = True) -> str:
        """Convert context window to text for LLM input.

        Args:
            include_citations: Whether to include citation markers.

        Returns:
            Formatted context text.
        """
        parts: List[str] = []

        for chunk in self.chunks:
            if include_citations:
                parts.append(f"{chunk.content} {chunk.to_citation()}")
            else:
                parts.append(chunk.content)

        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "total_tokens": self.total_tokens,
            "token_budget": self.token_budget,
            "source_documents": self.source_documents,
            "chunk_count": self.chunk_count,
            "document_count": self.document_count,
            "budget_utilization": self.budget_utilization,
            "metadata": self.metadata,
        }

    def get_citations(self) -> List[Dict[str, Any]]:
        """Get all citation references for provenance.

        Returns:
            List of citation dictionaries with artifact_id, document_id, etc.
        """
        return [
            {
                "artifact_id": c.artifact_id,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "citation": c.to_citation(),
            }
            for c in self.chunks
        ]


class ContextAggregator:
    """Aggregates retrieved chunks into context windows for synthesis.

    GWT-1: Multi-document context assembly.
    GWT-2: Provenance preservation.
    GWT-3: Token budget enforcement.
    GWT-4: Overlap deduplication.
    GWT-5: Context metadata generation.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        default_token_budget: int = DEFAULT_TOKEN_BUDGET,
        preserve_document_diversity: bool = True,
        min_chunks_per_document: int = 1,
    ) -> None:
        """Initialize the context aggregator.

        Args:
            default_token_budget: Default token budget if not specified.
            preserve_document_diversity: Ensure diverse document representation.
            min_chunks_per_document: Minimum chunks to keep per document.

        Rule #5: Assert preconditions.
        """
        assert (
            MIN_TOKEN_BUDGET <= default_token_budget <= MAX_TOKEN_BUDGET
        ), f"default_token_budget must be between {MIN_TOKEN_BUDGET} and {MAX_TOKEN_BUDGET}"
        assert (
            min_chunks_per_document >= 0
        ), "min_chunks_per_document cannot be negative"

        self._default_budget = default_token_budget
        self._preserve_diversity = preserve_document_diversity
        self._min_chunks_per_doc = min_chunks_per_document

    def aggregate(
        self,
        chunks: List["IFChunkArtifact"],
        token_budget: Optional[int] = None,
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> ContextWindow:
        """Aggregate chunks into a context window.

        GWT-1: Multi-document context assembly.
        Rule #2: MAX_CONTEXT_CHUNKS bound.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            chunks: Retrieved chunks to aggregate.
            token_budget: Maximum token budget.
            relevance_scores: Optional mapping of artifact_id to relevance score.

        Returns:
            ContextWindow with aggregated chunks and metadata.
        """
        assert chunks is not None, "chunks cannot be None"

        # Validate budget BEFORE early return (JPL Rule #5)
        budget = token_budget or self._default_budget
        assert (
            MIN_TOKEN_BUDGET <= budget <= MAX_TOKEN_BUDGET
        ), f"token_budget must be between {MIN_TOKEN_BUDGET} and {MAX_TOKEN_BUDGET}"

        if not chunks:
            return ContextWindow(token_budget=budget)

        # Convert to ContextChunks with scores
        context_chunks = self._convert_chunks(chunks, relevance_scores)

        # Deduplicate by content
        unique_chunks = self._deduplicate_chunks(context_chunks)

        # Sort by relevance
        sorted_chunks = sorted(
            unique_chunks, key=lambda c: c.relevance_score, reverse=True
        )

        # Apply bounds
        bounded_chunks = sorted_chunks[:MAX_CONTEXT_CHUNKS]

        # Prune to token budget
        final_chunks = self._prune_to_budget(bounded_chunks, budget)

        # Build context window
        return self._build_context_window(final_chunks, budget)

    def aggregate_from_dicts(
        self,
        chunk_dicts: List[Dict[str, Any]],
        token_budget: Optional[int] = None,
    ) -> ContextWindow:
        """Aggregate from dictionary representations.

        Convenience method for working with serialized chunk data.

        Args:
            chunk_dicts: List of chunk dictionaries.
            token_budget: Maximum token budget.

        Returns:
            ContextWindow with aggregated chunks.
        """
        assert chunk_dicts is not None, "chunk_dicts cannot be None"

        if not chunk_dicts:
            return ContextWindow(token_budget=token_budget or self._default_budget)

        # Convert dicts to ContextChunks
        context_chunks: List[ContextChunk] = []
        for d in chunk_dicts[:MAX_CONTEXT_CHUNKS]:
            chunk = ContextChunk(
                content=str(d.get("content", ""))[:MAX_CONTENT_LENGTH],
                artifact_id=str(d.get("artifact_id", d.get("chunk_id", "unknown"))),
                document_id=str(d.get("document_id", "unknown")),
                chunk_index=int(d.get("chunk_index", 0)),
                relevance_score=float(d.get("relevance_score", d.get("score", 1.0))),
                page_number=d.get("page_number"),
            )
            context_chunks.append(chunk)

        # Deduplicate and prune
        unique_chunks = self._deduplicate_chunks(context_chunks)
        sorted_chunks = sorted(
            unique_chunks, key=lambda c: c.relevance_score, reverse=True
        )
        budget = token_budget or self._default_budget
        final_chunks = self._prune_to_budget(sorted_chunks, budget)

        return self._build_context_window(final_chunks, budget)

    def _convert_chunks(
        self,
        chunks: List["IFChunkArtifact"],
        relevance_scores: Optional[Dict[str, float]],
    ) -> List[ContextChunk]:
        """Convert IFChunkArtifacts to ContextChunks.

        Rule #4: Function < 60 lines.
        """
        context_chunks: List[ContextChunk] = []
        default_score = 1.0

        for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS]):
            score = default_score
            if relevance_scores and chunk.artifact_id in relevance_scores:
                score = relevance_scores[chunk.artifact_id]

            # Extract page number from metadata if available
            page_num = chunk.metadata.get("page_number")
            if page_num is None:
                page_num = chunk.metadata.get("page")

            context_chunk = ContextChunk(
                content=chunk.content[:MAX_CONTENT_LENGTH],
                artifact_id=chunk.artifact_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                relevance_score=score,
                page_number=page_num,
                char_start=chunk.metadata.get("char_start"),
                char_end=chunk.metadata.get("char_end"),
            )
            context_chunks.append(context_chunk)

        return context_chunks

    def _deduplicate_chunks(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Remove duplicate chunks by content hash.

        GWT-4: Overlap deduplication.
        Rule #4: Function < 60 lines.
        """
        seen_hashes: Set[str] = set()
        unique_chunks: List[ContextChunk] = []

        for chunk in chunks:
            if chunk.content_hash not in seen_hashes:
                seen_hashes.add(chunk.content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    def _prune_to_budget(
        self,
        chunks: List[ContextChunk],
        budget: int,
    ) -> List[ContextChunk]:
        """Prune chunks to fit within token budget.

        GWT-3: Token budget enforcement.
        Rule #4: Function < 60 lines.
        """
        if not chunks:
            return []

        # If diversity preservation is enabled, ensure min chunks per doc
        if self._preserve_diversity and self._min_chunks_per_doc > 0:
            return self._prune_with_diversity(chunks, budget)

        # Simple greedy pruning by relevance
        selected: List[ContextChunk] = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = chunk.estimated_tokens
            if total_tokens + chunk_tokens <= budget:
                selected.append(chunk)
                total_tokens += chunk_tokens
            elif not selected:
                # Always include at least one chunk
                selected.append(chunk)
                break

        return selected

    def _prune_with_diversity(
        self,
        chunks: List[ContextChunk],
        budget: int,
    ) -> List[ContextChunk]:
        """Prune while preserving document diversity.

        Rule #4: Function < 60 lines.
        """
        # Group chunks by document
        doc_chunks: Dict[str, List[ContextChunk]] = {}
        for chunk in chunks:
            if chunk.document_id not in doc_chunks:
                doc_chunks[chunk.document_id] = []
            doc_chunks[chunk.document_id].append(chunk)

        # First pass: ensure minimum per document
        selected: List[ContextChunk] = []
        total_tokens = 0

        for doc_id, doc_chunk_list in doc_chunks.items():
            for chunk in doc_chunk_list[: self._min_chunks_per_doc]:
                if total_tokens + chunk.estimated_tokens <= budget:
                    selected.append(chunk)
                    total_tokens += chunk.estimated_tokens

        # Second pass: add remaining by relevance
        selected_ids = {c.artifact_id for c in selected}
        remaining = [c for c in chunks if c.artifact_id not in selected_ids]

        for chunk in remaining:
            if total_tokens + chunk.estimated_tokens <= budget:
                selected.append(chunk)
                total_tokens += chunk.estimated_tokens

        # Ensure at least one chunk is included (GWT-3 guarantee)
        if not selected and chunks:
            selected.append(chunks[0])

        # Sort by relevance for final ordering
        return sorted(selected, key=lambda c: c.relevance_score, reverse=True)

    def _build_context_window(
        self,
        chunks: List[ContextChunk],
        budget: int,
    ) -> ContextWindow:
        """Build final context window with metadata.

        GWT-5: Context metadata generation.
        Rule #4: Function < 60 lines.
        """
        # Calculate total tokens
        total_tokens = sum(c.estimated_tokens for c in chunks)

        # Get unique document IDs
        doc_ids = sorted(set(c.document_id for c in chunks))

        # Build metadata
        metadata: Dict[str, Any] = {
            "aggregation_method": "relevance_sorted",
            "diversity_preserved": self._preserve_diversity,
            "deduplication_applied": True,
        }

        return ContextWindow(
            chunks=chunks,
            total_tokens=total_tokens,
            token_budget=budget,
            source_documents=doc_ids,
            metadata=metadata,
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text.

        Uses word count * 1.3 approximation.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        word_count = len(text.split())
        return int(word_count * TOKENS_PER_WORD_ESTIMATE)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def aggregate_context(
    chunks: List["IFChunkArtifact"],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    relevance_scores: Optional[Dict[str, float]] = None,
) -> ContextWindow:
    """Convenience function to aggregate chunks into context.

    Args:
        chunks: Retrieved chunks to aggregate.
        token_budget: Maximum token budget.
        relevance_scores: Optional mapping of artifact_id to relevance score.

    Returns:
        ContextWindow with aggregated chunks.
    """
    aggregator = ContextAggregator(default_token_budget=token_budget)
    return aggregator.aggregate(chunks, token_budget, relevance_scores)


def create_context_aggregator(
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    preserve_diversity: bool = True,
) -> ContextAggregator:
    """Factory function to create a configured aggregator.

    Args:
        token_budget: Default token budget.
        preserve_diversity: Whether to preserve document diversity.

    Returns:
        Configured ContextAggregator.
    """
    return ContextAggregator(
        default_token_budget=token_budget,
        preserve_document_diversity=preserve_diversity,
    )
