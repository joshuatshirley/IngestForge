"""Tests for Context Aggregator.

Multi-Source Context Aggregation implementation tests.
Tests context assembly, provenance, token budgets, and deduplication."""

from __future__ import annotations


import pytest

from ingestforge.query.context_aggregator import (
    ContextAggregator,
    ContextChunk,
    ContextWindow,
    aggregate_context,
    create_context_aggregator,
    MAX_CONTEXT_CHUNKS,
    MAX_TOKEN_BUDGET,
    DEFAULT_TOKEN_BUDGET,
    TOKENS_PER_WORD_ESTIMATE,
)
from ingestforge.core.pipeline.artifacts import IFChunkArtifact


# ---------------------------------------------------------------------------
# ContextChunk Tests
# ---------------------------------------------------------------------------


class TestContextChunk:
    """Tests for ContextChunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test creating a context chunk."""
        chunk = ContextChunk(
            content="Hello world.",
            artifact_id="art-001",
            document_id="doc-001",
            chunk_index=0,
            relevance_score=0.95,
        )
        assert chunk.content == "Hello world."
        assert chunk.artifact_id == "art-001"
        assert chunk.relevance_score == 0.95

    def test_content_hash_generated(self) -> None:
        """Test that content hash is auto-generated."""
        chunk = ContextChunk(
            content="Test content here.",
            artifact_id="a1",
            document_id="d1",
            chunk_index=0,
            relevance_score=1.0,
        )
        assert chunk.content_hash != ""
        assert len(chunk.content_hash) == 16

    def test_same_content_same_hash(self) -> None:
        """Test that same content produces same hash."""
        chunk1 = ContextChunk(
            content="Same content",
            artifact_id="a1",
            document_id="d1",
            chunk_index=0,
            relevance_score=1.0,
        )
        chunk2 = ContextChunk(
            content="Same content",
            artifact_id="a2",
            document_id="d2",
            chunk_index=1,
            relevance_score=0.5,
        )
        assert chunk1.content_hash == chunk2.content_hash

    def test_estimated_tokens(self) -> None:
        """Test token estimation."""
        chunk = ContextChunk(
            content="One two three four five",  # 5 words
            artifact_id="a1",
            document_id="d1",
            chunk_index=0,
            relevance_score=1.0,
        )
        expected = int(5 * TOKENS_PER_WORD_ESTIMATE)
        assert chunk.estimated_tokens == expected

    def test_to_citation(self) -> None:
        """Test citation string generation."""
        chunk = ContextChunk(
            content="Content",
            artifact_id="a1",
            document_id="doc-001",
            chunk_index=5,
            relevance_score=1.0,
            page_number=42,
        )
        citation = chunk.to_citation()
        assert "doc-001" in citation
        assert "p.42" in citation
        assert "chunk 5" in citation

    def test_to_citation_no_page(self) -> None:
        """Test citation without page number."""
        chunk = ContextChunk(
            content="Content",
            artifact_id="a1",
            document_id="doc-001",
            chunk_index=0,
            relevance_score=1.0,
        )
        citation = chunk.to_citation()
        assert "doc-001" in citation
        assert "p." not in citation

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        chunk = ContextChunk(
            content="Test",
            artifact_id="a1",
            document_id="d1",
            chunk_index=0,
            relevance_score=0.8,
        )
        d = chunk.to_dict()
        assert d["content"] == "Test"
        assert d["artifact_id"] == "a1"
        assert d["relevance_score"] == 0.8


# ---------------------------------------------------------------------------
# ContextWindow Tests
# ---------------------------------------------------------------------------


class TestContextWindow:
    """Tests for ContextWindow dataclass."""

    def test_empty_window(self) -> None:
        """Test creating an empty context window."""
        window = ContextWindow()
        assert window.chunk_count == 0
        assert window.document_count == 0
        assert window.total_tokens == 0

    def test_window_with_chunks(self) -> None:
        """Test window with chunks."""
        chunks = [
            ContextChunk(
                content="First chunk",
                artifact_id="a1",
                document_id="d1",
                chunk_index=0,
                relevance_score=1.0,
            ),
            ContextChunk(
                content="Second chunk",
                artifact_id="a2",
                document_id="d2",
                chunk_index=0,
                relevance_score=0.9,
            ),
        ]
        window = ContextWindow(
            chunks=chunks,
            total_tokens=100,
            token_budget=1000,
            source_documents=["d1", "d2"],
        )
        assert window.chunk_count == 2
        assert window.document_count == 2

    def test_budget_utilization(self) -> None:
        """Test budget utilization calculation."""
        window = ContextWindow(
            total_tokens=500,
            token_budget=1000,
        )
        assert window.budget_utilization == 0.5

    def test_budget_utilization_zero_budget(self) -> None:
        """Test budget utilization with zero budget."""
        window = ContextWindow(total_tokens=100, token_budget=0)
        assert window.budget_utilization == 0.0

    def test_is_within_budget(self) -> None:
        """Test within budget check."""
        window = ContextWindow(total_tokens=500, token_budget=1000)
        assert window.is_within_budget is True

        window2 = ContextWindow(total_tokens=1500, token_budget=1000)
        assert window2.is_within_budget is False

    def test_to_text_with_citations(self) -> None:
        """Test text generation with citations."""
        chunks = [
            ContextChunk(
                content="First content",
                artifact_id="a1",
                document_id="doc-1",
                chunk_index=0,
                relevance_score=1.0,
            ),
        ]
        window = ContextWindow(chunks=chunks)
        text = window.to_text(include_citations=True)
        assert "First content" in text
        assert "[doc-1" in text

    def test_to_text_without_citations(self) -> None:
        """Test text generation without citations."""
        chunks = [
            ContextChunk(
                content="First content",
                artifact_id="a1",
                document_id="doc-1",
                chunk_index=0,
                relevance_score=1.0,
            ),
        ]
        window = ContextWindow(chunks=chunks)
        text = window.to_text(include_citations=False)
        assert "First content" in text
        assert "[doc-1" not in text

    def test_get_citations(self) -> None:
        """Test getting all citations."""
        chunks = [
            ContextChunk(
                content="C1",
                artifact_id="a1",
                document_id="d1",
                chunk_index=0,
                relevance_score=1.0,
            ),
            ContextChunk(
                content="C2",
                artifact_id="a2",
                document_id="d2",
                chunk_index=1,
                relevance_score=0.9,
            ),
        ]
        window = ContextWindow(chunks=chunks)
        citations = window.get_citations()
        assert len(citations) == 2
        assert citations[0]["artifact_id"] == "a1"
        assert citations[1]["artifact_id"] == "a2"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        window = ContextWindow(
            total_tokens=100,
            token_budget=1000,
            source_documents=["d1"],
        )
        d = window.to_dict()
        assert d["total_tokens"] == 100
        assert d["token_budget"] == 1000
        assert "budget_utilization" in d


# ---------------------------------------------------------------------------
# ContextAggregator Tests
# ---------------------------------------------------------------------------


class TestContextAggregatorInit:
    """Tests for ContextAggregator initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        agg = ContextAggregator()
        assert agg._default_budget == DEFAULT_TOKEN_BUDGET

    def test_custom_budget(self) -> None:
        """Test custom token budget."""
        agg = ContextAggregator(default_token_budget=4000)
        assert agg._default_budget == 4000

    def test_invalid_budget_too_low(self) -> None:
        """Test rejection of too-low budget."""
        with pytest.raises(AssertionError):
            ContextAggregator(default_token_budget=10)

    def test_invalid_budget_too_high(self) -> None:
        """Test rejection of too-high budget."""
        with pytest.raises(AssertionError):
            ContextAggregator(default_token_budget=MAX_TOKEN_BUDGET + 1)


# ---------------------------------------------------------------------------
# GWT-1: Multi-Document Context Assembly Tests
# ---------------------------------------------------------------------------


class TestGWT1MultiDocumentContextAssembly:
    """GWT-1: Multi-document context assembly tests."""

    def test_aggregate_multiple_documents(self) -> None:
        """GWT-1: Chunks from multiple documents are assembled."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="doc-1",
                content="Content from doc 1",
                chunk_index=0,
                total_chunks=1,
            ),
            IFChunkArtifact(
                artifact_id="a2",
                document_id="doc-2",
                content="Content from doc 2",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count == 2
        assert window.document_count == 2
        assert "doc-1" in window.source_documents
        assert "doc-2" in window.source_documents

    def test_aggregate_empty_chunks(self) -> None:
        """GWT-1: Empty chunk list produces empty window."""
        agg = ContextAggregator()
        window = agg.aggregate([])
        assert window.chunk_count == 0

    def test_aggregate_single_chunk(self) -> None:
        """GWT-1: Single chunk is preserved."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="doc-1",
                content="Single chunk content",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count == 1
        assert window.chunks[0].content == "Single chunk content"

    def test_aggregate_orders_by_relevance(self) -> None:
        """GWT-1: Chunks are ordered by relevance score."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="Low",
                chunk_index=0,
                total_chunks=1,
            ),
            IFChunkArtifact(
                artifact_id="a2",
                document_id="d2",
                content="High",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        scores = {"a1": 0.5, "a2": 0.9}
        agg = ContextAggregator()
        window = agg.aggregate(chunks, relevance_scores=scores)

        assert window.chunks[0].relevance_score == 0.9
        assert window.chunks[1].relevance_score == 0.5


# ---------------------------------------------------------------------------
# GWT-2: Provenance Preservation Tests
# ---------------------------------------------------------------------------


class TestGWT2ProvenancePreservation:
    """GWT-2: Provenance preservation tests."""

    def test_artifact_id_preserved(self) -> None:
        """GWT-2: Artifact IDs are preserved."""
        chunks = [
            IFChunkArtifact(
                artifact_id="unique-artifact-id",
                document_id="doc-1",
                content="Content",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunks[0].artifact_id == "unique-artifact-id"

    def test_document_id_preserved(self) -> None:
        """GWT-2: Document IDs are preserved."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="my-doc-id",
                content="Content",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunks[0].document_id == "my-doc-id"

    def test_page_number_from_metadata(self) -> None:
        """GWT-2: Page numbers are extracted from metadata."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="Content",
                chunk_index=0,
                total_chunks=1,
                metadata={"page_number": 42},
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunks[0].page_number == 42

    def test_citations_available(self) -> None:
        """GWT-2: Citations can be generated for all chunks."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="doc-1",
                content="Content",
                chunk_index=5,
                total_chunks=10,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        citations = window.get_citations()
        assert len(citations) == 1
        assert citations[0]["document_id"] == "doc-1"
        assert citations[0]["chunk_index"] == 5


# ---------------------------------------------------------------------------
# GWT-3: Token Budget Enforcement Tests
# ---------------------------------------------------------------------------


class TestGWT3TokenBudgetEnforcement:
    """GWT-3: Token budget enforcement tests."""

    def test_respects_token_budget(self) -> None:
        """GWT-3: Aggregation respects token budget."""
        # Create chunks that exceed budget
        chunks = [
            IFChunkArtifact(
                artifact_id=f"a{i}",
                document_id="d1",
                content="Word " * 100,  # ~130 tokens each
                chunk_index=i,
                total_chunks=10,
            )
            for i in range(10)
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks, token_budget=500)

        assert window.is_within_budget

    def test_at_least_one_chunk(self) -> None:
        """GWT-3: At least one chunk is included even if over budget."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="Word " * 1000,  # Large chunk
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks, token_budget=100)

        assert window.chunk_count >= 1

    def test_higher_relevance_prioritized(self) -> None:
        """GWT-3: Higher relevance chunks are kept when pruning."""
        chunks = [
            IFChunkArtifact(
                artifact_id="low",
                document_id="d1",
                content="Low relevance " * 50,
                chunk_index=0,
                total_chunks=2,
            ),
            IFChunkArtifact(
                artifact_id="high",
                document_id="d1",
                content="High relevance " * 50,
                chunk_index=1,
                total_chunks=2,
            ),
        ]
        scores = {"low": 0.3, "high": 0.9}
        agg = ContextAggregator()
        window = agg.aggregate(chunks, token_budget=200, relevance_scores=scores)

        # High relevance should be first/included
        if window.chunk_count == 1:
            assert window.chunks[0].artifact_id == "high"


# ---------------------------------------------------------------------------
# GWT-4: Overlap Deduplication Tests
# ---------------------------------------------------------------------------


class TestGWT4OverlapDeduplication:
    """GWT-4: Overlap deduplication tests."""

    def test_duplicate_content_removed(self) -> None:
        """GWT-4: Duplicate content is deduplicated."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="Same content here",
                chunk_index=0,
                total_chunks=2,
            ),
            IFChunkArtifact(
                artifact_id="a2",
                document_id="d1",
                content="Same content here",  # Duplicate
                chunk_index=1,
                total_chunks=2,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count == 1

    def test_similar_whitespace_deduplicated(self) -> None:
        """GWT-4: Content with different whitespace is deduplicated."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="Same   content   here",
                chunk_index=0,
                total_chunks=2,
            ),
            IFChunkArtifact(
                artifact_id="a2",
                document_id="d1",
                content="Same content here",
                chunk_index=1,
                total_chunks=2,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count == 1

    def test_different_content_preserved(self) -> None:
        """GWT-4: Different content is preserved."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="First unique content",
                chunk_index=0,
                total_chunks=2,
            ),
            IFChunkArtifact(
                artifact_id="a2",
                document_id="d1",
                content="Second unique content",
                chunk_index=1,
                total_chunks=2,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count == 2


# ---------------------------------------------------------------------------
# GWT-5: Context Metadata Generation Tests
# ---------------------------------------------------------------------------


class TestGWT5ContextMetadataGeneration:
    """GWT-5: Context metadata generation tests."""

    def test_source_documents_tracked(self) -> None:
        """GWT-5: Source documents are tracked."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="doc-A",
                content="A",
                chunk_index=0,
                total_chunks=1,
            ),
            IFChunkArtifact(
                artifact_id="a2",
                document_id="doc-B",
                content="B",
                chunk_index=0,
                total_chunks=1,
            ),
            IFChunkArtifact(
                artifact_id="a3",
                document_id="doc-A",
                content="C",
                chunk_index=1,
                total_chunks=2,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert set(window.source_documents) == {"doc-A", "doc-B"}

    def test_total_tokens_calculated(self) -> None:
        """GWT-5: Total tokens are calculated."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="One two three",  # 3 words
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        expected = int(3 * TOKENS_PER_WORD_ESTIMATE)
        assert window.total_tokens == expected

    def test_metadata_includes_method(self) -> None:
        """GWT-5: Metadata includes aggregation method."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="X",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert "aggregation_method" in window.metadata


# ---------------------------------------------------------------------------
# Document Diversity Tests
# ---------------------------------------------------------------------------


class TestDocumentDiversity:
    """Tests for document diversity preservation."""

    def test_diversity_enabled_by_default(self) -> None:
        """Test that diversity is enabled by default."""
        agg = ContextAggregator()
        assert agg._preserve_diversity is True

    def test_diversity_ensures_min_per_doc(self) -> None:
        """Test that minimum chunks per document is respected."""
        chunks = [
            # Doc A has high relevance chunks
            IFChunkArtifact(
                artifact_id="a1",
                document_id="doc-A",
                content="High A",
                chunk_index=0,
                total_chunks=2,
            ),
            IFChunkArtifact(
                artifact_id="a2",
                document_id="doc-A",
                content="High A2",
                chunk_index=1,
                total_chunks=2,
            ),
            # Doc B has lower relevance
            IFChunkArtifact(
                artifact_id="b1",
                document_id="doc-B",
                content="Low B",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        scores = {"a1": 0.9, "a2": 0.85, "b1": 0.5}
        agg = ContextAggregator(min_chunks_per_document=1)
        window = agg.aggregate(chunks, relevance_scores=scores)

        doc_ids = [c.document_id for c in window.chunks]
        assert "doc-B" in doc_ids  # Diversity preserved


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_aggregate_context_function(self) -> None:
        """Test aggregate_context convenience function."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="Content",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        window = aggregate_context(chunks)
        assert window.chunk_count == 1

    def test_create_context_aggregator_function(self) -> None:
        """Test create_context_aggregator factory function."""
        agg = create_context_aggregator(token_budget=4000, preserve_diversity=False)
        assert agg._default_budget == 4000
        assert agg._preserve_diversity is False


# ---------------------------------------------------------------------------
# Dict Aggregation Tests
# ---------------------------------------------------------------------------


class TestDictAggregation:
    """Tests for dictionary-based aggregation."""

    def test_aggregate_from_dicts(self) -> None:
        """Test aggregating from dictionary representations."""
        dicts = [
            {
                "content": "First content",
                "artifact_id": "a1",
                "document_id": "d1",
                "chunk_index": 0,
                "relevance_score": 0.9,
            },
            {
                "content": "Second content",
                "artifact_id": "a2",
                "document_id": "d2",
                "chunk_index": 0,
                "score": 0.8,  # Alternative key
            },
        ]
        agg = ContextAggregator()
        window = agg.aggregate_from_dicts(dicts)

        assert window.chunk_count == 2

    def test_aggregate_from_dicts_empty(self) -> None:
        """Test aggregating empty dict list."""
        agg = ContextAggregator()
        window = agg.aggregate_from_dicts([])
        assert window.chunk_count == 0


# ---------------------------------------------------------------------------
# JPL Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule2_max_chunks_constant(self) -> None:
        """JPL Rule #2: MAX_CONTEXT_CHUNKS is bounded."""
        assert MAX_CONTEXT_CHUNKS == 100
        assert MAX_CONTEXT_CHUNKS > 0

    def test_rule2_max_budget_constant(self) -> None:
        """JPL Rule #2: MAX_TOKEN_BUDGET is bounded."""
        assert MAX_TOKEN_BUDGET == 128000
        assert MAX_TOKEN_BUDGET > 0

    def test_rule5_aggregate_assertions(self) -> None:
        """JPL Rule #5: aggregate() asserts preconditions."""
        agg = ContextAggregator()

        with pytest.raises(AssertionError):
            agg.aggregate(None)  # type: ignore

    def test_rule5_budget_range_assertion(self) -> None:
        """JPL Rule #5: Budget range is validated."""
        agg = ContextAggregator()

        with pytest.raises(AssertionError):
            agg.aggregate([], token_budget=50)  # Too low

    def test_rule9_type_hints_chunk(self) -> None:
        """JPL Rule #9: ContextChunk has complete type hints."""
        from dataclasses import fields

        chunk_fields = {f.name for f in fields(ContextChunk)}
        required = {
            "content",
            "artifact_id",
            "document_id",
            "chunk_index",
            "relevance_score",
        }
        assert required.issubset(chunk_fields)

    def test_rule9_type_hints_window(self) -> None:
        """JPL Rule #9: ContextWindow has complete type hints."""
        from dataclasses import fields

        window_fields = {f.name for f in fields(ContextWindow)}
        required = {"chunks", "total_tokens", "token_budget", "source_documents"}
        assert required.issubset(window_fields)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_content_truncated(self) -> None:
        """Test that very long content is truncated."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="x" * 100000,  # Very long
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert len(window.chunks[0].content) <= 50000

    def test_unicode_content(self) -> None:
        """Test handling of unicode content."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="日本語のテスト。これは二番目の文です。",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count == 1

    def test_special_characters(self) -> None:
        """Test handling of special characters."""
        chunks = [
            IFChunkArtifact(
                artifact_id="a1",
                document_id="d1",
                content="Test @#$%^&*() here. Another sentence!",
                chunk_index=0,
                total_chunks=1,
            ),
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count == 1

    def test_many_chunks_bounded(self) -> None:
        """Test that many chunks are bounded."""
        chunks = [
            IFChunkArtifact(
                artifact_id=f"a{i}",
                document_id="d1",
                content=f"Chunk {i}",
                chunk_index=i,
                total_chunks=200,
            )
            for i in range(200)
        ]
        agg = ContextAggregator()
        window = agg.aggregate(chunks)

        assert window.chunk_count <= MAX_CONTEXT_CHUNKS

    def test_estimate_tokens_static(self) -> None:
        """Test static token estimation method."""
        tokens = ContextAggregator.estimate_tokens("One two three four five")
        expected = int(5 * TOKENS_PER_WORD_ESTIMATE)
        assert tokens == expected

    def test_estimate_tokens_empty(self) -> None:
        """Test token estimation for empty string."""
        tokens = ContextAggregator.estimate_tokens("")
        assert tokens == 0
