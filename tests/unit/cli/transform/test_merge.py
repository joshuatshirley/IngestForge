"""Tests for transform merge command.

Tests ChunkMerger class and merge command functionality.
"""

from __future__ import annotations

from typing import List

import pytest

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.cli.transform.merge import (
    ChunkMerger,
    MergeResult,
    DedupeStrategy,
    ConflictResolution,
)


# Fixtures
@pytest.fixture
def sample_chunks() -> List[ChunkRecord]:
    """Create sample chunks for testing."""
    return [
        ChunkRecord(
            chunk_id="chunk1",
            document_id="doc1",
            content="Introduction to machine learning concepts.",
            char_count=43,
            word_count=5,
            entities=["CONCEPT:machine learning"],
            concepts=["machine learning"],
            quality_score=0.8,
            ingested_at="2024-01-15T10:00:00",
        ),
        ChunkRecord(
            chunk_id="chunk2",
            document_id="doc1",
            content="Deep learning is a subset of machine learning.",
            char_count=47,
            word_count=8,
            entities=["CONCEPT:deep learning"],
            concepts=["deep learning", "machine learning"],
            quality_score=0.9,
            ingested_at="2024-01-16T10:00:00",
        ),
        ChunkRecord(
            chunk_id="chunk3",
            document_id="doc2",
            content="Neural networks power deep learning systems.",
            char_count=44,
            word_count=6,
            entities=["TECH:neural networks"],
            concepts=["neural networks"],
            quality_score=0.7,
            ingested_at="2024-01-17T10:00:00",
        ),
    ]


@pytest.fixture
def duplicate_chunks() -> List[ChunkRecord]:
    """Create chunks with duplicates."""
    return [
        ChunkRecord(
            chunk_id="chunk1",
            document_id="doc1",
            content="Machine learning is transforming industries.",
            char_count=45,
            word_count=5,
        ),
        ChunkRecord(
            chunk_id="chunk2",
            document_id="doc1",
            content="Machine learning is transforming industries.",  # Exact duplicate
            char_count=45,
            word_count=5,
        ),
        ChunkRecord(
            chunk_id="chunk3",
            document_id="doc1",
            content="machine learning is transforming industries",  # Near duplicate (case/punctuation)
            char_count=43,
            word_count=5,
        ),
        ChunkRecord(
            chunk_id="chunk4",
            document_id="doc1",
            content="ML is changing how we work in various fields.",  # Similar meaning
            char_count=45,
            word_count=9,
        ),
    ]


class TestChunkMergerExactDedupe:
    """Tests for exact deduplication."""

    def test_exact_dedupe_removes_duplicates(
        self, duplicate_chunks: List[ChunkRecord]
    ) -> None:
        """Test exact deduplication removes exact matches."""
        merger = ChunkMerger(dedupe_strategy=DedupeStrategy.EXACT)
        result = merger.deduplicate(duplicate_chunks)

        # Should remove exact duplicates
        assert len(result) == 3  # chunk2 removed as exact dup of chunk1

    def test_exact_dedupe_preserves_unique(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test exact deduplication preserves unique chunks."""
        merger = ChunkMerger(dedupe_strategy=DedupeStrategy.EXACT)
        result = merger.deduplicate(sample_chunks)

        assert len(result) == len(sample_chunks)

    def test_exact_dedupe_empty_list(self) -> None:
        """Test exact deduplication with empty list."""
        merger = ChunkMerger(dedupe_strategy=DedupeStrategy.EXACT)
        result = merger.deduplicate([])

        assert result == []

    def test_exact_dedupe_single_chunk(self) -> None:
        """Test exact deduplication with single chunk."""
        chunk = ChunkRecord(chunk_id="x", document_id="y", content="test")
        merger = ChunkMerger(dedupe_strategy=DedupeStrategy.EXACT)
        result = merger.deduplicate([chunk])

        assert len(result) == 1


class TestChunkMergerFuzzyDedupe:
    """Tests for fuzzy deduplication."""

    def test_fuzzy_dedupe_removes_similar(
        self, duplicate_chunks: List[ChunkRecord]
    ) -> None:
        """Test fuzzy deduplication removes similar chunks."""
        merger = ChunkMerger(
            dedupe_strategy=DedupeStrategy.FUZZY,
            similarity_threshold=0.8,
        )
        result = merger.deduplicate(duplicate_chunks)

        # Should remove chunk2 and chunk3 as similar to chunk1
        assert len(result) < len(duplicate_chunks)

    def test_fuzzy_dedupe_with_lower_threshold(
        self, duplicate_chunks: List[ChunkRecord]
    ) -> None:
        """Test fuzzy deduplication with lower threshold."""
        merger = ChunkMerger(
            dedupe_strategy=DedupeStrategy.FUZZY,
            similarity_threshold=0.5,
        )
        result = merger.deduplicate(duplicate_chunks)

        # Even more aggressive deduplication
        assert len(result) <= 2

    def test_fuzzy_dedupe_preserves_different(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test fuzzy deduplication preserves different chunks."""
        merger = ChunkMerger(
            dedupe_strategy=DedupeStrategy.FUZZY,
            similarity_threshold=0.9,
        )
        result = merger.deduplicate(sample_chunks)

        # All unique chunks should be preserved
        assert len(result) == len(sample_chunks)


class TestChunkMergerSemanticDedupe:
    """Tests for semantic deduplication."""

    def test_semantic_dedupe_fallback_to_fuzzy(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test semantic deduplication falls back when no embeddings."""
        merger = ChunkMerger(dedupe_strategy=DedupeStrategy.SEMANTIC)
        result = merger.deduplicate(sample_chunks)

        # Should fall back to fuzzy and preserve all
        assert len(result) == len(sample_chunks)

    def test_semantic_dedupe_with_embeddings(self) -> None:
        """Test semantic deduplication with embeddings."""
        # Create chunks with similar embeddings
        chunks = [
            ChunkRecord(
                chunk_id="1",
                document_id="doc",
                content="test 1",
                embedding=[1.0, 0.0, 0.0],
            ),
            ChunkRecord(
                chunk_id="2",
                document_id="doc",
                content="test 2",
                embedding=[0.99, 0.1, 0.0],  # Very similar
            ),
            ChunkRecord(
                chunk_id="3",
                document_id="doc",
                content="test 3",
                embedding=[0.0, 1.0, 0.0],  # Different
            ),
        ]

        merger = ChunkMerger(
            dedupe_strategy=DedupeStrategy.SEMANTIC,
            similarity_threshold=0.95,
        )
        result = merger.deduplicate(chunks)

        assert len(result) == 2  # chunk1 and chunk3


class TestChunkMergerMergeLibraries:
    """Tests for merging libraries."""

    def test_merge_libraries_combines(self) -> None:
        """Test merging multiple libraries."""
        lib1 = [ChunkRecord(chunk_id="1", document_id="doc1", content="content 1")]
        lib2 = [ChunkRecord(chunk_id="2", document_id="doc2", content="content 2")]

        merger = ChunkMerger()
        result = merger.merge_libraries({"lib1": lib1, "lib2": lib2}, deduplicate=False)

        assert result.total_input == 2
        assert result.total_output == 2
        assert len(result.chunks) == 2

    def test_merge_libraries_with_dedup(
        self, duplicate_chunks: List[ChunkRecord]
    ) -> None:
        """Test merging with deduplication."""
        merger = ChunkMerger(dedupe_strategy=DedupeStrategy.EXACT)
        result = merger.merge_libraries(
            {"source": duplicate_chunks},
            deduplicate=True,
        )

        assert result.duplicates_removed > 0
        assert result.total_output < result.total_input

    def test_merge_libraries_tracks_counts(self) -> None:
        """Test merge result tracks source counts."""
        lib1 = [
            ChunkRecord(chunk_id=f"a{i}", document_id="d", content=f"c{i}")
            for i in range(3)
        ]
        lib2 = [
            ChunkRecord(chunk_id=f"b{i}", document_id="d", content=f"c{i}")
            for i in range(5)
        ]

        merger = ChunkMerger()
        result = merger.merge_libraries({"lib1": lib1, "lib2": lib2}, deduplicate=False)

        assert result.source_counts["lib1"] == 3
        assert result.source_counts["lib2"] == 5
        assert result.total_input == 8


class TestChunkMergerConflictResolution:
    """Tests for conflict resolution."""

    def test_keep_first_resolution(self) -> None:
        """Test keep_first conflict resolution."""
        chunk1 = ChunkRecord(
            chunk_id="1",
            document_id="doc",
            content="content",
            ingested_at="2024-01-01T10:00:00",
        )
        chunk2 = ChunkRecord(
            chunk_id="2",
            document_id="doc",
            content="content",
            ingested_at="2024-01-02T10:00:00",
        )

        merger = ChunkMerger(conflict_resolution=ConflictResolution.KEEP_FIRST)
        result = merger.resolve_conflicts(chunk1, chunk2)

        assert result.chunk_id == "1"

    def test_keep_latest_resolution(self) -> None:
        """Test keep_latest conflict resolution."""
        chunk1 = ChunkRecord(
            chunk_id="1",
            document_id="doc",
            content="content",
            ingested_at="2024-01-01T10:00:00",
        )
        chunk2 = ChunkRecord(
            chunk_id="2",
            document_id="doc",
            content="content",
            ingested_at="2024-01-02T10:00:00",
        )

        merger = ChunkMerger(conflict_resolution=ConflictResolution.KEEP_LATEST)
        result = merger.resolve_conflicts(chunk1, chunk2)

        assert result.chunk_id == "2"

    def test_merge_metadata_resolution(self) -> None:
        """Test merge_metadata conflict resolution."""
        chunk1 = ChunkRecord(
            chunk_id="1",
            document_id="doc",
            content="content",
            entities=["PERSON:John"],
            concepts=["AI"],
            quality_score=0.7,
        )
        chunk2 = ChunkRecord(
            chunk_id="2",
            document_id="doc",
            content="content",
            entities=["ORG:OpenAI"],
            concepts=["ML"],
            quality_score=0.9,
        )

        merger = ChunkMerger(conflict_resolution=ConflictResolution.MERGE_METADATA)
        result = merger.resolve_conflicts(chunk1, chunk2)

        assert "PERSON:John" in result.entities
        assert "ORG:OpenAI" in result.entities
        assert "AI" in result.concepts
        assert "ML" in result.concepts
        assert result.quality_score == 0.9  # Max of both


class TestChunkMergerCombineChunks:
    """Tests for combining chunks."""

    def test_combine_by_document(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test combining chunks by document."""
        merger = ChunkMerger()
        result = merger.combine_chunks(sample_chunks, strategy="by_document")

        # Should have 2 combined chunks (doc1, doc2)
        assert len(result) == 2

    def test_combine_sequential(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test sequential combination returns original."""
        merger = ChunkMerger()
        result = merger.combine_chunks(sample_chunks, strategy="sequential")

        assert len(result) == len(sample_chunks)


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_dedup_rate_calculation(self) -> None:
        """Test deduplication rate calculation."""
        result = MergeResult(
            total_input=100,
            total_output=80,
            duplicates_removed=20,
        )

        assert result.dedup_rate == 20.0

    def test_dedup_rate_zero_input(self) -> None:
        """Test dedup rate with zero input."""
        result = MergeResult(total_input=0, total_output=0, duplicates_removed=0)

        assert result.dedup_rate == 0.0

    def test_dedup_rate_no_duplicates(self) -> None:
        """Test dedup rate with no duplicates."""
        result = MergeResult(
            total_input=50,
            total_output=50,
            duplicates_removed=0,
        )

        assert result.dedup_rate == 0.0


class TestChunkMergerJaccardSimilarity:
    """Tests for Jaccard similarity calculation."""

    def test_identical_sets(self) -> None:
        """Test Jaccard similarity for identical sets."""
        merger = ChunkMerger()
        words1 = {"machine", "learning", "ai"}
        words2 = {"machine", "learning", "ai"}

        similarity = merger._jaccard_similarity(words1, words2)

        assert similarity == 1.0

    def test_disjoint_sets(self) -> None:
        """Test Jaccard similarity for disjoint sets."""
        merger = ChunkMerger()
        words1 = {"machine", "learning"}
        words2 = {"deep", "neural"}

        similarity = merger._jaccard_similarity(words1, words2)

        assert similarity == 0.0

    def test_partial_overlap(self) -> None:
        """Test Jaccard similarity for partial overlap."""
        merger = ChunkMerger()
        words1 = {"machine", "learning", "ai"}
        words2 = {"machine", "learning", "deep"}

        similarity = merger._jaccard_similarity(words1, words2)

        # 2 common / 4 total = 0.5
        assert similarity == 0.5

    def test_empty_sets(self) -> None:
        """Test Jaccard similarity with empty sets."""
        merger = ChunkMerger()

        assert merger._jaccard_similarity(set(), {"a"}) == 0.0
        assert merger._jaccard_similarity({"a"}, set()) == 0.0
        assert merger._jaccard_similarity(set(), set()) == 0.0


class TestChunkMergerCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Test cosine similarity for identical vectors."""
        merger = ChunkMerger()
        vec = [1.0, 0.0, 0.0]

        similarity = merger._cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test cosine similarity for orthogonal vectors."""
        merger = ChunkMerger()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        similarity = merger._cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Test cosine similarity for opposite vectors."""
        merger = ChunkMerger()
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]

        similarity = merger._cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(-1.0)

    def test_empty_vectors(self) -> None:
        """Test cosine similarity with empty vectors."""
        merger = ChunkMerger()

        assert merger._cosine_similarity([], [1.0]) == 0.0
        assert merger._cosine_similarity([1.0], []) == 0.0
