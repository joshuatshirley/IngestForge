"""Tests for transform filter command.

Tests ChunkFilter class and filter command functionality.
"""

from __future__ import annotations

from datetime import date
from typing import List

import pytest

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.cli.transform.filter import (
    ChunkFilter,
    FilterResult,
)


# Fixtures
@pytest.fixture
def sample_chunks() -> List[ChunkRecord]:
    """Create sample chunks for testing."""
    return [
        ChunkRecord(
            chunk_id="chunk1",
            document_id="doc1",
            content="Short text about machine learning and AI.",
            char_count=42,
            word_count=7,
            entities=["PERSON:John", "ORG:OpenAI"],
            concepts=["machine learning", "AI"],
            quality_score=0.8,
            source_file="report.pdf",
            ingested_at="2024-01-15T10:00:00",
        ),
        ChunkRecord(
            chunk_id="chunk2",
            document_id="doc1",
            content="This is a much longer piece of text that discusses various topics including data science, statistics, and probability theory in great detail.",
            char_count=140,
            word_count=23,
            entities=["CONCEPT:statistics"],
            concepts=["data science", "statistics"],
            quality_score=0.9,
            source_file="report.pdf",
            ingested_at="2024-01-16T10:00:00",
        ),
        ChunkRecord(
            chunk_id="chunk3",
            document_id="doc2",
            content="Brief note.",
            char_count=11,
            word_count=2,
            entities=[],
            concepts=[],
            quality_score=0.3,
            source_file="notes.txt",
            ingested_at="2024-02-01T10:00:00",
        ),
        ChunkRecord(
            chunk_id="chunk4",
            document_id="doc2",
            content="A medium-length chunk about natural language processing and deep learning algorithms used in modern NLP systems.",
            char_count=110,
            word_count=17,
            entities=["TECH:NLP", "TECH:deep learning"],
            concepts=["NLP", "deep learning"],
            quality_score=0.7,
            source_file="paper.pdf",
            ingested_at="2024-02-15T10:00:00",
        ),
    ]


class TestChunkFilterLengthFilter:
    """Tests for length filtering."""

    def test_filter_by_min_length(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by minimum length."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_length(sample_chunks, min_len=50)

        assert len(filtered) == 2
        assert all(c.char_count >= 50 for c in filtered)

    def test_filter_by_max_length(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by maximum length."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_length(sample_chunks, max_len=50)

        assert len(filtered) == 2
        assert all(c.char_count <= 50 for c in filtered)

    def test_filter_by_length_range(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by length range."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_length(sample_chunks, min_len=40, max_len=120)

        assert len(filtered) == 2
        assert all(40 <= c.char_count <= 120 for c in filtered)

    def test_filter_by_length_no_constraints(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test filtering with no length constraints returns all."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_length(sample_chunks)

        assert len(filtered) == len(sample_chunks)

    def test_filter_by_length_empty_list(self) -> None:
        """Test filtering empty list."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_length([], min_len=50)

        assert filtered == []


class TestChunkFilterEntityFilter:
    """Tests for entity filtering."""

    def test_filter_has_entities(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering chunks with any entities."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_entity(sample_chunks, has_entities=True)

        assert len(filtered) == 3
        assert all(len(c.entities) > 0 for c in filtered)

    def test_filter_by_entity_type(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by entity type."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_entity(sample_chunks, entity_type="PERSON")

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "chunk1"

    def test_filter_by_entity_type_case_insensitive(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test entity type matching is case-insensitive."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_entity(sample_chunks, entity_type="person")

        assert len(filtered) == 1

    def test_filter_by_entity_type_org(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by ORG entity type."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_entity(sample_chunks, entity_type="ORG")

        assert len(filtered) == 1
        assert "ORG:OpenAI" in filtered[0].entities

    def test_filter_by_entity_type_tech(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by TECH entity type."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_entity(sample_chunks, entity_type="TECH")

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "chunk4"

    def test_filter_no_entity_match(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering with no matching entity type."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_entity(sample_chunks, entity_type="LOCATION")

        assert len(filtered) == 0


class TestChunkFilterTopicFilter:
    """Tests for topic filtering."""

    def test_filter_by_topic_single_word(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test filtering by single topic word."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_topic(sample_chunks, "machine", threshold=0.5)

        assert len(filtered) >= 1
        assert any("machine" in c.content.lower() for c in filtered)

    def test_filter_by_topic_phrase(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by topic phrase."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_topic(
            sample_chunks, "machine learning", threshold=0.5
        )

        assert len(filtered) >= 1

    def test_filter_by_topic_high_threshold(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test filtering with high threshold excludes partial matches."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_topic(
            sample_chunks, "machine learning AI", threshold=1.0
        )

        # Only chunk1 has all three words
        assert len(filtered) <= 1

    def test_filter_by_topic_low_threshold(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test filtering with low threshold is more permissive."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_topic(
            sample_chunks, "text about learning", threshold=0.3
        )

        # Should match chunks with at least one word
        assert len(filtered) >= 1

    def test_filter_by_topic_no_match(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering with no matching topic."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_topic(
            sample_chunks, "quantum physics", threshold=0.5
        )

        assert len(filtered) == 0


class TestChunkFilterSourceFilter:
    """Tests for source file filtering."""

    def test_filter_by_source_exact(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by exact source match."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_source(sample_chunks, r"report\.pdf")

        assert len(filtered) == 2
        assert all("report.pdf" in c.source_file for c in filtered)

    def test_filter_by_source_extension(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by file extension."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_source(sample_chunks, r"\.pdf$")

        assert len(filtered) == 3
        assert all(c.source_file.endswith(".pdf") for c in filtered)

    def test_filter_by_source_txt(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering for txt files."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_source(sample_chunks, r"\.txt$")

        assert len(filtered) == 1
        assert filtered[0].source_file == "notes.txt"

    def test_filter_by_source_case_insensitive(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test source filter is case-insensitive."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_source(sample_chunks, r"REPORT")

        assert len(filtered) == 2


class TestChunkFilterDateFilter:
    """Tests for date filtering."""

    def test_filter_by_date_start(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by start date."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_date(sample_chunks, start=date(2024, 2, 1))

        assert len(filtered) == 2

    def test_filter_by_date_end(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by end date."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_date(sample_chunks, end=date(2024, 1, 31))

        assert len(filtered) == 2

    def test_filter_by_date_range(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by date range."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_date(
            sample_chunks,
            start=date(2024, 1, 15),
            end=date(2024, 2, 1),
        )

        assert len(filtered) == 3

    def test_filter_by_date_excludes_none(
        self, sample_chunks: List[ChunkRecord]
    ) -> None:
        """Test that chunks without dates are excluded."""
        chunks_no_date = [ChunkRecord(chunk_id="x", document_id="y", content="test")]
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_date(chunks_no_date, start=date(2024, 1, 1))

        assert len(filtered) == 0


class TestChunkFilterQualityFilter:
    """Tests for quality score filtering."""

    def test_filter_by_quality(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by quality score."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_quality(sample_chunks, min_score=0.7)

        assert len(filtered) == 3
        assert all(c.quality_score >= 0.7 for c in filtered)

    def test_filter_by_quality_high(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering by high quality score."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_quality(sample_chunks, min_score=0.85)

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "chunk2"

    def test_filter_by_quality_zero(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filtering with zero threshold returns all."""
        chunk_filter = ChunkFilter()
        filtered = chunk_filter.filter_by_quality(sample_chunks, min_score=0.0)

        assert len(filtered) == len(sample_chunks)


class TestChunkFilterCombined:
    """Tests for combining multiple filters."""

    def test_combined_length_and_entity(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test combining length and entity filters."""
        chunk_filter = ChunkFilter()
        chunk_filter.add_length_filter(min_len=50)
        chunk_filter.add_entity_filter(has_entities=True)

        result = chunk_filter.apply(sample_chunks)

        assert result.filtered_count == 2
        assert all(c.char_count >= 50 and len(c.entities) > 0 for c in result.chunks)

    def test_combined_topic_and_quality(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test combining topic and quality filters."""
        chunk_filter = ChunkFilter()
        chunk_filter.add_topic_filter("learning", threshold=0.5)
        chunk_filter.add_quality_filter(min_score=0.7)

        result = chunk_filter.apply(sample_chunks)

        assert result.filtered_count >= 1
        assert all(c.quality_score >= 0.7 for c in result.chunks)

    def test_combined_multiple_filters(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test combining three filters."""
        chunk_filter = ChunkFilter()
        chunk_filter.add_length_filter(min_len=40, max_len=150)
        chunk_filter.add_entity_filter(has_entities=True)
        chunk_filter.add_source_filter(r"\.pdf$")

        result = chunk_filter.apply(sample_chunks)

        assert result.filtered_count >= 1
        assert len(result.filters_applied) == 3

    def test_filter_result_statistics(self, sample_chunks: List[ChunkRecord]) -> None:
        """Test filter result contains correct statistics."""
        chunk_filter = ChunkFilter()
        chunk_filter.add_length_filter(min_len=50)

        result = chunk_filter.apply(sample_chunks)

        assert result.original_count == 4
        assert result.filtered_count == 2
        assert result.removed_count == 2
        assert result.retention_rate == 50.0


class TestFilterResult:
    """Tests for FilterResult dataclass."""

    def test_retention_rate_calculation(self) -> None:
        """Test retention rate calculation."""
        result = FilterResult(original_count=100, filtered_count=75, removed_count=25)

        assert result.retention_rate == 75.0

    def test_retention_rate_zero_original(self) -> None:
        """Test retention rate with zero original count."""
        result = FilterResult(original_count=0, filtered_count=0, removed_count=0)

        assert result.retention_rate == 0.0

    def test_retention_rate_all_filtered(self) -> None:
        """Test retention rate when all chunks pass."""
        result = FilterResult(original_count=50, filtered_count=50, removed_count=0)

        assert result.retention_rate == 100.0


class TestChunkFilterMethodChaining:
    """Tests for method chaining support."""

    def test_method_chaining(self) -> None:
        """Test that filters can be chained."""
        chunk_filter = (
            ChunkFilter()
            .add_length_filter(min_len=50)
            .add_entity_filter(has_entities=True)
            .add_quality_filter(min_score=0.5)
        )

        assert len(chunk_filter._filters) == 3
        assert len(chunk_filter._filter_names) == 3

    def test_filter_names_tracked(self) -> None:
        """Test that filter names are tracked."""
        chunk_filter = ChunkFilter()
        chunk_filter.add_length_filter(min_len=100, max_len=500)
        chunk_filter.add_topic_filter("AI")

        assert "length(100-500)" in chunk_filter._filter_names
        assert "topic(AI)" in chunk_filter._filter_names
