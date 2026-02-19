"""Tests for related chunks module (QUIZ-002.2).

Tests the semantic context linker:
- RelatedChunk dataclass
- RelatedChunksResult
- RelatedChunksLinker searching
- Result filtering and conversion
"""

import pytest
from unittest.mock import Mock

from ingestforge.study.related_chunks import (
    RelatedChunk,
    RelatedChunksResult,
    RelatedChunksLinker,
    find_related_chunks,
    DEFAULT_MAX_RESULTS,
    MAX_ALLOWED_RESULTS,
    DEFAULT_SIMILARITY_THRESHOLD,
)


class TestRelatedChunk:
    """Test RelatedChunk dataclass."""

    def test_basic_chunk(self) -> None:
        """Should create basic chunk."""
        chunk = RelatedChunk(
            text="This is the content",
            source="document.pdf",
            similarity=0.85,
        )

        assert chunk.text == "This is the content"
        assert chunk.source == "document.pdf"
        assert chunk.similarity == 0.85

    def test_source_display_simple(self) -> None:
        """Should extract filename from source."""
        chunk = RelatedChunk(
            text="Content",
            source="/path/to/document.pdf",
            similarity=0.5,
        )

        assert chunk.source_display == "document.pdf"

    def test_source_display_windows_path(self) -> None:
        """Should handle Windows paths."""
        chunk = RelatedChunk(
            text="Content",
            source="C:\\Users\\docs\\file.txt",
            similarity=0.5,
        )

        assert chunk.source_display == "file.txt"

    def test_source_display_empty(self) -> None:
        """Should handle empty source."""
        chunk = RelatedChunk(
            text="Content",
            source="",
            similarity=0.5,
        )

        assert chunk.source_display == "Unknown Source"

    def test_truncate_short_text(self) -> None:
        """Should not truncate short text."""
        chunk = RelatedChunk(
            text="Short text",
            source="doc.txt",
            similarity=0.5,
        )

        assert chunk.truncate(100) == "Short text"

    def test_truncate_long_text(self) -> None:
        """Should truncate long text with ellipsis."""
        long_text = "A" * 250
        chunk = RelatedChunk(
            text=long_text,
            source="doc.txt",
            similarity=0.5,
        )

        truncated = chunk.truncate(200)
        assert len(truncated) <= 200
        assert truncated.endswith("...")

    def test_truncate_custom_length(self) -> None:
        """Should respect custom max length."""
        chunk = RelatedChunk(
            text="This is a test string",
            source="doc.txt",
            similarity=0.5,
        )

        truncated = chunk.truncate(10)
        assert len(truncated) <= 10


class TestRelatedChunksResult:
    """Test RelatedChunksResult dataclass."""

    def test_empty_result(self) -> None:
        """Empty result should have zero count."""
        result = RelatedChunksResult()

        assert result.count == 0
        assert not result.has_results
        assert result.avg_similarity == 0.0

    def test_with_chunks(self) -> None:
        """Should track chunks correctly."""
        chunks = [
            RelatedChunk(text="A", source="a.txt", similarity=0.8),
            RelatedChunk(text="B", source="b.txt", similarity=0.6),
        ]

        result = RelatedChunksResult(chunks=chunks)

        assert result.count == 2
        assert result.has_results

    def test_avg_similarity(self) -> None:
        """Should calculate average similarity."""
        chunks = [
            RelatedChunk(text="A", source="a.txt", similarity=0.8),
            RelatedChunk(text="B", source="b.txt", similarity=0.6),
            RelatedChunk(text="C", source="c.txt", similarity=0.4),
        ]

        result = RelatedChunksResult(chunks=chunks)

        # (0.8 + 0.6 + 0.4) / 3 = 0.6
        assert result.avg_similarity == 0.6

    def test_query_text(self) -> None:
        """Should store query text."""
        result = RelatedChunksResult(query_text="test query")

        assert result.query_text == "test query"


class TestRelatedChunksLinkerInit:
    """Test RelatedChunksLinker initialization."""

    def test_default_init(self) -> None:
        """Should initialize with defaults."""
        linker = RelatedChunksLinker()

        assert linker.max_results == DEFAULT_MAX_RESULTS
        assert linker.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert not linker.is_available

    def test_init_with_storage(self) -> None:
        """Should accept storage."""
        mock_storage = Mock()
        linker = RelatedChunksLinker(storage=mock_storage)

        assert linker.storage is mock_storage
        assert linker.is_available

    def test_init_custom_limits(self) -> None:
        """Should accept custom limits."""
        linker = RelatedChunksLinker(
            similarity_threshold=0.5,
            max_results=5,
        )

        assert linker.similarity_threshold == 0.5
        assert linker.max_results == 5

    def test_max_results_exceeds_limit(self) -> None:
        """Should reject max_results exceeding Rule #2 limit."""
        with pytest.raises(ValueError, match="cannot exceed"):
            RelatedChunksLinker(max_results=MAX_ALLOWED_RESULTS + 1)

    def test_max_results_zero(self) -> None:
        """Should reject zero max_results."""
        with pytest.raises(ValueError, match="positive"):
            RelatedChunksLinker(max_results=0)

    def test_max_results_negative(self) -> None:
        """Should reject negative max_results."""
        with pytest.raises(ValueError, match="positive"):
            RelatedChunksLinker(max_results=-1)

    def test_threshold_too_low(self) -> None:
        """Should reject threshold below 0."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            RelatedChunksLinker(similarity_threshold=-0.1)

    def test_threshold_too_high(self) -> None:
        """Should reject threshold above 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            RelatedChunksLinker(similarity_threshold=1.5)


class TestRelatedChunksLinkerSearch:
    """Test search functionality."""

    def test_find_related_with_storage(self) -> None:
        """Should search storage and return results."""
        mock_storage = Mock()
        mock_storage.search.return_value = [
            {
                "text": "Related content",
                "source": "doc.pdf",
                "similarity": 0.8,
                "id": "chunk1",
            }
        ]

        linker = RelatedChunksLinker(storage=mock_storage)
        result = linker.find_related(
            question="What is X?",
            answer="Y",
        )

        assert result.has_results
        assert result.count == 1
        assert result.chunks[0].text == "Related content"

    def test_find_related_empty_question(self) -> None:
        """Should return empty for empty question."""
        linker = RelatedChunksLinker()
        result = linker.find_related(question="")

        assert not result.has_results

    def test_find_related_no_storage(self) -> None:
        """Should return empty when no storage configured."""
        linker = RelatedChunksLinker()
        result = linker.find_related(question="What is X?")

        assert not result.has_results

    def test_find_related_query_includes_answer(self) -> None:
        """Query should include answer by default."""
        mock_storage = Mock()
        mock_storage.search.return_value = []

        linker = RelatedChunksLinker(storage=mock_storage)
        linker.find_related(
            question="What is X?",
            answer="Y",
        )

        call_args = mock_storage.search.call_args
        query = call_args[1]["query"]
        assert "X" in query
        assert "Y" in query

    def test_find_related_exclude_answer(self) -> None:
        """Should exclude answer when requested."""
        mock_storage = Mock()
        mock_storage.search.return_value = []

        linker = RelatedChunksLinker(storage=mock_storage)
        linker.find_related(
            question="What is X?",
            answer="Y",
            include_answer_in_query=False,
        )

        call_args = mock_storage.search.call_args
        query = call_args[1]["query"]
        assert "X" in query
        assert "Y" not in query

    def test_find_related_respects_max_results(self) -> None:
        """Should limit results to max_results."""
        mock_storage = Mock()
        mock_storage.search.return_value = [
            {"text": f"Result {i}", "source": "doc.pdf", "similarity": 0.9 - i * 0.1}
            for i in range(10)
        ]

        linker = RelatedChunksLinker(storage=mock_storage, max_results=3)
        result = linker.find_related(question="What is X?")

        assert result.count <= 3

    def test_find_related_filters_by_threshold(self) -> None:
        """Should filter results below threshold."""
        mock_storage = Mock()
        mock_storage.search.return_value = [
            {"text": "High sim", "source": "a.pdf", "similarity": 0.8},
            {"text": "Low sim", "source": "b.pdf", "similarity": 0.1},
            {"text": "Med sim", "source": "c.pdf", "similarity": 0.5},
        ]

        linker = RelatedChunksLinker(
            storage=mock_storage,
            similarity_threshold=0.4,
        )
        result = linker.find_related(question="What is X?")

        # Should include 0.8 and 0.5, exclude 0.1
        assert result.count == 2
        similarities = [c.similarity for c in result.chunks]
        assert 0.1 not in similarities

    def test_find_related_handles_error(self) -> None:
        """Should handle search errors gracefully."""
        mock_storage = Mock()
        mock_storage.search.side_effect = RuntimeError("Search failed")

        linker = RelatedChunksLinker(storage=mock_storage)
        result = linker.find_related(question="What is X?")

        # Should return empty result, not raise
        assert not result.has_results


class TestRelatedChunksLinkerConversion:
    """Test result conversion."""

    def test_extract_similarity_from_similarity_field(self) -> None:
        """Should extract from similarity field."""
        linker = RelatedChunksLinker()
        result = {"similarity": 0.75}

        similarity = linker._extract_similarity(result)
        assert similarity == 0.75

    def test_extract_similarity_from_score_field(self) -> None:
        """Should extract from score field."""
        linker = RelatedChunksLinker()
        result = {"score": 0.85}

        similarity = linker._extract_similarity(result)
        assert similarity == 0.85

    def test_extract_similarity_from_distance(self) -> None:
        """Should convert distance to similarity."""
        linker = RelatedChunksLinker()
        # Distance 0 = similarity 1, distance 2 = similarity 0
        result = {"distance": 0.5}

        similarity = linker._extract_similarity(result)
        assert 0.5 < similarity < 1.0

    def test_extract_similarity_default(self) -> None:
        """Should use default when no field present."""
        linker = RelatedChunksLinker()
        result = {}

        similarity = linker._extract_similarity(result)
        assert similarity == 0.5

    def test_convert_result_basic(self) -> None:
        """Should convert basic result."""
        linker = RelatedChunksLinker()
        result = {
            "text": "Content",
            "source": "doc.pdf",
            "id": "chunk123",
        }

        chunk = linker._convert_result(result, 0.8)

        assert chunk.text == "Content"
        assert chunk.source == "doc.pdf"
        assert chunk.chunk_id == "chunk123"
        assert chunk.similarity == 0.8

    def test_convert_result_content_field(self) -> None:
        """Should handle 'content' field name."""
        linker = RelatedChunksLinker()
        result = {
            "content": "Alternative content",
            "source": "doc.pdf",
        }

        chunk = linker._convert_result(result, 0.7)

        assert chunk.text == "Alternative content"

    def test_convert_result_nested_source(self) -> None:
        """Should extract source from metadata."""
        linker = RelatedChunksLinker()
        result = {
            "text": "Content",
            "metadata": {"source": "nested_doc.pdf"},
        }

        chunk = linker._convert_result(result, 0.6)

        assert chunk.source == "nested_doc.pdf"


class TestFindRelatedToText:
    """Test find_related_to_text method."""

    def test_find_related_to_text(self) -> None:
        """Should search for arbitrary text."""
        mock_storage = Mock()
        mock_storage.search.return_value = [
            {"text": "Related", "source": "doc.pdf", "similarity": 0.8}
        ]

        linker = RelatedChunksLinker(storage=mock_storage)
        result = linker.find_related_to_text("Some topic to explore")

        assert result.has_results


class TestConvenienceFunction:
    """Test find_related_chunks convenience function."""

    def test_find_related_chunks_no_storage(self) -> None:
        """Should return empty list without storage."""
        chunks = find_related_chunks(
            question="What is X?",
            answer="Y",
        )

        assert chunks == []

    def test_find_related_chunks_with_storage(self) -> None:
        """Should return chunks with storage."""
        mock_storage = Mock()
        mock_storage.search.return_value = [
            {"text": "Content", "source": "doc.pdf", "similarity": 0.8}
        ]

        chunks = find_related_chunks(
            question="What is X?",
            storage=mock_storage,
        )

        assert len(chunks) == 1
        assert isinstance(chunks[0], RelatedChunk)


class TestStorageSetter:
    """Test storage property setter."""

    def test_set_storage_later(self) -> None:
        """Should allow setting storage after init."""
        linker = RelatedChunksLinker()
        assert not linker.is_available

        mock_storage = Mock()
        linker.storage = mock_storage

        assert linker.is_available
        assert linker.storage is mock_storage
