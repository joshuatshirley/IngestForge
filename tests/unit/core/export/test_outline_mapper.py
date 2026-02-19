"""Tests for outline mapper module.

Tests thesis-evidence mapping with LLM relevance scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from ingestforge.core.export.outline_mapper import (
    OutlineMapper,
    OutlinePoint,
    EvidenceMatch,
    MappedOutline,
    MAX_OUTLINE_POINTS,
    MAX_CHUNKS_PER_POINT,
)

# Test fixtures


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock configuration."""
    config = MagicMock()
    config.llm = MagicMock()
    return config


@pytest.fixture
def mapper(mock_config: MagicMock) -> OutlineMapper:
    """Create mapper with mock config."""
    return OutlineMapper(config=mock_config)


@pytest.fixture
def sample_outline_text() -> str:
    """Sample markdown outline."""
    return """# Main Thesis

## First Supporting Point

### Detail A

### Detail B

## Second Supporting Point

- Bullet item one
- Bullet item two
"""


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    chunk_id: str
    content: str
    source_file: str = "test.pdf"
    page: Optional[int] = 1
    section_title: str = ""


# OutlinePoint tests


class TestOutlinePoint:
    """Tests for OutlinePoint dataclass."""

    def test_point_creation(self) -> None:
        """Test creating outline point."""
        point = OutlinePoint(
            id="point_1",
            title="Test Point",
            description="A test description",
            level=1,
        )

        assert point.id == "point_1"
        assert point.title == "Test Point"
        assert point.description == "A test description"
        assert point.level == 1
        assert point.parent_id is None

    def test_point_with_parent(self) -> None:
        """Test point with parent reference."""
        point = OutlinePoint(
            id="point_2",
            title="Child Point",
            level=2,
            parent_id="point_1",
        )

        assert point.parent_id == "point_1"
        assert point.level == 2

    def test_to_prompt_text_simple(self) -> None:
        """Test formatting point for LLM prompt."""
        point = OutlinePoint(id="p1", title="Test Title", level=1)
        result = point.to_prompt_text()

        assert result == "# Test Title"

    def test_to_prompt_text_with_description(self) -> None:
        """Test formatting with description."""
        point = OutlinePoint(
            id="p1",
            title="Test Title",
            description="Some details",
            level=2,
        )
        result = point.to_prompt_text()

        assert "## Test Title" in result
        assert "Some details" in result

    def test_to_prompt_text_level_3(self) -> None:
        """Test level 3 formatting."""
        point = OutlinePoint(id="p1", title="Detail", level=3)
        result = point.to_prompt_text()

        assert result == "### Detail"


# EvidenceMatch tests


class TestEvidenceMatch:
    """Tests for EvidenceMatch dataclass."""

    def test_match_creation(self) -> None:
        """Test creating evidence match."""
        match = EvidenceMatch(
            chunk_id="chunk_1",
            chunk_content="Evidence text here",
            point_id="point_1",
            relevance_score=0.85,
            source_file="doc.pdf",
        )

        assert match.chunk_id == "chunk_1"
        assert match.relevance_score == 0.85
        assert match.source_file == "doc.pdf"

    def test_to_citation_simple(self) -> None:
        """Test simple citation formatting."""
        match = EvidenceMatch(
            chunk_id="c1",
            chunk_content="text",
            point_id="p1",
            relevance_score=0.9,
            source_file="paper.pdf",
        )

        assert match.to_citation() == "paper.pdf"

    def test_to_citation_with_page(self) -> None:
        """Test citation with page number."""
        match = EvidenceMatch(
            chunk_id="c1",
            chunk_content="text",
            point_id="p1",
            relevance_score=0.9,
            source_file="paper.pdf",
            page=42,
        )

        citation = match.to_citation()
        assert "paper.pdf" in citation
        assert "p. 42" in citation

    def test_to_citation_with_section(self) -> None:
        """Test citation with section."""
        match = EvidenceMatch(
            chunk_id="c1",
            chunk_content="text",
            point_id="p1",
            relevance_score=0.9,
            source_file="paper.pdf",
            section="Chapter 3",
        )

        citation = match.to_citation()
        assert "Chapter 3" in citation


# MappedOutline tests


class TestMappedOutline:
    """Tests for MappedOutline dataclass."""

    def test_outline_creation(self) -> None:
        """Test creating mapped outline."""
        outline = MappedOutline(title="Test Research")

        assert outline.title == "Test Research"
        assert outline.points == []
        assert outline.evidence == {}

    def test_get_evidence_for_point_empty(self) -> None:
        """Test getting evidence for point with none."""
        outline = MappedOutline(title="Test")
        result = outline.get_evidence_for_point("nonexistent")

        assert result == []

    def test_get_evidence_for_point_with_matches(self) -> None:
        """Test getting evidence for point with matches."""
        match1 = EvidenceMatch(
            chunk_id="c1", chunk_content="t1", point_id="p1", relevance_score=0.8
        )
        match2 = EvidenceMatch(
            chunk_id="c2", chunk_content="t2", point_id="p1", relevance_score=0.7
        )

        outline = MappedOutline(title="Test", evidence={"p1": [match1, match2]})
        result = outline.get_evidence_for_point("p1")

        assert len(result) == 2
        assert result[0].chunk_id == "c1"

    def test_get_total_evidence_count(self) -> None:
        """Test counting total evidence."""
        match1 = EvidenceMatch(
            chunk_id="c1", chunk_content="t1", point_id="p1", relevance_score=0.8
        )
        match2 = EvidenceMatch(
            chunk_id="c2", chunk_content="t2", point_id="p2", relevance_score=0.7
        )

        outline = MappedOutline(
            title="Test",
            evidence={"p1": [match1], "p2": [match2]},
        )

        assert outline.get_total_evidence_count() == 2


# OutlineMapper tests


class TestOutlineMapperParsing:
    """Tests for outline parsing."""

    def test_parse_outline_empty(self, mapper: OutlineMapper) -> None:
        """Test parsing empty outline."""
        result = mapper.parse_outline("", title="Empty")

        assert result.title == "Empty"
        assert len(result.points) == 0

    def test_parse_outline_simple(
        self, mapper: OutlineMapper, sample_outline_text: str
    ) -> None:
        """Test parsing simple outline."""
        result = mapper.parse_outline(sample_outline_text, title="Test")

        assert result.title == "Test"
        assert len(result.points) > 0

    def test_parse_detects_levels(self, mapper: OutlineMapper) -> None:
        """Test that parser detects heading levels."""
        text = "# Level 1\n## Level 2\n### Level 3"
        result = mapper.parse_outline(text, "Test")

        levels = [p.level for p in result.points]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels

    def test_parse_bullet_points(self, mapper: OutlineMapper) -> None:
        """Test parsing bullet points."""
        text = "# Main\n- Bullet one\n* Bullet two"
        result = mapper.parse_outline(text, "Test")

        # Bullets should be level 2
        bullet_points = [p for p in result.points if p.level == 2]
        assert len(bullet_points) >= 2

    def test_parse_respects_max_points(self, mapper: OutlineMapper) -> None:
        """Test that parsing respects MAX_OUTLINE_POINTS."""
        lines = [f"# Point {i}" for i in range(MAX_OUTLINE_POINTS + 10)]
        text = "\n".join(lines)

        result = mapper.parse_outline(text, "Test")

        assert len(result.points) <= MAX_OUTLINE_POINTS


class TestOutlineMapperMapping:
    """Tests for chunk-to-outline mapping."""

    def test_map_chunks_empty(self, mapper: OutlineMapper) -> None:
        """Test mapping with no chunks."""
        outline = MappedOutline(
            title="Test",
            points=[OutlinePoint(id="p1", title="Point 1", level=1)],
        )

        result = mapper.map_chunks_to_outline(outline, [], min_relevance=0.5)

        assert result.get_total_evidence_count() == 0

    def test_map_chunks_respects_max_per_point(
        self, mapper: OutlineMapper, mock_config: MagicMock
    ) -> None:
        """Test that mapping respects MAX_CHUNKS_PER_POINT."""
        # Create many mock chunks
        chunks = [MockChunk(chunk_id=f"c{i}", content=f"Text {i}") for i in range(30)]

        outline = MappedOutline(
            title="Test",
            points=[OutlinePoint(id="p1", title="Point 1", level=1)],
        )

        # Mock LLM to always return high score
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "9"
        mapper._llm = mock_llm

        result = mapper.map_chunks_to_outline(outline, chunks, min_relevance=0.1)

        for evidence_list in result.evidence.values():
            assert len(evidence_list) <= MAX_CHUNKS_PER_POINT

    def test_map_filters_by_relevance(
        self, mapper: OutlineMapper, mock_config: MagicMock
    ) -> None:
        """Test that low relevance chunks are filtered."""
        chunks = [MockChunk(chunk_id="c1", content="Irrelevant text")]

        outline = MappedOutline(
            title="Test",
            points=[OutlinePoint(id="p1", title="Quantum Physics", level=1)],
        )

        # Mock LLM to return low score
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "2"
        mapper._llm = mock_llm

        result = mapper.map_chunks_to_outline(outline, chunks, min_relevance=0.5)

        # Should have no evidence since score (0.2) < min_relevance (0.5)
        assert result.get_total_evidence_count() == 0


class TestOutlineMapperRelevance:
    """Tests for relevance scoring."""

    def test_parse_relevance_response_valid(self, mapper: OutlineMapper) -> None:
        """Test parsing valid relevance response."""
        result = mapper._parse_relevance_response("7")
        assert result == 0.7

    def test_parse_relevance_response_with_text(self, mapper: OutlineMapper) -> None:
        """Test parsing response with surrounding text."""
        result = mapper._parse_relevance_response("I rate this 8 out of 10")
        assert result == 0.8

    def test_parse_relevance_response_invalid(self, mapper: OutlineMapper) -> None:
        """Test parsing invalid response."""
        result = mapper._parse_relevance_response("not a number")
        assert result == 0.0

    def test_parse_relevance_response_capped(self, mapper: OutlineMapper) -> None:
        """Test that scores are capped at 1.0."""
        result = mapper._parse_relevance_response("15")
        assert result == 1.0


class TestOutlineMapperChunkExtraction:
    """Tests for chunk content extraction."""

    def test_extract_chunk_text_object_content(self, mapper: OutlineMapper) -> None:
        """Test extracting from object with content attr."""
        chunk = MockChunk(chunk_id="c1", content="Test content")
        result = mapper._extract_chunk_text(chunk)

        assert result == "Test content"

    def test_extract_chunk_text_dict(self, mapper: OutlineMapper) -> None:
        """Test extracting from dict."""
        chunk = {"content": "Dict content", "id": "c1"}
        result = mapper._extract_chunk_text(chunk)

        assert result == "Dict content"

    def test_extract_chunk_text_dict_text_key(self, mapper: OutlineMapper) -> None:
        """Test extracting from dict with text key."""
        chunk = {"text": "Text content", "id": "c1"}
        result = mapper._extract_chunk_text(chunk)

        assert result == "Text content"

    def test_extract_chunk_text_string(self, mapper: OutlineMapper) -> None:
        """Test extracting from plain string."""
        result = mapper._extract_chunk_text("Plain string")

        assert result == "Plain string"


class TestOutlineMapperEvidenceCreation:
    """Tests for evidence match creation."""

    def test_create_evidence_from_object(self, mapper: OutlineMapper) -> None:
        """Test creating evidence from chunk object."""
        chunk = MockChunk(
            chunk_id="c1",
            content="Evidence text",
            source_file="paper.pdf",
            page=10,
            section_title="Results",
        )

        result = mapper._create_evidence_match(chunk, "p1", 0.85)

        assert result.chunk_id == "c1"
        assert result.point_id == "p1"
        assert result.relevance_score == 0.85
        assert result.source_file == "paper.pdf"
        assert result.page == 10
        assert result.section == "Results"

    def test_create_evidence_from_dict(self, mapper: OutlineMapper) -> None:
        """Test creating evidence from dict chunk."""
        chunk = {
            "chunk_id": "c2",
            "content": "Dict evidence",
            "source_file": "doc.pdf",
            "page": 5,
        }

        result = mapper._create_evidence_match(chunk, "p2", 0.75)

        assert result.chunk_id == "c2"
        assert result.point_id == "p2"
        assert result.source_file == "doc.pdf"
