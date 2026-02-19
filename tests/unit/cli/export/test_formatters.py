"""Tests for document formatters module.

Tests Markdown and DOCX document generation from mapped outlines."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestforge.core.export.outline_mapper import (
    MappedOutline,
    OutlinePoint,
    EvidenceMatch,
)
from ingestforge.cli.export.formatters import (
    MarkdownFormatter,
    DocxFormatter,
    get_formatter,
    MAX_EVIDENCE_PER_POINT,
    MAX_CONTENT_LENGTH,
)

# Test fixtures


@pytest.fixture
def sample_outline() -> MappedOutline:
    """Create sample outline with evidence."""
    points = [
        OutlinePoint(id="p1", title="Introduction", level=1),
        OutlinePoint(id="p2", title="Methods", level=1),
        OutlinePoint(id="p3", title="Sub-method A", level=2, parent_id="p2"),
    ]

    evidence = {
        "p1": [
            EvidenceMatch(
                chunk_id="c1",
                chunk_content="This is introduction evidence.",
                point_id="p1",
                relevance_score=0.9,
                source_file="paper.pdf",
                page=1,
            ),
        ],
        "p2": [
            EvidenceMatch(
                chunk_id="c2",
                chunk_content="Method description here.",
                point_id="p2",
                relevance_score=0.85,
                source_file="paper.pdf",
                page=5,
            ),
            EvidenceMatch(
                chunk_id="c3",
                chunk_content="Additional method info.",
                point_id="p2",
                relevance_score=0.75,
                source_file="notes.md",
            ),
        ],
    }

    return MappedOutline(title="Test Research", points=points, evidence=evidence)


@pytest.fixture
def empty_outline() -> MappedOutline:
    """Create outline with no evidence."""
    points = [OutlinePoint(id="p1", title="Empty Point", level=1)]
    return MappedOutline(title="Empty Research", points=points)


# MarkdownFormatter tests


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter."""

    def test_format_includes_title(self, sample_outline: MappedOutline) -> None:
        """Test that formatted output includes title."""
        formatter = MarkdownFormatter()
        result = formatter.format(sample_outline)

        assert "# Test Research" in result

    def test_format_includes_metadata(self, sample_outline: MappedOutline) -> None:
        """Test that metadata is included."""
        formatter = MarkdownFormatter(include_metadata=True)
        result = formatter.format(sample_outline)

        assert "Generated:" in result
        assert "Points:" in result
        assert "Evidence:" in result

    def test_format_excludes_metadata(self, sample_outline: MappedOutline) -> None:
        """Test that metadata can be excluded."""
        formatter = MarkdownFormatter(include_metadata=False)
        result = formatter.format(sample_outline)

        assert "Generated:" not in result

    def test_format_includes_points(self, sample_outline: MappedOutline) -> None:
        """Test that outline points are included."""
        formatter = MarkdownFormatter()
        result = formatter.format(sample_outline)

        assert "Introduction" in result
        assert "Methods" in result
        assert "Sub-method A" in result

    def test_format_correct_heading_levels(self, sample_outline: MappedOutline) -> None:
        """Test that heading levels are correct."""
        formatter = MarkdownFormatter()
        result = formatter.format(sample_outline)

        # Title is #, so level 1 points are ##
        assert "## Introduction" in result
        assert "## Methods" in result
        # Level 2 points are ###
        assert "### Sub-method A" in result

    def test_format_includes_evidence(self, sample_outline: MappedOutline) -> None:
        """Test that evidence is included."""
        formatter = MarkdownFormatter()
        result = formatter.format(sample_outline)

        assert "Evidence:" in result
        assert "introduction evidence" in result

    def test_format_evidence_as_blockquotes(
        self, sample_outline: MappedOutline
    ) -> None:
        """Test evidence is formatted as blockquotes."""
        formatter = MarkdownFormatter()
        result = formatter.format(sample_outline)

        assert "> " in result

    def test_format_includes_citations(self, sample_outline: MappedOutline) -> None:
        """Test that citations are included."""
        formatter = MarkdownFormatter(include_citations=True)
        result = formatter.format(sample_outline)

        assert "Sources" in result
        assert "paper.pdf" in result

    def test_format_excludes_citations(self, sample_outline: MappedOutline) -> None:
        """Test that citations can be excluded."""
        formatter = MarkdownFormatter(include_citations=False)
        result = formatter.format(sample_outline)

        assert "## Sources" not in result

    def test_format_empty_outline(self, empty_outline: MappedOutline) -> None:
        """Test formatting outline with no evidence."""
        formatter = MarkdownFormatter()
        result = formatter.format(empty_outline)

        assert "# Empty Research" in result
        assert "Empty Point" in result
        # No evidence section for this point
        assert result.count("**Evidence:**") == 0

    def test_format_truncates_long_content(self) -> None:
        """Test that long content is truncated."""
        long_content = "x" * (MAX_CONTENT_LENGTH + 100)
        evidence = EvidenceMatch(
            chunk_id="c1",
            chunk_content=long_content,
            point_id="p1",
            relevance_score=0.9,
        )
        outline = MappedOutline(
            title="Test",
            points=[OutlinePoint(id="p1", title="Point", level=1)],
            evidence={"p1": [evidence]},
        )

        formatter = MarkdownFormatter()
        result = formatter.format(outline)

        # Should be truncated with ellipsis
        assert "..." in result
        # Should not contain full length
        assert long_content not in result

    def test_save_creates_file(
        self, sample_outline: MappedOutline, tmp_path: Path
    ) -> None:
        """Test saving to file."""
        output_file = tmp_path / "output.md"
        formatter = MarkdownFormatter()

        formatter.save(sample_outline, output_file)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Test Research" in content


# DocxFormatter tests


class TestDocxFormatter:
    """Tests for DocxFormatter."""

    def test_docx_available_check(self) -> None:
        """Test that docx availability is checked."""
        # Should not raise even if docx not installed
        try:
            formatter = DocxFormatter()
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_format_returns_empty(self) -> None:
        """Test that format returns empty string for binary format."""
        try:
            formatter = DocxFormatter()
        except ImportError:
            pytest.skip("python-docx not installed")

        outline = MappedOutline(title="Test", points=[])
        result = formatter.format(outline)

        assert result == ""

    def test_save_creates_file(
        self, sample_outline: MappedOutline, tmp_path: Path
    ) -> None:
        """Test saving DOCX file."""
        try:
            from docx import Document
        except ImportError:
            pytest.skip("python-docx not installed")

        output_file = tmp_path / "output.docx"
        formatter = DocxFormatter()

        formatter.save(sample_outline, output_file)

        assert output_file.exists()

    def test_save_docx_content(
        self, sample_outline: MappedOutline, tmp_path: Path
    ) -> None:
        """Test DOCX content is correct."""
        try:
            from docx import Document
        except ImportError:
            pytest.skip("python-docx not installed")

        output_file = tmp_path / "output.docx"
        formatter = DocxFormatter()
        formatter.save(sample_outline, output_file)

        # Read back and verify
        doc = Document(str(output_file))
        text = "\n".join([p.text for p in doc.paragraphs])

        assert "Test Research" in text
        assert "Introduction" in text


# get_formatter tests


class TestGetFormatter:
    """Tests for get_formatter factory function."""

    def test_get_markdown_formatter(self) -> None:
        """Test getting markdown formatter."""
        formatter = get_formatter("markdown")
        assert isinstance(formatter, MarkdownFormatter)

    def test_get_md_formatter(self) -> None:
        """Test getting formatter with 'md' alias."""
        formatter = get_formatter("md")
        assert isinstance(formatter, MarkdownFormatter)

    def test_get_docx_formatter(self) -> None:
        """Test getting docx formatter."""
        try:
            formatter = get_formatter("docx")
            assert isinstance(formatter, DocxFormatter)
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_get_word_formatter(self) -> None:
        """Test getting formatter with 'word' alias."""
        try:
            formatter = get_formatter("word")
            assert isinstance(formatter, DocxFormatter)
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_get_formatter_case_insensitive(self) -> None:
        """Test that format name is case insensitive."""
        formatter = get_formatter("MARKDOWN")
        assert isinstance(formatter, MarkdownFormatter)

        formatter = get_formatter("Md")
        assert isinstance(formatter, MarkdownFormatter)

    def test_get_formatter_unknown_raises(self) -> None:
        """Test that unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output format"):
            get_formatter("unknown_format")

    def test_get_formatter_with_citations(self) -> None:
        """Test formatter with citations parameter."""
        formatter = get_formatter("markdown", include_citations=True)
        assert isinstance(formatter, MarkdownFormatter)
        assert formatter.include_citations is True

    def test_get_formatter_without_citations(self) -> None:
        """Test formatter without citations."""
        formatter = get_formatter("markdown", include_citations=False)
        assert isinstance(formatter, MarkdownFormatter)
        assert formatter.include_citations is False


# Evidence limit tests


class TestEvidenceLimits:
    """Tests for evidence limiting."""

    def test_max_evidence_per_point_enforced(self) -> None:
        """Test that MAX_EVIDENCE_PER_POINT is respected."""
        # Create more evidence than the limit
        evidence_list = [
            EvidenceMatch(
                chunk_id=f"c{i}",
                chunk_content=f"Evidence {i}",
                point_id="p1",
                relevance_score=0.9 - (i * 0.01),
            )
            for i in range(MAX_EVIDENCE_PER_POINT + 5)
        ]

        outline = MappedOutline(
            title="Test",
            points=[OutlinePoint(id="p1", title="Point", level=1)],
            evidence={"p1": evidence_list},
        )

        formatter = MarkdownFormatter()
        result = formatter.format(outline)

        # Count evidence blockquotes - should be at most MAX_EVIDENCE_PER_POINT
        evidence_count = result.count("> Evidence")
        assert evidence_count <= MAX_EVIDENCE_PER_POINT
