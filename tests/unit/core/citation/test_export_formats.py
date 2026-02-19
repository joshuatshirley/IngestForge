"""Tests for export_formats module (CITE-002.1).

Tests the BibTeX and RIS citation formatters:
- Citation dataclass
- BibTeX formatting
- RIS formatting
- Export functionality
- Metadata conversion
"""

import pytest
from pathlib import Path
import tempfile

from ingestforge.core.citation.export_formats import (
    Citation,
    CitationType,
    BibTeXFormatter,
    RISFormatter,
    format_bibtex,
    format_ris,
    export_bibliography,
    citation_from_metadata,
)


class TestCitation:
    """Test Citation dataclass."""

    def test_citation_creation(self) -> None:
        """Citation should be creatable with minimal fields."""
        citation = Citation(title="Test Article")
        assert citation.title == "Test Article"
        assert citation.authors == []
        assert citation.citation_type == CitationType.MISC

    def test_citation_full(self) -> None:
        """Citation should accept all fields."""
        citation = Citation(
            title="Machine Learning",
            authors=["John Smith", "Jane Doe"],
            year=2024,
            citation_type=CitationType.ARTICLE,
            journal="AI Journal",
            volume="10",
            issue="2",
            pages="1-15",
            doi="10.1234/example",
        )
        assert citation.title == "Machine Learning"
        assert len(citation.authors) == 2
        assert citation.year == 2024

    def test_generate_key_with_author(self) -> None:
        """generate_key should use first author's last name."""
        citation = Citation(
            title="Test",
            authors=["John Smith"],
            year=2024,
        )
        key = citation.generate_key()
        assert key == "smith2024"

    def test_generate_key_no_author(self) -> None:
        """generate_key should use 'unknown' when no authors."""
        citation = Citation(title="Test", year=2024)
        key = citation.generate_key()
        assert key == "unknown2024"

    def test_generate_key_complex_name(self) -> None:
        """generate_key should handle complex names."""
        citation = Citation(
            title="Test",
            authors=["María García-López"],
            year=2024,
        )
        key = citation.generate_key()
        assert "garca" in key or "lpez" in key


class TestBibTeXFormatter:
    """Test BibTeXFormatter class."""

    def test_format_basic(self) -> None:
        """Should format basic citation."""
        citation = Citation(
            title="Test Article",
            authors=["John Smith"],
            year=2024,
        )
        formatter = BibTeXFormatter()
        result = formatter.format(citation)

        assert "@misc{smith2024," in result
        assert "title = {Test Article}" in result
        assert "author = {John Smith}" in result
        assert "year = {2024}" in result

    def test_format_article(self) -> None:
        """Should format article with journal info."""
        citation = Citation(
            title="Research Paper",
            authors=["Alice Johnson"],
            year=2023,
            citation_type=CitationType.ARTICLE,
            journal="Nature",
            volume="100",
            issue="5",
            pages="10-20",
        )
        formatter = BibTeXFormatter()
        result = formatter.format(citation)

        assert "@article{" in result
        assert "journal = {Nature}" in result
        assert "volume = {100}" in result
        assert "number = {5}" in result
        assert "pages = {10-20}" in result

    def test_format_multiple_authors(self) -> None:
        """Should join multiple authors with 'and'."""
        citation = Citation(
            title="Collaborative Work",
            authors=["Alice", "Bob", "Charlie"],
            year=2024,
        )
        formatter = BibTeXFormatter()
        result = formatter.format(citation)

        assert "author = {Alice and Bob and Charlie}" in result

    def test_escape_special_chars(self) -> None:
        """Should escape special BibTeX characters."""
        citation = Citation(
            title="Using & and % in LaTeX",
            authors=["Test Author"],
            year=2024,
        )
        formatter = BibTeXFormatter()
        result = formatter.format(citation)

        assert r"\&" in result
        assert r"\%" in result

    def test_format_with_doi(self) -> None:
        """Should include DOI field."""
        citation = Citation(
            title="Test",
            doi="10.1234/example.2024",
            year=2024,
        )
        formatter = BibTeXFormatter()
        result = formatter.format(citation)

        assert "doi = {10.1234/example.2024}" in result

    def test_format_many(self) -> None:
        """Should format multiple citations."""
        citations = [
            Citation(title="First", year=2024),
            Citation(title="Second", year=2023),
        ]
        formatter = BibTeXFormatter()
        result = formatter.format_many(citations)

        assert "First" in result
        assert "Second" in result
        assert result.count("@misc{") == 2


class TestRISFormatter:
    """Test RISFormatter class."""

    def test_format_basic(self) -> None:
        """Should format basic RIS entry."""
        citation = Citation(
            title="Test Article",
            authors=["John Smith"],
            year=2024,
        )
        formatter = RISFormatter()
        result = formatter.format(citation)

        assert "TY  - GEN" in result
        assert "TI  - Test Article" in result
        assert "AU  - John Smith" in result
        assert "PY  - 2024" in result
        assert "ER  - " in result

    def test_format_journal_article(self) -> None:
        """Should format journal article as JOUR type."""
        citation = Citation(
            title="Journal Paper",
            authors=["Jane Doe"],
            year=2023,
            citation_type=CitationType.ARTICLE,
            journal="Science",
            volume="50",
            issue="3",
        )
        formatter = RISFormatter()
        result = formatter.format(citation)

        assert "TY  - JOUR" in result
        assert "JO  - Science" in result
        assert "VL  - 50" in result
        assert "IS  - 3" in result

    def test_format_pages_range(self) -> None:
        """Should split page range into SP and EP."""
        citation = Citation(
            title="Test",
            pages="100-150",
            year=2024,
        )
        formatter = RISFormatter()
        result = formatter.format(citation)

        assert "SP  - 100" in result
        assert "EP  - 150" in result

    def test_format_single_page(self) -> None:
        """Should handle single page number."""
        citation = Citation(
            title="Test",
            pages="42",
            year=2024,
        )
        formatter = RISFormatter()
        result = formatter.format(citation)

        assert "SP  - 42" in result
        assert "EP  -" not in result

    def test_format_multiple_authors(self) -> None:
        """Should have separate AU line per author."""
        citation = Citation(
            title="Team Work",
            authors=["Alice", "Bob", "Charlie"],
            year=2024,
        )
        formatter = RISFormatter()
        result = formatter.format(citation)

        assert result.count("AU  - ") == 3
        assert "AU  - Alice" in result
        assert "AU  - Bob" in result
        assert "AU  - Charlie" in result

    def test_format_keywords(self) -> None:
        """Should include keywords as KW lines."""
        citation = Citation(
            title="Tagged Article",
            keywords=["machine learning", "AI", "neural networks"],
            year=2024,
        )
        formatter = RISFormatter()
        result = formatter.format(citation)

        assert result.count("KW  - ") == 3
        assert "KW  - machine learning" in result

    def test_format_many(self) -> None:
        """Should format multiple citations."""
        citations = [
            Citation(title="First", year=2024),
            Citation(title="Second", year=2023),
        ]
        formatter = RISFormatter()
        result = formatter.format_many(citations)

        assert "First" in result
        assert "Second" in result
        assert result.count("TY  - ") == 2
        assert result.count("ER  - ") == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_format_bibtex(self) -> None:
        """format_bibtex should work."""
        citation = Citation(title="Test", year=2024)
        result = format_bibtex(citation)
        assert "@misc{" in result

    def test_format_ris(self) -> None:
        """format_ris should work."""
        citation = Citation(title="Test", year=2024)
        result = format_ris(citation)
        assert "TY  - " in result


class TestExportBibliography:
    """Test export_bibliography function."""

    def test_export_bibtex(self) -> None:
        """Should export BibTeX file."""
        citations = [
            Citation(title="Article One", year=2024),
            Citation(title="Article Two", year=2023),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "refs.bib"
            export_bibliography(citations, output, format="bibtex")

            assert output.exists()
            content = output.read_text()
            assert "Article One" in content
            assert "Article Two" in content

    def test_export_ris(self) -> None:
        """Should export RIS file."""
        citations = [Citation(title="Test Paper", year=2024)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "refs.ris"
            export_bibliography(citations, output, format="ris")

            assert output.exists()
            content = output.read_text()
            assert "Test Paper" in content
            assert "TY  - " in content

    def test_export_invalid_format(self) -> None:
        """Should raise error for invalid format."""
        citations = [Citation(title="Test", year=2024)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "refs.txt"

            with pytest.raises(ValueError, match="Unsupported format"):
                export_bibliography(citations, output, format="invalid")


class TestCitationFromMetadata:
    """Test citation_from_metadata function."""

    def test_basic_metadata(self) -> None:
        """Should create citation from basic metadata."""
        metadata = {
            "title": "Test Document",
            "author": "John Smith",
            "year": "2024",
        }
        citation = citation_from_metadata(metadata)

        assert citation.title == "Test Document"
        assert citation.authors == ["John Smith"]
        assert citation.year == 2024

    def test_multiple_authors_semicolon(self) -> None:
        """Should split authors by semicolon."""
        metadata = {
            "title": "Collaboration",
            "author": "Alice; Bob; Charlie",
        }
        citation = citation_from_metadata(metadata)

        assert len(citation.authors) == 3
        assert "Alice" in citation.authors

    def test_multiple_authors_and(self) -> None:
        """Should split authors by 'and'."""
        metadata = {
            "title": "Team Work",
            "author": "Alice and Bob and Charlie",
        }
        citation = citation_from_metadata(metadata)

        assert len(citation.authors) == 3

    def test_authors_list(self) -> None:
        """Should handle author list."""
        metadata = {
            "title": "Listed Authors",
            "author": ["Alice", "Bob"],
        }
        citation = citation_from_metadata(metadata)

        assert citation.authors == ["Alice", "Bob"]

    def test_pdf_source(self) -> None:
        """Should detect PDF as article type."""
        metadata = {
            "source": "document.pdf",
            "title": "PDF Document",
        }
        citation = citation_from_metadata(metadata)

        assert citation.citation_type == CitationType.ARTICLE

    def test_url_source(self) -> None:
        """Should detect URL as webpage type."""
        metadata = {
            "source": "https://example.com/page",
            "title": "Web Page",
        }
        citation = citation_from_metadata(metadata)

        assert citation.citation_type == CitationType.WEBPAGE

    def test_year_from_date(self) -> None:
        """Should extract year from date string."""
        metadata = {
            "title": "Dated Document",
            "date": "2024-03-15",
        }
        citation = citation_from_metadata(metadata)

        assert citation.year == 2024

    def test_fallback_title(self) -> None:
        """Should use source as fallback title."""
        metadata = {"source": "document.pdf"}
        citation = citation_from_metadata(metadata)

        assert citation.title == "document.pdf"
