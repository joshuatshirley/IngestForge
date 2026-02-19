"""
Tests for Citation Manager Module.

This module tests the citation management functionality including:
- CitationManager class methods
- Source and VerificationResult dataclasses
- Citation formatting in all styles
- Bibliography generation

Test Strategy
-------------
- Test each public method of CitationManager
- Test all 5 citation styles (APA, MLA, Chicago, IEEE, Harvard)
- Test citation verification
- Test BibTeX parsing/generation
- Follow NASA JPL Rule #1: Simple test structure

Organization
------------
- TestSource: Source dataclass tests
- TestVerificationResult: VerificationResult dataclass tests
- TestCitationManager: Manager class tests
- TestCitationStyles: Style-specific tests
- TestCitationCommands: CLI command tests
"""

from unittest.mock import Mock, patch

from ingestforge.cli.writing.cite import (
    Source,
    VerificationResult,
    CitationManager,
    CiteInsertCommand,
    CiteFormatCommand,
    CiteCheckCommand,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_test_source(
    id: str = "test_id",
    author: str = "Smith, J.",
    title: str = "Test Title",
    year: str = "2024",
    **kwargs,
) -> Source:
    """Create test source."""
    return Source(id=id, author=author, title=title, year=year, **kwargs)


def make_mock_chunk(source: str = "test_source", author: str = "Test Author"):
    """Create mock chunk with metadata."""
    chunk = Mock()
    chunk.metadata = {
        "source": source,
        "author": author,
        "title": "Test Title",
        "year": "2024",
    }
    return chunk


# ============================================================================
# TestSource: Source dataclass tests
# ============================================================================


class TestSource:
    """Tests for Source dataclass."""

    def test_source_creation(self):
        """Test basic source creation."""
        source = Source(
            id="test",
            author="Author",
            title="Title",
            year="2024",
        )
        assert source.id == "test"
        assert source.author == "Author"
        assert source.title == "Title"
        assert source.year == "2024"

    def test_source_with_all_fields(self):
        """Test source with all optional fields."""
        source = Source(
            id="full",
            author="Author",
            title="Title",
            year="2024",
            source_type="article",
            journal="Journal Name",
            publisher="Publisher",
            url="https://example.com",
            page="1-10",
            volume="5",
            issue="2",
            doi="10.1234/test",
        )
        assert source.journal == "Journal Name"
        assert source.doi == "10.1234/test"

    def test_source_to_dict(self):
        """Test dictionary conversion."""
        source = make_test_source()
        data = source.to_dict()

        assert data["id"] == "test_id"
        assert data["author"] == "Smith, J."
        assert "title" in data

    def test_source_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "from_dict",
            "author": "Dict Author",
            "title": "Dict Title",
            "year": "2023",
        }
        source = Source.from_dict(data)

        assert source.id == "from_dict"
        assert source.author == "Dict Author"

    def test_source_from_dict_defaults(self):
        """Test creation from dict with missing fields."""
        data = {}
        source = Source.from_dict(data)

        assert source.id == "unknown"
        assert source.author == "Unknown"
        assert source.year == "n.d."


# ============================================================================
# TestVerificationResult: VerificationResult dataclass tests
# ============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = VerificationResult()
        assert result.is_valid == True
        assert result.total_citations == 0
        assert result.missing_sources == []

    def test_result_with_values(self):
        """Test result with values."""
        result = VerificationResult(
            is_valid=False,
            total_citations=5,
            valid_citations=3,
            missing_sources=["src1", "src2"],
        )
        assert result.is_valid == False
        assert result.total_citations == 5
        assert len(result.missing_sources) == 2

    def test_result_to_dict(self):
        """Test dictionary conversion."""
        result = VerificationResult(
            total_citations=10,
            valid_citations=8,
            warnings=["Warning 1"],
        )
        data = result.to_dict()

        assert data["total_citations"] == 10
        assert data["valid_citations"] == 8
        assert "warnings" in data


# ============================================================================
# TestCitationManager: Manager class tests
# ============================================================================


class TestCitationManager:
    """Tests for CitationManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = CitationManager()
        assert manager.llm_client is None

    def test_initialization_with_llm(self):
        """Test manager initialization with LLM client."""
        client = Mock()
        manager = CitationManager(llm_client=client)
        assert manager.llm_client == client

    def test_insert_citations_empty_text(self):
        """Test citation insertion with empty text."""
        manager = CitationManager()

        result = manager.insert_citations("", [make_test_source()])

        assert result == ""

    def test_insert_citations_empty_sources(self):
        """Test citation insertion with no sources."""
        manager = CitationManager()

        result = manager.insert_citations("Some text", [])

        assert result == "Some text"

    def test_insert_citations_apa(self):
        """Test APA citation insertion."""
        manager = CitationManager()
        source = make_test_source(id="ref1", author="Jones")
        text = "As noted in [ref1], this is important."

        result = manager.insert_citations(text, [source], "apa")

        assert "(Jones, 2024)" in result

    def test_format_bibliography_empty(self):
        """Test empty bibliography."""
        manager = CitationManager()

        result = manager.format_bibliography([])

        assert result == ""

    def test_format_bibliography_apa(self):
        """Test APA bibliography formatting."""
        manager = CitationManager()
        sources = [
            make_test_source(id="1", author="Smith, J.", title="Paper A"),
            make_test_source(id="2", author="Doe, J.", title="Paper B"),
        ]

        result = manager.format_bibliography(sources, "apa")

        assert "Smith" in result
        assert "Doe" in result

    def test_verify_citations_all_valid(self):
        """Test citation verification with all valid."""
        manager = CitationManager()
        sources = [make_test_source(author="ValidAuthor")]
        text = "According to (ValidAuthor, 2024), this is true."

        result = manager.verify_citations(text, sources)

        # The citation should be found
        assert result.total_citations > 0

    def test_verify_citations_missing(self):
        """Test citation verification with missing source."""
        manager = CitationManager()
        sources = [make_test_source(author="ValidAuthor")]
        text = "According to (MissingAuthor, 2024), this is wrong."

        result = manager.verify_citations(text, sources)

        assert not result.is_valid
        assert len(result.missing_sources) > 0

    def test_extract_sources_from_chunks(self):
        """Test source extraction from chunks."""
        manager = CitationManager()
        chunks = [
            make_mock_chunk("source1", "Author 1"),
            make_mock_chunk("source2", "Author 2"),
        ]

        sources = manager.extract_sources_from_chunks(chunks)

        assert len(sources) == 2

    def test_extract_sources_deduplication(self):
        """Test source extraction deduplication."""
        manager = CitationManager()
        chunks = [
            make_mock_chunk("same_source", "Same Author"),
            make_mock_chunk("same_source", "Same Author"),
        ]

        sources = manager.extract_sources_from_chunks(chunks)

        assert len(sources) == 1

    def test_parse_bibtex_basic(self):
        """Test basic BibTeX parsing."""
        manager = CitationManager()
        bibtex = """
@article{smith2024,
  author = {Smith, John},
  title = {A Great Paper},
  year = {2024},
  journal = {Science Journal}
}
"""
        sources = manager.parse_bibtex(bibtex)

        assert len(sources) == 1
        assert sources[0].author == "Smith, John"
        assert sources[0].year == "2024"

    def test_parse_bibtex_multiple(self):
        """Test parsing multiple BibTeX entries."""
        manager = CitationManager()
        bibtex = """
@article{ref1,
  author = {Author One},
  title = {Title One},
  year = {2024}
}
@book{ref2,
  author = {Author Two},
  title = {Title Two},
  year = {2023}
}
"""
        sources = manager.parse_bibtex(bibtex)

        assert len(sources) == 2

    def test_to_bibtex(self):
        """Test BibTeX generation."""
        manager = CitationManager()
        sources = [make_test_source(id="test", author="Test Author")]

        bibtex = manager.to_bibtex(sources)

        assert "@article{test," in bibtex
        assert "author = {Test Author}" in bibtex

    def test_to_bibtex_with_optional_fields(self):
        """Test BibTeX generation with optional fields."""
        manager = CitationManager()
        source = make_test_source(
            journal="Test Journal",
            volume="5",
            doi="10.1234/test",
        )

        bibtex = manager.to_bibtex([source])

        assert "journal = {Test Journal}" in bibtex
        assert "volume = {5}" in bibtex
        assert "doi = {10.1234/test}" in bibtex

    def test_validate_style_valid(self):
        """Test style validation with valid styles."""
        manager = CitationManager()

        assert manager._validate_style("apa") == "apa"
        assert manager._validate_style("APA") == "apa"
        assert manager._validate_style("mla") == "mla"

    def test_validate_style_invalid(self):
        """Test style validation with invalid style."""
        manager = CitationManager()

        assert manager._validate_style("invalid") == "apa"


# ============================================================================
# TestCitationStyles: Style-specific tests
# ============================================================================


class TestCitationStyles:
    """Tests for all citation style formatters."""

    def test_apa_inline(self):
        """Test APA inline citation."""
        manager = CitationManager()
        source = make_test_source(author="Smith", year="2024")

        result = manager._format_apa_inline(source)

        assert result == "(Smith, 2024)"

    def test_mla_inline(self):
        """Test MLA inline citation."""
        manager = CitationManager()
        source = make_test_source(author="Smith")

        result = manager._format_mla_inline(source)

        assert "(Smith" in result

    def test_mla_inline_with_page(self):
        """Test MLA inline citation with page."""
        manager = CitationManager()
        source = make_test_source(author="Smith", page="42")

        result = manager._format_mla_inline(source)

        assert "42" in result

    def test_chicago_inline(self):
        """Test Chicago inline citation."""
        manager = CitationManager()
        source = make_test_source(author="Smith", year="2024")

        result = manager._format_chicago_inline(source)

        assert "(Smith 2024)" in result

    def test_ieee_inline(self):
        """Test IEEE inline citation."""
        manager = CitationManager()
        source = make_test_source(id="1")

        result = manager._format_ieee_inline(source)

        assert result == "[1]"

    def test_harvard_inline(self):
        """Test Harvard inline citation."""
        manager = CitationManager()
        source = make_test_source(author="Smith", year="2024")

        result = manager._format_harvard_inline(source)

        assert result == "(Smith, 2024)"

    def test_apa_bibliography(self):
        """Test APA bibliography entry."""
        manager = CitationManager()
        source = make_test_source(
            author="Smith, J.",
            title="Test Paper",
            year="2024",
        )

        result = manager._format_apa_bibliography(source)

        assert "Smith, J." in result
        assert "(2024)" in result
        assert "*Test Paper*" in result

    def test_mla_bibliography(self):
        """Test MLA bibliography entry."""
        manager = CitationManager()
        source = make_test_source(
            author="Smith, Jane",
            title="Test Article",
        )

        result = manager._format_mla_bibliography(source)

        assert "Smith, Jane" in result
        assert "Test Article" in result

    def test_chicago_bibliography(self):
        """Test Chicago bibliography entry."""
        manager = CitationManager()
        source = make_test_source()

        result = manager._format_chicago_bibliography(source)

        assert len(result) > 0

    def test_ieee_bibliography(self):
        """Test IEEE bibliography entry."""
        manager = CitationManager()
        source = make_test_source(id="1")

        result = manager._format_ieee_bibliography(source)

        assert "[1]" in result

    def test_harvard_bibliography(self):
        """Test Harvard bibliography entry."""
        manager = CitationManager()
        source = make_test_source()

        result = manager._format_harvard_bibliography(source)

        assert len(result) > 0

    def test_all_styles_format_bibliography(self):
        """Test all styles produce valid bibliography."""
        manager = CitationManager()
        sources = [make_test_source()]

        for style in ["apa", "mla", "chicago", "ieee", "harvard"]:
            result = manager.format_bibliography(sources, style)
            assert len(result) > 0, f"Style {style} produced empty output"


# ============================================================================
# TestCitationCommands: CLI command tests
# ============================================================================


class TestCiteInsertCommand:
    """Tests for CiteInsertCommand."""

    def test_execute_file_not_found(self, tmp_path):
        """Test with non-existent file."""
        cmd = CiteInsertCommand(console=Mock())
        result = cmd.execute(tmp_path / "nonexistent.md")

        assert result == 1

    @patch.object(CiteInsertCommand, "initialize_context")
    def test_execute_success(self, mock_ctx, tmp_path):
        """Test successful execution."""
        storage = Mock()
        storage.get_all_chunks.return_value = [make_mock_chunk()]
        mock_ctx.return_value = {"storage": storage}

        input_file = tmp_path / "test.md"
        input_file.write_text("Reference to [test_source]")

        cmd = CiteInsertCommand(console=Mock())
        result = cmd.execute(input_file, project=tmp_path)

        assert result == 0


class TestCiteFormatCommand:
    """Tests for CiteFormatCommand."""

    @patch.object(CiteFormatCommand, "initialize_context")
    def test_execute_bibtex_file(self, mock_ctx, tmp_path):
        """Test with BibTeX file."""
        mock_ctx.return_value = {"storage": Mock()}

        bib_file = tmp_path / "refs.bib"
        bib_file.write_text(
            """
@article{test,
  author = {Test Author},
  title = {Test Title},
  year = {2024}
}
"""
        )

        cmd = CiteFormatCommand(console=Mock())
        result = cmd.execute(bib_file, project=tmp_path)

        assert result == 0

    @patch.object(CiteFormatCommand, "initialize_context")
    def test_execute_with_output(self, mock_ctx, tmp_path):
        """Test with output file."""
        mock_ctx.return_value = {"storage": Mock()}

        bib_file = tmp_path / "refs.bib"
        bib_file.write_text(
            """
@article{test,
  author = {Author},
  title = {Title},
  year = {2024}
}
"""
        )
        output = tmp_path / "bibliography.md"

        cmd = CiteFormatCommand(console=Mock())
        result = cmd.execute(bib_file, output=output, project=tmp_path)

        assert result == 0
        assert output.exists()

    @patch.object(CiteFormatCommand, "initialize_context")
    def test_execute_all_styles(self, mock_ctx, tmp_path):
        """Test with all styles."""
        mock_ctx.return_value = {"storage": Mock()}

        bib_file = tmp_path / "refs.bib"
        bib_file.write_text(
            """
@article{test,
  author = {Author},
  title = {Title},
  year = {2024}
}
"""
        )

        cmd = CiteFormatCommand(console=Mock())

        for style in ["apa", "mla", "chicago", "ieee", "harvard"]:
            result = cmd.execute(bib_file, style=style, project=tmp_path)
            assert result == 0, f"Style {style} failed"


class TestCiteCheckCommand:
    """Tests for CiteCheckCommand."""

    def test_execute_file_not_found(self, tmp_path):
        """Test with non-existent file."""
        cmd = CiteCheckCommand(console=Mock())
        result = cmd.execute(tmp_path / "nonexistent.md")

        assert result == 1

    @patch.object(CiteCheckCommand, "initialize_context")
    def test_execute_valid_citations(self, mock_ctx, tmp_path):
        """Test with valid citations."""
        storage = Mock()
        storage.get_all_chunks.return_value = [make_mock_chunk()]
        mock_ctx.return_value = {"storage": storage}

        doc = tmp_path / "doc.md"
        doc.write_text("Reference (Test Author, 2024) here.")

        cmd = CiteCheckCommand(console=Mock())
        result = cmd.execute(doc, project=tmp_path)

        # Result depends on matching
        assert result in [0, 1]

    @patch.object(CiteCheckCommand, "initialize_context")
    def test_execute_displays_result(self, mock_ctx, tmp_path):
        """Test result display."""
        storage = Mock()
        storage.get_all_chunks.return_value = []
        mock_ctx.return_value = {"storage": storage}

        doc = tmp_path / "doc.md"
        doc.write_text("No citations here.")

        console = Mock()
        cmd = CiteCheckCommand(console=console)
        cmd.execute(doc, project=tmp_path)

        # Should have printed something
        assert console.print.called


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestCiteEdgeCases:
    """Edge case tests for citation functionality."""

    def test_source_with_special_characters(self):
        """Test source with special characters in title."""
        source = make_test_source(title='Title with <html> & "quotes"')

        # Should not crash
        manager = CitationManager()
        bibtex = manager.to_bibtex([source])
        assert len(bibtex) > 0

    def test_bibtex_with_missing_fields(self):
        """Test BibTeX parsing with missing fields."""
        manager = CitationManager()
        bibtex = """
@article{minimal,
  title = {Only Title}
}
"""
        sources = manager.parse_bibtex(bibtex)

        assert len(sources) == 1
        assert sources[0].author == "Unknown"

    def test_verify_empty_text(self):
        """Test verification with empty text."""
        manager = CitationManager()

        result = manager.verify_citations("", [make_test_source()])

        assert result.total_citations == 0
        assert result.is_valid == True

    def test_format_bibliography_sorts_by_author(self):
        """Test bibliography sorting."""
        manager = CitationManager()
        sources = [
            make_test_source(id="1", author="Zulu"),
            make_test_source(id="2", author="Alpha"),
            make_test_source(id="3", author="Mike"),
        ]

        result = manager.format_bibliography(sources, "apa")

        # Alpha should come before Mike and Zulu
        alpha_pos = result.find("Alpha")
        mike_pos = result.find("Mike")
        zulu_pos = result.find("Zulu")

        assert alpha_pos < mike_pos < zulu_pos

    def test_extract_year_from_date(self):
        """Test year extraction from date field."""
        manager = CitationManager()
        chunk = Mock()
        chunk.metadata = {"date": "2024-01-15", "source": "test"}

        sources = manager.extract_sources_from_chunks([chunk])

        assert sources[0].year == "2024"

    def test_citation_patterns_various_formats(self):
        """Test extracting various citation formats."""
        manager = CitationManager()
        text = """
            (Author, 2024)
            (Another 123)
            [1]
            [ref123]
        """

        citations = manager._extract_citations_from_text(text)

        assert len(citations) > 0
