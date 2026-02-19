"""Tests for CSL style engine.

Tests bibliography generation functionality."""

from __future__ import annotations


from ingestforge.citation.csl_engine import (
    CitationType,
    OutputFormat,
    Author,
    DateParts,
    Reference,
    Citation,
    Bibliography,
    CSLEngine,
    create_engine,
    format_references,
    MAX_REFERENCES,
)

# Test fixtures


def make_reference(
    id: str = "ref1",
    title: str = "Test Article",
) -> Reference:
    """Create a test reference."""
    return Reference(
        id=id,
        title=title,
        authors=[
            Author(family="Smith", given="John"),
            Author(family="Jones", given="Jane"),
        ],
        citation_type=CitationType.ARTICLE,
        issued=DateParts(year=2023, month=6),
        container_title="Journal of Testing",
        volume="10",
        issue="2",
        page="100-120",
        doi="10.1234/test.2023.001",
    )


# CitationType tests


class TestCitationType:
    """Tests for CitationType enum."""

    def test_types_defined(self) -> None:
        """Test all types are defined."""
        types = [t.value for t in CitationType]

        assert "article-journal" in types
        assert "book" in types
        assert "webpage" in types

    def test_type_count(self) -> None:
        """Test correct number of types."""
        assert len(CitationType) == 8


# OutputFormat tests


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_formats_defined(self) -> None:
        """Test all formats are defined."""
        formats = [f.value for f in OutputFormat]

        assert "text" in formats
        assert "html" in formats
        assert "markdown" in formats


# Author tests


class TestAuthor:
    """Tests for Author dataclass."""

    def test_author_creation(self) -> None:
        """Test creating an author."""
        author = Author(family="Smith", given="John")

        assert author.family == "Smith"
        assert author.given == "John"

    def test_to_csl(self) -> None:
        """Test converting to CSL format."""
        author = Author(family="Smith", given="John", suffix="Jr.")

        csl = author.to_csl()

        assert csl["family"] == "Smith"
        assert csl["given"] == "John"
        assert csl["suffix"] == "Jr."

    def test_str(self) -> None:
        """Test string conversion."""
        author = Author(family="Smith", given="John")

        assert str(author) == "Smith, John"

    def test_str_no_given(self) -> None:
        """Test string with no given name."""
        author = Author(family="Smith")

        assert str(author) == "Smith"


# DateParts tests


class TestDateParts:
    """Tests for DateParts dataclass."""

    def test_date_creation(self) -> None:
        """Test creating a date."""
        date = DateParts(year=2023, month=6, day=15)

        assert date.year == 2023
        assert date.month == 6

    def test_to_csl_full(self) -> None:
        """Test full date CSL conversion."""
        date = DateParts(year=2023, month=6, day=15)

        csl = date.to_csl()

        assert csl["date-parts"] == [[2023, 6, 15]]

    def test_to_csl_year_only(self) -> None:
        """Test year-only CSL conversion."""
        date = DateParts(year=2023)

        csl = date.to_csl()

        assert csl["date-parts"] == [[2023]]


# Reference tests


class TestReference:
    """Tests for Reference dataclass."""

    def test_reference_creation(self) -> None:
        """Test creating a reference."""
        ref = make_reference()

        assert ref.id == "ref1"
        assert ref.title == "Test Article"
        assert len(ref.authors) == 2

    def test_to_csl(self) -> None:
        """Test CSL conversion."""
        ref = make_reference()

        csl = ref.to_csl()

        assert csl["id"] == "ref1"
        assert csl["type"] == "article-journal"
        assert "author" in csl
        assert "DOI" in csl

    def test_title_truncation(self) -> None:
        """Test long title is truncated."""
        long_title = "x" * 1000
        ref = Reference(id="test", title=long_title)

        assert len(ref.title) == 500


# Citation tests


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self) -> None:
        """Test creating a citation."""
        citation = Citation(reference_ids=["ref1", "ref2"])

        assert len(citation.reference_ids) == 2

    def test_citation_with_locator(self) -> None:
        """Test citation with page locator."""
        citation = Citation(
            reference_ids=["ref1"],
            locator="42",
            locator_type="page",
        )

        csl = citation.to_csl()

        assert csl[0]["locator"] == "42"
        assert csl[0]["label"] == "page"

    def test_citation_with_prefix(self) -> None:
        """Test citation with prefix/suffix."""
        citation = Citation(
            reference_ids=["ref1"],
            prefix="see",
            suffix="for details",
        )

        csl = citation.to_csl()

        assert csl[0]["prefix"] == "see"
        assert csl[0]["suffix"] == "for details"


# Bibliography tests


class TestBibliography:
    """Tests for Bibliography dataclass."""

    def test_bibliography_creation(self) -> None:
        """Test creating a bibliography."""
        bib = Bibliography(
            entries=["Entry 1", "Entry 2"],
            format=OutputFormat.TEXT,
            style="apa",
            reference_count=2,
        )

        assert bib.reference_count == 2

    def test_to_string(self) -> None:
        """Test converting to string."""
        bib = Bibliography(
            entries=["Entry 1", "Entry 2"],
            format=OutputFormat.TEXT,
            style="apa",
            reference_count=2,
        )

        result = bib.to_string()

        assert "Entry 1" in result
        assert "Entry 2" in result

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        bib = Bibliography(
            entries=["Entry"],
            format=OutputFormat.HTML,
            style="mla",
            reference_count=1,
        )

        d = bib.to_dict()

        assert d["format"] == "html"
        assert d["style"] == "mla"


# CSLEngine tests


class TestCSLEngine:
    """Tests for CSLEngine class."""

    def test_engine_creation(self) -> None:
        """Test creating an engine."""
        engine = CSLEngine()

        assert engine.style == "apa"
        assert engine.reference_count == 0

    def test_engine_with_style(self) -> None:
        """Test engine with custom style."""
        engine = CSLEngine(style="mla")

        assert engine.style == "mla"

    def test_add_reference(self) -> None:
        """Test adding a reference."""
        engine = CSLEngine()
        ref = make_reference()

        result = engine.add_reference(ref)

        assert result is True
        assert engine.reference_count == 1

    def test_remove_reference(self) -> None:
        """Test removing a reference."""
        engine = CSLEngine()
        ref = make_reference()
        engine.add_reference(ref)

        result = engine.remove_reference("ref1")

        assert result is True
        assert engine.reference_count == 0

    def test_get_reference(self) -> None:
        """Test getting a reference."""
        engine = CSLEngine()
        ref = make_reference()
        engine.add_reference(ref)

        result = engine.get_reference("ref1")

        assert result is not None
        assert result.title == "Test Article"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent reference."""
        engine = CSLEngine()

        result = engine.get_reference("missing")

        assert result is None


class TestCitationFormatting:
    """Tests for citation formatting."""

    def test_format_apa_citation(self) -> None:
        """Test APA citation format."""
        engine = CSLEngine(style="apa")
        ref = make_reference()
        engine.add_reference(ref)
        citation = Citation(reference_ids=["ref1"])

        result = engine.format_citation(citation)

        assert "Smith" in result
        assert "2023" in result

    def test_format_ieee_citation(self) -> None:
        """Test IEEE citation format."""
        engine = CSLEngine(style="ieee")
        ref = make_reference()
        engine.add_reference(ref)
        citation = Citation(reference_ids=["ref1"])

        result = engine.format_citation(citation)

        assert "[" in result
        assert "]" in result

    def test_format_empty_citation(self) -> None:
        """Test empty citation."""
        engine = CSLEngine()
        citation = Citation(reference_ids=[])

        result = engine.format_citation(citation)

        assert result == ""

    def test_format_missing_reference(self) -> None:
        """Test citation with missing reference."""
        engine = CSLEngine()
        citation = Citation(reference_ids=["missing"])

        result = engine.format_citation(citation)

        assert result == "[?]"


class TestBibliographyGeneration:
    """Tests for bibliography generation."""

    def test_generate_apa(self) -> None:
        """Test APA bibliography."""
        engine = CSLEngine(style="apa")
        engine.add_reference(make_reference("ref1", "First Article"))
        engine.add_reference(make_reference("ref2", "Second Article"))

        bib = engine.generate_bibliography()

        assert bib.reference_count == 2
        assert len(bib.entries) == 2

    def test_generate_mla(self) -> None:
        """Test MLA bibliography."""
        engine = CSLEngine(style="mla")
        engine.add_reference(make_reference())

        bib = engine.generate_bibliography()

        assert bib.style == "mla"

    def test_generate_ieee(self) -> None:
        """Test IEEE bibliography."""
        engine = CSLEngine(style="ieee")
        engine.add_reference(make_reference())

        bib = engine.generate_bibliography()

        assert bib.style == "ieee"

    def test_generate_html_format(self) -> None:
        """Test HTML output format."""
        engine = CSLEngine()
        engine.add_reference(make_reference())

        bib = engine.generate_bibliography(format=OutputFormat.HTML)

        assert bib.format == OutputFormat.HTML
        assert "<i>" in bib.entries[0]

    def test_generate_markdown_format(self) -> None:
        """Test Markdown output format."""
        engine = CSLEngine()
        engine.add_reference(make_reference())

        bib = engine.generate_bibliography(format=OutputFormat.MARKDOWN)

        assert "*" in bib.entries[0]

    def test_generate_specific_refs(self) -> None:
        """Test generating specific references."""
        engine = CSLEngine()
        engine.add_reference(make_reference("a", "Article A"))
        engine.add_reference(make_reference("b", "Article B"))
        engine.add_reference(make_reference("c", "Article C"))

        bib = engine.generate_bibliography(ref_ids=["a", "c"])

        assert bib.reference_count == 2


class TestEngineSorting:
    """Tests for reference sorting."""

    def test_sort_by_author(self) -> None:
        """Test sorting by author name."""
        engine = CSLEngine()
        ref_z = Reference(
            id="z",
            title="Z Article",
            authors=[Author(family="Zeta")],
        )
        ref_a = Reference(
            id="a",
            title="A Article",
            authors=[Author(family="Alpha")],
        )
        engine.add_reference(ref_z)
        engine.add_reference(ref_a)

        bib = engine.generate_bibliography()

        # Alpha should come before Zeta
        assert "Alpha" in bib.entries[0]


# Factory function tests


class TestCreateEngine:
    """Tests for create_engine factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        engine = create_engine()

        assert engine.style == "apa"

    def test_create_custom(self) -> None:
        """Test creating with custom style."""
        engine = create_engine(style="mla", locale="en-GB")

        assert engine.style == "mla"
        assert engine.locale == "en-GB"


class TestFormatReferences:
    """Tests for format_references function."""

    def test_format(self) -> None:
        """Test convenience formatting function."""
        refs = [make_reference()]

        result = format_references(refs)

        assert len(result) > 0
        assert "Smith" in result

    def test_format_multiple(self) -> None:
        """Test formatting multiple references."""
        refs = [
            make_reference("a", "Article A"),
            make_reference("b", "Article B"),
        ]

        result = format_references(refs)

        assert "Article A" in result
        assert "Article B" in result

    def test_format_with_style(self) -> None:
        """Test formatting with specific style."""
        refs = [make_reference()]

        result = format_references(refs, style="mla")

        assert len(result) > 0


# Constant tests


class TestConstants:
    """Tests for module constants."""

    def test_max_references(self) -> None:
        """Test MAX_REFERENCES is reasonable."""
        assert MAX_REFERENCES > 0
        assert MAX_REFERENCES == 500


class TestMaxReferencesLimit:
    """Tests for reference limit."""

    def test_max_references_enforced(self) -> None:
        """Test MAX_REFERENCES limit."""
        engine = CSLEngine()

        # Fill engine
        for i in range(MAX_REFERENCES):
            ref = Reference(id=f"ref{i}", title=f"Article {i}")
            engine.add_reference(ref)

        # Next should fail
        overflow = Reference(id="overflow", title="Overflow")
        result = engine.add_reference(overflow)

        assert result is False
        assert engine.reference_count == MAX_REFERENCES
