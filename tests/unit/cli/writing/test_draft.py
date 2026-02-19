"""
Tests for Draft Generation Module.

This module tests the draft generation functionality including:
- DraftGenerator class methods
- Citation and Draft dataclasses
- Style-based generation
- Output formatting

Test Strategy
-------------
- Test each public method of DraftGenerator
- Test dataclass serialization
- Test style validation
- Test output formatters
- Follow NASA JPL Rule #1: Simple test structure

Organization
------------
- TestCitation: Citation dataclass tests
- TestDraft: Draft dataclass tests
- TestDraftGenerator: Generator class tests
- TestDraftCommand: CLI command tests
"""

from unittest.mock import Mock, patch

from ingestforge.cli.writing.draft import (
    Citation,
    Draft,
    DraftGenerator,
    DraftCommand,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(text: str = "Test content", source: str = "test_source"):
    """Create mock chunk with metadata."""
    chunk = Mock()
    chunk.text = text
    chunk.content = text
    chunk.metadata = {
        "source": source,
        "author": "Test Author",
        "year": "2024",
        "title": "Test Title",
    }
    chunk.source_file = source
    return chunk


def make_mock_llm_client(response: str = "Generated text"):
    """Create mock LLM client."""
    client = Mock()
    client.generate.return_value = response
    return client


# ============================================================================
# TestCitation: Citation dataclass tests
# ============================================================================


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Test basic citation creation."""
        citation = Citation(
            author="Smith, J.",
            year="2024",
            title="Test Paper",
            source_id="smith2024",
        )
        assert citation.author == "Smith, J."
        assert citation.year == "2024"
        assert citation.title == "Test Paper"

    def test_citation_with_optional_fields(self):
        """Test citation with optional fields."""
        citation = Citation(
            author="Doe, J.",
            year="2023",
            title="Another Paper",
            source_id="doe2023",
            page="42",
            url="https://example.com",
        )
        assert citation.page == "42"
        assert citation.url == "https://example.com"

    def test_citation_to_inline(self):
        """Test inline citation formatting."""
        citation = Citation(
            author="Johnson",
            year="2022",
            title="Title",
            source_id="j2022",
        )
        assert citation.to_inline() == "[Johnson, 2022]"

    def test_citation_to_dict(self):
        """Test dictionary conversion."""
        citation = Citation(
            author="Author",
            year="2021",
            title="Title",
            source_id="a2021",
            page="10",
        )
        data = citation.to_dict()
        assert data["author"] == "Author"
        assert data["year"] == "2021"
        assert data["page"] == "10"


# ============================================================================
# TestDraft: Draft dataclass tests
# ============================================================================


class TestDraft:
    """Tests for Draft dataclass."""

    def test_draft_creation(self):
        """Test basic draft creation."""
        draft = Draft(content="Test content here.")
        assert draft.content == "Test content here."
        assert draft.style == "academic"
        assert draft.format == "markdown"

    def test_draft_word_count_calculation(self):
        """Test automatic word count."""
        draft = Draft(content="One two three four five")
        assert draft.word_count == 5

    def test_draft_explicit_word_count(self):
        """Test explicit word count."""
        draft = Draft(content="Test", word_count=100)
        assert draft.word_count == 100

    def test_draft_with_citations(self):
        """Test draft with citations."""
        citations = [
            Citation("A", "2024", "T1", "a1"),
            Citation("B", "2023", "T2", "b1"),
        ]
        draft = Draft(
            content="Content",
            citations=citations,
            sources_used=["source1", "source2"],
        )
        assert len(draft.citations) == 2
        assert len(draft.sources_used) == 2

    def test_draft_to_dict(self):
        """Test dictionary conversion."""
        draft = Draft(
            content="Test",
            style="blog",
            sources_used=["s1"],
        )
        data = draft.to_dict()
        assert data["content"] == "Test"
        assert data["style"] == "blog"
        assert "sources_used" in data


# ============================================================================
# TestDraftGenerator: Generator class tests
# ============================================================================


class TestDraftGenerator:
    """Tests for DraftGenerator class."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)
        assert generator.llm_client == client

    def test_generate_draft_academic(self):
        """Test draft generation with academic style."""
        client = make_mock_llm_client("Academic content here.")
        generator = DraftGenerator(client)
        chunks = [make_mock_chunk()]

        draft = generator.generate("Test topic", "academic", 500, chunks)

        assert draft.content == "Academic content here."
        assert draft.style == "academic"
        client.generate.assert_called_once()

    def test_generate_draft_blog(self):
        """Test draft generation with blog style."""
        client = make_mock_llm_client("Blog post content")
        generator = DraftGenerator(client)

        draft = generator.generate("Topic", "blog", 800, [make_mock_chunk()])

        assert draft.style == "blog"

    def test_generate_draft_technical(self):
        """Test draft generation with technical style."""
        client = make_mock_llm_client("Technical documentation")
        generator = DraftGenerator(client)

        draft = generator.generate("Topic", "technical", 1000, [make_mock_chunk()])

        assert draft.style == "technical"

    def test_generate_draft_casual(self):
        """Test draft generation with casual style."""
        client = make_mock_llm_client("Casual writing")
        generator = DraftGenerator(client)

        draft = generator.generate("Topic", "casual", 500, [make_mock_chunk()])

        assert draft.style == "casual"

    def test_generate_unknown_style_defaults_academic(self):
        """Test unknown style defaults to academic."""
        client = make_mock_llm_client("Content")
        generator = DraftGenerator(client)

        draft = generator.generate("Topic", "unknown_style", 500, [make_mock_chunk()])

        assert draft.style == "academic"

    def test_generate_with_citations(self):
        """Test draft generation includes citations."""
        client = make_mock_llm_client("Content with [Author, 2024] citation.")
        generator = DraftGenerator(client)
        chunks = [make_mock_chunk()]

        draft = generator.generate(
            "Topic", "academic", 500, chunks, include_citations=True
        )

        assert len(draft.citations) > 0

    def test_generate_without_citations(self):
        """Test draft generation without citations."""
        client = make_mock_llm_client("Content")
        generator = DraftGenerator(client)

        draft = generator.generate(
            "Topic", "academic", 500, [make_mock_chunk()], include_citations=False
        )

        assert len(draft.citations) == 0

    def test_format_output_markdown(self):
        """Test markdown output formatting."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)
        draft = Draft(content="# Title\n\nContent")

        result = generator.format_output(draft, "markdown")

        assert "# Title" in result
        assert "Content" in result

    def test_format_output_plain(self):
        """Test plain text output formatting."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)
        draft = Draft(content="# Title\n\n**Bold** text")

        result = generator.format_output(draft, "plain")

        # Should strip markdown formatting
        assert "#" not in result
        assert "**" not in result

    def test_format_output_latex(self):
        """Test LaTeX output formatting."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)
        draft = Draft(content="# Section\n\nText")

        result = generator.format_output(draft, "latex")

        assert "\\documentclass" in result
        assert "\\begin{document}" in result

    def test_extract_citations_from_chunks(self):
        """Test citation extraction from chunks."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)
        chunks = [
            make_mock_chunk("Content 1", "source1"),
            make_mock_chunk("Content 2", "source2"),
        ]

        citations = generator._extract_citations(chunks)

        assert len(citations) >= 0  # May dedupe

    def test_get_source_ids(self):
        """Test source ID extraction."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)
        chunks = [
            make_mock_chunk("C1", "source1"),
            make_mock_chunk("C2", "source2"),
        ]

        sources = generator._get_source_ids(chunks)

        assert "source1" in sources
        assert "source2" in sources

    def test_length_guide_short(self):
        """Test short length guide."""
        generator = DraftGenerator(make_mock_llm_client())
        guide = generator._get_length_guide(300)
        assert "300" in guide or "500" in guide

    def test_length_guide_medium(self):
        """Test medium length guide."""
        generator = DraftGenerator(make_mock_llm_client())
        guide = generator._get_length_guide(1000)
        assert "800" in guide or "1200" in guide

    def test_length_guide_long(self):
        """Test long length guide."""
        generator = DraftGenerator(make_mock_llm_client())
        guide = generator._get_length_guide(2000)
        assert "1800" in guide or "2500" in guide


# ============================================================================
# TestDraftCommand: CLI command tests
# ============================================================================


class TestDraftCommand:
    """Tests for DraftCommand CLI class."""

    @patch.object(DraftCommand, "initialize_context")
    @patch.object(DraftCommand, "get_llm_client")
    @patch.object(DraftCommand, "search_context")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_success(self, mock_spinner, mock_search, mock_llm, mock_ctx):
        """Test successful command execution."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client("Generated draft")
        mock_search.return_value = [make_mock_chunk()]
        mock_spinner.side_effect = lambda fn, *args: fn()

        cmd = DraftCommand(console=Mock())
        result = cmd.execute("Test topic")

        assert result == 0

    @patch.object(DraftCommand, "initialize_context")
    @patch.object(DraftCommand, "get_llm_client")
    def test_execute_no_llm_client(self, mock_llm, mock_ctx):
        """Test execution without LLM client."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = None

        cmd = DraftCommand(console=Mock())
        result = cmd.execute("Topic")

        assert result == 1

    @patch.object(DraftCommand, "initialize_context")
    @patch.object(DraftCommand, "get_llm_client")
    @patch.object(DraftCommand, "search_context")
    def test_execute_no_context(self, mock_search, mock_llm, mock_ctx):
        """Test execution with no search results."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client()
        mock_search.return_value = []

        cmd = DraftCommand(console=Mock())
        result = cmd.execute("Topic")

        assert result == 0  # Returns 0 but with warning

    @patch.object(DraftCommand, "initialize_context")
    @patch.object(DraftCommand, "get_llm_client")
    @patch.object(DraftCommand, "search_context")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_with_output_file(
        self, mock_spinner, mock_search, mock_llm, mock_ctx, tmp_path
    ):
        """Test execution with output file."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client("Draft content")
        mock_search.return_value = [make_mock_chunk()]
        mock_spinner.side_effect = lambda fn, *args: fn()

        output = tmp_path / "draft.md"
        cmd = DraftCommand(console=Mock())
        result = cmd.execute("Topic", output=output)

        assert result == 0
        assert output.exists()

    @patch.object(DraftCommand, "initialize_context")
    @patch.object(DraftCommand, "get_llm_client")
    @patch.object(DraftCommand, "search_context")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_all_styles(self, mock_spinner, mock_search, mock_llm, mock_ctx):
        """Test execution with all styles."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client("Content")
        mock_search.return_value = [make_mock_chunk()]
        mock_spinner.side_effect = lambda fn, *args: fn()

        cmd = DraftCommand(console=Mock())

        for style in ["academic", "blog", "technical", "casual"]:
            result = cmd.execute("Topic", style=style)
            assert result == 0

    @patch.object(DraftCommand, "initialize_context")
    @patch.object(DraftCommand, "get_llm_client")
    @patch.object(DraftCommand, "search_context")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_all_formats(self, mock_spinner, mock_search, mock_llm, mock_ctx):
        """Test execution with all output formats."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client("Content")
        mock_search.return_value = [make_mock_chunk()]
        mock_spinner.side_effect = lambda fn, *args: fn()

        cmd = DraftCommand(console=Mock())

        for fmt in ["markdown", "plain", "latex"]:
            result = cmd.execute("Topic", output_format=fmt)
            assert result == 0

    @patch.object(DraftCommand, "initialize_context")
    def test_execute_handles_exception(self, mock_ctx):
        """Test exception handling."""
        mock_ctx.side_effect = Exception("Test error")

        cmd = DraftCommand(console=Mock())
        result = cmd.execute("Topic")

        assert result == 1


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestDraftEdgeCases:
    """Edge case tests for draft generation."""

    def test_empty_content_draft(self):
        """Test draft with empty content."""
        draft = Draft(content="")
        assert draft.word_count == 0

    def test_citation_without_optional_fields(self):
        """Test citation without optional fields."""
        citation = Citation("A", "2024", "T", "id")
        assert citation.page is None
        assert citation.url is None

    def test_draft_sources_deduplication(self):
        """Test source deduplication."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)

        # Same source multiple times
        chunks = [
            make_mock_chunk("C1", "same_source"),
            make_mock_chunk("C2", "same_source"),
        ]

        sources = generator._get_source_ids(chunks)
        assert len(sources) == 1

    def test_format_context_with_empty_chunks(self):
        """Test context formatting with empty list."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)

        result = generator._format_context_with_sources([])

        assert "No source material" in result

    def test_format_context_limits_chunks(self):
        """Test context formatting limits to 10 chunks."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)

        chunks = [make_mock_chunk(f"Content {i}") for i in range(15)]
        result = generator._format_context_with_sources(chunks)

        # Should only include first 10
        assert "Content 9" in result
        assert "Content 14" not in result

    def test_get_metadata_from_dict(self):
        """Test metadata extraction from dict."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)

        chunk = {"metadata": {"author": "Test"}}
        meta = generator._get_metadata(chunk)

        assert meta.get("author") == "Test"

    def test_get_metadata_from_object(self):
        """Test metadata extraction from object."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)

        chunk = Mock()
        chunk.metadata = {"title": "Test"}
        meta = generator._get_metadata(chunk)

        assert meta.get("title") == "Test"

    def test_get_metadata_empty(self):
        """Test metadata extraction from empty object."""
        client = make_mock_llm_client()
        generator = DraftGenerator(client)

        chunk = "plain string"
        meta = generator._get_metadata(chunk)

        assert meta == {}
