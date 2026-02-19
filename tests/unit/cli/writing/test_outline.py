"""
Tests for Outline Creation Module.

This module tests the outline creation functionality including:
- OutlineBuilder class methods
- OutlineSection and Outline dataclasses
- Theme organization
- Output formatting

Test Strategy
-------------
- Test each public method of OutlineBuilder
- Test dataclass operations
- Test recursive section handling
- Test output formatters
- Follow NASA JPL Rule #1: Simple test structure

Organization
------------
- TestOutlineSection: Section dataclass tests
- TestOutline: Outline dataclass tests
- TestOutlineBuilder: Builder class tests
- TestOutlineFormatter: Formatter tests
- TestOutlineCommand: CLI command tests
"""

from unittest.mock import Mock, patch
import json

from ingestforge.cli.writing.outline import (
    OutlineSection,
    Outline,
    OutlineBuilder,
    OutlineFormatter,
    OutlineCommand,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(text: str = "Test content", source: str = "test_source"):
    """Create mock chunk."""
    chunk = Mock()
    chunk.text = text
    chunk.content = text
    chunk.metadata = {"source": source}
    chunk.source_file = source
    return chunk


def make_mock_llm_client(response: str = '{"title": "Test", "sections": []}'):
    """Create mock LLM client."""
    client = Mock()
    client.generate.return_value = response
    return client


# ============================================================================
# TestOutlineSection: Section dataclass tests
# ============================================================================


class TestOutlineSection:
    """Tests for OutlineSection dataclass."""

    def test_section_creation(self):
        """Test basic section creation."""
        section = OutlineSection(title="Test Section")
        assert section.title == "Test Section"
        assert section.level == 1
        assert section.notes == []
        assert section.sources == []

    def test_section_with_notes(self):
        """Test section with notes."""
        section = OutlineSection(
            title="Section",
            notes=["Note 1", "Note 2"],
        )
        assert len(section.notes) == 2

    def test_section_with_sources(self):
        """Test section with sources."""
        section = OutlineSection(
            title="Section",
            sources=["source1", "source2"],
        )
        assert len(section.sources) == 2

    def test_add_subsection(self):
        """Test adding subsection."""
        parent = OutlineSection(title="Parent", level=1)
        child = OutlineSection(title="Child")

        parent.add_subsection(child)

        assert len(parent.subsections) == 1
        assert parent.subsections[0].level == 2

    def test_nested_subsections(self):
        """Test nested subsections."""
        root = OutlineSection(title="Root", level=1)
        child = OutlineSection(title="Child")
        grandchild = OutlineSection(title="Grandchild")

        # Add in order: grandchild to child first, then child to root
        # Note: add_subsection sets level relative to parent, so
        # we need to add in proper order
        root.add_subsection(child)  # child becomes level 2
        child.add_subsection(grandchild)  # grandchild becomes level 3

        assert root.level == 1
        assert root.subsections[0].level == 2
        assert root.subsections[0].subsections[0].level == 3

    def test_to_dict(self):
        """Test dictionary conversion."""
        section = OutlineSection(
            title="Test",
            level=2,
            notes=["Note"],
            sources=["src"],
        )
        data = section.to_dict()

        assert data["title"] == "Test"
        assert data["level"] == 2
        assert "notes" in data
        assert "sources" in data

    def test_to_dict_with_subsections(self):
        """Test dictionary conversion with subsections."""
        parent = OutlineSection(title="Parent")
        parent.add_subsection(OutlineSection(title="Child"))

        data = parent.to_dict()

        assert len(data["subsections"]) == 1
        assert data["subsections"][0]["title"] == "Child"

    def test_get_all_sections_flat(self):
        """Test flattening section hierarchy."""
        root = OutlineSection(title="Root")
        child1 = OutlineSection(title="Child 1")
        child2 = OutlineSection(title="Child 2")
        grandchild = OutlineSection(title="Grandchild")

        child1.add_subsection(grandchild)
        root.add_subsection(child1)
        root.add_subsection(child2)

        flat = root.get_all_sections_flat()

        assert len(flat) == 4  # Root + 2 children + 1 grandchild


# ============================================================================
# TestOutline: Outline dataclass tests
# ============================================================================


class TestOutline:
    """Tests for Outline dataclass."""

    def test_outline_creation(self):
        """Test basic outline creation."""
        outline = Outline(title="Test Outline")
        assert outline.title == "Test Outline"
        assert outline.depth == 3
        assert outline.format == "numbered"
        assert outline.sections == []

    def test_add_section(self):
        """Test adding section."""
        outline = Outline(title="Outline")
        section = OutlineSection(title="Section 1")

        outline.add_section(section)

        assert len(outline.sections) == 1
        assert outline.sections[0].level == 1

    def test_get_section_count_empty(self):
        """Test section count for empty outline."""
        outline = Outline(title="Empty")
        assert outline.get_section_count() == 0

    def test_get_section_count_with_sections(self):
        """Test section count with sections."""
        outline = Outline(title="Outline")
        outline.add_section(OutlineSection(title="S1"))
        outline.add_section(OutlineSection(title="S2"))

        assert outline.get_section_count() == 2

    def test_get_section_count_with_subsections(self):
        """Test section count including subsections."""
        outline = Outline(title="Outline")
        section = OutlineSection(title="Main")
        section.add_subsection(OutlineSection(title="Sub1"))
        section.add_subsection(OutlineSection(title="Sub2"))
        outline.add_section(section)

        assert outline.get_section_count() == 3

    def test_to_dict(self):
        """Test dictionary conversion."""
        outline = Outline(
            title="Test",
            depth=4,
            format="bulleted",
            sources_used=["s1", "s2"],
        )
        data = outline.to_dict()

        assert data["title"] == "Test"
        assert data["depth"] == 4
        assert data["format"] == "bulleted"
        assert "section_count" in data


# ============================================================================
# TestOutlineBuilder: Builder class tests
# ============================================================================


class TestOutlineBuilder:
    """Tests for OutlineBuilder class."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        client = make_mock_llm_client()
        builder = OutlineBuilder(client)
        assert builder.llm_client == client

    def test_build_basic_outline(self):
        """Test basic outline building."""
        response = json.dumps(
            {
                "title": "Test Outline",
                "sections": [
                    {"title": "Section 1", "notes": ["Note"], "subsections": []},
                ],
            }
        )
        client = make_mock_llm_client(response)
        builder = OutlineBuilder(client)

        outline = builder.build("Test Topic", [make_mock_chunk()], depth=3)

        assert outline.title == "Test Outline"
        assert len(outline.sections) == 1

    def test_build_with_nested_sections(self):
        """Test building with nested sections."""
        response = json.dumps(
            {
                "title": "Outline",
                "sections": [
                    {
                        "title": "Main",
                        "notes": [],
                        "subsections": [
                            {"title": "Sub", "notes": [], "subsections": []}
                        ],
                    }
                ],
            }
        )
        client = make_mock_llm_client(response)
        builder = OutlineBuilder(client)

        outline = builder.build("Topic", [make_mock_chunk()], depth=3)

        assert len(outline.sections[0].subsections) == 1

    def test_build_respects_depth_limit(self):
        """Test depth limit is respected."""
        response = json.dumps(
            {
                "title": "Outline",
                "sections": [
                    {
                        "title": "L1",
                        "subsections": [
                            {
                                "title": "L2",
                                "subsections": [
                                    {
                                        "title": "L3",
                                        "subsections": [
                                            {"title": "L4", "subsections": []}
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        client = make_mock_llm_client(response)
        builder = OutlineBuilder(client)

        outline = builder.build("Topic", [make_mock_chunk()], depth=2)

        # Should only go 2 levels deep
        assert len(outline.sections[0].subsections[0].subsections) == 0

    def test_build_with_empty_sources(self):
        """Test building with no sources."""
        client = make_mock_llm_client('{"title": "T", "sections": []}')
        builder = OutlineBuilder(client)

        outline = builder.build("Topic", [], depth=3)

        assert outline.title == "T"

    def test_format_sources(self):
        """Test source formatting."""
        client = make_mock_llm_client()
        builder = OutlineBuilder(client)
        chunks = [make_mock_chunk("Content 1"), make_mock_chunk("Content 2")]

        result = builder._format_sources(chunks)

        assert "Content 1" in result
        assert "Content 2" in result

    def test_format_sources_empty(self):
        """Test source formatting with empty list."""
        client = make_mock_llm_client()
        builder = OutlineBuilder(client)

        result = builder._format_sources([])

        assert "No source material" in result

    def test_suggest_sections(self):
        """Test section suggestion."""
        response = json.dumps(
            {
                "sections": [
                    {"title": "Introduction", "notes": ["Overview"]},
                    {"title": "Background", "notes": ["Context"]},
                ]
            }
        )
        client = make_mock_llm_client(response)
        builder = OutlineBuilder(client)

        sections = builder.suggest_sections("Topic", "Some context")

        assert len(sections) == 2

    def test_organize_by_theme(self):
        """Test theme organization."""
        response = json.dumps({"themes": ["Theme A", "Theme B"]})
        client = make_mock_llm_client(response)
        builder = OutlineBuilder(client)

        chunks = [
            make_mock_chunk("Theme A content"),
            make_mock_chunk("Theme B content"),
            make_mock_chunk("Other content"),
        ]

        result = builder.organize_by_theme(chunks)

        assert "Theme A" in result or "Other" in result

    def test_organize_by_theme_empty(self):
        """Test theme organization with empty chunks."""
        client = make_mock_llm_client()
        builder = OutlineBuilder(client)

        result = builder.organize_by_theme([])

        assert result == {}

    def test_parse_outline_response_valid_json(self):
        """Test parsing valid JSON response."""
        client = make_mock_llm_client()
        builder = OutlineBuilder(client)

        data = builder._parse_outline_response('{"title": "Test", "sections": []}')

        assert data["title"] == "Test"

    def test_parse_outline_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        client = make_mock_llm_client()
        builder = OutlineBuilder(client)

        data = builder._parse_outline_response("Not JSON at all")

        assert "sections" in data  # Returns default

    def test_parse_outline_response_json_in_text(self):
        """Test parsing JSON embedded in text."""
        client = make_mock_llm_client()
        builder = OutlineBuilder(client)

        data = builder._parse_outline_response('Here is {"title": "X", "sections": []}')

        assert data["title"] == "X"


# ============================================================================
# TestOutlineFormatter: Formatter tests
# ============================================================================


class TestOutlineFormatter:
    """Tests for OutlineFormatter class."""

    def test_format_numbered(self):
        """Test numbered format."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test")
        outline.add_section(OutlineSection(title="Section 1"))
        outline.add_section(OutlineSection(title="Section 2"))

        result = formatter.format(outline, "numbered")

        assert "1. Section 1" in result
        assert "2. Section 2" in result

    def test_format_numbered_with_subsections(self):
        """Test numbered format with subsections."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test")
        section = OutlineSection(title="Main")
        section.add_subsection(OutlineSection(title="Sub"))
        outline.add_section(section)

        result = formatter.format(outline, "numbered")

        assert "1. Main" in result
        assert "1.1. Sub" in result

    def test_format_bulleted(self):
        """Test bulleted format."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test")
        outline.add_section(OutlineSection(title="Section"))

        result = formatter.format(outline, "bulleted")

        assert "* Section" in result

    def test_format_bulleted_with_notes(self):
        """Test bulleted format with notes."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test")
        section = OutlineSection(title="Section", notes=["Note 1"])
        outline.add_section(section)

        result = formatter.format(outline, "bulleted")

        assert "- Note 1" in result

    def test_format_markdown(self):
        """Test markdown format."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test")
        outline.add_section(OutlineSection(title="Section"))

        result = formatter.format(outline, "markdown")

        assert "# Test" in result
        assert "## Section" in result

    def test_format_markdown_with_sources(self):
        """Test markdown format includes sources."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test", sources_used=["source1"])
        outline.add_section(OutlineSection(title="Section"))

        result = formatter.format(outline, "markdown")

        assert "## Sources" in result
        assert "source1" in result

    def test_format_unknown_defaults_numbered(self):
        """Test unknown format defaults to numbered."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test")
        outline.add_section(OutlineSection(title="Section"))

        result = formatter.format(outline, "unknown_format")

        assert "1. Section" in result

    def test_to_tree(self):
        """Test tree conversion."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test")
        outline.add_section(OutlineSection(title="Section"))

        tree = formatter.to_tree(outline)

        assert tree is not None


# ============================================================================
# TestOutlineCommand: CLI command tests
# ============================================================================


class TestOutlineCommand:
    """Tests for OutlineCommand CLI class."""

    @patch.object(OutlineCommand, "initialize_context")
    @patch.object(OutlineCommand, "get_llm_client")
    @patch.object(OutlineCommand, "search_context")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_success(self, mock_spinner, mock_search, mock_llm, mock_ctx):
        """Test successful execution."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client('{"title": "T", "sections": []}')
        mock_search.return_value = [make_mock_chunk()]
        mock_spinner.side_effect = lambda fn, *args: fn()

        cmd = OutlineCommand(console=Mock())
        result = cmd.execute("Topic")

        assert result == 0

    @patch.object(OutlineCommand, "initialize_context")
    @patch.object(OutlineCommand, "get_llm_client")
    def test_execute_no_llm(self, mock_llm, mock_ctx):
        """Test execution without LLM client."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = None

        cmd = OutlineCommand(console=Mock())
        result = cmd.execute("Topic")

        assert result == 1

    @patch.object(OutlineCommand, "initialize_context")
    @patch.object(OutlineCommand, "get_llm_client")
    @patch.object(OutlineCommand, "search_context")
    def test_execute_no_context(self, mock_search, mock_llm, mock_ctx):
        """Test execution with no context."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client()
        mock_search.return_value = []

        cmd = OutlineCommand(console=Mock())
        result = cmd.execute("Topic")

        assert result == 0

    @patch.object(OutlineCommand, "initialize_context")
    @patch.object(OutlineCommand, "get_llm_client")
    @patch.object(OutlineCommand, "search_context")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_with_output(
        self, mock_spinner, mock_search, mock_llm, mock_ctx, tmp_path
    ):
        """Test execution with output file."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client('{"title": "T", "sections": []}')
        mock_search.return_value = [make_mock_chunk()]
        mock_spinner.side_effect = lambda fn, *args: fn()

        output = tmp_path / "outline.md"
        cmd = OutlineCommand(console=Mock())
        result = cmd.execute("Topic", output=output)

        assert result == 0
        assert output.exists()

    @patch.object(OutlineCommand, "initialize_context")
    @patch.object(OutlineCommand, "get_llm_client")
    @patch.object(OutlineCommand, "search_context")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_all_formats(self, mock_spinner, mock_search, mock_llm, mock_ctx):
        """Test execution with all formats."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client('{"title": "T", "sections": []}')
        mock_search.return_value = [make_mock_chunk()]
        mock_spinner.side_effect = lambda fn, *args: fn()

        cmd = OutlineCommand(console=Mock())

        for fmt in ["numbered", "bulleted", "markdown"]:
            result = cmd.execute("Topic", format_type=fmt)
            assert result == 0

    @patch.object(OutlineCommand, "initialize_context")
    def test_execute_handles_exception(self, mock_ctx):
        """Test exception handling."""
        mock_ctx.side_effect = Exception("Test error")

        cmd = OutlineCommand(console=Mock())
        result = cmd.execute("Topic")

        assert result == 1


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestOutlineEdgeCases:
    """Edge case tests for outline functionality."""

    def test_deeply_nested_sections(self):
        """Test handling deeply nested sections."""
        root = OutlineSection(title="L1")
        current = root
        for i in range(10):
            child = OutlineSection(title=f"L{i+2}")
            current.add_subsection(child)
            current = child

        flat = root.get_all_sections_flat()
        assert len(flat) == 11

    def test_section_with_empty_title(self):
        """Test section with empty title."""
        section = OutlineSection(title="")
        assert section.title == ""

    def test_outline_with_many_sections(self):
        """Test outline with many sections."""
        outline = Outline(title="Large")
        for i in range(100):
            outline.add_section(OutlineSection(title=f"Section {i}"))

        assert outline.get_section_count() == 100

    def test_formatter_with_special_characters(self):
        """Test formatter with special characters."""
        formatter = OutlineFormatter()
        outline = Outline(title="Test & <Special>")
        outline.add_section(OutlineSection(title="Section with 'quotes'"))

        result = formatter.format(outline, "markdown")

        assert "Test & <Special>" in result
        assert "'quotes'" in result
