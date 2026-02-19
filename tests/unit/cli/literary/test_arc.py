"""
Tests for Story Arc CLI Command and StoryArcAnalyzer.

Tests story structure analysis, plot point identification, and visualization.

Test Strategy
-------------
- Test StoryArcAnalyzer class methods
- Test ArcCommand execution
- Test structure templates
- Mock LLM and storage for unit testing
- Keep tests simple (NASA JPL Rule #1)
"""

import json
from unittest.mock import Mock, patch

import pytest
import typer

from ingestforge.cli.literary.arc import (
    ArcCommand,
    StoryArcAnalyzer,
    command,
    visualize_command,
    analyze_command,
    STRUCTURE_TEMPLATES,
    THREE_ACT_TEMPLATE,
    HERO_JOURNEY_TEMPLATE,
    FIVE_ACT_TEMPLATE,
    FREYTAG_TEMPLATE,
)
from ingestforge.cli.literary.models import (
    PlotPoint,
    StoryArc,
    STRUCTURE_TYPES,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "The hero faced danger and fought bravely.",
    **metadata,
):
    """Create a mock chunk object."""
    chunk = Mock()
    chunk.chunk_id = chunk_id
    chunk.content = content
    chunk.text = content
    chunk.metadata = metadata
    return chunk


def make_mock_context(has_storage: bool = True):
    """Create a mock context dictionary."""
    ctx = {"config": Mock()}
    if has_storage:
        ctx["storage"] = Mock()
        ctx["storage"].search = Mock(return_value=[make_mock_chunk()])
    return ctx


def make_mock_llm_client():
    """Create a mock LLM client."""
    llm = Mock()
    llm.generate = Mock(return_value="The story follows a classic three-act structure.")
    return llm


# ============================================================================
# Structure Template Tests
# ============================================================================


class TestStructureTemplates:
    """Tests for structure templates."""

    def test_three_act_template_defined(self):
        """Test three-act template is defined."""
        assert len(THREE_ACT_TEMPLATE) > 0

    def test_three_act_has_key_points(self):
        """Test three-act has key plot points."""
        names = [p["name"] for p in THREE_ACT_TEMPLATE]
        assert "Exposition" in names
        assert "Climax" in names
        assert "Resolution" in names

    def test_hero_journey_template_defined(self):
        """Test hero's journey template is defined."""
        assert len(HERO_JOURNEY_TEMPLATE) > 0

    def test_hero_journey_has_key_points(self):
        """Test hero's journey has key plot points."""
        names = [p["name"] for p in HERO_JOURNEY_TEMPLATE]
        assert "Call to Adventure" in names
        assert "Ordeal" in names

    def test_five_act_template_defined(self):
        """Test five-act template is defined."""
        assert len(FIVE_ACT_TEMPLATE) > 0

    def test_freytag_template_defined(self):
        """Test Freytag's pyramid template is defined."""
        assert len(FREYTAG_TEMPLATE) > 0

    def test_all_templates_in_structure_templates(self):
        """Test all templates are in STRUCTURE_TEMPLATES dict."""
        assert "three-act" in STRUCTURE_TEMPLATES
        assert "hero-journey" in STRUCTURE_TEMPLATES
        assert "five-act" in STRUCTURE_TEMPLATES
        assert "freytag" in STRUCTURE_TEMPLATES

    def test_templates_have_valid_positions(self):
        """Test all template points have valid positions (0-1)."""
        for name, template in STRUCTURE_TEMPLATES.items():
            for point in template:
                assert 0.0 <= point["position"] <= 1.0, f"Invalid position in {name}"


# ============================================================================
# StoryArcAnalyzer Tests
# ============================================================================


class TestStoryArcAnalyzerInit:
    """Tests for StoryArcAnalyzer initialization."""

    def test_create_analyzer(self):
        """Test creating StoryArcAnalyzer instance."""
        analyzer = StoryArcAnalyzer()
        assert analyzer is not None
        assert analyzer.llm_client is None

    def test_create_analyzer_with_llm(self):
        """Test StoryArcAnalyzer with LLM client."""
        llm = make_mock_llm_client()
        analyzer = StoryArcAnalyzer(llm_client=llm)
        assert analyzer.llm_client is not None

    def test_tension_keywords_defined(self):
        """Test tension keywords are initialized."""
        analyzer = StoryArcAnalyzer()
        assert "high" in analyzer._tension_keywords
        assert "rising" in analyzer._tension_keywords
        assert "low" in analyzer._tension_keywords


class TestStructureAnalysis:
    """Tests for structure analysis."""

    def test_analyze_structure_default(self):
        """Test analyzing with default structure."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk() for _ in range(10)]

        result = analyzer.analyze_structure(chunks)

        assert isinstance(result, StoryArc)
        assert result.structure_type == "three-act"

    def test_analyze_structure_hero_journey(self):
        """Test analyzing with hero's journey structure."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk() for _ in range(12)]

        result = analyzer.analyze_structure(chunks, structure="hero-journey")

        assert result.structure_type == "hero-journey"

    def test_analyze_structure_invalid_fallback(self):
        """Test invalid structure falls back to three-act."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk()]

        result = analyzer.analyze_structure(chunks, structure="invalid")

        assert result.structure_type == "three-act"

    def test_analyze_structure_returns_plot_points(self):
        """Test analysis returns plot points."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk() for _ in range(10)]

        result = analyzer.analyze_structure(chunks)

        assert len(result.plot_points) > 0
        assert all(isinstance(p, PlotPoint) for p in result.plot_points)

    def test_analyze_structure_returns_tension_curve(self):
        """Test analysis returns tension curve."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk() for _ in range(10)]

        result = analyzer.analyze_structure(chunks)

        assert len(result.tension_curve) == len(chunks)


class TestPlotPointIdentification:
    """Tests for plot point identification."""

    def test_identify_plot_points(self):
        """Test identifying plot points from template."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk(chunk_id=f"c{i}") for i in range(10)]

        result = analyzer.identify_plot_points(chunks, THREE_ACT_TEMPLATE)

        assert len(result) == len(THREE_ACT_TEMPLATE)

    def test_plot_points_have_descriptions(self):
        """Test plot points have descriptions."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk(content="Story content here") for _ in range(5)]

        result = analyzer.identify_plot_points(chunks, THREE_ACT_TEMPLATE[:3])

        for point in result:
            assert point.description != ""


class TestTensionCurve:
    """Tests for tension curve calculation."""

    def test_calculate_tension_curve(self):
        """Test tension curve calculation."""
        analyzer = StoryArcAnalyzer()
        chunks = [
            make_mock_chunk(content="The danger was immense."),
            make_mock_chunk(content="Peace at last."),
        ]

        result = analyzer.calculate_tension_curve(chunks)

        assert len(result) == 2
        assert all(0.0 <= t <= 1.0 for t in result)

    def test_tension_high_keywords(self):
        """Test high tension from danger keywords."""
        analyzer = StoryArcAnalyzer()
        content = "The battle raged. Death and danger everywhere. Kill or be killed."

        result = analyzer._calculate_chunk_tension(content.lower())

        assert result > 0.5  # Should be high tension

    def test_tension_low_keywords(self):
        """Test low tension from peaceful keywords."""
        analyzer = StoryArcAnalyzer()
        content = "Peace reigned. Everyone was safe and happy at home."

        result = analyzer._calculate_chunk_tension(content.lower())

        assert result < 0.5  # Should be lower tension

    def test_smooth_curve(self):
        """Test curve smoothing."""
        analyzer = StoryArcAnalyzer()
        values = [0.0, 1.0, 0.0, 1.0, 0.0]

        result = analyzer._smooth_curve(values, window=3)

        # Smoothed values should be less extreme
        assert min(result) > 0.0
        assert max(result) < 1.0

    def test_smooth_curve_short(self):
        """Test smoothing with short input."""
        analyzer = StoryArcAnalyzer()
        values = [0.5, 0.6]

        result = analyzer._smooth_curve(values, window=5)

        assert len(result) == 2


class TestActBuilding:
    """Tests for act/section building."""

    def test_build_acts(self):
        """Test building act divisions."""
        analyzer = StoryArcAnalyzer()
        chunks = [make_mock_chunk() for _ in range(9)]

        result = analyzer._build_acts(THREE_ACT_TEMPLATE, chunks)

        assert len(result) == 3  # Three acts
        assert result[0]["number"] == 1


class TestSummaryGeneration:
    """Tests for summary generation."""

    def test_generate_summary_with_llm(self):
        """Test summary generation with LLM."""
        llm = make_mock_llm_client()
        analyzer = StoryArcAnalyzer(llm_client=llm)

        arc = StoryArc(structure_type="three-act")
        chunks = [make_mock_chunk() for _ in range(3)]

        result = analyzer._generate_summary(arc, chunks)

        assert result != ""
        llm.generate.assert_called_once()

    def test_generate_summary_llm_failure(self):
        """Test summary generation when LLM fails."""
        llm = Mock()
        llm.generate = Mock(side_effect=Exception("API error"))
        analyzer = StoryArcAnalyzer(llm_client=llm)

        arc = StoryArc(structure_type="three-act")
        chunks = [make_mock_chunk()]

        result = analyzer._generate_summary(arc, chunks)

        assert "three-act" in result


class TestVisualization:
    """Tests for arc visualization."""

    def test_export_mermaid(self):
        """Test exporting as Mermaid diagram."""
        analyzer = StoryArcAnalyzer()
        arc = StoryArc(
            structure_type="three-act",
            plot_points=[
                PlotPoint(
                    name="Start", type="exposition", position=0.0, description=""
                ),
                PlotPoint(name="End", type="resolution", position=1.0, description=""),
            ],
        )

        result = analyzer.export_visualization(arc, format="mermaid")

        assert "graph LR" in result

    def test_export_ascii(self):
        """Test exporting as ASCII art."""
        analyzer = StoryArcAnalyzer()
        arc = StoryArc(
            structure_type="three-act",
            tension_curve=[0.2, 0.5, 0.8, 0.3],
        )

        result = analyzer.export_visualization(arc, format="ascii")

        assert "Tension Curve" in result


# ============================================================================
# ArcCommand Tests
# ============================================================================


class TestArcCommandInit:
    """Tests for ArcCommand initialization."""

    def test_create_command(self):
        """Test creating ArcCommand instance."""
        cmd = ArcCommand()
        assert cmd is not None

    def test_inherits_from_literary_command(self):
        """Test ArcCommand inherits from LiteraryCommand."""
        from ingestforge.cli.literary.base import LiteraryCommand

        cmd = ArcCommand()
        assert isinstance(cmd, LiteraryCommand)


class TestArcCommandValidation:
    """Tests for input validation."""

    def test_validate_structure_valid(self):
        """Test validation passes for valid structure."""
        cmd = ArcCommand()

        for structure in STRUCTURE_TYPES:
            cmd._validate_structure(structure)  # Should not raise

    def test_validate_structure_invalid(self):
        """Test validation fails for invalid structure."""
        cmd = ArcCommand()

        with pytest.raises(typer.BadParameter):
            cmd._validate_structure("invalid-structure")


class TestArcCommandExecution:
    """Tests for ArcCommand.execute()."""

    def test_execute_no_llm(self):
        """Test execution fails without LLM."""
        cmd = ArcCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client", return_value=None):
                mock_init.return_value = make_mock_context()

                result = cmd.execute("The Odyssey")

                assert result == 1

    def test_execute_no_context(self):
        """Test execution with no search results."""
        cmd = ArcCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_structure", return_value=[]):
                    mock_init.return_value = make_mock_context()
                    mock_llm.return_value = make_mock_llm_client()

                    result = cmd.execute("Unknown Work")

                    assert result == 0  # Returns 0 with warning

    def test_execute_success(self):
        """Test successful execution."""
        cmd = ArcCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_structure") as mock_search:
                    with patch.object(cmd, "_display_arc"):
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = [make_mock_chunk() for _ in range(5)]

                        result = cmd.execute("The Odyssey")

                        assert result == 0


class TestArcCommandOutput:
    """Tests for ArcCommand output methods."""

    def test_save_arc_json(self, tmp_path):
        """Test saving arc as JSON."""
        cmd = ArcCommand()
        output = tmp_path / "arc.json"

        arc = StoryArc(
            structure_type="three-act",
            summary="A classic story",
        )

        cmd._save_arc(output, "Test Work", arc)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["work"] == "Test Work"

    def test_save_arc_markdown(self, tmp_path):
        """Test saving arc as Markdown."""
        cmd = ArcCommand()
        output = tmp_path / "arc.md"

        arc = StoryArc(
            structure_type="three-act",
            plot_points=[
                PlotPoint(
                    name="Start",
                    type="exposition",
                    position=0.0,
                    description="Beginning",
                ),
            ],
            tension_curve=[0.5],
            summary="A classic story",
        )

        cmd._save_arc(output, "Test Work", arc)

        assert output.exists()
        content = output.read_text()
        assert "# Story Arc" in content


# ============================================================================
# Typer Command Tests
# ============================================================================


class TestTyperCommands:
    """Tests for typer command wrappers."""

    def test_command_exists(self):
        """Test main command exists."""
        assert callable(command)

    def test_visualize_command_exists(self):
        """Test visualize command exists."""
        assert callable(visualize_command)

    def test_analyze_command_exists(self):
        """Test analyze command exists."""
        assert callable(analyze_command)
