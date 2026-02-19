"""Tests for graph_viz module (CITE-001.3).

Tests the Mermaid graph visualization:
- Diagram generation
- Node styling
- Edge generation
- Markdown export
"""


from ingestforge.core.citation.link_extractor import LinkMap, InternalReference
from ingestforge.core.citation.analytics import GraphAnalytics
from ingestforge.cli.citation.graph_viz import (
    GraphDirection,
    MermaidStyle,
    MermaidExporter,
    export_to_mermaid,
    generate_citation_graph_markdown,
    MAX_NODES_DEFAULT,
)


def _create_test_link_map() -> LinkMap:
    """Create a test link map: A -> B -> C, A -> C"""
    link_map = LinkMap()
    refs = [
        InternalReference(source_chunk_id="A", target_chunk_id="B", resolved=True),
        InternalReference(source_chunk_id="B", target_chunk_id="C", resolved=True),
        InternalReference(source_chunk_id="A", target_chunk_id="C", resolved=True),
    ]
    for ref in refs:
        link_map.add_reference(ref)
    return link_map


class TestGraphDirection:
    """Test GraphDirection enum."""

    def test_all_directions_defined(self) -> None:
        """All directions should be defined."""
        assert GraphDirection.TOP_DOWN
        assert GraphDirection.BOTTOM_UP
        assert GraphDirection.LEFT_RIGHT
        assert GraphDirection.RIGHT_LEFT


class TestMermaidStyle:
    """Test MermaidStyle dataclass."""

    def test_default_style(self) -> None:
        """Default style should have sensible defaults."""
        style = MermaidStyle()
        assert style.direction == GraphDirection.TOP_DOWN
        assert style.show_scores is False
        assert style.highlight_hubs is True
        assert style.max_nodes == MAX_NODES_DEFAULT

    def test_custom_style(self) -> None:
        """Should accept custom values."""
        style = MermaidStyle(
            direction=GraphDirection.LEFT_RIGHT,
            show_scores=True,
            max_nodes=20,
        )
        assert style.direction == GraphDirection.LEFT_RIGHT
        assert style.show_scores is True
        assert style.max_nodes == 20


class TestMermaidExporterInit:
    """Test MermaidExporter initialization."""

    def test_init_with_link_map(self) -> None:
        """Should initialize with link map."""
        link_map = _create_test_link_map()
        exporter = MermaidExporter(link_map)
        assert exporter is not None

    def test_init_with_analytics(self) -> None:
        """Should initialize with analytics."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)
        exporter = MermaidExporter(link_map, analytics)
        assert exporter is not None


class TestDiagramGeneration:
    """Test diagram generation."""

    def test_generate_basic_diagram(self) -> None:
        """Should generate basic diagram."""
        link_map = _create_test_link_map()
        exporter = MermaidExporter(link_map)
        diagram = exporter.generate_diagram()
        assert "flowchart" in diagram
        assert "TD" in diagram

    def test_generate_with_title(self) -> None:
        """Should include title when provided."""
        link_map = _create_test_link_map()
        exporter = MermaidExporter(link_map)
        diagram = exporter.generate_diagram(title="Test Graph")
        assert "Test Graph" in diagram

    def test_contains_edges(self) -> None:
        """Should contain edges."""
        link_map = _create_test_link_map()
        exporter = MermaidExporter(link_map)
        diagram = exporter.generate_diagram()
        assert "-->" in diagram


class TestNodeStyling:
    """Test node styling."""

    def test_style_classes_included(self) -> None:
        """Should include style class definitions."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)
        analytics.calculate_all_metrics()
        style = MermaidStyle(highlight_hubs=True)
        exporter = MermaidExporter(link_map, analytics, style)
        diagram = exporter.generate_diagram()
        assert "classDef" in diagram

    def test_legend_included(self) -> None:
        """Should include legend when requested."""
        link_map = _create_test_link_map()
        style = MermaidStyle(include_legend=True)
        exporter = MermaidExporter(link_map, style=style)
        diagram = exporter.generate_diagram()
        assert "Legend" in diagram


class TestConvenienceFunction:
    """Test export_to_mermaid convenience function."""

    def test_export_convenience(self) -> None:
        """Should work as standalone function."""
        link_map = _create_test_link_map()
        diagram = export_to_mermaid(link_map)
        assert "flowchart" in diagram


class TestMarkdownExport:
    """Test full Markdown document generation."""

    def test_generate_markdown(self) -> None:
        """Should generate complete Markdown."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)
        analytics.calculate_all_metrics()
        markdown = generate_citation_graph_markdown(link_map, analytics)
        assert "# " in markdown
        assert "```mermaid" in markdown

    def test_markdown_without_analytics(self) -> None:
        """Should work without analytics."""
        link_map = _create_test_link_map()
        markdown = generate_citation_graph_markdown(link_map)
        assert "```mermaid" in markdown


class TestEmptyGraph:
    """Test handling of empty graph."""

    def test_empty_link_map(self) -> None:
        """Should handle empty link map."""
        link_map = LinkMap()
        exporter = MermaidExporter(link_map)
        diagram = exporter.generate_diagram()
        assert "flowchart" in diagram
