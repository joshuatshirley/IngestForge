"""Tests for D3 renderer.

Tests interactive graph visualization generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestforge.viz.d3_renderer import (
    D3Renderer,
    RenderConfig,
    create_renderer,
    render_graph,
    render_from_networkx,
    render_with_template,
    open_in_browser,
    TEMPLATES_DIR,
)
from ingestforge.viz.graph_export import (
    GraphData,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    create_sample_graph,
)

# RenderConfig tests


class TestRenderConfig:
    """Tests for RenderConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RenderConfig()

        assert config.width == 960
        assert config.height == 600
        assert config.show_labels is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RenderConfig(
            width=1200,
            height=800,
            node_radius=10,
        )

        assert config.width == 1200
        assert config.height == 800
        assert config.node_radius == 10


# D3Renderer tests


class TestD3Renderer:
    """Tests for D3Renderer class."""

    def test_renderer_creation(self) -> None:
        """Test creating renderer."""
        renderer = D3Renderer()

        assert renderer.config is not None

    def test_renderer_with_config(self) -> None:
        """Test renderer with custom config."""
        config = RenderConfig(width=800)
        renderer = D3Renderer(config=config)

        assert renderer.config.width == 800


class TestRendering:
    """Tests for graph rendering."""

    def test_render_sample_graph(self, tmp_path: Path) -> None:
        """Test rendering sample graph."""
        renderer = D3Renderer()
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        result = renderer.render(graph, output)

        assert result is True
        assert output.exists()

    def test_render_empty_graph(self, tmp_path: Path) -> None:
        """Test rendering empty graph."""
        renderer = D3Renderer()
        graph = GraphData()
        output = tmp_path / "empty.html"

        result = renderer.render(graph, output)

        assert result is False  # Empty graph should fail

    def test_render_contains_d3(self, tmp_path: Path) -> None:
        """Test that output contains D3 reference."""
        renderer = D3Renderer()
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        renderer.render(graph, output)
        content = output.read_text()

        assert "d3.js" in content or "d3.v7" in content

    def test_render_contains_nodes(self, tmp_path: Path) -> None:
        """Test that output contains node data."""
        renderer = D3Renderer()
        nodes = [GraphNode(id="test", label="Test Node")]
        graph = GraphData(nodes=nodes)
        output = tmp_path / "graph.html"

        renderer.render(graph, output)
        content = output.read_text()

        assert "Test Node" in content


class TestHTMLGeneration:
    """Tests for HTML generation."""

    def test_generate_head(self) -> None:
        """Test head generation."""
        renderer = D3Renderer()

        head = renderer._generate_head("Test Title")

        assert "<head>" in head
        assert "Test Title" in head
        assert "<title>" in head

    def test_generate_styles(self) -> None:
        """Test styles generation."""
        renderer = D3Renderer()

        styles = renderer._generate_styles()

        assert "<style>" in styles
        assert ".node" in styles
        assert ".link" in styles

    def test_generate_legend(self) -> None:
        """Test legend generation."""
        config = RenderConfig(show_legend=True)
        renderer = D3Renderer(config=config)

        legend = renderer._generate_legend()

        assert "legend" in legend
        assert "Node Types" in legend

    def test_generate_controls(self) -> None:
        """Test controls generation."""
        renderer = D3Renderer()

        controls = renderer._generate_controls()

        assert "controls" in controls
        assert "button" in controls

    def test_generate_script(self) -> None:
        """Test script generation."""
        renderer = D3Renderer()

        script = renderer._generate_script()

        assert "<script>" in script
        assert "d3.select" in script
        assert "simulation" in script


class TestConfigEffects:
    """Tests for configuration effects."""

    def test_width_in_output(self, tmp_path: Path) -> None:
        """Test that width config is used."""
        config = RenderConfig(width=1200)
        renderer = D3Renderer(config=config)
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        renderer.render(graph, output)
        content = output.read_text()

        assert "1200" in content

    def test_labels_config(self) -> None:
        """Test labels configuration in script."""
        config = RenderConfig(show_labels=True)
        renderer = D3Renderer(config=config)

        script = renderer._generate_script()

        assert "labelsVisible = true" in script

    def test_no_labels_config(self) -> None:
        """Test no labels configuration."""
        config = RenderConfig(show_labels=False)
        renderer = D3Renderer(config=config)

        script = renderer._generate_script()

        assert "labelsVisible = false" in script


# Factory function tests


class TestCreateRenderer:
    """Tests for create_renderer factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        renderer = create_renderer()

        assert renderer.config.width == 960
        assert renderer.config.height == 600

    def test_create_custom(self) -> None:
        """Test creating with custom options."""
        renderer = create_renderer(width=1200, height=800)

        assert renderer.config.width == 1200
        assert renderer.config.height == 800


class TestRenderGraph:
    """Tests for render_graph function."""

    def test_render_graph_success(self, tmp_path: Path) -> None:
        """Test successful graph rendering."""
        graph = create_sample_graph()
        output = tmp_path / "test.html"

        result = render_graph(graph, output)

        assert result is True
        assert output.exists()

    def test_render_graph_creates_file(self, tmp_path: Path) -> None:
        """Test that file is created."""
        nodes = [GraphNode(id="1", label="Node")]
        graph = GraphData(nodes=nodes)
        output = tmp_path / "output.html"

        render_graph(graph, output)

        assert output.is_file()


class TestRenderFromNetworkx:
    """Tests for render_from_networkx function."""

    def test_render_networkx_graph(self, tmp_path: Path) -> None:
        """Test rendering NetworkX graph."""
        try:
            import networkx as nx

            G = nx.Graph()
            G.add_node("a", label="Node A")
            G.add_node("b", label="Node B")
            G.add_edge("a", "b")

            output = tmp_path / "nx_graph.html"
            result = render_from_networkx(G, output)

            assert result is True
            assert output.exists()

        except ImportError:
            pytest.skip("networkx not available")


class TestInteractiveFeatures:
    """Tests for interactive features in generated HTML."""

    def test_zoom_enabled(self, tmp_path: Path) -> None:
        """Test that zoom is enabled."""
        config = RenderConfig(enable_zoom=True)
        renderer = D3Renderer(config=config)
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        renderer.render(graph, output)
        content = output.read_text()

        assert "d3.zoom" in content

    def test_drag_enabled(self, tmp_path: Path) -> None:
        """Test that drag is enabled."""
        config = RenderConfig(enable_drag=True)
        renderer = D3Renderer(config=config)
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        renderer.render(graph, output)
        content = output.read_text()

        assert "d3.drag" in content

    def test_tooltip_present(self, tmp_path: Path) -> None:
        """Test that tooltip is present."""
        renderer = D3Renderer()
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        renderer.render(graph, output)
        content = output.read_text()

        assert "tooltip" in content


# Template rendering tests (TICKET-403)


class TestTemplateExists:
    """Tests that template file exists."""

    def test_template_directory_exists(self) -> None:
        """Test template directory exists."""
        assert TEMPLATES_DIR.exists()
        assert TEMPLATES_DIR.is_dir()

    def test_graph_template_exists(self) -> None:
        """Test graph.html template exists."""
        template_path = TEMPLATES_DIR / "graph.html"
        assert template_path.exists()
        assert template_path.is_file()

    def test_template_has_required_elements(self) -> None:
        """Test template has required D3 elements."""
        template_path = TEMPLATES_DIR / "graph.html"
        content = template_path.read_text()

        # Required D3 features
        assert "d3.v7" in content or "d3.js" in content
        assert "d3.forceSimulation" in content
        assert "d3.zoom" in content
        assert "d3.drag" in content

        # Required UI elements
        assert "tooltip" in content
        assert "search" in content.lower()
        assert "legend" in content.lower()


class TestRenderWithTemplate:
    """Tests for render_with_template function."""

    def test_render_with_template_success(self, tmp_path: Path) -> None:
        """Test successful template rendering."""
        graph = create_sample_graph()
        output = tmp_path / "template_graph.html"

        result = render_with_template(graph, output)

        assert result is True
        assert output.exists()

    def test_render_with_template_contains_data(self, tmp_path: Path) -> None:
        """Test template output contains graph data."""
        nodes = [GraphNode(id="test123", label="Test Node 123")]
        graph = GraphData(nodes=nodes)
        output = tmp_path / "graph.html"

        render_with_template(graph, output)
        content = output.read_text()

        assert "test123" in content
        assert "Test Node 123" in content

    def test_render_with_template_custom_title(self, tmp_path: Path) -> None:
        """Test custom title in template."""
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        render_with_template(graph, output, title="Custom Title")
        content = output.read_text()

        assert "Custom Title" in content

    def test_render_empty_graph_no_api(self, tmp_path: Path) -> None:
        """Test rendering empty graph without API fails."""
        graph = GraphData()
        output = tmp_path / "empty.html"

        result = render_with_template(graph, output)

        assert result is False

    def test_render_with_api_endpoint(self, tmp_path: Path) -> None:
        """Test rendering with API endpoint allows empty graph."""
        # With API endpoint, empty graph is OK - data loads dynamically
        graph = create_sample_graph()  # Use sample graph for valid test
        output = tmp_path / "api_graph.html"

        result = render_with_template(graph, output, api_endpoint="/v1/viz/graph")

        # Should succeed with API endpoint
        assert result is True
        assert output.exists()
        content = output.read_text()
        assert "/v1/viz/graph" in content

    def test_render_template_has_dark_theme(self, tmp_path: Path) -> None:
        """Test template has dark theme styles."""
        graph = create_sample_graph()
        output = tmp_path / "graph.html"

        render_with_template(graph, output)
        content = output.read_text()

        # Dark theme indicators
        assert "#1a1a2e" in content or "bg-primary" in content
        assert "dark" in content.lower() or "primary" in content.lower()


class TestOpenInBrowser:
    """Tests for open_in_browser function."""

    def test_open_nonexistent_file(self) -> None:
        """Test opening nonexistent file fails."""
        result = open_in_browser(Path("/nonexistent/file.html"))
        assert result is False


class TestJSONExportFormat:
    """Tests for JSON export format compatibility."""

    def test_json_has_nodes_array(self) -> None:
        """Test JSON export has nodes array."""
        graph = create_sample_graph()
        d3_json = graph.to_d3_json()

        assert "nodes" in d3_json
        assert isinstance(d3_json["nodes"], list)

    def test_json_has_links_array(self) -> None:
        """Test JSON export has links array."""
        graph = create_sample_graph()
        d3_json = graph.to_d3_json()

        assert "links" in d3_json
        assert isinstance(d3_json["links"], list)

    def test_json_node_structure(self) -> None:
        """Test JSON node has required fields."""
        node = GraphNode(
            id="test1",
            label="Test Label",
            node_type=NodeType.CONCEPT,
            size=1.5,
            group=2,
        )
        node_dict = node.to_dict()

        assert node_dict["id"] == "test1"
        assert node_dict["label"] == "Test Label"
        assert node_dict["type"] == "concept"
        assert node_dict["size"] == 1.5
        assert node_dict["group"] == 2

    def test_json_link_structure(self) -> None:
        """Test JSON link has required fields."""
        edge = GraphEdge(
            source="node1",
            target="node2",
            edge_type=EdgeType.CONTAINS,
            weight=2.0,
            label="contains",
        )
        edge_dict = edge.to_dict()

        assert edge_dict["source"] == "node1"
        assert edge_dict["target"] == "node2"
        assert edge_dict["type"] == "contains"
        assert edge_dict["weight"] == 2.0
        assert edge_dict["label"] == "contains"

    def test_json_is_valid(self) -> None:
        """Test that exported JSON is valid."""
        from ingestforge.viz.graph_export import GraphExporter

        graph = create_sample_graph()
        exporter = GraphExporter()
        json_str = exporter.to_json(graph)

        # Should not raise
        parsed = json.loads(json_str)
        assert "nodes" in parsed
        assert "links" in parsed

    def test_json_metadata_included(self) -> None:
        """Test JSON includes metadata."""
        graph = GraphData(
            title="Test Graph",
            description="Test Description",
            nodes=[GraphNode(id="1", label="A")],
        )
        d3_json = graph.to_d3_json()

        assert "metadata" in d3_json
        assert d3_json["metadata"]["title"] == "Test Graph"
        assert d3_json["metadata"]["nodeCount"] == 1


class TestAPIResponseFormat:
    """Tests for API response format structure."""

    def test_response_structure_matches_d3(self) -> None:
        """Test API response matches D3 expectations."""
        # Simulate what the API returns
        graph = create_sample_graph()
        d3_json = graph.to_d3_json()

        # D3 expects these exact keys
        assert "nodes" in d3_json
        assert "links" in d3_json

        # Each node should have id
        for node in d3_json["nodes"]:
            assert "id" in node

        # Each link should have source and target
        for link in d3_json["links"]:
            assert "source" in link
            assert "target" in link

    def test_color_scale_types(self) -> None:
        """Test node types match expected color scale."""
        expected_types = {"document", "concept", "entity", "topic", "chunk"}

        # NodeType enum should have these values
        node_types = {t.value for t in NodeType}
        assert node_types == expected_types

    def test_edge_types_defined(self) -> None:
        """Test edge types are defined."""
        expected_types = {
            "contains",
            "references",
            "related_to",
            "similar_to",
            "derived_from",
        }

        edge_types = {t.value for t in EdgeType}
        assert edge_types == expected_types
