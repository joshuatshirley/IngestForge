"""Tests for graph data exporter.

Tests conversion of graphs to D3-compatible JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestforge.viz.graph_export import (
    GraphExporter,
    GraphData,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    create_exporter,
    export_to_d3_json,
    create_sample_graph,
    MAX_NODES,
    MAX_EDGES,
    MAX_LABEL_LENGTH,
)

# NodeType tests


class TestNodeType:
    """Tests for NodeType enum."""

    def test_types_defined(self) -> None:
        """Test all node types are defined."""
        types = [t.value for t in NodeType]

        assert "document" in types
        assert "concept" in types
        assert "entity" in types

    def test_type_count(self) -> None:
        """Test correct number of types."""
        assert len(NodeType) == 5


# EdgeType tests


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_types_defined(self) -> None:
        """Test all edge types are defined."""
        types = [t.value for t in EdgeType]

        assert "contains" in types
        assert "references" in types
        assert "related_to" in types

    def test_type_count(self) -> None:
        """Test correct number of types."""
        assert len(EdgeType) == 5


# GraphNode tests


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_node_creation(self) -> None:
        """Test creating a node."""
        node = GraphNode(
            id="1",
            label="Test Node",
            node_type=NodeType.CONCEPT,
        )

        assert node.id == "1"
        assert node.label == "Test Node"

    def test_node_to_dict(self) -> None:
        """Test converting node to dict."""
        node = GraphNode(
            id="1",
            label="Test",
            node_type=NodeType.DOCUMENT,
            group=2,
        )

        d = node.to_dict()

        assert d["id"] == "1"
        assert d["type"] == "document"
        assert d["group"] == 2

    def test_node_label_truncated(self) -> None:
        """Test that long labels are truncated."""
        long_label = "x" * (MAX_LABEL_LENGTH + 10)
        node = GraphNode(id="1", label=long_label)

        d = node.to_dict()

        assert len(d["label"]) == MAX_LABEL_LENGTH


# GraphEdge tests


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_edge_creation(self) -> None:
        """Test creating an edge."""
        edge = GraphEdge(
            source="1",
            target="2",
            edge_type=EdgeType.CONTAINS,
        )

        assert edge.source == "1"
        assert edge.target == "2"

    def test_edge_to_dict(self) -> None:
        """Test converting edge to dict."""
        edge = GraphEdge(
            source="a",
            target="b",
            edge_type=EdgeType.REFERENCES,
            weight=2.5,
        )

        d = edge.to_dict()

        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["type"] == "references"
        assert d["weight"] == 2.5


# GraphData tests


class TestGraphData:
    """Tests for GraphData dataclass."""

    def test_default_data(self) -> None:
        """Test default graph data."""
        data = GraphData()

        assert data.nodes == []
        assert data.edges == []

    def test_node_count(self) -> None:
        """Test node count property."""
        nodes = [
            GraphNode(id="1", label="A"),
            GraphNode(id="2", label="B"),
        ]
        data = GraphData(nodes=nodes)

        assert data.node_count == 2

    def test_edge_count(self) -> None:
        """Test edge count property."""
        edges = [
            GraphEdge(source="1", target="2"),
        ]
        data = GraphData(edges=edges)

        assert data.edge_count == 1

    def test_to_d3_json(self) -> None:
        """Test conversion to D3 JSON."""
        nodes = [GraphNode(id="1", label="A")]
        edges = [GraphEdge(source="1", target="1")]
        data = GraphData(nodes=nodes, edges=edges, title="Test")

        d3 = data.to_d3_json()

        assert "nodes" in d3
        assert "links" in d3
        assert "metadata" in d3
        assert d3["metadata"]["title"] == "Test"


# GraphExporter tests


class TestGraphExporter:
    """Tests for GraphExporter class."""

    def test_exporter_creation(self) -> None:
        """Test creating exporter."""
        exporter = GraphExporter()

        assert exporter is not None

    def test_to_json(self) -> None:
        """Test JSON export."""
        exporter = GraphExporter()
        data = create_sample_graph()

        json_str = exporter.to_json(data)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "nodes" in parsed
        assert "links" in parsed

    def test_to_file(self, tmp_path: Path) -> None:
        """Test file export."""
        exporter = GraphExporter()
        data = create_sample_graph()
        output = tmp_path / "graph.json"

        result = exporter.to_file(data, output)

        assert result is True
        assert output.exists()

    def test_color_palette(self) -> None:
        """Test color palette."""
        exporter = GraphExporter()

        # Should have colors
        assert len(exporter._color_palette) > 0

    def test_get_color_cycles(self) -> None:
        """Test that colors cycle."""
        exporter = GraphExporter()
        palette_size = len(exporter._color_palette)

        color0 = exporter._get_color(0)
        color_cycle = exporter._get_color(palette_size)

        assert color0 == color_cycle


class TestNetworkXConversion:
    """Tests for NetworkX conversion."""

    def test_from_networkx_none(self) -> None:
        """Test converting None graph."""
        exporter = GraphExporter()

        result = exporter.from_networkx(None)

        assert result.node_count == 0

    def test_from_networkx_valid(self) -> None:
        """Test converting valid NetworkX graph."""
        try:
            import networkx as nx

            G = nx.Graph()
            G.add_node("a", label="Node A", type="concept")
            G.add_node("b", label="Node B", type="entity")
            G.add_edge("a", "b", type="related_to")

            exporter = GraphExporter()
            result = exporter.from_networkx(G)

            assert result.node_count == 2
            assert result.edge_count == 1

        except ImportError:
            pytest.skip("networkx not available")


# Factory function tests


class TestCreateExporter:
    """Tests for create_exporter factory."""

    def test_create(self) -> None:
        """Test creating exporter."""
        exporter = create_exporter()

        assert isinstance(exporter, GraphExporter)


class TestExportToD3Json:
    """Tests for export_to_d3_json function."""

    def test_export_with_networkx(self) -> None:
        """Test exporting NetworkX graph."""
        try:
            import networkx as nx

            G = nx.Graph()
            G.add_node("a")
            G.add_node("b")
            G.add_edge("a", "b")

            json_str = export_to_d3_json(G)

            parsed = json.loads(json_str)
            assert len(parsed["nodes"]) == 2

        except ImportError:
            pytest.skip("networkx not available")


class TestCreateSampleGraph:
    """Tests for create_sample_graph function."""

    def test_sample_graph(self) -> None:
        """Test creating sample graph."""
        sample = create_sample_graph()

        assert sample.node_count > 0
        assert sample.edge_count > 0
        assert sample.title != ""


class TestMaxLimits:
    """Tests for max limits enforcement."""

    def test_max_nodes_constant(self) -> None:
        """Test MAX_NODES is defined."""
        assert MAX_NODES > 0
        assert MAX_NODES == 500

    def test_max_edges_constant(self) -> None:
        """Test MAX_EDGES is defined."""
        assert MAX_EDGES > 0
        assert MAX_EDGES == 2000
