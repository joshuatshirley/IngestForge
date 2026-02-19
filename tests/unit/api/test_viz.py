"""Tests for visualization API endpoint (TICKET-403).

Tests the /v1/viz/graph endpoint for knowledge graph data."""

from __future__ import annotations


from ingestforge.api.routes.viz import (
    GraphNodeResponse,
    GraphLinkResponse,
    GraphMetadataResponse,
    GraphResponse,
    _convert_to_response,
    MAX_NODES,
    MAX_EDGES,
)
from ingestforge.viz.graph_export import (
    GraphData,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    create_sample_graph,
)


class TestGraphNodeResponse:
    """Tests for GraphNodeResponse model."""

    def test_node_response_creation(self) -> None:
        """Test creating node response."""
        node = GraphNodeResponse(
            id="test1",
            label="Test Node",
            type="concept",
        )

        assert node.id == "test1"
        assert node.label == "Test Node"
        assert node.type == "concept"

    def test_node_response_defaults(self) -> None:
        """Test node response defaults."""
        node = GraphNodeResponse(id="x", label="X")

        assert node.type == "concept"
        assert node.size == 1.0
        assert node.color == ""
        assert node.group == 0
        assert node.metadata == {}


class TestGraphLinkResponse:
    """Tests for GraphLinkResponse model."""

    def test_link_response_creation(self) -> None:
        """Test creating link response."""
        link = GraphLinkResponse(
            source="node1",
            target="node2",
            type="contains",
        )

        assert link.source == "node1"
        assert link.target == "node2"
        assert link.type == "contains"

    def test_link_response_defaults(self) -> None:
        """Test link response defaults."""
        link = GraphLinkResponse(source="a", target="b")

        assert link.type == "related_to"
        assert link.weight == 1.0
        assert link.label == ""


class TestGraphMetadataResponse:
    """Tests for GraphMetadataResponse model."""

    def test_metadata_response_creation(self) -> None:
        """Test creating metadata response."""
        meta = GraphMetadataResponse(
            title="Test Graph",
            description="Test Description",
            nodeCount=10,
            linkCount=5,
        )

        assert meta.title == "Test Graph"
        assert meta.nodeCount == 10

    def test_metadata_response_defaults(self) -> None:
        """Test metadata response defaults."""
        meta = GraphMetadataResponse()

        assert meta.title == "Knowledge Graph"
        assert meta.description == ""
        assert meta.nodeCount == 0
        assert meta.linkCount == 0


class TestGraphResponse:
    """Tests for GraphResponse model."""

    def test_response_creation(self) -> None:
        """Test creating full response."""
        nodes = [GraphNodeResponse(id="1", label="A")]
        links = [GraphLinkResponse(source="1", target="1")]
        metadata = GraphMetadataResponse(nodeCount=1, linkCount=1)

        response = GraphResponse(
            nodes=nodes,
            links=links,
            metadata=metadata,
        )

        assert len(response.nodes) == 1
        assert len(response.links) == 1
        assert response.metadata.nodeCount == 1


class TestConvertToResponse:
    """Tests for _convert_to_response function."""

    def test_convert_sample_graph(self) -> None:
        """Test converting sample graph to response."""
        graph = create_sample_graph()
        response = _convert_to_response(graph)

        assert len(response.nodes) == graph.node_count
        assert len(response.links) == graph.edge_count

    def test_convert_preserves_node_ids(self) -> None:
        """Test that node IDs are preserved."""
        nodes = [
            GraphNode(id="unique_id_1", label="Node 1"),
            GraphNode(id="unique_id_2", label="Node 2"),
        ]
        graph = GraphData(nodes=nodes)
        response = _convert_to_response(graph)

        node_ids = [n.id for n in response.nodes]
        assert "unique_id_1" in node_ids
        assert "unique_id_2" in node_ids

    def test_convert_preserves_link_connections(self) -> None:
        """Test that link connections are preserved."""
        nodes = [
            GraphNode(id="a", label="A"),
            GraphNode(id="b", label="B"),
        ]
        edges = [
            GraphEdge(source="a", target="b", edge_type=EdgeType.CONTAINS),
        ]
        graph = GraphData(nodes=nodes, edges=edges)
        response = _convert_to_response(graph)

        assert len(response.links) == 1
        assert response.links[0].source == "a"
        assert response.links[0].target == "b"

    def test_convert_node_type_to_string(self) -> None:
        """Test node type is converted to string."""
        nodes = [
            GraphNode(id="1", label="Doc", node_type=NodeType.DOCUMENT),
        ]
        graph = GraphData(nodes=nodes)
        response = _convert_to_response(graph)

        assert response.nodes[0].type == "document"

    def test_convert_edge_type_to_string(self) -> None:
        """Test edge type is converted to string."""
        nodes = [GraphNode(id="1", label="A")]
        edges = [
            GraphEdge(source="1", target="1", edge_type=EdgeType.REFERENCES),
        ]
        graph = GraphData(nodes=nodes, edges=edges)
        response = _convert_to_response(graph)

        assert response.links[0].type == "references"


class TestMaxLimits:
    """Tests for max limit constants."""

    def test_max_nodes_defined(self) -> None:
        """Test MAX_NODES is defined."""
        assert MAX_NODES > 0
        assert MAX_NODES == 500

    def test_max_edges_defined(self) -> None:
        """Test MAX_EDGES is defined."""
        assert MAX_EDGES > 0
        assert MAX_EDGES == 2000


class TestResponseJSONSerializable:
    """Tests that response is JSON serializable."""

    def test_node_response_json(self) -> None:
        """Test node response is JSON serializable."""
        node = GraphNodeResponse(
            id="1",
            label="Test",
            type="concept",
            metadata={"key": "value"},
        )

        # Should not raise
        json_data = node.model_dump()
        assert json_data["id"] == "1"
        assert json_data["metadata"]["key"] == "value"

    def test_full_response_json(self) -> None:
        """Test full response is JSON serializable."""
        graph = create_sample_graph()
        response = _convert_to_response(graph)

        # Should not raise
        json_data = response.model_dump()
        assert "nodes" in json_data
        assert "links" in json_data
        assert "metadata" in json_data


class TestD3Compatibility:
    """Tests for D3.js compatibility."""

    def test_response_has_nodes_and_links(self) -> None:
        """Test response structure matches D3 expectations."""
        graph = create_sample_graph()
        response = _convert_to_response(graph)
        json_data = response.model_dump()

        # D3 expects 'nodes' and 'links' arrays
        assert isinstance(json_data["nodes"], list)
        assert isinstance(json_data["links"], list)

    def test_nodes_have_id(self) -> None:
        """Test all nodes have id field."""
        graph = create_sample_graph()
        response = _convert_to_response(graph)

        for node in response.nodes:
            assert node.id is not None
            assert len(node.id) > 0

    def test_links_have_source_target(self) -> None:
        """Test all links have source and target."""
        graph = create_sample_graph()
        response = _convert_to_response(graph)

        for link in response.links:
            assert link.source is not None
            assert link.target is not None

    def test_links_reference_valid_nodes(self) -> None:
        """Test links reference existing node IDs."""
        nodes = [
            GraphNode(id="n1", label="N1"),
            GraphNode(id="n2", label="N2"),
        ]
        edges = [
            GraphEdge(source="n1", target="n2"),
        ]
        graph = GraphData(nodes=nodes, edges=edges)
        response = _convert_to_response(graph)

        node_ids = {n.id for n in response.nodes}
        for link in response.links:
            assert link.source in node_ids
            assert link.target in node_ids
