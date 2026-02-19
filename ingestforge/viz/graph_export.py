"""Graph Data Exporter for knowledge graph visualization.

Converts NetworkX graphs to D3-compatible JSON format
for interactive visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_NODES = 500
MAX_EDGES = 2000
MAX_LABEL_LENGTH = 50
MAX_METADATA_SIZE = 1000


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""

    DOCUMENT = "document"
    CHUNK = "chunk"
    ENTITY = "entity"
    CONCEPT = "concept"
    TOPIC = "topic"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""

    CONTAINS = "contains"
    REFERENCES = "references"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    label: str
    node_type: NodeType = NodeType.CONCEPT
    metadata: Dict[str, Any] = field(default_factory=dict)
    size: float = 1.0
    color: str = ""
    group: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to D3-compatible dict."""
        return {
            "id": self.id,
            "label": self.label[:MAX_LABEL_LENGTH],
            "type": self.node_type.value,
            "size": self.size,
            "color": self.color,
            "group": self.group,
            "metadata": dict(list(self.metadata.items())[:MAX_METADATA_SIZE]),
        }


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""

    source: str
    target: str
    edge_type: EdgeType = EdgeType.RELATED_TO
    weight: float = 1.0
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to D3-compatible dict."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "weight": self.weight,
            "label": self.label[:MAX_LABEL_LENGTH] if self.label else "",
        }


@dataclass
class GraphData:
    """Complete graph data structure."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    title: str = "Knowledge Graph"
    description: str = ""

    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return len(self.edges)

    def to_d3_json(self) -> Dict[str, Any]:
        """Convert to D3-compatible JSON structure."""
        return {
            "nodes": [n.to_dict() for n in self.nodes[:MAX_NODES]],
            "links": [e.to_dict() for e in self.edges[:MAX_EDGES]],
            "metadata": {
                "title": self.title,
                "description": self.description,
                "nodeCount": min(len(self.nodes), MAX_NODES),
                "linkCount": min(len(self.edges), MAX_EDGES),
            },
        }


class GraphExporter:
    """Exports knowledge graphs to various formats.

    Converts internal graph representations to D3-compatible
    JSON for browser-based visualization.
    """

    def __init__(self) -> None:
        """Initialize the exporter."""
        self._color_palette = self._default_color_palette()

    def from_networkx(self, graph: Any) -> GraphData:
        """Convert NetworkX graph to GraphData.

        Args:
            graph: NetworkX graph object

        Returns:
            GraphData structure
        """
        if graph is None:
            return GraphData()

        try:
            import networkx as nx

            if not isinstance(graph, (nx.Graph, nx.DiGraph)):
                logger.error("Invalid graph type")
                return GraphData()

        except ImportError:
            logger.error("networkx not available")
            return GraphData()

        # Extract nodes
        nodes = self._extract_nodes(graph)

        # Extract edges
        edges = self._extract_edges(graph)

        return GraphData(nodes=nodes, edges=edges)

    def _extract_nodes(self, graph: Any) -> List[GraphNode]:
        """Extract nodes from NetworkX graph.

        Args:
            graph: NetworkX graph

        Returns:
            List of GraphNode
        """
        nodes: List[GraphNode] = []

        for node_id, attrs in list(graph.nodes(data=True))[:MAX_NODES]:
            # Get node type
            node_type_str = attrs.get("type", "concept")
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.CONCEPT

            # Get label
            label = attrs.get("label", str(node_id))

            # Get group for coloring
            group = attrs.get("group", 0)

            node = GraphNode(
                id=str(node_id),
                label=str(label),
                node_type=node_type,
                metadata=dict(attrs),
                size=attrs.get("size", 1.0),
                color=self._get_color(group),
                group=group,
            )
            nodes.append(node)

        return nodes

    def _extract_edges(self, graph: Any) -> List[GraphEdge]:
        """Extract edges from NetworkX graph.

        Args:
            graph: NetworkX graph

        Returns:
            List of GraphEdge
        """
        edges: List[GraphEdge] = []

        for source, target, attrs in list(graph.edges(data=True))[:MAX_EDGES]:
            # Get edge type
            edge_type_str = attrs.get("type", "related_to")
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                edge_type = EdgeType.RELATED_TO

            edge = GraphEdge(
                source=str(source),
                target=str(target),
                edge_type=edge_type,
                weight=attrs.get("weight", 1.0),
                label=attrs.get("label", ""),
                metadata=dict(attrs),
            )
            edges.append(edge)

        return edges

    def to_json(self, graph_data: GraphData) -> str:
        """Export graph to JSON string.

        Args:
            graph_data: Graph data structure

        Returns:
            JSON string
        """
        d3_data = graph_data.to_d3_json()
        return json.dumps(d3_data, indent=2)

    def to_file(self, graph_data: GraphData, output_path: Path) -> bool:
        """Export graph to JSON file.

        Args:
            graph_data: Graph data structure
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            json_str = self.to_json(graph_data)
            output_path.write_text(json_str, encoding="utf-8")
            logger.info(f"Graph exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return False

    def _get_color(self, group: int) -> str:
        """Get color for group.

        Args:
            group: Group index

        Returns:
            Color hex code
        """
        return self._color_palette[group % len(self._color_palette)]

    def _default_color_palette(self) -> List[str]:
        """Get default color palette.

        Returns:
            List of hex color codes
        """
        return [
            "#4e79a7",  # Blue
            "#f28e2c",  # Orange
            "#e15759",  # Red
            "#76b7b2",  # Teal
            "#59a14f",  # Green
            "#edc949",  # Yellow
            "#af7aa1",  # Purple
            "#ff9da7",  # Pink
            "#9c755f",  # Brown
            "#bab0ab",  # Gray
        ]


def create_exporter() -> GraphExporter:
    """Factory function to create exporter.

    Returns:
        Configured GraphExporter
    """
    return GraphExporter()


def export_to_d3_json(
    graph: Any,
    output_path: Optional[Path] = None,
) -> str:
    """Convenience function to export graph.

    Args:
        graph: NetworkX graph
        output_path: Optional output file path

    Returns:
        JSON string
    """
    exporter = create_exporter()
    graph_data = exporter.from_networkx(graph)

    if output_path:
        exporter.to_file(graph_data, output_path)

    return exporter.to_json(graph_data)


def create_sample_graph() -> GraphData:
    """Create a sample graph for testing.

    Returns:
        Sample GraphData
    """
    nodes = [
        GraphNode(id="1", label="Document A", node_type=NodeType.DOCUMENT, group=0),
        GraphNode(id="2", label="Concept X", node_type=NodeType.CONCEPT, group=1),
        GraphNode(id="3", label="Entity Y", node_type=NodeType.ENTITY, group=2),
        GraphNode(id="4", label="Topic Z", node_type=NodeType.TOPIC, group=1),
    ]

    edges = [
        GraphEdge(source="1", target="2", edge_type=EdgeType.CONTAINS),
        GraphEdge(source="1", target="3", edge_type=EdgeType.CONTAINS),
        GraphEdge(source="2", target="3", edge_type=EdgeType.RELATED_TO),
        GraphEdge(source="2", target="4", edge_type=EdgeType.SIMILAR_TO),
    ]

    return GraphData(
        nodes=nodes,
        edges=edges,
        title="Sample Knowledge Graph",
        description="A sample graph for testing visualization",
    )
