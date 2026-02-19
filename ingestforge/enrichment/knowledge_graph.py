"""Knowledge Graph Storage and Query.

Stores entities and relationships in a graph structure using NetworkX.
Supports multi-hop queries, graph traversal, and visualization."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from ingestforge.enrichment.relationships import Relationship
from ingestforge.enrichment.entity_linker import EntityProfile
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str  # Unique identifier
    label: str  # Display label
    node_type: str  # Entity type (PERSON, ORG, etc.)
    properties: Dict[str, Any] = field(default_factory=dict)  # Metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type,
            "properties": self.properties,
        }


@dataclass
class GraphEdge:
    """An edge (relationship) in the knowledge graph."""

    source: str  # Source node ID
    target: str  # Target node ID
    relation: str  # Relationship type
    confidence: float = 1.0  # Confidence score
    properties: Dict[str, Any] = field(default_factory=dict)  # Metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
            "properties": self.properties,
        }


class KnowledgeGraph:
    """
    Knowledge graph storage using NetworkX.

    Stores entities as nodes and relationships as edges.
    Supports multi-hop queries and graph visualization.
    """

    def __init__(self) -> None:
        """Initialize knowledge graph."""
        try:
            import networkx as nx

            self._graph = nx.MultiDiGraph()  # Directed, multiple edges
            self._nx = nx
            self._networkx_available = True
        except ImportError:
            logger.warning(
                "NetworkX not installed. Graph features limited. "
                "Install with: pip install networkx"
            )
            self._graph = None
            self._nx = None
            self._networkx_available = False

            # Fallback: simple dict-based storage
            self._nodes: Dict[str, GraphNode] = {}
            self._edges: List[GraphEdge] = []

    def is_available(self) -> bool:
        """Check if graph storage is available."""
        return self._networkx_available

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add node to graph.

        Args:
            node_id: Unique node identifier
            label: Display label
            node_type: Entity type
            properties: Optional metadata
        """
        if not node_id or not label:
            logger.warning("Invalid node: empty ID or label")
            return

        node = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties or {},
        )

        if self._networkx_available:
            self._graph.add_node(
                node_id,
                label=label,
                type=node_type,
                **properties or {},
            )
        else:
            self._nodes[node_id] = node

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        confidence: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add edge (relationship) to graph.

        Args:
            source: Source node ID
            target: Target node ID
            relation: Relationship type
            confidence: Confidence score
            properties: Optional metadata
        """
        if not source or not target or not relation:
            logger.warning("Invalid edge: empty source, target, or relation")
            return

        edge = GraphEdge(
            source=source,
            target=target,
            relation=relation,
            confidence=confidence,
            properties=properties or {},
        )

        if self._networkx_available:
            self._graph.add_edge(
                source,
                target,
                relation=relation,
                confidence=confidence,
                **properties or {},
            )
        else:
            self._edges.append(edge)

    def add_entity(self, entity_profile: EntityProfile) -> None:
        """
        Add entity from EntityProfile.

        Args:
            entity_profile: Entity profile from entity linker
        """
        self.add_node(
            node_id=entity_profile.canonical_name,
            label=entity_profile.canonical_name,
            node_type=entity_profile.entity_type,
            properties={
                "variations": list(entity_profile.variations),
                "documents": list(entity_profile.documents),
                "mention_count": entity_profile.mention_count,
            },
        )

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add relationship to graph.

        Args:
            relationship: Relationship object
        """
        # Ensure nodes exist
        self.add_node(relationship.subject, relationship.subject, "ENTITY")
        self.add_node(relationship.object, relationship.object, "ENTITY")

        # Add edge
        self.add_edge(
            source=relationship.subject,
            target=relationship.object,
            relation=relationship.predicate,
            confidence=relationship.confidence,
            properties={
                "context": relationship.context,
                "start": relationship.start_char,
                "end": relationship.end_char,
            },
        )

    def add_spatial_link(self, node_id: str, spatial_data: Dict[str, Any]) -> bool:
        """
        Add spatial linkage metadata to a node.

        Persist visual evidence coordinates on graph nodes.
        Rule #2: Bound links list.
        Rule #7: Return success status.
        """
        if not self._networkx_available or node_id not in self._graph:
            return False

        links = self._graph.nodes[node_id].get("spatial_links", [])
        links.append(spatial_data)

        # JPL Rule #2: Strictly bound to 10 visual links per node
        self._graph.nodes[node_id]["spatial_links"] = links[:10]
        return True

    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get immediate neighbors of a node.

        Args:
            node_id: Node identifier

        Returns:
            List of neighbor node IDs
        """
        if self._networkx_available:
            if node_id not in self._graph:
                return []
            # Get both successors (outgoing) and predecessors (incoming)
            return list(
                set(
                    list(self._graph.successors(node_id))
                    + list(self._graph.predecessors(node_id))
                )
            )
        else:
            # Fallback
            neighbors = set()
            for edge in self._edges:
                if edge.source == node_id:
                    neighbors.add(edge.target)
                if edge.target == node_id:
                    neighbors.add(edge.source)
            return list(neighbors)

    def get_paths(self, source: str, target: str, max_hops: int = 3) -> List[List[str]]:
        """
        Find paths between two nodes (multi-hop query).

        Args:
            source: Source node ID
            target: Target node ID
            max_hops: Maximum path length

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if not self._networkx_available:
            logger.warning("Multi-hop queries require NetworkX")
            return []

        if source not in self._graph or target not in self._graph:
            return []

        try:
            # Find all simple paths up to max length
            paths = list(
                self._nx.all_simple_paths(self._graph, source, target, cutoff=max_hops)
            )
            return paths
        except self._nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Path finding error: {e}")
            return []

    def _expand_neighbors(self, node: str, visited: set) -> set:
        """
        Find all unvisited neighbors of a node.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            node: Node ID to expand
            visited: Set of already visited nodes

        Returns:
            Set of new neighbors not in visited
        """
        assert node is not None, "Node cannot be None"
        assert visited is not None, "Visited set cannot be None"

        new_neighbors = set()
        for neighbor in self.get_neighbors(node):
            if neighbor not in visited:
                new_neighbors.add(neighbor)
        return new_neighbors

    def _find_nodes_within_depth(self, node_id: str, depth: int) -> set:
        """
        Find all nodes within depth hops using BFS.

        Rule #1: Zero nesting - helper extracts neighbor finding
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            node_id: Center node ID
            depth: Number of hops to include

        Returns:
            Set of node IDs within depth
        """
        assert node_id is not None, "Node ID cannot be None"
        assert depth >= 0, "Depth must be non-negative"

        visited = {node_id}
        current_level = {node_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                new_neighbors = self._expand_neighbors(node, visited)
                visited.update(new_neighbors)
                next_level.update(new_neighbors)
            current_level = next_level

        return visited

    def _copy_nodes_to_subgraph(
        self, nx_subgraph: Any, target_graph: "KnowledgeGraph"
    ) -> None:
        """
        Copy nodes from NetworkX subgraph to KnowledgeGraph.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            nx_subgraph: NetworkX subgraph
            target_graph: Target KnowledgeGraph to copy to
        """
        assert nx_subgraph is not None, "Subgraph cannot be None"
        assert target_graph is not None, "Target graph cannot be None"

        for node in nx_subgraph.nodes(data=True):
            target_graph.add_node(
                node_id=node[0],
                label=node[1].get("label", node[0]),
                node_type=node[1].get("type", "ENTITY"),
                properties={
                    k: v for k, v in node[1].items() if k not in ["label", "type"]
                },
            )

    def _copy_edges_to_subgraph(
        self, nx_subgraph: Any, target_graph: "KnowledgeGraph"
    ) -> None:
        """
        Copy edges from NetworkX subgraph to KnowledgeGraph.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            nx_subgraph: NetworkX subgraph
            target_graph: Target KnowledgeGraph to copy to
        """
        assert nx_subgraph is not None, "Subgraph cannot be None"
        assert target_graph is not None, "Target graph cannot be None"

        for edge in nx_subgraph.edges(data=True, keys=True):
            target_graph.add_edge(
                source=edge[0],
                target=edge[1],
                relation=edge[3].get("relation", "RELATES_TO"),
                confidence=edge[3].get("confidence", 1.0),
                properties={
                    k: v
                    for k, v in edge[3].items()
                    if k not in ["relation", "confidence"]
                },
            )

    def get_subgraph(self, node_id: str, depth: int = 2) -> "KnowledgeGraph":
        """
        Extract subgraph around a node.

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #4: Function <60 lines (reduced from 61)
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            node_id: Center node ID
            depth: Number of hops to include

        Returns:
            New KnowledgeGraph with subgraph
        """
        subgraph = KnowledgeGraph()
        if not self._networkx_available:
            return subgraph
        if node_id not in self._graph:
            return subgraph
        visited = self._find_nodes_within_depth(node_id, depth)

        # Extract NetworkX subgraph
        nx_subgraph = self._graph.subgraph(visited)
        self._copy_nodes_to_subgraph(nx_subgraph, subgraph)
        self._copy_edges_to_subgraph(nx_subgraph, subgraph)

        return subgraph

    def get_node_count(self) -> int:
        """Get number of nodes in graph."""
        if self._networkx_available:
            return self._graph.number_of_nodes()
        return len(self._nodes)

    def get_edge_count(self) -> int:
        """Get number of edges in graph."""
        if self._networkx_available:
            return self._graph.number_of_edges()
        return len(self._edges)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
            "networkx_available": self._networkx_available,
        }

        if self._networkx_available and self._graph.number_of_nodes() > 0:
            stats["density"] = self._nx.density(self._graph)
            stats["is_connected"] = self._nx.is_weakly_connected(self._graph)

        return stats

    def export_json(self, filepath: Path) -> None:
        """
        Export graph to JSON format.

        Args:
            filepath: Output file path
        """
        if self._networkx_available:
            data = self._nx.node_link_data(self._graph)
        else:
            # Fallback
            data = {
                "nodes": [node.to_dict() for node in self._nodes.values()],
                "links": [edge.to_dict() for edge in self._edges],
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Graph exported to: {filepath}")

    def export_dot(self, filepath: Path) -> None:
        """
        Export graph to DOT format (GraphViz).

        Args:
            filepath: Output file path
        """
        if not self._networkx_available:
            logger.warning("DOT export requires NetworkX")
            return

        try:
            from networkx.drawing.nx_pydot import write_dot

            write_dot(self._graph, filepath)
            logger.info(f"Graph exported to DOT: {filepath}")
        except ImportError:
            logger.warning("pydot not installed. DOT export unavailable.")

    def visualize_text(self, max_nodes: int = 20) -> str:
        """
        Create text-based visualization of graph.

        Args:
            max_nodes: Maximum nodes to show

        Returns:
            Text visualization
        """
        lines = []
        lines.append("Knowledge Graph")
        lines.append("=" * 50)
        lines.append(f"Nodes: {self.get_node_count()}")
        lines.append(f"Edges: {self.get_edge_count()}")
        lines.append("")

        if self._networkx_available:
            # Show some relationships
            lines.append("Sample Relationships:")
            count = 0
            for source, target, data in self._graph.edges(data=True):
                if count >= max_nodes:
                    break
                relation = data.get("relation", "relates_to")
                lines.append(f"  {source} --[{relation}]--> {target}")
                count += 1

            if self.get_edge_count() > max_nodes:
                lines.append(f"  ... and {self.get_edge_count() - max_nodes} more")
        else:
            # Fallback
            lines.append("Sample Relationships:")
            for i, edge in enumerate(self._edges[:max_nodes]):
                lines.append(f"  {edge.source} --[{edge.relation}]--> {edge.target}")

        return "\n".join(lines)


def build_knowledge_graph_from_chunks(
    chunks: List[Any],
    entity_extractor: Any,
    relationship_extractor: Any,
) -> KnowledgeGraph:
    """
    Build knowledge graph from chunks.

    Args:
        chunks: List of chunks
        entity_extractor: Entity extractor instance
        relationship_extractor: Relationship extractor instance

    Returns:
        Populated KnowledgeGraph
    """
    graph = KnowledgeGraph()

    for chunk in chunks:
        # Extract relationships
        text = getattr(chunk, "content", str(chunk))
        relationships = relationship_extractor.extract(text)

        # Add to graph
        for rel in relationships:
            graph.add_relationship(rel)

    logger.info(
        f"Built graph: {graph.get_node_count()} nodes, {graph.get_edge_count()} edges"
    )
    return graph


# =============================================================================
# KnowledgeGraphBuilder - Fluent API for Graph Construction
# =============================================================================


class KnowledgeGraphBuilder:
    """Fluent builder for constructing knowledge graphs.

    Provides a step-by-step API for adding entities and relationships,
    then building and exporting the graph in various formats.

    Example:
        >>> builder = KnowledgeGraphBuilder()
        >>> builder.add_entities(entities)
        >>> builder.add_relationships(relationships)
        >>> graph = builder.build()
        >>> mermaid = builder.to_mermaid()

            - Rule #1: Early returns, max 3 nesting
        - Rule #4: Functions <60 lines
        - Rule #9: Full type hints
    """

    def __init__(self) -> None:
        """Initialize builder with empty graph."""
        self._graph = KnowledgeGraph()
        self._entities: List[Any] = []
        self._relationships: List[Relationship] = []

    def add_entities(self, entities: List[Any]) -> "KnowledgeGraphBuilder":
        """Add entities to the graph.

        Args:
            entities: List of Entity objects from NER

        Returns:
            Self for method chaining

        Rule #4: Function <60 lines
        """
        for entity in entities:
            # Extract entity details
            text = getattr(entity, "text", str(entity))
            entity_type = getattr(entity, "type", "ENTITY")
            normalized = getattr(entity, "normalized", text)

            # Add as node
            self._graph.add_node(
                node_id=normalized or text,
                label=text,
                node_type=entity_type,
                properties={
                    "original_text": text,
                    "confidence": getattr(entity, "confidence", 1.0),
                },
            )
            self._entities.append(entity)

        return self

    def add_relationships(
        self, relationships: List[Relationship]
    ) -> "KnowledgeGraphBuilder":
        """Add relationships to the graph.

        Args:
            relationships: List of Relationship objects

        Returns:
            Self for method chaining

        Rule #4: Function <60 lines
        """
        for rel in relationships:
            self._graph.add_relationship(rel)
            self._relationships.append(rel)

        return self

    def add_spatial_evidence(self, artifacts: List[Any]) -> "KnowledgeGraphBuilder":
        """
        Link spatial evidence from artifacts to the graph.

        Triggers the SpatialLinkageEngine to map bboxes to nodes.
        Rule #4: Function < 60 lines.
        """
        from ingestforge.enrichment.spatial_linker import SpatialLinkageEngine

        linker = SpatialLinkageEngine(self._graph)
        linker.link_visual_artifacts(artifacts)

        return self

    def build(self) -> KnowledgeGraph:
        """Build and return the knowledge graph.

        Returns:
            Populated KnowledgeGraph
        """
        return self._graph

    def _sanitize_mermaid_id(self, text: str) -> str:
        """Sanitize text for use as Mermaid node ID.

        Args:
            text: Raw text to sanitize

        Returns:
            Sanitized ID safe for Mermaid

        Rule #4: Function <60 lines
        """
        # Replace non-alphanumeric with underscore
        sanitized = "".join(c if c.isalnum() else "_" for c in text)
        # Ensure starts with letter
        if sanitized and sanitized[0].isdigit():
            sanitized = "n" + sanitized
        return sanitized or "node"

    def _get_mermaid_shape(self, node_type: str) -> Tuple[str, str]:
        """Get Mermaid shape brackets for node type.

        Args:
            node_type: Entity type

        Returns:
            Tuple of (open_bracket, close_bracket)

        Rule #4: Function <60 lines
        """
        shapes = {
            "PERSON": ("([", "])"),  # Stadium shape
            "ORG": ("[[", "]]"),  # Subroutine shape
            "GPE": ("{{", "}}"),  # Hexagon
            "DATE": ("[/", "/]"),  # Parallelogram
            "EVENT": ("((", "))"),  # Circle
            "WORK_OF_ART": (">", "]"),  # Flag
        }
        return shapes.get(node_type, ("[", "]"))

    def to_mermaid(self, max_nodes: int = 50) -> str:
        """Export graph as Mermaid diagram syntax.

        Creates a flowchart diagram that can be rendered by Mermaid.

        Args:
            max_nodes: Maximum number of nodes to include

        Returns:
            Mermaid diagram syntax as string

        Rule #1: Early return for empty graph
        Rule #4: Function <60 lines
        """
        if self._graph.get_node_count() == 0:
            return "graph LR\n    empty[No data]"

        lines = ["graph LR"]

        # Track nodes already added
        added_nodes: Set[str] = set()
        edge_count = 0

        # Add edges with node definitions
        if self._graph._networkx_available:
            for source, target, data in self._graph._graph.edges(data=True):
                if edge_count >= max_nodes:
                    break

                # Sanitize IDs
                src_id = self._sanitize_mermaid_id(source)
                tgt_id = self._sanitize_mermaid_id(target)

                # Get node types for shapes
                src_type = self._graph._graph.nodes[source].get("type", "ENTITY")
                tgt_type = self._graph._graph.nodes[target].get("type", "ENTITY")

                # Add source node if not yet added
                if src_id not in added_nodes:
                    open_b, close_b = self._get_mermaid_shape(src_type)
                    lines.append(f"    {src_id}{open_b}{source}{close_b}")
                    added_nodes.add(src_id)

                # Add target node if not yet added
                if tgt_id not in added_nodes:
                    open_b, close_b = self._get_mermaid_shape(tgt_type)
                    lines.append(f"    {tgt_id}{open_b}{target}{close_b}")
                    added_nodes.add(tgt_id)

                # Add edge
                relation = data.get("relation", "relates_to")
                lines.append(f"    {src_id} -->|{relation}| {tgt_id}")
                edge_count += 1

        else:
            # Fallback for non-NetworkX
            for edge in self._graph._edges[:max_nodes]:
                src_id = self._sanitize_mermaid_id(edge.source)
                tgt_id = self._sanitize_mermaid_id(edge.target)

                if src_id not in added_nodes:
                    lines.append(f"    {src_id}[{edge.source}]")
                    added_nodes.add(src_id)

                if tgt_id not in added_nodes:
                    lines.append(f"    {tgt_id}[{edge.target}]")
                    added_nodes.add(tgt_id)

                lines.append(f"    {src_id} -->|{edge.relation}| {tgt_id}")

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Export graph as JSON-serializable dictionary.

        Returns:
            Dictionary with nodes and edges

        Rule #4: Function <60 lines
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        if self._graph._networkx_available:
            for node, data in self._graph._graph.nodes(data=True):
                nodes.append(
                    {
                        "id": node,
                        "label": data.get("label", node),
                        "type": data.get("type", "ENTITY"),
                        **{k: v for k, v in data.items() if k not in ["label", "type"]},
                    }
                )

            for source, target, data in self._graph._graph.edges(data=True):
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": data.get("relation", "relates_to"),
                        "confidence": data.get("confidence", 1.0),
                    }
                )
        else:
            for node in self._graph._nodes.values():
                nodes.append(node.to_dict())
            for edge in self._graph._edges:
                edges.append(edge.to_dict())

        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": self._graph.get_statistics(),
        }

    def query(self, entity: str, depth: int = 2) -> "KnowledgeGraphBuilder":
        """Query subgraph around an entity.

        Args:
            entity: Entity name to center query on
            depth: Number of hops to include

        Returns:
            New KnowledgeGraphBuilder with subgraph

        Rule #1: Early return if entity not found
        Rule #4: Function <60 lines
        """
        subgraph = self._graph.get_subgraph(entity, depth)

        # Create new builder with subgraph
        new_builder = KnowledgeGraphBuilder()
        new_builder._graph = subgraph

        return new_builder

    def get_entity_count(self) -> int:
        """Get number of entities added."""
        return len(self._entities)

    def get_relationship_count(self) -> int:
        """Get number of relationships added."""
        return len(self._relationships)


# =============================================================================
# Convenience Functions
# =============================================================================


def build_graph_from_text(
    text: str,
    use_spacy: bool = True,
) -> KnowledgeGraphBuilder:
    """Build knowledge graph from raw text.

    Extracts entities and relationships, then builds graph.

    Args:
        text: Raw text to analyze
        use_spacy: Whether to use spaCy (default: True)

    Returns:
        KnowledgeGraphBuilder with populated graph

    Rule #4: Function <60 lines
    """
    from ingestforge.enrichment.ner import NERExtractor
    from ingestforge.enrichment.relationships import SpacyRelationshipExtractor

    # Extract entities
    ner = NERExtractor()
    entities = ner.extract(text)

    # Extract relationships
    rel_extractor = SpacyRelationshipExtractor(use_spacy=use_spacy)
    relationships = rel_extractor.extract(text)

    # Build graph
    builder = KnowledgeGraphBuilder()
    builder.add_entities(entities)
    builder.add_relationships(relationships)

    return builder


def export_to_mermaid_file(
    builder: KnowledgeGraphBuilder,
    filepath: Path,
    title: str = "Knowledge Graph",
) -> None:
    """Export graph to Mermaid markdown file.

    Args:
        builder: KnowledgeGraphBuilder with graph
        filepath: Output file path
        title: Optional title for the document

    Rule #4: Function <60 lines
    """
    mermaid_syntax = builder.to_mermaid()

    content = f"# {title}\n\n```mermaid\n{mermaid_syntax}\n```\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Mermaid diagram exported to: {filepath}")
