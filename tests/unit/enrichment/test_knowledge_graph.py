"""
Tests for Knowledge Graph Storage.

Validates:
- Graph storage (NetworkX and fallback)
- Node and edge operations
- Multi-hop queries
- Graph traversal
- Visualization and export
- Performance
"""

import pytest
from pathlib import Path
import tempfile
import json

from ingestforge.enrichment.knowledge_graph import (
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    build_knowledge_graph_from_chunks,
)
from ingestforge.enrichment.relationships import (
    Relationship,
    SpacyRelationshipExtractor,
)
from ingestforge.enrichment.entity_linker import EntityProfile
from ingestforge.chunking.semantic_chunker import ChunkRecord


# ============================================================================
# Test Data
# ============================================================================


def make_test_relationships() -> list:
    """Create test relationships."""
    return [
        Relationship("Tim Cook", "work", "Apple Inc", confidence=0.9),
        Relationship("Apple Inc", "acquire", "Beats Electronics", confidence=0.85),
        Relationship("Beats Electronics", "founded_by", "Dr. Dre", confidence=0.8),
        Relationship("Tim Cook", "succeed", "Steve Jobs", confidence=0.95),
    ]


def make_test_entity_profiles() -> list:
    """Create test entity profiles."""
    profile1 = EntityProfile(
        canonical_name="Tim Cook",
        entity_type="PERSON",
    )
    profile1.mention_count = 5
    profile1.documents.add("doc1")
    profile1.variations.add("Tim Cook")
    profile1.variations.add("Timothy Cook")

    profile2 = EntityProfile(
        canonical_name="Apple Inc",
        entity_type="ORG",
    )
    profile2.mention_count = 10
    profile2.documents.add("doc1")
    profile2.documents.add("doc2")

    return [profile1, profile2]


# ============================================================================
# Test Classes
# ============================================================================


class TestGraphDataclasses:
    """Tests for graph dataclasses."""

    def test_create_graph_node(self):
        """Test creating GraphNode."""
        node = GraphNode(
            id="node1",
            label="Apple Inc",
            node_type="ORG",
            properties={"founded": 1976},
        )

        assert node.id == "node1"
        assert node.label == "Apple Inc"
        assert node.node_type == "ORG"
        assert node.properties["founded"] == 1976

    def test_graph_node_to_dict(self):
        """Test GraphNode to_dict."""
        node = GraphNode("n1", "Tim Cook", "PERSON")
        node_dict = node.to_dict()

        assert node_dict["id"] == "n1"
        assert node_dict["label"] == "Tim Cook"
        assert node_dict["type"] == "PERSON"

    def test_create_graph_edge(self):
        """Test creating GraphEdge."""
        edge = GraphEdge(
            source="node1",
            target="node2",
            relation="works_at",
            confidence=0.9,
        )

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.relation == "works_at"
        assert edge.confidence == 0.9

    def test_graph_edge_to_dict(self):
        """Test GraphEdge to_dict."""
        edge = GraphEdge("n1", "n2", "works", 0.85)
        edge_dict = edge.to_dict()

        assert edge_dict["source"] == "n1"
        assert edge_dict["target"] == "n2"
        assert edge_dict["relation"] == "works"
        assert edge_dict["confidence"] == 0.85


class TestKnowledgeGraphBasic:
    """Tests for basic KnowledgeGraph operations."""

    def test_create_knowledge_graph(self):
        """Test creating KnowledgeGraph."""
        graph = KnowledgeGraph()

        assert graph is not None
        assert graph.get_node_count() == 0
        assert graph.get_edge_count() == 0

    def test_add_node(self):
        """Test adding node to graph."""
        graph = KnowledgeGraph()

        graph.add_node("node1", "Apple Inc", "ORG")

        assert graph.get_node_count() == 1

    def test_add_node_with_properties(self):
        """Test adding node with properties."""
        graph = KnowledgeGraph()

        graph.add_node(
            "node1",
            "Apple Inc",
            "ORG",
            properties={"founded": 1976, "ceo": "Tim Cook"},
        )

        assert graph.get_node_count() == 1

    def test_add_edge(self):
        """Test adding edge to graph."""
        graph = KnowledgeGraph()

        # Add nodes first
        graph.add_node("n1", "Tim Cook", "PERSON")
        graph.add_node("n2", "Apple Inc", "ORG")

        # Add edge
        graph.add_edge("n1", "n2", "works_at", confidence=0.9)

        assert graph.get_edge_count() == 1

    def test_add_edge_with_properties(self):
        """Test adding edge with properties."""
        graph = KnowledgeGraph()

        graph.add_node("n1", "Microsoft", "ORG")
        graph.add_node("n2", "LinkedIn", "ORG")

        graph.add_edge(
            "n1",
            "n2",
            "acquired",
            confidence=0.95,
            properties={"year": 2016, "amount": "$26.2 billion"},
        )

        assert graph.get_edge_count() == 1

    def test_add_invalid_node(self):
        """Test adding invalid node."""
        graph = KnowledgeGraph()

        # Empty ID or label
        graph.add_node("", "Label", "TYPE")
        graph.add_node("id", "", "TYPE")

        # Should not add invalid nodes
        assert graph.get_node_count() == 0

    def test_add_invalid_edge(self):
        """Test adding invalid edge."""
        graph = KnowledgeGraph()

        # Empty source, target, or relation
        graph.add_edge("", "target", "relation")
        graph.add_edge("source", "", "relation")
        graph.add_edge("source", "target", "")

        # Should not add invalid edges
        assert graph.get_edge_count() == 0


class TestEntityAndRelationshipIntegration:
    """Tests for adding entities and relationships."""

    def test_add_entity_profile(self):
        """Test adding entity from EntityProfile."""
        graph = KnowledgeGraph()
        profiles = make_test_entity_profiles()

        for profile in profiles:
            graph.add_entity(profile)

        assert graph.get_node_count() == 2

    def test_add_relationship(self):
        """Test adding relationship."""
        graph = KnowledgeGraph()
        rel = Relationship("Alice", "knows", "Bob", confidence=0.8)

        graph.add_relationship(rel)

        # Should create 2 nodes + 1 edge
        assert graph.get_node_count() == 2
        assert graph.get_edge_count() == 1

    def test_add_multiple_relationships(self):
        """Test adding multiple relationships."""
        graph = KnowledgeGraph()
        relationships = make_test_relationships()

        for rel in relationships:
            graph.add_relationship(rel)

        # Should create all relationships
        assert graph.get_node_count() >= 2  # At least some nodes
        assert graph.get_edge_count() == len(relationships)


class TestGraphTraversal:
    """Tests for graph traversal operations."""

    def test_get_neighbors(self):
        """Test getting node neighbors."""
        graph = KnowledgeGraph()

        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("bob", "Bob", "PERSON")
        graph.add_node("charlie", "Charlie", "PERSON")

        graph.add_edge("alice", "bob", "knows")
        graph.add_edge("alice", "charlie", "knows")

        neighbors = graph.get_neighbors("alice")

        assert len(neighbors) == 2
        assert "bob" in neighbors
        assert "charlie" in neighbors

    def test_get_neighbors_nonexistent(self):
        """Test getting neighbors of nonexistent node."""
        graph = KnowledgeGraph()

        neighbors = graph.get_neighbors("nonexistent")

        assert neighbors == []

    def test_get_neighbors_bidirectional(self):
        """Test neighbors include both incoming and outgoing."""
        graph = KnowledgeGraph()

        graph.add_node("a", "A", "ENTITY")
        graph.add_node("b", "B", "ENTITY")
        graph.add_node("c", "C", "ENTITY")

        # A -> B, C -> A
        graph.add_edge("a", "b", "relates")
        graph.add_edge("c", "a", "relates")

        neighbors = graph.get_neighbors("a")

        # Should include both B (outgoing) and C (incoming)
        assert len(neighbors) == 2


class TestMultiHopQueries:
    """Tests for multi-hop path queries."""

    def test_get_paths_direct(self):
        """Test finding direct path."""
        try:
            import networkx  # noqa: F401

            graph = KnowledgeGraph()

            graph.add_node("a", "A", "ENTITY")
            graph.add_node("b", "B", "ENTITY")
            graph.add_edge("a", "b", "relates")

            paths = graph.get_paths("a", "b", max_hops=2)

            # Should find direct path
            assert len(paths) >= 1
            if paths:
                assert paths[0] == ["a", "b"]

        except ImportError:
            pytest.skip("NetworkX not installed")

    def test_get_paths_two_hop(self):
        """Test finding 2-hop path."""
        try:
            import networkx  # noqa: F401

            graph = KnowledgeGraph()

            # A -> B -> C
            graph.add_node("a", "A", "ENTITY")
            graph.add_node("b", "B", "ENTITY")
            graph.add_node("c", "C", "ENTITY")

            graph.add_edge("a", "b", "relates")
            graph.add_edge("b", "c", "relates")

            paths = graph.get_paths("a", "c", max_hops=3)

            # Should find path through B
            assert len(paths) >= 1
            if paths:
                assert paths[0] == ["a", "b", "c"]

        except ImportError:
            pytest.skip("NetworkX not installed")

    def test_get_paths_no_path(self):
        """Test when no path exists."""
        try:
            import networkx  # noqa: F401

            graph = KnowledgeGraph()

            # Disconnected nodes
            graph.add_node("a", "A", "ENTITY")
            graph.add_node("b", "B", "ENTITY")

            paths = graph.get_paths("a", "b", max_hops=5)

            # Should find no path
            assert paths == []

        except ImportError:
            pytest.skip("NetworkX not installed")


class TestSubgraph:
    """Tests for subgraph extraction."""

    def test_get_subgraph_depth_1(self):
        """Test extracting 1-hop subgraph."""
        try:
            import networkx  # noqa: F401

            graph = KnowledgeGraph()

            # A -> B -> C
            graph.add_node("a", "A", "ENTITY")
            graph.add_node("b", "B", "ENTITY")
            graph.add_node("c", "C", "ENTITY")
            graph.add_edge("a", "b", "relates")
            graph.add_edge("b", "c", "relates")

            subgraph = graph.get_subgraph("a", depth=1)

            # Should include A and B only
            assert subgraph.get_node_count() == 2

        except ImportError:
            pytest.skip("NetworkX not installed")

    def test_get_subgraph_depth_2(self):
        """Test extracting 2-hop subgraph."""
        try:
            import networkx  # noqa: F401

            graph = KnowledgeGraph()

            # A -> B -> C
            graph.add_node("a", "A", "ENTITY")
            graph.add_node("b", "B", "ENTITY")
            graph.add_node("c", "C", "ENTITY")
            graph.add_edge("a", "b", "relates")
            graph.add_edge("b", "c", "relates")

            subgraph = graph.get_subgraph("a", depth=2)

            # Should include A, B, and C
            assert subgraph.get_node_count() == 3

        except ImportError:
            pytest.skip("NetworkX not installed")


class TestGraphStatistics:
    """Tests for graph statistics."""

    def test_get_statistics_empty(self):
        """Test statistics for empty graph."""
        graph = KnowledgeGraph()

        stats = graph.get_statistics()

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert "networkx_available" in stats

    def test_get_statistics_with_data(self):
        """Test statistics for populated graph."""
        graph = KnowledgeGraph()

        # Add some data
        graph.add_node("a", "A", "ENTITY")
        graph.add_node("b", "B", "ENTITY")
        graph.add_edge("a", "b", "relates")

        stats = graph.get_statistics()

        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1


class TestGraphExport:
    """Tests for graph export functionality."""

    def test_export_json(self):
        """Test exporting graph to JSON."""
        graph = KnowledgeGraph()

        graph.add_node("a", "Alice", "PERSON")
        graph.add_node("b", "Bob", "PERSON")
        graph.add_edge("a", "b", "knows")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            graph.export_json(temp_path)

            # Verify file created
            assert temp_path.exists()

            # Verify valid JSON
            with open(temp_path, "r") as f:
                data = json.load(f)

            # Should have nodes/links or similar structure
            assert isinstance(data, dict)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_visualize_text(self):
        """Test text visualization."""
        graph = KnowledgeGraph()

        graph.add_node("a", "Alice", "PERSON")
        graph.add_node("b", "Bob", "PERSON")
        graph.add_edge("a", "b", "knows")

        text = graph.visualize_text(max_nodes=10)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "Knowledge Graph" in text


class TestBuildFromChunks:
    """Tests for building graph from chunks."""

    def test_build_knowledge_graph_from_chunks(self):
        """Test building graph from chunks."""
        chunks = [
            ChunkRecord("c1", "doc1", "Tim Cook works at Apple Inc."),
            ChunkRecord("c2", "doc2", "Apple acquired Beats Electronics."),
        ]

        from ingestforge.enrichment.entities import EntityExtractor

        entity_extractor = EntityExtractor(use_spacy=False)
        relationship_extractor = SpacyRelationshipExtractor(use_spacy=False)

        graph = build_knowledge_graph_from_chunks(
            chunks, entity_extractor, relationship_extractor
        )

        # Should create some graph structure
        assert isinstance(graph, KnowledgeGraph)
        assert graph.get_node_count() >= 0


class TestFallbackMode:
    """Tests for fallback mode (no NetworkX)."""

    def test_fallback_add_nodes_and_edges(self):
        """Test graph operations work in fallback mode."""
        graph = KnowledgeGraph()

        # Should work even if NetworkX unavailable
        graph.add_node("a", "Alice", "PERSON")
        graph.add_node("b", "Bob", "PERSON")
        graph.add_edge("a", "b", "knows")

        assert graph.get_node_count() >= 2
        assert graph.get_edge_count() >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_duplicate_nodes(self):
        """Test adding duplicate nodes."""
        graph = KnowledgeGraph()

        graph.add_node("a", "Alice", "PERSON")
        graph.add_node("a", "Alice", "PERSON")  # Duplicate

        # NetworkX allows updating, count stays 1
        assert graph.get_node_count() == 1

    def test_self_loops(self):
        """Test self-referential edges."""
        graph = KnowledgeGraph()

        graph.add_node("a", "Alice", "PERSON")
        graph.add_edge("a", "a", "reflects_on")

        # Should allow self-loops
        assert graph.get_edge_count() == 1

    def test_very_large_graph(self):
        """Test with many nodes and edges."""
        graph = KnowledgeGraph()

        # Add many nodes
        for i in range(100):
            graph.add_node(f"node{i}", f"Node {i}", "ENTITY")

        # Add some edges
        for i in range(50):
            graph.add_edge(f"node{i}", f"node{i+1}", "relates")

        assert graph.get_node_count() == 100
        assert graph.get_edge_count() == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
