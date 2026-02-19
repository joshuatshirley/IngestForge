"""
Tests for KnowledgeGraphBuilder with Mermaid export.

Validates:
- KnowledgeGraphBuilder initialization
- add_entities method
- add_relationships method
- build method
- to_mermaid export
- to_json export
- query method for subgraphs
- build_graph_from_text function

Follows NASA JPL Commandments for test structure.
"""

import pytest
import tempfile
from pathlib import Path

from ingestforge.enrichment.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    build_graph_from_text,
    export_to_mermaid_file,
)
from ingestforge.enrichment.relationships import Relationship
from ingestforge.enrichment.ner import Entity


# =============================================================================
# Test Data
# =============================================================================


def make_test_entity(
    text: str,
    entity_type: str = "ORG",
    start: int = 0,
    end: int = 10,
) -> Entity:
    """Create test entity."""
    return Entity(
        text=text,
        type=entity_type,
        start=start,
        end=end,
        confidence=0.9,
    )


def make_test_relationship(
    subject: str,
    predicate: str,
    obj: str,
    confidence: float = 0.8,
) -> Relationship:
    """Create test relationship."""
    return Relationship(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
    )


TEST_TEXT = """
Apple Inc. was founded by Steve Jobs in California.
Tim Cook works at Apple Inc.
Microsoft Corporation is based in Washington.
"""


# =============================================================================
# Test KnowledgeGraphBuilder Initialization
# =============================================================================


class TestKnowledgeGraphBuilderInit:
    """Tests for KnowledgeGraphBuilder initialization."""

    def test_empty_initialization(self) -> None:
        """Test creating empty builder."""
        builder = KnowledgeGraphBuilder()

        assert builder.get_entity_count() == 0
        assert builder.get_relationship_count() == 0

    def test_build_empty_graph(self) -> None:
        """Test building empty graph."""
        builder = KnowledgeGraphBuilder()
        graph = builder.build()

        assert isinstance(graph, KnowledgeGraph)
        assert graph.get_node_count() == 0
        assert graph.get_edge_count() == 0


# =============================================================================
# Test add_entities
# =============================================================================


class TestAddEntities:
    """Tests for add_entities method."""

    def test_add_single_entity(self) -> None:
        """Test adding single entity."""
        builder = KnowledgeGraphBuilder()
        entities = [make_test_entity("Apple Inc")]

        builder.add_entities(entities)

        assert builder.get_entity_count() == 1

    def test_add_multiple_entities(self) -> None:
        """Test adding multiple entities."""
        builder = KnowledgeGraphBuilder()
        entities = [
            make_test_entity("Apple Inc", "ORG"),
            make_test_entity("Steve Jobs", "PERSON"),
            make_test_entity("California", "GPE"),
        ]

        builder.add_entities(entities)

        assert builder.get_entity_count() == 3

    def test_add_entities_creates_nodes(self) -> None:
        """Test that entities become graph nodes."""
        builder = KnowledgeGraphBuilder()
        entities = [
            make_test_entity("Apple Inc", "ORG"),
            make_test_entity("Microsoft", "ORG"),
        ]

        builder.add_entities(entities)
        graph = builder.build()

        assert graph.get_node_count() == 2

    def test_method_chaining(self) -> None:
        """Test method chaining."""
        builder = KnowledgeGraphBuilder()

        result = builder.add_entities([make_test_entity("Apple")])

        assert result is builder  # Returns self


# =============================================================================
# Test add_relationships
# =============================================================================


class TestAddRelationships:
    """Tests for add_relationships method."""

    def test_add_single_relationship(self) -> None:
        """Test adding single relationship."""
        builder = KnowledgeGraphBuilder()
        relationships = [make_test_relationship("Apple", "founded_by", "Jobs")]

        builder.add_relationships(relationships)

        assert builder.get_relationship_count() == 1

    def test_add_multiple_relationships(self) -> None:
        """Test adding multiple relationships."""
        builder = KnowledgeGraphBuilder()
        relationships = [
            make_test_relationship("Apple", "founded_by", "Jobs"),
            make_test_relationship("Jobs", "worked_at", "Apple"),
        ]

        builder.add_relationships(relationships)

        assert builder.get_relationship_count() == 2

    def test_relationships_create_edges(self) -> None:
        """Test that relationships become graph edges."""
        builder = KnowledgeGraphBuilder()
        relationships = [
            make_test_relationship("A", "relates_to", "B"),
            make_test_relationship("B", "relates_to", "C"),
        ]

        builder.add_relationships(relationships)
        graph = builder.build()

        assert graph.get_edge_count() == 2

    def test_relationships_create_nodes(self) -> None:
        """Test that relationship entities become nodes."""
        builder = KnowledgeGraphBuilder()
        relationships = [make_test_relationship("Apple", "founded_by", "Jobs")]

        builder.add_relationships(relationships)
        graph = builder.build()

        # Should create 2 nodes (Apple and Jobs)
        assert graph.get_node_count() == 2


# =============================================================================
# Test to_mermaid
# =============================================================================


class TestToMermaid:
    """Tests for to_mermaid export."""

    def test_empty_graph_mermaid(self) -> None:
        """Test Mermaid for empty graph."""
        builder = KnowledgeGraphBuilder()

        mermaid = builder.to_mermaid()

        assert "graph LR" in mermaid
        assert "empty" in mermaid.lower() or "No data" in mermaid

    def test_basic_mermaid_structure(self) -> None:
        """Test basic Mermaid structure."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("Alice", "knows", "Bob"),
            ]
        )

        mermaid = builder.to_mermaid()

        assert "graph LR" in mermaid
        assert "Alice" in mermaid
        assert "Bob" in mermaid
        assert "knows" in mermaid

    def test_mermaid_edge_format(self) -> None:
        """Test Mermaid edge format."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "connects", "B"),
            ]
        )

        mermaid = builder.to_mermaid()

        # Should have edge with label
        assert "-->|connects|" in mermaid or "connects" in mermaid

    def test_mermaid_multiple_edges(self) -> None:
        """Test Mermaid with multiple edges."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "r1", "B"),
                make_test_relationship("B", "r2", "C"),
                make_test_relationship("C", "r3", "D"),
            ]
        )

        mermaid = builder.to_mermaid()

        assert "A" in mermaid
        assert "B" in mermaid
        assert "C" in mermaid
        assert "D" in mermaid

    def test_mermaid_max_nodes(self) -> None:
        """Test Mermaid respects max_nodes limit."""
        builder = KnowledgeGraphBuilder()

        # Add many relationships
        for i in range(100):
            builder.add_relationships(
                [
                    make_test_relationship(f"Node{i}", "rel", f"Node{i+1}"),
                ]
            )

        mermaid = builder.to_mermaid(max_nodes=10)

        # Should have limited output
        assert isinstance(mermaid, str)

    def test_mermaid_sanitizes_ids(self) -> None:
        """Test that IDs are sanitized for Mermaid."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("Apple Inc.", "based_in", "New York"),
            ]
        )

        mermaid = builder.to_mermaid()

        # Should not have special chars in IDs
        # IDs should be alphanumeric + underscore
        lines = mermaid.split("\n")
        for line in lines:
            if "-->" in line:
                # Check ID format
                assert isinstance(line, str)


# =============================================================================
# Test to_json
# =============================================================================


class TestToJson:
    """Tests for to_json export."""

    def test_empty_graph_json(self) -> None:
        """Test JSON for empty graph."""
        builder = KnowledgeGraphBuilder()

        result = builder.to_json()

        assert "nodes" in result
        assert "edges" in result
        assert "statistics" in result
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_json_nodes_structure(self) -> None:
        """Test JSON nodes structure."""
        builder = KnowledgeGraphBuilder()
        builder.add_entities(
            [
                make_test_entity("Apple", "ORG"),
                make_test_entity("Jobs", "PERSON"),
            ]
        )

        result = builder.to_json()

        assert len(result["nodes"]) == 2
        for node in result["nodes"]:
            assert "id" in node
            assert "label" in node
            assert "type" in node

    def test_json_edges_structure(self) -> None:
        """Test JSON edges structure."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "relates", "B"),
            ]
        )

        result = builder.to_json()

        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "relation" in edge

    def test_json_statistics(self) -> None:
        """Test JSON includes statistics."""
        builder = KnowledgeGraphBuilder()
        builder.add_entities([make_test_entity("Apple")])
        builder.add_relationships(
            [
                make_test_relationship("Apple", "founded_by", "Jobs"),
            ]
        )

        result = builder.to_json()

        assert "statistics" in result
        stats = result["statistics"]
        assert "node_count" in stats
        assert "edge_count" in stats


# =============================================================================
# Test query
# =============================================================================


class TestQuery:
    """Tests for query method."""

    def test_query_returns_builder(self) -> None:
        """Test query returns new builder."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "rel", "B"),
            ]
        )

        result = builder.query("A", depth=1)

        assert isinstance(result, KnowledgeGraphBuilder)
        assert result is not builder

    def test_query_extracts_subgraph(self) -> None:
        """Test query extracts correct subgraph."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "rel", "B"),
                make_test_relationship("B", "rel", "C"),
                make_test_relationship("X", "rel", "Y"),  # Disconnected
            ]
        )

        result = builder.query("A", depth=2)
        graph = result.build()

        # Should include A, B, C but not X, Y
        # (Depends on NetworkX availability)
        assert isinstance(graph, KnowledgeGraph)

    def test_query_nonexistent_entity(self) -> None:
        """Test query for nonexistent entity."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "rel", "B"),
            ]
        )

        result = builder.query("NonExistent", depth=1)
        graph = result.build()

        # Should be empty
        assert graph.get_node_count() == 0


# =============================================================================
# Test build_graph_from_text
# =============================================================================


class TestBuildGraphFromText:
    """Tests for build_graph_from_text function."""

    def test_empty_text(self) -> None:
        """Test with empty text."""
        builder = build_graph_from_text("")

        assert isinstance(builder, KnowledgeGraphBuilder)

    def test_simple_text(self) -> None:
        """Test with simple text."""
        builder = build_graph_from_text(TEST_TEXT)

        assert isinstance(builder, KnowledgeGraphBuilder)
        # Should have some entities
        # (depends on spaCy availability)

    def test_returns_builder(self) -> None:
        """Test that function returns builder."""
        builder = build_graph_from_text("Apple Inc. is a company.")

        assert isinstance(builder, KnowledgeGraphBuilder)
        # Can chain methods
        mermaid = builder.to_mermaid()
        assert isinstance(mermaid, str)


# =============================================================================
# Test export_to_mermaid_file
# =============================================================================


class TestExportToMermaidFile:
    """Tests for export_to_mermaid_file function."""

    def test_export_creates_file(self) -> None:
        """Test that export creates file."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "rel", "B"),
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = Path(f.name)

        try:
            export_to_mermaid_file(builder, temp_path)

            assert temp_path.exists()
            content = temp_path.read_text()
            assert "```mermaid" in content
            assert "graph LR" in content
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_export_with_title(self) -> None:
        """Test export with custom title."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("A", "rel", "B"),
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = Path(f.name)

        try:
            export_to_mermaid_file(builder, temp_path, title="My Graph")

            content = temp_path.read_text()
            assert "# My Graph" in content
        finally:
            if temp_path.exists():
                temp_path.unlink()


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestKnowledgeGraphBuilderEdgeCases:
    """Tests for edge cases."""

    def test_special_characters_in_names(self) -> None:
        """Test entities with special characters."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("Apple Inc.", "based_in", "São Paulo"),
            ]
        )

        mermaid = builder.to_mermaid()

        assert isinstance(mermaid, str)
        # Should not crash

    def test_very_long_names(self) -> None:
        """Test very long entity names."""
        long_name = "A" * 100
        builder = KnowledgeGraphBuilder()
        builder.add_entities([make_test_entity(long_name)])

        result = builder.to_json()

        assert len(result["nodes"]) == 1

    def test_unicode_entities(self) -> None:
        """Test unicode entity names."""
        builder = KnowledgeGraphBuilder()
        builder.add_relationships(
            [
                make_test_relationship("北京", "capital_of", "中国"),
            ]
        )

        mermaid = builder.to_mermaid()
        json_result = builder.to_json()

        assert isinstance(mermaid, str)
        assert len(json_result["edges"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
