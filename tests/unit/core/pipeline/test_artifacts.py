import pytest
from pathlib import Path
from pydantic import ValidationError
from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFChunkArtifact,
    IFFailureArtifact,
)


def test_if_file_artifact_validation():
    """
    GWT:
    Given a FileArtifact
    When initialized with valid data
    Then it must store the path and mime_type.
    """
    path = Path("/tmp/test.pdf").absolute()
    art = IFFileArtifact(
        artifact_id="file-1", file_path=path, mime_type="application/pdf"
    )
    assert art.file_path == path
    assert art.mime_type == "application/pdf"
    assert art.artifact_id == "file-1"


def test_if_text_artifact_validation():
    """
    GWT:
    Given a TextArtifact
    When initialized
    Then it must store the content string.
    """
    art = IFTextArtifact(artifact_id="text-1", content="Hello World")
    assert art.content == "Hello World"


def test_if_chunk_artifact_validation():
    """
    GWT:
    Given a ChunkArtifact
    When initialized
    Then it must store document identity and position.
    """
    art = IFChunkArtifact(
        artifact_id="chunk-1",
        document_id="doc-123",
        content="some content",
        chunk_index=5,
        total_chunks=10,
    )
    assert art.document_id == "doc-123"
    assert art.chunk_index == 5
    assert art.total_chunks == 10


def test_if_failure_artifact_validation():
    """
    GWT:
    Given a FailureArtifact
    When initialized
    Then it must store error details.
    """
    art = IFFailureArtifact(
        artifact_id="fail-1",
        error_message="Something went wrong",
        failed_processor_id="proc-xyz",
    )
    assert art.error_message == "Something went wrong"
    assert art.failed_processor_id == "proc-xyz"


def test_artifact_immutability_concrete():
    """
    Verify immutability across all concrete models.
    """
    art = IFTextArtifact(artifact_id="t1", content="immutable")
    with pytest.raises(ValidationError):
        art.content = "mutable?"


# Lineage - Deterministic Tracking Tests


def test_root_artifact_has_correct_lineage_defaults():
    """
    GWT:
    Given a new artifact without derivation
    When initialized
    Then it must have lineage_depth=0, no parent_id, no root_artifact_id.
    """
    art = IFTextArtifact(artifact_id="root-1", content="root content")
    assert art.lineage_depth == 0
    assert art.parent_id is None
    assert art.root_artifact_id is None
    assert art.is_root is True
    assert art.effective_root_id == "root-1"


def test_derive_sets_lineage_fields():
    """
    GWT:
    Given a root artifact
    When derive() is called
    Then the child must have parent_id, root_artifact_id, and lineage_depth=1.
    """
    root = IFTextArtifact(artifact_id="root-1", content="root content")
    child = root.derive("processor-a", artifact_id="child-1", content="child content")

    assert child.parent_id == "root-1"
    assert child.root_artifact_id == "root-1"
    assert child.lineage_depth == 1
    assert child.provenance == ["processor-a"]
    assert child.is_root is False
    assert child.effective_root_id == "root-1"


def test_multi_level_lineage_tracking():
    """
    GWT:
    Given a chain of derived artifacts
    When traversing the lineage
    Then root_artifact_id remains consistent and depth increments correctly.
    """
    root = IFTextArtifact(artifact_id="root", content="level 0")
    child1 = root.derive("proc-1", artifact_id="child-1", content="level 1")
    child2 = child1.derive("proc-2", artifact_id="child-2", content="level 2")
    child3 = child2.derive("proc-3", artifact_id="child-3", content="level 3")

    # Verify chain
    assert child3.lineage_depth == 3
    assert child3.root_artifact_id == "root"
    assert child3.parent_id == "child-2"
    assert child3.provenance == ["proc-1", "proc-2", "proc-3"]

    # All children point to same root
    assert child1.root_artifact_id == "root"
    assert child2.root_artifact_id == "root"
    assert child3.root_artifact_id == "root"


def test_lineage_consistency_validation_root():
    """
    GWT:
    Given a root artifact
    When validate_lineage_consistency() is called
    Then it must return True.
    """
    root = IFTextArtifact(artifact_id="root", content="root")
    assert root.validate_lineage_consistency() is True


def test_lineage_consistency_validation_derived():
    """
    GWT:
    Given a properly derived artifact
    When validate_lineage_consistency() is called
    Then it must return True.
    """
    root = IFTextArtifact(artifact_id="root", content="root")
    child = root.derive("proc-1", artifact_id="child", content="child")
    assert child.validate_lineage_consistency() is True


def test_chunk_artifact_lineage():
    """
    GWT:
    Given a ChunkArtifact
    When derived
    Then lineage fields are correctly propagated.
    """
    chunk1 = IFChunkArtifact(
        artifact_id="chunk-1",
        document_id="doc-1",
        content="chunk content",
        chunk_index=0,
        total_chunks=1,
    )
    chunk2 = chunk1.derive("enricher-1", artifact_id="chunk-2", content="enriched")

    assert chunk2.root_artifact_id == "chunk-1"
    assert chunk2.lineage_depth == 1
    assert chunk2.parent_id == "chunk-1"


def test_failure_artifact_lineage():
    """
    GWT:
    Given a FailureArtifact derived from another artifact
    When created
    Then lineage is preserved.
    """
    root = IFTextArtifact(artifact_id="source", content="source")
    failure = root.derive(
        "broken-proc",
        artifact_id="fail-1",
    )
    # Note: derive returns same type, but in practice failures are created differently
    assert failure.root_artifact_id == "source"
    assert failure.lineage_depth == 1


def test_file_artifact_lineage():
    """
    GWT:
    Given a FileArtifact
    When derived
    Then lineage fields are correctly set.
    """
    path = Path("/tmp/test.txt").absolute()
    file_art = IFFileArtifact(artifact_id="file-1", file_path=path)
    derived = file_art.derive("extractor", artifact_id="derived-1")

    assert derived.root_artifact_id == "file-1"
    assert derived.lineage_depth == 1
    assert derived.parent_id == "file-1"


# === Serialization - Validated Metadata Tests ===


class TestMetadataValidation:
    """Tests for metadata validation ()."""

    def test_metadata_accepts_valid_dict(self):
        """
        GWT:
        Given valid metadata with JSON-serializable values,
        When artifact is created,
        Then it accepts the metadata.
        """
        art = IFTextArtifact(
            artifact_id="meta-1",
            content="test",
            metadata={"key1": "value1", "count": 42, "nested": {"a": 1}},
        )
        assert art.metadata["key1"] == "value1"
        assert art.metadata["count"] == 42
        assert art.metadata["nested"]["a"] == 1

    def test_metadata_rejects_too_many_keys(self):
        """
        GWT:
        Given metadata with more than 128 keys,
        When artifact is created,
        Then it raises ValidationError.
        """
        too_many_keys = {f"key_{i}": i for i in range(129)}
        with pytest.raises(ValidationError) as exc_info:
            IFTextArtifact(
                artifact_id="meta-fail", content="test", metadata=too_many_keys
            )
        assert "128 keys" in str(exc_info.value)

    def test_metadata_accepts_exactly_128_keys(self):
        """
        GWT:
        Given metadata with exactly 128 keys,
        When artifact is created,
        Then it is accepted.
        """
        max_keys = {f"key_{i}": i for i in range(128)}
        art = IFTextArtifact(artifact_id="meta-max", content="test", metadata=max_keys)
        assert len(art.metadata) == 128

    def test_metadata_rejects_non_serializable_values(self):
        """
        GWT:
        Given metadata with non-JSON-serializable value,
        When artifact is created,
        Then it raises ValidationError.
        """
        with pytest.raises(ValidationError) as exc_info:
            IFTextArtifact(
                artifact_id="meta-bad",
                content="test",
                metadata={"func": lambda x: x},  # Functions not serializable
            )
        assert "not JSON-serializable" in str(exc_info.value)

    def test_metadata_rejects_oversized_value(self):
        """
        GWT:
        Given metadata with value exceeding 64KB,
        When artifact is created,
        Then it raises ValidationError.
        """
        # Create a string larger than 64KB
        large_value = "x" * 70000
        with pytest.raises(ValidationError) as exc_info:
            IFTextArtifact(
                artifact_id="meta-large", content="test", metadata={"big": large_value}
            )
        assert "exceeds" in str(exc_info.value)


class TestSerializationRoundTrip:
    """Tests for JSON serialization/deserialization ()."""

    def test_text_artifact_serialization_roundtrip(self):
        """
        GWT:
        Given a TextArtifact with metadata,
        When serialized to JSON and deserialized,
        Then all fields are preserved.
        """
        original = IFTextArtifact(
            artifact_id="ser-1",
            content="Hello World",
            metadata={"source": "test", "count": 42},
            schema_version="1.0.0",
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        restored = IFTextArtifact.model_validate_json(json_str)

        assert restored.artifact_id == original.artifact_id
        assert restored.content == original.content
        assert restored.metadata == original.metadata
        assert restored.content_hash == original.content_hash
        assert restored.schema_version == original.schema_version

    def test_chunk_artifact_serialization_roundtrip(self):
        """
        GWT:
        Given a ChunkArtifact with lineage,
        When serialized and deserialized,
        Then lineage fields are preserved.
        """
        root = IFChunkArtifact(
            artifact_id="chunk-root",
            document_id="doc-1",
            content="Root content",
            chunk_index=0,
            total_chunks=3,
        )
        derived = root.derive("proc-1", artifact_id="chunk-derived", content="Derived")

        # Serialize and restore
        json_str = derived.model_dump_json()
        restored = IFChunkArtifact.model_validate_json(json_str)

        assert restored.parent_id == "chunk-root"
        assert restored.root_artifact_id == "chunk-root"
        assert restored.lineage_depth == 1
        assert restored.provenance == ["proc-1"]
        assert restored.document_id == "doc-1"

    def test_file_artifact_serialization_roundtrip(self):
        """
        GWT:
        Given a FileArtifact,
        When serialized and deserialized,
        Then path is correctly restored.
        """
        path = Path("/tmp/test.pdf").absolute()
        original = IFFileArtifact(
            artifact_id="file-ser",
            file_path=path,
            mime_type="application/pdf",
            metadata={"pages": 10},
        )

        json_str = original.model_dump_json()
        restored = IFFileArtifact.model_validate_json(json_str)

        assert restored.file_path == path
        assert restored.mime_type == "application/pdf"
        assert restored.metadata == {"pages": 10}

    def test_failure_artifact_serialization_roundtrip(self):
        """
        GWT:
        Given a FailureArtifact with error details,
        When serialized and deserialized,
        Then error info is preserved.
        """
        original = IFFailureArtifact(
            artifact_id="fail-ser",
            error_message="Test error",
            stack_trace="Traceback...",
            failed_processor_id="broken-proc",
        )

        json_str = original.model_dump_json()
        restored = IFFailureArtifact.model_validate_json(json_str)

        assert restored.error_message == "Test error"
        assert restored.stack_trace == "Traceback..."
        assert restored.failed_processor_id == "broken-proc"

    def test_nested_metadata_serialization(self):
        """
        GWT:
        Given an artifact with deeply nested metadata,
        When serialized and deserialized,
        Then nested structure is preserved.
        """
        nested_meta = {
            "level1": {"level2": {"level3": {"value": [1, 2, 3]}}},
            "list": [{"a": 1}, {"b": 2}],
        }
        original = IFTextArtifact(
            artifact_id="nested-ser", content="test", metadata=nested_meta
        )

        json_str = original.model_dump_json()
        restored = IFTextArtifact.model_validate_json(json_str)

        assert restored.metadata == nested_meta
        assert restored.metadata["level1"]["level2"]["level3"]["value"] == [1, 2, 3]


# === Entity Artifact Model Tests ===

from ingestforge.core.pipeline.artifacts import (
    SourceProvenance,
    EntityNode,
    RelationshipEdge,
    IFEntityArtifact,
)


class TestSourceProvenance:
    """Tests for SourceProvenance model ()."""

    def test_source_provenance_creation(self):
        """
        GWT:
        Given valid provenance data,
        When SourceProvenance is created,
        Then it stores all fields correctly.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-123",
            char_offset_start=10,
            char_offset_end=25,
            confidence=0.95,
            extraction_method="spacy",
        )
        assert prov.source_artifact_id == "chunk-123"
        assert prov.char_offset_start == 10
        assert prov.char_offset_end == 25
        assert prov.confidence == 0.95
        assert prov.extraction_method == "spacy"

    def test_source_provenance_default_values(self):
        """
        GWT:
        Given minimal provenance data,
        When SourceProvenance is created,
        Then defaults are applied.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        assert prov.confidence == 1.0
        assert prov.extraction_method == "unknown"

    def test_source_provenance_is_frozen(self):
        """
        GWT:
        Given a SourceProvenance instance,
        When attempting to modify it,
        Then it raises ValidationError.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        with pytest.raises(ValidationError):
            prov.source_artifact_id = "modified"


class TestEntityNode:
    """Tests for EntityNode model ()."""

    def test_entity_node_creation(self):
        """
        GWT:
        Given valid entity data,
        When EntityNode is created,
        Then it stores all fields correctly.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        node = EntityNode(
            entity_id="ent-1",
            entity_type="PERSON",
            name="John Doe",
            aliases=["J. Doe", "JD"],
            source_provenance=prov,
            properties={"age": 30},
        )
        assert node.entity_id == "ent-1"
        assert node.entity_type == "PERSON"
        assert node.name == "John Doe"
        assert node.aliases == ["J. Doe", "JD"]
        assert node.properties == {"age": 30}

    def test_entity_node_repr(self):
        """
        GWT:
        Given an EntityNode,
        When repr is called,
        Then it shows type and name.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        node = EntityNode(
            entity_id="ent-1",
            entity_type="ORG",
            name="Acme Inc",
            source_provenance=prov,
        )
        assert "ORG" in repr(node)
        assert "Acme Inc" in repr(node)

    def test_entity_node_is_frozen(self):
        """
        GWT:
        Given an EntityNode instance,
        When attempting to modify it,
        Then it raises ValidationError.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        node = EntityNode(
            entity_id="ent-1", entity_type="PERSON", name="John", source_provenance=prov
        )
        with pytest.raises(ValidationError):
            node.name = "Jane"


class TestRelationshipEdge:
    """Tests for RelationshipEdge model ()."""

    def test_relationship_edge_creation(self):
        """
        GWT:
        Given valid relationship data,
        When RelationshipEdge is created,
        Then it stores all fields correctly.
        """
        edge = RelationshipEdge(
            source_entity_id="ent-1",
            target_entity_id="ent-2",
            predicate="works_for",
            confidence=0.85,
            properties={"start_date": "2020-01-01"},
        )
        assert edge.source_entity_id == "ent-1"
        assert edge.target_entity_id == "ent-2"
        assert edge.predicate == "works_for"
        assert edge.confidence == 0.85
        assert edge.properties["start_date"] == "2020-01-01"

    def test_relationship_edge_repr(self):
        """
        GWT:
        Given a RelationshipEdge,
        When repr is called,
        Then it shows the relationship triple.
        """
        edge = RelationshipEdge(
            source_entity_id="person-1",
            target_entity_id="org-1",
            predicate="employed_by",
        )
        assert "person-1" in repr(edge)
        assert "org-1" in repr(edge)
        assert "employed_by" in repr(edge)

    def test_relationship_edge_optional_provenance(self):
        """
        GWT:
        Given a RelationshipEdge without provenance,
        When created,
        Then source_provenance is None.
        """
        edge = RelationshipEdge(
            source_entity_id="ent-1", target_entity_id="ent-2", predicate="related_to"
        )
        assert edge.source_provenance is None


class TestIFEntityArtifact:
    """Tests for IFEntityArtifact model ()."""

    def test_entity_artifact_creation(self):
        """
        GWT:
        Given entity nodes and relationship edges,
        When IFEntityArtifact is created,
        Then it stores all data correctly.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        node = EntityNode(
            entity_id="ent-1", entity_type="PERSON", name="John", source_provenance=prov
        )
        edge = RelationshipEdge(
            source_entity_id="ent-1", target_entity_id="ent-2", predicate="knows"
        )
        artifact = IFEntityArtifact(
            artifact_id="entity-art-1",
            nodes=[node],
            edges=[edge],
            extraction_model="spacy-en_core_web_lg",
            source_document_id="doc-123",
        )
        assert artifact.node_count == 1
        assert artifact.edge_count == 1
        assert artifact.extraction_model == "spacy-en_core_web_lg"
        assert artifact.source_document_id == "doc-123"

    def test_entity_artifact_repr(self):
        """
        GWT:
        Given an IFEntityArtifact,
        When repr is called,
        Then it shows node and edge counts.
        """
        artifact = IFEntityArtifact(artifact_id="ent-art-1", nodes=[], edges=[])
        repr_str = repr(artifact)
        assert "nodes=0" in repr_str
        assert "edges=0" in repr_str

    def test_entity_artifact_derive(self):
        """
        GWT:
        Given an IFEntityArtifact,
        When derive is called,
        Then lineage is properly set.
        """
        artifact = IFEntityArtifact(artifact_id="ent-art-1", nodes=[], edges=[])
        derived = artifact.derive("entity-enricher", artifact_id="ent-art-2")
        assert derived.parent_id == "ent-art-1"
        assert derived.root_artifact_id == "ent-art-1"
        assert derived.lineage_depth == 1
        assert derived.provenance == ["entity-enricher"]

    def test_entity_artifact_get_node_by_id(self):
        """
        GWT:
        Given an IFEntityArtifact with nodes,
        When get_node_by_id is called,
        Then the correct node is returned.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        node1 = EntityNode(
            entity_id="ent-1", entity_type="PERSON", name="John", source_provenance=prov
        )
        node2 = EntityNode(
            entity_id="ent-2", entity_type="ORG", name="Acme", source_provenance=prov
        )
        artifact = IFEntityArtifact(
            artifact_id="ent-art-1", nodes=[node1, node2], edges=[]
        )
        found = artifact.get_node_by_id("ent-2")
        assert found is not None
        assert found.name == "Acme"
        assert artifact.get_node_by_id("nonexistent") is None

    def test_entity_artifact_get_nodes_by_type(self):
        """
        GWT:
        Given an IFEntityArtifact with multiple nodes,
        When get_nodes_by_type is called,
        Then only nodes of that type are returned.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        nodes = [
            EntityNode(
                entity_id="p1",
                entity_type="PERSON",
                name="John",
                source_provenance=prov,
            ),
            EntityNode(
                entity_id="p2",
                entity_type="PERSON",
                name="Jane",
                source_provenance=prov,
            ),
            EntityNode(
                entity_id="o1", entity_type="ORG", name="Acme", source_provenance=prov
            ),
        ]
        artifact = IFEntityArtifact(artifact_id="ent-art-1", nodes=nodes, edges=[])
        people = artifact.get_nodes_by_type("PERSON")
        assert len(people) == 2
        assert all(n.entity_type == "PERSON" for n in people)

    def test_entity_artifact_get_edges_for_node(self):
        """
        GWT:
        Given an IFEntityArtifact with edges,
        When get_edges_for_node is called,
        Then edges involving that node are returned.
        """
        edges = [
            RelationshipEdge(
                source_entity_id="ent-1", target_entity_id="ent-2", predicate="knows"
            ),
            RelationshipEdge(
                source_entity_id="ent-2",
                target_entity_id="ent-3",
                predicate="works_with",
            ),
            RelationshipEdge(
                source_entity_id="ent-1", target_entity_id="ent-3", predicate="manages"
            ),
        ]
        artifact = IFEntityArtifact(artifact_id="ent-art-1", nodes=[], edges=edges)
        # ent-1 is source in 2 edges
        ent1_edges = artifact.get_edges_for_node("ent-1")
        assert len(ent1_edges) == 2

        # ent-2 is source in 1 and target in 1
        ent2_edges = artifact.get_edges_for_node("ent-2")
        assert len(ent2_edges) == 2

    def test_entity_artifact_serialization_roundtrip(self):
        """
        GWT:
        Given an IFEntityArtifact with nodes and edges,
        When serialized and deserialized,
        Then all data is preserved.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1",
            char_offset_start=0,
            char_offset_end=10,
            confidence=0.95,
            extraction_method="llm",
        )
        node = EntityNode(
            entity_id="ent-1",
            entity_type="PERSON",
            name="John Doe",
            aliases=["JD"],
            source_provenance=prov,
            properties={"role": "engineer"},
        )
        edge = RelationshipEdge(
            source_entity_id="ent-1",
            target_entity_id="ent-2",
            predicate="works_for",
            confidence=0.8,
            source_provenance=prov,
            properties={"since": "2020"},
        )
        original = IFEntityArtifact(
            artifact_id="ent-art-1",
            nodes=[node],
            edges=[edge],
            extraction_model="gpt-4",
            source_document_id="doc-1",
            metadata={"processing_time": 1.5},
        )

        json_str = original.model_dump_json()
        restored = IFEntityArtifact.model_validate_json(json_str)

        assert restored.artifact_id == original.artifact_id
        assert restored.node_count == 1
        assert restored.edge_count == 1
        assert restored.nodes[0].name == "John Doe"
        assert restored.nodes[0].source_provenance.confidence == 0.95
        assert restored.edges[0].predicate == "works_for"
        assert restored.extraction_model == "gpt-4"
        assert restored.metadata == {"processing_time": 1.5}


# === Chain-of-Custody Integrity Tests ===


class TestChainOfCustodyIntegrity:
    """Tests for chain-of-custody integrity via parent_hash."""

    def test_entity_node_with_parent_hash(self):
        """
        GWT:
        Given an EntityNode with parent_hash,
        When created,
        Then the hash is stored.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        node = EntityNode(
            entity_id="ent-1",
            entity_type="PERSON",
            name="John Doe",
            source_provenance=prov,
            parent_hash="abc123def456",
        )
        assert node.parent_hash == "abc123def456"

    def test_entity_node_parent_hash_optional(self):
        """
        GWT:
        Given an EntityNode without parent_hash,
        When created,
        Then parent_hash is None.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        node = EntityNode(
            entity_id="ent-1", entity_type="PERSON", name="John", source_provenance=prov
        )
        assert node.parent_hash is None

    def test_verify_parent_integrity_matching_hash(self):
        """
        GWT:
        Given an EntityNode with parent_hash matching artifact,
        When verify_parent_integrity is called,
        Then it returns True.
        """
        artifact = IFTextArtifact(
            artifact_id="text-1", content="John Doe works at Acme"
        )
        prov = SourceProvenance(
            source_artifact_id=artifact.artifact_id,
            char_offset_start=0,
            char_offset_end=8,
        )
        node = EntityNode(
            entity_id="ent-1",
            entity_type="PERSON",
            name="John Doe",
            source_provenance=prov,
            parent_hash=artifact.content_hash,
        )

        assert node.verify_parent_integrity(artifact) is True

    def test_verify_parent_integrity_mismatched_hash(self):
        """
        GWT:
        Given an EntityNode with parent_hash not matching artifact,
        When verify_parent_integrity is called,
        Then it returns False.
        """
        original = IFTextArtifact(
            artifact_id="text-1", content="John Doe works at Acme"
        )
        prov = SourceProvenance(
            source_artifact_id=original.artifact_id,
            char_offset_start=0,
            char_offset_end=8,
        )
        node = EntityNode(
            entity_id="ent-1",
            entity_type="PERSON",
            name="John Doe",
            source_provenance=prov,
            parent_hash=original.content_hash,
        )

        # Modified artifact with different content
        modified = IFTextArtifact(
            artifact_id="text-1",
            content="Jane Doe works at Acme",  # Changed name
        )

        assert node.verify_parent_integrity(modified) is False

    def test_verify_parent_integrity_no_hash_set(self):
        """
        GWT:
        Given an EntityNode without parent_hash,
        When verify_parent_integrity is called,
        Then it returns True (no hash to verify).
        """
        artifact = IFTextArtifact(artifact_id="text-1", content="Some content")
        prov = SourceProvenance(
            source_artifact_id=artifact.artifact_id,
            char_offset_start=0,
            char_offset_end=5,
        )
        node = EntityNode(
            entity_id="ent-1",
            entity_type="PERSON",
            name="John",
            source_provenance=prov,
            parent_hash=None,  # No hash set
        )

        assert node.verify_parent_integrity(artifact) is True

    def test_chain_of_custody_preserved_in_serialization(self):
        """
        GWT:
        Given an EntityNode with parent_hash,
        When serialized and deserialized,
        Then parent_hash is preserved.
        """
        prov = SourceProvenance(
            source_artifact_id="chunk-1", char_offset_start=0, char_offset_end=10
        )
        original = EntityNode(
            entity_id="ent-1",
            entity_type="PERSON",
            name="John Doe",
            source_provenance=prov,
            parent_hash="sha256_hash_value_here",
        )

        json_str = original.model_dump_json()
        restored = EntityNode.model_validate_json(json_str)

        assert restored.parent_hash == original.parent_hash

    def test_entity_creation_fails_with_mismatched_hash(self):
        """
        GWT:
        Given an entity created with a hash from artifact A,
        When verified against artifact B (different content),
        Then verification fails (chain-of-custody broken).

        This test demonstrates the integrity check that should be
        performed at extraction time.
        """
        # Source artifact
        source = IFTextArtifact(
            artifact_id="source-1", content="The CEO John Smith announced..."
        )

        # Create entity with source hash
        prov = SourceProvenance(
            source_artifact_id=source.artifact_id,
            char_offset_start=8,
            char_offset_end=18,
        )
        entity = EntityNode(
            entity_id="ent-ceo",
            entity_type="PERSON",
            name="John Smith",
            source_provenance=prov,
            parent_hash=source.content_hash,
        )

        # Simulate tampered content
        tampered = IFTextArtifact(
            artifact_id="source-1",
            content="The CEO Jane Smith announced...",  # Name changed
        )

        # Verification should detect tampering
        assert entity.verify_parent_integrity(source) is True
        assert entity.verify_parent_integrity(tampered) is False
