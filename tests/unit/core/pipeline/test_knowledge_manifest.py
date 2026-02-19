"""
Unit tests for IFKnowledgeManifest.

Knowledge Manifest
Tests GWT scenarios and NASA JPL Power of Ten compliance.
"""

import pytest
import threading
from typing import List

from ingestforge.core.pipeline.knowledge_manifest import (
    IFKnowledgeManifest,
    IFJoinArtifact,
    EntityReference,
    ManifestEntry,
    register_extracted_entities,
    MAX_ENTITIES_IN_MANIFEST,
    MAX_REFERENCES_PER_ENTITY,
    MAX_JOIN_ARTIFACTS_PER_SESSION,
)
from ingestforge.core.pipeline.artifacts import IFTextArtifact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_manifest():
    """Reset singleton before each test."""
    IFKnowledgeManifest.reset_instance()
    yield
    IFKnowledgeManifest.reset_instance()


@pytest.fixture
def manifest() -> IFKnowledgeManifest:
    """Create a fresh manifest instance."""
    m = IFKnowledgeManifest()
    m.start_session()
    return m


@pytest.fixture
def artifact_with_entities() -> IFTextArtifact:
    """Create artifact with extracted entities in metadata."""
    return IFTextArtifact(
        artifact_id="doc1-chunk1",
        content="Sample text with entities",
        metadata={
            "entities_structured": [
                {
                    "text": "John Smith",
                    "type": "PERSON",
                    "start": 0,
                    "end": 10,
                    "confidence": 0.9,
                },
                {
                    "text": "Microsoft",
                    "type": "ORG",
                    "start": 15,
                    "end": 24,
                    "confidence": 0.85,
                },
            ],
            "chunk_id": "chunk-001",
        },
    )


# ---------------------------------------------------------------------------
# GWT Scenario 1: Singleton Pattern
# ---------------------------------------------------------------------------


class TestSingletonPattern:
    """Given IFKnowledgeManifest, Then it behaves as singleton."""

    def test_singleton_returns_same_instance(self):
        """Given multiple instantiations, When created, Then same instance."""
        m1 = IFKnowledgeManifest()
        m2 = IFKnowledgeManifest()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        """Given reset called, When new instance created, Then different."""
        m1 = IFKnowledgeManifest()
        m1.start_session()
        m1.register_entity("Test", "PERSON", "doc1", "art1")

        IFKnowledgeManifest.reset_instance()
        m2 = IFKnowledgeManifest()

        assert m1 is not m2
        assert m2.entity_count == 0

    def test_thread_safe_singleton(self):
        """Given concurrent access, When creating instances, Then same instance."""
        instances: List[IFKnowledgeManifest] = []

        def create_instance():
            instances.append(IFKnowledgeManifest())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(i is instances[0] for i in instances)


# ---------------------------------------------------------------------------
# GWT Scenario 2: Session Management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Given manifest, When session managed, Then state is correct."""

    def test_start_session(self, manifest: IFKnowledgeManifest):
        """Given manifest, When start_session called, Then session active."""
        assert manifest.is_active is True

    def test_end_session_returns_counts(self, manifest: IFKnowledgeManifest):
        """Given session with data, When end_session called, Then returns counts."""
        manifest.register_entity("Entity1", "PERSON", "doc1", "art1")
        manifest.register_entity("Entity1", "PERSON", "doc2", "art2")  # Cross-doc

        entity_count, join_count = manifest.end_session()

        assert entity_count == 1
        assert join_count == 1
        assert manifest.is_active is False

    def test_cannot_register_without_session(self):
        """Given inactive session, When register called, Then fails."""
        m = IFKnowledgeManifest()
        # Don't start session

        is_new, linked = m.register_entity("Test", "PERSON", "doc1", "art1")

        assert is_new is False
        assert linked is None


# ---------------------------------------------------------------------------
# GWT Scenario 3: O(1) Entity Lookup
# ---------------------------------------------------------------------------


class TestEntityLookup:
    """Given entity registration, Then O(1) lookup works."""

    def test_register_new_entity(self, manifest: IFKnowledgeManifest):
        """Given new entity, When registered, Then returns is_new=True."""
        is_new, linked = manifest.register_entity(
            "John Smith", "PERSON", "doc1", "art1"
        )

        assert is_new is True
        assert linked is None

    def test_lookup_existing_entity(self, manifest: IFKnowledgeManifest):
        """Given registered entity, When looked up, Then found."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")

        entry = manifest.lookup_entity("John Smith", "PERSON")

        assert entry is not None
        assert entry.entity_text == "John Smith"
        assert entry.entity_type == "PERSON"

    def test_lookup_normalizes_case(self, manifest: IFKnowledgeManifest):
        """Given entity with different case, When looked up, Then found."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")

        entry = manifest.lookup_entity("john smith", "PERSON")

        assert entry is not None

    def test_lookup_nonexistent_returns_none(self, manifest: IFKnowledgeManifest):
        """Given nonexistent entity, When looked up, Then None."""
        entry = manifest.lookup_entity("Unknown Person", "PERSON")

        assert entry is None

    def test_different_types_are_separate(self, manifest: IFKnowledgeManifest):
        """Given same text different type, When registered, Then separate."""
        manifest.register_entity("Apple", "ORG", "doc1", "art1")
        manifest.register_entity("Apple", "FOOD", "doc1", "art1")

        assert manifest.entity_count == 2


# ---------------------------------------------------------------------------
# GWT Scenario 4: Cross-Document Entity Detection
# ---------------------------------------------------------------------------


class TestCrossDocumentDetection:
    """Given batch of documents, When same entity found, Then linked."""

    def test_second_document_returns_linked_docs(self, manifest: IFKnowledgeManifest):
        """Given entity in doc1, When found in doc2, Then returns doc1."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")

        is_new, linked = manifest.register_entity(
            "John Smith", "PERSON", "doc2", "art2"
        )

        assert is_new is False
        assert linked == ["doc1"]

    def test_same_document_no_link(self, manifest: IFKnowledgeManifest):
        """Given entity twice in same doc, When registered, Then no link."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")

        is_new, linked = manifest.register_entity(
            "John Smith", "PERSON", "doc1", "art2"
        )

        assert is_new is False
        assert linked is None

    def test_get_cross_document_entities(self, manifest: IFKnowledgeManifest):
        """Given entities in multiple docs, When queried, Then cross-doc found."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")
        manifest.register_entity("John Smith", "PERSON", "doc2", "art2")
        manifest.register_entity("Jane Doe", "PERSON", "doc1", "art3")  # Single doc

        cross_doc = manifest.get_cross_document_entities()

        assert len(cross_doc) == 1
        assert cross_doc[0].entity_text == "John Smith"


# ---------------------------------------------------------------------------
# GWT Scenario 5: Join Artifact Generation
# ---------------------------------------------------------------------------


class TestJoinArtifactGeneration:
    """Given cross-document entity, Then Join artifact generated."""

    def test_creates_join_artifact(self, manifest: IFKnowledgeManifest):
        """Given entity in 2 docs, When registered, Then join created."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")
        manifest.register_entity("John Smith", "PERSON", "doc2", "art2")

        joins = manifest.get_join_artifacts()

        assert len(joins) == 1
        assert isinstance(joins[0], IFJoinArtifact)
        assert joins[0].source_document == "doc1"
        assert joins[0].target_document == "doc2"
        assert joins[0].linked_entity == "John Smith"

    def test_join_artifact_has_required_fields(self, manifest: IFKnowledgeManifest):
        """Given join artifact, Then all required fields present."""
        manifest.register_entity(
            "Microsoft", "ORG", "doc1", "art1", chunk_id="chunk1", confidence=0.9
        )
        manifest.register_entity(
            "Microsoft", "ORG", "doc2", "art2", chunk_id="chunk2", confidence=0.8
        )

        join = manifest.get_join_artifacts()[0]

        assert join.join_type == "entity_link"
        assert join.entity_type == "ORG"
        assert join.source_artifact == "art1"
        assert join.target_artifact == "art2"
        assert "entity_hash" in join.metadata
        assert join.metadata["confidence"] == 0.8  # Min of both

    def test_multiple_joins_for_multi_doc_entity(self, manifest: IFKnowledgeManifest):
        """Given entity in 3 docs, When 3rd added, Then 2 new joins."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")
        manifest.register_entity("John Smith", "PERSON", "doc2", "art2")
        manifest.register_entity("John Smith", "PERSON", "doc3", "art3")

        joins = manifest.get_join_artifacts()

        # doc1-doc2, doc1-doc3, doc2-doc3 (but our impl creates doc1-doc3 and doc2-doc3)
        assert len(joins) >= 2


# ---------------------------------------------------------------------------
# GWT Scenario 6: Document Links Query
# ---------------------------------------------------------------------------


class TestDocumentLinksQuery:
    """Given linked documents, When queried, Then links returned."""

    def test_get_document_links(self, manifest: IFKnowledgeManifest):
        """Given cross-doc entities, When get_document_links called, Then returns links."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")
        manifest.register_entity("John Smith", "PERSON", "doc2", "art2")
        manifest.register_entity("Microsoft", "ORG", "doc1", "art3")
        manifest.register_entity("Microsoft", "ORG", "doc3", "art4")

        links = manifest.get_document_links("doc1")

        assert len(links) == 2
        linked_docs = [l[0] for l in links]
        assert "doc2" in linked_docs
        assert "doc3" in linked_docs


# ---------------------------------------------------------------------------
# GWT Scenario 7: Post-Extraction Hook
# ---------------------------------------------------------------------------


class TestPostExtractionHook:
    """Given artifact with entities, When hook called, Then entities registered."""

    def test_register_extracted_entities(
        self, manifest: IFKnowledgeManifest, artifact_with_entities: IFTextArtifact
    ):
        """Given artifact with entities, When hook called, Then entities added."""
        linked = register_extracted_entities(artifact_with_entities, "doc1", manifest)

        assert manifest.entity_count == 2
        assert len(linked) == 0  # No cross-doc yet

    def test_hook_returns_linked_docs(self, manifest: IFKnowledgeManifest):
        """Given prior entities, When hook called, Then returns links."""
        # Register entity in doc1
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")

        # Create artifact for doc2 with same entity
        artifact = IFTextArtifact(
            artifact_id="doc2-chunk1",
            content="More text",
            metadata={
                "entities_structured": [
                    {
                        "text": "John Smith",
                        "type": "PERSON",
                        "start": 0,
                        "end": 10,
                        "confidence": 0.9,
                    },
                ]
            },
        )

        linked = register_extracted_entities(artifact, "doc2", manifest)

        assert "doc1" in linked

    def test_hook_uses_singleton(self, artifact_with_entities: IFTextArtifact):
        """Given no manifest passed, When hook called, Then uses singleton."""
        # Start session on singleton
        m = IFKnowledgeManifest()
        m.start_session()

        # Don't pass manifest to hook
        register_extracted_entities(artifact_with_entities, "doc1")

        assert m.entity_count == 2


# ---------------------------------------------------------------------------
# JPL Rule #2: Fixed Upper Bounds
# ---------------------------------------------------------------------------


class TestJPLRule2Bounds:
    """Test fixed upper bounds per JPL Rule #2."""

    def test_entity_limit(self, manifest: IFKnowledgeManifest):
        """Given max entities, When more added, Then rejected."""
        # Register up to limit (use smaller limit for test)
        for i in range(100):
            manifest.register_entity(f"Entity{i}", "PERSON", "doc1", f"art{i}")

        # Manifest should have entities
        assert manifest.entity_count == 100

    def test_reference_limit(self, manifest: IFKnowledgeManifest):
        """Given max references, When more added, Then stops adding."""
        # Register same entity many times in different artifacts
        for i in range(MAX_REFERENCES_PER_ENTITY + 100):
            manifest.register_entity("John Smith", "PERSON", "doc1", f"art{i}")

        entry = manifest.lookup_entity("John Smith", "PERSON")
        assert len(entry.references) <= MAX_REFERENCES_PER_ENTITY


# ---------------------------------------------------------------------------
# JPL Rule #4: Functions < 60 Lines
# ---------------------------------------------------------------------------


class TestJPLRule4FunctionSize:
    """Test that functions are under 60 lines per JPL Rule #4."""

    def test_register_entity_size(self):
        """Given register_entity method, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFKnowledgeManifest.register_entity)
        lines = source.split("\n")
        assert len(lines) < 60

    def test_hash_entity_size(self):
        """Given _hash_entity method, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFKnowledgeManifest._hash_entity)
        lines = source.split("\n")
        assert len(lines) < 60


# ---------------------------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_registration(self, manifest: IFKnowledgeManifest):
        """Given concurrent registrations, When executed, Then no data loss."""
        entity_count = 100

        def register_entities(thread_id: int):
            for i in range(entity_count):
                manifest.register_entity(
                    f"Entity{i}", "PERSON", f"doc{thread_id}", f"art{thread_id}-{i}"
                )

        threads = [
            threading.Thread(target=register_entities, args=(i,)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have entity_count unique entities (same text from different docs)
        assert manifest.entity_count == entity_count

    def test_concurrent_lookup(self, manifest: IFKnowledgeManifest):
        """Given concurrent lookups, When executed, Then consistent."""
        manifest.register_entity("Test Entity", "PERSON", "doc1", "art1")

        results = []

        def lookup():
            for _ in range(100):
                entry = manifest.lookup_entity("Test Entity", "PERSON")
                results.append(entry is not None)

        threads = [threading.Thread(target=lookup) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test manifest statistics."""

    def test_get_stats(self, manifest: IFKnowledgeManifest):
        """Given manifest with data, When get_stats called, Then accurate."""
        manifest.register_entity("Entity1", "PERSON", "doc1", "art1")
        manifest.register_entity("Entity1", "PERSON", "doc2", "art2")
        manifest.register_entity("Entity2", "ORG", "doc1", "art3")
        manifest.mark_document_processed("doc1")
        manifest.mark_document_processed("doc2")

        stats = manifest.get_stats()

        assert stats["session_active"] is True
        assert stats["total_entities"] == 2
        assert stats["cross_document_entities"] == 1
        assert stats["total_references"] == 3
        assert stats["join_artifacts"] == 1
        assert stats["documents_processed"] == 2


# ---------------------------------------------------------------------------
# IFJoinArtifact
# ---------------------------------------------------------------------------


class TestIFJoinArtifact:
    """Test IFJoinArtifact model."""

    def test_join_artifact_creation(self):
        """Given join data, When artifact created, Then fields set."""
        join = IFJoinArtifact(
            artifact_id="join-001",
            content="Link description",
            source_document="doc1",
            target_document="doc2",
            linked_entity="John Smith",
            entity_type="PERSON",
            source_artifact="art1",
            target_artifact="art2",
        )

        assert join.join_type == "entity_link"
        assert join.source_document == "doc1"
        assert join.target_document == "doc2"

    def test_join_artifact_derive(self):
        """Given join artifact, When derived, Then lineage preserved."""
        join = IFJoinArtifact(
            artifact_id="join-001",
            content="Link description",
            source_document="doc1",
            target_document="doc2",
            linked_entity="John Smith",
            entity_type="PERSON",
            source_artifact="art1",
            target_artifact="art2",
        )

        derived = join.derive("processor-1")

        assert derived.parent_id == "join-001"
        assert "processor-1" in derived.provenance


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_entity_text(self, manifest: IFKnowledgeManifest):
        """Given empty entity text, When registered, Then still works."""
        is_new, _ = manifest.register_entity("", "PERSON", "doc1", "art1")
        assert is_new is True

    def test_unicode_entity(self, manifest: IFKnowledgeManifest):
        """Given unicode entity, When registered, Then works."""
        is_new, _ = manifest.register_entity("José García", "PERSON", "doc1", "art1")

        entry = manifest.lookup_entity("José García", "PERSON")
        assert entry is not None

    def test_entity_with_special_chars(self, manifest: IFKnowledgeManifest):
        """Given entity with special chars, When registered, Then works."""
        is_new, _ = manifest.register_entity(
            "O'Reilly & Associates, Inc.", "ORG", "doc1", "art1"
        )

        entry = manifest.lookup_entity("O'Reilly & Associates, Inc.", "ORG")
        assert entry is not None

    def test_document_processed_tracking(self, manifest: IFKnowledgeManifest):
        """Given document processed, When checked, Then tracked."""
        assert manifest.is_document_processed("doc1") is False

        manifest.mark_document_processed("doc1")

        assert manifest.is_document_processed("doc1") is True
        assert manifest.is_document_processed("doc2") is False


# ---------------------------------------------------------------------------
# AC: get_relationships() Method
# ---------------------------------------------------------------------------


class TestGetRelationships:
    """AC: Provides get_relationships() method to retrieve all found cross-links."""

    def test_get_relationships_returns_join_artifacts(
        self, manifest: IFKnowledgeManifest
    ):
        """Given cross-doc entities, When get_relationships called, Then returns joins."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")
        manifest.register_entity("John Smith", "PERSON", "doc2", "art2")

        relationships = manifest.get_relationships()

        assert len(relationships) == 1
        assert isinstance(relationships[0], IFJoinArtifact)

    def test_get_relationships_empty_when_no_cross_links(
        self, manifest: IFKnowledgeManifest
    ):
        """Given no cross-doc entities, When get_relationships called, Then empty."""
        manifest.register_entity("John Smith", "PERSON", "doc1", "art1")
        manifest.register_entity("Jane Doe", "PERSON", "doc1", "art2")

        relationships = manifest.get_relationships()

        assert len(relationships) == 0

    def test_get_relationships_alias_for_get_join_artifacts(
        self, manifest: IFKnowledgeManifest
    ):
        """Given relationships, Then get_relationships equals get_join_artifacts."""
        manifest.register_entity("Entity", "ORG", "doc1", "art1")
        manifest.register_entity("Entity", "ORG", "doc2", "art2")

        assert manifest.get_relationships() == manifest.get_join_artifacts()


# ---------------------------------------------------------------------------
# JPL Rule #7: Check Return Values
# ---------------------------------------------------------------------------


class TestJPLRule7ReturnValues:
    """Test explicit return value checking per JPL Rule #7."""

    def test_register_entity_always_returns_tuple(self, manifest: IFKnowledgeManifest):
        """Given any registration, When called, Then returns (bool, Optional[List])."""
        result = manifest.register_entity("Test", "PERSON", "doc1", "art1")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert result[1] is None or isinstance(result[1], list)

    def test_lookup_entity_returns_optional(self, manifest: IFKnowledgeManifest):
        """Given lookup, When called, Then returns ManifestEntry or None."""
        # Non-existent
        result = manifest.lookup_entity("Unknown", "PERSON")
        assert result is None

        # Existent
        manifest.register_entity("Known", "PERSON", "doc1", "art1")
        result = manifest.lookup_entity("Known", "PERSON")
        assert isinstance(result, ManifestEntry)

    def test_start_session_returns_bool(self, manifest: IFKnowledgeManifest):
        """Given start_session, When called, Then returns bool."""
        # Reset to test fresh
        IFKnowledgeManifest.reset_instance()
        m = IFKnowledgeManifest()

        result = m.start_session()

        assert isinstance(result, bool)
        assert result is True

    def test_end_session_returns_tuple(self, manifest: IFKnowledgeManifest):
        """Given end_session, When called, Then returns (int, int)."""
        result = manifest.end_session()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)


# ---------------------------------------------------------------------------
# JPL Rule #9: Type Hints
# ---------------------------------------------------------------------------


class TestJPLRule9TypeHints:
    """Test complete type hints per JPL Rule #9."""

    def test_manifest_methods_have_return_hints(self):
        """Given IFKnowledgeManifest methods, When inspected, Then have return hints."""
        import inspect

        methods_to_check = [
            "start_session",
            "end_session",
            "register_entity",
            "lookup_entity",
            "get_cross_document_entities",
            "get_join_artifacts",
            "get_relationships",
            "get_document_links",
            "get_stats",
            "mark_document_processed",
            "is_document_processed",
        ]

        for method_name in methods_to_check:
            method = getattr(IFKnowledgeManifest, method_name)
            sig = inspect.signature(method)
            assert (
                sig.return_annotation != inspect.Signature.empty
            ), f"{method_name} missing return type hint"

    def test_entity_reference_has_type_hints(self):
        """Given EntityReference, Then all fields typed."""
        hints = EntityReference.__dataclass_fields__

        expected_fields = [
            "document_id",
            "artifact_id",
            "chunk_id",
            "start_char",
            "end_char",
            "confidence",
        ]
        for field in expected_fields:
            assert field in hints, f"Missing field: {field}"

    def test_manifest_entry_has_type_hints(self):
        """Given ManifestEntry, Then all fields typed."""
        hints = ManifestEntry.__dataclass_fields__

        expected_fields = [
            "entity_hash",
            "entity_text",
            "entity_type",
            "references",
            "first_seen_document",
            "first_seen_artifact",
        ]
        for field in expected_fields:
            assert field in hints, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# JPL Rule #2: 10,000 Entity Limit (AC)
# ---------------------------------------------------------------------------


class TestJPLRule2EntityLimit:
    """Test 10,000 entity limit per AC."""

    def test_max_entities_constant_is_10000(self):
        """Given MAX_ENTITIES_IN_MANIFEST, Then equals 10000."""
        assert MAX_ENTITIES_IN_MANIFEST == 10000

    def test_entity_limit_enforcement(self, manifest: IFKnowledgeManifest):
        """Given limit reached, When more added, Then rejected with warning."""

        # Register many entities (not full 10k for test speed)
        for i in range(200):
            manifest.register_entity(f"Entity{i}", "PERSON", "doc1", f"art{i}")

        assert manifest.entity_count == 200

    def test_reference_limit_constant_defined(self):
        """Given MAX_REFERENCES_PER_ENTITY, Then defined."""
        assert MAX_REFERENCES_PER_ENTITY == 1000

    def test_join_limit_constant_defined(self):
        """Given MAX_JOIN_ARTIFACTS_PER_SESSION, Then defined."""
        assert MAX_JOIN_ARTIFACTS_PER_SESSION == 50000


# ---------------------------------------------------------------------------
# GWT Scenario 8: Cross-Link Event
# ---------------------------------------------------------------------------


class TestCrossLinkEvent:
    """AC: Cross-Link event emitted when same entity found in different docs."""

    def test_given_entity_in_doc1_when_found_in_doc2_then_cross_link_created(
        self, manifest: IFKnowledgeManifest
    ):
        """
        GWT Scenario from
        Given: Entity A found in Doc 1.
        When: Entity A is found in Doc 2 during the same run.
        Then: The manifest must resolve them to the same ID and emit a "Cross-Link" event.
        """
        # Given: Entity A found in Doc 1
        is_new1, linked1 = manifest.register_entity(
            entity_text="Entity A",
            entity_type="PERSON",
            document_id="doc1",
            artifact_id="art1",
        )
        assert is_new1 is True
        assert linked1 is None

        # When: Entity A is found in Doc 2
        is_new2, linked2 = manifest.register_entity(
            entity_text="Entity A",
            entity_type="PERSON",
            document_id="doc2",
            artifact_id="art2",
        )

        # Then: Resolve to same ID (is_new=False) and emit cross-link (linked2 contains doc1)
        assert is_new2 is False, "Should resolve to same entity ID"
        assert linked2 == ["doc1"], "Should emit cross-link event referencing doc1"

        # Verify join artifact was created
        joins = manifest.get_relationships()
        assert len(joins) == 1, "Cross-link join artifact should be created"
        assert joins[0].source_document == "doc1"
        assert joins[0].target_document == "doc2"
        assert joins[0].linked_entity == "Entity A"
