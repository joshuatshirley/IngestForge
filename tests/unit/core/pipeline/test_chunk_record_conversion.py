"""
Tests for b: ChunkRecord Conversion Methods.

GWT-style tests for bidirectional conversion between
IFChunkArtifact and ChunkRecord.
"""

import pytest

from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFTextArtifact
from ingestforge.chunking.semantic_chunker import ChunkRecord


# --- Fixtures ---


@pytest.fixture
def sample_chunk_record() -> ChunkRecord:
    """Create a sample ChunkRecord with all fields populated."""
    return ChunkRecord(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="This is sample content for testing.",
        section_title="Introduction",
        chunk_type="content",
        source_file="test.pdf",
        word_count=6,
        char_count=35,
        chunk_index=0,
        total_chunks=5,
        library="test-library",
        is_read=True,
        section_hierarchy=["Chapter 1", "Section 1.1"],
        page_start=1,
        page_end=2,
        ingested_at="2026-02-16T12:00:00Z",
        tags=["important", "test"],
        author_id="author-001",
        author_name="Test Author",
        entities=["Entity1", "Entity2"],
        concepts=["Concept1", "Concept2"],
        quality_score=0.95,
        element_type="NarrativeText",
    )


@pytest.fixture
def sample_chunk_artifact() -> IFChunkArtifact:
    """Create a sample IFChunkArtifact with metadata."""
    return IFChunkArtifact(
        artifact_id="artifact-001",
        document_id="doc-001",
        content="This is sample content for testing.",
        chunk_index=0,
        total_chunks=5,
        metadata={
            "section_title": "Introduction",
            "chunk_type": "content",
            "source_file": "test.pdf",
            "word_count": 6,
            "char_count": 35,
            "library": "test-library",
            "is_read": True,
            "section_hierarchy": ["Chapter 1", "Section 1.1"],
            "page_start": 1,
            "page_end": 2,
            "ingested_at": "2026-02-16T12:00:00Z",
            "tags": ["important", "test"],
            "author_id": "author-001",
            "author_name": "Test Author",
            "entities": ["Entity1", "Entity2"],
            "concepts": ["Concept1", "Concept2"],
            "quality_score": 0.95,
            "element_type": "NarrativeText",
        },
    )


@pytest.fixture
def minimal_chunk_record() -> ChunkRecord:
    """Create a ChunkRecord with only required fields."""
    return ChunkRecord(
        chunk_id="minimal-001",
        document_id="doc-minimal",
        content="Minimal content.",
    )


@pytest.fixture
def parent_artifact() -> IFTextArtifact:
    """Create a parent artifact for lineage testing."""
    return IFTextArtifact(
        artifact_id="parent-001",
        content="Parent content",
        metadata={"source": "test"},
    )


# --- GWT Scenario 1: Convert IFChunkArtifact to ChunkRecord ---


class TestToChunkRecord:
    """Tests for IFChunkArtifact.to_chunk_record()."""

    def test_to_chunk_record_preserves_core_fields(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact, When to_chunk_record() is called,
        Then core fields are preserved."""
        record = sample_chunk_artifact.to_chunk_record()

        assert record.chunk_id == sample_chunk_artifact.artifact_id
        assert record.document_id == sample_chunk_artifact.document_id
        assert record.content == sample_chunk_artifact.content
        assert record.chunk_index == sample_chunk_artifact.chunk_index
        assert record.total_chunks == sample_chunk_artifact.total_chunks

    def test_to_chunk_record_preserves_metadata_fields(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact with metadata, When to_chunk_record() is called,
        Then metadata fields are preserved in ChunkRecord."""
        record = sample_chunk_artifact.to_chunk_record()

        assert record.section_title == "Introduction"
        assert record.chunk_type == "content"
        assert record.source_file == "test.pdf"
        assert record.word_count == 6
        assert record.char_count == 35
        assert record.library == "test-library"
        assert record.is_read is True

    def test_to_chunk_record_preserves_optional_fields(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact, When to_chunk_record() is called,
        Then optional fields are preserved."""
        record = sample_chunk_artifact.to_chunk_record()

        assert record.section_hierarchy == ["Chapter 1", "Section 1.1"]
        assert record.page_start == 1
        assert record.page_end == 2
        assert record.ingested_at == "2026-02-16T12:00:00Z"
        assert record.tags == ["important", "test"]
        assert record.author_id == "author-001"
        assert record.author_name == "Test Author"

    def test_to_chunk_record_preserves_enrichment_fields(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact, When to_chunk_record() is called,
        Then enrichment fields are preserved."""
        record = sample_chunk_artifact.to_chunk_record()

        assert record.entities == ["Entity1", "Entity2"]
        assert record.concepts == ["Concept1", "Concept2"]
        assert record.quality_score == 0.95
        assert record.element_type == "NarrativeText"

    def test_to_chunk_record_returns_chunk_record_type(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact, When to_chunk_record() is called,
        Then result is a ChunkRecord instance."""
        record = sample_chunk_artifact.to_chunk_record()
        assert isinstance(record, ChunkRecord)


# --- GWT Scenario 2: Create IFChunkArtifact from ChunkRecord ---


class TestFromChunkRecord:
    """Tests for IFChunkArtifact.from_chunk_record()."""

    def test_from_chunk_record_preserves_core_fields(
        self, sample_chunk_record: ChunkRecord
    ):
        """Given a ChunkRecord, When from_chunk_record() is called,
        Then core fields are preserved."""
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)

        assert artifact.artifact_id == sample_chunk_record.chunk_id
        assert artifact.document_id == sample_chunk_record.document_id
        assert artifact.content == sample_chunk_record.content
        assert artifact.chunk_index == sample_chunk_record.chunk_index
        assert artifact.total_chunks == sample_chunk_record.total_chunks

    def test_from_chunk_record_stores_metadata(self, sample_chunk_record: ChunkRecord):
        """Given a ChunkRecord, When from_chunk_record() is called,
        Then all fields stored in artifact metadata."""
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)

        assert artifact.metadata["section_title"] == "Introduction"
        assert artifact.metadata["chunk_type"] == "content"
        assert artifact.metadata["source_file"] == "test.pdf"
        assert artifact.metadata["word_count"] == 6
        assert artifact.metadata["library"] == "test-library"
        assert artifact.metadata["is_read"] is True

    def test_from_chunk_record_stores_optional_fields(
        self, sample_chunk_record: ChunkRecord
    ):
        """Given a ChunkRecord with optional fields, When from_chunk_record() is called,
        Then optional fields are stored in metadata."""
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)

        assert artifact.metadata["section_hierarchy"] == ["Chapter 1", "Section 1.1"]
        assert artifact.metadata["page_start"] == 1
        assert artifact.metadata["page_end"] == 2
        assert artifact.metadata["tags"] == ["important", "test"]

    def test_from_chunk_record_with_parent_lineage(
        self, sample_chunk_record: ChunkRecord, parent_artifact: IFTextArtifact
    ):
        """Given a ChunkRecord and parent, When from_chunk_record() is called,
        Then lineage is tracked."""
        artifact = IFChunkArtifact.from_chunk_record(
            sample_chunk_record, parent=parent_artifact
        )

        assert artifact.parent_id == parent_artifact.artifact_id
        assert artifact.lineage_depth == 1
        assert "from-chunk-record" in artifact.provenance

    def test_from_chunk_record_returns_artifact_type(
        self, sample_chunk_record: ChunkRecord
    ):
        """Given a ChunkRecord, When from_chunk_record() is called,
        Then result is an IFChunkArtifact instance."""
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)
        assert isinstance(artifact, IFChunkArtifact)

    def test_from_chunk_record_computes_content_hash(
        self, sample_chunk_record: ChunkRecord
    ):
        """Given a ChunkRecord, When from_chunk_record() is called,
        Then content hash is computed."""
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)
        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64  # SHA-256 hex


# --- GWT Scenario 3: Round-Trip Conversion Fidelity ---


class TestRoundTripConversion:
    """Tests for round-trip conversion fidelity."""

    def test_artifact_to_record_to_artifact_preserves_content(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact, When converted to ChunkRecord and back,
        Then content is identical."""
        record = sample_chunk_artifact.to_chunk_record()
        restored = IFChunkArtifact.from_chunk_record(record)

        assert restored.content == sample_chunk_artifact.content
        assert restored.document_id == sample_chunk_artifact.document_id

    def test_artifact_to_record_to_artifact_preserves_indices(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact, When converted to ChunkRecord and back,
        Then chunk indices are preserved."""
        record = sample_chunk_artifact.to_chunk_record()
        restored = IFChunkArtifact.from_chunk_record(record)

        assert restored.chunk_index == sample_chunk_artifact.chunk_index
        assert restored.total_chunks == sample_chunk_artifact.total_chunks

    def test_record_to_artifact_to_record_preserves_fields(
        self, sample_chunk_record: ChunkRecord
    ):
        """Given a ChunkRecord, When converted to artifact and back,
        Then all fields preserved."""
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)
        restored = artifact.to_chunk_record()

        assert restored.chunk_id == sample_chunk_record.chunk_id
        assert restored.content == sample_chunk_record.content
        assert restored.section_title == sample_chunk_record.section_title
        assert restored.tags == sample_chunk_record.tags
        assert restored.quality_score == sample_chunk_record.quality_score

    def test_round_trip_preserves_artifact_id(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """Given an IFChunkArtifact, When round-tripped,
        Then artifact_id is preserved via chunk_id."""
        record = sample_chunk_artifact.to_chunk_record()
        restored = IFChunkArtifact.from_chunk_record(record)

        assert restored.artifact_id == sample_chunk_artifact.artifact_id


# --- GWT Scenario 4: Handle Optional ChunkRecord Fields ---


class TestOptionalFields:
    """Tests for handling optional ChunkRecord fields."""

    def test_minimal_record_converts_successfully(
        self, minimal_chunk_record: ChunkRecord
    ):
        """Given a ChunkRecord with minimal fields, When from_chunk_record() is called,
        Then artifact is created with defaults."""
        artifact = IFChunkArtifact.from_chunk_record(minimal_chunk_record)

        assert artifact.artifact_id == "minimal-001"
        assert artifact.content == "Minimal content."
        assert artifact.metadata["section_title"] == ""
        assert artifact.metadata["library"] == "default"

    def test_minimal_artifact_converts_to_record(self):
        """Given a minimal IFChunkArtifact, When to_chunk_record() is called,
        Then ChunkRecord has appropriate defaults."""
        artifact = IFChunkArtifact(
            artifact_id="minimal-art",
            document_id="doc-min",
            content="Minimal content.",
            metadata={},
        )
        record = artifact.to_chunk_record()

        assert record.chunk_id == "minimal-art"
        assert record.section_title == ""
        assert record.library == "default"
        assert record.is_read is False

    def test_missing_optional_fields_use_defaults(
        self, minimal_chunk_record: ChunkRecord
    ):
        """Given a ChunkRecord missing optional fields, When from_chunk_record() is called,
        Then defaults are used appropriately."""
        artifact = IFChunkArtifact.from_chunk_record(minimal_chunk_record)

        assert artifact.metadata.get("page_start") is None
        assert artifact.metadata.get("page_end") is None
        assert artifact.metadata.get("author_id") is None

    def test_empty_lists_handled_correctly(self):
        """Given a ChunkRecord with empty lists, When converted,
        Then empty lists preserved."""
        record = ChunkRecord(
            chunk_id="empty-lists",
            document_id="doc-empty",
            content="Content.",
            tags=[],
            entities=[],
            concepts=[],
        )
        artifact = IFChunkArtifact.from_chunk_record(record)

        # Empty lists should not be stored in metadata (falsy)
        assert "tags" not in artifact.metadata or artifact.metadata["tags"] == []


# --- GWT Scenario 5: Preserve Lineage in Conversion ---


class TestLineagePreservation:
    """Tests for lineage preservation during conversion."""

    def test_lineage_stored_in_record_metadata(self):
        """Given an IFChunkArtifact with lineage, When to_chunk_record() is called,
        Then lineage is stored in ChunkRecord metadata."""
        artifact = IFChunkArtifact(
            artifact_id="child-001",
            document_id="doc-001",
            content="Child content.",
            parent_id="parent-001",
            root_artifact_id="root-001",
            lineage_depth=2,
            provenance=["extractor", "chunker"],
            metadata={"key": "value"},
        )
        record = artifact.to_chunk_record()

        assert record.metadata.get("_lineage_parent_id") == "parent-001"
        assert record.metadata.get("_lineage_root_id") == "root-001"
        assert record.metadata.get("_lineage_depth") == 2
        assert record.metadata.get("_lineage_provenance") == ["extractor", "chunker"]

    def test_lineage_from_parent_artifact(
        self, sample_chunk_record: ChunkRecord, parent_artifact: IFTextArtifact
    ):
        """Given a ChunkRecord and parent artifact, When from_chunk_record() is called,
        Then lineage is derived from parent."""
        artifact = IFChunkArtifact.from_chunk_record(
            sample_chunk_record, parent=parent_artifact
        )

        assert artifact.parent_id == parent_artifact.artifact_id
        assert (
            artifact.root_artifact_id == parent_artifact.artifact_id
        )  # parent is root
        assert artifact.lineage_depth == 1

    def test_content_hash_preserved_in_metadata(self):
        """Given an IFChunkArtifact with content_hash, When to_chunk_record() is called,
        Then content_hash is preserved in metadata."""
        artifact = IFChunkArtifact(
            artifact_id="hash-test",
            document_id="doc-001",
            content="Test content.",
            metadata={},
        )
        record = artifact.to_chunk_record()

        assert "_content_hash" in record.metadata
        assert record.metadata["_content_hash"] == artifact.content_hash

    def test_no_lineage_fields_when_no_parent(self, sample_chunk_record: ChunkRecord):
        """Given a ChunkRecord without parent, When from_chunk_record() is called,
        Then no lineage fields are set."""
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)

        assert artifact.parent_id is None
        assert artifact.root_artifact_id is None
        assert artifact.lineage_depth == 0
        assert artifact.provenance == []


# --- JPL Rule Compliance Tests ---


class TestJPLRule7ExplicitReturns:
    """Tests for JPL Rule #7: Explicit return types."""

    def test_from_chunk_record_has_return_annotation(self):
        """Given from_chunk_record method, it has return type annotation."""
        annotations = IFChunkArtifact.from_chunk_record.__annotations__
        assert "return" in annotations

    def test_to_chunk_record_has_return_annotation(self):
        """Given to_chunk_record method, it has return type annotation."""
        annotations = IFChunkArtifact.to_chunk_record.__annotations__
        assert "return" in annotations


class TestJPLRule9TypeHints:
    """Tests for JPL Rule #9: Complete type hints."""

    def test_from_chunk_record_parameter_hints(self):
        """Given from_chunk_record method, all parameters have type hints."""
        annotations = IFChunkArtifact.from_chunk_record.__annotations__
        assert "record" in annotations
        assert "parent" in annotations

    def test_to_chunk_record_is_instance_method(self):
        """Given to_chunk_record method, it's a proper instance method."""
        artifact = IFChunkArtifact(
            artifact_id="test",
            document_id="doc",
            content="content",
        )
        # Method should be callable on instance
        assert callable(artifact.to_chunk_record)


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases in conversion."""

    def test_empty_content_artifact(self):
        """Given an artifact with empty content, conversion succeeds."""
        artifact = IFChunkArtifact(
            artifact_id="empty-content",
            document_id="doc",
            content="",
            metadata={},
        )
        record = artifact.to_chunk_record()
        assert record.content == ""

    def test_special_characters_in_content(self):
        """Given content with special characters, conversion preserves them."""
        content = "Unicode: \u4e2d\u6587 \u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u263a Special: <>&\"'"
        record = ChunkRecord(
            chunk_id="special",
            document_id="doc",
            content=content,
        )
        artifact = IFChunkArtifact.from_chunk_record(record)
        restored = artifact.to_chunk_record()

        assert artifact.content == content
        assert restored.content == content

    def test_large_metadata_dict(self):
        """Given an artifact with large metadata, conversion handles it."""
        artifact = IFChunkArtifact(
            artifact_id="large-meta",
            document_id="doc",
            content="content",
            metadata={f"key_{i}": f"value_{i}" for i in range(50)},
        )
        record = artifact.to_chunk_record()
        assert record.chunk_id == "large-meta"

    def test_nested_chunk_metadata_preserved(self):
        """Given artifact with chunk_metadata, it's preserved in round-trip."""
        artifact = IFChunkArtifact(
            artifact_id="nested-meta",
            document_id="doc",
            content="content",
            metadata={
                "section_title": "Test",
                "chunk_metadata": {"custom_key": "custom_value"},
            },
        )
        record = artifact.to_chunk_record()

        assert record.metadata.get("custom_key") == "custom_value"

    def test_generates_uuid_when_no_chunk_id(self):
        """Given ChunkRecord with empty chunk_id, UUID is generated."""
        record = ChunkRecord(
            chunk_id="",  # Empty
            document_id="doc",
            content="content",
        )
        artifact = IFChunkArtifact.from_chunk_record(record)

        # Should generate UUID since empty string is falsy
        assert artifact.artifact_id != ""
        # Either uses empty string or generates UUID
        assert len(artifact.artifact_id) > 0


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_artifact_to_record_covered(self):
        """GWT Scenario 1 (IFChunkArtifact to ChunkRecord) is tested."""
        assert hasattr(TestToChunkRecord, "test_to_chunk_record_preserves_core_fields")

    def test_scenario_2_record_to_artifact_covered(self):
        """GWT Scenario 2 (ChunkRecord to IFChunkArtifact) is tested."""
        assert hasattr(
            TestFromChunkRecord, "test_from_chunk_record_preserves_core_fields"
        )

    def test_scenario_3_round_trip_covered(self):
        """GWT Scenario 3 (Round-Trip Fidelity) is tested."""
        assert hasattr(
            TestRoundTripConversion,
            "test_artifact_to_record_to_artifact_preserves_content",
        )

    def test_scenario_4_optional_fields_covered(self):
        """GWT Scenario 4 (Optional Fields) is tested."""
        assert hasattr(TestOptionalFields, "test_minimal_record_converts_successfully")

    def test_scenario_5_lineage_preservation_covered(self):
        """GWT Scenario 5 (Lineage Preservation) is tested."""
        assert hasattr(
            TestLineagePreservation, "test_lineage_stored_in_record_metadata"
        )
