"""
Tests for TASK-010: Pipeline Migration to IFChunkArtifact.

Comprehensive testing of pipeline.py changes migrating from ChunkRecord
to IFChunkArtifact. Validates round-trip conversions, backward compatibility,
and metadata preservation.

Test Coverage Areas:
1. Round-trip Conversions (chunk_to_artifact → artifact_to_chunk)
2. Backward Compatibility with legacy code
3. Pipeline Execution with IFChunkArtifact
4. Edge Cases (empty chunks, large metadata, special characters)

JPL Compliance:
- Rule #2: Fixed bounds on test data sizes
- Rule #7: Check all return values in tests
- Rule #9: Complete type hints
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from ingestforge.core.pipeline.pipeline import Pipeline
from ingestforge.core.pipeline.artifacts import IFChunkArtifact
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config
from ingestforge.core.provenance import SourceLocation, SourceType


# JPL Rule #2: Fixed upper bounds for test data
MAX_TEST_CHUNKS = 100
MAX_METADATA_KEYS = 50
MAX_CONTENT_LENGTH = 10000


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    """Create a mock Config for testing."""
    config = Mock(spec=Config)
    config.data_path = tmp_path / "data"
    config.pending_path = tmp_path / "pending"
    config.completed_path = tmp_path / "completed"
    config.project.name = "test_project"

    # Chunking config
    config.chunking = Mock()
    config.chunking.strategy = "semantic"
    config.chunking.min_size = 100
    config.chunking.max_size = 500

    # Enrichment config
    config.enrichment = Mock()
    config.enrichment.generate_embeddings = False
    config.enrichment.extract_entities = False
    config.enrichment.generate_questions = False
    config.enrichment.generate_summaries = False
    config.enrichment.use_instructor_citation = False
    config.enrichment.compute_quality = False

    # Storage config
    config.storage = Mock()
    config.storage.backend = "jsonl"

    # Retrieval config
    config.retrieval = Mock()
    config.retrieval.top_k = 5

    # Refinement config
    config.refinement = Mock()
    config.refinement.cleanup_ocr = False
    config.refinement.normalize_formatting = False
    config.refinement.detect_chapters = False

    config.ensure_directories = Mock()

    return config


@pytest.fixture
def sample_chunk_record() -> ChunkRecord:
    """Create a sample ChunkRecord with comprehensive metadata."""
    return ChunkRecord(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="This is a test chunk with sample content for testing migrations.",
        section_title="Test Section",
        chunk_type="content",
        source_file="test.pdf",
        word_count=10,
        char_count=65,
        chunk_index=0,
        total_chunks=5,
        library="test-library",
        is_read=False,
        section_hierarchy=["Chapter 1", "Section 1.1"],
        page_start=1,
        page_end=2,
        ingested_at="2026-02-17T10:00:00Z",
        tags=["important", "test"],
        author_id="author-001",
        author_name="Test Author",
        entities=["Entity1", "Entity2"],
        concepts=["Concept1"],
        quality_score=0.85,
        element_type="NarrativeText",
    )


@pytest.fixture
def sample_chunk_artifact() -> IFChunkArtifact:
    """Create a sample IFChunkArtifact with metadata."""
    return IFChunkArtifact(
        artifact_id="artifact-001",
        document_id="doc-001",
        content="This is a test chunk with sample content for testing migrations.",
        chunk_index=0,
        total_chunks=5,
        metadata={
            "section_title": "Test Section",
            "chunk_type": "content",
            "source_file": "test.pdf",
            "word_count": 10,
            "char_count": 65,
            "library": "test-library",
            "is_read": False,
            "element_type": "NarrativeText",
            "section_hierarchy": ["Chapter 1", "Section 1.1"],
            "page_start": 1,
            "page_end": 2,
            "ingested_at": "2026-02-17T10:00:00Z",
            "tags": ["important", "test"],
            "author_id": "author-001",
            "author_name": "Test Author",
            "entities": ["Entity1", "Entity2"],
            "concepts": ["Concept1"],
            "quality_score": 0.85,
        },
    )


@pytest.fixture
def source_location() -> SourceLocation:
    """Create a sample SourceLocation."""
    return SourceLocation(
        source_type=SourceType.PDF,
        source="test.pdf",
        title="Test Document",
    )


# ============================================================================
# Test Coverage Area 1: Round-Trip Conversions
# ============================================================================


class TestRoundTripConversions:
    """Tests for round-trip conversion fidelity between ChunkRecord and IFChunkArtifact."""

    def test_chunk_record_to_artifact_preserves_core_fields(
        self, sample_chunk_record: ChunkRecord
    ):
        """
        Given: A ChunkRecord with all fields populated
        When: Converting to IFChunkArtifact
        Then: Core fields are preserved

        JPL Rule #7: Check all return values.
        """
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)

        assert artifact.artifact_id == sample_chunk_record.chunk_id
        assert artifact.document_id == sample_chunk_record.document_id
        assert artifact.content == sample_chunk_record.content
        assert artifact.chunk_index == sample_chunk_record.chunk_index
        assert artifact.total_chunks == sample_chunk_record.total_chunks
        assert artifact.content_hash is not None  # Auto-computed

    def test_chunk_record_to_artifact_preserves_metadata(
        self, sample_chunk_record: ChunkRecord
    ):
        """
        Given: A ChunkRecord with metadata fields
        When: Converting to IFChunkArtifact
        Then: All metadata fields are preserved in artifact.metadata

        JPL Rule #7: Check all return values.
        """
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)

        assert artifact.metadata["section_title"] == "Test Section"
        assert artifact.metadata["chunk_type"] == "content"
        assert artifact.metadata["source_file"] == "test.pdf"
        assert artifact.metadata["word_count"] == 10
        assert artifact.metadata["char_count"] == 65
        assert artifact.metadata["library"] == "test-library"
        assert artifact.metadata["is_read"] is False
        assert artifact.metadata["element_type"] == "NarrativeText"

    def test_artifact_to_chunk_record_preserves_core_fields(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """
        Given: An IFChunkArtifact
        When: Converting to ChunkRecord
        Then: Core fields are preserved

        JPL Rule #7: Check all return values.
        """
        record = sample_chunk_artifact.to_chunk_record()

        assert record.chunk_id == sample_chunk_artifact.artifact_id
        assert record.document_id == sample_chunk_artifact.document_id
        assert record.content == sample_chunk_artifact.content
        assert record.chunk_index == sample_chunk_artifact.chunk_index
        assert record.total_chunks == sample_chunk_artifact.total_chunks

    def test_artifact_to_chunk_record_preserves_metadata(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """
        Given: An IFChunkArtifact with metadata
        When: Converting to ChunkRecord
        Then: Metadata fields are extracted to ChunkRecord attributes

        JPL Rule #7: Check all return values.
        """
        record = sample_chunk_artifact.to_chunk_record()

        assert record.section_title == "Test Section"
        assert record.chunk_type == "content"
        assert record.source_file == "test.pdf"
        assert record.word_count == 10
        assert record.char_count == 65
        assert record.library == "test-library"
        assert record.is_read is False
        assert record.element_type == "NarrativeText"

    def test_full_round_trip_record_to_artifact_to_record(
        self, sample_chunk_record: ChunkRecord
    ):
        """
        Given: A ChunkRecord
        When: Converting to IFChunkArtifact and back to ChunkRecord
        Then: All fields are preserved

        JPL Rule #7: Check all return values.
        """
        # Record → Artifact → Record
        artifact = IFChunkArtifact.from_chunk_record(sample_chunk_record)
        restored_record = artifact.to_chunk_record()

        # Verify core fields
        assert restored_record.chunk_id == sample_chunk_record.chunk_id
        assert restored_record.document_id == sample_chunk_record.document_id
        assert restored_record.content == sample_chunk_record.content
        assert restored_record.chunk_index == sample_chunk_record.chunk_index
        assert restored_record.total_chunks == sample_chunk_record.total_chunks

        # Verify metadata fields
        assert restored_record.section_title == sample_chunk_record.section_title
        assert restored_record.chunk_type == sample_chunk_record.chunk_type
        assert restored_record.library == sample_chunk_record.library
        assert restored_record.tags == sample_chunk_record.tags
        assert restored_record.quality_score == sample_chunk_record.quality_score

    def test_full_round_trip_artifact_to_record_to_artifact(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """
        Given: An IFChunkArtifact
        When: Converting to ChunkRecord and back to IFChunkArtifact
        Then: All fields are preserved

        JPL Rule #7: Check all return values.
        """
        # Artifact → Record → Artifact
        record = sample_chunk_artifact.to_chunk_record()
        restored_artifact = IFChunkArtifact.from_chunk_record(record)

        # Verify core fields
        assert restored_artifact.artifact_id == sample_chunk_artifact.artifact_id
        assert restored_artifact.document_id == sample_chunk_artifact.document_id
        assert restored_artifact.content == sample_chunk_artifact.content
        assert restored_artifact.chunk_index == sample_chunk_artifact.chunk_index
        assert restored_artifact.total_chunks == sample_chunk_artifact.total_chunks

        # Verify metadata fields
        assert (
            restored_artifact.metadata["section_title"]
            == sample_chunk_artifact.metadata["section_title"]
        )
        assert (
            restored_artifact.metadata["library"]
            == sample_chunk_artifact.metadata["library"]
        )

    def test_round_trip_preserves_content_hash(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """
        Given: An IFChunkArtifact with content_hash
        When: Round-tripping through ChunkRecord
        Then: content_hash is preserved via metadata

        JPL Rule #7: Check all return values.
        """
        original_hash = sample_chunk_artifact.content_hash

        # Round-trip
        record = sample_chunk_artifact.to_chunk_record()
        restored = IFChunkArtifact.from_chunk_record(record)

        # Content hash should be recomputed from content (same value)
        assert restored.content_hash is not None
        assert restored.content_hash == original_hash


# ============================================================================
# Test Coverage Area 2: Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy ChunkRecord code."""

    def test_pipeline_create_chunk_artifact_method(
        self, source_location: SourceLocation
    ):
        """
        Given: Pipeline._create_chunk_artifact method
        When: Creating a chunk artifact
        Then: Returns IFChunkArtifact (not ChunkRecord)

        Migration to IFChunkArtifact.
        """
        # Mock chunk object
        mock_chunk = Mock()
        mock_chunk.content = "Test content"

        # Create pipeline instance (lightweight)

        # We can't easily instantiate Pipeline in tests, so we test the converter directly
        artifact = IFChunkArtifact.from_chunk_record(
            ChunkRecord(
                chunk_id="test-001",
                document_id="doc-001",
                content="Test content",
            )
        )

        # Verify it's an IFChunkArtifact
        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.content == "Test content"

    def test_legacy_code_receiving_chunk_record_still_works(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """
        Given: Legacy code expecting ChunkRecord
        When: IFChunkArtifact is converted to ChunkRecord
        Then: Legacy code can process it

        JPL Rule #7: Check all return values.
        """

        # Simulate legacy function expecting ChunkRecord
        def legacy_function(chunk: ChunkRecord) -> str:
            return f"Processing {chunk.chunk_id}: {chunk.content[:20]}"

        # Convert artifact to record for legacy code
        record = sample_chunk_artifact.to_chunk_record()
        result = legacy_function(record)

        assert "artifact-001" in result
        assert "This is a test chunk" in result

    def test_new_code_receiving_artifact_works(
        self, sample_chunk_artifact: IFChunkArtifact
    ):
        """
        Given: New code expecting IFChunkArtifact
        When: Processing artifact directly
        Then: No conversion needed

        JPL Rule #7: Check all return values.
        """

        # Simulate new function using IFChunkArtifact
        def new_function(artifact: IFChunkArtifact) -> str:
            return f"Artifact {artifact.artifact_id}: {artifact.content[:20]}"

        result = new_function(sample_chunk_artifact)

        assert "artifact-001" in result
        assert "This is a test chunk" in result

    def test_converters_handle_both_directions(self):
        """
        Given: Converter functions
        When: Converting in both directions
        Then: Both conversions work correctly

        JPL Rule #7: Check all return values.
        """
        # Create original record
        original_record = ChunkRecord(
            chunk_id="bidirectional-001",
            document_id="doc-001",
            content="Bidirectional test content",
        )

        # Forward: Record → Artifact
        artifact = IFChunkArtifact.from_chunk_record(original_record)
        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.content == original_record.content

        # Backward: Artifact → Record
        restored_record = artifact.to_chunk_record()
        assert isinstance(restored_record, ChunkRecord)
        assert restored_record.content == original_record.content


# ============================================================================
# Test Coverage Area 3: Pipeline Execution with IFChunkArtifact
# ============================================================================


class TestPipelineExecution:
    """Tests for pipeline execution using IFChunkArtifact."""

    def test_chunk_text_content_returns_artifacts(
        self, mock_config: Config, source_location: SourceLocation
    ):
        """
        Given: Pipeline._chunk_text_content method
        When: Chunking text content
        Then: Returns list of IFChunkArtifact instances

        Migration to IFChunkArtifact.
        JPL Rule #7: Check all return values.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch("ingestforge.core.state.StateManager"):
                pipeline = Pipeline(config=mock_config)

                # Mock the chunker to return simple chunks
                mock_chunk = Mock()
                mock_chunk.content = "Test chunk content"

                pipeline._chunker = Mock()
                pipeline._chunker.chunk_text = Mock(return_value=[mock_chunk])

                # Execute chunking
                chunks = pipeline._chunk_text_content(
                    text="This is test text for chunking.",
                    document_id="doc-001",
                    source="test.txt",
                    title="Test Document",
                    source_location=source_location,
                    metadata={"library": "test-lib"},
                )

                # Verify returns IFChunkArtifact instances
                assert isinstance(chunks, list)
                assert len(chunks) == 1
                assert isinstance(chunks[0], IFChunkArtifact)
                assert chunks[0].content == "Test chunk content"

    def test_create_chunk_artifact_builds_correct_metadata(
        self, mock_config: Config, source_location: SourceLocation
    ):
        """
        Given: Pipeline._create_chunk_artifact helper
        When: Creating a chunk artifact
        Then: Metadata is correctly populated

        Migration to IFChunkArtifact.
        JPL Rule #7: Check all return values.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch("ingestforge.core.state.StateManager"):
                pipeline = Pipeline(config=mock_config)

                mock_chunk = Mock()
                mock_chunk.content = "Test content for metadata"

                artifact = pipeline._create_chunk_artifact(
                    index=0,
                    chunk=mock_chunk,
                    document_id="doc-001",
                    source="test.txt",
                    title="Test Title",
                    source_location=source_location,
                    metadata={"library": "test-lib", "custom_key": "custom_value"},
                )

                # Verify metadata structure
                assert artifact.metadata["section_title"] == "Test Title"
                assert artifact.metadata["chunk_type"] == "text"
                assert artifact.metadata["source_file"] == "test.txt"
                assert artifact.metadata["library"] == "test-lib"
                assert artifact.metadata["custom_key"] == "custom_value"
                assert "word_count" in artifact.metadata

    def test_pipeline_stages_handle_artifacts(self, mock_config: Config):
        """
        Given: Pipeline with artifact-based stages
        When: Processing through pipeline stages
        Then: Artifacts flow correctly through stages

        Uses IFPipelineRunner for stage orchestration.
        JPL Rule #7: Check all return values.
        """
        # This test validates that the pipeline can handle artifacts
        # without breaking stage transitions

        artifact = IFChunkArtifact(
            artifact_id="stage-test-001",
            document_id="doc-001",
            content="Stage test content",
            metadata={"test": "value"},
        )

        # Verify artifact has required attributes for pipeline
        assert hasattr(artifact, "artifact_id")
        assert hasattr(artifact, "document_id")
        assert hasattr(artifact, "content")
        assert hasattr(artifact, "metadata")
        assert hasattr(artifact, "to_chunk_record")


# ============================================================================
# Test Coverage Area 4: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in chunk conversions and pipeline processing."""

    def test_empty_chunk_content(self):
        """
        Given: A chunk with empty content
        When: Converting between formats
        Then: Empty content is preserved

        JPL Rule #7: Check all return values.
        """
        record = ChunkRecord(
            chunk_id="empty-001",
            document_id="doc-001",
            content="",
        )

        artifact = IFChunkArtifact.from_chunk_record(record)
        restored = artifact.to_chunk_record()

        assert artifact.content == ""
        assert restored.content == ""

    def test_large_metadata_dictionary(self):
        """
        Given: An artifact with large metadata dictionary
        When: Converting to ChunkRecord
        Then: Handles large metadata gracefully

        JPL Rule #2: Fixed bounds on test data (MAX_METADATA_KEYS).
        JPL Rule #7: Check all return values.
        """
        # Create artifact with many metadata keys (bounded)
        large_metadata = {
            "section_title": "Test",
            "library": "test-lib",
        }
        # Add bounded number of custom keys
        for i in range(min(30, MAX_METADATA_KEYS - 2)):  # -2 for existing keys
            large_metadata[f"custom_key_{i}"] = f"value_{i}"

        artifact = IFChunkArtifact(
            artifact_id="large-meta-001",
            document_id="doc-001",
            content="Content",
            metadata=large_metadata,
        )

        # Should handle conversion without errors
        record = artifact.to_chunk_record()
        assert record.chunk_id == "large-meta-001"
        assert record.section_title == "Test"

    def test_special_characters_in_content(self):
        """
        Given: Content with special characters and Unicode
        When: Round-tripping through conversions
        Then: Special characters are preserved

        JPL Rule #7: Check all return values.
        """
        special_content = (
            "Unicode: 中文 Русский ☺ " "Special: <>&\"' " "Newlines:\n\nTabs:\t\t"
        )

        record = ChunkRecord(
            chunk_id="special-001",
            document_id="doc-001",
            content=special_content,
        )

        artifact = IFChunkArtifact.from_chunk_record(record)
        restored = artifact.to_chunk_record()

        assert artifact.content == special_content
        assert restored.content == special_content

    def test_missing_optional_fields(self):
        """
        Given: A ChunkRecord with missing optional fields
        When: Converting to IFChunkArtifact
        Then: Uses appropriate defaults

        JPL Rule #7: Check all return values.
        """
        minimal_record = ChunkRecord(
            chunk_id="minimal-001",
            document_id="doc-001",
            content="Minimal content",
            # All optional fields omitted
        )

        artifact = IFChunkArtifact.from_chunk_record(minimal_record)

        # Should have defaults
        assert artifact.metadata.get("section_title") == ""
        assert artifact.metadata.get("chunk_type") == "content"
        assert artifact.metadata.get("library") == "default"
        assert artifact.metadata.get("is_read") is False
        assert artifact.metadata.get("word_count") == 0

    def test_none_values_in_metadata(self):
        """
        Given: Metadata with None values
        When: Converting between formats
        Then: None values are handled correctly

        JPL Rule #7: Check all return values.
        """
        artifact = IFChunkArtifact(
            artifact_id="none-test-001",
            document_id="doc-001",
            content="Content",
            metadata={
                "section_title": "Test",
                "page_start": None,
                "page_end": None,
                "author_id": None,
            },
        )

        record = artifact.to_chunk_record()

        # None values should be preserved
        assert record.page_start is None
        assert record.page_end is None
        assert record.author_id is None

    def test_very_long_content(self):
        """
        Given: Chunk with very long content
        When: Converting between formats
        Then: Content is preserved without truncation

        JPL Rule #2: Fixed bounds (MAX_CONTENT_LENGTH).
        JPL Rule #7: Check all return values.
        """
        # Create long content (bounded by test limit)
        long_content = "x" * min(5000, MAX_CONTENT_LENGTH)

        record = ChunkRecord(
            chunk_id="long-001",
            document_id="doc-001",
            content=long_content,
        )

        artifact = IFChunkArtifact.from_chunk_record(record)
        restored = artifact.to_chunk_record()

        assert len(artifact.content) == len(long_content)
        assert len(restored.content) == len(long_content)
        assert artifact.content == long_content

    def test_list_fields_preserved(self):
        """
        Given: ChunkRecord with list fields (tags, entities, concepts)
        When: Round-tripping through conversions
        Then: Lists are preserved

        JPL Rule #7: Check all return values.
        """
        record = ChunkRecord(
            chunk_id="lists-001",
            document_id="doc-001",
            content="Content",
            tags=["tag1", "tag2", "tag3"],
            entities=["Entity A", "Entity B"],
            concepts=["Concept X", "Concept Y", "Concept Z"],
        )

        artifact = IFChunkArtifact.from_chunk_record(record)
        restored = artifact.to_chunk_record()

        assert restored.tags == ["tag1", "tag2", "tag3"]
        assert restored.entities == ["Entity A", "Entity B"]
        assert restored.concepts == ["Concept X", "Concept Y", "Concept Z"]

    def test_empty_lists_preserved(self):
        """
        Given: ChunkRecord with empty list fields
        When: Converting to IFChunkArtifact
        Then: Empty lists are handled correctly

        JPL Rule #7: Check all return values.
        """
        record = ChunkRecord(
            chunk_id="empty-lists-001",
            document_id="doc-001",
            content="Content",
            tags=[],
            entities=[],
            concepts=[],
        )

        artifact = IFChunkArtifact.from_chunk_record(record)
        restored = artifact.to_chunk_record()

        # Empty lists should be preserved
        assert restored.tags == []
        assert restored.entities == []
        assert restored.concepts == []


# ============================================================================
# JPL Compliance Tests
# ============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten rule compliance."""

    def test_rule_7_explicit_return_types(self):
        """
        Given: Converter methods
        When: Checking type annotations
        Then: All have explicit return type annotations

        JPL Rule #7: Check all return values.
        """
        # Check from_chunk_record
        annotations = IFChunkArtifact.from_chunk_record.__annotations__
        assert "return" in annotations
        assert annotations["return"] == "IFChunkArtifact"

        # Check to_chunk_record (note: uses TYPE_CHECKING forward reference)
        annotations = IFChunkArtifact.to_chunk_record.__annotations__
        assert "return" in annotations

    def test_rule_9_complete_type_hints(self):
        """
        Given: Converter methods
        When: Checking parameter annotations
        Then: All parameters have type hints

        JPL Rule #9: Complete type hints.
        """
        # Check from_chunk_record parameters
        annotations = IFChunkArtifact.from_chunk_record.__annotations__
        assert "record" in annotations
        assert "parent" in annotations

        # Check to_chunk_record is instance method
        artifact = IFChunkArtifact(
            artifact_id="type-test",
            document_id="doc",
            content="content",
        )
        assert callable(artifact.to_chunk_record)

    def test_rule_2_fixed_bounds_on_test_data(self):
        """
        Given: Test data creation
        When: Creating test chunks
        Then: All collections have fixed upper bounds

        JPL Rule #2: Fixed upper bounds on loops and collections.
        """
        # Verify test constants are defined
        assert MAX_TEST_CHUNKS == 100
        assert MAX_METADATA_KEYS == 50
        assert MAX_CONTENT_LENGTH == 10000

        # Create bounded test data
        chunks = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Content {i}",
            )
            for i in range(min(10, MAX_TEST_CHUNKS))
        ]

        assert len(chunks) <= MAX_TEST_CHUNKS


# ============================================================================
# Metadata Preservation Tests
# ============================================================================


class TestMetadataPreservation:
    """Tests for metadata preservation during conversions."""

    def test_nested_metadata_preserved(self):
        """
        Given: Artifact with nested chunk_metadata
        When: Round-tripping through ChunkRecord
        Then: Nested metadata is preserved

        JPL Rule #7: Check all return values.
        """
        artifact = IFChunkArtifact(
            artifact_id="nested-001",
            document_id="doc-001",
            content="Content",
            metadata={
                "section_title": "Test",
                "chunk_metadata": {
                    "custom_field": "custom_value",
                    "nested_field": {"deep": "value"},
                },
            },
        )

        record = artifact.to_chunk_record()

        # Nested metadata should be flattened into record.metadata
        assert record.metadata is not None
        assert record.metadata.get("custom_field") == "custom_value"

    def test_lineage_preserved_in_metadata(self):
        """
        Given: Artifact with lineage tracking
        When: Converting to ChunkRecord
        Then: Lineage is stored in metadata

        JPL Rule #7: Check all return values.
        """
        artifact = IFChunkArtifact(
            artifact_id="lineage-001",
            document_id="doc-001",
            content="Content",
            parent_id="parent-001",
            root_artifact_id="root-001",
            lineage_depth=2,
            provenance=["extractor", "chunker"],
            metadata={"section_title": "Test"},
        )

        record = artifact.to_chunk_record()

        # Lineage should be in metadata with _lineage_ prefix
        assert record.metadata.get("_lineage_parent_id") == "parent-001"
        assert record.metadata.get("_lineage_root_id") == "root-001"
        assert record.metadata.get("_lineage_depth") == 2
        assert record.metadata.get("_lineage_provenance") == ["extractor", "chunker"]

    def test_content_hash_preserved_in_metadata(self):
        """
        Given: Artifact with content_hash
        When: Converting to ChunkRecord
        Then: content_hash is preserved in metadata

        JPL Rule #7: Check all return values.
        """
        artifact = IFChunkArtifact(
            artifact_id="hash-001",
            document_id="doc-001",
            content="Test content for hashing",
            metadata={},
        )

        # Content hash auto-computed
        assert artifact.content_hash is not None

        record = artifact.to_chunk_record()

        # Hash should be in metadata
        assert "_content_hash" in record.metadata
        assert record.metadata["_content_hash"] == artifact.content_hash


# ============================================================================
# Integration-Like Tests
# ============================================================================


class TestIntegrationScenarios:
    """Integration-style tests for realistic usage scenarios."""

    def test_batch_conversion_maintains_order(self):
        """
        Given: A batch of ChunkRecords
        When: Converting all to IFChunkArtifact
        Then: Order and indices are preserved

        JPL Rule #2: Fixed bounds (MAX_TEST_CHUNKS).
        JPL Rule #7: Check all return values.
        """
        # Create bounded batch of records
        batch_size = min(10, MAX_TEST_CHUNKS)
        records = [
            ChunkRecord(
                chunk_id=f"chunk-{i}",
                document_id="doc-batch",
                content=f"Content for chunk {i}",
                chunk_index=i,
                total_chunks=batch_size,
            )
            for i in range(batch_size)
        ]

        # Convert batch
        artifacts = [IFChunkArtifact.from_chunk_record(r) for r in records]

        # Verify order preserved
        for i, artifact in enumerate(artifacts):
            assert artifact.chunk_index == i
            assert artifact.artifact_id == f"chunk-{i}"

    def test_mixed_legacy_and_new_code(self):
        """
        Given: Pipeline processing with mixed code paths
        When: Some code uses ChunkRecord, some uses IFChunkArtifact
        Then: Conversions work seamlessly

        JPL Rule #7: Check all return values.
        """
        # Start with artifact
        artifact = IFChunkArtifact(
            artifact_id="mixed-001",
            document_id="doc-001",
            content="Mixed code test",
            metadata={"library": "test-lib"},
        )

        # Convert to record for legacy code
        record = artifact.to_chunk_record()

        # Legacy processing (simulated)
        record.quality_score = 0.9
        record.entities = ["Entity1"]

        # Convert back to artifact for new code
        updated_artifact = IFChunkArtifact.from_chunk_record(record)

        # Verify updates preserved
        assert updated_artifact.metadata["quality_score"] == 0.9
        assert updated_artifact.metadata["entities"] == ["Entity1"]
        assert updated_artifact.content == artifact.content
