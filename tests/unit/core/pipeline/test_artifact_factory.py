"""
Unit tests for Artifact Factory.

a: Create ArtifactFactory
Tests all GWT scenarios and JPL rule compliance.
"""

import tempfile
from pathlib import Path
from typing import List, get_type_hints
from dataclasses import dataclass, field

import pytest

from ingestforge.core.pipeline.artifact_factory import (
    ArtifactFactory,
    text_artifact,
    file_artifact,
    chunk_artifact,
    MAX_BATCH_CONVERSION,
    MAX_CONTENT_SIZE,
)
from ingestforge.core.pipeline.artifacts import (
    IFTextArtifact,
    IFChunkArtifact,
    IFFileArtifact,
)


# =============================================================================
# Mock ChunkRecord for testing (avoid import cycle)
# =============================================================================


@dataclass
class MockChunkRecord:
    """Mock ChunkRecord for testing without importing semantic_chunker."""

    chunk_id: str
    document_id: str
    content: str
    section_title: str = ""
    chunk_type: str = "content"
    source_file: str = ""
    word_count: int = 0
    char_count: int = 0
    section_hierarchy: List[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1
    page_start: int = None
    page_end: int = None
    library: str = "default"
    source_location: str = None
    ingested_at: str = None
    is_read: bool = False
    tags: List[str] = field(default_factory=list)
    author_id: str = None
    author_name: str = None
    embedding: List[float] = None
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    quality_score: float = 0.0


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a mock PDF file for testing."""
    file_path = temp_dir / "document.pdf"
    file_path.write_bytes(b"%PDF-1.4 mock content")
    return file_path


@pytest.fixture
def sample_record():
    """Create a sample ChunkRecord."""
    return MockChunkRecord(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="This is test content for the chunk.",
        section_title="Introduction",
        chunk_type="content",
        source_file="test.pdf",
        word_count=7,
        char_count=36,
        chunk_index=0,
        total_chunks=5,
        library="test-library",
    )


@pytest.fixture
def sample_records():
    """Create multiple sample ChunkRecords."""
    return [
        MockChunkRecord(
            chunk_id=f"chunk-{i:03d}",
            document_id="doc-001",
            content=f"Content for chunk {i}",
            chunk_index=i,
            total_chunks=5,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_dict():
    """Create a sample chunk dictionary."""
    return {
        "chunk_id": "dict-chunk-001",
        "document_id": "doc-002",
        "content": "Dictionary content here.",
        "chunk_index": 0,
        "total_chunks": 3,
        "section_title": "Section A",
        "custom_field": "custom_value",
    }


@pytest.fixture
def parent_artifact():
    """Create a parent artifact for lineage testing."""
    return IFTextArtifact(
        artifact_id="parent-001",
        content="Parent content",
        provenance=["extractor-1"],
    )


# =============================================================================
# GWT Scenario 1: Create IFTextArtifact from String
# =============================================================================


class TestTextFromString:
    """Tests for GWT Scenario 1: Create IFTextArtifact from String."""

    def test_create_text_artifact_from_string(self):
        """Given raw text, when text_from_string called, then IFTextArtifact created."""
        content = "This is some test content."
        artifact = ArtifactFactory.text_from_string(content)

        assert isinstance(artifact, IFTextArtifact)
        assert artifact.content == content

    def test_text_artifact_has_content_hash(self):
        """Given text, when created, then content hash is computed."""
        content = "Hash this content"
        artifact = ArtifactFactory.text_from_string(content)

        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64  # SHA-256 hex

    def test_text_artifact_with_source_path(self):
        """Given source path, when created, then path in metadata."""
        artifact = ArtifactFactory.text_from_string(
            content="Content", source_path="/path/to/source.txt"
        )

        assert artifact.metadata["source_path"] == "/path/to/source.txt"

    def test_text_artifact_with_metadata(self):
        """Given metadata, when created, then metadata preserved."""
        metadata = {"author": "Test", "version": "1.0"}
        artifact = ArtifactFactory.text_from_string(
            content="Content", metadata=metadata
        )

        assert artifact.metadata["author"] == "Test"
        assert artifact.metadata["version"] == "1.0"

    def test_text_artifact_with_custom_id(self):
        """Given custom ID, when created, then ID used."""
        artifact = ArtifactFactory.text_from_string(
            content="Content", artifact_id="custom-id-123"
        )

        assert artifact.artifact_id == "custom-id-123"

    def test_convenience_text_artifact_function(self):
        """Given convenience function, when called, then works correctly."""
        artifact = text_artifact(
            "Test content", source_path="/test.txt", custom="value"
        )

        assert isinstance(artifact, IFTextArtifact)
        assert artifact.content == "Test content"
        assert artifact.metadata["source_path"] == "/test.txt"
        assert artifact.metadata["custom"] == "value"


# =============================================================================
# GWT Scenario 2: Create IFFileArtifact from Path
# =============================================================================


class TestFileFromPath:
    """Tests for GWT Scenario 2: Create IFFileArtifact from Path."""

    def test_create_file_artifact_from_path(self, sample_file):
        """Given file path, when file_from_path called, then IFFileArtifact created."""
        artifact = ArtifactFactory.file_from_path(sample_file)

        assert isinstance(artifact, IFFileArtifact)
        assert artifact.file_path == sample_file.absolute()

    def test_file_artifact_has_mime_type(self, sample_file):
        """Given file, when created, then mime type detected."""
        artifact = ArtifactFactory.file_from_path(sample_file)

        assert artifact.mime_type == "text/plain"

    def test_file_artifact_pdf_mime_type(self, sample_pdf_file):
        """Given PDF file, when created, then PDF mime type."""
        artifact = ArtifactFactory.file_from_path(sample_pdf_file)

        assert artifact.mime_type == "application/pdf"

    def test_file_artifact_has_hash(self, sample_file):
        """Given existing file, when created, then hash computed."""
        artifact = ArtifactFactory.file_from_path(sample_file)

        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64

    def test_file_artifact_with_string_path(self, sample_file):
        """Given string path, when created, then converted to Path."""
        artifact = ArtifactFactory.file_from_path(str(sample_file))

        assert isinstance(artifact.file_path, Path)

    def test_file_artifact_metadata(self, sample_file):
        """Given metadata, when created, then metadata preserved."""
        artifact = ArtifactFactory.file_from_path(
            sample_file, metadata={"source": "upload"}
        )

        assert artifact.metadata["source"] == "upload"
        assert artifact.metadata["file_name"] == "sample.txt"

    def test_convenience_file_artifact_function(self, sample_file):
        """Given convenience function, when called, then works correctly."""
        artifact = file_artifact(sample_file, custom="value")

        assert isinstance(artifact, IFFileArtifact)
        assert artifact.metadata["custom"] == "value"


# =============================================================================
# GWT Scenario 3: Create IFChunkArtifact from ChunkRecord
# =============================================================================


class TestChunkFromRecord:
    """Tests for GWT Scenario 3: Create IFChunkArtifact from ChunkRecord."""

    def test_create_chunk_from_record(self, sample_record):
        """Given ChunkRecord, when chunk_from_record called, then IFChunkArtifact created."""
        artifact = ArtifactFactory.chunk_from_record(sample_record)

        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.content == sample_record.content

    def test_chunk_preserves_document_id(self, sample_record):
        """Given record, when converted, then document_id preserved."""
        artifact = ArtifactFactory.chunk_from_record(sample_record)

        assert artifact.document_id == sample_record.document_id

    def test_chunk_preserves_index(self, sample_record):
        """Given record, when converted, then index preserved."""
        artifact = ArtifactFactory.chunk_from_record(sample_record)

        assert artifact.chunk_index == sample_record.chunk_index
        assert artifact.total_chunks == sample_record.total_chunks

    def test_chunk_preserves_metadata_fields(self, sample_record):
        """Given record, when converted, then metadata fields preserved."""
        artifact = ArtifactFactory.chunk_from_record(sample_record)

        assert artifact.metadata["section_title"] == sample_record.section_title
        assert artifact.metadata["chunk_type"] == sample_record.chunk_type
        assert artifact.metadata["source_file"] == sample_record.source_file
        assert artifact.metadata["word_count"] == sample_record.word_count
        assert artifact.metadata["library"] == sample_record.library

    def test_chunk_with_parent_lineage(self, sample_record, parent_artifact):
        """Given parent, when converted, then lineage tracked."""
        artifact = ArtifactFactory.chunk_from_record(
            sample_record, parent=parent_artifact
        )

        assert artifact.parent_id == parent_artifact.artifact_id
        assert artifact.root_artifact_id == parent_artifact.artifact_id
        assert artifact.lineage_depth == 1
        assert "artifact-factory" in artifact.provenance

    def test_chunk_uses_record_chunk_id(self, sample_record):
        """Given record with ID, when converted, then ID used."""
        artifact = ArtifactFactory.chunk_from_record(sample_record)

        assert artifact.artifact_id == sample_record.chunk_id


# =============================================================================
# GWT Scenario 4: Create IFChunkArtifact from Dictionary
# =============================================================================


class TestChunkFromDict:
    """Tests for GWT Scenario 4: Create IFChunkArtifact from Dictionary."""

    def test_create_chunk_from_dict(self, sample_dict):
        """Given dictionary, when chunk_from_dict called, then IFChunkArtifact created."""
        artifact = ArtifactFactory.chunk_from_dict(sample_dict)

        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.content == sample_dict["content"]

    def test_chunk_dict_document_id(self, sample_dict):
        """Given dict, when converted, then document_id extracted."""
        artifact = ArtifactFactory.chunk_from_dict(sample_dict)

        assert artifact.document_id == sample_dict["document_id"]

    def test_chunk_dict_with_doc_id_alias(self):
        """Given dict with doc_id, when converted, then used."""
        data = {"content": "Test", "doc_id": "doc-alias"}
        artifact = ArtifactFactory.chunk_from_dict(data)

        assert artifact.document_id == "doc-alias"

    def test_chunk_dict_custom_fields_in_metadata(self, sample_dict):
        """Given dict with custom fields, when converted, then in metadata."""
        artifact = ArtifactFactory.chunk_from_dict(sample_dict)

        assert artifact.metadata["custom_field"] == "custom_value"
        assert artifact.metadata["section_title"] == "Section A"

    def test_chunk_dict_with_parent(self, sample_dict, parent_artifact):
        """Given dict and parent, when converted, then lineage tracked."""
        artifact = ArtifactFactory.chunk_from_dict(sample_dict, parent=parent_artifact)

        assert artifact.parent_id == parent_artifact.artifact_id
        assert artifact.lineage_depth == 1

    def test_convenience_chunk_artifact_function(self, parent_artifact):
        """Given convenience function, when called, then works correctly."""
        artifact = chunk_artifact(
            content="Test chunk",
            document_id="doc-123",
            chunk_index=2,
            total_chunks=10,
            parent=parent_artifact,
            custom="value",
        )

        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.content == "Test chunk"
        assert artifact.document_id == "doc-123"
        assert artifact.chunk_index == 2
        assert artifact.metadata["custom"] == "value"
        assert artifact.parent_id == parent_artifact.artifact_id


# =============================================================================
# GWT Scenario 5: Batch Conversion
# =============================================================================


class TestBatchConversion:
    """Tests for GWT Scenario 5: Batch Conversion."""

    def test_batch_convert_records(self, sample_records):
        """Given list of records, when chunks_from_records called, then all converted."""
        artifacts = ArtifactFactory.chunks_from_records(sample_records)

        assert len(artifacts) == len(sample_records)
        for artifact in artifacts:
            assert isinstance(artifact, IFChunkArtifact)

    def test_batch_convert_with_parent(self, sample_records, parent_artifact):
        """Given batch and parent, when converted, then all have lineage."""
        artifacts = ArtifactFactory.chunks_from_records(
            sample_records, parent=parent_artifact
        )

        for artifact in artifacts:
            assert artifact.parent_id == parent_artifact.artifact_id

    def test_batch_convert_dicts(self):
        """Given list of dicts, when chunks_from_dicts called, then all converted."""
        dicts = [
            {"content": f"Content {i}", "document_id": "doc-1", "chunk_index": i}
            for i in range(5)
        ]

        artifacts = ArtifactFactory.chunks_from_dicts(dicts)

        assert len(artifacts) == 5
        for i, artifact in enumerate(artifacts):
            assert artifact.chunk_index == i

    def test_batch_preserves_order(self, sample_records):
        """Given ordered records, when converted, then order preserved."""
        artifacts = ArtifactFactory.chunks_from_records(sample_records)

        for i, artifact in enumerate(artifacts):
            assert artifact.chunk_index == i


# =============================================================================
# JPL Rule #2: Fixed Upper Bounds
# =============================================================================


class TestJPLRule2FixedBounds:
    """Tests for JPL Rule #2: Fixed upper bounds."""

    def test_max_batch_conversion_constant(self):
        """Given constant, MAX_BATCH_CONVERSION is defined."""
        assert MAX_BATCH_CONVERSION == 1000

    def test_max_content_size_constant(self):
        """Given constant, MAX_CONTENT_SIZE is defined."""
        assert MAX_CONTENT_SIZE == 10_000_000

    def test_batch_size_limit_enforced_records(self):
        """Given oversized batch, when converted, then error raised."""
        records = [
            MockChunkRecord(
                chunk_id=f"chunk-{i}", document_id="doc", content=f"Content {i}"
            )
            for i in range(MAX_BATCH_CONVERSION + 1)
        ]

        with pytest.raises(ValueError, match="exceeds maximum"):
            ArtifactFactory.chunks_from_records(records)

    def test_batch_size_limit_enforced_dicts(self):
        """Given oversized dict batch, when converted, then error raised."""
        dicts = [
            {"content": f"Content {i}", "document_id": "doc"}
            for i in range(MAX_BATCH_CONVERSION + 1)
        ]

        with pytest.raises(ValueError, match="exceeds maximum"):
            ArtifactFactory.chunks_from_dicts(dicts)


# =============================================================================
# JPL Rule #7: Check Return Values
# =============================================================================


class TestJPLRule7ReturnValues:
    """Tests for JPL Rule #7: Check all return values."""

    def test_text_from_string_returns_artifact(self):
        """Given text_from_string call, returns IFTextArtifact."""
        result = ArtifactFactory.text_from_string("content")
        assert isinstance(result, IFTextArtifact)

    def test_file_from_path_returns_artifact(self, sample_file):
        """Given file_from_path call, returns IFFileArtifact."""
        result = ArtifactFactory.file_from_path(sample_file)
        assert isinstance(result, IFFileArtifact)

    def test_chunk_from_record_returns_artifact(self, sample_record):
        """Given chunk_from_record call, returns IFChunkArtifact."""
        result = ArtifactFactory.chunk_from_record(sample_record)
        assert isinstance(result, IFChunkArtifact)

    def test_chunk_from_dict_returns_artifact(self, sample_dict):
        """Given chunk_from_dict call, returns IFChunkArtifact."""
        result = ArtifactFactory.chunk_from_dict(sample_dict)
        assert isinstance(result, IFChunkArtifact)

    def test_chunks_from_records_returns_list(self, sample_records):
        """Given chunks_from_records call, returns list."""
        result = ArtifactFactory.chunks_from_records(sample_records)
        assert isinstance(result, list)


# =============================================================================
# JPL Rule #9: Complete Type Hints
# =============================================================================


class TestJPLRule9TypeHints:
    """Tests for JPL Rule #9: Complete type hints."""

    def test_text_from_string_has_type_hints(self):
        """Given text_from_string, has return type hint."""
        hints = get_type_hints(ArtifactFactory.text_from_string)
        assert "return" in hints

    def test_file_from_path_has_type_hints(self):
        """Given file_from_path, has return type hint."""
        hints = get_type_hints(ArtifactFactory.file_from_path)
        assert "return" in hints

    def test_chunk_from_record_has_annotations(self):
        """Given chunk_from_record, has type annotations."""
        # Uses TYPE_CHECKING import, so check __annotations__ instead
        annotations = ArtifactFactory.chunk_from_record.__annotations__
        assert "return" in annotations

    def test_chunks_from_records_has_annotations(self):
        """Given chunks_from_records, has type annotations."""
        # Uses TYPE_CHECKING import, so check __annotations__ instead
        annotations = ArtifactFactory.chunks_from_records.__annotations__
        assert "return" in annotations


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content_text_artifact(self):
        """Given empty string, when created, then valid artifact."""
        artifact = ArtifactFactory.text_from_string("")

        assert artifact.content == ""
        assert artifact.content_hash is not None

    def test_empty_batch_records(self):
        """Given empty list, when batch converted, then empty list returned."""
        artifacts = ArtifactFactory.chunks_from_records([])

        assert artifacts == []

    def test_empty_batch_dicts(self):
        """Given empty list, when batch converted, then empty list returned."""
        artifacts = ArtifactFactory.chunks_from_dicts([])

        assert artifacts == []

    def test_dict_with_missing_fields(self):
        """Given minimal dict, when converted, then defaults used."""
        data = {"content": "Just content"}
        artifact = ArtifactFactory.chunk_from_dict(data)

        assert artifact.content == "Just content"
        assert artifact.document_id == ""
        assert artifact.chunk_index == 0

    def test_nonexistent_file(self, temp_dir):
        """Given nonexistent file, when created, then no hash computed."""
        artifact = ArtifactFactory.file_from_path(
            temp_dir / "nonexistent.txt", compute_hash=True
        )

        assert artifact.content_hash is None

    def test_record_with_optional_fields(self):
        """Given record with all fields, when converted, then all in metadata."""
        record = MockChunkRecord(
            chunk_id="full-001",
            document_id="doc",
            content="Full content",
            section_hierarchy=["Ch1", "Sec1"],
            page_start=1,
            page_end=5,
            tags=["tag1", "tag2"],
            author_id="author-1",
            author_name="John Doe",
            entities=["Entity1"],
            concepts=["Concept1"],
            quality_score=0.95,
        )

        artifact = ArtifactFactory.chunk_from_record(record)

        assert artifact.metadata["section_hierarchy"] == ["Ch1", "Sec1"]
        assert artifact.metadata["page_start"] == 1
        assert artifact.metadata["tags"] == ["tag1", "tag2"]
        assert artifact.metadata["author_name"] == "John Doe"
        assert artifact.metadata["quality_score"] == 0.95


# =============================================================================
# GWT Scenario Completeness
# =============================================================================


class TestGWTScenarioCompleteness:
    """Meta-tests to ensure all GWT scenarios are covered."""

    def test_scenario_1_text_from_string_covered(self):
        """Verify Scenario 1 tests exist."""
        test_methods = [m for m in dir(TestTextFromString) if m.startswith("test_")]
        assert len(test_methods) >= 5

    def test_scenario_2_file_from_path_covered(self):
        """Verify Scenario 2 tests exist."""
        test_methods = [m for m in dir(TestFileFromPath) if m.startswith("test_")]
        assert len(test_methods) >= 5

    def test_scenario_3_chunk_from_record_covered(self):
        """Verify Scenario 3 tests exist."""
        test_methods = [m for m in dir(TestChunkFromRecord) if m.startswith("test_")]
        assert len(test_methods) >= 5

    def test_scenario_4_chunk_from_dict_covered(self):
        """Verify Scenario 4 tests exist."""
        test_methods = [m for m in dir(TestChunkFromDict) if m.startswith("test_")]
        assert len(test_methods) >= 5

    def test_scenario_5_batch_conversion_covered(self):
        """Verify Scenario 5 tests exist."""
        test_methods = [m for m in dir(TestBatchConversion) if m.startswith("test_")]
        assert len(test_methods) >= 4
