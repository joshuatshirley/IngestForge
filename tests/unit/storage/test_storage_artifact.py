"""
Storage Layer Artifact Integration Tests.

GWT-style tests verifying g implementation:
- Storage layer accepts IFChunkArtifact
- Artifact metadata (lineage) preserved
- Backward compatibility with ChunkRecord
- Mixed list handling
"""

import warnings
from pathlib import Path
from typing import List

import pytest

from ingestforge.core.pipeline.artifacts import IFChunkArtifact
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import normalize_to_chunk_record, ChunkInput


# --- Fixtures ---


@pytest.fixture
def sample_artifact() -> IFChunkArtifact:
    """Create a sample IFChunkArtifact."""
    return IFChunkArtifact(
        artifact_id="art-001",
        document_id="doc-001",
        content="Sample artifact content for storage.",
        chunk_index=0,
        total_chunks=1,
        parent_id="text-001",
        root_artifact_id="file-001",
        lineage_depth=2,
        provenance=["pdf-extractor", "semantic-chunker"],
        metadata={
            "section_title": "Test Section",
            "chunk_type": "content",
            "source_file": "test.pdf",
            "word_count": 5,
            "library": "test-library",
        },
    )


@pytest.fixture
def sample_artifact_list() -> List[IFChunkArtifact]:
    """Create a list of sample IFChunkArtifacts."""
    return [
        IFChunkArtifact(
            artifact_id=f"art-{i:03d}",
            document_id="doc-001",
            content=f"Content for chunk {i}.",
            chunk_index=i,
            total_chunks=3,
            parent_id="text-001",
            root_artifact_id="file-001",
            lineage_depth=2,
            provenance=["extractor", "chunker"],
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_chunk_record() -> ChunkRecord:
    """Create a sample ChunkRecord."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Sample chunk record content.",
            chunk_index=0,
            total_chunks=1,
        )


@pytest.fixture
def mixed_list(
    sample_artifact: IFChunkArtifact,
    sample_chunk_record: ChunkRecord,
) -> List[ChunkInput]:
    """Create a mixed list of artifacts and chunk records."""
    return [sample_artifact, sample_chunk_record]


# --- GWT Scenario 1: Artifact Stored via add_chunk ---


class TestArtifactStoredViaAddChunk:
    """Tests for storing single artifacts."""

    def test_normalize_converts_artifact_to_chunk_record(
        self,
        sample_artifact: IFChunkArtifact,
    ) -> None:
        """Given an IFChunkArtifact, When normalize_to_chunk_record called,
        Then a valid ChunkRecord is returned."""
        result = normalize_to_chunk_record(sample_artifact)

        assert isinstance(result, ChunkRecord)
        assert result.chunk_id == "art-001"
        assert result.document_id == "doc-001"
        assert result.content == "Sample artifact content for storage."

    def test_normalize_passes_through_chunk_record(
        self,
        sample_chunk_record: ChunkRecord,
    ) -> None:
        """Given a ChunkRecord, When normalize_to_chunk_record called,
        Then the same ChunkRecord is returned."""
        result = normalize_to_chunk_record(sample_chunk_record)

        assert result is sample_chunk_record

    def test_normalize_preserves_artifact_metadata(
        self,
        sample_artifact: IFChunkArtifact,
    ) -> None:
        """Given artifact with metadata, When normalized,
        Then metadata fields are preserved."""
        result = normalize_to_chunk_record(sample_artifact)

        assert result.section_title == "Test Section"
        assert result.chunk_type == "content"
        assert result.source_file == "test.pdf"
        assert result.library == "test-library"


# --- GWT Scenario 2: Lineage Preserved in Storage ---


class TestLineagePreservedInStorage:
    """Tests for lineage preservation during storage."""

    def test_lineage_fields_in_converted_record(
        self,
        sample_artifact: IFChunkArtifact,
    ) -> None:
        """Given artifact with lineage, When converted,
        Then lineage info is in metadata."""
        result = normalize_to_chunk_record(sample_artifact)

        # Lineage preserved in ChunkRecord.metadata
        assert result.metadata.get("_lineage_parent_id") == "text-001"
        assert result.metadata.get("_lineage_root_id") == "file-001"
        assert result.metadata.get("_lineage_depth") == 2

    def test_provenance_preserved_in_converted_record(
        self,
        sample_artifact: IFChunkArtifact,
    ) -> None:
        """Given artifact with provenance, When converted,
        Then provenance is in metadata."""
        result = normalize_to_chunk_record(sample_artifact)

        assert "_lineage_provenance" in result.metadata
        assert "pdf-extractor" in result.metadata["_lineage_provenance"]
        assert "semantic-chunker" in result.metadata["_lineage_provenance"]

    def test_content_hash_preserved(
        self,
        sample_artifact: IFChunkArtifact,
    ) -> None:
        """Given artifact with content_hash, When converted,
        Then hash is in metadata."""
        result = normalize_to_chunk_record(sample_artifact)

        assert "_content_hash" in result.metadata
        assert len(result.metadata["_content_hash"]) == 64  # SHA-256


# --- GWT Scenario 3: List of Artifacts Stored ---


class TestListOfArtifactsStored:
    """Tests for storing lists of artifacts."""

    def test_all_artifacts_normalized(
        self,
        sample_artifact_list: List[IFChunkArtifact],
    ) -> None:
        """Given list of artifacts, When all normalized,
        Then all become ChunkRecords."""
        results = [normalize_to_chunk_record(a) for a in sample_artifact_list]

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, ChunkRecord)
            assert result.chunk_id == f"art-{i:03d}"


# --- GWT Scenario 4: Mixed List Handled ---


class TestMixedListHandled:
    """Tests for handling mixed lists."""

    def test_mixed_list_all_normalized(
        self,
        mixed_list: List[ChunkInput],
    ) -> None:
        """Given mixed list, When all normalized,
        Then all become ChunkRecords."""
        results = [normalize_to_chunk_record(item) for item in mixed_list]

        assert len(results) == 2
        assert all(isinstance(r, ChunkRecord) for r in results)

    def test_mixed_list_preserves_identities(
        self,
        mixed_list: List[ChunkInput],
    ) -> None:
        """Given mixed list, When normalized,
        Then IDs preserved correctly."""
        results = [normalize_to_chunk_record(item) for item in mixed_list]

        ids = [r.chunk_id for r in results]
        assert "art-001" in ids  # Artifact ID
        assert "chunk-001" in ids  # ChunkRecord ID


# --- GWT Scenario 5: ChunkRecord Unchanged (Backward Compatibility) ---


class TestChunkRecordUnchanged:
    """Tests for backward compatibility with ChunkRecord."""

    def test_chunk_record_direct_passthrough(
        self,
        sample_chunk_record: ChunkRecord,
    ) -> None:
        """Given ChunkRecord, When normalized,
        Then same object returned."""
        result = normalize_to_chunk_record(sample_chunk_record)

        assert result is sample_chunk_record

    def test_chunk_record_fields_unchanged(
        self,
        sample_chunk_record: ChunkRecord,
    ) -> None:
        """Given ChunkRecord, When normalized,
        Then all fields match."""
        result = normalize_to_chunk_record(sample_chunk_record)

        assert result.chunk_id == sample_chunk_record.chunk_id
        assert result.document_id == sample_chunk_record.document_id
        assert result.content == sample_chunk_record.content


# --- Edge Cases ---


class TestStorageEdgeCases:
    """Edge case tests for storage artifact support."""

    def test_invalid_type_raises_type_error(self) -> None:
        """Given invalid type, When normalized,
        Then TypeError raised."""
        with pytest.raises(TypeError) as exc_info:
            normalize_to_chunk_record("not a chunk")

        assert "Expected ChunkRecord or IFChunkArtifact" in str(exc_info.value)

    def test_artifact_without_metadata_handled(self) -> None:
        """Given artifact with empty metadata, When normalized,
        Then conversion succeeds."""
        artifact = IFChunkArtifact(
            artifact_id="minimal-001",
            document_id="doc-001",
            content="Minimal content.",
        )

        result = normalize_to_chunk_record(artifact)

        assert isinstance(result, ChunkRecord)
        assert result.chunk_id == "minimal-001"


# --- JSONL Repository Integration Tests ---


class TestJSONLRepositoryArtifacts:
    """Tests for JSONL repository artifact support."""

    def test_jsonl_add_chunk_accepts_artifact(
        self,
        sample_artifact: IFChunkArtifact,
        tmp_path: Path,
    ) -> None:
        """Given JSONL repository, When artifact added,
        Then storage succeeds."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        result = repo.add_chunk(sample_artifact)

        assert result is True
        assert repo.count() == 1

    def test_jsonl_add_chunks_accepts_artifact_list(
        self,
        sample_artifact_list: List[IFChunkArtifact],
        tmp_path: Path,
    ) -> None:
        """Given JSONL repository, When artifact list added,
        Then all stored."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        count = repo.add_chunks(sample_artifact_list)

        assert count == 3
        assert repo.count() == 3

    def test_jsonl_add_chunks_accepts_mixed_list(
        self,
        mixed_list: List[ChunkInput],
        tmp_path: Path,
    ) -> None:
        """Given JSONL repository, When mixed list added,
        Then all stored."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        count = repo.add_chunks(mixed_list)

        assert count == 2
        assert repo.count() == 2

    def test_jsonl_retrieves_stored_artifact_content(
        self,
        sample_artifact: IFChunkArtifact,
        tmp_path: Path,
    ) -> None:
        """Given stored artifact, When retrieved,
        Then content matches."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        repo.add_chunk(sample_artifact)

        retrieved = repo.get_chunk("art-001")

        assert retrieved is not None
        assert retrieved.content == sample_artifact.content
        assert retrieved.document_id == sample_artifact.document_id


# --- JPL Compliance Tests ---


class TestJPLComplianceStorage:
    """JPL Power of Ten compliance tests for storage layer."""

    def test_normalize_function_under_60_lines(self) -> None:
        """Given normalize_to_chunk_record, When lines counted,
        Then count < 60."""
        import inspect
        from ingestforge.storage.base import normalize_to_chunk_record

        source = inspect.getsource(normalize_to_chunk_record)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"

    def test_normalize_has_return_type(self) -> None:
        """Given normalize_to_chunk_record, When annotations checked,
        Then return type is present."""
        from ingestforge.storage.base import normalize_to_chunk_record

        annotations = normalize_to_chunk_record.__annotations__
        assert "return" in annotations


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_artifact_stored_covered(self) -> None:
        """GWT Scenario 1 (Artifact Stored) is tested."""
        assert hasattr(
            TestArtifactStoredViaAddChunk,
            "test_normalize_converts_artifact_to_chunk_record",
        )

    def test_scenario_2_lineage_preserved_covered(self) -> None:
        """GWT Scenario 2 (Lineage Preserved) is tested."""
        assert hasattr(
            TestLineagePreservedInStorage, "test_lineage_fields_in_converted_record"
        )

    def test_scenario_3_list_stored_covered(self) -> None:
        """GWT Scenario 3 (List Stored) is tested."""
        assert hasattr(TestListOfArtifactsStored, "test_all_artifacts_normalized")

    def test_scenario_4_mixed_list_covered(self) -> None:
        """GWT Scenario 4 (Mixed List) is tested."""
        assert hasattr(TestMixedListHandled, "test_mixed_list_all_normalized")

    def test_scenario_5_backward_compat_covered(self) -> None:
        """GWT Scenario 5 (Backward Compatibility) is tested."""
        assert hasattr(TestChunkRecordUnchanged, "test_chunk_record_direct_passthrough")
