"""
Comprehensive Error Handling Tests for Artifact System.

GWT-style tests verifying robust error handling and edge cases
for the artifact creation and conversion pipeline.

Tests cover:
- Invalid input handling
- Boundary conditions
- Type validation
- Null/empty handling
- Conversion failures
"""

import warnings
from pathlib import Path

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    calculate_sha256,
)
from ingestforge.core.pipeline.artifact_factory import (
    ArtifactFactory,
    MAX_BATCH_CONVERSION,
    MAX_CONTENT_SIZE,
)
from ingestforge.chunking.semantic_chunker import SemanticChunker, ChunkRecord


# --- GWT Scenario 1: Empty/Null Input Handling ---


class TestEmptyInputHandling:
    """Tests for handling empty or null inputs."""

    def test_empty_string_produces_valid_artifact(self) -> None:
        """Given empty string, When text_from_string called,
        Then valid artifact with empty content is created."""
        artifact = ArtifactFactory.text_from_string("")

        assert artifact is not None
        assert artifact.content == ""
        assert artifact.content_hash is not None

    def test_empty_dict_produces_artifact_with_defaults(self) -> None:
        """Given empty dict, When chunk_from_dict called,
        Then artifact with default values is created."""
        artifact = ArtifactFactory.chunk_from_dict({})

        assert artifact is not None
        assert artifact.content == ""
        assert artifact.document_id == ""

    def test_empty_list_returns_empty_list(self) -> None:
        """Given empty list, When chunks_from_dicts called,
        Then empty list is returned."""
        result = ArtifactFactory.chunks_from_dicts([])

        assert result == []

    def test_none_metadata_handled_gracefully(self) -> None:
        """Given None metadata, When artifact created,
        Then default empty metadata is used."""
        artifact = ArtifactFactory.text_from_string(
            content="Test",
            metadata=None,
        )

        assert artifact.metadata is not None
        assert isinstance(artifact.metadata, dict)

    def test_none_source_path_handled(self) -> None:
        """Given None source_path, When text_from_string called,
        Then artifact created without source_path in metadata."""
        artifact = ArtifactFactory.text_from_string(
            content="Test",
            source_path=None,
        )

        assert "source_path" not in artifact.metadata

    def test_empty_content_has_valid_hash(self) -> None:
        """Given empty content, When hash calculated,
        Then valid SHA-256 hash returned."""
        hash_result = calculate_sha256("")

        assert hash_result is not None
        assert len(hash_result) == 64


# --- GWT Scenario 2: Boundary Condition Handling ---


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_exactly_max_batch_size_accepted(self) -> None:
        """Given exactly MAX_BATCH_CONVERSION items, When converted,
        Then all are processed."""
        dicts = [
            {"content": f"c{i}", "document_id": "doc"}
            for i in range(MAX_BATCH_CONVERSION)
        ]

        result = ArtifactFactory.chunks_from_dicts(dicts)

        assert len(result) == MAX_BATCH_CONVERSION

    def test_one_over_max_batch_size_rejected(self) -> None:
        """Given MAX_BATCH_CONVERSION + 1 items, When converted,
        Then ValueError raised."""
        dicts = [
            {"content": f"c{i}", "document_id": "doc"}
            for i in range(MAX_BATCH_CONVERSION + 1)
        ]

        with pytest.raises(ValueError, match="exceeds maximum"):
            ArtifactFactory.chunks_from_dicts(dicts)

    def test_exactly_max_content_size_not_truncated(self) -> None:
        """Given exactly MAX_CONTENT_SIZE chars, When artifact created,
        Then content not truncated."""
        content = "x" * MAX_CONTENT_SIZE

        artifact = ArtifactFactory.text_from_string(content)

        assert len(artifact.content) == MAX_CONTENT_SIZE

    def test_one_over_max_content_size_truncated(self) -> None:
        """Given MAX_CONTENT_SIZE + 1 chars, When artifact created,
        Then content truncated to MAX_CONTENT_SIZE."""
        content = "x" * (MAX_CONTENT_SIZE + 1)

        artifact = ArtifactFactory.text_from_string(content)

        assert len(artifact.content) == MAX_CONTENT_SIZE

    def test_single_character_content(self) -> None:
        """Given single character, When artifact created,
        Then valid artifact produced."""
        artifact = ArtifactFactory.text_from_string("x")

        assert artifact.content == "x"
        assert artifact.content_hash is not None

    def test_single_item_batch(self) -> None:
        """Given single-item list, When batch converted,
        Then single artifact returned."""
        dicts = [{"content": "test", "document_id": "doc"}]

        result = ArtifactFactory.chunks_from_dicts(dicts)

        assert len(result) == 1


# --- GWT Scenario 3: Type Coercion and Validation ---


class TestTypeCoercion:
    """Tests for type handling and coercion."""

    def test_numeric_content_coerced_to_string(self) -> None:
        """Given numeric content in dict, When chunk_from_dict called,
        Then content coerced to string."""
        artifact = ArtifactFactory.chunk_from_dict(
            {
                "content": 12345,
                "document_id": "doc",
            }
        )

        assert artifact.content == "12345"

    def test_numeric_document_id_coerced_to_string(self) -> None:
        """Given numeric document_id, When chunk_from_dict called,
        Then document_id coerced to string."""
        artifact = ArtifactFactory.chunk_from_dict(
            {
                "content": "test",
                "document_id": 999,
            }
        )

        assert artifact.document_id == "999"

    def test_string_chunk_index_coerced_to_int(self) -> None:
        """Given string chunk_index, When chunk_from_dict called,
        Then index coerced to int."""
        artifact = ArtifactFactory.chunk_from_dict(
            {
                "content": "test",
                "document_id": "doc",
                "chunk_index": "5",
            }
        )

        assert artifact.chunk_index == 5

    def test_path_coercion_string_to_path(self) -> None:
        """Given string path, When file_from_path called,
        Then Path object created."""
        artifact = ArtifactFactory.file_from_path("/tmp/test.pdf")

        assert isinstance(artifact.file_path, Path)


# --- GWT Scenario 4: Conversion Error Handling ---


class TestConversionErrors:
    """Tests for conversion error scenarios."""

    def test_chunk_record_with_none_fields(self) -> None:
        """Given ChunkRecord with None optional fields, When converted,
        Then artifact handles gracefully."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = ChunkRecord(
                chunk_id="test-001",
                document_id="doc-001",
                content="Test content",
                section_hierarchy=None,
                page_start=None,
                page_end=None,
                source_location=None,
            )

        artifact = IFChunkArtifact.from_chunk_record(record)

        assert artifact is not None
        assert artifact.content == "Test content"

    def test_chunk_record_to_artifact_preserves_all_fields(self) -> None:
        """Given ChunkRecord with all fields, When converted,
        Then all fields preserved in metadata."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = ChunkRecord(
                chunk_id="full-001",
                document_id="doc-full",
                content="Full content",
                section_title="Test Section",
                chunk_type="narrative",
                source_file="/path/to/file.pdf",
                word_count=2,
                char_count=12,
                library="test-lib",
                is_read=True,
                tags=["tag1", "tag2"],
                quality_score=0.95,
            )

        artifact = IFChunkArtifact.from_chunk_record(record)

        assert artifact.metadata.get("section_title") == "Test Section"
        assert artifact.metadata.get("chunk_type") == "narrative"
        assert artifact.metadata.get("library") == "test-lib"

    def test_artifact_to_chunk_record_round_trip(self) -> None:
        """Given IFChunkArtifact, When converted to ChunkRecord and back,
        Then content preserved."""
        original = IFChunkArtifact(
            artifact_id="round-trip-001",
            document_id="doc-001",
            content="Round trip test content",
            chunk_index=3,
            total_chunks=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = original.to_chunk_record()
            restored = IFChunkArtifact.from_chunk_record(record)

        assert restored.content == original.content
        assert restored.document_id == original.document_id


# --- GWT Scenario 5: File System Edge Cases ---


class TestFileSystemEdgeCases:
    """Tests for file system related edge cases."""

    def test_nonexistent_file_path(self, tmp_path: Path) -> None:
        """Given nonexistent file path, When file_from_path called,
        Then artifact created without hash."""
        nonexistent = tmp_path / "does_not_exist.pdf"

        artifact = ArtifactFactory.file_from_path(nonexistent, compute_hash=True)

        assert artifact is not None
        assert artifact.content_hash is None

    def test_existing_file_gets_hash(self, tmp_path: Path) -> None:
        """Given existing file, When file_from_path called,
        Then hash is computed."""
        existing = tmp_path / "exists.txt"
        existing.write_text("test content")

        artifact = ArtifactFactory.file_from_path(existing, compute_hash=True)

        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64

    def test_compute_hash_false_still_computes_via_model(self, tmp_path: Path) -> None:
        """Given compute_hash=False, When file_from_path called,
        Then IFFileArtifact's model_post_init still computes hash for integrity.

        Note: IFFileArtifact intentionally always computes hash for existing
        files to ensure content integrity, overriding the factory parameter.
        """
        existing = tmp_path / "exists.txt"
        existing.write_text("test content")

        artifact = ArtifactFactory.file_from_path(existing, compute_hash=False)

        # Model ensures hash integrity even when factory says skip
        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64

    def test_path_with_spaces_handled(self, tmp_path: Path) -> None:
        """Given path with spaces, When file_from_path called,
        Then artifact created correctly."""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        spaced_file = spaced_dir / "file name.txt"
        spaced_file.write_text("content")

        artifact = ArtifactFactory.file_from_path(spaced_file)

        assert artifact is not None
        assert "file name.txt" in str(artifact.file_path)


# --- GWT Scenario 6: Chunker Error Handling ---


class TestChunkerErrorHandling:
    """Tests for chunker error scenarios."""

    def test_chunker_with_very_short_text(self) -> None:
        """Given very short text, When chunk_to_artifacts called,
        Then single artifact produced."""
        chunker = SemanticChunker(
            max_chunk_size=1000,
            min_chunk_size=100,
            use_embeddings=False,
        )

        result = chunker.chunk_to_artifacts("Hi", document_id="doc")

        assert len(result) == 1

    def test_chunker_with_whitespace_only(self) -> None:
        """Given whitespace-only text, When chunk_to_artifacts called,
        Then result handles gracefully."""
        chunker = SemanticChunker(
            max_chunk_size=1000,
            min_chunk_size=100,
            use_embeddings=False,
        )

        result = chunker.chunk_to_artifacts("   \n\t  ", document_id="doc")

        # Should produce result (possibly single empty/whitespace chunk)
        assert isinstance(result, list)

    def test_chunker_with_unicode_content(self) -> None:
        """Given Unicode content, When chunk_to_artifacts called,
        Then Unicode preserved in artifacts."""
        chunker = SemanticChunker(
            max_chunk_size=500,
            min_chunk_size=50,
            use_embeddings=False,
        )

        unicode_text = "日本語テスト。中文测试。한국어 테스트。"
        result = chunker.chunk_to_artifacts(unicode_text, document_id="doc")

        for artifact in result:
            # Verify Unicode preserved
            assert any(c in artifact.content for c in ["日", "中", "한"])


# --- GWT Scenario 7: Hash Consistency ---


class TestHashConsistency:
    """Tests for content hash consistency."""

    def test_same_content_produces_same_hash(self) -> None:
        """Given same content, When multiple artifacts created,
        Then hashes match."""
        artifact1 = ArtifactFactory.text_from_string("identical content")
        artifact2 = ArtifactFactory.text_from_string("identical content")

        assert artifact1.content_hash == artifact2.content_hash

    def test_different_content_produces_different_hash(self) -> None:
        """Given different content, When artifacts created,
        Then hashes differ."""
        artifact1 = ArtifactFactory.text_from_string("content A")
        artifact2 = ArtifactFactory.text_from_string("content B")

        assert artifact1.content_hash != artifact2.content_hash

    def test_whitespace_difference_affects_hash(self) -> None:
        """Given content differing only in whitespace, When hashed,
        Then hashes differ."""
        artifact1 = ArtifactFactory.text_from_string("hello world")
        artifact2 = ArtifactFactory.text_from_string("hello  world")

        assert artifact1.content_hash != artifact2.content_hash

    def test_calculate_sha256_deterministic(self) -> None:
        """Given same input, When calculate_sha256 called multiple times,
        Then same hash returned."""
        hash1 = calculate_sha256("test data")
        hash2 = calculate_sha256("test data")
        hash3 = calculate_sha256("test data")

        assert hash1 == hash2 == hash3


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all error scenarios are covered."""

    def test_scenario_1_empty_input_covered(self) -> None:
        """GWT Scenario 1 (Empty Input) is tested."""
        assert hasattr(
            TestEmptyInputHandling, "test_empty_string_produces_valid_artifact"
        )

    def test_scenario_2_boundary_covered(self) -> None:
        """GWT Scenario 2 (Boundary Conditions) is tested."""
        assert hasattr(TestBoundaryConditions, "test_exactly_max_batch_size_accepted")

    def test_scenario_3_type_coercion_covered(self) -> None:
        """GWT Scenario 3 (Type Coercion) is tested."""
        assert hasattr(TestTypeCoercion, "test_numeric_content_coerced_to_string")

    def test_scenario_4_conversion_errors_covered(self) -> None:
        """GWT Scenario 4 (Conversion Errors) is tested."""
        assert hasattr(TestConversionErrors, "test_chunk_record_with_none_fields")

    def test_scenario_5_file_system_covered(self) -> None:
        """GWT Scenario 5 (File System) is tested."""
        assert hasattr(TestFileSystemEdgeCases, "test_nonexistent_file_path")

    def test_scenario_6_chunker_errors_covered(self) -> None:
        """GWT Scenario 6 (Chunker Errors) is tested."""
        assert hasattr(TestChunkerErrorHandling, "test_chunker_with_very_short_text")

    def test_scenario_7_hash_consistency_covered(self) -> None:
        """GWT Scenario 7 (Hash Consistency) is tested."""
        assert hasattr(TestHashConsistency, "test_same_content_produces_same_hash")
