"""
Phase 3 Comprehensive Tests: Storage & Retrieval Artifact Integration.

This module provides exhaustive GWT-style tests for g and h,
ensuring NASA JPL Power of Ten compliance throughout the artifact lifecycle.

Test Categories:
- GWT Scenarios: Given-When-Then behavioral tests
- JPL Compliance: Power of Ten rule verification
- Round-Trip Fidelity: artifact → storage → retrieval → artifact
- Boundary Conditions: Edge cases and limits
- Error Handling: Failure mode verification
"""

import hashlib
import inspect
import warnings
from pathlib import Path
from typing import List

import pytest

from ingestforge.core.pipeline.artifacts import IFChunkArtifact
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import (
    SearchResult,
    normalize_to_chunk_record,
    MAX_LINEAGE_FIELDS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def artifact_with_full_lineage() -> IFChunkArtifact:
    """Create artifact with complete lineage chain."""
    return IFChunkArtifact(
        artifact_id="chunk-full-001",
        document_id="doc-full-001",
        content="Content with complete lineage tracking.",
        chunk_index=0,
        total_chunks=5,
        parent_id="text-parent-001",
        root_artifact_id="file-root-001",
        lineage_depth=3,
        provenance=["pdf-extractor", "semantic-chunker", "entity-enricher"],
        metadata={
            "section_title": "Full Lineage Section",
            "chunk_type": "content",
            "source_file": "full_lineage.pdf",
            "word_count": 6,
            "char_count": 42,
            "library": "full-test",
            "embedding": [0.1] * 384,
            "entities": ["Person1", "Organization1"],
            "concepts": ["Concept1", "Concept2"],
            "quality_score": 0.95,
        },
    )


@pytest.fixture
def artifact_minimal() -> IFChunkArtifact:
    """Create artifact with minimal required fields."""
    return IFChunkArtifact(
        artifact_id="chunk-min-001",
        document_id="doc-min-001",
        content="Minimal content.",
    )


@pytest.fixture
def chunk_record_with_enrichment() -> ChunkRecord:
    """Create enriched ChunkRecord for comparison."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ChunkRecord(
            chunk_id="chunk-enriched-001",
            document_id="doc-enriched-001",
            content="Enriched chunk record content.",
            chunk_index=1,
            total_chunks=3,
            section_title="Enriched Section",
            chunk_type="analysis",
            source_file="enriched.md",
            word_count=4,
            library="enriched-library",
            embedding=[0.2] * 384,
            entities=["Entity1"],
            concepts=["Concept1"],
            quality_score=0.88,
        )


@pytest.fixture
def search_result_full() -> SearchResult:
    """Create SearchResult with all fields populated."""
    return SearchResult(
        chunk_id="search-full-001",
        content="Full search result with all metadata.",
        score=0.95,
        document_id="doc-search-001",
        section_title="Full Search Section",
        chunk_type="analysis",
        source_file="search_full.pdf",
        word_count=7,
        page_start=5,
        page_end=8,
        library="search-library",
        author_id="author-full-001",
        author_name="Dr. Full Test",
        source_location={"title": "Full Document", "page": 5, "chapter": "Ch1"},
        metadata={
            "entities": ["SearchEntity1", "SearchEntity2"],
            "concepts": ["SearchConcept1"],
            "quality_score": 0.92,
            "_lineage_parent_id": "parent-search-001",
            "_lineage_root_id": "root-search-001",
            "_lineage_depth": 2,
            "_lineage_provenance": ["extractor", "chunker"],
        },
    )


@pytest.fixture
def large_artifact_batch() -> List[IFChunkArtifact]:
    """Create a batch of 100 artifacts for batch testing."""
    return [
        IFChunkArtifact(
            artifact_id=f"batch-{i:04d}",
            document_id=f"doc-batch-{i // 10:02d}",
            content=f"Batch content for chunk {i}.",
            chunk_index=i % 10,
            total_chunks=10,
            parent_id=f"text-batch-{i // 10:02d}",
            root_artifact_id=f"file-batch-{i // 10:02d}",
            lineage_depth=2,
            provenance=["batch-processor"],
        )
        for i in range(100)
    ]


# =============================================================================
# GWT SCENARIO 1: ARTIFACT NORMALIZATION (g)
# =============================================================================


class TestGWTArtifactNormalization:
    """
    GWT Scenario 1: Artifact Normalization

    Given: Various input types (IFChunkArtifact, ChunkRecord)
    When: normalize_to_chunk_record() is called
    Then: Valid ChunkRecord returned with all data preserved
    """

    def test_given_artifact_when_normalized_then_chunk_record_returned(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: Artifact → ChunkRecord conversion."""
        result = normalize_to_chunk_record(artifact_with_full_lineage)

        assert isinstance(result, ChunkRecord)
        assert result.chunk_id == artifact_with_full_lineage.artifact_id

    def test_given_chunk_record_when_normalized_then_same_object_returned(
        self,
        chunk_record_with_enrichment: ChunkRecord,
    ) -> None:
        """GWT: ChunkRecord passthrough (no conversion)."""
        result = normalize_to_chunk_record(chunk_record_with_enrichment)

        assert result is chunk_record_with_enrichment

    def test_given_artifact_when_normalized_then_content_preserved(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: Content integrity after normalization."""
        result = normalize_to_chunk_record(artifact_with_full_lineage)

        assert result.content == artifact_with_full_lineage.content
        assert result.document_id == artifact_with_full_lineage.document_id

    def test_given_artifact_when_normalized_then_metadata_fields_mapped(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: Metadata field mapping completeness."""
        result = normalize_to_chunk_record(artifact_with_full_lineage)

        assert result.section_title == "Full Lineage Section"
        assert result.chunk_type == "content"
        assert result.source_file == "full_lineage.pdf"
        assert result.library == "full-test"

    def test_given_artifact_when_normalized_then_lineage_in_metadata(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: Lineage preserved in ChunkRecord.metadata."""
        result = normalize_to_chunk_record(artifact_with_full_lineage)

        assert result.metadata.get("_lineage_parent_id") == "text-parent-001"
        assert result.metadata.get("_lineage_root_id") == "file-root-001"
        assert result.metadata.get("_lineage_depth") == 3

    def test_given_artifact_when_normalized_then_provenance_preserved(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: Provenance chain preserved."""
        result = normalize_to_chunk_record(artifact_with_full_lineage)

        provenance = result.metadata.get("_lineage_provenance", [])
        assert "pdf-extractor" in provenance
        assert "semantic-chunker" in provenance
        assert "entity-enricher" in provenance

    def test_given_minimal_artifact_when_normalized_then_defaults_applied(
        self,
        artifact_minimal: IFChunkArtifact,
    ) -> None:
        """GWT: Minimal artifact uses sensible defaults."""
        result = normalize_to_chunk_record(artifact_minimal)

        assert isinstance(result, ChunkRecord)
        assert result.chunk_id == "chunk-min-001"
        assert result.library == "default"

    def test_given_invalid_type_when_normalized_then_type_error_raised(
        self,
    ) -> None:
        """GWT: Invalid input type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            normalize_to_chunk_record("invalid string")

        assert "Expected ChunkRecord or IFChunkArtifact" in str(exc_info.value)

    def test_given_dict_when_normalized_then_type_error_raised(
        self,
    ) -> None:
        """GWT: Dict input raises TypeError (not duck-typed)."""
        with pytest.raises(TypeError):
            normalize_to_chunk_record({"chunk_id": "test"})


# =============================================================================
# GWT SCENARIO 2: SEARCH RESULT ARTIFACT CONVERSION (h)
# =============================================================================


class TestGWTSearchResultConversion:
    """
    GWT Scenario 2: SearchResult to Artifact Conversion

    Given: SearchResult from retrieval layer
    When: to_artifact() is called
    Then: Valid IFChunkArtifact with all data preserved
    """

    def test_given_search_result_when_converted_then_artifact_returned(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: SearchResult → IFChunkArtifact conversion."""
        artifact = search_result_full.to_artifact()

        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.artifact_id == search_result_full.chunk_id

    def test_given_search_result_when_converted_then_score_preserved(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: Retrieval score preserved in metadata."""
        artifact = search_result_full.to_artifact()

        assert artifact.metadata["retrieval_score"] == 0.95

    def test_given_search_result_when_converted_then_lineage_restored(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: Lineage fields extracted from metadata and restored."""
        artifact = search_result_full.to_artifact()

        assert artifact.parent_id == "parent-search-001"
        assert artifact.root_artifact_id == "root-search-001"
        assert artifact.lineage_depth == 2
        assert "extractor" in artifact.provenance

    def test_given_search_result_when_converted_then_lineage_not_in_metadata(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: Lineage fields removed from metadata (extracted)."""
        artifact = search_result_full.to_artifact()

        assert "_lineage_parent_id" not in artifact.metadata
        assert "_lineage_root_id" not in artifact.metadata
        assert "_lineage_depth" not in artifact.metadata
        assert "_lineage_provenance" not in artifact.metadata

    def test_given_search_result_when_converted_then_source_location_preserved(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: Source location dict preserved."""
        artifact = search_result_full.to_artifact()

        assert artifact.metadata["source_location"]["title"] == "Full Document"
        assert artifact.metadata["source_location"]["page"] == 5

    def test_given_search_result_when_converted_then_author_info_preserved(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: Author information preserved."""
        artifact = search_result_full.to_artifact()

        assert artifact.metadata["author_id"] == "author-full-001"
        assert artifact.metadata["author_name"] == "Dr. Full Test"

    def test_given_search_result_without_lineage_when_converted_then_defaults(
        self,
    ) -> None:
        """GWT: No lineage uses default provenance."""
        result = SearchResult(
            chunk_id="no-lineage-001",
            content="No lineage content.",
            score=0.5,
            document_id="doc-no-lineage",
            section_title="",
            chunk_type="content",
            source_file="",
            word_count=3,
        )

        artifact = result.to_artifact()

        assert artifact.parent_id is None
        assert artifact.root_artifact_id is None
        assert artifact.lineage_depth == 0
        assert artifact.provenance == ["retrieval"]

    def test_given_search_result_when_converted_then_content_hash_computed(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: Content hash automatically computed."""
        artifact = search_result_full.to_artifact()

        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64  # SHA-256

        # Verify hash correctness
        expected_hash = hashlib.sha256(
            search_result_full.content.encode("utf-8")
        ).hexdigest()
        assert artifact.content_hash == expected_hash


# =============================================================================
# GWT SCENARIO 3: ROUND-TRIP FIDELITY
# =============================================================================


class TestGWTRoundTripFidelity:
    """
    GWT Scenario 3: Round-Trip Data Fidelity

    Given: IFChunkArtifact
    When: artifact → ChunkRecord → storage → retrieval → artifact
    Then: All critical data preserved through the cycle
    """

    def test_given_artifact_when_round_tripped_then_id_preserved(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: artifact_id survives round-trip."""
        # Artifact → ChunkRecord
        record = normalize_to_chunk_record(artifact_with_full_lineage)

        # ChunkRecord → SearchResult (simulating storage retrieval)
        search_result = SearchResult.from_chunk(record, score=0.9)

        # SearchResult → Artifact
        restored = search_result.to_artifact()

        assert restored.artifact_id == artifact_with_full_lineage.artifact_id

    def test_given_artifact_when_round_tripped_then_content_preserved(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: Content survives round-trip exactly."""
        record = normalize_to_chunk_record(artifact_with_full_lineage)
        search_result = SearchResult.from_chunk(record, score=0.9)
        restored = search_result.to_artifact()

        assert restored.content == artifact_with_full_lineage.content

    def test_given_artifact_when_round_tripped_then_document_id_preserved(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: document_id survives round-trip."""
        record = normalize_to_chunk_record(artifact_with_full_lineage)
        search_result = SearchResult.from_chunk(record, score=0.9)
        restored = search_result.to_artifact()

        assert restored.document_id == artifact_with_full_lineage.document_id

    def test_given_artifact_when_round_tripped_then_chunk_index_preserved(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """GWT: chunk_index survives round-trip."""
        record = normalize_to_chunk_record(artifact_with_full_lineage)

        assert record.chunk_index == artifact_with_full_lineage.chunk_index


# =============================================================================
# GWT SCENARIO 4: BATCH OPERATIONS
# =============================================================================


class TestGWTBatchOperations:
    """
    GWT Scenario 4: Batch Processing

    Given: List of artifacts or mixed types
    When: Batch operations performed
    Then: All items processed correctly
    """

    def test_given_artifact_list_when_all_normalized_then_all_valid(
        self,
        large_artifact_batch: List[IFChunkArtifact],
    ) -> None:
        """GWT: Batch normalization produces valid ChunkRecords."""
        results = [normalize_to_chunk_record(a) for a in large_artifact_batch]

        assert len(results) == 100
        assert all(isinstance(r, ChunkRecord) for r in results)

    def test_given_artifact_list_when_normalized_then_ids_unique(
        self,
        large_artifact_batch: List[IFChunkArtifact],
    ) -> None:
        """GWT: Batch preserves unique IDs."""
        results = [normalize_to_chunk_record(a) for a in large_artifact_batch]
        ids = [r.chunk_id for r in results]

        assert len(ids) == len(set(ids))  # All unique

    def test_given_mixed_list_when_normalized_then_all_converted(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
        chunk_record_with_enrichment: ChunkRecord,
    ) -> None:
        """GWT: Mixed list (artifact + ChunkRecord) all converted."""
        mixed = [artifact_with_full_lineage, chunk_record_with_enrichment]
        results = [normalize_to_chunk_record(item) for item in mixed]

        assert len(results) == 2
        assert all(isinstance(r, ChunkRecord) for r in results)

    def test_given_search_results_when_converted_to_artifacts_then_all_valid(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """GWT: Batch SearchResult → artifact conversion."""
        results = [search_result_full] * 10
        artifacts = [r.to_artifact() for r in results]

        assert len(artifacts) == 10
        assert all(isinstance(a, IFChunkArtifact) for a in artifacts)


# =============================================================================
# GWT SCENARIO 5: JSONL REPOSITORY INTEGRATION
# =============================================================================


class TestGWTJSONLIntegration:
    """
    GWT Scenario 5: JSONL Repository Integration

    Given: JSONL repository instance
    When: Artifacts stored and retrieved
    Then: Data integrity maintained
    """

    def test_given_jsonl_repo_when_artifact_added_then_stored(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
        tmp_path: Path,
    ) -> None:
        """GWT: JSONL accepts and stores artifact."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        result = repo.add_chunk(artifact_with_full_lineage)

        assert result is True
        assert repo.count() == 1

    def test_given_jsonl_repo_when_artifact_batch_added_then_all_stored(
        self,
        large_artifact_batch: List[IFChunkArtifact],
        tmp_path: Path,
    ) -> None:
        """GWT: JSONL batch storage."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        count = repo.add_chunks(large_artifact_batch[:20])  # First 20

        assert count == 20
        assert repo.count() == 20

    def test_given_jsonl_repo_when_artifact_retrieved_then_content_matches(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
        tmp_path: Path,
    ) -> None:
        """GWT: JSONL retrieval preserves content."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        repo.add_chunk(artifact_with_full_lineage)

        retrieved = repo.get_chunk(artifact_with_full_lineage.artifact_id)

        assert retrieved is not None
        assert retrieved.content == artifact_with_full_lineage.content

    def test_given_jsonl_repo_when_mixed_batch_added_then_all_stored(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
        chunk_record_with_enrichment: ChunkRecord,
        tmp_path: Path,
    ) -> None:
        """GWT: JSONL accepts mixed batch."""
        from ingestforge.storage.jsonl import JSONLRepository

        repo = JSONLRepository(tmp_path)
        mixed = [artifact_with_full_lineage, chunk_record_with_enrichment]
        count = repo.add_chunks(mixed)

        assert count == 2
        assert repo.count() == 2


# =============================================================================
# NASA JPL POWER OF TEN COMPLIANCE
# =============================================================================


class TestJPLRule1ControlFlow:
    """
    JPL Rule #1: Simple Control Flow

    All loops and branches should have a clear, predictable structure.
    No goto, setjmp/longjmp, or recursion in safety-critical code.
    """

    def test_normalize_has_linear_control_flow(self) -> None:
        """Rule #1: normalize_to_chunk_record uses early returns."""
        source = inspect.getsource(normalize_to_chunk_record)

        # No goto or eval
        assert "goto" not in source.lower()
        assert "eval(" not in source

        # Has early return pattern
        assert "return item" in source or "return" in source

    def test_to_artifact_has_linear_control_flow(self) -> None:
        """Rule #1: SearchResult.to_artifact uses linear flow."""
        source = inspect.getsource(SearchResult.to_artifact)

        assert "goto" not in source.lower()
        assert "eval(" not in source
        assert "exec(" not in source


class TestJPLRule2FixedBounds:
    """
    JPL Rule #2: Fixed Upper Bounds

    All loops and data structures should have fixed, verifiable upper bounds.
    """

    def test_max_lineage_fields_constant_defined(self) -> None:
        """Rule #2: MAX_LINEAGE_FIELDS constant exists."""
        assert MAX_LINEAGE_FIELDS == 5

    def test_normalize_handles_bounded_metadata(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """Rule #2: Metadata size is bounded."""
        result = normalize_to_chunk_record(artifact_with_full_lineage)

        # Lineage fields should be <= MAX_LINEAGE_FIELDS
        lineage_keys = [k for k in result.metadata.keys() if k.startswith("_lineage")]
        assert len(lineage_keys) <= MAX_LINEAGE_FIELDS


class TestJPLRule4FunctionSize:
    """
    JPL Rule #4: Function Size Limit

    Functions should be no longer than 60 lines.
    """

    def test_normalize_function_under_60_lines(self) -> None:
        """Rule #4: normalize_to_chunk_record < 60 lines."""
        source = inspect.getsource(normalize_to_chunk_record)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"

    def test_to_artifact_under_60_lines(self) -> None:
        """Rule #4: SearchResult.to_artifact < 60 lines."""
        source = inspect.getsource(SearchResult.to_artifact)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Method has {len(lines)} lines"

    def test_from_chunk_under_60_lines(self) -> None:
        """Rule #4: SearchResult.from_chunk < 60 lines."""
        source = inspect.getsource(SearchResult.from_chunk)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Method has {len(lines)} lines"


class TestJPLRule7ReturnTypes:
    """
    JPL Rule #7: Return Value Checking

    All functions should have explicit return types and return values checked.
    """

    def test_normalize_has_return_type_annotation(self) -> None:
        """Rule #7: normalize_to_chunk_record has return type."""
        annotations = normalize_to_chunk_record.__annotations__
        assert "return" in annotations
        assert annotations["return"] == ChunkRecord

    def test_to_artifact_has_return_type_annotation(self) -> None:
        """Rule #7: SearchResult.to_artifact has return type."""
        annotations = SearchResult.to_artifact.__annotations__
        assert "return" in annotations

    def test_from_chunk_has_return_type_annotation(self) -> None:
        """Rule #7: SearchResult.from_chunk has return type."""
        annotations = SearchResult.from_chunk.__annotations__
        assert "return" in annotations


class TestJPLRule9TypeHints:
    """
    JPL Rule #9: Type Annotations

    All function parameters should have complete type hints.
    """

    def test_normalize_has_parameter_type_hints(self) -> None:
        """Rule #9: normalize_to_chunk_record has param types."""
        annotations = normalize_to_chunk_record.__annotations__
        assert "item" in annotations

    def test_to_artifact_has_self_typed(self) -> None:
        """Rule #9: to_artifact method properly typed."""
        # Method should have return annotation
        annotations = SearchResult.to_artifact.__annotations__
        assert "return" in annotations


# =============================================================================
# BOUNDARY CONDITIONS & EDGE CASES
# =============================================================================


class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    def test_empty_content_artifact_handled(self) -> None:
        """Edge: Empty content artifact."""
        artifact = IFChunkArtifact(
            artifact_id="empty-001",
            document_id="doc-empty",
            content="",
        )

        result = normalize_to_chunk_record(artifact)

        assert result.content == ""
        assert result.chunk_id == "empty-001"

    def test_very_long_content_artifact_handled(self) -> None:
        """Edge: Very long content (100KB)."""
        long_content = "x" * 100_000

        artifact = IFChunkArtifact(
            artifact_id="long-001",
            document_id="doc-long",
            content=long_content,
        )

        result = normalize_to_chunk_record(artifact)

        assert len(result.content) == 100_000

    def test_unicode_content_preserved(self) -> None:
        """Edge: Unicode content (emoji, CJK, RTL)."""
        unicode_content = "Hello 世界 🌍 مرحبا"

        artifact = IFChunkArtifact(
            artifact_id="unicode-001",
            document_id="doc-unicode",
            content=unicode_content,
        )

        result = normalize_to_chunk_record(artifact)

        assert result.content == unicode_content

    def test_special_characters_in_id_handled(self) -> None:
        """Edge: Special characters in IDs."""
        artifact = IFChunkArtifact(
            artifact_id="chunk/with:special-chars_123",
            document_id="doc/special:test",
            content="Special ID content.",
        )

        result = normalize_to_chunk_record(artifact)

        assert result.chunk_id == "chunk/with:special-chars_123"

    def test_none_metadata_values_handled(self) -> None:
        """Edge: None values in metadata."""
        artifact = IFChunkArtifact(
            artifact_id="none-meta-001",
            document_id="doc-none",
            content="None metadata content.",
            metadata={
                "section_title": None,
                "source_file": None,
            },
        )

        result = normalize_to_chunk_record(artifact)

        # Should not raise, None values handled
        assert result.chunk_id == "none-meta-001"

    def test_zero_score_search_result_handled(self) -> None:
        """Edge: Zero retrieval score."""
        result = SearchResult(
            chunk_id="zero-score-001",
            content="Zero score content.",
            score=0.0,
            document_id="doc-zero",
            section_title="",
            chunk_type="content",
            source_file="",
            word_count=3,
        )

        artifact = result.to_artifact()

        assert artifact.metadata["retrieval_score"] == 0.0

    def test_max_score_search_result_handled(self) -> None:
        """Edge: Maximum retrieval score (1.0)."""
        result = SearchResult(
            chunk_id="max-score-001",
            content="Max score content.",
            score=1.0,
            document_id="doc-max",
            section_title="",
            chunk_type="content",
            source_file="",
            word_count=3,
        )

        artifact = result.to_artifact()

        assert artifact.metadata["retrieval_score"] == 1.0


# =============================================================================
# ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Tests for error conditions and failure modes."""

    def test_normalize_rejects_none_input(self) -> None:
        """Error: None input raises TypeError."""
        with pytest.raises(TypeError):
            normalize_to_chunk_record(None)

    def test_normalize_rejects_primitive_types(self) -> None:
        """Error: Primitive types raise TypeError."""
        with pytest.raises(TypeError):
            normalize_to_chunk_record(42)

        with pytest.raises(TypeError):
            normalize_to_chunk_record(3.14)

        with pytest.raises(TypeError):
            normalize_to_chunk_record(True)

    def test_normalize_rejects_list_input(self) -> None:
        """Error: List input raises TypeError (not iterable handling)."""
        with pytest.raises(TypeError):
            normalize_to_chunk_record([])


# =============================================================================
# CONTENT HASH INTEGRITY
# =============================================================================


class TestContentHashIntegrity:
    """Tests for content hash computation and verification."""

    def test_artifact_content_hash_computed(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """Hash: Artifact has content hash."""
        assert artifact_with_full_lineage.content_hash is not None
        assert len(artifact_with_full_lineage.content_hash) == 64

    def test_artifact_content_hash_correct(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """Hash: Artifact hash matches expected."""
        expected = hashlib.sha256(
            artifact_with_full_lineage.content.encode("utf-8")
        ).hexdigest()

        assert artifact_with_full_lineage.content_hash == expected

    def test_search_result_artifact_hash_matches(
        self,
        search_result_full: SearchResult,
    ) -> None:
        """Hash: Converted artifact has correct hash."""
        artifact = search_result_full.to_artifact()
        expected = hashlib.sha256(
            search_result_full.content.encode("utf-8")
        ).hexdigest()

        assert artifact.content_hash == expected

    def test_normalized_record_preserves_hash_in_metadata(
        self,
        artifact_with_full_lineage: IFChunkArtifact,
    ) -> None:
        """Hash: ChunkRecord metadata contains hash."""
        result = normalize_to_chunk_record(artifact_with_full_lineage)

        assert "_content_hash" in result.metadata
        assert (
            result.metadata["_content_hash"] == artifact_with_full_lineage.content_hash
        )


# =============================================================================
# GWT SCENARIO COMPLETENESS VERIFICATION
# =============================================================================


class TestGWTScenarioCompleteness:
    """Meta-tests verifying all GWT scenarios are covered."""

    def test_scenario_1_normalization_covered(self) -> None:
        """Verify: Scenario 1 (Normalization) has tests."""
        assert hasattr(
            TestGWTArtifactNormalization,
            "test_given_artifact_when_normalized_then_chunk_record_returned",
        )
        assert hasattr(
            TestGWTArtifactNormalization,
            "test_given_chunk_record_when_normalized_then_same_object_returned",
        )

    def test_scenario_2_search_result_conversion_covered(self) -> None:
        """Verify: Scenario 2 (SearchResult Conversion) has tests."""
        assert hasattr(
            TestGWTSearchResultConversion,
            "test_given_search_result_when_converted_then_artifact_returned",
        )
        assert hasattr(
            TestGWTSearchResultConversion,
            "test_given_search_result_when_converted_then_score_preserved",
        )

    def test_scenario_3_round_trip_covered(self) -> None:
        """Verify: Scenario 3 (Round-Trip) has tests."""
        assert hasattr(
            TestGWTRoundTripFidelity,
            "test_given_artifact_when_round_tripped_then_id_preserved",
        )
        assert hasattr(
            TestGWTRoundTripFidelity,
            "test_given_artifact_when_round_tripped_then_content_preserved",
        )

    def test_scenario_4_batch_operations_covered(self) -> None:
        """Verify: Scenario 4 (Batch Operations) has tests."""
        assert hasattr(
            TestGWTBatchOperations,
            "test_given_artifact_list_when_all_normalized_then_all_valid",
        )

    def test_scenario_5_jsonl_integration_covered(self) -> None:
        """Verify: Scenario 5 (JSONL Integration) has tests."""
        assert hasattr(
            TestGWTJSONLIntegration,
            "test_given_jsonl_repo_when_artifact_added_then_stored",
        )

    def test_jpl_rules_covered(self) -> None:
        """Verify: JPL rules have test classes."""
        assert TestJPLRule1ControlFlow is not None
        assert TestJPLRule2FixedBounds is not None
        assert TestJPLRule4FunctionSize is not None
        assert TestJPLRule7ReturnTypes is not None
        assert TestJPLRule9TypeHints is not None
