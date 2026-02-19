"""
SearchResult to Artifact Conversion Tests.

GWT-style tests verifying h implementation:
- SearchResult converts to IFChunkArtifact
- Retrieval score preserved
- Lineage metadata restored
- All fields mapped correctly
"""


import pytest

from ingestforge.core.pipeline.artifacts import IFChunkArtifact
from ingestforge.storage.base import SearchResult


# --- Fixtures ---


@pytest.fixture
def basic_search_result() -> SearchResult:
    """Create a basic SearchResult."""
    return SearchResult(
        chunk_id="search-001",
        content="Search result content for testing.",
        score=0.85,
        document_id="doc-001",
        section_title="Test Section",
        chunk_type="content",
        source_file="test.pdf",
        word_count=6,
        library="test-library",
    )


@pytest.fixture
def search_result_with_metadata() -> SearchResult:
    """Create a SearchResult with rich metadata."""
    return SearchResult(
        chunk_id="search-002",
        content="Content with entities and concepts.",
        score=0.92,
        document_id="doc-002",
        section_title="Analysis Section",
        chunk_type="analysis",
        source_file="analysis.md",
        word_count=5,
        page_start=10,
        page_end=12,
        library="research",
        author_id="author-001",
        author_name="Dr. Test",
        source_location={"title": "Test Document", "page": 10},
        metadata={
            "entities": ["Entity1", "Entity2"],
            "concepts": ["Concept1"],
            "quality_score": 0.88,
        },
    )


@pytest.fixture
def search_result_with_lineage() -> SearchResult:
    """Create a SearchResult with lineage info in metadata (from g storage)."""
    return SearchResult(
        chunk_id="search-003",
        content="Content with preserved lineage.",
        score=0.78,
        document_id="doc-003",
        section_title="Lineage Section",
        chunk_type="content",
        source_file="lineage.txt",
        word_count=4,
        metadata={
            "_lineage_parent_id": "text-001",
            "_lineage_root_id": "file-001",
            "_lineage_depth": 2,
            "_lineage_provenance": ["extractor", "chunker", "enricher"],
            "entities": ["Person1"],
        },
    )


# --- GWT Scenario 1: Basic Conversion ---


class TestBasicConversion:
    """Tests for basic SearchResult to artifact conversion."""

    def test_to_artifact_returns_if_chunk_artifact(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When to_artifact called,
        Then IFChunkArtifact is returned."""
        artifact = basic_search_result.to_artifact()

        assert isinstance(artifact, IFChunkArtifact)

    def test_to_artifact_preserves_chunk_id(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When converted,
        Then artifact_id matches chunk_id."""
        artifact = basic_search_result.to_artifact()

        assert artifact.artifact_id == "search-001"

    def test_to_artifact_preserves_content(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When converted,
        Then content is preserved."""
        artifact = basic_search_result.to_artifact()

        assert artifact.content == "Search result content for testing."

    def test_to_artifact_preserves_document_id(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When converted,
        Then document_id is preserved."""
        artifact = basic_search_result.to_artifact()

        assert artifact.document_id == "doc-001"


# --- GWT Scenario 2: Retrieval Score Preserved ---


class TestRetrievalScorePreserved:
    """Tests for retrieval score preservation."""

    def test_score_in_artifact_metadata(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult with score, When converted,
        Then score is in metadata as retrieval_score."""
        artifact = basic_search_result.to_artifact()

        assert "retrieval_score" in artifact.metadata
        assert artifact.metadata["retrieval_score"] == 0.85

    def test_high_score_preserved(
        self,
        search_result_with_metadata: SearchResult,
    ) -> None:
        """Given SearchResult with high score, When converted,
        Then exact score value preserved."""
        artifact = search_result_with_metadata.to_artifact()

        assert artifact.metadata["retrieval_score"] == 0.92


# --- GWT Scenario 3: Lineage Metadata Restored ---


class TestLineageMetadataRestored:
    """Tests for lineage restoration from stored metadata."""

    def test_parent_id_restored(
        self,
        search_result_with_lineage: SearchResult,
    ) -> None:
        """Given SearchResult with lineage metadata, When converted,
        Then parent_id is restored."""
        artifact = search_result_with_lineage.to_artifact()

        assert artifact.parent_id == "text-001"

    def test_root_artifact_id_restored(
        self,
        search_result_with_lineage: SearchResult,
    ) -> None:
        """Given SearchResult with lineage metadata, When converted,
        Then root_artifact_id is restored."""
        artifact = search_result_with_lineage.to_artifact()

        assert artifact.root_artifact_id == "file-001"

    def test_lineage_depth_restored(
        self,
        search_result_with_lineage: SearchResult,
    ) -> None:
        """Given SearchResult with lineage metadata, When converted,
        Then lineage_depth is restored."""
        artifact = search_result_with_lineage.to_artifact()

        assert artifact.lineage_depth == 2

    def test_provenance_restored(
        self,
        search_result_with_lineage: SearchResult,
    ) -> None:
        """Given SearchResult with lineage metadata, When converted,
        Then provenance is restored."""
        artifact = search_result_with_lineage.to_artifact()

        assert artifact.provenance == ["extractor", "chunker", "enricher"]

    def test_lineage_fields_not_in_final_metadata(
        self,
        search_result_with_lineage: SearchResult,
    ) -> None:
        """Given SearchResult with lineage metadata, When converted,
        Then lineage fields are extracted, not in metadata."""
        artifact = search_result_with_lineage.to_artifact()

        assert "_lineage_parent_id" not in artifact.metadata
        assert "_lineage_root_id" not in artifact.metadata
        assert "_lineage_depth" not in artifact.metadata
        assert "_lineage_provenance" not in artifact.metadata


# --- GWT Scenario 4: Metadata Fields Mapped ---


class TestMetadataFieldsMapped:
    """Tests for metadata field mapping."""

    def test_section_title_in_metadata(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When converted,
        Then section_title is in metadata."""
        artifact = basic_search_result.to_artifact()

        assert artifact.metadata["section_title"] == "Test Section"

    def test_source_file_in_metadata(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When converted,
        Then source_file is in metadata."""
        artifact = basic_search_result.to_artifact()

        assert artifact.metadata["source_file"] == "test.pdf"

    def test_library_in_metadata(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When converted,
        Then library is in metadata."""
        artifact = basic_search_result.to_artifact()

        assert artifact.metadata["library"] == "test-library"

    def test_page_info_in_metadata(
        self,
        search_result_with_metadata: SearchResult,
    ) -> None:
        """Given SearchResult with page info, When converted,
        Then page_start and page_end are in metadata."""
        artifact = search_result_with_metadata.to_artifact()

        assert artifact.metadata["page_start"] == 10
        assert artifact.metadata["page_end"] == 12

    def test_author_info_in_metadata(
        self,
        search_result_with_metadata: SearchResult,
    ) -> None:
        """Given SearchResult with author info, When converted,
        Then author_id and author_name are in metadata."""
        artifact = search_result_with_metadata.to_artifact()

        assert artifact.metadata["author_id"] == "author-001"
        assert artifact.metadata["author_name"] == "Dr. Test"

    def test_source_location_in_metadata(
        self,
        search_result_with_metadata: SearchResult,
    ) -> None:
        """Given SearchResult with source_location, When converted,
        Then source_location is in metadata."""
        artifact = search_result_with_metadata.to_artifact()

        assert artifact.metadata["source_location"] == {
            "title": "Test Document",
            "page": 10,
        }

    def test_entities_merged_from_existing_metadata(
        self,
        search_result_with_metadata: SearchResult,
    ) -> None:
        """Given SearchResult with entities in metadata, When converted,
        Then entities are preserved."""
        artifact = search_result_with_metadata.to_artifact()

        assert artifact.metadata["entities"] == ["Entity1", "Entity2"]


# --- GWT Scenario 5: Default Values ---


class TestDefaultValues:
    """Tests for default value handling."""

    def test_no_lineage_uses_empty_values(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult without lineage, When converted,
        Then lineage fields have default values."""
        artifact = basic_search_result.to_artifact()

        assert artifact.parent_id is None
        assert artifact.root_artifact_id is None
        assert artifact.lineage_depth == 0

    def test_no_lineage_uses_retrieval_provenance(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult without provenance, When converted,
        Then provenance defaults to ['retrieval']."""
        artifact = basic_search_result.to_artifact()

        assert artifact.provenance == ["retrieval"]


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case tests for SearchResult conversion."""

    def test_empty_metadata_handled(self) -> None:
        """Given SearchResult with no metadata, When converted,
        Then conversion succeeds."""
        result = SearchResult(
            chunk_id="minimal-001",
            content="Minimal content",
            score=0.5,
            document_id="doc-min",
            section_title="",
            chunk_type="content",
            source_file="",
            word_count=2,
        )

        artifact = result.to_artifact()

        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.metadata["retrieval_score"] == 0.5

    def test_content_hash_computed(
        self,
        basic_search_result: SearchResult,
    ) -> None:
        """Given SearchResult, When converted,
        Then content_hash is computed by artifact."""
        artifact = basic_search_result.to_artifact()

        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64  # SHA-256


# --- List Conversion Tests ---


class TestListConversion:
    """Tests for converting lists of SearchResults."""

    def test_multiple_results_converted(
        self,
        basic_search_result: SearchResult,
        search_result_with_metadata: SearchResult,
    ) -> None:
        """Given list of SearchResults, When all converted,
        Then all become IFChunkArtifacts."""
        results = [basic_search_result, search_result_with_metadata]
        artifacts = [r.to_artifact() for r in results]

        assert len(artifacts) == 2
        assert all(isinstance(a, IFChunkArtifact) for a in artifacts)

    def test_converted_list_preserves_order(
        self,
        basic_search_result: SearchResult,
        search_result_with_metadata: SearchResult,
    ) -> None:
        """Given list of SearchResults, When converted,
        Then artifact_ids match original order."""
        results = [basic_search_result, search_result_with_metadata]
        artifacts = [r.to_artifact() for r in results]

        assert artifacts[0].artifact_id == "search-001"
        assert artifacts[1].artifact_id == "search-002"


# --- JPL Compliance Tests ---


class TestJPLComplianceSearchResult:
    """JPL Power of Ten compliance tests for SearchResult.to_artifact()."""

    def test_to_artifact_under_60_lines(self) -> None:
        """Given to_artifact method, When lines counted,
        Then count < 60."""
        import inspect

        source = inspect.getsource(SearchResult.to_artifact)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Method has {len(lines)} lines"

    def test_to_artifact_has_return_type(self) -> None:
        """Given to_artifact, When annotations checked,
        Then return type is present."""
        annotations = SearchResult.to_artifact.__annotations__
        assert "return" in annotations


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_basic_conversion_covered(self) -> None:
        """GWT Scenario 1 (Basic Conversion) is tested."""
        assert hasattr(
            TestBasicConversion, "test_to_artifact_returns_if_chunk_artifact"
        )

    def test_scenario_2_score_preserved_covered(self) -> None:
        """GWT Scenario 2 (Score Preserved) is tested."""
        assert hasattr(TestRetrievalScorePreserved, "test_score_in_artifact_metadata")

    def test_scenario_3_lineage_restored_covered(self) -> None:
        """GWT Scenario 3 (Lineage Restored) is tested."""
        assert hasattr(TestLineageMetadataRestored, "test_parent_id_restored")

    def test_scenario_4_metadata_mapped_covered(self) -> None:
        """GWT Scenario 4 (Metadata Mapped) is tested."""
        assert hasattr(TestMetadataFieldsMapped, "test_section_title_in_metadata")
