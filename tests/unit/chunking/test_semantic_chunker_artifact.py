"""
Tests for e: SemanticChunker.chunk_to_artifacts().

GWT-style tests verifying that chunking produces IFChunkArtifact
with proper lineage, metadata, and backward compatibility.
"""

import warnings

import pytest

from ingestforge.chunking.semantic_chunker import SemanticChunker, ChunkRecord
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFTextArtifact


# --- Fixtures ---


@pytest.fixture
def chunker() -> SemanticChunker:
    """Create SemanticChunker instance with embeddings disabled for speed."""
    return SemanticChunker(
        max_chunk_size=500,
        min_chunk_size=50,
        use_embeddings=False,
    )


@pytest.fixture
def sample_text() -> str:
    """Sample text for chunking tests."""
    return """
    Machine learning is a subset of artificial intelligence.
    It enables computers to learn from data without explicit programming.

    Deep learning is a type of machine learning.
    It uses neural networks with many layers.
    These networks can learn complex patterns.

    Natural language processing deals with text and speech.
    It helps computers understand human language.
    Applications include translation and sentiment analysis.
    """


@pytest.fixture
def parent_artifact() -> IFTextArtifact:
    """Create a parent text artifact for lineage testing."""
    return IFTextArtifact(
        artifact_id="parent-text-001",
        content="Sample parent text content",
        metadata={"source_path": "/tmp/test.pdf"},
    )


# --- GWT Scenario 1: Chunk Stage Produces IFChunkArtifact ---


class TestChunkToArtifacts:
    """Tests that chunk_to_artifacts produces artifacts."""

    def test_returns_list_of_artifacts(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given text, When chunk_to_artifacts called,
        Then list of IFChunkArtifact is returned."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        assert isinstance(result, list)
        assert all(isinstance(item, IFChunkArtifact) for item in result)

    def test_artifacts_have_content(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given text, When chunk_to_artifacts called,
        Then each artifact has content."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        for artifact in result:
            assert artifact.content is not None
            assert len(artifact.content) > 0

    def test_artifacts_have_document_id(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given document_id, When chunk_to_artifacts called,
        Then each artifact has document_id."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="test-doc-123",
        )

        for artifact in result:
            assert artifact.document_id == "test-doc-123"

    def test_artifacts_have_content_hash(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given text, When chunk_to_artifacts called,
        Then each artifact has content_hash."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        for artifact in result:
            assert artifact.content_hash is not None
            assert len(artifact.content_hash) == 64  # SHA-256

    def test_artifacts_have_chunk_indices(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given text, When chunk_to_artifacts called,
        Then artifacts have sequential chunk_index."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        indices = [a.chunk_index for a in result]
        assert indices == list(range(len(result)))


# --- GWT Scenario 2: Lineage Tracking ---


class TestLineageTracking:
    """Tests that lineage is properly tracked from parent."""

    def test_artifact_with_parent_has_parent_id(
        self,
        chunker: SemanticChunker,
        sample_text: str,
        parent_artifact: IFTextArtifact,
    ) -> None:
        """Given parent artifact, When chunk_to_artifacts with parent,
        Then each artifact has parent_id set."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
            parent=parent_artifact,
        )

        for artifact in result:
            assert artifact.parent_id == parent_artifact.artifact_id

    def test_artifact_with_parent_has_root_id(
        self,
        chunker: SemanticChunker,
        sample_text: str,
        parent_artifact: IFTextArtifact,
    ) -> None:
        """Given parent artifact, When chunk_to_artifacts with parent,
        Then each artifact has root_artifact_id set."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
            parent=parent_artifact,
        )

        for artifact in result:
            assert artifact.root_artifact_id == parent_artifact.effective_root_id

    def test_artifact_with_parent_has_lineage_depth(
        self,
        chunker: SemanticChunker,
        sample_text: str,
        parent_artifact: IFTextArtifact,
    ) -> None:
        """Given parent artifact, When chunk_to_artifacts with parent,
        Then lineage_depth is parent + 1."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
            parent=parent_artifact,
        )

        for artifact in result:
            assert artifact.lineage_depth == parent_artifact.lineage_depth + 1

    def test_artifact_with_parent_has_provenance(
        self,
        chunker: SemanticChunker,
        sample_text: str,
        parent_artifact: IFTextArtifact,
    ) -> None:
        """Given parent artifact, When chunk_to_artifacts with parent,
        Then provenance includes from-chunk-record."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
            parent=parent_artifact,
        )

        for artifact in result:
            assert "from-chunk-record" in artifact.provenance

    def test_artifact_without_parent_has_no_lineage(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given no parent, When chunk_to_artifacts called,
        Then artifacts have no parent lineage."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        for artifact in result:
            assert artifact.parent_id is None
            assert artifact.lineage_depth == 0


# --- GWT Scenario 3: Backward Compatibility ---


class TestBackwardCompatibility:
    """Tests backward compatibility with ChunkRecord."""

    def test_artifact_can_convert_to_chunk_record(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given IFChunkArtifact, When to_chunk_record() called,
        Then valid ChunkRecord is returned."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        for artifact in result:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = artifact.to_chunk_record()
                assert isinstance(record, ChunkRecord)

    def test_round_trip_preserves_content(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given artifact converted to ChunkRecord,
        Then content is preserved."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        for artifact in result:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = artifact.to_chunk_record()
                assert record.content == artifact.content

    def test_round_trip_preserves_document_id(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given artifact converted to ChunkRecord,
        Then document_id is preserved."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="test-doc-456",
        )

        for artifact in result:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = artifact.to_chunk_record()
                assert record.document_id == "test-doc-456"


# --- GWT Scenario 4: Metadata Preservation ---


class TestMetadataPreservation:
    """Tests that metadata is preserved in artifacts."""

    def test_artifacts_have_word_count(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given chunking, When artifact examined,
        Then metadata includes word_count."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        for artifact in result:
            assert "word_count" in artifact.metadata

    def test_artifacts_have_char_count(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given chunking, When artifact examined,
        Then metadata includes char_count."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        for artifact in result:
            assert "char_count" in artifact.metadata

    def test_custom_metadata_preserved(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given custom metadata, When chunk_to_artifacts called,
        Then custom metadata is in artifacts (nested in chunk_metadata)."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
            metadata={"custom_key": "custom_value"},
        )

        for artifact in result:
            # Custom metadata is nested under chunk_metadata due to ChunkRecord structure
            chunk_meta = artifact.metadata.get("chunk_metadata", {})
            assert chunk_meta.get("custom_key") == "custom_value"


# --- GWT Scenario 5: Consistency with chunk() ---


class TestConsistencyWithChunk:
    """Tests that chunk_to_artifacts is consistent with chunk()."""

    def test_same_number_of_results(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given same input, When chunk and chunk_to_artifacts called,
        Then same number of results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk_records = chunker.chunk(sample_text, "doc-001")

        artifacts = chunker.chunk_to_artifacts(sample_text, "doc-001")

        assert len(artifacts) == len(chunk_records)

    def test_same_content_in_results(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given same input, When chunk and chunk_to_artifacts called,
        Then content matches."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk_records = chunker.chunk(sample_text, "doc-001")

        artifacts = chunker.chunk_to_artifacts(sample_text, "doc-001")

        for record, artifact in zip(chunk_records, artifacts):
            assert record.content == artifact.content


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_text_produces_artifact(self, chunker: SemanticChunker) -> None:
        """Given empty text, When chunk_to_artifacts called,
        Then single artifact with empty content."""
        result = chunker.chunk_to_artifacts(
            "",
            document_id="doc-empty",
        )

        assert len(result) == 1
        assert result[0].content == ""

    def test_short_text_produces_single_artifact(
        self, chunker: SemanticChunker
    ) -> None:
        """Given short text, When chunk_to_artifacts called,
        Then single artifact is created."""
        result = chunker.chunk_to_artifacts(
            "Short text.",
            document_id="doc-short",
        )

        assert len(result) == 1

    def test_multiple_chunks_unique_ids(
        self, chunker: SemanticChunker, sample_text: str
    ) -> None:
        """Given multiple chunks, When artifacts examined,
        Then each has unique artifact_id."""
        result = chunker.chunk_to_artifacts(
            sample_text,
            document_id="doc-001",
        )

        ids = [a.artifact_id for a in result]
        assert len(ids) == len(set(ids))


# --- JPL Rules Compliance ---


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_bounded_chunk_count(self, chunker: SemanticChunker) -> None:
        """Given long text, When chunk_to_artifacts called,
        Then reasonable number of chunks created."""
        long_text = "Sentence number {}. " * 100
        long_text = long_text.format(*range(100))

        result = chunker.chunk_to_artifacts(
            long_text,
            document_id="doc-long",
        )

        # Should create chunks, not infinite
        assert 1 <= len(result) <= 100

    def test_explicit_return_type(self) -> None:
        """Given chunk_to_artifacts method, When annotations checked,
        Then return type is explicit."""
        annotations = SemanticChunker.chunk_to_artifacts.__annotations__
        assert "return" in annotations


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_artifact_production_covered(self) -> None:
        """GWT Scenario 1 (Artifact Production) is tested."""
        assert hasattr(TestChunkToArtifacts, "test_returns_list_of_artifacts")

    def test_scenario_2_lineage_covered(self) -> None:
        """GWT Scenario 2 (Lineage Tracking) is tested."""
        assert hasattr(TestLineageTracking, "test_artifact_with_parent_has_parent_id")

    def test_scenario_3_backward_compat_covered(self) -> None:
        """GWT Scenario 3 (Backward Compatibility) is tested."""
        assert hasattr(
            TestBackwardCompatibility, "test_artifact_can_convert_to_chunk_record"
        )

    def test_scenario_4_metadata_covered(self) -> None:
        """GWT Scenario 4 (Metadata Preservation) is tested."""
        assert hasattr(TestMetadataPreservation, "test_artifacts_have_word_count")

    def test_scenario_5_consistency_covered(self) -> None:
        """GWT Scenario 5 (Consistency) is tested."""
        assert hasattr(TestConsistencyWithChunk, "test_same_number_of_results")
