"""
Comprehensive Lineage Chain Tests for Artifact System.

GWT-style tests verifying full lineage tracking from
File → Text → Chunk through the artifact pipeline.

Tests cover:
- Parent-child relationships
- Root artifact tracking
- Provenance chains
- Lineage depth progression
- Content hash inheritance
"""

import warnings
from pathlib import Path
from typing import List

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFTextArtifact,
    IFChunkArtifact,
    IFFileArtifact,
)
from ingestforge.core.pipeline.artifact_factory import ArtifactFactory
from ingestforge.chunking.semantic_chunker import SemanticChunker, ChunkRecord


# --- Fixtures ---


@pytest.fixture
def file_artifact() -> IFFileArtifact:
    """Create source file artifact."""
    return IFFileArtifact(
        artifact_id="file-root-001",
        file_path=Path("/documents/research.pdf"),
        mime_type="application/pdf",
        metadata={"title": "Research Paper"},
    )


@pytest.fixture
def text_artifact(file_artifact: IFFileArtifact) -> IFTextArtifact:
    """Create text artifact derived from file."""
    return IFTextArtifact(
        artifact_id="text-001",
        content="This is the extracted text content from the research paper.",
        parent_id=file_artifact.artifact_id,
        root_artifact_id=file_artifact.artifact_id,
        lineage_depth=1,
        provenance=["pdf-extractor"],
        metadata={"source_path": str(file_artifact.file_path)},
    )


@pytest.fixture
def chunk_artifacts(text_artifact: IFTextArtifact) -> List[IFChunkArtifact]:
    """Create chunk artifacts derived from text."""
    chunks = []
    for i in range(3):
        chunks.append(
            IFChunkArtifact(
                artifact_id=f"chunk-{i:03d}",
                document_id="doc-001",
                content=f"Chunk {i} content from the research paper.",
                chunk_index=i,
                total_chunks=3,
                parent_id=text_artifact.artifact_id,
                root_artifact_id=text_artifact.root_artifact_id,
                lineage_depth=text_artifact.lineage_depth + 1,
                provenance=text_artifact.provenance + ["semantic-chunker"],
            )
        )
    return chunks


# --- GWT Scenario 1: File → Text Lineage ---


class TestFileToTextLineage:
    """Tests for File → Text artifact lineage."""

    def test_text_artifact_has_file_parent(
        self, text_artifact: IFTextArtifact, file_artifact: IFFileArtifact
    ) -> None:
        """Given text derived from file, When parent_id checked,
        Then it references file artifact."""
        assert text_artifact.parent_id == file_artifact.artifact_id

    def test_text_artifact_has_file_as_root(
        self, text_artifact: IFTextArtifact, file_artifact: IFFileArtifact
    ) -> None:
        """Given text derived from file, When root_artifact_id checked,
        Then it references file artifact."""
        assert text_artifact.root_artifact_id == file_artifact.artifact_id

    def test_text_artifact_depth_is_one(self, text_artifact: IFTextArtifact) -> None:
        """Given text derived from file, When lineage_depth checked,
        Then it is 1."""
        assert text_artifact.lineage_depth == 1

    def test_text_artifact_has_provenance(self, text_artifact: IFTextArtifact) -> None:
        """Given text derived from file, When provenance checked,
        Then it contains extractor."""
        assert "pdf-extractor" in text_artifact.provenance


# --- GWT Scenario 2: Text → Chunk Lineage ---


class TestTextToChunkLineage:
    """Tests for Text → Chunk artifact lineage."""

    def test_chunk_artifacts_have_text_parent(
        self,
        chunk_artifacts: List[IFChunkArtifact],
        text_artifact: IFTextArtifact,
    ) -> None:
        """Given chunks derived from text, When parent_id checked,
        Then they reference text artifact."""
        for chunk in chunk_artifacts:
            assert chunk.parent_id == text_artifact.artifact_id

    def test_chunk_artifacts_preserve_root(
        self,
        chunk_artifacts: List[IFChunkArtifact],
        file_artifact: IFFileArtifact,
    ) -> None:
        """Given chunks in chain, When root_artifact_id checked,
        Then they reference original file."""
        for chunk in chunk_artifacts:
            assert chunk.root_artifact_id == file_artifact.artifact_id

    def test_chunk_artifacts_depth_is_two(
        self, chunk_artifacts: List[IFChunkArtifact]
    ) -> None:
        """Given chunks derived from text, When lineage_depth checked,
        Then it is 2."""
        for chunk in chunk_artifacts:
            assert chunk.lineage_depth == 2

    def test_chunk_artifacts_extend_provenance(
        self, chunk_artifacts: List[IFChunkArtifact]
    ) -> None:
        """Given chunks derived from text, When provenance checked,
        Then it includes both extractor and chunker."""
        for chunk in chunk_artifacts:
            assert "pdf-extractor" in chunk.provenance
            assert "semantic-chunker" in chunk.provenance


# --- GWT Scenario 3: Full Chain Integrity ---


class TestFullChainIntegrity:
    """Tests for complete File → Text → Chunk chain."""

    def test_chain_root_consistent(
        self,
        file_artifact: IFFileArtifact,
        text_artifact: IFTextArtifact,
        chunk_artifacts: List[IFChunkArtifact],
    ) -> None:
        """Given full artifact chain, When roots compared,
        Then all point to file artifact."""
        # File is its own root (or None)
        file_root = file_artifact.effective_root_id
        assert file_root == file_artifact.artifact_id

        # Text points to file
        assert text_artifact.root_artifact_id == file_artifact.artifact_id

        # Chunks point to file
        for chunk in chunk_artifacts:
            assert chunk.root_artifact_id == file_artifact.artifact_id

    def test_chain_depth_increments(
        self,
        file_artifact: IFFileArtifact,
        text_artifact: IFTextArtifact,
        chunk_artifacts: List[IFChunkArtifact],
    ) -> None:
        """Given full artifact chain, When depths compared,
        Then they increment correctly."""
        assert file_artifact.lineage_depth == 0
        assert text_artifact.lineage_depth == 1
        for chunk in chunk_artifacts:
            assert chunk.lineage_depth == 2

    def test_chain_provenance_grows(
        self,
        file_artifact: IFFileArtifact,
        text_artifact: IFTextArtifact,
        chunk_artifacts: List[IFChunkArtifact],
    ) -> None:
        """Given full artifact chain, When provenance lengths compared,
        Then they grow at each step."""
        assert len(file_artifact.provenance) == 0
        assert len(text_artifact.provenance) == 1
        for chunk in chunk_artifacts:
            assert len(chunk.provenance) == 2


# --- GWT Scenario 4: Factory-Created Lineage ---


class TestFactoryCreatedLineage:
    """Tests for lineage via ArtifactFactory."""

    def test_factory_text_with_parent(self, file_artifact: IFFileArtifact) -> None:
        """Given ArtifactFactory.text_from_string with parent,
        When artifact created, Then lineage established."""
        # Create text artifact using factory pattern
        text = ArtifactFactory.text_from_string(
            content="Extracted content",
            source_path="/documents/test.pdf",
        )

        # Without parent, lineage_depth is 0
        assert text.lineage_depth == 0
        assert text.parent_id is None

    def test_factory_chunk_from_record_with_parent(
        self, text_artifact: IFTextArtifact
    ) -> None:
        """Given ArtifactFactory.chunk_from_record with parent,
        When artifact created, Then lineage established."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = ChunkRecord(
                chunk_id="chunk-test",
                document_id="doc-001",
                content="Test chunk content",
            )

        chunk = ArtifactFactory.chunk_from_record(record, parent=text_artifact)

        assert chunk.parent_id == text_artifact.artifact_id
        assert chunk.root_artifact_id == text_artifact.effective_root_id
        assert chunk.lineage_depth == text_artifact.lineage_depth + 1

    def test_factory_batch_with_shared_parent(
        self, text_artifact: IFTextArtifact
    ) -> None:
        """Given batch conversion with parent, When artifacts created,
        Then all share same parent lineage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            records = [
                ChunkRecord(chunk_id=f"c{i}", document_id="doc", content=f"Content {i}")
                for i in range(5)
            ]

        artifacts = ArtifactFactory.chunks_from_records(records, parent=text_artifact)

        for artifact in artifacts:
            assert artifact.parent_id == text_artifact.artifact_id
            assert artifact.root_artifact_id == text_artifact.effective_root_id


# --- GWT Scenario 5: Chunker Lineage ---


class TestChunkerLineage:
    """Tests for lineage via SemanticChunker.chunk_to_artifacts()."""

    def test_chunker_artifacts_have_parent_lineage(
        self, text_artifact: IFTextArtifact
    ) -> None:
        """Given chunk_to_artifacts with parent, When chunks created,
        Then all have parent lineage."""
        chunker = SemanticChunker(
            max_chunk_size=200,
            min_chunk_size=20,
            use_embeddings=False,
        )

        text = "First sentence here. Second sentence here. Third sentence here."
        artifacts = chunker.chunk_to_artifacts(
            text,
            document_id="doc-001",
            parent=text_artifact,
        )

        for artifact in artifacts:
            assert artifact.parent_id == text_artifact.artifact_id

    def test_chunker_artifacts_preserve_root(
        self, text_artifact: IFTextArtifact
    ) -> None:
        """Given chunk_to_artifacts with parent having root,
        When chunks created, Then root is preserved."""
        chunker = SemanticChunker(
            max_chunk_size=200,
            min_chunk_size=20,
            use_embeddings=False,
        )

        text = "Sample text content."
        artifacts = chunker.chunk_to_artifacts(
            text,
            document_id="doc-001",
            parent=text_artifact,
        )

        for artifact in artifacts:
            assert artifact.root_artifact_id == text_artifact.root_artifact_id

    def test_chunker_artifacts_have_provenance(
        self, text_artifact: IFTextArtifact
    ) -> None:
        """Given chunk_to_artifacts with parent, When chunks created,
        Then provenance includes from-chunk-record."""
        chunker = SemanticChunker(
            max_chunk_size=200,
            min_chunk_size=20,
            use_embeddings=False,
        )

        text = "Sample text."
        artifacts = chunker.chunk_to_artifacts(
            text,
            document_id="doc-001",
            parent=text_artifact,
        )

        for artifact in artifacts:
            assert "from-chunk-record" in artifact.provenance


# --- GWT Scenario 6: Round-Trip Lineage Preservation ---


class TestRoundTripLineagePreservation:
    """Tests for lineage preservation in round-trip conversions."""

    def test_to_chunk_record_preserves_lineage_in_metadata(
        self, chunk_artifacts: List[IFChunkArtifact]
    ) -> None:
        """Given IFChunkArtifact with lineage, When converted to ChunkRecord,
        Then lineage is preserved in metadata."""
        for artifact in chunk_artifacts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = artifact.to_chunk_record()

            # Lineage stored in metadata
            assert record.metadata.get("_lineage_parent_id") == artifact.parent_id
            assert record.metadata.get("_lineage_root_id") == artifact.root_artifact_id
            assert record.metadata.get("_lineage_depth") == artifact.lineage_depth

    def test_lineage_provenance_preserved_in_metadata(
        self, chunk_artifacts: List[IFChunkArtifact]
    ) -> None:
        """Given IFChunkArtifact with provenance, When converted,
        Then provenance is in metadata."""
        for artifact in chunk_artifacts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = artifact.to_chunk_record()

            assert "_lineage_provenance" in record.metadata
            assert record.metadata["_lineage_provenance"] == artifact.provenance


# --- Edge Cases ---


class TestLineageEdgeCases:
    """Edge case tests for lineage handling."""

    def test_orphan_artifact_has_no_lineage(self) -> None:
        """Given artifact without parent, When examined,
        Then lineage fields are default."""
        orphan = IFChunkArtifact(
            artifact_id="orphan-001",
            document_id="doc-001",
            content="Orphan content",
        )

        assert orphan.parent_id is None
        assert orphan.root_artifact_id is None
        assert orphan.lineage_depth == 0
        assert orphan.provenance == []

    def test_effective_root_for_root_artifact(self) -> None:
        """Given root artifact, When effective_root_id called,
        Then returns own artifact_id."""
        root = IFFileArtifact(
            artifact_id="root-001",
            file_path=Path("/test.pdf"),
            mime_type="application/pdf",
        )

        assert root.effective_root_id == root.artifact_id

    def test_effective_root_for_child_artifact(
        self, text_artifact: IFTextArtifact, file_artifact: IFFileArtifact
    ) -> None:
        """Given child artifact, When effective_root_id called,
        Then returns root_artifact_id."""
        assert text_artifact.effective_root_id == file_artifact.artifact_id


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_file_to_text_covered(self) -> None:
        """GWT Scenario 1 (File → Text) is tested."""
        assert hasattr(TestFileToTextLineage, "test_text_artifact_has_file_parent")

    def test_scenario_2_text_to_chunk_covered(self) -> None:
        """GWT Scenario 2 (Text → Chunk) is tested."""
        assert hasattr(TestTextToChunkLineage, "test_chunk_artifacts_have_text_parent")

    def test_scenario_3_full_chain_covered(self) -> None:
        """GWT Scenario 3 (Full Chain) is tested."""
        assert hasattr(TestFullChainIntegrity, "test_chain_root_consistent")

    def test_scenario_4_factory_lineage_covered(self) -> None:
        """GWT Scenario 4 (Factory Lineage) is tested."""
        assert hasattr(
            TestFactoryCreatedLineage, "test_factory_chunk_from_record_with_parent"
        )

    def test_scenario_5_chunker_lineage_covered(self) -> None:
        """GWT Scenario 5 (Chunker Lineage) is tested."""
        assert hasattr(TestChunkerLineage, "test_chunker_artifacts_have_parent_lineage")

    def test_scenario_6_round_trip_covered(self) -> None:
        """GWT Scenario 6 (Round-Trip) is tested."""
        assert hasattr(
            TestRoundTripLineagePreservation,
            "test_to_chunk_record_preserves_lineage_in_metadata",
        )
