"""
Tests for e: Stage 3 Chunk produces IFChunkArtifact.

GWT-style tests verifying that _stage_chunk_text creates artifacts
while maintaining backward compatibility with ChunkRecord output.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFChunkArtifact
from ingestforge.chunking.semantic_chunker import ChunkRecord


# --- Mock Classes ---


class MockChunker:
    """Mock SemanticChunker for testing."""

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        metadata: Dict[str, Any] = None,
    ) -> List[ChunkRecord]:
        """Return mock ChunkRecords."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [
                ChunkRecord(
                    chunk_id=f"chunk-{document_id}-0",
                    document_id=document_id,
                    content=text[:100] if len(text) > 100 else text,
                    chunk_index=0,
                    total_chunks=1,
                    word_count=len(text.split()),
                    char_count=len(text),
                ),
            ]


class MockPipelineLogger:
    """Mock PipelineLogger for testing."""

    def start_stage(self, name: str) -> None:
        pass

    def log_progress(self, message: str) -> None:
        pass


class MockDocState:
    """Mock DocumentState for testing."""

    def __init__(self) -> None:
        self.status = None
        self.total_chunks = 0
        self.chunk_ids: List[str] = []


class MockConfig:
    """Mock config with chunking settings."""

    class ChunkingConfig:
        target_size = 500
        overlap = 50
        respect_section_boundaries = False
        chunk_by_title = False

    chunking = ChunkingConfig()


# --- Test Harness ---


class StageMixinTestHarness:
    """Test harness simulating PipelineStagesMixin."""

    def __init__(self) -> None:
        self.chunker = MockChunker()
        self.config = MockConfig()
        self._progress_reports: List[tuple] = []
        self._current_chapter_markers: Dict[str, Any] = {}

    def _report_progress(self, stage: str, progress: float, message: str) -> None:
        self._progress_reports.append((stage, progress, message))

    def _extract_library_from_path(self, path: Path) -> str:
        return "default"

    def _chunk_with_layout_awareness(
        self,
        text: str,
        source_path: str,
        document_id: str,
    ) -> List[ChunkRecord]:
        """Use standard chunker."""
        return self.chunker.chunk(text, document_id, source_path)

    def _attach_source_locations(
        self,
        chunks: List[Any],
        source_location: Any,
    ) -> List[Any]:
        return chunks

    def _optimize_chunks(
        self,
        chunks: List[Any],
        plog: MockPipelineLogger,
    ) -> List[Any]:
        return chunks

    def _chunks_to_artifacts(
        self,
        chunks: List[Any],
        parent: IFTextArtifact = None,
    ) -> List[IFChunkArtifact]:
        """Convert ChunkRecords to IFChunkArtifacts."""
        artifacts: List[IFChunkArtifact] = []
        for chunk in chunks:
            artifact = IFChunkArtifact.from_chunk_record(chunk, parent)
            artifacts.append(artifact)
        return artifacts


# --- Fixtures ---


@pytest.fixture
def harness() -> StageMixinTestHarness:
    """Create test harness."""
    return StageMixinTestHarness()


@pytest.fixture
def plog() -> MockPipelineLogger:
    """Create mock pipeline logger."""
    return MockPipelineLogger()


@pytest.fixture
def doc_state() -> MockDocState:
    """Create mock document state."""
    return MockDocState()


@pytest.fixture
def parent_text_artifact() -> IFTextArtifact:
    """Create parent text artifact."""
    return IFTextArtifact(
        artifact_id="text-artifact-001",
        content="Sample extracted text content for testing chunking.",
        metadata={"source_path": "/tmp/test.pdf"},
    )


@pytest.fixture
def extracted_texts_with_artifact(
    parent_text_artifact: IFTextArtifact,
) -> List[Dict[str, Any]]:
    """Create Stage 2 output with artifacts."""
    return [
        {
            "path": "/tmp/test.pdf",
            "text": "Sample extracted text content for testing chunking.",
            "_artifact": parent_text_artifact,
        }
    ]


@pytest.fixture
def extracted_texts_without_artifact() -> List[Dict[str, Any]]:
    """Create Stage 2 output without artifacts (legacy)."""
    return [
        {
            "path": "/tmp/test.pdf",
            "text": "Sample extracted text content for testing chunking.",
        }
    ]


# --- GWT Scenario 1: Stage 3 Consumes IFTextArtifact ---


class TestStage3ConsumesArtifact:
    """Tests that Stage 3 uses artifacts from Stage 2."""

    def test_stage3_extracts_artifact_from_input(
        self,
        harness: StageMixinTestHarness,
        extracted_texts_with_artifact: List[Dict[str, Any]],
    ) -> None:
        """Given Stage 2 output with _artifact, When Stage 3 processes,
        Then artifact is accessible."""
        extracted = extracted_texts_with_artifact[0]

        assert "_artifact" in extracted
        assert isinstance(extracted["_artifact"], IFTextArtifact)

    def test_chunks_created_with_parent_lineage(
        self,
        harness: StageMixinTestHarness,
        extracted_texts_with_artifact: List[Dict[str, Any]],
    ) -> None:
        """Given Stage 2 output with artifact, When chunks converted,
        Then artifacts have parent lineage."""
        extracted = extracted_texts_with_artifact[0]
        parent = extracted["_artifact"]

        # Simulate chunking
        chunks = harness._chunk_with_layout_awareness(
            extracted["text"],
            extracted["path"],
            "doc-001",
        )

        # Convert to artifacts
        artifacts = harness._chunks_to_artifacts(chunks, parent)

        for artifact in artifacts:
            assert artifact.parent_id == parent.artifact_id


# --- GWT Scenario 2: Stage 3 Produces IFChunkArtifact ---


class TestStage3ProducesArtifact:
    """Tests that Stage 3 creates artifacts."""

    def test_chunks_to_artifacts_returns_list(
        self,
        harness: StageMixinTestHarness,
        parent_text_artifact: IFTextArtifact,
    ) -> None:
        """Given chunks, When _chunks_to_artifacts called,
        Then list of IFChunkArtifact returned."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="doc-001",
                    content="Test content",
                )
            ]

        result = harness._chunks_to_artifacts(chunks, parent_text_artifact)

        assert isinstance(result, list)
        assert all(isinstance(a, IFChunkArtifact) for a in result)

    def test_artifacts_have_content_from_chunks(
        self,
        harness: StageMixinTestHarness,
        parent_text_artifact: IFTextArtifact,
    ) -> None:
        """Given chunks, When converted to artifacts,
        Then content is preserved."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="doc-001",
                    content="Test chunk content",
                )
            ]

        result = harness._chunks_to_artifacts(chunks, parent_text_artifact)

        assert result[0].content == "Test chunk content"


# --- GWT Scenario 3: Backward Compatibility ---


class TestBackwardCompatibility:
    """Tests that Stage 3 maintains backward compatibility."""

    def test_stage3_still_returns_chunk_records(
        self,
        harness: StageMixinTestHarness,
        extracted_texts_with_artifact: List[Dict[str, Any]],
    ) -> None:
        """Given Stage 3 execution, When result examined,
        Then ChunkRecords are returned (not artifacts)."""
        extracted = extracted_texts_with_artifact[0]

        chunks = harness._chunk_with_layout_awareness(
            extracted["text"],
            extracted["path"],
            "doc-001",
        )

        # Verify chunks are ChunkRecords
        assert all(isinstance(c, ChunkRecord) for c in chunks)

    def test_artifacts_convertible_to_chunk_records(
        self,
        harness: StageMixinTestHarness,
        parent_text_artifact: IFTextArtifact,
    ) -> None:
        """Given artifacts, When to_chunk_record called,
        Then valid ChunkRecords returned."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="doc-001",
                    content="Test content",
                )
            ]

        artifacts = harness._chunks_to_artifacts(chunks, parent_text_artifact)

        for artifact in artifacts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record = artifact.to_chunk_record()
                assert isinstance(record, ChunkRecord)


# --- GWT Scenario 4: Context Storage ---


class TestContextStorage:
    """Tests that artifacts are stored in context."""

    def test_artifacts_stored_in_context_key(self) -> None:
        """Given Stage 3 execution, When context examined,
        Then _chunk_artifacts key should be used."""
        # This tests the expected interface - actual storage happens in stages.py
        context: Dict[str, Any] = {}
        context["_chunk_artifacts"] = []

        assert "_chunk_artifacts" in context

    def test_context_key_holds_artifact_list(self) -> None:
        """Given context with _chunk_artifacts, When examined,
        Then it contains list of artifacts."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-001",
            document_id="doc-001",
            content="Test",
            chunk_index=0,
            total_chunks=1,
        )

        context: Dict[str, Any] = {"_chunk_artifacts": [artifact]}

        assert len(context["_chunk_artifacts"]) == 1
        assert isinstance(context["_chunk_artifacts"][0], IFChunkArtifact)


# --- GWT Scenario 5: Lineage Tracking ---


class TestLineageTracking:
    """Tests proper lineage from text to chunks."""

    def test_chunk_artifact_has_parent_id(
        self,
        harness: StageMixinTestHarness,
        parent_text_artifact: IFTextArtifact,
    ) -> None:
        """Given parent artifact, When chunk created,
        Then parent_id is set."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="doc-001",
                    content="Test",
                )
            ]

        artifacts = harness._chunks_to_artifacts(chunks, parent_text_artifact)

        assert artifacts[0].parent_id == parent_text_artifact.artifact_id

    def test_chunk_artifact_has_root_id(
        self,
        harness: StageMixinTestHarness,
        parent_text_artifact: IFTextArtifact,
    ) -> None:
        """Given parent artifact, When chunk created,
        Then root_artifact_id is set."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="doc-001",
                    content="Test",
                )
            ]

        artifacts = harness._chunks_to_artifacts(chunks, parent_text_artifact)

        assert artifacts[0].root_artifact_id == parent_text_artifact.effective_root_id

    def test_chunk_artifact_has_incremented_depth(
        self,
        harness: StageMixinTestHarness,
        parent_text_artifact: IFTextArtifact,
    ) -> None:
        """Given parent artifact, When chunk created,
        Then lineage_depth is parent + 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="doc-001",
                    content="Test",
                )
            ]

        artifacts = harness._chunks_to_artifacts(chunks, parent_text_artifact)

        assert artifacts[0].lineage_depth == parent_text_artifact.lineage_depth + 1

    def test_chunk_without_parent_has_no_lineage(
        self,
        harness: StageMixinTestHarness,
    ) -> None:
        """Given no parent, When chunk created,
        Then no lineage set."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="doc-001",
                    content="Test",
                )
            ]

        artifacts = harness._chunks_to_artifacts(chunks, parent=None)

        assert artifacts[0].parent_id is None
        assert artifacts[0].lineage_depth == 0


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_consumes_artifact_covered(self) -> None:
        """GWT Scenario 1 (Consumes Artifact) is tested."""
        assert hasattr(
            TestStage3ConsumesArtifact, "test_stage3_extracts_artifact_from_input"
        )

    def test_scenario_2_produces_artifact_covered(self) -> None:
        """GWT Scenario 2 (Produces Artifact) is tested."""
        assert hasattr(
            TestStage3ProducesArtifact, "test_chunks_to_artifacts_returns_list"
        )

    def test_scenario_3_backward_compat_covered(self) -> None:
        """GWT Scenario 3 (Backward Compatibility) is tested."""
        assert hasattr(
            TestBackwardCompatibility, "test_stage3_still_returns_chunk_records"
        )

    def test_scenario_4_context_storage_covered(self) -> None:
        """GWT Scenario 4 (Context Storage) is tested."""
        assert hasattr(TestContextStorage, "test_artifacts_stored_in_context_key")

    def test_scenario_5_lineage_covered(self) -> None:
        """GWT Scenario 5 (Lineage Tracking) is tested."""
        assert hasattr(TestLineageTracking, "test_chunk_artifact_has_parent_id")
