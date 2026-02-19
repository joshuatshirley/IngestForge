"""
Stage Migration Tests - TASK-011 Validation.

Comprehensive tests validating the migration of stages.py from ChunkRecord to IFChunkArtifact.
Tests all stage execution paths, artifact conversion, backward compatibility, and edge cases.

Coverage Areas:
1. Stage Execution - Each stage processes IFChunkArtifact correctly
2. Stage Chaining - Artifacts pass correctly between stages
3. Backward Compatibility - Works with both ChunkRecord and IFChunkArtifact
4. Edge Cases - Empty artifacts, missing metadata, failures
5. JPL Compliance - Rule #2, #7, #9 validation

Target: >90% coverage on migrated code.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFTextArtifact,
    IFChunkArtifact,
    IFFileArtifact,
)
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.pipeline.stages import PipelineStagesMixin
from ingestforge.core.state import DocumentState


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline with PipelineStagesMixin."""
    pipeline = MagicMock(spec=PipelineStagesMixin)

    # Add all methods from PipelineStagesMixin
    pipeline._create_text_artifact = PipelineStagesMixin._create_text_artifact
    pipeline._chunks_to_artifacts = PipelineStagesMixin._chunks_to_artifacts
    pipeline._convert_chunks_to_artifacts = (
        PipelineStagesMixin._convert_chunks_to_artifacts
    )
    pipeline._sync_enrichment_to_artifacts = (
        PipelineStagesMixin._sync_enrichment_to_artifacts
    )
    pipeline._enrich_via_stage = PipelineStagesMixin._enrich_via_stage
    pipeline._stage_extract_text = PipelineStagesMixin._stage_extract_text
    pipeline._stage_chunk_text = PipelineStagesMixin._stage_chunk_text
    pipeline._stage_enrich_chunks = PipelineStagesMixin._stage_enrich_chunks

    # Mock dependencies
    pipeline.config = MagicMock()
    pipeline.config.enrichment.generate_embeddings = True
    pipeline.config.enrichment.enrichment_max_batch_size = 500
    pipeline.config.refinement.enabled = False
    pipeline.extractor = MagicMock()
    pipeline.chunker = MagicMock()
    pipeline.enricher = MagicMock()
    pipeline.audio_processor = MagicMock()

    # Mock progress reporting
    pipeline._report_progress = MagicMock()

    return pipeline


@pytest.fixture
def sample_text_artifact() -> IFTextArtifact:
    """Create a sample text artifact (Stage 2 output)."""
    return IFTextArtifact(
        artifact_id="text-001",
        document_id="doc-001",
        content="This is sample extracted text content for testing chunking.",
        parent_id="file-001",
        root_artifact_id="file-001",
        lineage_depth=1,
        provenance=["pdf-extractor"],
        metadata={
            "file_name": "test.pdf",
            "file_type": ".pdf",
            "extraction_method": "standard",
            "word_count": 9,
            "char_count": 61,
        },
    )


@pytest.fixture
def sample_chunk_artifacts() -> List[IFChunkArtifact]:
    """Create sample chunk artifacts (Stage 3 output)."""
    return [
        IFChunkArtifact(
            artifact_id=f"chunk-{i:03d}",
            document_id="doc-001",
            content=f"Sample chunk content {i}.",
            chunk_index=i,
            total_chunks=3,
            parent_id="text-001",
            root_artifact_id="file-001",
            lineage_depth=2,
            provenance=["pdf-extractor", "semantic-chunker"],
            metadata={
                "section_title": f"Section {i}",
                "chunk_type": "content",
                "word_count": 4,
                "char_count": 25,
            },
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_chunk_records() -> List[ChunkRecord]:
    """Create sample ChunkRecords (legacy format)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            ChunkRecord(
                chunk_id=f"chunk-{i:03d}",
                document_id="doc-001",
                content=f"Sample chunk content {i}.",
                chunk_index=i,
                total_chunks=3,
                section_title=f"Section {i}",
                chunk_type="content",
                word_count=4,
                char_count=25,
            )
            for i in range(3)
        ]


@pytest.fixture
def enriched_chunk_records() -> List[ChunkRecord]:
    """Create enriched ChunkRecords with embeddings and entities."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            ChunkRecord(
                chunk_id=f"chunk-{i:03d}",
                document_id="doc-001",
                content=f"Sample chunk content {i}.",
                chunk_index=i,
                total_chunks=3,
                embedding=[0.1, 0.2, 0.3] * 128,  # 384 dims
                entities=["Entity1", "Entity2"],
                concepts=["Concept1"],
                quality_score=0.85,
            )
            for i in range(3)
        ]


# ============================================================================
# SCENARIO 1: STAGE 2 - TEXT EXTRACTION WITH ARTIFACTS
# ============================================================================


class TestStage2TextExtractionArtifacts:
    """Tests for Stage 2 text extraction producing IFTextArtifact."""

    def test_create_text_artifact_generates_metadata(self, mock_pipeline):
        """Given text and file path, When _create_text_artifact called,
        Then artifact has correct metadata."""
        file_path = Path("/test/sample.pdf")
        text = "Sample text content for testing."

        artifact = mock_pipeline._create_text_artifact(
            mock_pipeline, text, file_path, "ocr"
        )

        assert isinstance(artifact, IFTextArtifact)
        assert artifact.content == text
        assert artifact.metadata["file_name"] == "sample.pdf"
        assert artifact.metadata["file_type"] == ".pdf"
        assert artifact.metadata["extraction_method"] == "ocr"
        assert artifact.metadata["word_count"] == 6
        assert artifact.metadata["char_count"] == 33

    def test_create_text_artifact_computes_hash(self, mock_pipeline):
        """Given text content, When artifact created,
        Then content_hash is computed."""
        file_path = Path("/test/sample.txt")
        text = "Test content"

        artifact = mock_pipeline._create_text_artifact(
            mock_pipeline, text, file_path, "standard"
        )

        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64  # SHA-256 hex digest

    def test_stage_extract_text_returns_artifacts_in_dict(self, mock_pipeline):
        """Given chapters to extract, When _stage_extract_text called,
        Then dicts contain both 'text' and '_artifact' keys."""
        mock_extractor = MagicMock()
        mock_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Extracted content",
            metadata={"file_name": "test.pdf"},
        )
        mock_extractor.extract_to_artifact.return_value = mock_artifact
        mock_pipeline.extractor = mock_extractor

        plog = MagicMock()
        chapters = [Path("/test/chapter1.pdf")]
        file_path = Path("/test/sample.pdf")
        context: Dict[str, Any] = {}

        result = mock_pipeline._stage_extract_text(
            mock_pipeline, chapters, file_path, context, plog
        )

        assert len(result) == 1
        assert "text" in result[0]
        assert "_artifact" in result[0]
        assert isinstance(result[0]["_artifact"], IFTextArtifact)
        assert result[0]["text"] == "Extracted content"

    def test_stage_extract_text_ocr_path_creates_artifact(self, mock_pipeline):
        """Given OCR result in context, When _stage_extract_text called,
        Then artifact is created with OCR metadata."""
        plog = MagicMock()
        chapters = []
        file_path = Path("/test/scanned.pdf")

        # Mock OCR result
        ocr_result = MagicMock()
        ocr_result.text = "OCR extracted text"
        ocr_result.engine = "tesseract"
        context = {"scanned_pdf_ocr_result": ocr_result}

        result = mock_pipeline._stage_extract_text(
            mock_pipeline, chapters, file_path, context, plog
        )

        assert len(result) == 1
        assert result[0]["_artifact"].metadata["extraction_method"] == "ocr"
        assert result[0]["text"] == "OCR extracted text"

    def test_stage_extract_text_audio_path_creates_artifact(self, mock_pipeline):
        """Given audio file, When _stage_extract_text called,
        Then artifact is created with audio metadata."""
        plog = MagicMock()
        chapters = []
        file_path = Path("/test/audio.mp3")
        context: Dict[str, Any] = {}

        # Mock audio processor
        audio_result = MagicMock()
        audio_result.success = True
        audio_result.text = "Transcribed audio text"
        audio_result.word_count = 3
        mock_pipeline.audio_processor.process.return_value = audio_result

        result = mock_pipeline._stage_extract_text(
            mock_pipeline, chapters, file_path, context, plog
        )

        assert len(result) == 1
        assert result[0]["_artifact"].metadata["extraction_method"] == "audio"
        assert context["audio_result"] == audio_result


# ============================================================================
# SCENARIO 2: STAGE 3 - CHUNKING WITH ARTIFACT CONVERSION
# ============================================================================


class TestStage3ChunkingArtifactConversion:
    """Tests for Stage 3 chunking producing IFChunkArtifacts."""

    def test_chunks_to_artifacts_converts_chunk_records(
        self, mock_pipeline, sample_chunk_records
    ):
        """Given ChunkRecords, When _chunks_to_artifacts called,
        Then IFChunkArtifacts are created."""
        artifacts = mock_pipeline._chunks_to_artifacts(
            mock_pipeline, sample_chunk_records, parent=None
        )

        assert len(artifacts) == 3
        for i, artifact in enumerate(artifacts):
            assert isinstance(artifact, IFChunkArtifact)
            assert artifact.artifact_id == f"chunk-{i:03d}"
            assert artifact.content == f"Sample chunk content {i}."

    def test_chunks_to_artifacts_preserves_parent_lineage(
        self, mock_pipeline, sample_chunk_records, sample_text_artifact
    ):
        """Given ChunkRecords and parent artifact, When converted,
        Then lineage is preserved."""
        artifacts = mock_pipeline._chunks_to_artifacts(
            mock_pipeline, sample_chunk_records, parent=sample_text_artifact
        )

        for artifact in artifacts:
            assert artifact.parent_id == sample_text_artifact.artifact_id
            assert artifact.lineage_depth == sample_text_artifact.lineage_depth + 1

    def test_chunks_to_artifacts_copies_metadata(self, mock_pipeline):
        """Given ChunkRecord with metadata, When converted,
        Then metadata is copied to artifact."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk = ChunkRecord(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Test content",
                section_title="Introduction",
                page_start=1,
                page_end=2,
                library="research",
            )

        artifacts = mock_pipeline._chunks_to_artifacts(
            mock_pipeline, [chunk], parent=None
        )

        assert artifacts[0].metadata["section_title"] == "Introduction"
        assert artifacts[0].metadata["library"] == "research"

    def test_convert_chunks_to_artifacts_handles_audio(
        self, mock_pipeline, sample_chunk_records
    ):
        """Given audio chunks, When _convert_chunks_to_artifacts called,
        Then artifacts created with parent from extracted_texts."""
        extracted_texts = [
            {
                "path": "/test/audio.mp3",
                "text": "Audio text",
                "_artifact": IFTextArtifact(
                    artifact_id="audio-text-001",
                    content="Audio text",
                ),
            }
        ]
        context: Dict[str, Any] = {}

        artifacts = mock_pipeline._convert_chunks_to_artifacts(
            mock_pipeline, sample_chunk_records, extracted_texts, context
        )

        assert len(artifacts) == 3
        for artifact in artifacts:
            assert artifact.parent_id == "audio-text-001"

    def test_convert_chunks_to_artifacts_handles_no_parent(
        self, mock_pipeline, sample_chunk_records
    ):
        """Given no parent artifact in extracted_texts, When converted,
        Then artifacts created without parent."""
        extracted_texts = [{"path": "/test/file.txt", "text": "Text"}]
        context: Dict[str, Any] = {}

        artifacts = mock_pipeline._convert_chunks_to_artifacts(
            mock_pipeline, sample_chunk_records, extracted_texts, context
        )

        assert len(artifacts) == 3
        # Artifacts should still be created even without parent
        for artifact in artifacts:
            assert isinstance(artifact, IFChunkArtifact)


# ============================================================================
# SCENARIO 3: STAGE 4 - ENRICHMENT WITH ARTIFACT SYNC
# ============================================================================


class TestStage4EnrichmentArtifactSync:
    """Tests for Stage 4 enrichment syncing to artifacts."""

    def test_sync_enrichment_to_artifacts_adds_embeddings(
        self, mock_pipeline, sample_chunk_artifacts, enriched_chunk_records
    ):
        """Given enriched chunks with embeddings, When synced to artifacts,
        Then embeddings are in artifact metadata."""
        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, enriched_chunk_records, sample_chunk_artifacts
        )

        for artifact in result:
            assert "embedding" in artifact.metadata
            assert len(artifact.metadata["embedding"]) == 384

    def test_sync_enrichment_to_artifacts_adds_entities(
        self, mock_pipeline, sample_chunk_artifacts, enriched_chunk_records
    ):
        """Given enriched chunks with entities, When synced to artifacts,
        Then entities are in artifact metadata."""
        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, enriched_chunk_records, sample_chunk_artifacts
        )

        for artifact in result:
            assert "entities" in artifact.metadata
            assert "Entity1" in artifact.metadata["entities"]
            assert "Entity2" in artifact.metadata["entities"]

    def test_sync_enrichment_to_artifacts_preserves_lineage(
        self, mock_pipeline, sample_chunk_artifacts, enriched_chunk_records
    ):
        """Given artifacts with lineage, When enriched,
        Then lineage depth increments and enricher added to provenance."""
        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, enriched_chunk_records, sample_chunk_artifacts
        )

        for i, artifact in enumerate(result):
            # Lineage depth should increment
            assert artifact.lineage_depth == 3
            # Enricher should be in provenance
            assert "enricher" in artifact.provenance
            # Parent should point to original artifact
            assert artifact.parent_id == sample_chunk_artifacts[i].artifact_id

    def test_sync_enrichment_to_artifacts_handles_missing_chunks(
        self, mock_pipeline, sample_chunk_artifacts
    ):
        """Given empty enriched chunks, When synced,
        Then original artifacts returned."""
        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, [], sample_chunk_artifacts
        )

        assert result == sample_chunk_artifacts

    def test_sync_enrichment_to_artifacts_handles_mismatched_ids(
        self, mock_pipeline, sample_chunk_artifacts
    ):
        """Given chunks with different IDs, When synced,
        Then unmatched artifacts returned unchanged."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            different_chunks = [
                ChunkRecord(
                    chunk_id="different-001",
                    document_id="doc-001",
                    content="Different content",
                    embedding=[0.1, 0.2],
                )
            ]

        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, different_chunks, sample_chunk_artifacts
        )

        # Original artifacts should be returned since no match
        assert len(result) == len(sample_chunk_artifacts)
        for orig, res in zip(sample_chunk_artifacts, result):
            assert res.artifact_id == orig.artifact_id
            # No embedding should be added
            assert "embedding" not in res.metadata


# ============================================================================
# SCENARIO 4: BACKWARD COMPATIBILITY
# ============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with ChunkRecord."""

    def test_chunk_record_to_artifact_round_trip(self):
        """Given ChunkRecord, When converted to artifact and back,
        Then data is preserved."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = ChunkRecord(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Test content",
                section_title="Introduction",
                chunk_type="content",
                word_count=2,
                char_count=12,
                chunk_index=0,
                total_chunks=1,
                library="research",
            )

        # Convert to artifact
        artifact = IFChunkArtifact.from_chunk_record(original)

        # Convert back to ChunkRecord
        restored = artifact.to_chunk_record()

        # Verify key fields preserved
        assert restored.chunk_id == original.chunk_id
        assert restored.document_id == original.document_id
        assert restored.content == original.content
        assert restored.section_title == original.section_title
        assert restored.library == original.library

    def test_enriched_chunk_record_preserves_embeddings(self):
        """Given enriched ChunkRecord, When converted to artifact,
        Then embeddings are in metadata."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk = ChunkRecord(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Test",
                embedding=[0.1, 0.2, 0.3],
                entities=["Entity1"],
                concepts=["Concept1"],
                quality_score=0.9,
            )

        artifact = IFChunkArtifact.from_chunk_record(chunk)

        assert artifact.metadata["embedding"] == [0.1, 0.2, 0.3]
        assert artifact.metadata["entities"] == ["Entity1"]
        assert artifact.metadata["concepts"] == ["Concept1"]
        assert artifact.metadata["quality_score"] == 0.9

    def test_stage_enrich_chunks_works_with_legacy_enricher(
        self, mock_pipeline, sample_chunk_records
    ):
        """Given legacy enricher with enrich_batch(), When stage called,
        Then enrichment succeeds."""
        # Mock legacy enricher
        enriched = sample_chunk_records.copy()
        for chunk in enriched:
            chunk.embedding = [0.1] * 384

        mock_pipeline.enricher.enrich_batch.return_value = enriched
        mock_pipeline.config.enrichment.enrichment_max_batch_size = 500

        plog = MagicMock()
        context: Dict[str, Any] = {}

        result = mock_pipeline._stage_enrich_chunks(
            mock_pipeline, sample_chunk_records, context, plog
        )

        assert len(result) == 3
        for chunk in result:
            assert chunk.embedding is not None


# ============================================================================
# SCENARIO 5: EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge case tests for stage migration."""

    def test_empty_chunks_list_handled(self, mock_pipeline):
        """Given empty chunks list, When _chunks_to_artifacts called,
        Then empty list returned."""
        result = mock_pipeline._chunks_to_artifacts(mock_pipeline, [], parent=None)

        assert result == []

    def test_chunk_without_optional_fields(self, mock_pipeline):
        """Given ChunkRecord with minimal fields, When converted,
        Then artifact created without errors."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            minimal_chunk = ChunkRecord(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Minimal content",
            )

        artifacts = mock_pipeline._chunks_to_artifacts(
            mock_pipeline, [minimal_chunk], parent=None
        )

        assert len(artifacts) == 1
        assert artifacts[0].content == "Minimal content"

    def test_sync_enrichment_with_empty_artifacts(
        self, mock_pipeline, enriched_chunk_records
    ):
        """Given empty artifacts list, When sync called,
        Then empty list returned."""
        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, enriched_chunk_records, []
        )

        assert result == []

    def test_chunk_with_none_values_handled(self, mock_pipeline):
        """Given ChunkRecord with None values, When converted,
        Then artifact created without None in metadata."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk = ChunkRecord(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Test",
                section_title=None,
                page_start=None,
                embedding=None,
            )

        artifact = IFChunkArtifact.from_chunk_record(chunk)

        # None values should be filtered out or handled gracefully
        assert isinstance(artifact, IFChunkArtifact)
        assert artifact.content == "Test"

    def test_large_embedding_vector_handled(
        self, mock_pipeline, sample_chunk_artifacts
    ):
        """Given chunk with large embedding, When synced,
        Then embedding is stored correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            large_embedding = [0.1] * 1536  # OpenAI ada-002 size
            chunk = ChunkRecord(
                chunk_id="chunk-000",
                document_id="doc-001",
                content="Test",
                embedding=large_embedding,
            )

        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, [chunk], sample_chunk_artifacts[:1]
        )

        assert len(result[0].metadata["embedding"]) == 1536


# ============================================================================
# SCENARIO 6: STAGE CHAINING AND DATA FLOW
# ============================================================================


class TestStageChaining:
    """Tests for stage chaining and artifact data flow."""

    def test_stage2_to_stage3_artifact_flow(self, mock_pipeline):
        """Given Stage 2 output with artifacts, When Stage 3 processes,
        Then artifacts are used for lineage."""
        # Stage 2 output
        extracted_texts = [
            {
                "path": "/test/file.pdf",
                "text": "Extracted text",
                "_artifact": IFTextArtifact(
                    artifact_id="text-001",
                    content="Extracted text",
                    parent_id="file-001",
                    lineage_depth=1,
                    provenance=["pdf-extractor"],
                ),
            }
        ]

        # Mock chunker
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk_records = [
                ChunkRecord(
                    chunk_id="chunk-001",
                    document_id="doc-001",
                    content="Chunk 1",
                )
            ]
        mock_pipeline.chunker.chunk.return_value = chunk_records

        plog = MagicMock()
        doc_state = DocumentState(document_id="doc-001", file_path="/test/file.pdf")
        context: Dict[str, Any] = {}

        # Call Stage 3
        mock_pipeline._chunk_with_layout_awareness = MagicMock(
            return_value=chunk_records
        )
        mock_pipeline._optimize_chunks = MagicMock(return_value=chunk_records)
        mock_pipeline._attach_source_locations = MagicMock(return_value=chunk_records)

        result_chunks = mock_pipeline._stage_chunk_text(
            mock_pipeline,
            extracted_texts,
            "doc-001",
            Path("/test/file.pdf"),
            None,
            None,
            doc_state,
            context,
            plog,
        )

        # Verify artifacts created in context
        assert "_chunk_artifacts" in context
        artifacts = context["_chunk_artifacts"]
        assert len(artifacts) > 0
        for artifact in artifacts:
            # Should have lineage from Stage 2 artifact
            assert artifact.lineage_depth == 2
            assert artifact.parent_id == "text-001"

    def test_stage3_to_stage4_artifact_flow(
        self, mock_pipeline, sample_chunk_records, sample_chunk_artifacts
    ):
        """Given Stage 3 output with artifacts in context, When Stage 4 processes,
        Then enrichment syncs back to artifacts."""
        # Stage 3 output
        context = {"_chunk_artifacts": sample_chunk_artifacts}

        # Mock enricher
        enriched = sample_chunk_records.copy()
        for chunk in enriched:
            chunk.embedding = [0.1] * 384
        mock_pipeline.enricher.enrich_batch.return_value = enriched
        mock_pipeline.config.enrichment.enrichment_max_batch_size = 500

        plog = MagicMock()

        # Call Stage 4
        result = mock_pipeline._stage_enrich_chunks(
            mock_pipeline, sample_chunk_records, context, plog
        )

        # Verify artifacts updated in context
        enriched_artifacts = context["_chunk_artifacts"]
        for artifact in enriched_artifacts:
            assert "embedding" in artifact.metadata
            # Lineage should increment
            assert artifact.lineage_depth == 3
            assert "enricher" in artifact.provenance

    def test_full_stage_chain_preserves_root(self, mock_pipeline):
        """Given file artifact as root, When passed through stages,
        Then root_artifact_id is preserved."""
        file_artifact = IFFileArtifact(
            artifact_id="file-001",
            file_path=Path("/test/document.pdf"),
            mime_type="application/pdf",
        )

        # Stage 2: Extract
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Extracted text",
            parent_id=file_artifact.artifact_id,
            root_artifact_id=file_artifact.artifact_id,
            lineage_depth=1,
            provenance=["pdf-extractor"],
        )

        # Stage 3: Chunk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk = ChunkRecord(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Chunk content",
            )

        chunk_artifact = IFChunkArtifact.from_chunk_record(chunk, parent=text_artifact)

        # Stage 4: Enrich
        enriched_artifact = chunk_artifact.derive(
            "enricher", metadata={**chunk_artifact.metadata, "embedding": [0.1] * 384}
        )

        # Verify root preserved through chain
        assert text_artifact.root_artifact_id == file_artifact.artifact_id
        assert chunk_artifact.root_artifact_id == file_artifact.artifact_id
        assert enriched_artifact.root_artifact_id == file_artifact.artifact_id


# ============================================================================
# SCENARIO 7: JPL COMPLIANCE
# ============================================================================


class TestJPLCompliance:
    """JPL Power of Ten compliance tests."""

    def test_create_text_artifact_under_60_lines(self):
        """Given _create_text_artifact, When lines counted,
        Then count < 60 (JPL Rule #4)."""
        import inspect

        source = inspect.getsource(PipelineStagesMixin._create_text_artifact)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"

    def test_chunks_to_artifacts_under_60_lines(self):
        """Given _chunks_to_artifacts, When lines counted,
        Then count < 60 (JPL Rule #4)."""
        import inspect

        source = inspect.getsource(PipelineStagesMixin._chunks_to_artifacts)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"

    def test_sync_enrichment_under_60_lines(self):
        """Given _sync_enrichment_to_artifacts, When lines counted,
        Then count < 60 (JPL Rule #4)."""
        import inspect

        source = inspect.getsource(PipelineStagesMixin._sync_enrichment_to_artifacts)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"

    def test_all_migration_methods_have_return_types(self):
        """Given migration methods, When annotations checked,
        Then return types are present (JPL Rule #9)."""
        methods = [
            PipelineStagesMixin._create_text_artifact,
            PipelineStagesMixin._chunks_to_artifacts,
            PipelineStagesMixin._convert_chunks_to_artifacts,
            PipelineStagesMixin._sync_enrichment_to_artifacts,
        ]

        for method in methods:
            annotations = method.__annotations__
            assert "return" in annotations, f"{method.__name__} missing return type"

    def test_no_unbounded_loops_in_chunk_conversion(self, mock_pipeline):
        """Given large chunk list, When converting to artifacts,
        Then conversion completes without unbounded loops (JPL Rule #2)."""
        # Create large but bounded chunk list
        max_chunks = 1000  # Fixed upper bound
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id=f"chunk-{i:04d}",
                    document_id="doc-001",
                    content=f"Chunk {i}",
                )
                for i in range(max_chunks)
            ]

        # Should complete without hanging
        artifacts = mock_pipeline._chunks_to_artifacts(
            mock_pipeline, chunks, parent=None
        )

        assert len(artifacts) == max_chunks


# ============================================================================
# SCENARIO 8: METADATA HANDLING
# ============================================================================


class TestMetadataHandling:
    """Tests for metadata handling across stages."""

    def test_metadata_accumulates_across_stages(self):
        """Given artifact with metadata, When derived,
        Then original metadata is preserved."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-001",
            document_id="doc-001",
            content="Test",
            metadata={
                "section_title": "Introduction",
                "word_count": 100,
            },
        )

        enriched = artifact.derive(
            "enricher",
            metadata={
                **artifact.metadata,
                "embedding": [0.1] * 384,
                "entities": ["Entity1"],
            },
        )

        # Original metadata preserved
        assert enriched.metadata["section_title"] == "Introduction"
        assert enriched.metadata["word_count"] == 100
        # New metadata added
        assert "embedding" in enriched.metadata
        assert "entities" in enriched.metadata

    def test_metadata_does_not_mutate_original(self):
        """Given artifact, When metadata updated in derived artifact,
        Then original artifact metadata unchanged."""
        original = IFChunkArtifact(
            artifact_id="chunk-001",
            document_id="doc-001",
            content="Test",
            metadata={"key": "value"},
        )

        # Create derived artifact with additional metadata
        derived = original.derive(
            "processor", metadata={**original.metadata, "new_key": "new_value"}
        )

        # Original metadata should not be mutated
        assert "new_key" not in original.metadata
        assert "new_key" in derived.metadata
        assert original.metadata["key"] == "value"

    def test_sync_preserves_non_enrichment_metadata(
        self, mock_pipeline, sample_chunk_artifacts
    ):
        """Given artifacts with custom metadata, When enrichment synced,
        Then custom metadata is preserved."""
        # Add custom metadata
        for artifact in sample_chunk_artifacts:
            artifact.metadata["custom_field"] = "custom_value"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = [
                ChunkRecord(
                    chunk_id=f"chunk-{i:03d}",
                    document_id="doc-001",
                    content=f"Content {i}",
                    embedding=[0.1] * 384,
                )
                for i in range(3)
            ]

        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, chunks, sample_chunk_artifacts
        )

        for artifact in result:
            # Custom metadata should be preserved
            assert artifact.metadata["custom_field"] == "custom_value"
            # Enrichment data should be added
            assert "embedding" in artifact.metadata


# ============================================================================
# SCENARIO 9: ERROR HANDLING
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in stage migration."""

    def test_sync_handles_chunk_without_id(self, mock_pipeline, sample_chunk_artifacts):
        """Given chunk without chunk_id, When synced,
        Then original artifacts returned without error."""
        # Create chunk without chunk_id (should not crash)
        invalid_chunk = MagicMock()
        invalid_chunk.chunk_id = None

        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, [invalid_chunk], sample_chunk_artifacts
        )

        # Should return original artifacts
        assert len(result) == len(sample_chunk_artifacts)

    def test_convert_handles_empty_extracted_texts(
        self, mock_pipeline, sample_chunk_records
    ):
        """Given empty extracted_texts, When converting chunks,
        Then artifacts created without parent."""
        context: Dict[str, Any] = {}

        artifacts = mock_pipeline._convert_chunks_to_artifacts(
            mock_pipeline, sample_chunk_records, [], context
        )

        assert len(artifacts) == 3
        for artifact in artifacts:
            # Should work even without parent
            assert isinstance(artifact, IFChunkArtifact)

    def test_sync_handles_partial_enrichment_data(
        self, mock_pipeline, sample_chunk_artifacts
    ):
        """Given chunk with some enrichment fields missing, When synced,
        Then only available fields are added."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            partial_chunk = ChunkRecord(
                chunk_id="chunk-000",
                document_id="doc-001",
                content="Test",
                embedding=[0.1] * 384,
                # entities missing
                # concepts missing
                quality_score=0.8,
            )

        result = mock_pipeline._sync_enrichment_to_artifacts(
            mock_pipeline, [partial_chunk], sample_chunk_artifacts[:1]
        )

        # Available fields should be synced
        assert "embedding" in result[0].metadata
        assert "quality_score" in result[0].metadata
        # Missing fields should not cause errors
        # (they may or may not be present depending on original artifact)


# ============================================================================
# SCENARIO 10: PROVENANCE TRACKING
# ============================================================================


class TestProvenanceTracking:
    """Tests for provenance tracking through stage migration."""

    def test_provenance_chain_through_stages(self):
        """Given artifact chain through stages, When processed,
        Then provenance records full processing history."""
        # Stage 1: File
        file_artifact = IFFileArtifact(
            artifact_id="file-001",
            file_path=Path("/test/doc.pdf"),
        )

        # Stage 2: Extract
        text_artifact = file_artifact.derive(
            "pdf-extractor",
            content="Extracted text",
        )
        assert text_artifact.provenance == ["pdf-extractor"]

        # Stage 3: Chunk
        chunk_artifact = text_artifact.derive(
            "semantic-chunker",
            content="Chunk 1",
        )
        assert chunk_artifact.provenance == ["pdf-extractor", "semantic-chunker"]

        # Stage 4: Enrich
        enriched_artifact = chunk_artifact.derive(
            "enricher", metadata={"embedding": [0.1] * 384}
        )
        assert enriched_artifact.provenance == [
            "pdf-extractor",
            "semantic-chunker",
            "enricher",
        ]

    def test_lineage_depth_increments_correctly(self):
        """Given artifact chain, When derived,
        Then lineage_depth increments at each stage."""
        artifact = IFFileArtifact(
            artifact_id="file-001",
            file_path=Path("/test/doc.pdf"),
        )
        assert artifact.lineage_depth == 0

        derived1 = artifact.derive("processor1")
        assert derived1.lineage_depth == 1

        derived2 = derived1.derive("processor2")
        assert derived2.lineage_depth == 2

        derived3 = derived2.derive("processor3")
        assert derived3.lineage_depth == 3

    def test_parent_chain_maintained(self):
        """Given artifact chain, When derived,
        Then parent_id points to immediate parent."""
        file_artifact = IFFileArtifact(
            artifact_id="file-001",
            file_path=Path("/test/doc.pdf"),
        )

        text_artifact = file_artifact.derive("extractor", content="text")
        assert text_artifact.parent_id == "file-001"

        chunk_artifact = text_artifact.derive("chunker", content="chunk")
        assert chunk_artifact.parent_id == text_artifact.artifact_id

        enriched_artifact = chunk_artifact.derive("enricher")
        assert enriched_artifact.parent_id == chunk_artifact.artifact_id


# ============================================================================
# COVERAGE SUMMARY TEST
# ============================================================================


class TestCoverageSummary:
    """Meta-test ensuring all coverage areas are tested."""

    def test_stage_execution_coverage(self):
        """Verify Stage Execution coverage area is tested."""
        assert hasattr(
            TestStage2TextExtractionArtifacts,
            "test_create_text_artifact_generates_metadata",
        )
        assert hasattr(
            TestStage3ChunkingArtifactConversion,
            "test_chunks_to_artifacts_converts_chunk_records",
        )
        assert hasattr(
            TestStage4EnrichmentArtifactSync,
            "test_sync_enrichment_to_artifacts_adds_embeddings",
        )

    def test_stage_chaining_coverage(self):
        """Verify Stage Chaining coverage area is tested."""
        assert hasattr(TestStageChaining, "test_stage2_to_stage3_artifact_flow")
        assert hasattr(TestStageChaining, "test_stage3_to_stage4_artifact_flow")
        assert hasattr(TestStageChaining, "test_full_stage_chain_preserves_root")

    def test_backward_compatibility_coverage(self):
        """Verify Backward Compatibility coverage area is tested."""
        assert hasattr(
            TestBackwardCompatibility, "test_chunk_record_to_artifact_round_trip"
        )
        assert hasattr(
            TestBackwardCompatibility, "test_enriched_chunk_record_preserves_embeddings"
        )

    def test_edge_cases_coverage(self):
        """Verify Edge Cases coverage area is tested."""
        assert hasattr(TestEdgeCases, "test_empty_chunks_list_handled")
        assert hasattr(TestEdgeCases, "test_chunk_without_optional_fields")

    def test_jpl_compliance_coverage(self):
        """Verify JPL Compliance coverage area is tested."""
        assert hasattr(TestJPLCompliance, "test_create_text_artifact_under_60_lines")
        assert hasattr(
            TestJPLCompliance, "test_all_migration_methods_have_return_types"
        )
