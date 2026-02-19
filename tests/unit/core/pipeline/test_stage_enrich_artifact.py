"""
Stage 4 (Enrich) Artifact Integration Tests.

GWT-style tests verifying f implementation:
- Stage 4 reads artifacts from context
- Enrichment synced back to artifacts
- Lineage preserved through enrichment
- Backward compatibility with ChunkRecord
"""

import warnings
from typing import List
from unittest.mock import MagicMock

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
)
from ingestforge.chunking.semantic_chunker import ChunkRecord


# --- Fixtures ---


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock config for testing."""
    config = MagicMock()
    config.enrichment.generate_embeddings = True
    config.enrichment.enrichment_max_batch_size = 500
    return config


@pytest.fixture
def sample_artifacts() -> List[IFChunkArtifact]:
    """Create sample artifacts as Stage 3 output."""
    return [
        IFChunkArtifact(
            artifact_id=f"chunk-{i:03d}",
            document_id="doc-001",
            content=f"Sample content for chunk {i}.",
            chunk_index=i,
            total_chunks=3,
            parent_id="text-001",
            root_artifact_id="file-001",
            lineage_depth=2,
            provenance=["pdf-extractor", "semantic-chunker"],
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_chunk_records() -> List[ChunkRecord]:
    """Create sample ChunkRecords matching the artifacts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            ChunkRecord(
                chunk_id=f"chunk-{i:03d}",
                document_id="doc-001",
                content=f"Sample content for chunk {i}.",
                chunk_index=i,
                total_chunks=3,
            )
            for i in range(3)
        ]


@pytest.fixture
def enriched_chunk_records() -> List[ChunkRecord]:
    """Create enriched ChunkRecords with embedding and entities."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            ChunkRecord(
                chunk_id=f"chunk-{i:03d}",
                document_id="doc-001",
                content=f"Sample content for chunk {i}.",
                chunk_index=i,
                total_chunks=3,
                embedding=[0.1, 0.2, 0.3] * 128,  # 384 dims
                entities=["Entity1", "Entity2"],
                concepts=["Concept1"],
                quality_score=0.85,
            )
            for i in range(3)
        ]


# --- GWT Scenario 1: Stage 4 Reads Artifacts from Context ---


class TestStage4ReadsArtifactsFromContext:
    """Tests for reading artifacts from context."""

    def test_sync_enrichment_accesses_context_artifacts(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given artifacts in context, When _sync_enrichment_to_artifacts called,
        Then it processes the context artifacts."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        # Create mock stages instance
        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        # Call the method
        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        # Verify result
        assert len(result) == len(sample_artifacts)

    def test_empty_context_artifacts_handled(
        self,
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given empty artifacts list, When _sync_enrichment_to_artifacts called,
        Then empty list returned."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, []
        )

        assert result == []


# --- GWT Scenario 2: Enriched Artifacts Preserve Lineage ---


class TestEnrichedArtifactsPreserveLineage:
    """Tests for lineage preservation through enrichment."""

    def test_enriched_artifact_has_provenance_entry(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given artifact with provenance, When enriched,
        Then provenance includes 'enricher'."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            assert "enricher" in enriched.provenance

    def test_enriched_artifact_preserves_parent_id(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given artifact with parent_id, When enriched,
        Then parent_id chain is preserved."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            # parent_id now points to original artifact (derived pattern)
            assert enriched.parent_id is not None

    def test_enriched_artifact_preserves_root_id(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given artifact with root_artifact_id, When enriched,
        Then root remains the original root."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            # Root should be preserved from derive()
            assert enriched.root_artifact_id == "file-001"

    def test_enriched_artifact_increments_lineage_depth(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given artifact with lineage_depth=2, When enriched,
        Then depth increments to 3."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            assert enriched.lineage_depth == 3


# --- GWT Scenario 3: Embedding Added to Artifact Metadata ---


class TestEmbeddingAddedToMetadata:
    """Tests for embedding storage in metadata."""

    def test_embedding_stored_in_metadata(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given enriched chunk with embedding, When artifact created,
        Then embedding is in metadata."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            assert "embedding" in enriched.metadata
            assert len(enriched.metadata["embedding"]) == 384

    def test_entities_stored_in_metadata(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given enriched chunk with entities, When artifact created,
        Then entities are in metadata."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            assert "entities" in enriched.metadata
            assert "Entity1" in enriched.metadata["entities"]

    def test_quality_score_stored_in_metadata(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given enriched chunk with quality_score, When artifact created,
        Then quality_score is in metadata."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            assert enriched.metadata.get("quality_score") == 0.85


# --- GWT Scenario 4: Backward Compatibility with ChunkRecord ---


class TestBackwardCompatibility:
    """Tests for ChunkRecord backward compatibility."""

    def test_chunk_records_unchanged_after_sync(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given enriched ChunkRecords, When sync called,
        Then original ChunkRecords are not modified."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        # Store original state
        original_embeddings = [c.embedding for c in enriched_chunk_records]

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        _ = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        # Verify ChunkRecords unchanged
        for i, chunk in enumerate(enriched_chunk_records):
            assert chunk.embedding == original_embeddings[i]

    def test_mismatched_chunk_ids_handled(
        self,
        sample_artifacts: List[IFChunkArtifact],
    ) -> None:
        """Given ChunkRecords with different IDs, When sync called,
        Then unmatched artifacts returned as-is."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            different_chunks = [
                ChunkRecord(
                    chunk_id="different-id",
                    document_id="doc-001",
                    content="Different content",
                )
            ]

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, different_chunks, sample_artifacts
        )

        # Original artifacts returned since no match
        assert len(result) == len(sample_artifacts)
        for orig, res in zip(sample_artifacts, result):
            assert res.artifact_id == orig.artifact_id


# --- GWT Scenario 5: Multiple Enrichers Chained ---


class TestMultipleEnrichersChained:
    """Tests for chained enrichment scenarios."""

    def test_all_enrichment_fields_synced(
        self,
        sample_artifacts: List[IFChunkArtifact],
        enriched_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given chunk with multiple enrichment fields, When synced,
        Then all fields present in artifact metadata."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, enriched_chunk_records, sample_artifacts
        )

        for enriched in result:
            assert "embedding" in enriched.metadata
            assert "entities" in enriched.metadata
            assert "concepts" in enriched.metadata
            assert "quality_score" in enriched.metadata


# --- Edge Cases ---


class TestEnrichmentEdgeCases:
    """Edge case tests for enrichment artifact sync."""

    def test_chunk_without_embedding_handled(
        self,
        sample_artifacts: List[IFChunkArtifact],
    ) -> None:
        """Given chunk without embedding, When synced,
        Then artifact created without embedding in metadata."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks_no_embedding = [
                ChunkRecord(
                    chunk_id=f"chunk-{i:03d}",
                    document_id="doc-001",
                    content=f"Content {i}",
                )
                for i in range(3)
            ]

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(
            stages, chunks_no_embedding, sample_artifacts
        )

        for enriched in result:
            # No embedding added
            assert enriched.metadata.get("embedding") is None

    def test_empty_enriched_chunks_handled(
        self,
        sample_artifacts: List[IFChunkArtifact],
    ) -> None:
        """Given empty enriched chunks list, When synced,
        Then original artifacts returned."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        stages = MagicMock(spec=PipelineStagesMixin)
        stages._sync_enrichment_to_artifacts = (
            PipelineStagesMixin._sync_enrichment_to_artifacts
        )

        result = stages._sync_enrichment_to_artifacts(stages, [], sample_artifacts)

        assert result == sample_artifacts


# --- JPL Compliance Tests ---


class TestJPLComplianceEnrich:
    """JPL Power of Ten compliance tests for Stage 4."""

    def test_sync_function_under_60_lines(self) -> None:
        """Given _sync_enrichment_to_artifacts, When lines counted,
        Then count < 60."""
        import inspect
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        source = inspect.getsource(PipelineStagesMixin._sync_enrichment_to_artifacts)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"

    def test_enrich_in_batches_under_60_lines(self) -> None:
        """Given _enrich_in_batches, When lines counted,
        Then count < 60."""
        import inspect
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        source = inspect.getsource(PipelineStagesMixin._enrich_in_batches)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"

    def test_sync_has_return_type(self) -> None:
        """Given _sync_enrichment_to_artifacts, When annotations checked,
        Then return type is present."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        annotations = PipelineStagesMixin._sync_enrichment_to_artifacts.__annotations__
        assert "return" in annotations


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_reads_context_covered(self) -> None:
        """GWT Scenario 1 (Reads Context) is tested."""
        assert hasattr(
            TestStage4ReadsArtifactsFromContext,
            "test_sync_enrichment_accesses_context_artifacts",
        )

    def test_scenario_2_lineage_preserved_covered(self) -> None:
        """GWT Scenario 2 (Lineage Preserved) is tested."""
        assert hasattr(
            TestEnrichedArtifactsPreserveLineage,
            "test_enriched_artifact_has_provenance_entry",
        )

    def test_scenario_3_embedding_metadata_covered(self) -> None:
        """GWT Scenario 3 (Embedding Metadata) is tested."""
        assert hasattr(
            TestEmbeddingAddedToMetadata, "test_embedding_stored_in_metadata"
        )

    def test_scenario_4_backward_compat_covered(self) -> None:
        """GWT Scenario 4 (Backward Compatibility) is tested."""
        assert hasattr(
            TestBackwardCompatibility, "test_chunk_records_unchanged_after_sync"
        )

    def test_scenario_5_chained_enrichers_covered(self) -> None:
        """GWT Scenario 5 (Chained Enrichers) is tested."""
        assert hasattr(
            TestMultipleEnrichersChained, "test_all_enrichment_fields_synced"
        )


# --- BUG001: IFStage vs Legacy Enricher Routing ---


class TestBUG001IFStageEnricherRouting:
    """
    Tests for BUG001 fix: IFEnrichmentStage vs legacy enricher routing.

    BUG001: Stage 4 crashed with AttributeError when calling enrich_batch()
    on IFEnrichmentStage (which uses execute() interface instead).
    """

    def test_ifstage_detected_for_routing(self) -> None:
        """Given IFEnrichmentStage enricher, When type checked,
        Then it is recognized as IFStage."""
        from ingestforge.core.pipeline.interfaces import IFStage
        from ingestforge.core.pipeline.enrichment_stage import IFEnrichmentStage

        stage = IFEnrichmentStage(processors=[], stage_name="test")
        assert isinstance(stage, IFStage)

    def test_legacy_enricher_not_ifstage(self) -> None:
        """Given _NoOpEnricher, When type checked,
        Then it is NOT recognized as IFStage."""
        from ingestforge.core.pipeline.interfaces import IFStage

        class MockLegacyEnricher:
            def enrich_batch(self, chunks):
                return chunks

        enricher = MockLegacyEnricher()
        assert not isinstance(enricher, IFStage)

    def test_enrich_via_stage_converts_chunk_record(
        self,
        sample_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given ChunkRecords, When _enrich_via_stage called,
        Then each record is converted and processed."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin
        from ingestforge.core.pipeline.enrichment_stage import IFEnrichmentStage

        # Create a mock stage that returns input unchanged
        mock_stage = IFEnrichmentStage(processors=[], stage_name="test")

        stages = MagicMock(spec=PipelineStagesMixin)
        stages.enricher = mock_stage
        stages._enrich_via_stage = PipelineStagesMixin._enrich_via_stage

        result = stages._enrich_via_stage(stages, sample_chunk_records)

        assert len(result) == len(sample_chunk_records)
        for i, chunk in enumerate(result):
            assert chunk.document_id == sample_chunk_records[i].document_id
            assert chunk.content == sample_chunk_records[i].content

    def test_enrich_via_stage_returns_chunk_records(
        self,
        sample_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given ChunkRecords processed via IFStage, When returned,
        Then result is List[ChunkRecord] for Stage 5 compatibility."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin
        from ingestforge.core.pipeline.enrichment_stage import IFEnrichmentStage
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        mock_stage = IFEnrichmentStage(processors=[], stage_name="test")

        stages = MagicMock(spec=PipelineStagesMixin)
        stages.enricher = mock_stage
        stages._enrich_via_stage = PipelineStagesMixin._enrich_via_stage

        result = stages._enrich_via_stage(stages, sample_chunk_records)

        for chunk in result:
            assert isinstance(chunk, ChunkRecord)

    def test_enrich_via_stage_handles_failure_artifact(
        self,
        sample_chunk_records: List[ChunkRecord],
    ) -> None:
        """Given IFStage that returns IFFailureArtifact, When processed,
        Then original chunk is kept (graceful degradation)."""
        from ingestforge.core.pipeline.stages import PipelineStagesMixin
        from ingestforge.core.pipeline.interfaces import IFStage, IFArtifact
        from ingestforge.core.pipeline.artifacts import IFFailureArtifact

        # Create a stage that always returns failure
        class FailingStage(IFStage):
            @property
            def name(self) -> str:
                return "failing"

            @property
            def input_type(self):
                return IFArtifact

            @property
            def output_type(self):
                return IFArtifact

            def execute(self, artifact: IFArtifact) -> IFArtifact:
                return IFFailureArtifact(
                    artifact_id="fail-001",
                    error_message="Test failure",
                    failed_processor_id="test",
                )

        stages = MagicMock(spec=PipelineStagesMixin)
        stages.enricher = FailingStage()
        stages._enrich_via_stage = PipelineStagesMixin._enrich_via_stage

        result = stages._enrich_via_stage(stages, sample_chunk_records)

        # Original chunks should be returned when failure occurs
        assert len(result) == len(sample_chunk_records)
        for i, chunk in enumerate(result):
            assert chunk.content == sample_chunk_records[i].content

    def test_enrich_via_stage_jpl_under_60_lines(self) -> None:
        """Given _enrich_via_stage, When lines counted,
        Then count < 60 (JPL Rule #4)."""
        import inspect
        from ingestforge.core.pipeline.stages import PipelineStagesMixin

        source = inspect.getsource(PipelineStagesMixin._enrich_via_stage)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Function has {len(lines)} lines"
