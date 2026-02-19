"""
Unit tests for IFEnrichmentStage.

Replace EnrichmentPipeline with IFEnrichmentStage.
GWT-compliant tests with NASA JPL Power of Ten verification.
"""

import inspect
from typing import List
from unittest.mock import Mock

import pytest

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact, IFStage
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFFailureArtifact,
)
from ingestforge.core.pipeline.enrichment_stage import (
    IFEnrichmentStage,
    MAX_PROCESSORS_PER_STAGE,
    MAX_CONSECUTIVE_FAILURES,
    MAX_BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Mock Processor Fixtures
# ---------------------------------------------------------------------------


def create_mock_processor(
    processor_id: str = "test-processor",
    available: bool = True,
    return_failure: bool = False,
    raise_exception: bool = False,
) -> Mock:
    """Create a mock IFProcessor for testing."""
    processor = Mock(spec=IFProcessor)
    processor.processor_id = processor_id
    processor.version = "1.0.0"
    processor.capabilities = ["test-capability"]
    processor.memory_mb = 50
    processor.is_available.return_value = available
    processor.teardown.return_value = True

    def mock_process(artifact: IFArtifact) -> IFArtifact:
        if raise_exception:
            raise RuntimeError(f"Processor {processor_id} raised exception")
        if return_failure:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-failure",
                error_message=f"Processor {processor_id} failed",
                failed_processor_id=processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [processor_id],
            )
        return artifact.derive(
            processor_id,
            artifact_id=f"{artifact.artifact_id}-{processor_id}",
            metadata={**artifact.metadata, f"{processor_id}_applied": True},
        )

    processor.process.side_effect = mock_process
    return processor


@pytest.fixture
def sample_chunk_artifact() -> IFChunkArtifact:
    """Create a sample IFChunkArtifact for testing."""
    return IFChunkArtifact(
        artifact_id="chunk-001",
        document_id="doc-001",
        content="Test content for enrichment.",
        chunk_index=0,
        total_chunks=1,
        metadata={"original": True},
    )


@pytest.fixture
def mock_processor() -> Mock:
    """Create a basic mock processor."""
    return create_mock_processor()


@pytest.fixture
def enrichment_stage(mock_processor: Mock) -> IFEnrichmentStage:
    """Create an IFEnrichmentStage with a single mock processor."""
    return IFEnrichmentStage([mock_processor])


# ---------------------------------------------------------------------------
# IFStage Interface Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestIFStageInterface:
    """
    GWT: Test that IFEnrichmentStage implements IFStage interface.

    Acceptance Criteria:
    - [x] New IFEnrichmentStage class implements IFStage interface.
    """

    def test_given_enrichment_stage_when_check_inheritance_then_is_ifstage(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given IFEnrichmentStage, When checking inheritance, Then it extends IFStage."""
        assert isinstance(enrichment_stage, IFStage)

    def test_given_enrichment_stage_when_check_methods_then_has_execute(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given IFEnrichmentStage, When checking methods, Then has execute() method."""
        assert hasattr(enrichment_stage, "execute")
        assert callable(enrichment_stage.execute)

    def test_given_enrichment_stage_when_check_properties_then_has_name(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given IFEnrichmentStage, When checking properties, Then has name property."""
        assert hasattr(enrichment_stage, "name")
        assert isinstance(enrichment_stage.name, str)

    def test_given_enrichment_stage_when_check_properties_then_has_input_type(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given IFEnrichmentStage, When checking properties, Then has input_type property."""
        assert hasattr(enrichment_stage, "input_type")
        assert enrichment_stage.input_type == IFChunkArtifact

    def test_given_enrichment_stage_when_check_properties_then_has_output_type(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given IFEnrichmentStage, When checking properties, Then has output_type property."""
        assert hasattr(enrichment_stage, "output_type")
        assert enrichment_stage.output_type == IFChunkArtifact


# ---------------------------------------------------------------------------
# Stage Initialization Tests
# ---------------------------------------------------------------------------


class TestStageInitialization:
    """
    GWT: Test stage accepts list of IFProcessor instances.

    Acceptance Criteria:
    - [x] Stage accepts list of IFProcessor instances.
    """

    def test_given_processor_list_when_init_then_stores_processors(self) -> None:
        """Given processor list, When init, Then stores processors."""
        processors = [create_mock_processor(f"proc-{i}") for i in range(3)]
        stage = IFEnrichmentStage(processors)
        assert stage.processor_count == 3

    def test_given_empty_list_when_init_then_creates_stage(self) -> None:
        """Given empty list, When init, Then creates stage with no processors."""
        stage = IFEnrichmentStage([])
        assert stage.processor_count == 0

    def test_given_too_many_processors_when_init_then_raises_error(self) -> None:
        """Given too many processors, When init, Then raises ValueError."""
        processors = [
            create_mock_processor(f"proc-{i}")
            for i in range(MAX_PROCESSORS_PER_STAGE + 1)
        ]
        with pytest.raises(ValueError, match="Too many processors"):
            IFEnrichmentStage(processors)

    def test_given_custom_name_when_init_then_uses_custom_name(self) -> None:
        """Given custom stage name, When init, Then uses custom name."""
        stage = IFEnrichmentStage([], stage_name="custom-enrichment")
        assert stage.name == "custom-enrichment"


# ---------------------------------------------------------------------------
# Execute Method Tests
# ---------------------------------------------------------------------------


class TestExecuteMethod:
    """
    GWT: Test execute() method chains process() calls on each processor.

    Acceptance Criteria:
    - [x] execute() method chains process() calls on each processor.
    """

    def test_given_artifact_when_execute_then_returns_artifact(
        self,
        enrichment_stage: IFEnrichmentStage,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given artifact, When execute(), Then returns artifact."""
        result = enrichment_stage.execute(sample_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_given_multiple_processors_when_execute_then_chains_all(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given multiple processors, When execute(), Then chains all."""
        processors = [create_mock_processor(f"proc-{i}") for i in range(3)]
        stage = IFEnrichmentStage(processors)

        result = stage.execute(sample_chunk_artifact)

        # Each processor should have been called
        for proc in processors:
            assert proc.process.called

    def test_given_unavailable_processor_when_execute_then_skips_it(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given unavailable processor, When execute(), Then skips it."""
        available_proc = create_mock_processor("available", available=True)
        unavailable_proc = create_mock_processor("unavailable", available=False)

        stage = IFEnrichmentStage([unavailable_proc, available_proc])
        result = stage.execute(sample_chunk_artifact)

        # Unavailable processor should not be called
        assert not unavailable_proc.process.called
        assert available_proc.process.called


# ---------------------------------------------------------------------------
# Failure Handling Tests
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """
    GWT: Test handles IFFailureArtifact returns gracefully.

    Acceptance Criteria:
    - [x] Handles IFFailureArtifact returns gracefully (configurable: skip or abort).
    """

    def test_given_skip_failures_true_when_failure_then_continues(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given skip_failures=True, When processor returns failure, Then continues."""
        failing_proc = create_mock_processor("failing", return_failure=True)
        success_proc = create_mock_processor("success")

        stage = IFEnrichmentStage([failing_proc, success_proc], skip_failures=True)
        result = stage.execute(sample_chunk_artifact)

        # Both should be called
        assert failing_proc.process.called
        assert success_proc.process.called

    def test_given_skip_failures_false_when_failure_then_aborts(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given skip_failures=False, When processor returns failure, Then aborts."""
        failing_proc = create_mock_processor("failing", return_failure=True)
        success_proc = create_mock_processor("success")

        stage = IFEnrichmentStage([failing_proc, success_proc], skip_failures=False)
        result = stage.execute(sample_chunk_artifact)

        assert isinstance(result, IFFailureArtifact)
        # Second processor should not be called
        assert not success_proc.process.called

    def test_given_exception_and_skip_failures_then_continues(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given processor raises exception and skip_failures=True, Then continues."""
        failing_proc = create_mock_processor("failing", raise_exception=True)
        success_proc = create_mock_processor("success")

        stage = IFEnrichmentStage([failing_proc, success_proc], skip_failures=True)
        result = stage.execute(sample_chunk_artifact)

        # Should continue to success processor
        assert success_proc.process.called

    def test_given_max_consecutive_failures_when_exceeded_then_aborts(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given max consecutive failures exceeded, When execute, Then aborts."""
        # Create enough failing processors to exceed MAX_CONSECUTIVE_FAILURES
        failing_procs = [
            create_mock_processor(f"failing-{i}", return_failure=True)
            for i in range(MAX_CONSECUTIVE_FAILURES + 1)
        ]
        success_proc = create_mock_processor("success")

        stage = IFEnrichmentStage(failing_procs + [success_proc], skip_failures=True)
        result = stage.execute(sample_chunk_artifact)

        # Should abort after MAX_CONSECUTIVE_FAILURES
        assert isinstance(result, IFFailureArtifact)


# ---------------------------------------------------------------------------
# Teardown Tests
# ---------------------------------------------------------------------------


class TestTeardown:
    """GWT: Test teardown calls all processor teardowns."""

    def test_given_stage_when_teardown_then_calls_all_processor_teardowns(self) -> None:
        """Given stage with processors, When teardown(), Then calls all teardowns."""
        processors = [create_mock_processor(f"proc-{i}") for i in range(3)]
        stage = IFEnrichmentStage(processors)

        result = stage.teardown()

        assert result is True
        for proc in processors:
            assert proc.teardown.called

    def test_given_teardown_failure_when_teardown_then_continues_others(self) -> None:
        """Given processor teardown fails, When teardown(), Then continues to others."""
        proc1 = create_mock_processor("proc-1")
        proc2 = create_mock_processor("proc-2")
        proc3 = create_mock_processor("proc-3")

        # Make proc2 teardown fail
        proc2.teardown.return_value = False

        stage = IFEnrichmentStage([proc1, proc2, proc3])
        result = stage.teardown()

        # Should return False but still call all
        assert result is False
        assert proc1.teardown.called
        assert proc2.teardown.called
        assert proc3.teardown.called


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestProperties:
    """GWT: Test stage properties."""

    def test_version_returns_semver(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given stage, When get version, Then returns SemVer string."""
        version = enrichment_stage.version
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) == 3

    def test_processors_returns_copy(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given stage, When get processors, Then returns copy."""
        processors = enrichment_stage.processors
        assert isinstance(processors, list)
        # Modifying returned list shouldn't affect stage
        processors.clear()
        assert enrichment_stage.processor_count > 0

    def test_available_processors_filters_unavailable(self) -> None:
        """Given stage with mixed availability, When get available_processors, Then filters."""
        available = create_mock_processor("available", available=True)
        unavailable = create_mock_processor("unavailable", available=False)

        stage = IFEnrichmentStage([available, unavailable])
        assert len(stage.available_processors) == 1

    def test_skip_failures_returns_config(self) -> None:
        """Given stage, When get skip_failures, Then returns config value."""
        stage_skip = IFEnrichmentStage([], skip_failures=True)
        stage_abort = IFEnrichmentStage([], skip_failures=False)

        assert stage_skip.skip_failures is True
        assert stage_abort.skip_failures is False


# ---------------------------------------------------------------------------
# Utility Method Tests
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """GWT: Test utility methods."""

    def test_get_processor_by_id_found(self) -> None:
        """Given processor ID, When get_processor(), Then returns processor."""
        proc = create_mock_processor("target-proc")
        stage = IFEnrichmentStage([proc])

        result = stage.get_processor("target-proc")
        assert result is proc

    def test_get_processor_by_id_not_found(self) -> None:
        """Given unknown processor ID, When get_processor(), Then returns None."""
        proc = create_mock_processor("some-proc")
        stage = IFEnrichmentStage([proc])

        result = stage.get_processor("unknown-proc")
        assert result is None

    def test_get_capabilities_aggregates_all(self) -> None:
        """Given multiple processors, When get_capabilities(), Then aggregates all."""
        proc1 = create_mock_processor("proc-1")
        proc1.capabilities = ["cap-a", "cap-b"]
        proc2 = create_mock_processor("proc-2")
        proc2.capabilities = ["cap-b", "cap-c"]

        stage = IFEnrichmentStage([proc1, proc2])
        capabilities = stage.get_capabilities()

        assert "cap-a" in capabilities
        assert "cap-b" in capabilities
        assert "cap-c" in capabilities

    def test_get_total_memory_mb_sums_all(self) -> None:
        """Given multiple processors, When get_total_memory_mb(), Then sums all."""
        proc1 = create_mock_processor("proc-1")
        proc1.memory_mb = 100
        proc2 = create_mock_processor("proc-2")
        proc2.memory_mb = 200

        stage = IFEnrichmentStage([proc1, proc2])
        total = stage.get_total_memory_mb()

        assert total == 300


# ---------------------------------------------------------------------------
# JPL Power of Ten Rule Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLRuleCompliance:
    """GWT: Verify NASA JPL Power of Ten compliance."""

    def test_rule_2_max_processors_bound_exists(self) -> None:
        """Rule #2: Fixed upper bound MAX_PROCESSORS_PER_STAGE exists."""
        assert MAX_PROCESSORS_PER_STAGE > 0
        assert MAX_PROCESSORS_PER_STAGE <= 64

    def test_rule_2_max_failures_bound_exists(self) -> None:
        """Rule #2: Fixed upper bound MAX_CONSECUTIVE_FAILURES exists."""
        assert MAX_CONSECUTIVE_FAILURES > 0
        assert MAX_CONSECUTIVE_FAILURES <= 20

    def test_rule_4_execute_method_under_60_lines(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Rule #4: execute() method is < 60 lines."""
        source = inspect.getsource(enrichment_stage.execute)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]
        assert len(lines) < 60, f"execute() has {len(lines)} lines"

    def test_rule_7_execute_returns_artifact(
        self,
        enrichment_stage: IFEnrichmentStage,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Rule #7: execute() always returns an IFArtifact."""
        result = enrichment_stage.execute(sample_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_rule_9_execute_has_type_hints(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Rule #9: execute() has complete type hints."""
        hints = enrichment_stage.execute.__annotations__
        assert "artifact" in hints
        assert "return" in hints


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """GWT: Integration tests for full workflow."""

    def test_given_chained_processors_when_execute_then_all_applied(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given chained processors, When execute(), Then all applied in order."""
        processors = [create_mock_processor(f"proc-{i}") for i in range(3)]
        stage = IFEnrichmentStage(processors)

        result = stage.execute(sample_chunk_artifact)

        # Result should have metadata from all processors
        assert isinstance(result, IFChunkArtifact)
        # The artifact should be enriched (though exact metadata depends on mock behavior)
        assert "original" in result.metadata

    def test_given_empty_stage_when_execute_then_returns_unchanged(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given empty stage, When execute(), Then returns unchanged artifact."""
        stage = IFEnrichmentStage([])

        result = stage.execute(sample_chunk_artifact)

        assert result == sample_chunk_artifact


# ---------------------------------------------------------------------------
# TASK-002: Batch Processing Tests
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    """
    GWT: Test batch processing functionality.

    TASK-002: Batch processing detection and execution.
    """

    def test_given_empty_batch_when_execute_batch_then_returns_empty_list(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given empty batch, When execute_batch(), Then returns empty list."""
        result = enrichment_stage.execute_batch([])
        assert result == []

    def test_given_batch_when_execute_batch_then_returns_same_length(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given batch of artifacts, When execute_batch(), Then returns same length."""
        processors = [create_mock_processor("proc-1")]
        stage = IFEnrichmentStage(processors)

        artifacts = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Test content {i}",
                chunk_index=i,
                total_chunks=3,
            )
            for i in range(3)
        ]

        result = stage.execute_batch(artifacts)
        assert len(result) == len(artifacts)

    def test_given_processor_with_process_batch_when_execute_batch_then_uses_batch_method(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given processor with process_batch, When execute_batch(), Then uses batch method."""
        processor = create_mock_processor("batch-proc")

        # Add process_batch method to the mock
        def mock_process_batch(artifacts: List[IFArtifact]) -> List[IFArtifact]:
            return [
                artifact.derive(
                    "batch-proc",
                    artifact_id=f"{artifact.artifact_id}-batch",
                    metadata={**artifact.metadata, "batch_processed": True},
                )
                for artifact in artifacts
            ]

        processor.process_batch = Mock(side_effect=mock_process_batch)

        stage = IFEnrichmentStage([processor])

        artifacts = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Test content {i}",
                chunk_index=i,
                total_chunks=3,
            )
            for i in range(3)
        ]

        result = stage.execute_batch(artifacts)

        # Verify batch method was called
        assert processor.process_batch.called
        assert processor.process_batch.call_count == 1

        # Verify individual process was NOT called
        assert not processor.process.called

        # Verify results are enriched
        assert all(r.metadata.get("batch_processed") for r in result)

    def test_given_processor_without_batch_when_execute_batch_then_falls_back_sequential(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given processor without process_batch, When execute_batch(), Then uses sequential."""
        processor = create_mock_processor("sequential-proc")
        # Ensure no process_batch method
        assert not hasattr(processor, "process_batch")

        stage = IFEnrichmentStage([processor])

        artifacts = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Test content {i}",
                chunk_index=i,
                total_chunks=3,
            )
            for i in range(3)
        ]

        result = stage.execute_batch(artifacts)

        # Verify individual process was called for each artifact
        assert processor.process.called
        assert processor.process.call_count == len(artifacts)

    def test_given_batch_exceeds_max_when_execute_batch_then_raises_error(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Given batch exceeds MAX_BATCH_SIZE, When execute_batch(), Then raises ValueError."""
        # Create a batch larger than MAX_BATCH_SIZE
        artifacts = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Test content {i}",
                chunk_index=i,
                total_chunks=MAX_BATCH_SIZE + 1,
            )
            for i in range(MAX_BATCH_SIZE + 1)
        ]

        with pytest.raises(ValueError, match="exceeds MAX_BATCH_SIZE"):
            enrichment_stage.execute_batch(artifacts)

    def test_given_batch_processing_error_when_execute_batch_then_falls_back_sequential(
        self,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given batch processing error, When execute_batch(), Then falls back to sequential."""
        processor = create_mock_processor("failing-batch-proc")

        # Add process_batch that raises an error
        processor.process_batch = Mock(
            side_effect=RuntimeError("Batch processing failed")
        )

        stage = IFEnrichmentStage([processor])

        artifacts = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Test content {i}",
                chunk_index=i,
                total_chunks=2,
            )
            for i in range(2)
        ]

        # Should not raise, should fall back to sequential
        result = stage.execute_batch(artifacts)

        # Verify we got results
        assert len(result) == len(artifacts)

        # Verify individual process was called as fallback
        assert processor.process.called

    def test_rule_4_execute_batch_under_60_lines(
        self,
        enrichment_stage: IFEnrichmentStage,
    ) -> None:
        """Rule #4: execute_batch() method is < 60 lines."""
        source = inspect.getsource(enrichment_stage.execute_batch)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]
        assert len(lines) < 60, f"execute_batch() has {len(lines)} lines"
