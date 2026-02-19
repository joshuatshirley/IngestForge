"""
Processor Teardown in Pipeline Tests.

Tests for automatic teardown of processors and stages after pipeline execution.
GWT-compliant behavioral specifications with NASA JPL Power of Ten coverage.
"""

import pytest
from unittest.mock import MagicMock, patch

from ingestforge.core.pipeline.interfaces import IFArtifact, IFStage
from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFFailureArtifact
from ingestforge.core.pipeline.runner import IFPipelineRunner


# =============================================================================
# Mock Classes
# =============================================================================


class MockStageWithTeardown(IFStage):
    """Mock stage that supports teardown."""

    def __init__(
        self,
        name: str = "mock-stage",
        fail: bool = False,
        teardown_success: bool = True,
        teardown_raises: bool = False,
    ):
        self._name = name
        self._fail = fail
        self._teardown_success = teardown_success
        self._teardown_raises = teardown_raises
        self.teardown_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_type(self):
        return IFArtifact

    @property
    def output_type(self):
        return IFArtifact

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        if self._fail:
            return IFFailureArtifact(
                artifact_id=artifact.artifact_id,
                error_message="forced fail",
                provenance=artifact.provenance,
            )
        return artifact.derive(processor_id=f"stage-{self.name}")

    def teardown(self) -> bool:
        self.teardown_called = True
        if self._teardown_raises:
            raise RuntimeError("Teardown explosion!")
        return self._teardown_success


class MockStageWithoutTeardown(IFStage):
    """Mock stage that does not support teardown."""

    def __init__(self, name: str = "no-teardown"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_type(self):
        return IFArtifact

    @property
    def output_type(self):
        return IFArtifact

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        return artifact.derive(processor_id=f"stage-{self.name}")


class CrashStage(IFStage):
    """Mock stage that raises an exception during execution."""

    def __init__(self):
        self.teardown_called = False

    @property
    def name(self) -> str:
        return "crash"

    @property
    def input_type(self):
        return IFArtifact

    @property
    def output_type(self):
        return IFArtifact

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        raise RuntimeError("boom")

    def teardown(self) -> bool:
        self.teardown_called = True
        return True


# =============================================================================
# Scenario 1: Normal Completion Tests
# =============================================================================


class TestTeardownOnNormalCompletion:
    """Tests for teardown on successful pipeline completion."""

    def test_teardown_called_on_success(self):
        """
        GWT - Scenario 1:
        Given a pipeline with multiple IFProcessor instances.
        When pipeline execution completes successfully.
        Then teardown() is called on all processors.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage1 = MockStageWithTeardown("stage1")
        stage2 = MockStageWithTeardown("stage2")
        stages = [stage1, stage2]

        result = runner.run(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert stage1.teardown_called
        assert stage2.teardown_called

    def test_teardown_called_in_order(self):
        """
        GWT:
        Given a pipeline with stages A, B, C.
        When pipeline execution completes.
        Then teardown is called on all stages.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stages = [
            MockStageWithTeardown("A"),
            MockStageWithTeardown("B"),
            MockStageWithTeardown("C"),
        ]

        runner.run(artifact, stages, document_id="doc-1")

        for stage in stages:
            assert stage.teardown_called

    def test_teardown_skipped_for_stages_without_method(self):
        """
        GWT:
        Given a pipeline with mixed stages (some with teardown, some without).
        When pipeline execution completes.
        Then only stages with teardown() are called.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage_with = MockStageWithTeardown("with-teardown")
        stage_without = MockStageWithoutTeardown("no-teardown")
        stages = [stage_with, stage_without]

        result = runner.run(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert stage_with.teardown_called
        # stage_without doesn't have teardown_called attribute


# =============================================================================
# Scenario 2: Error During Execution Tests
# =============================================================================


class TestTeardownOnError:
    """Tests for teardown when pipeline fails mid-execution."""

    def test_teardown_called_on_stage_failure(self):
        """
        GWT - Scenario 2:
        Given a pipeline that fails mid-execution.
        When an exception is raised.
        Then teardown() is still called on all initialized processors.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage1 = MockStageWithTeardown("stage1")
        stage2 = MockStageWithTeardown("stage2", fail=True)
        stage3 = MockStageWithTeardown("stage3")
        stages = [stage1, stage2, stage3]

        result = runner.run(artifact, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        # All stages should have teardown called (even ones not executed)
        assert stage1.teardown_called
        assert stage2.teardown_called
        assert stage3.teardown_called

    def test_teardown_called_on_exception(self):
        """
        GWT:
        Given a pipeline with a stage that throws an exception.
        When the exception is raised during execution.
        Then teardown() is still called on all stages.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage1 = MockStageWithTeardown("stage1")
        crash = CrashStage()
        stage2 = MockStageWithTeardown("stage2")
        stages = [stage1, crash, stage2]

        result = runner.run(artifact, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "boom" in result.error_message
        assert stage1.teardown_called
        assert crash.teardown_called
        assert stage2.teardown_called


# =============================================================================
# Teardown Error Isolation Tests (JPL Rule #7)
# =============================================================================


class TestTeardownErrorIsolation:
    """Tests for isolated teardown - one failure doesn't block others."""

    def test_teardown_continues_on_failure(self):
        """
        GWT:
        Given a pipeline with multiple stages where one teardown fails.
        When pipeline completes and teardown is called.
        Then other teardowns are still executed.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage1 = MockStageWithTeardown("stage1", teardown_success=True)
        stage2 = MockStageWithTeardown("stage2", teardown_raises=True)
        stage3 = MockStageWithTeardown("stage3", teardown_success=True)
        stages = [stage1, stage2, stage3]

        result = runner.run(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        # All teardowns should be called even though stage2 raises
        assert stage1.teardown_called
        assert stage2.teardown_called
        assert stage3.teardown_called

    def test_teardown_failure_doesnt_affect_pipeline_result(self):
        """
        GWT:
        Given a pipeline that completes successfully.
        When teardown() fails on one stage.
        Then the pipeline result is still successful.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage = MockStageWithTeardown("stage", teardown_success=False)
        stages = [stage]

        result = runner.run(artifact, stages, document_id="doc-1")

        # Pipeline result should still be successful
        assert not isinstance(result, IFFailureArtifact)
        assert stage.teardown_called


# =============================================================================
# Public API Tests
# =============================================================================


class TestTeardownStagesPublicAPI:
    """Tests for the public teardown_stages() method."""

    def test_teardown_stages_returns_true_on_all_success(self):
        """
        GWT:
        Given multiple stages with successful teardowns.
        When teardown_stages() is called.
        Then True is returned.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStageWithTeardown("stage1", teardown_success=True),
            MockStageWithTeardown("stage2", teardown_success=True),
        ]

        result = runner.teardown_stages(stages)

        assert result is True
        assert all(s.teardown_called for s in stages)

    def test_teardown_stages_returns_false_on_any_failure(self):
        """
        GWT:
        Given multiple stages where one returns False.
        When teardown_stages() is called.
        Then False is returned but all teardowns are attempted.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStageWithTeardown("stage1", teardown_success=True),
            MockStageWithTeardown("stage2", teardown_success=False),
            MockStageWithTeardown("stage3", teardown_success=True),
        ]

        result = runner.teardown_stages(stages)

        assert result is False
        assert all(s.teardown_called for s in stages)

    def test_teardown_stages_returns_false_on_exception(self):
        """
        GWT:
        Given multiple stages where one raises an exception.
        When teardown_stages() is called.
        Then False is returned but all teardowns are attempted.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStageWithTeardown("stage1", teardown_success=True),
            MockStageWithTeardown("stage2", teardown_raises=True),
            MockStageWithTeardown("stage3", teardown_success=True),
        ]

        result = runner.teardown_stages(stages)

        assert result is False
        assert all(s.teardown_called for s in stages)


# =============================================================================
# run_bounded Teardown Tests
# =============================================================================


class TestRunBoundedTeardown:
    """Tests for teardown in run_bounded method."""

    def test_run_bounded_calls_teardown_on_success(self):
        """
        GWT:
        Given a pipeline using run_bounded.
        When execution completes successfully.
        Then teardown is called on all stages.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage = MockStageWithTeardown("stage")
        stages = [stage]

        result = runner.run_bounded(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert stage.teardown_called

    def test_run_bounded_calls_teardown_on_failure(self):
        """
        GWT:
        Given a pipeline using run_bounded that fails.
        When execution fails mid-pipeline.
        Then teardown is still called on all stages.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage1 = MockStageWithTeardown("stage1")
        stage2 = MockStageWithTeardown("stage2", fail=True)
        stages = [stage1, stage2]

        result = runner.run_bounded(artifact, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert stage1.teardown_called
        assert stage2.teardown_called


# =============================================================================
# run_monitored Teardown Tests
# =============================================================================


class TestRunMonitoredTeardown:
    """Tests for teardown in run_monitored method."""

    def test_run_monitored_calls_teardown_on_success(self):
        """
        GWT:
        Given a pipeline using run_monitored.
        When execution completes successfully.
        Then teardown is called on all stages.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage = MockStageWithTeardown("stage")
        stages = [stage]

        result = runner.run_monitored(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert stage.teardown_called

    def test_run_monitored_calls_teardown_on_failure(self):
        """
        GWT:
        Given a pipeline using run_monitored that fails.
        When execution fails mid-pipeline.
        Then teardown is still called on all stages.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        crash = CrashStage()
        stages = [crash]

        result = runner.run_monitored(artifact, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert crash.teardown_called


# =============================================================================
# JPL Power of Ten Compliance
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten rules for teardown."""

    def test_rule2_bounded_teardown_iterations(self):
        """
        JPL Rule #2:
        Given many stages.
        When teardown is called.
        Then it iterates once per stage (bounded).
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stages = [MockStageWithTeardown(f"stage-{i}") for i in range(10)]

        runner.run(artifact, stages, document_id="doc-1")

        teardown_count = sum(1 for s in stages if s.teardown_called)
        assert teardown_count == 10

    def test_rule7_teardown_exceptions_logged_not_raised(self):
        """
        JPL Rule #7:
        Given a stage whose teardown raises.
        When teardown is called.
        Then exception is logged but not propagated.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage = MockStageWithTeardown("boom", teardown_raises=True)
        stages = [stage]

        # Should not raise
        result = runner.run(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert stage.teardown_called


# =============================================================================
# Scenario 3: Context Manager Tests
# =============================================================================


class TestPipelineContextManager:
    """Tests for Pipeline context manager support (Scenario 3)."""

    @pytest.fixture
    def mock_config(self):
        """Create a minimal mock config for Pipeline."""
        config = MagicMock()
        config.data_path = MagicMock()
        config.data_path.__truediv__ = MagicMock(return_value="test_state.json")
        config.project = MagicMock()
        config.project.name = "test-project"
        config.ensure_directories = MagicMock()
        config.performance_mode = "balanced"  # Required for apply_performance_preset
        config.enrichment = MagicMock()
        config.enrichment.generate_embeddings = False
        config.enrichment.extract_entities = False
        config.enrichment.generate_questions = False
        config.enrichment.generate_summaries = False
        config.enrichment.use_instructor_citation = False
        config.enrichment.compute_quality = False
        return config

    def test_context_manager_returns_self(self, mock_config):
        """
        GWT - Scenario 3:
        Given a Pipeline instance.
        When used with `with` statement.
        Then __enter__ returns the pipeline.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)

                    result = pipeline.__enter__()

                    assert result is pipeline

    def test_context_manager_calls_teardown_on_exit(self, mock_config):
        """
        GWT - Scenario 3:
        Given a Pipeline used as context manager.
        When exiting the with block.
        Then _teardown_components is called.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)
                    pipeline._teardown_components = MagicMock(return_value=True)

                    pipeline.__exit__(None, None, None)

                    pipeline._teardown_components.assert_called_once()

    def test_context_manager_calls_teardown_on_exception(self, mock_config):
        """
        GWT:
        Given a Pipeline context manager with an exception.
        When exception occurs in the with block.
        Then teardown is still called.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)
                    pipeline._teardown_components = MagicMock(return_value=True)

                    # Simulate exception scenario
                    pipeline.__exit__(RuntimeError, RuntimeError("test error"), None)

                    pipeline._teardown_components.assert_called_once()

    def test_context_manager_does_not_suppress_exceptions(self, mock_config):
        """
        GWT - JPL Rule #7:
        Given a Pipeline context manager.
        When an exception occurs.
        Then the exception is not suppressed.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)

                    result = pipeline.__exit__(ValueError, ValueError("test"), None)

                    assert result is False  # Don't suppress


# =============================================================================
# auto_teardown Parameter Tests
# =============================================================================


class TestAutoTeardownParameter:
    """Tests for the auto_teardown constructor parameter."""

    def test_auto_teardown_default_true(self):
        """
        GWT:
        Given no auto_teardown parameter.
        When IFPipelineRunner is created.
        Then auto_teardown defaults to True.
        """
        runner = IFPipelineRunner()

        assert runner._auto_teardown is True

    def test_auto_teardown_can_be_disabled(self):
        """
        GWT:
        Given auto_teardown=False.
        When IFPipelineRunner is created.
        Then auto_teardown is False.
        """
        runner = IFPipelineRunner(auto_teardown=False)

        assert runner._auto_teardown is False

    def test_auto_teardown_false_skips_teardown(self):
        """
        GWT:
        Given auto_teardown=False.
        When run() completes.
        Then teardown is NOT called on stages.
        """
        runner = IFPipelineRunner(auto_teardown=False)
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage = MockStageWithTeardown("stage")
        stages = [stage]

        result = runner.run(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert not stage.teardown_called  # Teardown should NOT be called

    def test_auto_teardown_true_calls_teardown(self):
        """
        GWT:
        Given auto_teardown=True (default).
        When run() completes.
        Then teardown IS called on stages.
        """
        runner = IFPipelineRunner(auto_teardown=True)
        artifact = IFTextArtifact(artifact_id="1", content="test")
        stage = MockStageWithTeardown("stage")
        stages = [stage]

        result = runner.run(artifact, stages, document_id="doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert stage.teardown_called  # Teardown should be called


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestTeardownEdgeCases:
    """Edge case tests for teardown functionality."""

    def test_teardown_with_empty_stage_list(self):
        """
        GWT:
        Given an empty list of stages.
        When teardown_stages is called.
        Then True is returned (no failures).
        """
        runner = IFPipelineRunner()

        result = runner.teardown_stages([])

        assert result is True

    def test_teardown_with_single_stage(self):
        """
        GWT:
        Given a single stage.
        When teardown_stages is called.
        Then the stage is torn down.
        """
        runner = IFPipelineRunner()
        stage = MockStageWithTeardown("single")

        result = runner.teardown_stages([stage])

        assert result is True
        assert stage.teardown_called

    def test_teardown_with_mixed_success_and_failure(self):
        """
        GWT:
        Given stages with mixed teardown results.
        When teardown_stages is called.
        Then returns False but all stages attempted.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStageWithTeardown("s1", teardown_success=True),
            MockStageWithTeardown("s2", teardown_success=False),
            MockStageWithTeardown("s3", teardown_raises=True),
            MockStageWithTeardown("s4", teardown_success=True),
        ]

        result = runner.teardown_stages(stages)

        assert result is False
        assert all(s.teardown_called for s in stages)

    def test_run_with_no_stages(self):
        """
        GWT:
        Given an empty stage list.
        When run is called.
        Then the original artifact is returned.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="1", content="test")

        result = runner.run(artifact, [], document_id="doc-1")

        assert result == artifact


# =============================================================================
# Pipeline Teardown Components Tests
# =============================================================================


class TestPipelineTeardownComponents:
    """Tests for Pipeline._teardown_components method."""

    @pytest.fixture
    def mock_config(self):
        """Create a minimal mock config for Pipeline."""
        config = MagicMock()
        config.data_path = MagicMock()
        config.data_path.__truediv__ = MagicMock(return_value="test_state.json")
        config.project = MagicMock()
        config.project.name = "test-project"
        config.ensure_directories = MagicMock()
        config.performance_mode = "balanced"  # Required for apply_performance_preset
        config.enrichment = MagicMock()
        config.enrichment.generate_embeddings = False
        config.enrichment.extract_entities = False
        config.enrichment.generate_questions = False
        config.enrichment.generate_summaries = False
        config.enrichment.use_instructor_citation = False
        config.enrichment.compute_quality = False
        return config

    def test_teardown_enricher_called(self, mock_config):
        """
        GWT:
        Given a Pipeline with an enricher.
        When _teardown_components is called.
        Then enricher.teardown() is called.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)

                    # Mock enricher with teardown
                    mock_enricher = MagicMock()
                    mock_enricher.teardown.return_value = True
                    pipeline._enricher = mock_enricher

                    pipeline._teardown_components()

                    mock_enricher.teardown.assert_called_once()

    def test_teardown_storage_called(self, mock_config):
        """
        GWT:
        Given a Pipeline with a storage backend.
        When _teardown_components is called.
        Then storage.close() is called.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)

                    # Mock storage with close
                    mock_storage = MagicMock()
                    pipeline._storage = mock_storage

                    pipeline._teardown_components()

                    mock_storage.close.assert_called_once()

    def test_teardown_continues_on_enricher_exception(self, mock_config):
        """
        GWT - JPL Rule #7:
        Given enricher teardown raises exception.
        When _teardown_components is called.
        Then storage teardown still happens.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)

                    # Mock enricher that raises
                    mock_enricher = MagicMock()
                    mock_enricher.teardown.side_effect = RuntimeError("boom")
                    pipeline._enricher = mock_enricher

                    # Mock storage
                    mock_storage = MagicMock()
                    pipeline._storage = mock_storage

                    # Should not raise
                    pipeline._teardown_components()

                    # Both should be attempted
                    mock_enricher.teardown.assert_called_once()
                    mock_storage.close.assert_called_once()

    def test_explicit_teardown_returns_true_on_success(self, mock_config):
        """
        GWT:
        Given Pipeline with no errors.
        When teardown() is called explicitly.
        Then True is returned.
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)

                    result = pipeline.teardown()

                    assert result is True

    def test_explicit_teardown_returns_component_result(self, mock_config):
        """
        GWT:
        Given Pipeline where _teardown_components returns False.
        When teardown() is called explicitly.
        Then False is returned (delegates to _teardown_components).
        """
        with patch(
            "ingestforge.core.pipeline.pipeline.load_config", return_value=mock_config
        ):
            with patch(
                "ingestforge.core.pipeline.pipeline.apply_performance_preset",
                return_value=mock_config,
            ):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline(config=mock_config)

                    # Make _teardown_components return False (simulates partial failure)
                    pipeline._teardown_components = MagicMock(return_value=False)

                    result = pipeline.teardown()

                    assert result is False
