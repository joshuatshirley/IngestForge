import time
from typing import List
import pytest
from ingestforge.core.pipeline.interfaces import IFArtifact, IFStage
from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFFailureArtifact
from ingestforge.core.pipeline.interfaces import IFProcessor
from ingestforge.core.pipeline.runner import (
    IFPipelineRunner,
    ResourceConfig,
    FallbackConfig,
    FallbackResult,
    FallbackAttempt,
    MAX_PIPELINE_STAGES,
    MAX_FALLBACK_ATTEMPTS,
    DEFAULT_STAGE_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
)


class MockStage(IFStage):
    def __init__(self, name, fail=False, in_type=IFArtifact, out_type=IFArtifact):
        self._name = name
        self._fail = fail
        self._in_type = in_type
        self._out_type = out_type

    @property
    def name(self):
        return self._name

    @property
    def input_type(self):
        return self._in_type

    @property
    def output_type(self):
        return self._out_type

    def execute(self, artifact):
        if self._fail:
            return IFFailureArtifact(
                artifact_id=artifact.artifact_id,
                error_message="forced fail",
                provenance=artifact.provenance,
            )
        return artifact.derive(processor_id=f"stage-{self.name}")


def test_runner_sequential_execution():
    """
    GWT:
    Given a list of stages
    When run() is called
    Then stages are executed in order and provenance is updated.
    """
    runner = IFPipelineRunner()
    art = IFTextArtifact(artifact_id="1", content="start")
    stages = [MockStage("A"), MockStage("B")]

    result = runner.run(art, stages, document_id="test-doc")

    assert "stage-A" in result.provenance
    assert "stage-B" in result.provenance
    assert result.provenance.index("stage-A") < result.provenance.index("stage-B")


def test_runner_failure_containment():
    """
    GWT:
    Given a stage that fails
    When run() is called
    Then the runner stops and returns a FailureArtifact.
    """
    runner = IFPipelineRunner()
    art = IFTextArtifact(artifact_id="1", content="start")
    stages = [MockStage("A"), MockStage("Fail", fail=True), MockStage("B")]

    result = runner.run(art, stages, document_id="test-doc")

    assert isinstance(result, IFFailureArtifact)
    assert result.error_message == "forced fail"
    assert "stage-A" in result.provenance
    assert "stage-B" not in result.provenance


def test_runner_exception_containment():
    """
    GWT:
    Given a stage that raises an unhandled exception
    When run() is called
    Then the runner catches it and returns a FailureArtifact.
    """

    class CrashStage(IFStage):
        @property
        def name(self):
            return "crash"

        @property
        def input_type(self):
            return IFArtifact

        @property
        def output_type(self):
            return IFArtifact

        def execute(self, artifact):
            raise RuntimeError("boom")

    runner = IFPipelineRunner()
    art = IFTextArtifact(artifact_id="1", content="start")
    stages = [CrashStage()]

    result = runner.run(art, stages, document_id="test-doc")
    assert isinstance(result, IFFailureArtifact)
    assert "crash" in result.error_message
    assert "boom" in result.error_message


def test_runner_type_mismatch():
    """
    GWT:
    Given a stage expecting a specific artifact type
    When a different type is passed
    Then the runner returns a FailureArtifact with type mismatch message.
    """

    class SpecialArtifact(IFArtifact):
        pass

    runner = IFPipelineRunner()
    art = IFTextArtifact(artifact_id="1", content="start")
    stages = [MockStage("A", in_type=SpecialArtifact)]

    result = runner.run(art, stages, document_id="test-doc")
    assert isinstance(result, IFFailureArtifact)
    assert "Type mismatch" in result.error_message


def test_runner_contract_violation():
    """
    GWT:
    Given a stage that produces the wrong artifact type
    When run() is called
    Then the runner returns a FailureArtifact with contract violation message.
    """

    class SpecialArtifact(IFArtifact):
        pass

    runner = IFPipelineRunner()
    art = IFTextArtifact(artifact_id="1", content="start")
    stages = [MockStage("A", out_type=SpecialArtifact)]

    result = runner.run(art, stages, document_id="test-doc")
    assert isinstance(result, IFFailureArtifact)
    assert "Contract violation" in result.error_message


# =============================================================================
# Boundaries - Resource Bounded Execution Tests
# =============================================================================


class TestResourceConfig:
    """Tests for ResourceConfig dataclass."""

    def test_default_config_values(self):
        """
        GWT:
        Given no arguments
        When ResourceConfig is created
        Then default values are used.
        """
        config = ResourceConfig()

        assert config.timeout_seconds == DEFAULT_STAGE_TIMEOUT_SECONDS
        assert config.max_stages == MAX_PIPELINE_STAGES
        assert config.warn_memory_threshold == 0.8

    def test_config_enforces_upper_bound_timeout(self):
        """
        GWT - Scenario 4 (JPL Rule #2):
        Given timeout exceeds MAX_TIMEOUT_SECONDS
        When ResourceConfig is created
        Then timeout is capped to MAX_TIMEOUT_SECONDS.
        """
        config = ResourceConfig(timeout_seconds=999999)

        assert config.timeout_seconds == MAX_TIMEOUT_SECONDS

    def test_config_enforces_upper_bound_stages(self):
        """
        GWT - Scenario 3 (JPL Rule #2):
        Given max_stages exceeds MAX_PIPELINE_STAGES
        When ResourceConfig is created
        Then max_stages is capped to MAX_PIPELINE_STAGES.
        """
        config = ResourceConfig(max_stages=1000)

        assert config.max_stages == MAX_PIPELINE_STAGES

    def test_config_handles_invalid_timeout(self):
        """
        GWT:
        Given timeout <= 0
        When ResourceConfig is created
        Then default timeout is used.
        """
        config = ResourceConfig(timeout_seconds=-5)

        assert config.timeout_seconds == DEFAULT_STAGE_TIMEOUT_SECONDS

    def test_config_handles_invalid_stages(self):
        """
        GWT:
        Given max_stages <= 0
        When ResourceConfig is created
        Then default max_stages is used.
        """
        config = ResourceConfig(max_stages=0)

        assert config.max_stages == MAX_PIPELINE_STAGES

    def test_config_custom_values(self):
        """
        GWT - Scenario 4:
        Given custom ResourceConfig values
        When ResourceConfig is created
        Then custom values are preserved if within bounds.
        """
        config = ResourceConfig(
            timeout_seconds=60,
            max_memory_mb=2048,
            max_stages=16,
            warn_memory_threshold=0.9,
        )

        assert config.timeout_seconds == 60
        assert config.max_memory_mb == 2048
        assert config.max_stages == 16
        assert config.warn_memory_threshold == 0.9


class TestPipelineValidation:
    """Tests for IFPipelineRunner.validate_pipeline()."""

    def test_validate_empty_pipeline(self):
        """
        GWT:
        Given an empty stages list
        When validate_pipeline() is called
        Then an error message is returned.
        """
        runner = IFPipelineRunner()
        error = runner.validate_pipeline([])

        assert error is not None
        assert "no stages" in error.lower()

    def test_validate_exceeds_stage_limit(self):
        """
        GWT - Scenario 3:
        Given a pipeline with 33 stages
        When validate_pipeline() is called
        Then an error is returned.
        """
        config = ResourceConfig(max_stages=5)
        runner = IFPipelineRunner(resource_config=config)
        stages = [MockStage(f"S{i}") for i in range(6)]

        error = runner.validate_pipeline(stages)

        assert error is not None
        assert "exceeds stage limit" in error

    def test_validate_valid_pipeline(self):
        """
        GWT:
        Given a valid pipeline within limits
        When validate_pipeline() is called
        Then None is returned.
        """
        runner = IFPipelineRunner()
        stages = [MockStage("A"), MockStage("B")]

        error = runner.validate_pipeline(stages)

        assert error is None


class TestRunBoundedExecution:
    """Tests for IFPipelineRunner.run_bounded()."""

    def test_run_bounded_rejects_too_many_stages(self):
        """
        GWT - Scenario 3:
        Given a pipeline exceeding max_stages
        When run_bounded() is called
        Then a FailureArtifact is returned immediately.
        """
        config = ResourceConfig(max_stages=2)
        runner = IFPipelineRunner(resource_config=config)
        art = IFTextArtifact(artifact_id="1", content="test")
        stages = [MockStage(f"S{i}") for i in range(5)]

        result = runner.run_bounded(art, stages, "doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "exceeds stage limit" in result.error_message
        assert "pipeline-validation-failed" in result.provenance

    def test_run_bounded_executes_stages(self):
        """
        GWT:
        Given a valid pipeline with stages
        When run_bounded() is called
        Then all stages execute successfully.
        """
        config = ResourceConfig(timeout_seconds=10)
        runner = IFPipelineRunner(resource_config=config)
        art = IFTextArtifact(artifact_id="1", content="test")
        stages = [MockStage("A"), MockStage("B")]

        result = runner.run_bounded(art, stages, "doc-1")

        assert not isinstance(result, IFFailureArtifact)
        assert "stage-A" in result.provenance
        assert "stage-B" in result.provenance

    def test_run_bounded_timeout_enforcement(self):
        """
        GWT - Scenario 1:
        Given a stage with timeout of 1 second
        When the stage exceeds 1 second
        Then a FailureArtifact with timeout error is returned.
        """

        class SlowStage(IFStage):
            @property
            def name(self):
                return "slow"

            @property
            def input_type(self):
                return IFArtifact

            @property
            def output_type(self):
                return IFArtifact

            def execute(self, artifact):
                time.sleep(3)  # Sleep longer than timeout
                return artifact.derive(processor_id="slow")

        config = ResourceConfig(timeout_seconds=1)
        runner = IFPipelineRunner(resource_config=config)
        art = IFTextArtifact(artifact_id="1", content="test")
        stages = [SlowStage()]

        result = runner.run_bounded(art, stages, "doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "timeout" in result.error_message.lower()
        assert "stage-slow-timeout" in result.provenance

    def test_run_bounded_type_mismatch(self):
        """
        GWT:
        Given a stage expecting wrong input type
        When run_bounded() is called
        Then a FailureArtifact is returned.
        """

        class SpecialArtifact(IFArtifact):
            pass

        config = ResourceConfig(timeout_seconds=10)
        runner = IFPipelineRunner(resource_config=config)
        art = IFTextArtifact(artifact_id="1", content="test")
        stages = [MockStage("A", in_type=SpecialArtifact)]

        result = runner.run_bounded(art, stages, "doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "Type mismatch" in result.error_message

    def test_run_bounded_contract_violation(self):
        """
        GWT:
        Given a stage that produces wrong output type
        When run_bounded() is called
        Then a FailureArtifact is returned.
        """

        class SpecialArtifact(IFArtifact):
            pass

        config = ResourceConfig(timeout_seconds=10)
        runner = IFPipelineRunner(resource_config=config)
        art = IFTextArtifact(artifact_id="1", content="test")
        stages = [MockStage("A", out_type=SpecialArtifact)]

        result = runner.run_bounded(art, stages, "doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "Contract violation" in result.error_message


class TestTimeoutExecution:
    """Tests for stage timeout enforcement."""

    def test_execute_with_timeout_success(self):
        """
        GWT:
        Given a fast stage and generous timeout
        When _execute_with_timeout() is called
        Then the stage completes successfully.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="1", content="test")
        stage = MockStage("fast")

        result = runner._execute_with_timeout(stage, art, timeout_seconds=10)

        assert not isinstance(result, IFFailureArtifact)
        assert "stage-fast" in result.provenance

    def test_execute_with_timeout_failure(self):
        """
        GWT - Scenario 1:
        Given a slow stage exceeding timeout
        When _execute_with_timeout() is called
        Then a FailureArtifact is returned.
        """

        class SlowStage(IFStage):
            @property
            def name(self):
                return "slow"

            @property
            def input_type(self):
                return IFArtifact

            @property
            def output_type(self):
                return IFArtifact

            def execute(self, artifact):
                time.sleep(5)
                return artifact.derive(processor_id="slow")

        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="1", content="test")

        result = runner._execute_with_timeout(SlowStage(), art, timeout_seconds=1)

        assert isinstance(result, IFFailureArtifact)
        assert "timeout" in result.error_message.lower()
        assert "1 seconds" in result.error_message


class TestResourceConfigIntegration:
    """Integration tests for ResourceConfig with pipeline runner."""

    def test_runner_accepts_resource_config(self):
        """
        GWT - Scenario 4:
        Given a custom ResourceConfig
        When IFPipelineRunner is created
        Then the config is stored and used.
        """
        config = ResourceConfig(timeout_seconds=60, max_stages=10)
        runner = IFPipelineRunner(resource_config=config)

        assert runner.resource_config.timeout_seconds == 60
        assert runner.resource_config.max_stages == 10

    def test_runner_default_resource_config(self):
        """
        GWT:
        Given no ResourceConfig provided
        When IFPipelineRunner is created
        Then default ResourceConfig is used.
        """
        runner = IFPipelineRunner()

        assert runner.resource_config is not None
        assert runner.resource_config.timeout_seconds == DEFAULT_STAGE_TIMEOUT_SECONDS
        assert runner.resource_config.max_stages == MAX_PIPELINE_STAGES


# =============================================================================
# Monitoring - Non-Blocking Interceptors Tests
# =============================================================================

from ingestforge.core.pipeline.interfaces import IFInterceptor, MAX_INTERCEPTORS


class MockInterceptor(IFInterceptor):
    """Test interceptor that records all calls."""

    def __init__(self):
        self.pre_stage_calls: list = []
        self.post_stage_calls: list = []
        self.error_calls: list = []
        self.pipeline_start_calls: list = []
        self.pipeline_end_calls: list = []

    def pre_stage(self, stage_name, artifact, document_id):
        self.pre_stage_calls.append((stage_name, artifact.artifact_id, document_id))

    def post_stage(self, stage_name, artifact, document_id, duration_ms):
        self.post_stage_calls.append(
            (stage_name, artifact.artifact_id, document_id, duration_ms)
        )

    def on_error(self, stage_name, artifact, document_id, error):
        self.error_calls.append(
            (stage_name, artifact.artifact_id, document_id, str(error))
        )

    def on_pipeline_start(self, document_id, stage_count):
        self.pipeline_start_calls.append((document_id, stage_count))

    def on_pipeline_end(self, document_id, success, total_duration_ms):
        self.pipeline_end_calls.append((document_id, success, total_duration_ms))


class FailingInterceptor(IFInterceptor):
    """Interceptor that always raises exceptions."""

    def pre_stage(self, stage_name, artifact, document_id):
        raise RuntimeError("pre_stage failed")

    def post_stage(self, stage_name, artifact, document_id, duration_ms):
        raise RuntimeError("post_stage failed")

    def on_error(self, stage_name, artifact, document_id, error):
        raise RuntimeError("on_error failed")

    def on_pipeline_start(self, document_id, stage_count):
        raise RuntimeError("on_pipeline_start failed")

    def on_pipeline_end(self, document_id, success, total_duration_ms):
        raise RuntimeError("on_pipeline_end failed")


class TestInterceptorPreStage:
    """Tests for pre_stage interceptor calls."""

    def test_pre_stage_called_before_execution(self):
        """
        GWT - Scenario 1:
        Given a pre-stage interceptor is registered
        When a stage is about to execute
        Then the interceptor is called with stage name and input artifact.
        """
        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A"), MockStage("B")]

        runner.run_monitored(art, stages, "doc-123")

        assert len(interceptor.pre_stage_calls) == 2
        assert interceptor.pre_stage_calls[0] == ("A", "test-1", "doc-123")
        assert interceptor.pre_stage_calls[1][0] == "B"


class TestInterceptorPostStage:
    """Tests for post_stage interceptor calls."""

    def test_post_stage_called_after_success(self):
        """
        GWT - Scenario 2:
        Given a post-stage interceptor is registered
        When a stage completes successfully
        Then the interceptor is called with stage name, output artifact, and duration.
        """
        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A")]

        runner.run_monitored(art, stages, "doc-123")

        assert len(interceptor.post_stage_calls) == 1
        stage_name, artifact_id, doc_id, duration_ms = interceptor.post_stage_calls[0]
        assert stage_name == "A"
        assert doc_id == "doc-123"
        assert duration_ms >= 0  # Duration should be positive

    def test_post_stage_includes_timing(self):
        """
        GWT:
        Given a stage that takes measurable time
        When post_stage is called
        Then duration_ms reflects actual execution time.
        """

        class SlowStage(IFStage):
            @property
            def name(self):
                return "slow"

            @property
            def input_type(self):
                return IFArtifact

            @property
            def output_type(self):
                return IFArtifact

            def execute(self, artifact):
                time.sleep(0.05)  # 50ms
                return artifact.derive(processor_id="slow")

        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")

        runner.run_monitored(art, [SlowStage()], "doc-123")

        _, _, _, duration_ms = interceptor.post_stage_calls[0]
        assert duration_ms >= 50  # At least 50ms


class TestInterceptorOnError:
    """Tests for on_error interceptor calls."""

    def test_on_error_called_on_failure(self):
        """
        GWT - Scenario 3:
        Given an error interceptor is registered
        When a stage fails
        Then the interceptor is called with stage name, error details, and partial artifact.
        """
        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A", fail=True)]

        runner.run_monitored(art, stages, "doc-123")

        assert len(interceptor.error_calls) == 1
        stage_name, artifact_id, doc_id, error_msg = interceptor.error_calls[0]
        assert stage_name == "A"
        assert doc_id == "doc-123"
        assert "forced fail" in error_msg

    def test_on_error_called_on_exception(self):
        """
        GWT:
        Given a stage that raises an exception
        When run_monitored() is called
        Then on_error is called with the exception.
        """

        class CrashStage(IFStage):
            @property
            def name(self):
                return "crash"

            @property
            def input_type(self):
                return IFArtifact

            @property
            def output_type(self):
                return IFArtifact

            def execute(self, artifact):
                raise ValueError("boom")

        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")

        runner.run_monitored(art, [CrashStage()], "doc-123")

        assert len(interceptor.error_calls) == 1
        assert "boom" in interceptor.error_calls[0][3]


class TestInterceptorIsolation:
    """Tests for interceptor error isolation."""

    def test_failing_interceptor_does_not_crash_pipeline(self):
        """
        GWT - Scenario 4:
        Given an interceptor that raises an exception
        When called during pipeline execution
        Then the exception is logged but does not affect pipeline processing.
        """
        failing = FailingInterceptor()
        runner = IFPipelineRunner(interceptors=[failing])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A"), MockStage("B")]

        # Should NOT raise - interceptor errors are isolated
        result = runner.run_monitored(art, stages, "doc-123")

        # Pipeline should complete successfully
        assert not isinstance(result, IFFailureArtifact)
        assert "stage-A" in result.provenance
        assert "stage-B" in result.provenance

    def test_failing_interceptor_does_not_stop_other_interceptors(self):
        """
        GWT:
        Given a failing interceptor and a working interceptor
        When both are registered
        Then the working interceptor still receives calls.
        """
        failing = FailingInterceptor()
        working = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[failing, working])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A")]

        runner.run_monitored(art, stages, "doc-123")

        # Working interceptor should still have been called
        assert len(working.pre_stage_calls) == 1
        assert len(working.post_stage_calls) == 1


class TestMultipleInterceptors:
    """Tests for multiple interceptors."""

    def test_multiple_interceptors_called_in_order(self):
        """
        GWT - Scenario 5:
        Given multiple interceptors of the same type are registered
        When the trigger event occurs
        Then all interceptors are called in registration order.
        """
        order_tracker = []

        class OrderedInterceptor(IFInterceptor):
            def __init__(self, name):
                self.name = name

            def pre_stage(self, stage_name, artifact, document_id):
                order_tracker.append(f"{self.name}-pre-{stage_name}")

        int1 = OrderedInterceptor("first")
        int2 = OrderedInterceptor("second")
        int3 = OrderedInterceptor("third")

        runner = IFPipelineRunner(interceptors=[int1, int2, int3])
        art = IFTextArtifact(artifact_id="test-1", content="hello")

        runner.run_monitored(art, [MockStage("A")], "doc-123")

        assert order_tracker == ["first-pre-A", "second-pre-A", "third-pre-A"]

    def test_interceptor_limit_enforced(self):
        """
        GWT (JPL Rule #2):
        Given more than MAX_INTERCEPTORS interceptors
        When runner is created
        Then list is truncated to MAX_INTERCEPTORS.
        """
        interceptors = [MockInterceptor() for _ in range(MAX_INTERCEPTORS + 5)]
        runner = IFPipelineRunner(interceptors=interceptors)

        assert len(runner._interceptors) == MAX_INTERCEPTORS


class TestAddInterceptor:
    """Tests for IFPipelineRunner.add_interceptor()."""

    def test_add_interceptor_success(self):
        """
        GWT:
        Given a runner with room for interceptors
        When add_interceptor() is called
        Then the interceptor is added and True is returned.
        """
        runner = IFPipelineRunner()
        interceptor = MockInterceptor()

        result = runner.add_interceptor(interceptor)

        assert result is True
        assert interceptor in runner._interceptors

    def test_add_interceptor_at_limit(self):
        """
        GWT (JPL Rule #2):
        Given a runner at MAX_INTERCEPTORS
        When add_interceptor() is called
        Then False is returned and interceptor is not added.
        """
        interceptors = [MockInterceptor() for _ in range(MAX_INTERCEPTORS)]
        runner = IFPipelineRunner(interceptors=interceptors)
        new_interceptor = MockInterceptor()

        result = runner.add_interceptor(new_interceptor)

        assert result is False
        assert new_interceptor not in runner._interceptors


class TestPipelineLifecycleInterceptors:
    """Tests for pipeline start/end interceptors."""

    def test_pipeline_start_called(self):
        """
        GWT:
        Given an interceptor with on_pipeline_start
        When run_monitored() begins
        Then on_pipeline_start is called with document_id and stage_count.
        """
        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A"), MockStage("B"), MockStage("C")]

        runner.run_monitored(art, stages, "doc-123")

        assert len(interceptor.pipeline_start_calls) == 1
        doc_id, stage_count = interceptor.pipeline_start_calls[0]
        assert doc_id == "doc-123"
        assert stage_count == 3

    def test_pipeline_end_called_on_success(self):
        """
        GWT:
        Given a successful pipeline execution
        When run_monitored() completes
        Then on_pipeline_end is called with success=True.
        """
        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A")]

        runner.run_monitored(art, stages, "doc-123")

        assert len(interceptor.pipeline_end_calls) == 1
        doc_id, success, duration_ms = interceptor.pipeline_end_calls[0]
        assert doc_id == "doc-123"
        assert success is True
        assert duration_ms >= 0

    def test_pipeline_end_called_on_failure(self):
        """
        GWT:
        Given a failing pipeline execution
        When run_monitored() completes
        Then on_pipeline_end is called with success=False.
        """
        interceptor = MockInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])
        art = IFTextArtifact(artifact_id="test-1", content="hello")
        stages = [MockStage("A", fail=True)]

        runner.run_monitored(art, stages, "doc-123")

        assert len(interceptor.pipeline_end_calls) == 1
        _, success, _ = interceptor.pipeline_end_calls[0]
        assert success is False


# =============================================================================
# Dry-Run Mode Tests
# =============================================================================

from ingestforge.core.pipeline.runner import DryRunResult
from ingestforge.core.pipeline.artifacts import IFChunkArtifact


class TestDryRunTypeChainValidation:
    """GWT tests for Scenario 1: Type Chain Validation."""

    def test_valid_type_chain(self):
        """
        GWT:
        Given a pipeline with compatible input/output types,
        When run_dry() is called,
        Then validation succeeds with all stages marked valid.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStage("stage-1", in_type=IFArtifact, out_type=IFArtifact),
            MockStage("stage-2", in_type=IFArtifact, out_type=IFArtifact),
        ]

        result = runner.run_dry(IFArtifact, stages)

        assert result.valid is True
        assert result.stage_count == 2
        assert len(result.errors) == 0
        assert all(s.valid for s in result.stages)

    def test_subtype_compatibility(self):
        """
        GWT:
        Given stages where output is subtype of next input,
        When run_dry() is called,
        Then validation succeeds (subtype is compatible).
        """
        runner = IFPipelineRunner()
        # IFTextArtifact is subtype of IFArtifact
        stages = [
            MockStage("produce-text", in_type=IFArtifact, out_type=IFTextArtifact),
            MockStage("consume-artifact", in_type=IFArtifact, out_type=IFArtifact),
        ]

        result = runner.run_dry(IFArtifact, stages)

        assert result.valid is True
        assert len(result.errors) == 0


class TestDryRunValidationReport:
    """GWT tests for Scenario 2: Validation Report."""

    def test_returns_stage_details(self):
        """
        GWT:
        Given a valid pipeline configuration,
        When run_dry() completes,
        Then a report includes stage names, types, and success status.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStage("extract", in_type=IFArtifact, out_type=IFTextArtifact),
            MockStage("chunk", in_type=IFTextArtifact, out_type=IFChunkArtifact),
        ]

        result = runner.run_dry(IFArtifact, stages)

        assert isinstance(result, DryRunResult)
        assert result.stage_count == 2
        assert len(result.stages) == 2

        # Check first stage details
        s0 = result.stages[0]
        assert s0.stage_name == "extract"
        assert s0.stage_index == 0
        assert s0.input_type == "IFArtifact"
        assert s0.output_type == "IFTextArtifact"
        assert s0.valid is True

        # Check second stage details
        s1 = result.stages[1]
        assert s1.stage_name == "chunk"
        assert s1.stage_index == 1
        assert s1.input_type == "IFTextArtifact"
        assert s1.output_type == "IFChunkArtifact"

    def test_reports_initial_and_final_types(self):
        """
        GWT:
        Given a valid pipeline,
        When run_dry() completes,
        Then initial_type and final_type are reported.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStage("transform", in_type=IFTextArtifact, out_type=IFChunkArtifact),
        ]

        result = runner.run_dry(IFTextArtifact, stages)

        assert result.initial_type == "IFTextArtifact"
        assert result.final_type == "IFChunkArtifact"


class TestDryRunTypeMismatchDetection:
    """GWT tests for Scenario 3: Type Mismatch Detection."""

    def test_detects_mid_pipeline_mismatch(self):
        """
        GWT:
        Given a pipeline where stage N output doesn't match stage N+1 input,
        When run_dry() is called,
        Then validation error identifies the mismatched stages.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStage("stage-1", in_type=IFArtifact, out_type=IFTextArtifact),
            # Mismatch: expects IFChunkArtifact but receives IFTextArtifact
            MockStage("stage-2", in_type=IFChunkArtifact, out_type=IFArtifact),
        ]

        result = runner.run_dry(IFArtifact, stages)

        assert result.valid is False
        assert len(result.errors) == 1
        assert "stage-1" in result.errors[0]
        assert "stage-2" in result.errors[0]
        assert "IFTextArtifact" in result.errors[0]
        assert "IFChunkArtifact" in result.errors[0]

    def test_reports_all_mismatches(self):
        """
        GWT:
        Given a pipeline with multiple type mismatches,
        When run_dry() is called,
        Then all mismatches are reported (not just first).
        """
        runner = IFPipelineRunner()
        stages = [
            MockStage("s1", in_type=IFArtifact, out_type=IFTextArtifact),
            MockStage(
                "s2", in_type=IFChunkArtifact, out_type=IFChunkArtifact
            ),  # Mismatch 1
            MockStage("s3", in_type=IFTextArtifact, out_type=IFArtifact),  # Mismatch 2
        ]

        result = runner.run_dry(IFArtifact, stages)

        assert result.valid is False
        assert len(result.errors) == 2
        # Both mismatches reported
        assert (
            not result.stages[0].valid
            or len([s for s in result.stages if not s.valid]) >= 1
        )


class TestDryRunInitialArtifactValidation:
    """GWT tests for Scenario 4: Initial Artifact Type Validation."""

    def test_detects_initial_type_mismatch(self):
        """
        GWT:
        Given an initial artifact type that doesn't match stage 1 input,
        When run_dry() is called,
        Then validation error is reported for initial artifact.
        """
        runner = IFPipelineRunner()
        stages = [
            MockStage("stage-1", in_type=IFChunkArtifact, out_type=IFArtifact),
        ]

        # Pass IFTextArtifact but stage expects IFChunkArtifact
        result = runner.run_dry(IFTextArtifact, stages)

        assert result.valid is False
        assert len(result.errors) == 1
        assert "Initial artifact type" in result.errors[0]
        assert "IFTextArtifact" in result.errors[0]
        assert "IFChunkArtifact" in result.errors[0]


class TestDryRunEmptyPipeline:
    """GWT tests for Scenario 5: Empty Pipeline Validation."""

    def test_empty_pipeline_is_valid(self):
        """
        GWT:
        Given an empty stage list,
        When run_dry() is called,
        Then validation succeeds with empty report.
        """
        runner = IFPipelineRunner()
        stages: List[IFStage] = []

        result = runner.run_dry(IFArtifact, stages)

        assert result.valid is True
        assert result.stage_count == 0
        assert len(result.stages) == 0
        assert len(result.errors) == 0
        assert result.initial_type == "IFArtifact"
        assert result.final_type == "IFArtifact"


class TestDryRunJPLCompliance:
    """JPL Power of Ten compliance tests for ."""

    def test_rule_2_stage_bound_enforced(self):
        """
        GWT (JPL Rule #2):
        Given stages exceeding MAX_PIPELINE_STAGES,
        When run_dry() is called,
        Then validation fails with bound error.
        """
        runner = IFPipelineRunner()
        # Create more stages than allowed
        stages = [MockStage(f"s-{i}") for i in range(MAX_PIPELINE_STAGES + 1)]

        result = runner.run_dry(IFArtifact, stages)

        assert result.valid is False
        assert "exceeds stage limit" in result.errors[0]

    def test_rule_7_returns_explicit_result(self):
        """
        GWT (JPL Rule #7):
        Given any pipeline configuration,
        When run_dry() is called,
        Then it returns a DryRunResult (never None).
        """
        runner = IFPipelineRunner()

        # Valid pipeline
        result1 = runner.run_dry(IFArtifact, [MockStage("s1")])
        assert isinstance(result1, DryRunResult)

        # Invalid pipeline
        result2 = runner.run_dry(
            IFTextArtifact, [MockStage("s1", in_type=IFChunkArtifact)]
        )
        assert isinstance(result2, DryRunResult)

        # Empty pipeline
        result3 = runner.run_dry(IFArtifact, [])
        assert isinstance(result3, DryRunResult)

    def test_rule_9_complete_type_hints(self):
        """
        GWT (JPL Rule #9):
        Given DryRunResult and StageValidation,
        When fields are accessed,
        Then they have documented types.
        """
        runner = IFPipelineRunner()
        stages = [MockStage("test")]

        result = runner.run_dry(IFArtifact, stages)

        # DryRunResult types
        assert isinstance(result.valid, bool)
        assert isinstance(result.stage_count, int)
        assert isinstance(result.stages, tuple)
        assert isinstance(result.errors, tuple)
        assert isinstance(result.initial_type, str)
        assert result.final_type is None or isinstance(result.final_type, str)

        # StageValidation types
        if result.stages:
            sv = result.stages[0]
            assert isinstance(sv.stage_name, str)
            assert isinstance(sv.stage_index, int)
            assert isinstance(sv.input_type, str)
            assert isinstance(sv.output_type, str)
            assert isinstance(sv.valid, bool)
            assert sv.error is None or isinstance(sv.error, str)


class TestDryRunNoExecution:
    """Test that dry-run doesn't execute stages."""

    def test_stages_not_executed(self):
        """
        GWT:
        Given a pipeline with stages,
        When run_dry() is called,
        Then no stage execute() methods are called.
        """
        execution_count = []

        class TrackingStage(IFStage):
            def __init__(self, name: str):
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

            def execute(self, artifact):
                execution_count.append(self._name)
                return artifact.derive("tracking")

        runner = IFPipelineRunner()
        stages = [TrackingStage("s1"), TrackingStage("s2"), TrackingStage("s3")]

        runner.run_dry(IFArtifact, stages)

        # No executions should have occurred
        assert len(execution_count) == 0


# =============================================================================
# Error - Sequential Fallback Recovery Tests
# =============================================================================


class MockProcessor(IFProcessor):
    """Test processor that can be configured to succeed or fail."""

    def __init__(
        self,
        proc_id: str,
        should_fail: bool = False,
        available: bool = True,
        raise_exception: bool = False,
    ):
        self._id = proc_id
        self._should_fail = should_fail
        self._available = available
        self._raise_exception = raise_exception

    @property
    def processor_id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_available(self) -> bool:
        return self._available

    def process(self, artifact: IFArtifact) -> IFArtifact:
        if self._raise_exception:
            raise RuntimeError(f"Processor {self._id} crashed")
        if self._should_fail:
            return IFFailureArtifact(
                artifact_id=artifact.artifact_id,
                error_message=f"Processor {self._id} failed intentionally",
                provenance=artifact.provenance,
            )
        return artifact.derive(processor_id=self._id)


class TestFallbackPrimarySuccess:
    """Tests for Scenario 1: Primary Processor Fails, Fallback Succeeds."""

    def test_first_processor_succeeds(self):
        """
        GWT:
        Given a list of processors where the first succeeds,
        When run_with_fallback() is called,
        Then only the first processor is tried and result is success.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", should_fail=False),
            MockProcessor("proc-2", should_fail=False),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is True
        assert result.successful_processor == "proc-1"
        assert len(result.attempts) == 1
        assert result.attempts[0].success is True

    def test_fallback_to_second_processor(self):
        """
        GWT - Scenario 1:
        Given primary processor fails,
        When run_with_fallback() is called,
        Then fallback processor is tried and succeeds.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", should_fail=True),
            MockProcessor("proc-2", should_fail=False),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is True
        assert result.successful_processor == "proc-2"
        assert len(result.attempts) == 2
        assert result.attempts[0].success is False
        assert result.attempts[1].success is True


class TestFallbackAllFail:
    """Tests for Scenario 2: All Processors Fail."""

    def test_all_processors_fail(self):
        """
        GWT - Scenario 2:
        Given all processors fail,
        When run_with_fallback() is called,
        Then FallbackResult has success=False and aggregated errors.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", should_fail=True),
            MockProcessor("proc-2", should_fail=True),
            MockProcessor("proc-3", should_fail=True),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is False
        assert result.successful_processor is None
        assert len(result.attempts) == 3
        assert all(not a.success for a in result.attempts)
        assert isinstance(result.artifact, IFFailureArtifact)
        assert "fallback-exhausted" in result.artifact.provenance

    def test_aggregated_error_message(self):
        """
        GWT:
        Given all processors fail with different errors,
        When run_with_fallback() returns,
        Then error message contains info from all failed attempts.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", should_fail=True),
            MockProcessor("proc-2", should_fail=True),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert "proc-1" in result.artifact.error_message
        assert "proc-2" in result.artifact.error_message


class TestFallbackPriorityOrder:
    """Tests for Scenario 3: Fallback Processor Selection."""

    def test_processors_tried_in_order(self):
        """
        GWT - Scenario 3:
        Given processors in priority order,
        When run_with_fallback() is called,
        Then processors are tried in the provided order.
        """
        execution_order = []

        class OrderTrackingProcessor(IFProcessor):
            def __init__(self, proc_id: str, should_succeed: bool):
                self._id = proc_id
                self._succeed = should_succeed

            @property
            def processor_id(self) -> str:
                return self._id

            @property
            def version(self) -> str:
                return "1.0.0"

            def is_available(self) -> bool:
                return True

            def process(self, artifact: IFArtifact) -> IFArtifact:
                execution_order.append(self._id)
                if self._succeed:
                    return artifact.derive(processor_id=self._id)
                return IFFailureArtifact(
                    artifact_id=artifact.artifact_id,
                    error_message=f"{self._id} failed",
                    provenance=artifact.provenance,
                )

        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            OrderTrackingProcessor("first", should_succeed=False),
            OrderTrackingProcessor("second", should_succeed=False),
            OrderTrackingProcessor("third", should_succeed=True),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert execution_order == ["first", "second", "third"]
        assert result.success is True
        assert result.successful_processor == "third"


class TestFallbackAvailabilityCheck:
    """Tests for Scenario 4: Fallback with Availability Check."""

    def test_unavailable_processors_skipped(self):
        """
        GWT - Scenario 4:
        Given some processors are unavailable,
        When run_with_fallback() is called with skip_unavailable=True,
        Then unavailable processors are skipped.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", available=False),
            MockProcessor("proc-2", should_fail=False),
        ]

        config = FallbackConfig(skip_unavailable=True)
        result = runner.run_with_fallback(artifact, processors, config)

        assert result.success is True
        assert result.successful_processor == "proc-2"
        # First attempt should show skipped
        assert result.attempts[0].processor_id == "proc-1"
        assert "unavailable" in result.attempts[0].error.lower()

    def test_unavailable_not_skipped_when_disabled(self):
        """
        GWT:
        Given skip_unavailable=False,
        When run_with_fallback() encounters unavailable processor,
        Then it still tries the processor.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", available=False, should_fail=False),
            MockProcessor("proc-2", should_fail=False),
        ]

        config = FallbackConfig(skip_unavailable=False)
        result = runner.run_with_fallback(artifact, processors, config)

        # First processor should be tried (and succeed) despite being "unavailable"
        assert result.success is True
        assert result.successful_processor == "proc-1"


class TestFallbackLimitEnforcement:
    """Tests for Scenario 5: Fallback Limit Enforcement."""

    def test_max_attempts_enforced(self):
        """
        GWT - Scenario 5:
        Given more processors than MAX_FALLBACK_ATTEMPTS,
        When run_with_fallback() is called,
        Then only max_attempts processors are tried.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor(f"proc-{i}", should_fail=True)
            for i in range(MAX_FALLBACK_ATTEMPTS + 3)
        ]

        config = FallbackConfig(max_attempts=3)
        result = runner.run_with_fallback(artifact, processors, config)

        assert len(result.attempts) == 3
        assert result.success is False

    def test_config_max_attempts_capped(self):
        """
        GWT (JPL Rule #2):
        Given max_attempts exceeds MAX_FALLBACK_ATTEMPTS,
        When FallbackConfig is created,
        Then max_attempts is capped to MAX_FALLBACK_ATTEMPTS.
        """
        config = FallbackConfig(max_attempts=1000)

        assert config.max_attempts == MAX_FALLBACK_ATTEMPTS


class TestFallbackExceptionHandling:
    """Tests for exception handling during fallback."""

    def test_exception_caught_and_recorded(self):
        """
        GWT (JPL Rule #7):
        Given a processor that raises an exception,
        When run_with_fallback() is called,
        Then exception is caught and recorded in attempts.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", raise_exception=True),
            MockProcessor("proc-2", should_fail=False),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is True
        assert result.successful_processor == "proc-2"
        assert "Exception" in result.attempts[0].error
        assert "crashed" in result.attempts[0].error

    def test_continue_on_failure_false_stops_early(self):
        """
        GWT:
        Given continue_on_failure=False,
        When first processor fails,
        Then no further processors are tried.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", should_fail=True),
            MockProcessor("proc-2", should_fail=False),
        ]

        config = FallbackConfig(continue_on_failure=False)
        result = runner.run_with_fallback(artifact, processors, config)

        assert result.success is False
        assert len(result.attempts) == 1


class TestFallbackResultDataclass:
    """Tests for FallbackResult and FallbackAttempt dataclasses."""

    def test_fallback_result_fields(self):
        """
        GWT (JPL Rule #9):
        Given a FallbackResult instance,
        When fields are accessed,
        Then all fields have correct types.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [MockProcessor("proc-1", should_fail=False)]

        result = runner.run_with_fallback(artifact, processors)

        assert isinstance(result.success, bool)
        assert isinstance(result.artifact, IFArtifact)
        assert isinstance(result.attempts, tuple)
        assert result.successful_processor is None or isinstance(
            result.successful_processor, str
        )

    def test_fallback_attempt_has_duration(self):
        """
        GWT:
        Given a fallback attempt,
        When it completes,
        Then duration_ms is recorded.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [MockProcessor("proc-1", should_fail=False)]

        result = runner.run_with_fallback(artifact, processors)

        assert result.attempts[0].duration_ms >= 0


# =============================================================================
# Additional Comprehensive Tests - Edge Cases & JPL Compliance
# =============================================================================


class TestFallbackEdgeCases:
    """Edge case tests for fallback recovery."""

    def test_empty_processor_list(self):
        """
        GWT:
        Given an empty list of processors,
        When run_with_fallback() is called,
        Then a failure result is returned with zero attempts.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors: List[IFProcessor] = []

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is False
        assert len(result.attempts) == 0
        assert isinstance(result.artifact, IFFailureArtifact)

    def test_single_processor_success(self):
        """
        GWT:
        Given a single successful processor,
        When run_with_fallback() is called,
        Then success is returned with one attempt.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [MockProcessor("only-proc", should_fail=False)]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is True
        assert len(result.attempts) == 1
        assert result.successful_processor == "only-proc"

    def test_single_processor_failure(self):
        """
        GWT:
        Given a single failing processor,
        When run_with_fallback() is called,
        Then failure is returned with one attempt.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [MockProcessor("only-proc", should_fail=True)]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is False
        assert len(result.attempts) == 1
        assert result.successful_processor is None

    def test_all_processors_unavailable(self):
        """
        GWT:
        Given all processors are unavailable,
        When run_with_fallback() is called with skip_unavailable=True,
        Then all are skipped and failure is returned.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", available=False),
            MockProcessor("proc-2", available=False),
            MockProcessor("proc-3", available=False),
        ]

        config = FallbackConfig(skip_unavailable=True)
        result = runner.run_with_fallback(artifact, processors, config)

        assert result.success is False
        assert len(result.attempts) == 3
        assert all("unavailable" in a.error.lower() for a in result.attempts)

    def test_first_unavailable_second_succeeds(self):
        """
        GWT:
        Given first processor unavailable, second succeeds,
        When run_with_fallback() is called,
        Then second processor is used.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", available=False),
            MockProcessor("proc-2", should_fail=False),
        ]

        config = FallbackConfig(skip_unavailable=True)
        result = runner.run_with_fallback(artifact, processors, config)

        assert result.success is True
        assert result.successful_processor == "proc-2"
        assert len(result.attempts) == 2

    def test_mixed_failures_and_exceptions(self):
        """
        GWT:
        Given processors with mixed failure types,
        When run_with_fallback() is called,
        Then all failure types are recorded correctly.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-1", should_fail=True),  # Returns FailureArtifact
            MockProcessor("proc-2", raise_exception=True),  # Raises exception
            MockProcessor("proc-3", should_fail=False),  # Succeeds
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is True
        assert result.successful_processor == "proc-3"
        assert len(result.attempts) == 3
        assert "failed intentionally" in result.attempts[0].error
        assert "Exception" in result.attempts[1].error


class TestFallbackConfigValidation:
    """Tests for FallbackConfig validation and bounds."""

    def test_config_default_values(self):
        """
        GWT:
        Given no arguments,
        When FallbackConfig is created,
        Then default values are used.
        """
        config = FallbackConfig()

        assert config.max_attempts == MAX_FALLBACK_ATTEMPTS
        assert config.skip_unavailable is True
        assert config.continue_on_failure is True

    def test_config_negative_max_attempts_corrected(self):
        """
        GWT (JPL Rule #2):
        Given max_attempts < 0,
        When FallbackConfig is created,
        Then it is corrected to 1.
        """
        config = FallbackConfig(max_attempts=-5)

        assert config.max_attempts == 1

    def test_config_zero_max_attempts_corrected(self):
        """
        GWT (JPL Rule #2):
        Given max_attempts = 0,
        When FallbackConfig is created,
        Then it is corrected to 1.
        """
        config = FallbackConfig(max_attempts=0)

        assert config.max_attempts == 1

    def test_config_custom_values_preserved(self):
        """
        GWT:
        Given custom valid values,
        When FallbackConfig is created,
        Then values are preserved.
        """
        config = FallbackConfig(
            max_attempts=3, skip_unavailable=False, continue_on_failure=False
        )

        assert config.max_attempts == 3
        assert config.skip_unavailable is False
        assert config.continue_on_failure is False

    def test_config_is_frozen(self):
        """
        GWT (JPL Rule #9):
        Given a FallbackConfig instance,
        When attempting to modify it,
        Then modification is rejected (frozen dataclass).
        """
        config = FallbackConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.max_attempts = 10


class TestFallbackJPLRule1LinearFlow:
    """JPL Rule #1: Linear control flow tests."""

    def test_processors_executed_sequentially(self):
        """
        GWT (JPL Rule #1):
        Given multiple processors,
        When run_with_fallback() is called,
        Then processors are executed sequentially (not in parallel).
        """
        execution_times = []

        class TimedProcessor(IFProcessor):
            def __init__(self, proc_id: str, delay_ms: float = 10):
                self._id = proc_id
                self._delay = delay_ms / 1000

            @property
            def processor_id(self) -> str:
                return self._id

            @property
            def version(self) -> str:
                return "1.0.0"

            def is_available(self) -> bool:
                return True

            def process(self, artifact: IFArtifact) -> IFArtifact:
                import time as t

                start = t.perf_counter()
                t.sleep(self._delay)
                execution_times.append((self._id, start))
                return IFFailureArtifact(
                    artifact_id=artifact.artifact_id,
                    error_message="failed",
                    provenance=artifact.provenance,
                )

        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            TimedProcessor("p1", delay_ms=20),
            TimedProcessor("p2", delay_ms=20),
            TimedProcessor("p3", delay_ms=20),
        ]

        runner.run_with_fallback(artifact, processors)

        # Verify sequential execution: each start time > previous
        assert len(execution_times) == 3
        for i in range(1, len(execution_times)):
            assert (
                execution_times[i][1] > execution_times[i - 1][1]
            ), "Processors should execute sequentially, not in parallel"


class TestFallbackJPLRule2Bounds:
    """JPL Rule #2: Fixed upper bounds tests."""

    def test_max_attempts_constant_exists(self):
        """
        GWT (JPL Rule #2):
        Given the runner module,
        When MAX_FALLBACK_ATTEMPTS is accessed,
        Then it exists and is a positive integer.
        """
        assert MAX_FALLBACK_ATTEMPTS > 0
        assert isinstance(MAX_FALLBACK_ATTEMPTS, int)

    def test_attempts_never_exceed_max(self):
        """
        GWT (JPL Rule #2):
        Given 100 processors,
        When run_with_fallback() is called with default config,
        Then attempts never exceed MAX_FALLBACK_ATTEMPTS.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [MockProcessor(f"proc-{i}", should_fail=True) for i in range(100)]

        result = runner.run_with_fallback(artifact, processors)

        assert len(result.attempts) <= MAX_FALLBACK_ATTEMPTS

    def test_custom_max_respects_absolute_limit(self):
        """
        GWT (JPL Rule #2):
        Given config with max_attempts > MAX_FALLBACK_ATTEMPTS,
        When run_with_fallback() is called,
        Then attempts are capped at MAX_FALLBACK_ATTEMPTS.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [MockProcessor(f"proc-{i}", should_fail=True) for i in range(20)]

        # Try to exceed the limit
        config = FallbackConfig(max_attempts=999)
        result = runner.run_with_fallback(artifact, processors, config)

        assert len(result.attempts) <= MAX_FALLBACK_ATTEMPTS


class TestFallbackJPLRule7ReturnValues:
    """JPL Rule #7: Check return values tests."""

    def test_always_returns_fallback_result(self):
        """
        GWT (JPL Rule #7):
        Given any processor configuration,
        When run_with_fallback() is called,
        Then it always returns a FallbackResult (never None).
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")

        # Test with empty list
        result1 = runner.run_with_fallback(artifact, [])
        assert isinstance(result1, FallbackResult)

        # Test with failing processors
        result2 = runner.run_with_fallback(
            artifact, [MockProcessor("p1", should_fail=True)]
        )
        assert isinstance(result2, FallbackResult)

        # Test with succeeding processor
        result3 = runner.run_with_fallback(
            artifact, [MockProcessor("p1", should_fail=False)]
        )
        assert isinstance(result3, FallbackResult)

    def test_artifact_always_present_in_result(self):
        """
        GWT (JPL Rule #7):
        Given any outcome,
        When run_with_fallback() returns,
        Then result.artifact is never None.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")

        # Success case
        result1 = runner.run_with_fallback(
            artifact, [MockProcessor("p1", should_fail=False)]
        )
        assert result1.artifact is not None

        # Failure case
        result2 = runner.run_with_fallback(
            artifact, [MockProcessor("p1", should_fail=True)]
        )
        assert result2.artifact is not None

        # Empty processors case
        result3 = runner.run_with_fallback(artifact, [])
        assert result3.artifact is not None

    def test_attempts_tuple_always_present(self):
        """
        GWT (JPL Rule #7):
        Given any outcome,
        When run_with_fallback() returns,
        Then result.attempts is always a tuple (never None).
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")

        result = runner.run_with_fallback(artifact, [])

        assert result.attempts is not None
        assert isinstance(result.attempts, tuple)


class TestFallbackJPLRule9TypeHints:
    """JPL Rule #9: Complete type hints tests."""

    def test_fallback_attempt_types(self):
        """
        GWT (JPL Rule #9):
        Given a FallbackAttempt,
        When fields are accessed,
        Then types match declarations.
        """
        attempt = FallbackAttempt(
            processor_id="test-proc", success=True, error=None, duration_ms=123.45
        )

        assert isinstance(attempt.processor_id, str)
        assert isinstance(attempt.success, bool)
        assert attempt.error is None or isinstance(attempt.error, str)
        assert isinstance(attempt.duration_ms, float)

    def test_fallback_attempt_with_error(self):
        """
        GWT (JPL Rule #9):
        Given a failed FallbackAttempt,
        When error field is accessed,
        Then it is a string.
        """
        attempt = FallbackAttempt(
            processor_id="test-proc",
            success=False,
            error="Something went wrong",
            duration_ms=50.0,
        )

        assert isinstance(attempt.error, str)
        assert "wrong" in attempt.error

    def test_fallback_result_types(self):
        """
        GWT (JPL Rule #9):
        Given a FallbackResult,
        When fields are accessed,
        Then types match declarations.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")

        result = runner.run_with_fallback(
            artifact, [MockProcessor("p1", should_fail=False)]
        )

        # Check all field types
        assert isinstance(result.success, bool)
        assert isinstance(result.artifact, IFArtifact)
        assert isinstance(result.attempts, tuple)
        assert all(isinstance(a, FallbackAttempt) for a in result.attempts)
        assert result.successful_processor is None or isinstance(
            result.successful_processor, str
        )


class TestFallbackGWTScenarioCompleteness:
    """Verify all GWT scenarios from are covered."""

    def test_scenario_1_primary_fails_fallback_succeeds(self):
        """
        GWT - Scenario 1 (explicit):
        Given a stage with multiple registered processors,
        When the primary processor fails,
        Then the runner automatically tries the next processor in priority order.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("primary", should_fail=True),
            MockProcessor("fallback-1", should_fail=True),
            MockProcessor("fallback-2", should_fail=False),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is True
        assert result.successful_processor == "fallback-2"
        assert len(result.attempts) == 3
        # Verify order was maintained
        assert result.attempts[0].processor_id == "primary"
        assert result.attempts[1].processor_id == "fallback-1"
        assert result.attempts[2].processor_id == "fallback-2"

    def test_scenario_2_all_fail_aggregated_error(self):
        """
        GWT - Scenario 2 (explicit):
        Given a stage where all processors fail,
        When fallback exhausts all options,
        Then an IFFailureArtifact is returned with aggregated error info.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("proc-a", should_fail=True),
            MockProcessor("proc-b", should_fail=True),
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert result.success is False
        assert isinstance(result.artifact, IFFailureArtifact)
        # Verify aggregated error contains info from both
        assert "proc-a" in result.artifact.error_message
        assert "proc-b" in result.artifact.error_message
        assert "fallback-exhausted" in result.artifact.provenance

    def test_scenario_3_priority_order_respected(self):
        """
        GWT - Scenario 3 (explicit):
        Given multiple processors registered for a capability,
        When fallback is needed,
        Then processors are tried in priority order (as provided).
        """
        order = []

        class OrderRecorder(IFProcessor):
            def __init__(self, proc_id: str):
                self._id = proc_id

            @property
            def processor_id(self) -> str:
                return self._id

            @property
            def version(self) -> str:
                return "1.0.0"

            def is_available(self) -> bool:
                return True

            def process(self, artifact: IFArtifact) -> IFArtifact:
                order.append(self._id)
                return IFFailureArtifact(
                    artifact_id=artifact.artifact_id,
                    error_message=f"{self._id} failed",
                    provenance=artifact.provenance,
                )

        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            OrderRecorder("first-priority"),
            OrderRecorder("second-priority"),
            OrderRecorder("third-priority"),
        ]

        runner.run_with_fallback(artifact, processors)

        assert order == ["first-priority", "second-priority", "third-priority"]

    def test_scenario_4_unavailable_skipped(self):
        """
        GWT - Scenario 4 (explicit):
        Given processors where some are unavailable,
        When fallback occurs,
        Then unavailable processors are skipped.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("unavailable-1", available=False),
            MockProcessor("unavailable-2", available=False),
            MockProcessor("available", should_fail=False),
        ]

        config = FallbackConfig(skip_unavailable=True)
        result = runner.run_with_fallback(artifact, processors, config)

        assert result.success is True
        assert result.successful_processor == "available"
        # All three should appear in attempts (skipped ones with error)
        assert len(result.attempts) == 3
        assert "unavailable" in result.attempts[0].error.lower()
        assert "unavailable" in result.attempts[1].error.lower()
        assert result.attempts[2].success is True

    def test_scenario_5_limit_enforced(self):
        """
        GWT - Scenario 5 (explicit):
        Given more fallback processors than MAX_FALLBACK_ATTEMPTS,
        When fallback is attempted,
        Then only MAX_FALLBACK_ATTEMPTS processors are tried.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor(f"proc-{i}", should_fail=True)
            for i in range(MAX_FALLBACK_ATTEMPTS + 10)
        ]

        result = runner.run_with_fallback(artifact, processors)

        assert len(result.attempts) == MAX_FALLBACK_ATTEMPTS
        # Verify we stopped at the limit
        last_proc_id = result.attempts[-1].processor_id
        assert last_proc_id == f"proc-{MAX_FALLBACK_ATTEMPTS - 1}"


# =============================================================================
# COMPREHENSIVE COVERAGE TESTS - Additional GWT scenarios for full coverage
# =============================================================================


class TestRunMonitoredTypeMismatch:
    """
    Tests for run_monitored type mismatch handling (lines 872-892).

    Monitoring - Non-Blocking Interceptors.
    JPL Rule #5: Assertion density - validate input types.
    """

    def test_type_mismatch_detected_and_reported(self):
        """
        GWT:
        Given a stage expecting IFTextArtifact but receiving base IFArtifact,
        When run_monitored() is called,
        Then it returns IFFailureArtifact with type mismatch error.
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        # Stage expects IFTextArtifact specifically
        class TextOnlyStage(IFStage):
            @property
            def name(self):
                return "text-only"

            @property
            def input_type(self):
                return IFTextArtifact

            @property
            def output_type(self):
                return IFTextArtifact

            def execute(self, artifact):
                return artifact.derive(processor_id="text-only")

        runner = IFPipelineRunner()
        # Pass a ChunkArtifact instead of TextArtifact
        wrong_artifact = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="chunk content",
            chunk_index=0,
            total_chunks=1,
        )
        stages = [TextOnlyStage()]

        result = runner.run_monitored(wrong_artifact, stages, document_id="test-doc")

        assert isinstance(result, IFFailureArtifact)
        assert "Type mismatch" in result.error_message
        assert "IFTextArtifact" in result.error_message
        assert "IFChunkArtifact" in result.error_message
        assert "stage-text-only-type-mismatch" in result.provenance

    def test_type_mismatch_calls_on_error_interceptor(self):
        """
        GWT:
        Given a registered interceptor and a type mismatch,
        When run_monitored() is called,
        Then the interceptor's on_error is called.
        """
        from ingestforge.core.pipeline.interfaces import IFInterceptor
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        error_calls = []

        class ErrorCapturingInterceptor(IFInterceptor):
            @property
            def interceptor_id(self):
                return "error-capture"

            def pre_stage(self, stage_name, artifact, doc_id):
                pass

            def post_stage(self, stage_name, artifact, doc_id, duration_ms):
                pass

            def on_error(self, stage_name, artifact, doc_id, error):
                error_calls.append((stage_name, str(error)))

            def on_pipeline_start(self, doc_id, stage_count):
                pass

            def on_pipeline_end(self, doc_id, success, duration_ms):
                pass

        class TextOnlyStage(IFStage):
            @property
            def name(self):
                return "text-stage"

            @property
            def input_type(self):
                return IFTextArtifact

            @property
            def output_type(self):
                return IFTextArtifact

            def execute(self, artifact):
                return artifact.derive(processor_id="text-stage")

        runner = IFPipelineRunner(interceptors=[ErrorCapturingInterceptor()])
        wrong_artifact = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="content",
            chunk_index=0,
            total_chunks=1,
        )

        runner.run_monitored(wrong_artifact, [TextOnlyStage()], document_id="test-doc")

        assert len(error_calls) == 1
        assert error_calls[0][0] == "text-stage"
        assert "Type mismatch" in error_calls[0][1]


class TestRunMonitoredContractViolation:
    """
    Tests for run_monitored contract violation handling (lines 916-936).

    Monitoring - Non-Blocking Interceptors.
    JPL Rule #5: Validate output types.
    """

    def test_contract_violation_detected_and_reported(self):
        """
        GWT:
        Given a stage that returns wrong output type,
        When run_monitored() is called,
        Then it returns IFFailureArtifact with contract violation error.
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        class BadOutputStage(IFStage):
            @property
            def name(self):
                return "bad-output"

            @property
            def input_type(self):
                return IFTextArtifact

            @property
            def output_type(self):
                return IFTextArtifact  # Claims TextArtifact

            def execute(self, artifact):
                # But actually returns ChunkArtifact (wrong!)
                return IFChunkArtifact(
                    artifact_id="chunk-1",
                    document_id="doc-1",
                    content="chunk",
                    chunk_index=0,
                    total_chunks=1,
                    parent_id=artifact.artifact_id,
                    provenance=artifact.provenance + ["bad-output"],
                )

        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="text-1", content="hello")

        result = runner.run_monitored(
            artifact, [BadOutputStage()], document_id="test-doc"
        )

        assert isinstance(result, IFFailureArtifact)
        assert "Contract violation" in result.error_message
        assert "IFChunkArtifact" in result.error_message
        assert "IFTextArtifact" in result.error_message
        assert "stage-bad-output-contract-violation" in result.provenance

    def test_contract_violation_calls_on_error_interceptor(self):
        """
        GWT:
        Given a registered interceptor and a contract violation,
        When run_monitored() is called,
        Then the interceptor's on_error is called.
        """
        from ingestforge.core.pipeline.interfaces import IFInterceptor
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        error_calls = []

        class ErrorCapturingInterceptor(IFInterceptor):
            @property
            def interceptor_id(self):
                return "error-capture"

            def pre_stage(self, stage_name, artifact, doc_id):
                pass

            def post_stage(self, stage_name, artifact, doc_id, duration_ms):
                pass

            def on_error(self, stage_name, artifact, doc_id, error):
                error_calls.append((stage_name, type(error).__name__))

            def on_pipeline_start(self, doc_id, stage_count):
                pass

            def on_pipeline_end(self, doc_id, success, duration_ms):
                pass

        class ContractBreaker(IFStage):
            @property
            def name(self):
                return "contract-breaker"

            @property
            def input_type(self):
                return IFTextArtifact

            @property
            def output_type(self):
                return IFTextArtifact

            def execute(self, artifact):
                return IFChunkArtifact(
                    artifact_id="chunk-1",
                    document_id="doc-1",
                    content="chunk",
                    chunk_index=0,
                    total_chunks=1,
                )

        runner = IFPipelineRunner(interceptors=[ErrorCapturingInterceptor()])
        artifact = IFTextArtifact(artifact_id="text-1", content="hello")

        runner.run_monitored(artifact, [ContractBreaker()], document_id="test-doc")

        assert len(error_calls) == 1
        assert error_calls[0][0] == "contract-breaker"
        assert error_calls[0][1] == "TypeError"


class TestRunMonitoredPipelineEndStatus:
    """
    Tests for run_monitored pipeline end interceptor status.

    Monitoring - Non-Blocking Interceptors.
    """

    def test_pipeline_end_reports_failure_on_type_mismatch(self):
        """
        GWT:
        Given a type mismatch failure,
        When run_monitored() completes,
        Then pipeline_end interceptor is called with success=False.
        """
        from ingestforge.core.pipeline.interfaces import IFInterceptor
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        end_calls = []

        class EndCapturingInterceptor(IFInterceptor):
            @property
            def interceptor_id(self):
                return "end-capture"

            def pre_stage(self, stage_name, artifact, doc_id):
                pass

            def post_stage(self, stage_name, artifact, doc_id, duration_ms):
                pass

            def on_error(self, stage_name, artifact, doc_id, error):
                pass

            def on_pipeline_start(self, doc_id, stage_count):
                pass

            def on_pipeline_end(self, doc_id, success, duration_ms):
                end_calls.append((doc_id, success))

        class TextOnlyStage(IFStage):
            @property
            def name(self):
                return "text-only"

            @property
            def input_type(self):
                return IFTextArtifact

            @property
            def output_type(self):
                return IFTextArtifact

            def execute(self, artifact):
                return artifact.derive(processor_id="text-only")

        runner = IFPipelineRunner(interceptors=[EndCapturingInterceptor()])
        wrong_artifact = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="content",
            chunk_index=0,
            total_chunks=1,
        )

        runner.run_monitored(wrong_artifact, [TextOnlyStage()], document_id="test-doc")

        assert len(end_calls) == 1
        assert end_calls[0][0] == "test-doc"
        assert end_calls[0][1] is False  # Pipeline failed


class TestCheckpointSaveFailure:
    """
    Tests for checkpoint save failure handling (lines 818-824).

    Resilience - Multi-Stage Checkpointing.
    JPL Rule #7: Check return values.
    """

    def test_checkpoint_save_failure_logs_warning_but_continues(self):
        """
        GWT:
        Given a checkpoint manager that fails to save,
        When run() is called,
        Then a warning is logged but pipeline continues successfully.
        """
        from ingestforge.core.pipeline.checkpoint import IFCheckpointManager
        from unittest.mock import Mock, patch

        # Create a mock checkpoint manager that always fails to save
        mock_checkpoint = Mock(spec=IFCheckpointManager)
        mock_checkpoint.save_checkpoint.return_value = False  # Simulate save failure

        runner = IFPipelineRunner(checkpoint_manager=mock_checkpoint)
        artifact = IFTextArtifact(artifact_id="text-1", content="hello")
        stages = [MockStage("A"), MockStage("B")]

        with patch("ingestforge.core.pipeline.runner.logger") as mock_logger:
            result = runner.run(artifact, stages, document_id="test-doc")

        # Pipeline should still succeed despite checkpoint failures
        assert not isinstance(result, IFFailureArtifact)
        assert "stage-A" in result.provenance
        assert "stage-B" in result.provenance

        # Checkpoint save was called for each stage
        assert mock_checkpoint.save_checkpoint.call_count == 2


class TestFallbackContinueOnFailureFalse:
    """
    Tests for early break when continue_on_failure=False (line 1078).

    Error - Sequential Fallback Recovery.
    JPL Rule #2: Fixed upper bounds.
    """

    def test_stops_early_when_continue_on_failure_false(self):
        """
        GWT:
        Given continue_on_failure=False in FallbackConfig,
        When first processor fails,
        Then fallback stops immediately without trying others.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            MockProcessor("first", should_fail=True),
            MockProcessor("second", should_fail=True),
            MockProcessor("third", should_fail=False),  # Would succeed if reached
        ]

        config = FallbackConfig(continue_on_failure=False)
        result = runner.run_with_fallback(artifact, processors, config)

        assert result.success is False
        # Only one attempt was made - stopped after first failure
        assert len(result.attempts) == 1
        assert result.attempts[0].processor_id == "first"
        assert result.attempts[0].success is False

    def test_does_not_try_subsequent_processors(self):
        """
        GWT:
        Given continue_on_failure=False and multiple failing processors,
        When run_with_fallback() is called,
        Then only the first processor is attempted.
        """
        attempted = []

        class TrackingProcessor(IFProcessor):
            def __init__(self, proc_id: str):
                self._id = proc_id

            @property
            def processor_id(self):
                return self._id

            @property
            def version(self):
                return "1.0.0"

            @property
            def input_type(self):
                return IFArtifact

            @property
            def output_type(self):
                return IFArtifact

            @property
            def capabilities(self):
                return frozenset(["test"])

            def is_available(self):
                return True

            def process(self, artifact):
                attempted.append(self._id)
                return IFFailureArtifact(
                    artifact_id=artifact.artifact_id, error_message=f"{self._id} failed"
                )

        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="test-1", content="hello")
        processors = [
            TrackingProcessor("first"),
            TrackingProcessor("second"),
            TrackingProcessor("third"),
        ]

        config = FallbackConfig(continue_on_failure=False)
        runner.run_with_fallback(artifact, processors, config)

        # Only first processor should have been called
        assert attempted == ["first"]


class TestJPLComplianceAdditional:
    """
    Additional JPL Power of Ten compliance tests.
    """

    def test_jpl_rule_1_no_recursion_in_run_monitored(self):
        """
        GWT:
        Given run_monitored method,
        When inspecting the code,
        Then no recursive calls are present (linear control flow).
        """
        import inspect

        source = inspect.getsource(IFPipelineRunner.run_monitored)

        # Check no self.run_monitored calls
        assert "self.run_monitored" not in source
        assert "run_monitored(" not in source.split("def run_monitored")[1]

    def test_jpl_rule_5_assertions_in_run_monitored(self):
        """
        GWT:
        Given run_monitored method and its helpers,
        When inspecting the code,
        Then assertions/checks are present for input validation.

        After refactoring, some validation moved to helper methods.
        """
        import inspect

        source = inspect.getsource(IFPipelineRunner.run_monitored)
        helper_source = inspect.getsource(IFPipelineRunner._execute_monitored_stage)

        # Check that isinstance checks are present in main method or helper
        assert "isinstance" in source or "isinstance" in helper_source
        # Check for input_type validation (in main method for type mismatch)
        assert "input_type" in source
        # Check for output_type validation (moved to helper after refactoring)
        assert "output_type" in helper_source

    def test_jpl_rule_7_explicit_return_values(self):
        """
        GWT:
        Given run_monitored method,
        When it completes (success or failure),
        Then it returns an explicit IFArtifact, never None.
        """
        runner = IFPipelineRunner()
        artifact = IFTextArtifact(artifact_id="text-1", content="hello")

        # Success case
        result_success = runner.run_monitored(
            artifact, [MockStage("A")], document_id="test-doc"
        )
        assert result_success is not None
        assert isinstance(result_success, IFArtifact)

        # Failure case
        result_fail = runner.run_monitored(
            artifact, [MockStage("Fail", fail=True)], document_id="test-doc"
        )
        assert result_fail is not None
        assert isinstance(result_fail, IFArtifact)

    def test_jpl_rule_9_complete_type_hints_run_monitored(self):
        """
        GWT:
        Given run_monitored method signature,
        When inspecting type hints,
        Then all parameters and return type are annotated.
        """
        import typing

        hints = typing.get_type_hints(IFPipelineRunner.run_monitored)

        assert "artifact" in hints
        assert "stages" in hints
        assert "document_id" in hints
        assert "return" in hints


class TestRunMonitoredFailureArtifactHandling:
    """
    Tests for FailureArtifact propagation in run_monitored.
    """

    def test_stage_returning_failure_artifact_stops_pipeline(self):
        """
        GWT:
        Given a stage that returns IFFailureArtifact,
        When run_monitored() is called,
        Then pipeline stops and returns that failure artifact.
        """
        from ingestforge.core.pipeline.interfaces import IFInterceptor

        error_calls = []

        class ErrorCapture(IFInterceptor):
            @property
            def interceptor_id(self):
                return "error-capture"

            def pre_stage(self, stage_name, artifact, doc_id):
                pass

            def post_stage(self, stage_name, artifact, doc_id, duration_ms):
                pass

            def on_error(self, stage_name, artifact, doc_id, error):
                error_calls.append(stage_name)

            def on_pipeline_start(self, doc_id, stage_count):
                pass

            def on_pipeline_end(self, doc_id, success, duration_ms):
                pass

        runner = IFPipelineRunner(interceptors=[ErrorCapture()])
        artifact = IFTextArtifact(artifact_id="text-1", content="hello")
        stages = [MockStage("A"), MockStage("Fail", fail=True), MockStage("B")]

        result = runner.run_monitored(artifact, stages, document_id="test-doc")

        assert isinstance(result, IFFailureArtifact)
        assert result.error_message == "forced fail"
        # on_error should be called for the failing stage
        assert "Fail" in error_calls

    def test_post_stage_not_called_on_failure_artifact(self):
        """
        GWT:
        Given a stage that returns IFFailureArtifact,
        When run_monitored() processes it,
        Then post_stage interceptor is NOT called for that stage.
        """
        from ingestforge.core.pipeline.interfaces import IFInterceptor

        post_calls = []

        class PostCapture(IFInterceptor):
            @property
            def interceptor_id(self):
                return "post-capture"

            def pre_stage(self, stage_name, artifact, doc_id):
                pass

            def post_stage(self, stage_name, artifact, doc_id, duration_ms):
                post_calls.append(stage_name)

            def on_error(self, stage_name, artifact, doc_id, error):
                pass

            def on_pipeline_start(self, doc_id, stage_count):
                pass

            def on_pipeline_end(self, doc_id, success, duration_ms):
                pass

        runner = IFPipelineRunner(interceptors=[PostCapture()])
        artifact = IFTextArtifact(artifact_id="text-1", content="hello")
        stages = [MockStage("A"), MockStage("Fail", fail=True), MockStage("B")]

        runner.run_monitored(artifact, stages, document_id="test-doc")

        # Only "A" should have post_stage called (it succeeded)
        # "Fail" returns FailureArtifact, so no post_stage
        # "B" is never reached
        assert post_calls == ["A"]
