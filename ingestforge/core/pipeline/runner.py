"""
Sequential Pipeline Runner for IngestForge (IF).

Executes a series of IFStages in linear order.
Enhanced with streaming callback support.

Follows NASA JPL Power of Ten rules.
"""

import logging
import time
import traceback
import concurrent.futures
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Type
from ingestforge.core.pipeline.interfaces import (
    IFArtifact,
    IFStage,
    IFInterceptor,
    IFProcessor,
    MAX_INTERCEPTORS,
)
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact
from ingestforge.core.pipeline.checkpoint import IFCheckpointManager

# JPL Rule #2: Fixed upper bounds for resource limits
MAX_PIPELINE_STAGES = 32
DEFAULT_STAGE_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_MAX_MEMORY_MB = 4096  # 4 GB
MAX_TIMEOUT_SECONDS = 3600  # 1 hour absolute maximum
MAX_FALLBACK_ATTEMPTS = 5  # Maximum fallback processor attempts

logger = logging.getLogger(__name__)


# Import shared memory utility to avoid duplication (defines canonical version)
def get_available_memory_mb() -> int:
    """
    Get available system memory in megabytes.

    Boundaries - Resource Bounded Execution.
    Rule #7: Check return values (returns sensible default on failure).

    Returns:
        Available memory in MB, or DEFAULT_MAX_MEMORY_MB as fallback.
    """
    try:
        # Delegate to registry's implementation to avoid duplication
        from ingestforge.core.pipeline.registry import (
            get_available_memory_mb as _get_memory,
        )

        return _get_memory()
    except ImportError:
        # Fallback if registry not available
        try:
            import psutil

            mem = psutil.virtual_memory()
            return int(mem.available / (1024 * 1024))
        except ImportError:
            logger.debug("psutil not available, memory monitoring disabled")
            return DEFAULT_MAX_MEMORY_MB
        except Exception as e:
            logger.warning(f"Failed to get system memory: {e}")
            return DEFAULT_MAX_MEMORY_MB


@dataclass(frozen=True)
class StageValidation:
    """
    Validation result for a single stage.

    Validation - Dry-Run Mode.
    Rule #9: Complete type hints.

    Attributes:
        stage_name: Name of the validated stage.
        stage_index: Position in pipeline (0-indexed).
        input_type: Expected input artifact type name.
        output_type: Declared output artifact type name.
        valid: True if type chain is valid at this stage.
        error: Error message if validation failed.
    """

    stage_name: str
    stage_index: int
    input_type: str
    output_type: str
    valid: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class DryRunResult:
    """
    Result of pipeline dry-run validation.

    Validation - Dry-Run Mode.
    Rule #9: Complete type hints.

    Attributes:
        valid: True if entire pipeline type chain is valid.
        stage_count: Number of stages validated.
        stages: Per-stage validation results.
        errors: List of all validation error messages.
        initial_type: Type of initial artifact.
        final_type: Expected final artifact type (if valid).
    """

    valid: bool
    stage_count: int
    stages: Tuple[StageValidation, ...]
    errors: Tuple[str, ...]
    initial_type: str
    final_type: Optional[str] = None


@dataclass(frozen=True)
class FallbackAttempt:
    """
    Record of a single fallback attempt.

    Error - Sequential Fallback Recovery.
    Rule #9: Complete type hints.

    Attributes:
        processor_id: ID of the processor that was tried.
        success: True if this attempt succeeded.
        error: Error message if attempt failed.
        duration_ms: Execution time in milliseconds.
    """

    processor_id: str
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass(frozen=True)
class FallbackResult:
    """
    Result of fallback execution.

    Error - Sequential Fallback Recovery.
    Rule #9: Complete type hints.

    Attributes:
        success: True if any processor succeeded.
        artifact: Result artifact (success or failure).
        attempts: Record of all attempted processors.
        successful_processor: ID of processor that succeeded (if any).
    """

    success: bool
    artifact: IFArtifact
    attempts: Tuple[FallbackAttempt, ...]
    successful_processor: Optional[str] = None


@dataclass(frozen=True)
class FallbackConfig:
    """
    Configuration for fallback recovery.

    Error - Sequential Fallback Recovery.
    Rule #2: Fixed upper bounds.
    Rule #9: Complete type hints.

    Attributes:
        max_attempts: Maximum processors to try (default 5, max MAX_FALLBACK_ATTEMPTS).
        skip_unavailable: Skip processors that report unavailable (default True).
        continue_on_failure: Try next processor on failure (default True).
    """

    max_attempts: int = MAX_FALLBACK_ATTEMPTS
    skip_unavailable: bool = True
    continue_on_failure: bool = True

    def __post_init__(self) -> None:
        """Validate fallback configuration bounds."""
        if self.max_attempts > MAX_FALLBACK_ATTEMPTS:
            object.__setattr__(self, "max_attempts", MAX_FALLBACK_ATTEMPTS)
        if self.max_attempts <= 0:
            object.__setattr__(self, "max_attempts", 1)


@dataclass(frozen=True)
class ResourceConfig:
    """
    Configuration for pipeline resource limits.

    Boundaries - Resource Bounded Execution.
    Rule #2: Fixed upper bounds on all resources.

    Attributes:
        timeout_seconds: Maximum seconds per stage (default 300, max 3600).
        max_memory_mb: Maximum memory in MB for pipeline (default 4096).
        max_stages: Maximum number of stages (default 32).
        warn_memory_threshold: Percentage of max_memory_mb to warn at (default 0.8).
    """

    timeout_seconds: int = DEFAULT_STAGE_TIMEOUT_SECONDS
    max_memory_mb: int = DEFAULT_MAX_MEMORY_MB
    max_stages: int = MAX_PIPELINE_STAGES
    warn_memory_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate resource configuration bounds."""
        if self.timeout_seconds > MAX_TIMEOUT_SECONDS:
            object.__setattr__(self, "timeout_seconds", MAX_TIMEOUT_SECONDS)
        if self.timeout_seconds <= 0:
            object.__setattr__(self, "timeout_seconds", DEFAULT_STAGE_TIMEOUT_SECONDS)
        if self.max_stages > MAX_PIPELINE_STAGES:
            object.__setattr__(self, "max_stages", MAX_PIPELINE_STAGES)
        if self.max_stages <= 0:
            object.__setattr__(self, "max_stages", MAX_PIPELINE_STAGES)


class IFPipelineRunner:
    """
    Orchestrator for linear stage execution.

    Supports resource-bounded execution with timeouts and memory limits.
    Supports non-blocking interceptors for monitoring.
    Supports automatic teardown of stages after execution.
    """

    def __init__(
        self,
        checkpoint_manager: Optional[IFCheckpointManager] = None,
        resource_config: Optional[ResourceConfig] = None,
        interceptors: Optional[List[IFInterceptor]] = None,
        auto_teardown: bool = True,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.resource_config = resource_config or ResourceConfig()
        self._auto_teardown = auto_teardown  # Enable automatic teardown

        # Initialize interceptors with bound check (JPL Rule #2)
        if interceptors is None:
            self._interceptors: List[IFInterceptor] = []
        elif len(interceptors) > MAX_INTERCEPTORS:
            logger.warning(
                f"Interceptor count {len(interceptors)} exceeds limit "
                f"{MAX_INTERCEPTORS}, truncating"
            )
            self._interceptors = interceptors[:MAX_INTERCEPTORS]
        else:
            self._interceptors = list(interceptors)

    # -------------------------------------------------------------------------
    # Teardown Support
    # -------------------------------------------------------------------------

    def teardown_stages(self, stages: List[IFStage]) -> bool:
        """
        Call teardown on all stages that support it.

        Processor Teardown in Pipeline.
        Rule #7: Isolated teardown - one failure doesn't block others.

        Args:
            stages: List of stages to teardown.

        Returns:
            True if all teardowns successful, False if any failed.
        """
        all_success = True

        for stage in stages:
            if hasattr(stage, "teardown") and callable(getattr(stage, "teardown")):
                try:
                    result = stage.teardown()
                    if not result:
                        logger.warning(f"Stage {stage.name} teardown returned False")
                        all_success = False
                except Exception as e:
                    logger.warning(f"Stage {stage.name} teardown failed: {e}")
                    all_success = False

        return all_success

    def add_interceptor(self, interceptor: IFInterceptor) -> bool:
        """
        Add an interceptor to the pipeline.

        Monitoring - Non-Blocking Interceptors.
        Rule #2: Fixed upper bound on interceptors.

        Args:
            interceptor: The interceptor to add.

        Returns:
            True if added, False if limit reached.
        """
        if len(self._interceptors) >= MAX_INTERCEPTORS:
            logger.warning(f"Cannot add interceptor: limit {MAX_INTERCEPTORS} reached")
            return False
        self._interceptors.append(interceptor)
        return True

    def _call_interceptors_pre_stage(
        self, stage_name: str, artifact: IFArtifact, document_id: str
    ) -> None:
        """
        Call pre_stage on all interceptors (isolated).

        Monitoring - Non-Blocking Interceptors.
        Rule #7: Exceptions are caught and logged.
        """
        for interceptor in self._interceptors:
            try:
                interceptor.pre_stage(stage_name, artifact, document_id)
            except Exception as e:
                logger.warning(
                    f"Interceptor {type(interceptor).__name__}.pre_stage failed: {e}"
                )

    def _call_interceptors_post_stage(
        self,
        stage_name: str,
        artifact: IFArtifact,
        document_id: str,
        duration_ms: float,
    ) -> None:
        """
        Call post_stage on all interceptors (isolated).

        Monitoring - Non-Blocking Interceptors.
        Rule #7: Exceptions are caught and logged.
        """
        for interceptor in self._interceptors:
            try:
                interceptor.post_stage(stage_name, artifact, document_id, duration_ms)
            except Exception as e:
                logger.warning(
                    f"Interceptor {type(interceptor).__name__}.post_stage failed: {e}"
                )

    def _call_interceptors_on_error(
        self, stage_name: str, artifact: IFArtifact, document_id: str, error: Exception
    ) -> None:
        """
        Call on_error on all interceptors (isolated).

        Monitoring - Non-Blocking Interceptors.
        Rule #7: Exceptions are caught and logged.
        """
        for interceptor in self._interceptors:
            try:
                interceptor.on_error(stage_name, artifact, document_id, error)
            except Exception as e:
                logger.warning(
                    f"Interceptor {type(interceptor).__name__}.on_error failed: {e}"
                )

    def _call_interceptors_pipeline_start(
        self, document_id: str, stage_count: int
    ) -> None:
        """
        Call on_pipeline_start on all interceptors (isolated).

        Monitoring - Non-Blocking Interceptors.
        """
        for interceptor in self._interceptors:
            try:
                interceptor.on_pipeline_start(document_id, stage_count)
            except Exception as e:
                logger.warning(
                    f"Interceptor {type(interceptor).__name__}.on_pipeline_start failed: {e}"
                )

    def _call_interceptors_pipeline_end(
        self, document_id: str, success: bool, total_duration_ms: float
    ) -> None:
        """
        Call on_pipeline_end on all interceptors (isolated).

        Monitoring - Non-Blocking Interceptors.
        """
        for interceptor in self._interceptors:
            try:
                interceptor.on_pipeline_end(document_id, success, total_duration_ms)
            except Exception as e:
                logger.warning(
                    f"Interceptor {type(interceptor).__name__}.on_pipeline_end failed: {e}"
                )

    def validate_pipeline(self, stages: List[IFStage]) -> Optional[str]:
        """
        Validate pipeline before execution.

        Boundaries - Resource Bounded Execution.
        Rule #2: Fixed upper bounds on stages.

        Args:
            stages: List of stages to validate.

        Returns:
            Error message if validation fails, None if valid.
        """
        if len(stages) > self.resource_config.max_stages:
            return (
                f"Pipeline exceeds stage limit: {len(stages)} > "
                f"{self.resource_config.max_stages}"
            )
        if len(stages) == 0:
            return "Pipeline has no stages"
        return None

    def _validate_stage_type(
        self,
        stage: IFStage,
        current_type: Type[IFArtifact],
        idx: int,
        stages: List[IFStage],
    ) -> Tuple[StageValidation, Optional[str]]:
        """
        Validate type compatibility for a single stage.

        Extracted helper for JPL Rule #4 compliance.

        Returns:
            Tuple of (StageValidation, error_message or None).
        """
        stage_valid = True
        stage_error: Optional[str] = None

        if not issubclass(current_type, stage.input_type):
            stage_valid = False
            if idx == 0:
                stage_error = (
                    f"Initial artifact type {current_type.__name__} "
                    f"incompatible with stage '{stage.name}' "
                    f"(expects {stage.input_type.__name__})"
                )
            else:
                prev_stage = stages[idx - 1]
                stage_error = (
                    f"Stage '{prev_stage.name}' output type "
                    f"{current_type.__name__} incompatible with "
                    f"stage '{stage.name}' input "
                    f"(expects {stage.input_type.__name__})"
                )

        validation = StageValidation(
            stage_name=stage.name,
            stage_index=idx,
            input_type=stage.input_type.__name__,
            output_type=stage.output_type.__name__,
            valid=stage_valid,
            error=stage_error,
        )
        return validation, stage_error

    def run_dry(
        self, artifact_type: Type[IFArtifact], stages: List[IFStage]
    ) -> DryRunResult:
        """
        Validate pipeline type chain without execution.

        Validation - Dry-Run Mode.
        Refactored for JPL Rule #4 compliance.
        """
        if len(stages) > self.resource_config.max_stages:
            return DryRunResult(
                valid=False,
                stage_count=len(stages),
                stages=tuple(),
                errors=(
                    f"Pipeline exceeds stage limit: {len(stages)} > "
                    f"{self.resource_config.max_stages}",
                ),
                initial_type=artifact_type.__name__,
                final_type=None,
            )

        if len(stages) == 0:
            return DryRunResult(
                valid=True,
                stage_count=0,
                stages=tuple(),
                errors=tuple(),
                initial_type=artifact_type.__name__,
                final_type=artifact_type.__name__,
            )

        stage_validations: List[StageValidation] = []
        errors: List[str] = []
        current_type: Type[IFArtifact] = artifact_type

        for idx, stage in enumerate(stages):
            validation, error = self._validate_stage_type(
                stage, current_type, idx, stages
            )
            stage_validations.append(validation)
            if error:
                errors.append(error)
            current_type = stage.output_type

        all_valid = len(errors) == 0
        return DryRunResult(
            valid=all_valid,
            stage_count=len(stages),
            stages=tuple(stage_validations),
            errors=tuple(errors),
            initial_type=artifact_type.__name__,
            final_type=current_type.__name__ if all_valid else None,
        )

    def _check_memory_threshold(self) -> Optional[str]:
        """
        Check if memory usage is approaching threshold.

        Boundaries - Resource Bounded Execution.
        Rule #7: Check return values.

        Returns:
            Warning message if threshold exceeded, None otherwise.
        """
        available_mb = get_available_memory_mb()
        threshold_mb = int(
            self.resource_config.max_memory_mb
            * self.resource_config.warn_memory_threshold
        )

        if available_mb < threshold_mb:
            return (
                f"Memory warning: {available_mb}MB available, "
                f"threshold {threshold_mb}MB"
            )
        return None

    def _execute_with_timeout(
        self, stage: IFStage, artifact: IFArtifact, timeout_seconds: int
    ) -> IFArtifact:
        """
        Execute a stage with timeout enforcement.

        Boundaries - Resource Bounded Execution.
        Rule #7: Check return values.

        Args:
            stage: Stage to execute.
            artifact: Input artifact.
            timeout_seconds: Maximum execution time.

        Returns:
            Output artifact or IFFailureArtifact on timeout.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(stage.execute, artifact)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                # Cancel the future (best effort)
                future.cancel()
                error_msg = (
                    f"Stage {stage.name} exceeded timeout of "
                    f"{timeout_seconds} seconds"
                )
                logger.error(error_msg)
                return IFFailureArtifact(
                    artifact_id=artifact.artifact_id,
                    error_message=error_msg,
                    parent_id=artifact.artifact_id,
                    provenance=artifact.provenance + [f"stage-{stage.name}-timeout"],
                )

    def _teardown_stages(self, stages: List[IFStage]) -> None:
        """
        Teardown all stages that support it.

        Processor Teardown in Pipeline.
        Rule #1: Linear control flow.
        Rule #7: Exceptions are caught and logged.

        Args:
            stages: List of stages to teardown.
        """
        for stage in stages:
            if hasattr(stage, "teardown"):
                try:
                    success = stage.teardown()
                    if not success:
                        logger.warning(f"Stage {stage.name} teardown returned False")
                except Exception as e:
                    logger.warning(f"Stage {stage.name} teardown failed: {e}")

    def run(
        self, artifact: IFArtifact, stages: List[IFStage], document_id: str
    ) -> IFArtifact:
        """
        Execute stages in sequence.

        Teardown is called in finally block.
        Refactored for JPL Rule #4 compliance.
        Rule #1: Linear control flow (no recursion).
        Rule #7: Check return values (wrapped in try/except).
        """
        current_artifact = artifact

        try:
            for stage in stages:
                if not isinstance(current_artifact, stage.input_type):
                    return self._create_type_mismatch_failure(current_artifact, stage)

                try:
                    logger.info(f"Executing Stage: {stage.name}")
                    current_artifact = stage.execute(current_artifact)

                    if isinstance(current_artifact, IFFailureArtifact):
                        logger.error(
                            f"Stage {stage.name} failed: {current_artifact.error_message}"
                        )
                        return current_artifact

                    if not isinstance(current_artifact, stage.output_type):
                        return self._create_contract_violation_failure(
                            current_artifact, stage
                        )

                    self._save_checkpoint_if_configured(
                        current_artifact, document_id, stage.name
                    )

                except Exception as e:
                    return self._create_crash_failure(current_artifact, stage, e)

            return current_artifact
        finally:
            if self._auto_teardown:
                self._teardown_stages(stages)

    def run_streaming(
        self,
        artifact: IFArtifact,
        stages: List[IFStage],
        document_id: str,
        on_chunk_complete: Optional[Callable[[IFChunkArtifact], None]] = None,
    ) -> IFArtifact:
        """
        Execute stages with streaming callback support.

        Streaming Foundry API.
        Calls on_chunk_complete for each IFChunkArtifact produced.

        Args:
            artifact: Initial artifact
            stages: Pipeline stages to execute
            document_id: Document identifier
            on_chunk_complete: Optional callback for each chunk

        Returns:
            Final artifact

        Rule #1: Linear control flow (no recursion).
        Rule #4: < 60 lines.
        Rule #7: Check return values (wrapped in try/except).
        Rule #9: 100% type hints.
        """
        current_artifact = artifact

        try:
            for stage in stages:
                if not isinstance(current_artifact, stage.input_type):
                    return self._create_type_mismatch_failure(current_artifact, stage)

                try:
                    logger.info(f"Executing Stage: {stage.name}")
                    current_artifact = stage.execute(current_artifact)

                    if isinstance(current_artifact, IFFailureArtifact):
                        logger.error(
                            f"Stage {stage.name} failed: {current_artifact.error_message}"
                        )
                        return current_artifact

                    if not isinstance(current_artifact, stage.output_type):
                        return self._create_contract_violation_failure(
                            current_artifact, stage
                        )

                    # Stream chunk callback
                    if on_chunk_complete and isinstance(
                        current_artifact, IFChunkArtifact
                    ):
                        on_chunk_complete(current_artifact)

                    self._save_checkpoint_if_configured(
                        current_artifact, document_id, stage.name
                    )

                except Exception as e:
                    return self._create_crash_failure(current_artifact, stage, e)

            return current_artifact
        finally:
            if self._auto_teardown:
                self._teardown_stages(stages)

    def _load_checkpoint_artifact(
        self, document_id: str, stage_name: str, artifact_type: Type[IFArtifact]
    ) -> Optional[IFArtifact]:
        """
        Load and validate a checkpoint artifact.

        Extracted helper for JPL Rule #4 compliance.

        Returns:
            Loaded artifact or None on failure.
        """
        loaded_artifact = self.checkpoint_manager.load_checkpoint(
            artifact_type, document_id, stage_name
        )

        if loaded_artifact is None:
            logger.error(f"Failed to load checkpoint for {document_id} at {stage_name}")
            return None

        if not isinstance(loaded_artifact, artifact_type):
            logger.error(
                f"Checkpoint type mismatch: expected {artifact_type.__name__}, "
                f"got {type(loaded_artifact).__name__}"
            )
            return None

        return loaded_artifact

    def resume(
        self,
        stages: List[IFStage],
        document_id: str,
        artifact_type: Optional[Type[IFArtifact]] = None,
    ) -> Optional[Tuple[IFArtifact, List[IFStage]]]:
        """
        Resume processing from the latest checkpoint.

        Recovery - Deterministic Resumption.
        Refactored for JPL Rule #4 compliance.
        """
        if len(stages) > self.resource_config.max_stages:
            logger.error(f"Pipeline exceeds {self.resource_config.max_stages} stages")
            return None

        if not self.checkpoint_manager:
            logger.warning("No checkpoint manager configured, cannot resume")
            return None

        stage_order = [s.name for s in stages]
        checkpoint_info = self.checkpoint_manager.get_latest_checkpoint(
            document_id, stage_order
        )

        if checkpoint_info is None:
            logger.info(f"No checkpoint found for {document_id}, fresh start required")
            return None

        stage_name, stage_index = checkpoint_info
        resolved_type = artifact_type or stages[stage_index].output_type

        loaded_artifact = self._load_checkpoint_artifact(
            document_id, stage_name, resolved_type
        )
        if loaded_artifact is None:
            return None

        remaining_stages = stages[stage_index + 1 :]
        logger.info(
            f"Resuming {document_id} from checkpoint '{stage_name}' "
            f"({len(remaining_stages)} stages remaining)"
        )
        return (loaded_artifact, remaining_stages)

    def run_with_resume(
        self, artifact: IFArtifact, stages: List[IFStage], document_id: str
    ) -> IFArtifact:
        """
        Run pipeline with automatic resume from checkpoint if available.

        Recovery - Deterministic Resumption.
        Rule #1: Linear control flow.
        Rule #7: Check return values.

        Args:
            artifact: The initial artifact (used only if no checkpoint exists).
            stages: Full list of pipeline stages.
            document_id: The document identifier.

        Returns:
            Final processed artifact or IFFailureArtifact on error.
        """
        # Try to resume from checkpoint
        resume_result = self.resume(stages, document_id)

        if resume_result is not None:
            loaded_artifact, remaining_stages = resume_result

            # If no stages remaining, return the loaded artifact
            if not remaining_stages:
                logger.info(f"Document {document_id} already fully processed")
                return loaded_artifact

            # Continue from checkpoint
            return self.run(loaded_artifact, remaining_stages, document_id)

        # No checkpoint, start fresh
        logger.info(f"Starting fresh processing for {document_id}")
        return self.run(artifact, stages, document_id)

    def _execute_bounded_stage(
        self, stage: IFStage, artifact: IFArtifact, document_id: str, timeout: int
    ) -> Tuple[IFArtifact, bool]:
        """
        Execute a single stage with timeout and resource bounds.

        Extracted helper for JPL Rule #4 compliance.

        Returns:
            Tuple of (result_artifact, success).
        """
        memory_warning = self._check_memory_threshold()
        if memory_warning:
            logger.warning(memory_warning)

        try:
            logger.info(f"Executing Stage: {stage.name} (timeout: {timeout}s)")
            result = self._execute_with_timeout(stage, artifact, timeout)

            if isinstance(result, IFFailureArtifact):
                logger.error(f"Stage {stage.name} failed: {result.error_message}")
                return result, False

            if not isinstance(result, stage.output_type):
                failure = self._create_contract_violation_failure(result, stage)
                return failure, False

            self._save_checkpoint_if_configured(result, document_id, stage.name)
            return result, True

        except Exception as e:
            failure = self._create_crash_failure(artifact, stage, e)
            return failure, False

    def run_bounded(
        self, artifact: IFArtifact, stages: List[IFStage], document_id: str
    ) -> IFArtifact:
        """
        Execute pipeline with resource boundary enforcement.

        Boundaries - Resource Bounded Execution.
        Teardown is called in finally block.
        Refactored for JPL Rule #4 compliance.
        """
        validation_error = self.validate_pipeline(stages)
        if validation_error:
            logger.error(validation_error)
            return IFFailureArtifact(
                artifact_id=artifact.artifact_id,
                error_message=validation_error,
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + ["pipeline-validation-failed"],
            )

        current_artifact = artifact
        timeout = self.resource_config.timeout_seconds

        try:
            for stage in stages:
                if not isinstance(current_artifact, stage.input_type):
                    return self._create_type_mismatch_failure(current_artifact, stage)

                current_artifact, success = self._execute_bounded_stage(
                    stage, current_artifact, document_id, timeout
                )
                if not success:
                    return current_artifact

            return current_artifact
        finally:
            if self._auto_teardown:
                self._teardown_stages(stages)

    def _create_type_mismatch_failure(
        self, artifact: IFArtifact, stage: IFStage, context: str = "type-mismatch"
    ) -> IFFailureArtifact:
        """
        Create failure artifact for type mismatch.

        Extracted helper for JPL Rule #4 compliance.
        """
        error_msg = (
            f"Type mismatch: Stage {stage.name} expected "
            f"{stage.input_type.__name__}, got {type(artifact).__name__}"
        )
        logger.error(error_msg)
        return IFFailureArtifact(
            artifact_id=artifact.artifact_id,
            error_message=error_msg,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [f"stage-{stage.name}-{context}"],
        )

    def _create_contract_violation_failure(
        self, artifact: IFArtifact, stage: IFStage
    ) -> IFFailureArtifact:
        """
        Create failure artifact for contract violation.

        Extracted helper for JPL Rule #4 compliance.
        """
        error_msg = (
            f"Contract violation: Stage {stage.name} produced "
            f"{type(artifact).__name__}, expected {stage.output_type.__name__}"
        )
        logger.error(error_msg)
        return IFFailureArtifact(
            artifact_id=artifact.artifact_id,
            error_message=error_msg,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [f"stage-{stage.name}-contract-violation"],
        )

    def _create_crash_failure(
        self, artifact: IFArtifact, stage: IFStage, error: Exception
    ) -> IFFailureArtifact:
        """
        Create failure artifact for unexpected crash.

        Extracted helper for JPL Rule #4 compliance.
        """
        error_msg = f"Unexpected error in stage {stage.name}: {str(error)}"
        logger.exception(error_msg)
        return IFFailureArtifact(
            artifact_id=artifact.artifact_id,
            error_message=error_msg,
            stack_trace=traceback.format_exc(),
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [f"stage-{stage.name}-crash"],
        )

    def _save_checkpoint_if_configured(
        self, artifact: IFArtifact, document_id: str, stage_name: str
    ) -> None:
        """
        Save checkpoint if checkpoint manager is configured.

        Extracted helper for JPL Rule #4 compliance.
        """
        if self.checkpoint_manager:
            if not self.checkpoint_manager.save_checkpoint(
                artifact, document_id, stage_name
            ):
                logger.warning(
                    f"Failed to save checkpoint for {document_id} at {stage_name}"
                )

    def _execute_monitored_stage(
        self, stage: IFStage, artifact: IFArtifact, document_id: str
    ) -> Tuple[IFArtifact, bool, float]:
        """
        Execute a single stage with monitoring.

        Extracted helper for JPL Rule #4 compliance.

        Returns:
            Tuple of (result_artifact, success, duration_ms).
        """
        self._call_interceptors_pre_stage(stage.name, artifact, document_id)
        stage_start = time.perf_counter()

        try:
            logger.info(f"Executing Stage: {stage.name}")
            result = stage.execute(artifact)
            duration_ms = (time.perf_counter() - stage_start) * 1000

            if isinstance(result, IFFailureArtifact):
                error = RuntimeError(result.error_message)
                self._call_interceptors_on_error(stage.name, result, document_id, error)
                return result, False, duration_ms

            if not isinstance(result, stage.output_type):
                failure = self._create_contract_violation_failure(result, stage)
                error = TypeError(failure.error_message)
                self._call_interceptors_on_error(stage.name, result, document_id, error)
                return failure, False, duration_ms

            self._call_interceptors_post_stage(
                stage.name, result, document_id, duration_ms
            )
            self._save_checkpoint_if_configured(result, document_id, stage.name)
            return result, True, duration_ms

        except Exception as e:
            duration_ms = (time.perf_counter() - stage_start) * 1000
            self._call_interceptors_on_error(stage.name, artifact, document_id, e)
            failure = self._create_crash_failure(artifact, stage, e)
            return failure, False, duration_ms

    def run_monitored(
        self, artifact: IFArtifact, stages: List[IFStage], document_id: str
    ) -> IFArtifact:
        """
        Execute pipeline with interceptor monitoring.

        Monitoring - Non-Blocking Interceptors.
        Teardown is called in finally block.
        Refactored for JPL Rule #4 compliance.
        """
        pipeline_start = time.perf_counter()
        self._call_interceptors_pipeline_start(document_id, len(stages))

        current_artifact = artifact
        success = True

        try:
            for stage in stages:
                if not isinstance(current_artifact, stage.input_type):
                    success = False  # Set before return for correct finally behavior
                    failure = self._create_type_mismatch_failure(
                        current_artifact, stage
                    )
                    error = ValueError(failure.error_message)
                    self._call_interceptors_on_error(
                        stage.name, current_artifact, document_id, error
                    )
                    return failure

                current_artifact, stage_success, _ = self._execute_monitored_stage(
                    stage, current_artifact, document_id
                )
                if not stage_success:
                    success = False
                    break

            return current_artifact
        finally:
            if self._auto_teardown:
                self._teardown_stages(stages)
            total_duration_ms = (time.perf_counter() - pipeline_start) * 1000
            self._call_interceptors_pipeline_end(
                document_id, success, total_duration_ms
            )

    def _try_processor(
        self, processor: IFProcessor, artifact: IFArtifact
    ) -> Tuple[Optional[IFArtifact], FallbackAttempt]:
        """
        Try a single processor and return attempt record.

        Extracted helper for JPL Rule #4 compliance.

        Returns:
            Tuple of (result_artifact or None, attempt_record).
        """
        start_time = time.perf_counter()
        try:
            logger.info(f"Trying processor: {processor.processor_id}")
            result = processor.process(artifact)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if isinstance(result, IFFailureArtifact):
                logger.warning(
                    f"Processor {processor.processor_id} returned failure: {result.error_message}"
                )
                return None, FallbackAttempt(
                    processor_id=processor.processor_id,
                    success=False,
                    error=result.error_message,
                    duration_ms=duration_ms,
                )

            logger.info(f"Processor {processor.processor_id} succeeded")
            return result, FallbackAttempt(
                processor_id=processor.processor_id,
                success=True,
                error=None,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Processor {processor.processor_id} raised exception: {e}")
            return None, FallbackAttempt(
                processor_id=processor.processor_id,
                success=False,
                error=f"Exception: {str(e)}",
                duration_ms=duration_ms,
            )

    def _create_fallback_failure(
        self, artifact: IFArtifact, attempts: List[FallbackAttempt]
    ) -> IFFailureArtifact:
        """
        Create aggregated failure artifact from fallback attempts.

        Extracted helper for JPL Rule #4 compliance.
        """
        error_messages = [
            f"{a.processor_id}: {a.error}"
            for a in attempts
            if not a.success and a.error
        ]
        aggregated_error = (
            f"All {len(attempts)} fallback attempts failed: "
            + "; ".join(error_messages[:3])
        )
        if len(error_messages) > 3:
            aggregated_error += f" (+{len(error_messages) - 3} more)"

        return IFFailureArtifact(
            artifact_id=artifact.artifact_id,
            error_message=aggregated_error,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + ["fallback-exhausted"],
        )

    def run_with_fallback(
        self,
        artifact: IFArtifact,
        processors: List[IFProcessor],
        fallback_config: Optional[FallbackConfig] = None,
    ) -> FallbackResult:
        """
        Execute artifact processing with fallback recovery.

        Error - Sequential Fallback Recovery.
        Refactored for JPL Rule #4 compliance.
        """
        config = fallback_config or FallbackConfig()
        attempts: List[FallbackAttempt] = []
        processors_to_try = processors[: config.max_attempts]

        for processor in processors_to_try:
            if config.skip_unavailable and not processor.is_available():
                logger.debug(
                    f"Skipping unavailable processor: {processor.processor_id}"
                )
                attempts.append(
                    FallbackAttempt(
                        processor_id=processor.processor_id,
                        success=False,
                        error="Processor unavailable",
                        duration_ms=0.0,
                    )
                )
                continue

            result, attempt = self._try_processor(processor, artifact)
            attempts.append(attempt)

            if attempt.success and result is not None:
                return FallbackResult(
                    success=True,
                    artifact=result,
                    attempts=tuple(attempts),
                    successful_processor=processor.processor_id,
                )

            if not config.continue_on_failure:
                break

        failure = self._create_fallback_failure(artifact, attempts)
        return FallbackResult(
            success=False,
            artifact=failure,
            attempts=tuple(attempts),
            successful_processor=None,
        )
