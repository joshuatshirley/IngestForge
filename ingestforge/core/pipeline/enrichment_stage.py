"""
IFEnrichmentStage: Orchestrates IFProcessor-based enrichers.

Replace EnrichmentPipeline with IFEnrichmentStage.
NASA JPL Power of Ten compliant.
"""

import logging
from typing import List, Type

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact, IFStage
from ingestforge.core.errors import SafeErrorMessage
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFFailureArtifact

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_PROCESSORS_PER_STAGE = 32
MAX_CONSECUTIVE_FAILURES = 5
MAX_BATCH_SIZE = 1000  # Maximum artifacts per batch (TASK-002)


class IFEnrichmentStage(IFStage):
    """
    Orchestrates a sequence of IFProcessor enrichers.

    Implements IFStage interface.
    Chains `process()` calls on each enricher in sequence.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        processors: List[IFProcessor],
        skip_failures: bool = True,
        stage_name: str = "enrichment",
    ) -> None:
        """
        Initialize the enrichment stage.

        Args:
            processors: List of IFProcessor instances to apply in sequence.
            skip_failures: If True, continue on IFFailureArtifact. If False, abort.
            stage_name: Name for this stage instance.

        Raises:
            ValueError: If processors list exceeds MAX_PROCESSORS_PER_STAGE.
        """
        # JPL Rule #2: Enforce upper bound
        if len(processors) > MAX_PROCESSORS_PER_STAGE:
            raise ValueError(
                f"Too many processors: {len(processors)} > {MAX_PROCESSORS_PER_STAGE}"
            )

        self._processors = list(processors)  # Defensive copy
        self._skip_failures = skip_failures
        self._stage_name = stage_name
        self._version = "1.0.0"

    # -------------------------------------------------------------------------
    # IFStage Interface Implementation
    # -------------------------------------------------------------------------

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """
        Execute the enrichment stage by chaining all processors.

        Implements IFStage.execute().
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (typically IFChunkArtifact).

        Returns:
            Enriched artifact or IFFailureArtifact if aborted.
        """
        result = artifact
        consecutive_failures = 0

        for processor in self._processors:
            # Skip unavailable processors
            if not processor.is_available():
                logger.debug(
                    f"Processor {processor.processor_id} not available, skipping"
                )
                continue

            try:
                result = processor.process(result)

                # Handle failure artifacts
                if isinstance(result, IFFailureArtifact):
                    consecutive_failures += 1
                    logger.warning(
                        f"Processor {processor.processor_id} returned failure: "
                        f"{result.error_message}"
                    )

                    # Abort if configured to do so
                    if not self._skip_failures:
                        return result

                    # JPL Rule #2: Bound consecutive failures
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error("Max consecutive failures reached, aborting")
                        return result
                else:
                    consecutive_failures = 0  # Reset on success

            except Exception as e:
                # Catch unexpected exceptions and convert to failure artifact
                logger.exception(
                    f"Processor {processor.processor_id} raised exception: {e}"
                )
                consecutive_failures += 1

                if not self._skip_failures:
                    return IFFailureArtifact(
                        artifact_id=f"{artifact.artifact_id}-stage-failure",
                        # SEC-002: Sanitize error message
                        error_message=SafeErrorMessage.sanitize(
                            e, "processor_{processor.processor_id}_failed", logger
                        ),
                        failed_processor_id=processor.processor_id,
                        parent_id=artifact.artifact_id,
                        root_artifact_id=artifact.effective_root_id,
                        lineage_depth=artifact.lineage_depth + 1,
                        provenance=artifact.provenance + [self._stage_name],
                    )

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    return IFFailureArtifact(
                        artifact_id=f"{artifact.artifact_id}-max-failures",
                        error_message="Max consecutive processor failures reached",
                        failed_processor_id=processor.processor_id,
                        parent_id=artifact.artifact_id,
                        root_artifact_id=artifact.effective_root_id,
                        lineage_depth=artifact.lineage_depth + 1,
                        provenance=artifact.provenance + [self._stage_name],
                    )

        return result

    def execute_batch(self, artifacts: List[IFArtifact]) -> List[IFArtifact]:
        """
        Execute enrichment on a batch of artifacts.

        TASK-002: Adds batch processing detection and execution.
        Uses processor.process_batch() when available for 5-10x speedup.

        Rule #2: Bounded batch size.
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifacts: List of input artifacts (typically IFChunkArtifacts).

        Returns:
            List of enriched artifacts (same order as input).
        """
        # JPL Rule #2: Enforce batch size limit
        if len(artifacts) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(artifacts)} exceeds MAX_BATCH_SIZE {MAX_BATCH_SIZE}"
            )

        # Empty batch early return
        if not artifacts:
            return []

        results = list(artifacts)  # Defensive copy

        # Process through each processor in sequence
        for processor in self._processors:
            # Skip unavailable processors
            if not processor.is_available():
                logger.debug(
                    f"Processor {processor.processor_id} not available, skipping"
                )
                continue

            try:
                # TASK-002: Detect and use batch processing when available
                if hasattr(processor, "process_batch"):
                    logger.debug(f"Using batch processing for {processor.processor_id}")
                    results = processor.process_batch(results)
                else:
                    # Fallback to sequential processing
                    logger.debug(
                        f"Using sequential processing for {processor.processor_id}"
                    )
                    results = [processor.process(artifact) for artifact in results]

                # Rule #7: Verify return value
                if not isinstance(results, list):
                    raise ValueError(
                        f"Processor {processor.processor_id} did not return a list"
                    )

            except Exception as e:
                logger.exception(
                    f"Batch processing failed for {processor.processor_id}: {e}"
                )
                # On batch failure, fall back to sequential with error handling
                results = [self.execute(artifact) for artifact in results]

        return results

    @property
    def name(self) -> str:
        """Name of this stage."""
        return self._stage_name

    @property
    def input_type(self) -> Type[IFArtifact]:
        """Expected input artifact type."""
        return IFChunkArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        """Produced output artifact type."""
        return IFChunkArtifact

    # -------------------------------------------------------------------------
    # Additional Properties
    # -------------------------------------------------------------------------

    @property
    def version(self) -> str:
        """SemVer version of this stage."""
        return self._version

    @property
    def processors(self) -> List[IFProcessor]:
        """List of processors in this stage (read-only copy)."""
        return list(self._processors)

    @property
    def processor_count(self) -> int:
        """Number of processors in this stage."""
        return len(self._processors)

    @property
    def available_processors(self) -> List[IFProcessor]:
        """List of currently available processors."""
        return [p for p in self._processors if p.is_available()]

    @property
    def skip_failures(self) -> bool:
        """Whether this stage skips failure artifacts."""
        return self._skip_failures

    # -------------------------------------------------------------------------
    # Teardown Support
    # -------------------------------------------------------------------------

    def teardown(self) -> bool:
        """
        Teardown all processors in this stage.

        Resource cleanup support.
        Rule #7: Check return values.

        Returns:
            True if all teardowns successful, False if any failed.
        """
        all_success = True

        for processor in self._processors:
            try:
                if not processor.teardown():
                    logger.warning(
                        f"Processor {processor.processor_id} teardown returned False"
                    )
                    all_success = False
            except Exception as e:
                logger.warning(
                    f"Processor {processor.processor_id} teardown failed: {e}"
                )
                all_success = False

        return all_success

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_processor(self, processor_id: str) -> IFProcessor | None:
        """
        Get a processor by ID.

        Args:
            processor_id: The processor ID to find.

        Returns:
            The processor if found, None otherwise.
        """
        for processor in self._processors:
            if processor.processor_id == processor_id:
                return processor
        return None

    def get_capabilities(self) -> List[str]:
        """
        Get aggregated capabilities from all processors.

        Returns:
            Deduplicated list of capabilities.
        """
        capabilities = set()
        for processor in self._processors:
            capabilities.update(processor.capabilities)
        return sorted(capabilities)

    def get_total_memory_mb(self) -> int:
        """
        Get total estimated memory for all processors.

        Returns:
            Sum of memory_mb from all processors.
        """
        return sum(p.memory_mb for p in self._processors)

    # -------------------------------------------------------------------------
    # BUG001: Backward Compatibility Methods
    # -------------------------------------------------------------------------

    def enrich_batch(self, chunks: List, **kwargs) -> List:
        """
        Enrich a batch of chunks (backward compatibility).

        BUG001: Provides `enrich_batch` interface for legacy pipeline code.
        TASK-002: Now uses execute_batch() for performance when possible.
        Converts ChunkRecords to IFChunkArtifacts, processes, and converts back.

        Rule #2: Bounded batch size (via execute_batch).
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            chunks: List of ChunkRecord objects.
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            List of enriched ChunkRecord objects.
        """
        # JPL Rule #2: Enforce batch size limit
        if len(chunks) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(chunks)} exceeds MAX_BATCH_SIZE {MAX_BATCH_SIZE}"
            )

        # Convert all chunks to artifacts
        artifacts: List[IFArtifact] = []
        for chunk in chunks:
            if isinstance(chunk, IFChunkArtifact):
                artifacts.append(chunk)
            else:
                artifacts.append(IFChunkArtifact.from_chunk_record(chunk))

        # TASK-002: Use batch processing
        results = self.execute_batch(artifacts)

        # Convert results back to ChunkRecords
        enriched: List = []
        for i, result in enumerate(results):
            if isinstance(result, IFChunkArtifact):
                enriched.append(result.to_chunk_record())
            elif isinstance(result, IFFailureArtifact):
                # On failure, return original chunk with error metadata
                logger.warning(f"Enrichment failed: {result.error_message}")
                original_chunk = chunks[i]
                if hasattr(original_chunk, "__dict__"):
                    original_chunk.enrichment_error = result.error_message
                enriched.append(original_chunk)
            else:
                # Unknown result type, return original
                enriched.append(chunks[i])

        return enriched

    def enrich_chunk(self, chunk) -> any:
        """
        Enrich a single chunk (backward compatibility).

        BUG001: Provides `enrich_chunk` interface for legacy pipeline code.

        Args:
            chunk: ChunkRecord or IFChunkArtifact.

        Returns:
            Enriched chunk.
        """
        result = self.enrich_batch([chunk])
        return result[0] if result else chunk
