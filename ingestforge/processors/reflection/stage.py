"""
Reflection Stage for IngestForge Pipeline.

Agentic Reflection Loop - Pipeline stage integrating reflection processor.
Follows NASA JPL Power of Ten rules.
"""

from typing import Type

from ingestforge.core.errors import SafeErrorMessage
from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFArtifact, IFStage
from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFFailureArtifact
from ingestforge.processors.reflection.reflection_processor import (
    IFReflectionProcessor,
    IFReflectionArtifact,
    MAX_REFLECTION_PASSES,
    CONFIDENCE_THRESHOLD,
)

logger = get_logger(__name__)


class ReflectionStage(IFStage):
    """
    Pipeline stage that executes the reflection/critic pass.

    Agentic Reflection Loop.

    This stage:
    1. Verifies LLM extraction output against source text
    2. Identifies contradictions using entailment scoring
    3. Triggers re-extraction if confidence < threshold
    4. Stores reasoning in artifact provenance

    JPL Rule #2: Max 2 reflection passes.
    JPL Rule #4: Methods < 60 lines.
    JPL Rule #9: Complete type hints.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_passes: int = MAX_REFLECTION_PASSES,
        processor: IFReflectionProcessor = None,
    ) -> None:
        """
        Initialize the reflection stage.

        Args:
            confidence_threshold: Threshold for re-extraction (default 0.7).
            max_passes: Maximum reflection passes (default 2, capped).
            processor: Optional pre-configured processor.
        """
        self._threshold = confidence_threshold
        self._max_passes = min(max_passes, MAX_REFLECTION_PASSES)
        self._processor = processor or IFReflectionProcessor(
            confidence_threshold=self._threshold,
            max_passes=self._max_passes,
        )

    @property
    def name(self) -> str:
        """Name of the stage."""
        return "reflection"

    @property
    def input_type(self) -> Type[IFArtifact]:
        """Expected input artifact type."""
        return IFTextArtifact

    @property
    def output_type(self) -> Type[IFArtifact]:
        """Produced output artifact type."""
        return IFReflectionArtifact

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """
        Execute reflection analysis on artifact.

        Identifies contradictions and decides on re-extraction.
        Rule #4: < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact with extracted content.

        Returns:
            IFReflectionArtifact with analysis results, or IFFailureArtifact on error.
        """
        assert artifact is not None, "artifact cannot be None"

        logger.info(f"Reflection stage processing artifact: {artifact.artifact_id}")

        try:
            # Check processor availability
            if not self._processor.is_available():
                logger.warning("Reflection processor dependencies not available")
                return self._create_skip_result(artifact, "Dependencies not available")

            # Execute reflection
            result = self._processor.process(artifact)

            # Log outcome
            if isinstance(result, IFReflectionArtifact):
                logger.info(
                    f"Reflection complete: pass={result.reflection_pass}, "
                    f"confidence={result.confidence_score:.2%}, "
                    f"contradictions={result.contradictions_found}, "
                    f"reextract={result.should_reextract}"
                )

            return result

        except Exception as e:
            logger.error(f"Reflection stage failed: {e}")
            return IFFailureArtifact(
                artifact_id=artifact.artifact_id,
                # SEC-002: Sanitize error message
                error_message=SafeErrorMessage.sanitize(e, "reflection_failed", logger),
                provenance=list(artifact.provenance) + [self._processor.processor_id],
                parent_id=artifact.artifact_id,
            )

    def _create_skip_result(
        self, artifact: IFArtifact, reason: str
    ) -> IFReflectionArtifact:
        """Create a skip result when reflection cannot proceed."""
        content = ""
        if isinstance(artifact, IFTextArtifact):
            content = artifact.content

        return IFReflectionArtifact(
            artifact_id=artifact.artifact_id + "-reflection-skip",
            content=content,
            metadata={**artifact.metadata, "reflection_skipped": True},
            provenance=list(artifact.provenance) + [self._processor.processor_id],
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id
            if hasattr(artifact, "effective_root_id")
            else artifact.artifact_id,
            lineage_depth=artifact.lineage_depth + 1
            if hasattr(artifact, "lineage_depth")
            else 1,
            reflection_pass=0,
            confidence_score=1.0,
            contradictions_found=0,
            should_reextract=False,
            reflection_reasoning=f"Skipped: {reason}",
        )
