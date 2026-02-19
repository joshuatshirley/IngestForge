"""
Vertical-Aware Ingestion Entry-Point for IngestForge (IF).

Vertical-Aware Ingestion Entry-Point.
Provides a unified entry-point for document ingestion that routes
documents to the appropriate vertical pipeline.

NASA JPL Power of Ten compliant.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.pipeline.blueprint import (
    BlueprintRegistry,
    VerticalBlueprint,
)
from ingestforge.core.pipeline.factory import (
    PipelineFactory,
    AssembledPipeline,
    PipelineAssemblyError,
)
from ingestforge.core.pipeline.interfaces import IFArtifact
from ingestforge.core.pipeline.artifacts import IFFileArtifact

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_INGESTION_RETRIES = 3
MAX_DEFAULT_VERTICALS = 10


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VerticalNotFoundError(Exception):
    """
    Raised when a specified vertical is not found.

    GWT-3: Unknown vertical_id raises error.
    """

    def __init__(self, vertical_id: str, available: Optional[List[str]] = None) -> None:
        """
        Initialize with error details.

        Args:
            vertical_id: The unknown vertical ID.
            available: List of available vertical IDs.
        """
        self.vertical_id = vertical_id
        self.available = available or []

        msg = f"Vertical not found: {vertical_id}"
        if self.available:
            msg += f". Available: {', '.join(self.available[:5])}"
            if len(self.available) > 5:
                msg += f" (+{len(self.available) - 5} more)"
        super().__init__(msg)


class IngestionError(Exception):
    """Raised when ingestion fails."""

    def __init__(
        self,
        message: str,
        vertical_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize with error details.

        Args:
            message: Error description.
            vertical_id: Vertical being used.
            artifact_id: Artifact being processed.
            cause: Original exception.
        """
        self.vertical_id = vertical_id
        self.artifact_id = artifact_id
        self.cause = cause
        super().__init__(message)


class NoDefaultVerticalError(Exception):
    """Raised when no default vertical is configured."""

    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """
    Result of vertical-aware ingestion.

    Contains processed artifact and execution metadata.
    Rule #9: Complete type hints.
    """

    artifact: IFArtifact
    vertical_id: str
    stage_count: int
    duration_ms: float
    success: bool = True
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def stages_executed(self) -> int:
        """Alias for stage_count."""
        return self.stage_count


# ---------------------------------------------------------------------------
# VerticalIngestion
# ---------------------------------------------------------------------------


class VerticalIngestion:
    """
    Unified entry-point for vertical-aware document ingestion.

    GWT-1: Routes documents to appropriate vertical pipeline.
    GWT-2: Supports default vertical fallback.
    GWT-3: Raises VerticalNotFoundError for unknown verticals.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        blueprint_registry: Optional[BlueprintRegistry] = None,
        pipeline_factory: Optional[PipelineFactory] = None,
        default_vertical: Optional[str] = None,
    ) -> None:
        """
        Initialize the ingestion entry-point.

        Args:
            blueprint_registry: Registry for vertical blueprints.
            pipeline_factory: Factory for pipeline assembly.
            default_vertical: Default vertical ID when none specified.
        """
        self._blueprint_registry = blueprint_registry or BlueprintRegistry()
        self._pipeline_factory = pipeline_factory or PipelineFactory(
            blueprint_registry=self._blueprint_registry
        )
        self._default_vertical = default_vertical
        self._pipeline_cache: Dict[str, AssembledPipeline] = {}

    @property
    def default_vertical(self) -> Optional[str]:
        """Get the default vertical ID."""
        return self._default_vertical

    def set_default_vertical(self, vertical_id: str) -> None:
        """
        Set the default vertical for ingestion.

        Args:
            vertical_id: Vertical ID to use as default.

        Raises:
            VerticalNotFoundError: If vertical not registered.

        Rule #4: Function < 60 lines.
        """
        assert vertical_id, "vertical_id cannot be empty"

        # Validate vertical exists
        blueprint = self._blueprint_registry.get(vertical_id)
        if blueprint is None:
            raise VerticalNotFoundError(
                vertical_id,
                available=self._blueprint_registry.list_verticals(),
            )

        self._default_vertical = vertical_id
        logger.info(f"Set default vertical: {vertical_id}")

    def list_verticals(self) -> List[str]:
        """
        List all available vertical IDs.

        Returns:
            List of registered vertical IDs.
        """
        return self._blueprint_registry.list_verticals()

    def get_blueprint(self, vertical_id: str) -> Optional[VerticalBlueprint]:
        """
        Get a blueprint by vertical ID.

        Args:
            vertical_id: The vertical identifier.

        Returns:
            Blueprint if found, None otherwise.
        """
        return self._blueprint_registry.get(vertical_id)

    def ingest(
        self,
        artifact: IFArtifact,
        vertical_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> IngestionResult:
        """
        Ingest an artifact using a vertical pipeline.

        Args:
            artifact: The artifact to process.
            vertical_id: Vertical to use. Falls back to default if None.
            use_cache: Whether to cache assembled pipelines.

        Returns:
            IngestionResult with processed artifact and metadata.

        Raises:
            VerticalNotFoundError: If vertical not found.
            NoDefaultVerticalError: If no vertical specified and no default.
            IngestionError: If processing fails.

        GWT-1: Explicit vertical selection.
        GWT-2: Default vertical fallback.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.
        """
        assert artifact is not None, "Artifact cannot be None"

        # Resolve vertical ID
        effective_vertical = self._resolve_vertical_id(vertical_id)

        # Get or assemble pipeline
        pipeline = self._get_pipeline(effective_vertical, use_cache)

        # Execute pipeline
        start_time = time.perf_counter()
        try:
            result_artifact = pipeline.execute(artifact)
            duration_ms = (time.perf_counter() - start_time) * 1000

            return IngestionResult(
                artifact=result_artifact,
                vertical_id=effective_vertical,
                stage_count=len(pipeline.enabled_stages),
                duration_ms=duration_ms,
                success=True,
                warnings=pipeline.warnings,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Ingestion failed for {artifact.artifact_id}: {e}")
            raise IngestionError(
                f"Ingestion failed: {e}",
                vertical_id=effective_vertical,
                artifact_id=artifact.artifact_id,
                cause=e,
            ) from e

    def ingest_file(
        self,
        file_path: Path,
        vertical_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> IngestionResult:
        """
        Ingest a file using a vertical pipeline.

        Args:
            file_path: Path to the file to process.
            vertical_id: Vertical to use. Falls back to default if None.
            artifact_id: Optional artifact ID. Auto-generated if None.
            use_cache: Whether to cache assembled pipelines.

        Returns:
            IngestionResult with processed artifact and metadata.

        Raises:
            VerticalNotFoundError: If vertical not found.
            FileNotFoundError: If file doesn't exist.
            IngestionError: If processing fails.

        Rule #4: Function < 60 lines.
        """
        assert file_path is not None, "file_path cannot be None"

        if not file_path.exists():
            # SEC-002: Sanitize path disclosure
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError("File not found: [REDACTED]")

        # Create file artifact
        if artifact_id is None:
            artifact_id = f"file:{file_path.name}"

        artifact = IFFileArtifact(
            artifact_id=artifact_id,
            file_path=file_path,
        )

        return self.ingest(artifact, vertical_id=vertical_id, use_cache=use_cache)

    def _resolve_vertical_id(self, vertical_id: Optional[str]) -> str:
        """
        Resolve the vertical ID to use.

        Args:
            vertical_id: Explicit vertical ID or None.

        Returns:
            Resolved vertical ID.

        Raises:
            NoDefaultVerticalError: If no vertical and no default.
            VerticalNotFoundError: If vertical not found.

        Rule #4: Helper < 60 lines.
        """
        # Use explicit vertical if provided
        if vertical_id is not None:
            if self._blueprint_registry.get(vertical_id) is None:
                raise VerticalNotFoundError(
                    vertical_id,
                    available=self._blueprint_registry.list_verticals(),
                )
            return vertical_id

        # Fall back to default
        if self._default_vertical is None:
            raise NoDefaultVerticalError(
                "No vertical_id specified and no default configured. "
                f"Available verticals: {self.list_verticals()}"
            )

        return self._default_vertical

    def _get_pipeline(
        self,
        vertical_id: str,
        use_cache: bool,
    ) -> AssembledPipeline:
        """
        Get or assemble a pipeline for a vertical.

        Args:
            vertical_id: Vertical ID to get pipeline for.
            use_cache: Whether to use cached pipeline.

        Returns:
            AssembledPipeline ready for execution.

        Rule #4: Helper < 60 lines.
        """
        # Check cache
        if use_cache and vertical_id in self._pipeline_cache:
            return self._pipeline_cache[vertical_id]

        # Assemble pipeline
        try:
            pipeline = self._pipeline_factory.assemble_by_id(vertical_id)
        except PipelineAssemblyError as e:
            raise IngestionError(
                f"Failed to assemble pipeline for {vertical_id}: {e}",
                vertical_id=vertical_id,
                cause=e,
            ) from e

        # Cache if requested
        if use_cache:
            self._pipeline_cache[vertical_id] = pipeline

        return pipeline

    def clear_cache(self) -> None:
        """Clear the pipeline cache."""
        self._pipeline_cache.clear()
        logger.debug("Cleared pipeline cache")


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_vertical_ingestion(
    default_vertical: Optional[str] = None,
    blueprint_registry: Optional[BlueprintRegistry] = None,
) -> VerticalIngestion:
    """
    Create a VerticalIngestion instance.

    Args:
        default_vertical: Default vertical ID.
        blueprint_registry: Blueprint registry to use.

    Returns:
        Configured VerticalIngestion instance.
    """
    return VerticalIngestion(
        blueprint_registry=blueprint_registry,
        default_vertical=default_vertical,
    )


def ingest_with_vertical(
    artifact: IFArtifact,
    vertical_id: str,
) -> IngestionResult:
    """
    Convenience function to ingest with a specific vertical.

    Args:
        artifact: Artifact to process.
        vertical_id: Vertical to use.

    Returns:
        IngestionResult with processed artifact.

    Raises:
        VerticalNotFoundError: If vertical not found.
        IngestionError: If processing fails.
    """
    ingestion = VerticalIngestion()
    return ingestion.ingest(artifact, vertical_id=vertical_id)
