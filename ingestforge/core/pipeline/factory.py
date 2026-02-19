"""
Dynamic Pipeline Factory for IngestForge (IF).

Dynamic Pipeline Factory.
Assembles processing pipelines from vertical blueprints by resolving
stage definitions to registered IFProcessors.

NASA JPL Power of Ten compliant.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ingestforge.core.pipeline.blueprint import (
    VerticalBlueprint,
    StageConfig,
    BlueprintRegistry,
)
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.registry import IFRegistry

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_RESOLUTION_ATTEMPTS = 3
MAX_PIPELINE_STAGES = 20


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PipelineAssemblyError(Exception):
    """
    Raised when pipeline assembly fails.

    GWT-3: Includes stage index and processor name.
    """

    def __init__(
        self,
        message: str,
        stage_index: Optional[int] = None,
        processor_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """
        Initialize with assembly error details.

        Args:
            message: Error description.
            stage_index: Index of the failing stage (0-based).
            processor_name: Name of the processor that failed.
            reason: Specific reason for failure.
        """
        self.stage_index = stage_index
        self.processor_name = processor_name
        self.reason = reason

        details = []
        if stage_index is not None:
            details.append(f"stage={stage_index}")
        if processor_name:
            details.append(f"processor={processor_name}")
        if reason:
            details.append(f"reason={reason}")

        detail_str = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{detail_str}")


class ProcessorResolutionError(PipelineAssemblyError):
    """Raised when a processor cannot be resolved."""

    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ResolvedStage:
    """
    A resolved pipeline stage with processor instance.

    Rule #9: Complete type hints.
    """

    stage_index: int
    processor: IFProcessor
    config: Dict[str, Any]
    enabled: bool = True
    processor_name: str = ""

    def __post_init__(self) -> None:
        """Set processor_name from processor if not provided."""
        if not self.processor_name:
            self.processor_name = self.processor.processor_id


@dataclass
class AssembledPipeline:
    """
    Result of pipeline assembly from a blueprint.

    GWT-1: Ordered list of configured IFProcessor instances.
    Rule #9: Complete type hints.
    """

    blueprint: VerticalBlueprint
    stages: List[ResolvedStage] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def vertical_id(self) -> str:
        """Get the vertical ID from the blueprint."""
        return self.blueprint.vertical_id

    @property
    def stage_count(self) -> int:
        """Get the number of resolved stages."""
        return len(self.stages)

    @property
    def enabled_stages(self) -> List[ResolvedStage]:
        """Get only enabled stages."""
        return [s for s in self.stages if s.enabled]

    @property
    def processors(self) -> List[IFProcessor]:
        """Get ordered list of processor instances."""
        return [s.processor for s in self.stages if s.enabled]

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        """
        Execute the pipeline on an artifact.

        Args:
            artifact: Input artifact to process.

        Returns:
            Processed artifact after all stages.

        Rule #4: Function < 60 lines.
        """
        current = artifact
        for stage in self.enabled_stages:
            current = stage.processor.process(current)
        return current


# ---------------------------------------------------------------------------
# PipelineFactory
# ---------------------------------------------------------------------------


class PipelineFactory:
    """
    Factory for assembling pipelines from vertical blueprints.

    GWT-1: Assembles pipeline from blueprint.
    GWT-2: Resolves processors from IFRegistry.
    GWT-3: Raises PipelineAssemblyError on failure.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        registry: Optional[IFRegistry] = None,
        blueprint_registry: Optional[BlueprintRegistry] = None,
    ) -> None:
        """
        Initialize the factory.

        Args:
            registry: IFRegistry for processor lookup. Uses singleton if None.
            blueprint_registry: BlueprintRegistry for blueprint lookup. Uses singleton if None.
        """
        self._registry = registry or IFRegistry()
        self._blueprint_registry = blueprint_registry or BlueprintRegistry()
        self._processor_factories: Dict[str, Callable[..., IFProcessor]] = {}

    def register_processor_factory(
        self,
        name: str,
        factory: Callable[..., IFProcessor],
    ) -> None:
        """
        Register a custom processor factory.

        Args:
            name: Processor name to match in blueprints.
            factory: Callable that creates processor instances.

        Rule #4: Function < 60 lines.
        """
        assert name, "Processor name cannot be empty"
        assert factory is not None, "Factory cannot be None"
        self._processor_factories[name] = factory
        logger.debug(f"Registered processor factory: {name}")

    def assemble(self, blueprint: VerticalBlueprint) -> AssembledPipeline:
        """
        Assemble a pipeline from a blueprint.

        Args:
            blueprint: Validated blueprint to assemble.

        Returns:
            AssembledPipeline with resolved processors.

        Raises:
            PipelineAssemblyError: If assembly fails.

        GWT-1: Returns ordered list of IFProcessor instances.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.
        """
        assert blueprint is not None, "Blueprint cannot be None"
        assert (
            len(blueprint.stages) <= MAX_PIPELINE_STAGES
        ), f"Blueprint exceeds maximum {MAX_PIPELINE_STAGES} stages"

        resolved_stages: List[ResolvedStage] = []
        warnings: List[str] = []

        for idx, stage in enumerate(blueprint.stages):
            try:
                resolved = self._resolve_stage(idx, stage)
                resolved_stages.append(resolved)
            except ProcessorResolutionError:
                raise
            except Exception as e:
                raise PipelineAssemblyError(
                    f"Failed to resolve stage {idx}",
                    stage_index=idx,
                    processor_name=stage.processor,
                    reason=str(e),
                ) from e

        # Check for disabled stages
        disabled_count = sum(1 for s in resolved_stages if not s.enabled)
        if disabled_count > 0:
            warnings.append(f"{disabled_count} stage(s) are disabled")

        logger.info(
            f"Assembled pipeline '{blueprint.vertical_id}' with "
            f"{len(resolved_stages)} stages ({disabled_count} disabled)"
        )

        return AssembledPipeline(
            blueprint=blueprint,
            stages=resolved_stages,
            warnings=warnings,
        )

    def assemble_by_id(self, vertical_id: str) -> AssembledPipeline:
        """
        Assemble a pipeline by vertical ID.

        Args:
            vertical_id: ID of the registered blueprint.

        Returns:
            AssembledPipeline with resolved processors.

        Raises:
            PipelineAssemblyError: If blueprint not found or assembly fails.

        Rule #4: Function < 60 lines.
        """
        assert vertical_id, "vertical_id cannot be empty"

        blueprint = self._blueprint_registry.get(vertical_id)
        if blueprint is None:
            raise PipelineAssemblyError(
                f"Blueprint not found: {vertical_id}",
                reason="not_registered",
            )

        return self.assemble(blueprint)

    def _resolve_stage(self, index: int, stage: StageConfig) -> ResolvedStage:
        """
        Resolve a single stage to a processor instance.

        Args:
            index: Stage index in the blueprint.
            stage: Stage configuration to resolve.

        Returns:
            ResolvedStage with processor instance.

        Raises:
            ProcessorResolutionError: If processor cannot be found.

        GWT-2: Resolves from IFRegistry or factory.
        Rule #4: Function < 60 lines.
        """
        processor_name = stage.processor
        processor: Optional[IFProcessor] = None

        # Strategy 1: Check custom factories first
        processor = self._try_custom_factory(processor_name, stage.config)
        if processor is not None:
            return self._create_resolved_stage(index, stage, processor)

        # Strategy 2: Check IFRegistry by processor_id
        processor = self._try_registry_by_id(processor_name)
        if processor is not None:
            return self._create_resolved_stage(index, stage, processor)

        # Strategy 3: Check enricher factories by class name
        processor = self._try_enricher_factory(processor_name, stage.config)
        if processor is not None:
            return self._create_resolved_stage(index, stage, processor)

        # Strategy 4: Check by capability (if processor_name looks like capability)
        processor = self._try_by_capability(processor_name)
        if processor is not None:
            return self._create_resolved_stage(index, stage, processor)

        # Failed to resolve
        raise ProcessorResolutionError(
            f"Cannot resolve processor: {processor_name}",
            stage_index=index,
            processor_name=processor_name,
            reason="not_found",
        )

    def _create_resolved_stage(
        self,
        index: int,
        stage: StageConfig,
        processor: IFProcessor,
    ) -> ResolvedStage:
        """Create a ResolvedStage from components. Rule #4: Helper < 60 lines."""
        return ResolvedStage(
            stage_index=index,
            processor=processor,
            config=stage.config,
            enabled=stage.enabled,
            processor_name=stage.processor,
        )

    def _try_custom_factory(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> Optional[IFProcessor]:
        """Try to create processor from custom factory. Rule #4: Helper < 60 lines."""
        factory = self._processor_factories.get(name)
        if factory is None:
            return None

        try:
            return factory(**config) if config else factory()
        except Exception as e:
            logger.warning(f"Custom factory failed for {name}: {e}")
            return None

    def _try_registry_by_id(self, processor_id: str) -> Optional[IFProcessor]:
        """Try to get processor from registry by ID. Rule #4: Helper < 60 lines."""
        return self._registry._id_map.get(processor_id)

    def _try_enricher_factory(
        self,
        class_name: str,
        config: Dict[str, Any],
    ) -> Optional[IFProcessor]:
        """Try to create processor from enricher factory. Rule #4: Helper < 60 lines."""
        entry = self._registry._enricher_factories.get(class_name)
        if entry is None:
            return None

        try:
            return entry.factory(**config) if config else entry.factory()
        except Exception as e:
            logger.warning(f"Enricher factory failed for {class_name}: {e}")
            return None

    def _try_by_capability(self, capability: str) -> Optional[IFProcessor]:
        """Try to get processor by capability. Rule #4: Helper < 60 lines."""
        # Only try capability lookup for lowercase names without "IF" prefix
        if capability.startswith("IF") or capability[0].isupper():
            return None

        processors = self._registry.get_by_capability(capability)
        if processors:
            return processors[0]  # Return highest priority
        return None


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_pipeline_factory(
    registry: Optional[IFRegistry] = None,
    blueprint_registry: Optional[BlueprintRegistry] = None,
) -> PipelineFactory:
    """
    Create a PipelineFactory instance.

    Args:
        registry: IFRegistry for processor lookup.
        blueprint_registry: BlueprintRegistry for blueprint lookup.

    Returns:
        Configured PipelineFactory instance.
    """
    return PipelineFactory(
        registry=registry,
        blueprint_registry=blueprint_registry,
    )


def assemble_pipeline(blueprint: VerticalBlueprint) -> AssembledPipeline:
    """
    Convenience function to assemble a pipeline from a blueprint.

    Args:
        blueprint: Validated blueprint to assemble.

    Returns:
        AssembledPipeline with resolved processors.

    Raises:
        PipelineAssemblyError: If assembly fails.
    """
    factory = PipelineFactory()
    return factory.assemble(blueprint)


def assemble_pipeline_by_id(vertical_id: str) -> AssembledPipeline:
    """
    Convenience function to assemble a pipeline by vertical ID.

    Args:
        vertical_id: ID of the registered blueprint.

    Returns:
        AssembledPipeline with resolved processors.

    Raises:
        PipelineAssemblyError: If blueprint not found or assembly fails.
    """
    factory = PipelineFactory()
    return factory.assemble_by_id(vertical_id)
