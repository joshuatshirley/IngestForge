"""
Document Processing Pipeline for IngestForge.

This module orchestrates the full document ingestion workflow:
1. Split documents (PDF → chapters)
2. Extract text (PDF/EPUB → markdown)
3. Chunk content (semantic splitting)
4. Enrich chunks (embeddings, entities, questions)
5. Index chunks (storage backend)

Public API
----------
All pipeline classes are re-exported here for backward compatibility:

    from ingestforge.core.pipeline import Pipeline, PipelineResult

Architecture
------------
Pipeline is organized into focused modules:

    pipeline/
    ├── result.py      # PipelineResult dataclass
    ├── pipeline.py    # Main Pipeline class
    ├── stages.py      # PipelineStagesMixin (core processing stages)
    ├── splitters.py   # PipelineSplittersMixin (document splitting)
    ├── streaming.py   # PipelineStreamingMixin (streaming generators)
    └── utils.py       # PipelineUtilsMixin (utilities and helpers)

Usage Example
-------------
    from ingestforge.core.pipeline import Pipeline
    from ingestforge.core.config_loaders import load_config

    config = load_config()
    pipeline = Pipeline(config)
    result = pipeline.process_file("document.pdf")
"""

__all__ = [
    "Pipeline",
    "PipelineResult",
    "SchemaGenerator",
    "SchemaDefinitionError",
    "generate_schema_from_yaml",
    "IFConfigFactory",
    # IFEnrichmentStage
    "IFEnrichmentStage",
    # Blueprint Parser & Validator
    "VerticalBlueprint",
    "StageConfig",
    "BlueprintParser",
    "BlueprintRegistry",
    "BlueprintValidationError",
    "BlueprintLoadError",
    "load_blueprint",
    "load_blueprint_from_dict",
    "get_blueprint_registry",
    # Dynamic Pipeline Factory
    "PipelineFactory",
    "AssembledPipeline",
    "ResolvedStage",
    "PipelineAssemblyError",
    "ProcessorResolutionError",
    "create_pipeline_factory",
    "assemble_pipeline",
    "assemble_pipeline_by_id",
    # Vertical-Aware Ingestion
    "VerticalIngestion",
    "IngestionResult",
    "VerticalNotFoundError",
    "NoDefaultVerticalError",
    "IngestionError",
    "create_vertical_ingestion",
    "ingest_with_vertical",
    # Single-Object Aggregation
    "IFAggregator",
    "AggregationStrategy",
    "AggregationResult",
    "aggregate_artifacts",
    "aggregate_to_json",
    # Registry-Driven Discovery
    "IFRegistry",
    "register_processor",
    "register_enricher",
    # Incremental Runner
    "IncrementalRunner",
    "HashManifest",
    "FileHashRecord",
    "IncrementalCheckResult",
    "IncrementalRunReport",
    "create_incremental_runner",
    "filter_unchanged_files",
    # Incremental Foundry Runner
    "IncrementalFoundryRunner",
    "IncrementalFoundryConfig",
    "IncrementalFoundryReport",
    "IncrementalSkipResult",
    "HashRegistry",
    "create_incremental_foundry_runner",
    "run_incremental_batch",
]


def __getattr__(name):
    """Lazy imports to prevent circular dependencies at module load time."""
    if name == "Pipeline":
        from ingestforge.core.pipeline.pipeline import Pipeline

        return Pipeline
    elif name == "PipelineResult":
        from ingestforge.core.pipeline.result import PipelineResult

        return PipelineResult
    elif name == "SchemaGenerator":
        from ingestforge.core.pipeline.schema_generator import SchemaGenerator

        return SchemaGenerator
    elif name == "SchemaDefinitionError":
        from ingestforge.core.pipeline.schema_generator import SchemaDefinitionError

        return SchemaDefinitionError
    elif name == "generate_schema_from_yaml":
        from ingestforge.core.pipeline.schema_generator import generate_schema_from_yaml

        return generate_schema_from_yaml
    elif name == "IFConfigFactory":
        from ingestforge.core.pipeline.config_factory import IFConfigFactory

        return IFConfigFactory
    elif name == "IFEnrichmentStage":
        from ingestforge.core.pipeline.enrichment_stage import IFEnrichmentStage

        return IFEnrichmentStage
    # Blueprint Parser & Validator
    elif name == "VerticalBlueprint":
        from ingestforge.core.pipeline.blueprint import VerticalBlueprint

        return VerticalBlueprint
    elif name == "StageConfig":
        from ingestforge.core.pipeline.blueprint import StageConfig

        return StageConfig
    elif name == "BlueprintParser":
        from ingestforge.core.pipeline.blueprint import BlueprintParser

        return BlueprintParser
    elif name == "BlueprintRegistry":
        from ingestforge.core.pipeline.blueprint import BlueprintRegistry

        return BlueprintRegistry
    elif name == "BlueprintValidationError":
        from ingestforge.core.pipeline.blueprint import BlueprintValidationError

        return BlueprintValidationError
    elif name == "BlueprintLoadError":
        from ingestforge.core.pipeline.blueprint import BlueprintLoadError

        return BlueprintLoadError
    elif name == "load_blueprint":
        from ingestforge.core.pipeline.blueprint import load_blueprint

        return load_blueprint
    elif name == "load_blueprint_from_dict":
        from ingestforge.core.pipeline.blueprint import load_blueprint_from_dict

        return load_blueprint_from_dict
    elif name == "get_blueprint_registry":
        from ingestforge.core.pipeline.blueprint import get_blueprint_registry

        return get_blueprint_registry
    # Dynamic Pipeline Factory
    elif name == "PipelineFactory":
        from ingestforge.core.pipeline.factory import PipelineFactory

        return PipelineFactory
    elif name == "AssembledPipeline":
        from ingestforge.core.pipeline.factory import AssembledPipeline

        return AssembledPipeline
    elif name == "ResolvedStage":
        from ingestforge.core.pipeline.factory import ResolvedStage

        return ResolvedStage
    elif name == "PipelineAssemblyError":
        from ingestforge.core.pipeline.factory import PipelineAssemblyError

        return PipelineAssemblyError
    elif name == "ProcessorResolutionError":
        from ingestforge.core.pipeline.factory import ProcessorResolutionError

        return ProcessorResolutionError
    elif name == "create_pipeline_factory":
        from ingestforge.core.pipeline.factory import create_pipeline_factory

        return create_pipeline_factory
    elif name == "assemble_pipeline":
        from ingestforge.core.pipeline.factory import assemble_pipeline

        return assemble_pipeline
    elif name == "assemble_pipeline_by_id":
        from ingestforge.core.pipeline.factory import assemble_pipeline_by_id

        return assemble_pipeline_by_id
    # Vertical-Aware Ingestion
    elif name == "VerticalIngestion":
        from ingestforge.core.pipeline.ingestion import VerticalIngestion

        return VerticalIngestion
    elif name == "IngestionResult":
        from ingestforge.core.pipeline.ingestion import IngestionResult

        return IngestionResult
    elif name == "VerticalNotFoundError":
        from ingestforge.core.pipeline.ingestion import VerticalNotFoundError

        return VerticalNotFoundError
    elif name == "NoDefaultVerticalError":
        from ingestforge.core.pipeline.ingestion import NoDefaultVerticalError

        return NoDefaultVerticalError
    elif name == "IngestionError":
        from ingestforge.core.pipeline.ingestion import IngestionError

        return IngestionError
    elif name == "create_vertical_ingestion":
        from ingestforge.core.pipeline.ingestion import create_vertical_ingestion

        return create_vertical_ingestion
    elif name == "ingest_with_vertical":
        from ingestforge.core.pipeline.ingestion import ingest_with_vertical

        return ingest_with_vertical
    # Single-Object Aggregation
    elif name == "IFAggregator":
        from ingestforge.core.pipeline.aggregator import IFAggregator

        return IFAggregator
    elif name == "AggregationStrategy":
        from ingestforge.core.pipeline.aggregator import AggregationStrategy

        return AggregationStrategy
    elif name == "AggregationResult":
        from ingestforge.core.pipeline.aggregator import AggregationResult

        return AggregationResult
    elif name == "aggregate_artifacts":
        from ingestforge.core.pipeline.aggregator import aggregate_artifacts

        return aggregate_artifacts
    elif name == "aggregate_to_json":
        from ingestforge.core.pipeline.aggregator import aggregate_to_json

        return aggregate_to_json
    # Registry-Driven Discovery
    elif name == "IFRegistry":
        from ingestforge.core.pipeline.registry import IFRegistry

        return IFRegistry
    elif name == "register_processor":
        from ingestforge.core.pipeline.registry import register_processor

        return register_processor
    elif name == "register_enricher":
        from ingestforge.core.pipeline.registry import register_enricher

        return register_enricher
    # Incremental Runner
    elif name == "IncrementalRunner":
        from ingestforge.core.pipeline.incremental import IncrementalRunner

        return IncrementalRunner
    elif name == "HashManifest":
        from ingestforge.core.pipeline.incremental import HashManifest

        return HashManifest
    elif name == "FileHashRecord":
        from ingestforge.core.pipeline.incremental import FileHashRecord

        return FileHashRecord
    elif name == "IncrementalCheckResult":
        from ingestforge.core.pipeline.incremental import IncrementalCheckResult

        return IncrementalCheckResult
    elif name == "IncrementalRunReport":
        from ingestforge.core.pipeline.incremental import IncrementalRunReport

        return IncrementalRunReport
    elif name == "create_incremental_runner":
        from ingestforge.core.pipeline.incremental import create_incremental_runner

        return create_incremental_runner
    elif name == "filter_unchanged_files":
        from ingestforge.core.pipeline.incremental import filter_unchanged_files

        return filter_unchanged_files
    # Incremental Foundry Runner
    elif name == "IncrementalFoundryRunner":
        from ingestforge.core.pipeline.incremental_foundry import (
            IncrementalFoundryRunner,
        )

        return IncrementalFoundryRunner
    elif name == "IncrementalFoundryConfig":
        from ingestforge.core.pipeline.incremental_foundry import (
            IncrementalFoundryConfig,
        )

        return IncrementalFoundryConfig
    elif name == "IncrementalFoundryReport":
        from ingestforge.core.pipeline.incremental_foundry import (
            IncrementalFoundryReport,
        )

        return IncrementalFoundryReport
    elif name == "IncrementalSkipResult":
        from ingestforge.core.pipeline.incremental_foundry import IncrementalSkipResult

        return IncrementalSkipResult
    elif name == "HashRegistry":
        from ingestforge.core.pipeline.incremental_foundry import HashRegistry

        return HashRegistry
    elif name == "create_incremental_foundry_runner":
        from ingestforge.core.pipeline.incremental_foundry import (
            create_incremental_foundry_runner,
        )

        return create_incremental_foundry_runner
    elif name == "run_incremental_batch":
        from ingestforge.core.pipeline.incremental_foundry import run_incremental_batch

        return run_incremental_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
