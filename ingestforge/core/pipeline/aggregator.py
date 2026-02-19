"""
Single-Object Aggregator for IngestForge (IF).

Combines multiple IFArtifacts into a single structured output object
that conforms to a target schema.

Single-Object Aggregation.
Follows NASA JPL Power of Ten rules.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Set

from pydantic import BaseModel, ValidationError

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFChunkArtifact

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_AGGREGATION_CHUNKS = 1000
MAX_FIELDS = 64
MAX_FIELD_VALUE_LENGTH = 100000
MAX_LIST_ITEMS = 10000


class AggregationStrategy(Enum):
    """
    Strategy for merging field values from multiple artifacts.

    Field merging with conflict resolution.
    Rule #9: Complete type hints.
    """

    FIRST_WINS = "first_wins"  # Keep first non-null value
    LAST_WINS = "last_wins"  # Keep last non-null value
    ACCUMULATE = "accumulate"  # Collect all values into list
    MERGE_UNIQUE = "merge_unique"  # Collect unique values only


@dataclass
class AggregationResult:
    """
    Result of an aggregation operation.

    Result container with validation status.
    Rule #9: Complete type hints.
    """

    data: Dict[str, Any]
    schema_valid: bool
    missing_fields: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    source_artifact_ids: List[str] = field(default_factory=list)
    field_sources: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all required fields are present."""
        return len(self.missing_fields) == 0

    @property
    def artifact_count(self) -> int:
        """Return number of source artifacts."""
        return len(self.source_artifact_ids)


class IFAggregator(IFProcessor):
    """
    Aggregates multiple artifacts into a single structured object.

    GWT-1: Multi-chunk to single object.
    GWT-2: Field extraction across chunks.
    GWT-3: List field accumulation.
    GWT-4: Validation against schema.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        default_strategy: AggregationStrategy = AggregationStrategy.FIRST_WINS,
        field_strategies: Optional[Dict[str, AggregationStrategy]] = None,
        deduplicate_lists: bool = True,
    ) -> None:
        """
        Initialize the aggregator.

        Args:
            default_strategy: Default merge strategy for scalar fields.
            field_strategies: Per-field strategy overrides.
            deduplicate_lists: Whether to deduplicate list field values.
        """
        self._default_strategy = default_strategy
        self._field_strategies = field_strategies or {}
        self._deduplicate_lists = deduplicate_lists
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Return processor identifier."""
        return "aggregator"

    @property
    def version(self) -> str:
        """Return processor version."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Return processor capabilities."""
        return ["aggregate", "merge", "combine"]

    @property
    def memory_mb(self) -> int:
        """Return estimated memory usage in MB."""
        return 100  # Aggregation can hold multiple artifacts

    def is_available(self) -> bool:
        """Check if processor is available."""
        return True

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process a single artifact (passthrough for IFProcessor interface).

        For actual aggregation, use aggregate() with multiple artifacts.

        Args:
            artifact: Input artifact.

        Returns:
            Same artifact (passthrough).
        """
        return artifact

    def aggregate(
        self,
        artifacts: List[IFArtifact],
        schema: Optional[Type[BaseModel]] = None,
        schema_def: Optional[Dict[str, Any]] = None,
    ) -> AggregationResult:
        """
        Aggregate multiple artifacts into a single structured object.

        Args:
            artifacts: List of artifacts to aggregate.
            schema: Pydantic model for validation (optional).
            schema_def: Schema definition dict for field extraction (optional).

        Returns:
            AggregationResult with merged data and validation status.

        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.
        """
        assert artifacts is not None, "artifacts cannot be None"
        assert (
            len(artifacts) <= MAX_AGGREGATION_CHUNKS
        ), f"Too many artifacts: {len(artifacts)} > {MAX_AGGREGATION_CHUNKS}"

        # Determine target fields
        target_fields = self._get_target_fields(schema, schema_def)

        # Extract and merge data
        merged_data: Dict[str, Any] = {}
        field_sources: Dict[str, List[str]] = {}
        source_ids: List[str] = []

        for artifact in artifacts:
            source_ids.append(artifact.artifact_id)
            self._extract_and_merge(artifact, target_fields, merged_data, field_sources)

        # Validate against schema if provided
        validation_result = self._validate_against_schema(
            merged_data, schema, target_fields
        )

        return AggregationResult(
            data=merged_data,
            schema_valid=validation_result["valid"],
            missing_fields=validation_result["missing"],
            validation_errors=validation_result["errors"],
            source_artifact_ids=source_ids,
            field_sources=field_sources,
        )

    def aggregate_to_artifact(
        self,
        artifacts: List[IFArtifact],
        schema: Optional[Type[BaseModel]] = None,
        schema_def: Optional[Dict[str, Any]] = None,
    ) -> IFTextArtifact:
        """
        Aggregate artifacts and return as IFTextArtifact.

        Args:
            artifacts: List of artifacts to aggregate.
            schema: Pydantic model for validation.
            schema_def: Schema definition for field extraction.

        Returns:
            IFTextArtifact containing aggregated JSON data.

        Rule #4: Function < 60 lines.
        """
        assert artifacts, "artifacts list cannot be empty"

        result = self.aggregate(artifacts, schema, schema_def)

        # Create output artifact
        import json

        content = json.dumps(result.data, indent=2, default=str)

        # Use first artifact as parent for lineage
        parent = artifacts[0]

        return IFTextArtifact(
            artifact_id=f"aggregated:{parent.artifact_id}",
            content=content,
            parent_id=parent.artifact_id,
            root_artifact_id=parent.effective_root_id,
            lineage_depth=parent.lineage_depth + 1,
            provenance=list(parent.provenance) + [self.processor_id],
            metadata={
                "aggregation_result": True,
                "schema_valid": result.schema_valid,
                "missing_fields": result.missing_fields,
                "validation_errors": result.validation_errors,
                "source_count": result.artifact_count,
                "source_artifact_ids": result.source_artifact_ids,
            },
        )

    def _get_target_fields(
        self,
        schema: Optional[Type[BaseModel]],
        schema_def: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """
        Determine target fields from schema or schema_def.

        Args:
            schema: Pydantic model class.
            schema_def: Schema definition dictionary.

        Returns:
            Set of field names to extract.

        Rule #4: Helper < 60 lines.
        """
        fields: Set[str] = set()

        if schema is not None:
            # Extract fields from Pydantic model
            model_fields = getattr(schema, "model_fields", {})
            fields.update(model_fields.keys())

        if schema_def is not None:
            # Extract fields from schema definition dict
            fields.update(schema_def.keys())

        return fields

    def _extract_and_merge(
        self,
        artifact: IFArtifact,
        target_fields: Set[str],
        merged_data: Dict[str, Any],
        field_sources: Dict[str, List[str]],
    ) -> None:
        """
        Extract fields from artifact and merge into result.

        Args:
            artifact: Source artifact.
            target_fields: Fields to extract (empty = all fields).
            merged_data: Accumulated merged data (modified in place).
            field_sources: Tracks which artifacts contributed each field.

        Rule #4: Helper < 60 lines.
        """
        # Extract data from artifact
        artifact_data = self._extract_artifact_data(artifact)

        # If no target fields specified, extract all available fields
        fields_to_process = (
            target_fields if target_fields else set(artifact_data.keys())
        )

        # Merge each field
        for field_name in fields_to_process:
            if field_name not in artifact_data:
                continue

            value = artifact_data[field_name]
            if value is None:
                continue

            # Track source
            if field_name not in field_sources:
                field_sources[field_name] = []
            field_sources[field_name].append(artifact.artifact_id)

            # Apply merge strategy
            strategy = self._field_strategies.get(field_name, self._default_strategy)
            self._merge_field(field_name, value, merged_data, strategy)

    def _extract_artifact_data(self, artifact: IFArtifact) -> Dict[str, Any]:
        """
        Extract data dictionary from an artifact.

        Args:
            artifact: Source artifact.

        Returns:
            Dictionary of field values.

        Rule #4: Helper < 60 lines.
        """
        data: Dict[str, Any] = {}

        # Check metadata for structured data
        if artifact.metadata:
            data.update(artifact.metadata)

        # For chunk artifacts, extract from content if structured
        if isinstance(artifact, IFChunkArtifact):
            extracted = self._parse_structured_content(artifact.content)
            data.update(extracted)

        # For text artifacts, try to parse as JSON
        if isinstance(artifact, IFTextArtifact):
            extracted = self._parse_structured_content(artifact.content)
            data.update(extracted)

        return data

    def _parse_structured_content(self, content: str) -> Dict[str, Any]:
        """
        Try to parse content as structured data.

        Args:
            content: Text content to parse.

        Returns:
            Parsed dictionary or empty dict.

        Rule #4: Helper < 60 lines.
        """
        if not content:
            return {}

        # Try JSON parsing
        import json

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Try key-value extraction (simple pattern)
        return self._extract_key_values(content)

    def _extract_key_values(self, content: str) -> Dict[str, Any]:
        """
        Extract key-value pairs from text content.

        Supports patterns like:
        - "key: value"
        - "key = value"

        Args:
            content: Text content.

        Returns:
            Extracted key-value pairs.

        Rule #4: Helper < 60 lines.
        """
        result: Dict[str, Any] = {}

        # Pattern: "key: value" or "key = value"
        patterns = [
            r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(.+)$",
        ]

        lines = content.split("\n")
        for line in lines[:100]:  # JPL Rule #2: bounded iteration
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    key = match.group(1).lower()
                    value = match.group(2).strip()
                    # Clean quotes
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    result[key] = value
                    break

        return result

    def _merge_field(
        self,
        field_name: str,
        value: Any,
        merged_data: Dict[str, Any],
        strategy: AggregationStrategy,
    ) -> None:
        """
        Merge a field value using the specified strategy.

        Args:
            field_name: Field name.
            value: New value to merge.
            merged_data: Target dictionary (modified in place).
            strategy: Merge strategy to apply.

        Rule #4: Helper < 60 lines.
        """
        existing = merged_data.get(field_name)

        if strategy == AggregationStrategy.FIRST_WINS:
            if existing is None:
                merged_data[field_name] = value

        elif strategy == AggregationStrategy.LAST_WINS:
            merged_data[field_name] = value

        elif strategy == AggregationStrategy.ACCUMULATE:
            if existing is None:
                new_list = [value] if not isinstance(value, list) else list(value)
                # Apply bounds on initial value too
                merged_data[field_name] = new_list[:MAX_LIST_ITEMS]
            else:
                if not isinstance(existing, list):
                    existing = [existing]
                if isinstance(value, list):
                    existing.extend(value)
                else:
                    existing.append(value)
                # Apply bounds
                merged_data[field_name] = existing[:MAX_LIST_ITEMS]

        elif strategy == AggregationStrategy.MERGE_UNIQUE:
            if existing is None:
                new_list = [value] if not isinstance(value, list) else list(value)
                # Apply bounds on initial value too
                merged_data[field_name] = new_list[:MAX_LIST_ITEMS]
            else:
                if not isinstance(existing, list):
                    existing = [existing]
                if isinstance(value, list):
                    for v in value:
                        if v not in existing:
                            existing.append(v)
                elif value not in existing:
                    existing.append(value)
                # Apply bounds
                merged_data[field_name] = existing[:MAX_LIST_ITEMS]

    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Optional[Type[BaseModel]],
        target_fields: Set[str],
    ) -> Dict[str, Any]:
        """
        Validate aggregated data against schema.

        Args:
            data: Aggregated data dictionary.
            schema: Pydantic model for validation.
            target_fields: Expected fields.

        Returns:
            Dictionary with 'valid', 'missing', 'errors' keys.

        Rule #4: Helper < 60 lines.
        """
        result: Dict[str, Any] = {
            "valid": True,
            "missing": [],
            "errors": [],
        }

        if schema is None:
            return result

        # Check for missing required fields
        model_fields = getattr(schema, "model_fields", {})
        for field_name, field_info in model_fields.items():
            is_required = field_info.is_required()
            if is_required and field_name not in data:
                result["missing"].append(field_name)

        # Try to validate with Pydantic
        try:
            schema(**data)
        except ValidationError as e:
            result["valid"] = False
            for error in e.errors():
                loc = ".".join(str(x) for x in error.get("loc", []))
                msg = error.get("msg", "Unknown error")
                result["errors"].append(f"{loc}: {msg}")

        # Mark invalid if missing required fields
        if result["missing"]:
            result["valid"] = False

        return result


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def aggregate_artifacts(
    artifacts: List[IFArtifact],
    schema: Optional[Type[BaseModel]] = None,
    schema_def: Optional[Dict[str, Any]] = None,
    strategy: AggregationStrategy = AggregationStrategy.FIRST_WINS,
) -> AggregationResult:
    """
    Convenience function to aggregate artifacts.

    Args:
        artifacts: List of artifacts to aggregate.
        schema: Pydantic model for validation.
        schema_def: Schema definition for field extraction.
        strategy: Default merge strategy.

    Returns:
        AggregationResult with merged data.
    """
    aggregator = IFAggregator(default_strategy=strategy)
    return aggregator.aggregate(artifacts, schema, schema_def)


def aggregate_to_json(
    artifacts: List[IFArtifact],
    schema: Optional[Type[BaseModel]] = None,
    schema_def: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Aggregate artifacts and return as JSON string.

    Args:
        artifacts: List of artifacts to aggregate.
        schema: Pydantic model for validation.
        schema_def: Schema definition for field extraction.

    Returns:
        JSON string of aggregated data.
    """
    import json

    result = aggregate_artifacts(artifacts, schema, schema_def)
    return json.dumps(result.data, indent=2, default=str)
