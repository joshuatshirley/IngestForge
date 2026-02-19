"""
Blueprint Parser & Validator for IngestForge (IF).

Blueprint Parser & Validator.
Enables declarative vertical pipeline definitions via YAML blueprints.

NASA JPL Power of Ten compliant.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ingestforge.core.pipeline.schema_generator import (
    SchemaGenerator,
    SchemaDefinitionError,
)

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_STAGES = 20
MAX_BLUEPRINT_SIZE = 65536  # 64KB
MAX_CONFIG_DEPTH = 5
MAX_VERTICAL_ID_LENGTH = 64
MAX_NAME_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 1024
MAX_BLUEPRINTS = 64


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BlueprintValidationError(Exception):
    """
    Raised when blueprint validation fails.

    AC: Clear message indicating which fields are invalid.
    """

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        """
        Initialize with validation error details.

        Args:
            message: Error description.
            field: Optional field name that caused the error.
        """
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}" if field else message)


class BlueprintLoadError(Exception):
    """Raised when blueprint file cannot be loaded."""

    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class StageConfig(BaseModel):
    """
    Configuration for a single pipeline stage.

    Defines processor and its configuration within a blueprint.
    Rule #9: Complete type hints.
    """

    model_config = {"extra": "forbid"}

    processor: str = Field(
        ..., min_length=1, max_length=128, description="Processor class name or ID"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Processor-specific configuration"
    )
    enabled: bool = Field(default=True, description="Whether this stage is active")

    @field_validator("processor")
    @classmethod
    def validate_processor(cls, v: str) -> str:
        """Validate processor name is non-empty and reasonable."""
        assert v and v.strip(), "Processor name cannot be empty"
        return v.strip()


class VerticalBlueprint(BaseModel):
    """
    Complete vertical pipeline definition.

    GWT-1: Validated Pydantic model with all fields properly typed.
    Rule #9: Complete type hints.
    """

    model_config = {"extra": "forbid", "frozen": True}

    vertical_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_VERTICAL_ID_LENGTH,
        description="Unique identifier for this vertical",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_NAME_LENGTH,
        description="Human-readable vertical name",
    )
    version: str = Field(
        default="1.0.0", max_length=32, description="Blueprint version (semver)"
    )
    description: str = Field(
        default="",
        max_length=MAX_DESCRIPTION_LENGTH,
        description="Vertical description",
    )
    stages: List[StageConfig] = Field(
        ...,
        min_length=1,
        max_length=MAX_STAGES,
        description="Ordered list of processing stages",
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Target output schema definition"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("vertical_id")
    @classmethod
    def validate_vertical_id(cls, v: str) -> str:
        """Validate vertical_id format."""
        assert v and v.strip(), "vertical_id cannot be empty"
        # Allow alphanumeric, hyphens, underscores
        cleaned = v.strip().lower()
        for char in cleaned:
            if not (char.isalnum() or char in "-_"):
                raise ValueError(f"vertical_id contains invalid character: '{char}'")
        return cleaned

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v: List[StageConfig]) -> List[StageConfig]:
        """Validate stages list is non-empty and bounded."""
        assert v, "At least one stage is required"
        assert len(v) <= MAX_STAGES, f"Maximum {MAX_STAGES} stages allowed"
        return v

    @model_validator(mode="after")
    def validate_output_schema(self) -> "VerticalBlueprint":
        """Validate output_schema if provided."""
        if self.output_schema is not None:
            try:
                generator = SchemaGenerator()
                generator.generate(f"{self.vertical_id}_output", self.output_schema)
            except SchemaDefinitionError as e:
                raise ValueError(f"Invalid output_schema: {e}") from e
        return self


# ---------------------------------------------------------------------------
# BlueprintParser
# ---------------------------------------------------------------------------


class BlueprintParser:
    """
    Parse and validate YAML blueprint files.

    GWT-1: Returns validated VerticalBlueprint on success.
    GWT-2: Raises BlueprintValidationError on invalid input.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._schema_generator = SchemaGenerator()

    def _validate_required_fields(self, data: Dict[str, Any]) -> None:
        """Validate required fields exist. Rule #4: Helper < 60 lines."""
        required_fields = ["vertical_id", "name", "stages"]
        for field in required_fields:
            if field not in data:
                raise BlueprintValidationError(
                    f"Missing required field: {field}", field=field
                )
        if not isinstance(data.get("stages"), list):
            raise BlueprintValidationError("stages must be a list", field="stages")

    def _convert_stages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert stage dicts to StageConfig objects. Rule #4: Helper < 60 lines."""
        try:
            stages = [StageConfig(**stage) for stage in data["stages"]]
            result = dict(data)
            result["stages"] = stages
            return result
        except Exception as e:
            raise BlueprintValidationError(
                f"Invalid stage configuration: {e}", field="stages"
            ) from e

    def parse_file(self, path: Path) -> VerticalBlueprint:
        """
        Parse a YAML blueprint file.

        Args:
            path: Path to the YAML file.

        Returns:
            Validated VerticalBlueprint instance.

        Raises:
            BlueprintLoadError: If file cannot be read.
            BlueprintValidationError: If blueprint is invalid.

        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.
        """
        assert path is not None, "Path cannot be None"

        # Validate file exists and is readable
        if not path.exists():
            # SEC-002: Sanitize path disclosure
            logger.error(f"Blueprint file not found: {path}")
            raise BlueprintLoadError("Blueprint file not found: [REDACTED]")
        if not path.is_file():
            # SEC-002: Sanitize path disclosure
            logger.error(f"Not a file: {path}")
            raise BlueprintLoadError("Not a file: [REDACTED]")

        # Check file size (JPL Rule #2)
        file_size = path.stat().st_size
        if file_size > MAX_BLUEPRINT_SIZE:
            raise BlueprintLoadError(
                f"Blueprint exceeds maximum size of {MAX_BLUEPRINT_SIZE} bytes"
            )

        # Read and parse YAML
        try:
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise BlueprintLoadError(f"Invalid YAML syntax: {e}") from e
        except IOError as e:
            raise BlueprintLoadError(f"Cannot read file: {e}") from e

        if data is None:
            raise BlueprintValidationError("Blueprint file is empty")

        return self.parse_dict(data)

    def parse_dict(self, data: Dict[str, Any]) -> VerticalBlueprint:
        """
        Parse a dictionary into a VerticalBlueprint.

        Args:
            data: Blueprint data as dictionary.

        Returns:
            Validated VerticalBlueprint instance.

        Raises:
            BlueprintValidationError: If data is invalid.

        Rule #4: Function < 60 lines (uses helpers).
        """
        assert data is not None, "Data cannot be None"
        assert isinstance(data, dict), "Data must be a dictionary"

        # Validate and convert
        self._validate_required_fields(data)
        data = self._convert_stages(data)

        # Create and validate blueprint
        try:
            blueprint = VerticalBlueprint(**data)
        except Exception as e:
            field = None
            if hasattr(e, "errors") and callable(e.errors):
                errors = e.errors()
                if errors and "loc" in errors[0]:
                    field = str(errors[0]["loc"][0])
            raise BlueprintValidationError(str(e), field=field) from e

        logger.info(f"Parsed blueprint: {blueprint.vertical_id}")
        return blueprint

    def validate(self, blueprint: VerticalBlueprint) -> List[str]:
        """
        Perform additional validation on a parsed blueprint.

        Args:
            blueprint: Blueprint to validate.

        Returns:
            List of warning messages (empty if all valid).

        Rule #4: Function < 60 lines.
        """
        warnings: List[str] = []

        # Check for duplicate processor IDs
        processor_ids = [stage.processor for stage in blueprint.stages]
        seen: set[str] = set()
        for pid in processor_ids:
            if pid in seen:
                warnings.append(f"Duplicate processor in stages: {pid}")
            seen.add(pid)

        # Check for disabled stages
        disabled_count = sum(1 for s in blueprint.stages if not s.enabled)
        if disabled_count > 0:
            warnings.append(f"{disabled_count} stage(s) are disabled")

        # Validate output_schema can generate a model
        if blueprint.output_schema:
            try:
                self._schema_generator.generate(
                    f"{blueprint.vertical_id}_output", blueprint.output_schema
                )
            except SchemaDefinitionError as e:
                warnings.append(f"Output schema warning: {e}")

        return warnings


# ---------------------------------------------------------------------------
# BlueprintRegistry
# ---------------------------------------------------------------------------


class BlueprintRegistry:
    """
    Registry for discovered vertical blueprints.

    GWT-3: Scans directory and registers all valid blueprints.

    Rule #2: Fixed upper bound on registered blueprints.
    Rule #9: Complete type hints.
    """

    _instance: Optional["BlueprintRegistry"] = None
    _blueprints: Dict[str, VerticalBlueprint]
    _parser: BlueprintParser

    def __new__(cls) -> "BlueprintRegistry":
        """Singleton pattern. Rule #4: < 60 lines."""
        if cls._instance is None:
            cls._instance = super(BlueprintRegistry, cls).__new__(cls)
            cls._instance._blueprints = {}
            cls._instance._parser = BlueprintParser()
        return cls._instance

    @property
    def blueprints(self) -> Dict[str, VerticalBlueprint]:
        """Get all registered blueprints."""
        return dict(self._blueprints)

    def register(self, blueprint: VerticalBlueprint) -> None:
        """
        Register a blueprint.

        Args:
            blueprint: Validated blueprint to register.

        Raises:
            RuntimeError: If registry limit reached.

        Rule #2: Fixed upper bound.
        """
        if len(self._blueprints) >= MAX_BLUEPRINTS:
            raise RuntimeError(f"Blueprint registry limit reached: {MAX_BLUEPRINTS}")

        self._blueprints[blueprint.vertical_id] = blueprint
        logger.info(f"Registered blueprint: {blueprint.vertical_id}")

    def get(self, vertical_id: str) -> Optional[VerticalBlueprint]:
        """
        Get a blueprint by vertical_id.

        Args:
            vertical_id: The vertical identifier.

        Returns:
            Blueprint if found, None otherwise.
        """
        return self._blueprints.get(vertical_id)

    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """
        Scan a directory for blueprint files and register valid ones.

        Args:
            directory: Directory to scan for .yaml/.yml files.

        Returns:
            Summary dict with loaded, failed, and errors.

        Rule #4: Function < 60 lines.
        Rule #7: Check all return values.
        """
        assert directory is not None, "Directory cannot be None"

        if not directory.exists():
            logger.warning(f"Blueprint directory does not exist: {directory}")
            return {"loaded": 0, "failed": 0, "errors": []}

        if not directory.is_dir():
            logger.warning(f"Not a directory: {directory}")
            return {"loaded": 0, "failed": 0, "errors": []}

        loaded = 0
        failed = 0
        errors: List[str] = []

        # Find YAML files
        yaml_files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))

        for yaml_file in yaml_files:
            try:
                blueprint = self._parser.parse_file(yaml_file)
                self.register(blueprint)
                loaded += 1
            except (BlueprintLoadError, BlueprintValidationError) as e:
                failed += 1
                errors.append(f"{yaml_file.name}: {e}")
                logger.warning(f"Failed to load blueprint {yaml_file}: {e}")
            except RuntimeError as e:
                # Registry limit reached
                failed += 1
                errors.append(f"{yaml_file.name}: {e}")
                break

        logger.info(f"Blueprint scan complete: {loaded} loaded, {failed} failed")
        return {"loaded": loaded, "failed": failed, "errors": errors}

    def list_verticals(self) -> List[str]:
        """
        List all registered vertical IDs.

        Returns:
            List of vertical_id strings.
        """
        return list(self._blueprints.keys())

    def clear(self) -> None:
        """Clear all registered blueprints (for testing)."""
        self._blueprints.clear()

    def get_output_model(self, vertical_id: str) -> Optional[Type[BaseModel]]:
        """
        Get the generated output schema model for a vertical.

        Args:
            vertical_id: The vertical identifier.

        Returns:
            Generated Pydantic model, or None if no schema defined.
        """
        blueprint = self.get(vertical_id)
        if blueprint is None or blueprint.output_schema is None:
            return None

        generator = SchemaGenerator()
        return generator.generate(f"{vertical_id}_output", blueprint.output_schema)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def load_blueprint(path: Path) -> VerticalBlueprint:
    """
    Convenience function to load a single blueprint file.

    Args:
        path: Path to the YAML blueprint file.

    Returns:
        Validated VerticalBlueprint instance.

    Raises:
        BlueprintLoadError: If file cannot be read.
        BlueprintValidationError: If blueprint is invalid.
    """
    parser = BlueprintParser()
    return parser.parse_file(path)


def load_blueprint_from_dict(data: Dict[str, Any]) -> VerticalBlueprint:
    """
    Convenience function to load a blueprint from a dictionary.

    Args:
        data: Blueprint data as dictionary.

    Returns:
        Validated VerticalBlueprint instance.

    Raises:
        BlueprintValidationError: If data is invalid.
    """
    parser = BlueprintParser()
    return parser.parse_dict(data)


def get_blueprint_registry() -> BlueprintRegistry:
    """
    Get the singleton blueprint registry instance.

    Returns:
        The BlueprintRegistry singleton.
    """
    return BlueprintRegistry()
