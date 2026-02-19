"""
Configuration Factory for IngestForge (IF).

Slices the monolithic Config object into scoped sub-configs for processors.
Supports output schema loading from vertical blueprints ().
Follows NASA JPL Power of Ten rules.
"""

import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ingestforge.core.config.config import Config
from ingestforge.core.pipeline.schema_generator import (
    SchemaGenerator,
    SchemaDefinitionError,
)

# BlueprintError is an alias for schema errors in extraction context
BlueprintError = SchemaDefinitionError

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_SCHEMA_NAME_LENGTH = 128


class IFConfigFactory:
    """
    Factory to extract scoped configuration blocks.
    """

    @staticmethod
    def get_sub_config(config: Config, path: str) -> Any:
        """
        Extract a nested config object using a dot-separated path.

        Example: get_sub_config(config, "ingest") -> config.ingest

        Rule #5: Assertion density.
        """
        assert config is not None, "Root config cannot be None"
        assert path, "Config path cannot be empty"

        parts = path.split(".")
        current = config

        for part in parts:
            if not hasattr(current, part):
                logger.error(f"Config path not found: {path} (missing '{part}')")
                raise AttributeError(f"Configuration has no attribute '{part}'")

            current = getattr(current, part)
        assert current is not None, f"Sub-config at {path} is None"

        return current

    @staticmethod
    def get_output_schema(
        config: Config, schema_name: str = "OutputInsight"
    ) -> Optional[Type[BaseModel]]:
        """
        Extract and generate a Pydantic model from the output_schema config block.

        Looks for an `output_schema` attribute in the config (typically from a
        vertical blueprint). If found, generates a dynamic Pydantic model.

        Args:
            config: The configuration object (may have output_schema attribute).
            schema_name: Name for the generated Pydantic model class.

        Returns:
            Generated Pydantic model class, or None if no schema defined.

        Raises:
            SchemaDefinitionError: If schema definition is invalid.

        Programmatic Output Schema Definition.
        Rule #4: Function < 60 lines.
        Rule #7: Check all inputs.
        """
        assert config is not None, "Config cannot be None"
        assert schema_name, "Schema name cannot be empty"
        assert (
            len(schema_name) <= MAX_SCHEMA_NAME_LENGTH
        ), f"Schema name exceeds max length of {MAX_SCHEMA_NAME_LENGTH}"

        # Check if config has output_schema attribute
        if not hasattr(config, "output_schema"):
            logger.debug("No output_schema attribute in config")
            return None

        schema_def = getattr(config, "output_schema")
        if schema_def is None:
            logger.debug("output_schema is None")
            return None

        # Schema definition should be a dictionary
        if not isinstance(schema_def, dict):
            raise SchemaDefinitionError(
                f"output_schema must be a dict, got {type(schema_def).__name__}"
            )

        if not schema_def:
            logger.debug("output_schema is empty")
            return None

        # Generate the Pydantic model
        generator = SchemaGenerator()
        model = generator.generate(schema_name, schema_def)
        logger.info(f"Generated output schema: {schema_name}")

        return model

    @staticmethod
    def get_output_schema_from_dict(
        schema_def: Dict[str, Any], schema_name: str = "OutputInsight"
    ) -> Type[BaseModel]:
        """
        Generate a Pydantic model from a dictionary schema definition.

        Convenience method for when the schema is already extracted as a dict.

        Args:
            schema_def: Dictionary mapping field names to type definitions.
            schema_name: Name for the generated Pydantic model class.

        Returns:
            Generated Pydantic model class.

        Raises:
            SchemaDefinitionError: If schema definition is invalid.

        Direct schema dict to model conversion.
        """
        assert schema_def, "Schema definition cannot be empty"
        assert schema_name, "Schema name cannot be empty"

        generator = SchemaGenerator()
        return generator.generate(schema_name, schema_def)

    @staticmethod
    def create_extraction_model(
        schema_def: Dict[str, Any], model_name: str = "ExtractionModel"
    ) -> Type[BaseModel]:
        """
        Convert a YAML extraction schema into a live Pydantic BaseModel class.

        Dynamic Schema Factory. Uses pydantic.create_model internally.
        Supports: str, int, float, bool, list[<type>], date, datetime, optional.

        Args:
            schema_def: Dict mapping field names to types (e.g., {"name": "str"}).
            model_name: Name for the generated Pydantic model class.

        Returns:
            Dynamically created Pydantic model class ready for Instructor.

        Raises:
            BlueprintError: If schema is invalid or contains unknown types.

        Rule #4: < 60 lines. Rule #5: Assert dict input. Rule #7: Check inputs.
        """
        # Rule #5: Assert preconditions
        assert schema_def is not None, "schema_def cannot be None"
        assert isinstance(schema_def, dict), "schema_def must be a dictionary"
        assert model_name, "model_name cannot be empty"
        assert (
            len(model_name) <= MAX_SCHEMA_NAME_LENGTH
        ), f"model_name exceeds max length of {MAX_SCHEMA_NAME_LENGTH}"

        if not schema_def:
            raise BlueprintError("Schema definition cannot be empty")

        generator = SchemaGenerator()
        try:
            model = generator.generate(model_name, schema_def)
        except SchemaDefinitionError as e:
            raise BlueprintError(str(e)) from e

        logger.info(f"Created extraction model: {model_name}")
        return model
