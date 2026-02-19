"""
Schema Generator for IngestForge (IF).

Converts YAML output schema definitions into dynamic Pydantic models.
Supports vertical blueprints with declarative output schemas.

Programmatic Output Schema Definition.
Follows NASA JPL Power of Ten rules.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_SCHEMA_FIELDS = 64
MAX_FIELD_NAME_LENGTH = 128
MAX_NESTED_DEPTH = 5

# Supported type mappings from YAML strings to Python types
TYPE_MAPPINGS: Dict[str, Type] = {
    # Basic types
    "string": str,
    "str": str,
    "text": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    # Date/time types
    "date": date,
    "datetime": datetime,
    # Special types
    "any": Any,
}


class SchemaDefinitionError(Exception):
    """Raised when schema definition is invalid."""

    pass


class SchemaGenerator:
    """
    Generate dynamic Pydantic models from YAML schema definitions.

    Converts a dictionary-based schema definition (from YAML) into a
    Pydantic BaseModel class that can be used for validation and extraction.

    Programmatic Output Schema Definition.
    Rule #4: Functions < 60 lines.
    Rule #9: Complete type hints.

    Example:
        >>> schema_def = {
        ...     "name": "string",
        ...     "age": "int",
        ...     "tags": "list[string]",
        ...     "published": "date"
        ... }
        >>> generator = SchemaGenerator()
        >>> MyModel = generator.generate("ArticleInsight", schema_def)
        >>> instance = MyModel(name="Test", age=25, tags=["a", "b"], published=date.today())
    """

    def __init__(self, allow_optional: bool = True) -> None:
        """
        Initialize the schema generator.

        Args:
            allow_optional: If True, fields without 'required' marker are optional.
        """
        self._allow_optional = allow_optional
        self._generated_models: Dict[str, Type[BaseModel]] = {}

    def generate(
        self,
        model_name: str,
        schema_def: Dict[str, Any],
        base_class: Type[BaseModel] = BaseModel,
    ) -> Type[BaseModel]:
        """
        Generate a Pydantic model from a schema definition.

        Args:
            model_name: Name for the generated model class.
            schema_def: Dictionary mapping field names to type definitions.
            base_class: Optional base class to inherit from.

        Returns:
            Dynamically created Pydantic model class.

        Raises:
            SchemaDefinitionError: If schema is invalid.

        Rule #4: Function < 60 lines.
        Rule #5: Assert input types (JPL Rule #5).
        Rule #7: Check all inputs.
        """
        # JPL Rule #5: Assert input types
        assert isinstance(model_name, str), "model_name must be a string"
        assert isinstance(schema_def, dict), "schema_def must be a dictionary"

        # Validate inputs
        if not model_name:
            raise SchemaDefinitionError("Model name cannot be empty")
        if not schema_def:
            raise SchemaDefinitionError("Schema definition cannot be empty")
        if len(schema_def) > MAX_SCHEMA_FIELDS:
            raise SchemaDefinitionError(
                f"Schema exceeds maximum of {MAX_SCHEMA_FIELDS} fields"
            )

        # Parse field definitions
        field_definitions = self._parse_fields(schema_def)

        # Create the model
        model = create_model(model_name, __base__=base_class, **field_definitions)

        # Cache the generated model
        self._generated_models[model_name] = model
        logger.info(
            f"Generated schema model: {model_name} with {len(field_definitions)} fields"
        )

        return model

    def _parse_fields(
        self, schema_def: Dict[str, Any], depth: int = 0
    ) -> Dict[str, Any]:
        """
        Parse field definitions from schema dictionary.

        Rule #2: Bounded recursion depth.
        Rule #4: Function < 60 lines.
        """
        if depth > MAX_NESTED_DEPTH:
            raise SchemaDefinitionError(
                f"Schema nesting exceeds maximum depth of {MAX_NESTED_DEPTH}"
            )

        field_definitions = {}
        for field_name, field_type in schema_def.items():
            if len(field_name) > MAX_FIELD_NAME_LENGTH:
                raise SchemaDefinitionError(
                    f"Field name '{field_name[:20]}...' exceeds maximum length"
                )
            parsed_type, is_required = self._parse_type(field_type, depth)
            if is_required:
                field_definitions[field_name] = (parsed_type, ...)
            else:
                field_definitions[field_name] = (Optional[parsed_type], None)
        return field_definitions

    def _parse_type(self, type_def: Any, depth: int = 0) -> tuple[Type, bool]:
        """
        Parse a type definition into a Python type. Rule #4: < 60 lines.
        """
        if isinstance(type_def, dict):
            return self._parse_dict_type(type_def, depth)
        if isinstance(type_def, str):
            return self._parse_string_type(type_def, depth)
        raise SchemaDefinitionError(
            f"Invalid type definition: {type_def} (type: {type(type_def).__name__})"
        )

    def _parse_dict_type(self, type_def: Dict, depth: int) -> tuple[Type, bool]:
        """Parse dictionary type definitions (nested objects or specs)."""
        if "type" in type_def:
            is_required = type_def.get("required", True)
            return self._parse_type(type_def["type"], depth)[0], is_required
        # Nested object - create sub-model with incremented depth
        if depth + 1 > MAX_NESTED_DEPTH:
            raise SchemaDefinitionError(
                f"Schema nesting exceeds maximum depth of {MAX_NESTED_DEPTH}"
            )
        nested_name = f"NestedModel_{depth}_{len(type_def)}"
        nested_fields = self._parse_fields(type_def, depth + 1)
        nested_model = create_model(nested_name, **nested_fields)
        self._generated_models[nested_name] = nested_model
        return nested_model, True

    def _parse_string_type(self, type_def: str, depth: int) -> tuple[Type, bool]:
        """Parse string type definitions."""
        type_str = type_def.strip().lower()
        is_required = True

        if type_str.endswith("?"):
            type_str = type_str[:-1]
            is_required = False

        if type_str.startswith("list[") and type_str.endswith("]"):
            inner_type, _ = self._parse_type(type_str[5:-1], depth + 1)
            return List[inner_type], is_required

        if type_str.startswith("optional[") and type_str.endswith("]"):
            inner_type, _ = self._parse_type(type_str[9:-1], depth + 1)
            return inner_type, False

        if type_str in TYPE_MAPPINGS:
            return TYPE_MAPPINGS[type_str], is_required

        raise SchemaDefinitionError(
            f"Unknown type '{type_def}'. Supported: {list(TYPE_MAPPINGS.keys())}"
        )

    def get_model(self, model_name: str) -> Optional[Type[BaseModel]]:
        """
        Retrieve a previously generated model by name.

        Args:
            model_name: Name of the model to retrieve.

        Returns:
            The model class, or None if not found.
        """
        return self._generated_models.get(model_name)

    def list_models(self) -> List[str]:
        """
        List all generated model names.

        Returns:
            List of model names.
        """
        return list(self._generated_models.keys())

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._generated_models.clear()


def generate_schema_from_yaml(
    model_name: str, yaml_schema: Dict[str, Any]
) -> Type[BaseModel]:
    """
    Convenience function to generate a schema from YAML dict.

    Args:
        model_name: Name for the generated model.
        yaml_schema: Schema definition as parsed YAML dictionary.

    Returns:
        Generated Pydantic model class.

    Example:
        >>> schema = {"title": "string", "count": "int"}
        >>> Model = generate_schema_from_yaml("MyModel", schema)
    """
    generator = SchemaGenerator()
    return generator.generate(model_name, yaml_schema)


def attach_schema_to_artifact(
    artifact_metadata: Dict[str, Any], schema_model: Type[BaseModel]
) -> Dict[str, Any]:
    """
    Attach a schema model reference to artifact metadata.

    Since Pydantic models can't be directly serialized to JSON,
    we store the model name and schema definition.

    Args:
        artifact_metadata: Existing artifact metadata dict.
        schema_model: The Pydantic model to attach.

    Returns:
        Updated metadata dictionary with schema info.

    Schema attached to IFArtifact as target_model property.
    """
    # Extract schema info from the model
    schema_info = {
        "target_model_name": schema_model.__name__,
        "target_model_schema": schema_model.model_json_schema(),
    }

    # Merge with existing metadata
    updated = dict(artifact_metadata)
    updated["_target_schema"] = schema_info

    return updated
