"""
Tests for IFConfigFactory.

Verifies configuration extraction and output schema loading.
Programmatic Output Schema Definition integration.

Test Categories:
- GWT (Given-When-Then) behavioral specifications
- JPL Power of Ten rule compliance verification
- Error handling and edge cases
"""

import inspect
import pytest
from typing import get_type_hints
from unittest.mock import Mock

from pydantic import BaseModel

from ingestforge.core.config.config import Config
from ingestforge.core.pipeline.config_factory import (
    IFConfigFactory,
    MAX_SCHEMA_NAME_LENGTH,
    BlueprintError,
)
from ingestforge.core.config.base import IngestConfig
from ingestforge.core.pipeline.schema_generator import SchemaDefinitionError


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLRule2FixedUpperBounds:
    """
    JPL Rule #2: All loops must have a fixed upper-bound.

    Tests verify that constants are defined and enforced.
    """

    def test_max_schema_name_length_defined(self):
        """
        GWT:
        Given the config_factory module.
        When MAX_SCHEMA_NAME_LENGTH is accessed.
        Then it is a positive integer with reasonable bound.
        """
        assert isinstance(MAX_SCHEMA_NAME_LENGTH, int)
        assert MAX_SCHEMA_NAME_LENGTH > 0
        assert MAX_SCHEMA_NAME_LENGTH <= 256

    def test_schema_name_length_at_boundary(self):
        """
        GWT:
        Given a schema name at exactly MAX_SCHEMA_NAME_LENGTH.
        When get_output_schema() is called.
        Then the schema is accepted.
        """
        config = Mock()
        config.output_schema = {"field": "string"}

        schema_name = "A" * MAX_SCHEMA_NAME_LENGTH
        Model = IFConfigFactory.get_output_schema(config, schema_name=schema_name)

        assert Model is not None
        assert Model.__name__ == schema_name

    def test_schema_name_length_over_boundary(self):
        """
        GWT:
        Given a schema name exceeding MAX_SCHEMA_NAME_LENGTH.
        When get_output_schema() is called.
        Then AssertionError is raised.
        """
        config = Mock()
        config.output_schema = {"field": "string"}

        schema_name = "A" * (MAX_SCHEMA_NAME_LENGTH + 1)

        with pytest.raises(AssertionError, match="max length"):
            IFConfigFactory.get_output_schema(config, schema_name=schema_name)


class TestJPLRule4FunctionLength:
    """
    JPL Rule #4: No function should be longer than 60 lines.

    Tests verify that all public methods comply with this rule.
    """

    def test_get_sub_config_method_length(self):
        """
        GWT:
        Given the IFConfigFactory.get_sub_config() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(IFConfigFactory.get_sub_config)[0]
        assert len(source_lines) < 60, f"get_sub_config() has {len(source_lines)} lines"

    def test_get_output_schema_method_length(self):
        """
        GWT:
        Given the IFConfigFactory.get_output_schema() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(IFConfigFactory.get_output_schema)[0]
        assert (
            len(source_lines) < 60
        ), f"get_output_schema() has {len(source_lines)} lines"

    def test_get_output_schema_from_dict_method_length(self):
        """
        GWT:
        Given the IFConfigFactory.get_output_schema_from_dict() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(
            IFConfigFactory.get_output_schema_from_dict
        )[0]
        assert (
            len(source_lines) < 60
        ), f"get_output_schema_from_dict() has {len(source_lines)} lines"


class TestJPLRule5AssertionDensity:
    """
    JPL Rule #5: Assertions should be used liberally.

    Tests verify that invalid inputs are caught with meaningful errors.
    """

    def test_get_sub_config_none_config_rejected(self):
        """
        GWT:
        Given a None config.
        When get_sub_config() is called.
        Then AssertionError is raised.
        """
        with pytest.raises(AssertionError):
            IFConfigFactory.get_sub_config(None, "any")

    def test_get_sub_config_empty_path_rejected(self):
        """
        GWT:
        Given an empty path.
        When get_sub_config() is called.
        Then AssertionError is raised.
        """
        config = Config()
        with pytest.raises(AssertionError):
            IFConfigFactory.get_sub_config(config, "")

    def test_get_output_schema_none_config_rejected(self):
        """
        GWT:
        Given a None config.
        When get_output_schema() is called.
        Then AssertionError is raised.
        """
        with pytest.raises(AssertionError):
            IFConfigFactory.get_output_schema(None)

    def test_get_output_schema_empty_name_rejected(self):
        """
        GWT:
        Given an empty schema name.
        When get_output_schema() is called.
        Then AssertionError is raised.
        """
        config = Mock()
        config.output_schema = {"field": "string"}

        with pytest.raises(AssertionError):
            IFConfigFactory.get_output_schema(config, schema_name="")

    def test_get_output_schema_from_dict_empty_schema_rejected(self):
        """
        GWT:
        Given an empty schema definition.
        When get_output_schema_from_dict() is called.
        Then AssertionError is raised.
        """
        with pytest.raises(AssertionError):
            IFConfigFactory.get_output_schema_from_dict({})

    def test_get_output_schema_from_dict_empty_name_rejected(self):
        """
        GWT:
        Given an empty schema name.
        When get_output_schema_from_dict() is called.
        Then AssertionError is raised.
        """
        with pytest.raises(AssertionError):
            IFConfigFactory.get_output_schema_from_dict(
                {"field": "string"}, schema_name=""
            )


class TestJPLRule7ExplicitTypes:
    """
    JPL Rule #7: Return values and types must be explicit.

    Tests verify that all public methods have type hints.
    """

    def test_get_sub_config_has_type_hints(self):
        """
        GWT:
        Given the IFConfigFactory.get_sub_config() method.
        When type hints are inspected.
        Then return type is explicitly specified.
        """
        hints = get_type_hints(IFConfigFactory.get_sub_config)
        assert "return" in hints

    def test_get_output_schema_has_type_hints(self):
        """
        GWT:
        Given the IFConfigFactory.get_output_schema() method.
        When type hints are inspected.
        Then return type is explicitly specified.
        """
        hints = get_type_hints(IFConfigFactory.get_output_schema)
        assert "return" in hints

    def test_get_output_schema_from_dict_has_type_hints(self):
        """
        GWT:
        Given the IFConfigFactory.get_output_schema_from_dict() method.
        When type hints are inspected.
        Then return type is explicitly specified.
        """
        hints = get_type_hints(IFConfigFactory.get_output_schema_from_dict)
        assert "return" in hints


# =============================================================================
# GWT BEHAVIORAL TESTS - GET_SUB_CONFIG
# =============================================================================


class TestGetSubConfig:
    """Test get_sub_config() method."""

    def test_get_sub_config_simple(self):
        """
        GWT:
        Given a root Config.
        When get_sub_config is called with 'ingest'.
        Then it returns the IngestConfig object.
        """
        config = Config()
        sub = IFConfigFactory.get_sub_config(config, "ingest")
        assert isinstance(sub, IngestConfig)
        assert sub is config.ingest

    def test_get_sub_config_nested(self):
        """
        GWT:
        Given a root Config.
        When get_sub_config is called with 'storage.chromadb'.
        Then it returns the nested ChromaDBConfig.
        """
        config = Config()
        sub = IFConfigFactory.get_sub_config(config, "storage.chromadb")
        assert sub is config.storage.chromadb

    def test_get_sub_config_invalid(self):
        """
        GWT:
        Given an invalid path.
        When get_sub_config is called.
        Then AttributeError is raised.
        """
        config = Config()
        with pytest.raises(AttributeError):
            IFConfigFactory.get_sub_config(config, "non_existent")

    def test_get_sub_config_partial_invalid_path(self):
        """
        GWT:
        Given a path where only the first part is valid.
        When get_sub_config() is called.
        Then AttributeError is raised.
        """
        config = Config()
        with pytest.raises(AttributeError):
            IFConfigFactory.get_sub_config(config, "storage.nonexistent")

    def test_get_sub_config_single_level(self):
        """
        GWT:
        Given a single-level path.
        When get_sub_config() is called.
        Then the direct attribute is returned.
        """
        config = Config()
        sub = IFConfigFactory.get_sub_config(config, "storage")
        assert sub is config.storage

    def test_get_sub_config_deep_nesting(self):
        """
        GWT:
        Given a deeply nested config path.
        When get_sub_config() is called.
        Then the nested value is returned.
        """
        config = Config()
        # Access a deeply nested attribute
        sub = IFConfigFactory.get_sub_config(config, "retrieval.hybrid")
        assert sub is config.retrieval.hybrid


# =============================================================================
# GWT BEHAVIORAL TESTS - GET_OUTPUT_SCHEMA
# =============================================================================


class TestGetOutputSchema:
    """Test get_output_schema() method for ."""

    def test_get_output_schema_no_attribute(self):
        """
        GWT:
        Given a Config without output_schema attribute.
        When get_output_schema() is called.
        Then None is returned.
        """
        config = Config()
        result = IFConfigFactory.get_output_schema(config)
        assert result is None

    def test_get_output_schema_none_value(self):
        """
        GWT:
        Given a Config with output_schema = None.
        When get_output_schema() is called.
        Then None is returned.
        """
        config = Mock()
        config.output_schema = None
        result = IFConfigFactory.get_output_schema(config)
        assert result is None

    def test_get_output_schema_empty_dict(self):
        """
        GWT:
        Given a Config with output_schema = {}.
        When get_output_schema() is called.
        Then None is returned.
        """
        config = Mock()
        config.output_schema = {}
        result = IFConfigFactory.get_output_schema(config)
        assert result is None

    def test_get_output_schema_valid(self):
        """
        GWT:
        Given a Config with a valid output_schema dict.
        When get_output_schema() is called.
        Then a Pydantic model class is returned.
        """
        config = Mock()
        config.output_schema = {
            "title": "string",
            "score": "int",
            "tags": "list[string]",
        }

        Model = IFConfigFactory.get_output_schema(config, schema_name="ArticleInsight")

        assert Model is not None
        assert issubclass(Model, BaseModel)
        assert Model.__name__ == "ArticleInsight"

        # Verify the model works
        instance = Model(title="Test", score=95, tags=["tag1", "tag2"])
        assert instance.title == "Test"
        assert instance.score == 95
        assert instance.tags == ["tag1", "tag2"]

    def test_get_output_schema_invalid_type(self):
        """
        GWT:
        Given a Config with output_schema that is not a dict.
        When get_output_schema() is called.
        Then SchemaDefinitionError is raised.
        """
        config = Mock()
        config.output_schema = "not a dict"

        with pytest.raises(SchemaDefinitionError, match="must be a dict"):
            IFConfigFactory.get_output_schema(config)

    def test_get_output_schema_list_not_dict(self):
        """
        GWT:
        Given a Config with output_schema as a list.
        When get_output_schema() is called.
        Then SchemaDefinitionError is raised.
        """
        config = Mock()
        config.output_schema = ["field1", "field2"]

        with pytest.raises(SchemaDefinitionError, match="must be a dict"):
            IFConfigFactory.get_output_schema(config)

    def test_get_output_schema_default_name(self):
        """
        GWT:
        Given a Config with output_schema and no name specified.
        When get_output_schema() is called.
        Then the model is named 'OutputInsight'.
        """
        config = Mock()
        config.output_schema = {"field": "string"}

        Model = IFConfigFactory.get_output_schema(config)

        assert Model is not None
        assert Model.__name__ == "OutputInsight"

    def test_get_output_schema_custom_name(self):
        """
        GWT:
        Given a Config with output_schema and a custom name.
        When get_output_schema() is called.
        Then the model uses the custom name.
        """
        config = Mock()
        config.output_schema = {"field": "string"}

        Model = IFConfigFactory.get_output_schema(config, schema_name="CustomInsight")

        assert Model.__name__ == "CustomInsight"

    def test_get_output_schema_complex(self):
        """
        GWT:
        Given a Config with a complex output_schema.
        When get_output_schema() is called.
        Then all field types are properly handled.
        """
        config = Mock()
        config.output_schema = {
            "title": "string",
            "count": "int",
            "price": "float",
            "active": "bool",
            "tags": "list[string]",
            "notes": "string?",
        }

        Model = IFConfigFactory.get_output_schema(config, schema_name="ComplexInsight")

        instance = Model(
            title="Test", count=10, price=19.99, active=True, tags=["a", "b"]
        )

        assert instance.title == "Test"
        assert instance.count == 10
        assert instance.price == 19.99
        assert instance.active is True
        assert instance.tags == ["a", "b"]
        assert instance.notes is None


# =============================================================================
# GWT BEHAVIORAL TESTS - GET_OUTPUT_SCHEMA_FROM_DICT
# =============================================================================


class TestGetOutputSchemaFromDict:
    """Test get_output_schema_from_dict() convenience method."""

    def test_get_output_schema_from_dict(self):
        """
        GWT:
        Given a schema definition dict.
        When get_output_schema_from_dict() is called.
        Then a Pydantic model is returned.
        """
        schema_def = {"name": "string", "count": "int"}

        Model = IFConfigFactory.get_output_schema_from_dict(schema_def, "Counter")

        assert Model is not None
        assert issubclass(Model, BaseModel)

        instance = Model(name="items", count=10)
        assert instance.name == "items"
        assert instance.count == 10

    def test_get_output_schema_from_dict_default_name(self):
        """
        GWT:
        Given a schema definition and no name specified.
        When get_output_schema_from_dict() is called.
        Then the model is named 'OutputInsight'.
        """
        schema_def = {"field": "string"}

        Model = IFConfigFactory.get_output_schema_from_dict(schema_def)

        assert Model.__name__ == "OutputInsight"

    def test_get_output_schema_from_dict_with_optional(self):
        """
        GWT:
        Given a schema with optional fields.
        When get_output_schema_from_dict() is called.
        Then optional fields work correctly.
        """
        schema_def = {"required_field": "string", "optional_field": "int?"}

        Model = IFConfigFactory.get_output_schema_from_dict(schema_def, "OptionalModel")

        instance = Model(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field is None

    def test_get_output_schema_from_dict_with_nested(self):
        """
        GWT:
        Given a schema with nested objects.
        When get_output_schema_from_dict() is called.
        Then nested objects are properly generated.
        """
        schema_def = {
            "name": "string",
            "address": {"street": "string", "city": "string"},
        }

        Model = IFConfigFactory.get_output_schema_from_dict(schema_def, "NestedModel")

        instance = Model(name="Alice", address={"street": "123 Main", "city": "Boston"})
        assert instance.name == "Alice"
        assert instance.address.street == "123 Main"
        assert instance.address.city == "Boston"

    def test_get_output_schema_from_dict_invalid_type(self):
        """
        GWT:
        Given a schema with an invalid type.
        When get_output_schema_from_dict() is called.
        Then SchemaDefinitionError is raised.
        """
        schema_def = {"field": "invalid_type"}

        with pytest.raises(SchemaDefinitionError, match="Unknown type"):
            IFConfigFactory.get_output_schema_from_dict(schema_def)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for config factory."""

    def test_schema_model_serialization(self):
        """
        GWT:
        Given a model generated from config.
        When model_dump() is called.
        Then data is properly serialized.
        """
        config = Mock()
        config.output_schema = {"title": "string", "count": "int"}

        Model = IFConfigFactory.get_output_schema(config)
        instance = Model(title="Test", count=42)
        data = instance.model_dump()

        assert data == {"title": "Test", "count": 42}

    def test_schema_model_json_schema(self):
        """
        GWT:
        Given a model generated from config.
        When model_json_schema() is called.
        Then a valid JSON schema is returned.
        """
        config = Mock()
        config.output_schema = {"name": "string", "age": "int?"}

        Model = IFConfigFactory.get_output_schema(config)
        json_schema = Model.model_json_schema()

        assert "properties" in json_schema
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]

    def test_schema_model_validation(self):
        """
        GWT:
        Given a model generated from config.
        When invalid data is provided.
        Then ValidationError is raised.
        """
        from pydantic import ValidationError

        config = Mock()
        config.output_schema = {"count": "int"}

        Model = IFConfigFactory.get_output_schema(config)

        with pytest.raises(ValidationError):
            Model(count="not a number")

    def test_real_config_without_output_schema(self):
        """
        GWT:
        Given a real Config object (no output_schema by default).
        When get_output_schema() is called.
        Then None is returned gracefully.
        """
        config = Config()
        result = IFConfigFactory.get_output_schema(config)
        assert result is None

    def test_multiple_schema_generation(self):
        """
        GWT:
        Given multiple calls to get_output_schema_from_dict.
        When different schemas are provided.
        Then each generates a unique model.
        """
        Model1 = IFConfigFactory.get_output_schema_from_dict(
            {"field1": "string"}, "Model1"
        )
        Model2 = IFConfigFactory.get_output_schema_from_dict(
            {"field2": "int"}, "Model2"
        )

        assert Model1.__name__ == "Model1"
        assert Model2.__name__ == "Model2"
        assert Model1 is not Model2

        instance1 = Model1(field1="test")
        instance2 = Model2(field2=42)

        assert instance1.field1 == "test"
        assert instance2.field2 == 42


# =============================================================================
# TESTS - CREATE_EXTRACTION_MODEL
# =============================================================================


class TestCreateExtractionModel:
    """
    Dynamic Schema Factory tests.

    Tests verify that YAML extraction schemas can be converted into
    live Pydantic BaseModel classes at runtime.
    """

    def test_create_extraction_model_basic(self):
        """
        GWT:
        Given a schema dict from a blueprint (e.g., {"name": "str"}).
        When IFConfigFactory.create_extraction_model() is called.
        Then it returns a Pydantic class ready for use with Instructor.
        """
        schema_def = {"name": "str"}
        Model = IFConfigFactory.create_extraction_model(schema_def, "BasicEntity")

        assert Model is not None
        assert issubclass(Model, BaseModel)
        assert Model.__name__ == "BasicEntity"

        instance = Model(name="test")
        assert instance.name == "test"

    def test_create_extraction_model_all_basic_types(self):
        """
        GWT:
        Given a schema with str, int, float, bool types.
        When create_extraction_model() is called.
        Then all types are properly mapped.
        """
        schema_def = {"name": "str", "count": "int", "price": "float", "active": "bool"}
        Model = IFConfigFactory.create_extraction_model(schema_def, "MultiTypeEntity")

        instance = Model(name="test", count=42, price=19.99, active=True)

        assert instance.name == "test"
        assert instance.count == 42
        assert instance.price == 19.99
        assert instance.active is True

    def test_create_extraction_model_list_type(self):
        """
        GWT:
        Given a schema with list type.
        When create_extraction_model() is called.
        Then list types are properly handled with recursive mapping.
        """
        schema_def = {"tags": "list[str]", "scores": "list[int]"}
        Model = IFConfigFactory.create_extraction_model(schema_def, "ListEntity")

        instance = Model(tags=["a", "b", "c"], scores=[1, 2, 3])

        assert instance.tags == ["a", "b", "c"]
        assert instance.scores == [1, 2, 3]

    def test_create_extraction_model_unknown_type_raises_blueprint_error(self):
        """
        GWT:
        Given a YAML schema with an unknown type.
        When create_extraction_model() is called.
        Then BlueprintError is raised (AC: Negative Test).
        """
        schema_def = {"field": "unknown_type"}

        with pytest.raises(BlueprintError, match="Unknown type"):
            IFConfigFactory.create_extraction_model(schema_def)

    def test_create_extraction_model_assert_dict_input(self):
        """
        GWT:
        Given input that is not a dictionary.
        When create_extraction_model() is called.
        Then AssertionError is raised (JPL Rule #5).
        """
        with pytest.raises(AssertionError, match="must be a dictionary"):
            IFConfigFactory.create_extraction_model("not a dict")

    def test_create_extraction_model_assert_none_input(self):
        """
        GWT:
        Given None input.
        When create_extraction_model() is called.
        Then AssertionError is raised.
        """
        with pytest.raises(AssertionError, match="cannot be None"):
            IFConfigFactory.create_extraction_model(None)

    def test_create_extraction_model_empty_dict_raises(self):
        """
        GWT:
        Given an empty schema dict.
        When create_extraction_model() is called.
        Then BlueprintError is raised.
        """
        with pytest.raises(BlueprintError, match="empty"):
            IFConfigFactory.create_extraction_model({})

    def test_create_extraction_model_empty_name_rejected(self):
        """
        GWT:
        Given an empty model name.
        When create_extraction_model() is called.
        Then AssertionError is raised.
        """
        with pytest.raises(AssertionError):
            IFConfigFactory.create_extraction_model({"field": "str"}, model_name="")

    def test_create_extraction_model_name_over_limit(self):
        """
        GWT:
        Given a model name exceeding MAX_SCHEMA_NAME_LENGTH.
        When create_extraction_model() is called.
        Then AssertionError is raised.
        """
        long_name = "A" * (MAX_SCHEMA_NAME_LENGTH + 1)

        with pytest.raises(AssertionError, match="max length"):
            IFConfigFactory.create_extraction_model(
                {"field": "str"}, model_name=long_name
            )

    def test_create_extraction_model_default_name(self):
        """
        GWT:
        Given no model name specified.
        When create_extraction_model() is called.
        Then default name 'ExtractionModel' is used.
        """
        Model = IFConfigFactory.create_extraction_model({"field": "str"})
        assert Model.__name__ == "ExtractionModel"

    def test_create_extraction_model_optional_fields(self):
        """
        GWT:
        Given a schema with optional fields (using ? suffix).
        When create_extraction_model() is called.
        Then optional fields can be omitted.
        """
        schema_def = {"required_field": "str", "optional_field": "int?"}
        Model = IFConfigFactory.create_extraction_model(schema_def, "OptionalEntity")

        instance = Model(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field is None

    def test_create_extraction_model_nested_object(self):
        """
        GWT:
        Given a schema with nested objects.
        When create_extraction_model() is called.
        Then nested models are properly generated.
        """
        schema_def = {"name": "str", "location": {"city": "str", "zip": "str"}}
        Model = IFConfigFactory.create_extraction_model(schema_def, "NestedEntity")

        instance = Model(name="Test", location={"city": "Boston", "zip": "02101"})

        assert instance.name == "Test"
        assert instance.location.city == "Boston"
        assert instance.location.zip == "02101"

    def test_create_extraction_model_ready_for_instructor(self):
        """
        GWT:
        Given a generated model.
        When model_json_schema() is called.
        Then a valid JSON schema is returned (compatible with Instructor).
        """
        schema_def = {"title": "str", "summary": "str", "score": "int"}
        Model = IFConfigFactory.create_extraction_model(schema_def, "InstructorModel")

        json_schema = Model.model_json_schema()

        assert "properties" in json_schema
        assert "title" in json_schema["properties"]
        assert "summary" in json_schema["properties"]
        assert "score" in json_schema["properties"]

    def test_create_extraction_model_method_length(self):
        """
        GWT:
        Given the IFConfigFactory.create_extraction_model() method.
        When its source code is inspected.
        Then it has fewer than 60 lines (JPL Rule #4).
        """
        source_lines = inspect.getsourcelines(IFConfigFactory.create_extraction_model)[
            0
        ]
        assert (
            len(source_lines) < 60
        ), f"create_extraction_model() has {len(source_lines)} lines"

    def test_create_extraction_model_has_type_hints(self):
        """
        GWT:
        Given the IFConfigFactory.create_extraction_model() method.
        When type hints are inspected.
        Then return type is explicitly specified (JPL Rule #7).
        """
        hints = get_type_hints(IFConfigFactory.create_extraction_model)
        assert "return" in hints
