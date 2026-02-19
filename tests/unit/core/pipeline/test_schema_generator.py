"""
Tests for SchemaGenerator.

Programmatic Output Schema Definition.
Verifies dynamic Pydantic model generation from YAML schema definitions.

Test Categories:
- GWT (Given-When-Then) behavioral specifications
- JPL Power of Ten rule compliance verification
- Boundary condition testing
- Error handling and propagation
"""

import inspect
import pytest
from datetime import date, datetime
from typing import get_type_hints

from pydantic import BaseModel, ValidationError

from ingestforge.core.pipeline.schema_generator import (
    SchemaGenerator,
    SchemaDefinitionError,
    generate_schema_from_yaml,
    attach_schema_to_artifact,
    MAX_SCHEMA_FIELDS,
    MAX_FIELD_NAME_LENGTH,
    MAX_NESTED_DEPTH,
    TYPE_MAPPINGS,
)


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLRule2FixedUpperBounds:
    """
    JPL Rule #2: All loops must have a fixed upper-bound.

    For data structures, this means fixed maximum sizes are enforced.
    Tests verify that constants are defined and enforced.
    """

    def test_max_schema_fields_constant_defined(self):
        """
        GWT:
        Given the SchemaGenerator module.
        When MAX_SCHEMA_FIELDS is accessed.
        Then it is a positive integer with reasonable bound.
        """
        assert isinstance(MAX_SCHEMA_FIELDS, int)
        assert MAX_SCHEMA_FIELDS > 0
        assert MAX_SCHEMA_FIELDS <= 256  # Reasonable upper bound

    def test_max_field_name_length_constant_defined(self):
        """
        GWT:
        Given the SchemaGenerator module.
        When MAX_FIELD_NAME_LENGTH is accessed.
        Then it is a positive integer with reasonable bound.
        """
        assert isinstance(MAX_FIELD_NAME_LENGTH, int)
        assert MAX_FIELD_NAME_LENGTH > 0
        assert MAX_FIELD_NAME_LENGTH <= 512

    def test_max_nested_depth_constant_defined(self):
        """
        GWT:
        Given the SchemaGenerator module.
        When MAX_NESTED_DEPTH is accessed.
        Then it is a positive integer with reasonable bound.
        """
        assert isinstance(MAX_NESTED_DEPTH, int)
        assert MAX_NESTED_DEPTH > 0
        assert MAX_NESTED_DEPTH <= 10

    def test_schema_field_limit_enforced_at_boundary(self):
        """
        GWT:
        Given a schema with exactly MAX_SCHEMA_FIELDS fields.
        When generate() is called.
        Then the schema is accepted (boundary case).
        """
        generator = SchemaGenerator()
        schema_def = {f"field_{i}": "string" for i in range(MAX_SCHEMA_FIELDS)}

        # Should succeed at exact limit
        Model = generator.generate("BoundaryModel", schema_def)
        assert Model is not None

    def test_schema_field_limit_enforced_over_boundary(self):
        """
        GWT:
        Given a schema with MAX_SCHEMA_FIELDS + 1 fields.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {f"field_{i}": "string" for i in range(MAX_SCHEMA_FIELDS + 1)}

        with pytest.raises(SchemaDefinitionError, match="exceeds maximum"):
            generator.generate("OverLimitModel", schema_def)

    def test_field_name_length_at_boundary(self):
        """
        GWT:
        Given a field name with exactly MAX_FIELD_NAME_LENGTH characters.
        When generate() is called.
        Then the schema is accepted.
        """
        generator = SchemaGenerator()
        field_name = "a" * MAX_FIELD_NAME_LENGTH
        schema_def = {field_name: "string"}

        Model = generator.generate("BoundaryNameModel", schema_def)
        assert Model is not None

    def test_field_name_length_over_boundary(self):
        """
        GWT:
        Given a field name exceeding MAX_FIELD_NAME_LENGTH.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        field_name = "a" * (MAX_FIELD_NAME_LENGTH + 1)
        schema_def = {field_name: "string"}

        with pytest.raises(SchemaDefinitionError, match="exceeds maximum length"):
            generator.generate("OverLengthModel", schema_def)

    def test_nesting_depth_at_boundary(self):
        """
        GWT:
        Given a schema nested to exactly MAX_NESTED_DEPTH levels.
        When generate() is called.
        Then the schema is accepted.
        """
        generator = SchemaGenerator()

        # Build nested schema to MAX_NESTED_DEPTH
        schema_def: dict = {"value": "string"}
        for i in range(MAX_NESTED_DEPTH - 1):
            schema_def = {f"level_{i}": schema_def}

        Model = generator.generate("DeepModel", schema_def)
        assert Model is not None

    def test_nesting_depth_over_boundary(self):
        """
        GWT:
        Given a schema nested beyond MAX_NESTED_DEPTH levels.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()

        # Build nested schema exceeding MAX_NESTED_DEPTH
        schema_def: dict = {"value": "string"}
        for i in range(MAX_NESTED_DEPTH + 1):
            schema_def = {f"level_{i}": schema_def}

        with pytest.raises(SchemaDefinitionError, match="exceeds maximum depth"):
            generator.generate("TooDeepModel", schema_def)


class TestJPLRule4FunctionLength:
    """
    JPL Rule #4: No function should be longer than 60 lines.

    Tests verify that all public methods comply with this rule.
    """

    def test_generate_method_length(self):
        """
        GWT:
        Given the SchemaGenerator.generate() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(SchemaGenerator.generate)[0]
        assert len(source_lines) < 60, f"generate() has {len(source_lines)} lines"

    def test_parse_fields_method_length(self):
        """
        GWT:
        Given the SchemaGenerator._parse_fields() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(SchemaGenerator._parse_fields)[0]
        assert len(source_lines) < 60, f"_parse_fields() has {len(source_lines)} lines"

    def test_parse_type_method_length(self):
        """
        GWT:
        Given the SchemaGenerator._parse_type() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(SchemaGenerator._parse_type)[0]
        assert len(source_lines) < 60, f"_parse_type() has {len(source_lines)} lines"

    def test_parse_dict_type_method_length(self):
        """
        GWT:
        Given the SchemaGenerator._parse_dict_type() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(SchemaGenerator._parse_dict_type)[0]
        assert (
            len(source_lines) < 60
        ), f"_parse_dict_type() has {len(source_lines)} lines"

    def test_parse_string_type_method_length(self):
        """
        GWT:
        Given the SchemaGenerator._parse_string_type() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(SchemaGenerator._parse_string_type)[0]
        assert (
            len(source_lines) < 60
        ), f"_parse_string_type() has {len(source_lines)} lines"


class TestJPLRule5AssertionDensity:
    """
    JPL Rule #5: Assertions should be used liberally.

    Tests verify that invalid inputs are caught with meaningful errors.
    """

    def test_none_model_name_rejected(self):
        """
        GWT:
        Given a None model name.
        When generate() is called.
        Then an appropriate error is raised.
        """
        generator = SchemaGenerator()
        with pytest.raises((SchemaDefinitionError, TypeError)):
            generator.generate(None, {"field": "string"})  # type: ignore

    def test_empty_model_name_rejected(self):
        """
        GWT:
        Given an empty string model name.
        When generate() is called.
        Then SchemaDefinitionError is raised with clear message.
        """
        generator = SchemaGenerator()
        with pytest.raises(SchemaDefinitionError, match="cannot be empty"):
            generator.generate("", {"field": "string"})

    def test_none_schema_rejected(self):
        """
        GWT:
        Given a None schema definition.
        When generate() is called.
        Then an appropriate error is raised.
        """
        generator = SchemaGenerator()
        with pytest.raises((SchemaDefinitionError, TypeError)):
            generator.generate("Model", None)  # type: ignore

    def test_empty_schema_rejected(self):
        """
        GWT:
        Given an empty schema definition.
        When generate() is called.
        Then SchemaDefinitionError is raised with clear message.
        """
        generator = SchemaGenerator()
        with pytest.raises(SchemaDefinitionError, match="cannot be empty"):
            generator.generate("Model", {})

    def test_list_schema_rejected(self):
        """
        GWT:
        Given a list instead of dict as schema definition.
        When generate() is called.
        Then AssertionError is raised.

        AC / JPL Rule #5: Assert input schema is a valid dictionary.
        """
        generator = SchemaGenerator()
        with pytest.raises(AssertionError, match="must be a dictionary"):
            generator.generate("Model", ["field1", "field2"])  # type: ignore

    def test_non_string_model_name_rejected(self):
        """
        GWT:
        Given a non-string model name.
        When generate() is called.
        Then AssertionError is raised.

        AC / JPL Rule #5: Assert model_name is a string.
        """
        generator = SchemaGenerator()
        with pytest.raises(AssertionError, match="must be a string"):
            generator.generate(123, {"field": "string"})  # type: ignore


class TestJPLRule7ExplicitTypes:
    """
    JPL Rule #7: Return values and types must be explicit.

    Tests verify that all public methods have type hints and return expected types.
    """

    def test_generate_has_type_hints(self):
        """
        GWT:
        Given the SchemaGenerator.generate() method.
        When type hints are inspected.
        Then return type is explicitly Type[BaseModel].
        """
        hints = get_type_hints(SchemaGenerator.generate)
        assert "return" in hints
        # Return type should be Type[BaseModel]
        assert hints["return"] is not None

    def test_generate_returns_base_model_subclass(self):
        """
        GWT:
        Given a valid schema.
        When generate() is called.
        Then returned class is a BaseModel subclass.
        """
        generator = SchemaGenerator()
        Model = generator.generate("TypedModel", {"field": "string"})

        assert isinstance(Model, type)
        assert issubclass(Model, BaseModel)

    def test_get_model_returns_optional(self):
        """
        GWT:
        Given a non-existent model name.
        When get_model() is called.
        Then None is returned (not an exception).
        """
        generator = SchemaGenerator()
        result = generator.get_model("NonExistent")
        assert result is None

    def test_list_models_returns_list(self):
        """
        GWT:
        Given a SchemaGenerator with generated models.
        When list_models() is called.
        Then a List[str] is returned.
        """
        generator = SchemaGenerator()
        generator.generate("Model1", {"a": "string"})

        result = generator.list_models()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)


class TestJPLRule9TypeSafety:
    """
    JPL Rule #9: Data integrity and type safety.

    Tests verify that type mappings are complete and correct.
    """

    def test_type_mappings_complete(self):
        """
        GWT:
        Given the TYPE_MAPPINGS constant.
        When inspected.
        Then all documented types are present.
        """
        expected_types = {
            "string",
            "str",
            "text",
            "int",
            "integer",
            "float",
            "number",
            "bool",
            "boolean",
            "date",
            "datetime",
            "any",
        }
        assert expected_types.issubset(set(TYPE_MAPPINGS.keys()))

    def test_type_mappings_values_are_types(self):
        """
        GWT:
        Given the TYPE_MAPPINGS constant.
        When values are inspected.
        Then all values are valid Python types.
        """
        for type_name, python_type in TYPE_MAPPINGS.items():
            # 'Any' is a special form, not a regular type
            if type_name != "any":
                assert isinstance(python_type, type), f"{type_name} maps to non-type"

    def test_case_insensitive_type_parsing(self):
        """
        GWT:
        Given type names in various cases.
        When generate() is called.
        Then types are recognized case-insensitively.
        """
        generator = SchemaGenerator()
        schema_def = {
            "upper": "STRING",
            "mixed": "StRiNg",
            "lower": "string",
        }
        Model = generator.generate("CaseModel", schema_def)

        instance = Model(upper="a", mixed="b", lower="c")
        assert instance.upper == "a"
        assert instance.mixed == "b"
        assert instance.lower == "c"


# =============================================================================
# GWT BEHAVIORAL TESTS - BASIC TYPES
# =============================================================================


class TestSchemaGeneratorBasicTypes:
    """Test basic type parsing and model generation."""

    def test_generate_string_field(self):
        """
        GWT:
        Given a schema with a string field.
        When generate() is called.
        Then the model accepts string values for that field.
        """
        generator = SchemaGenerator()
        schema_def = {"name": "string"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model(name="Alice")
        assert instance.name == "Alice"

    def test_generate_int_field(self):
        """
        GWT:
        Given a schema with an int field.
        When generate() is called.
        Then the model accepts integer values for that field.
        """
        generator = SchemaGenerator()
        schema_def = {"count": "int"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model(count=42)
        assert instance.count == 42

    def test_generate_float_field(self):
        """
        GWT:
        Given a schema with a float field.
        When generate() is called.
        Then the model accepts float values for that field.
        """
        generator = SchemaGenerator()
        schema_def = {"price": "float"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model(price=19.99)
        assert instance.price == 19.99

    def test_generate_bool_field(self):
        """
        GWT:
        Given a schema with a bool field.
        When generate() is called.
        Then the model accepts boolean values for that field.
        """
        generator = SchemaGenerator()
        schema_def = {"active": "bool"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model(active=True)
        assert instance.active is True

    def test_generate_date_field(self):
        """
        GWT:
        Given a schema with a date field.
        When generate() is called.
        Then the model accepts date values for that field.
        """
        generator = SchemaGenerator()
        schema_def = {"published": "date"}
        Model = generator.generate("TestModel", schema_def)

        today = date.today()
        instance = Model(published=today)
        assert instance.published == today

    def test_generate_datetime_field(self):
        """
        GWT:
        Given a schema with a datetime field.
        When generate() is called.
        Then the model accepts datetime values for that field.
        """
        generator = SchemaGenerator()
        schema_def = {"created_at": "datetime"}
        Model = generator.generate("TestModel", schema_def)

        now = datetime.now()
        instance = Model(created_at=now)
        assert instance.created_at == now

    def test_type_aliases(self):
        """
        GWT:
        Given type aliases like 'text', 'integer', 'number', 'boolean'.
        When generate() is called.
        Then they map to the correct Python types.
        """
        generator = SchemaGenerator()
        schema_def = {
            "text_field": "text",
            "integer_field": "integer",
            "number_field": "number",
            "boolean_field": "boolean",
        }
        Model = generator.generate("TestModel", schema_def)

        instance = Model(
            text_field="hello",
            integer_field=10,
            number_field=3.14,
            boolean_field=False,
        )
        assert instance.text_field == "hello"
        assert instance.integer_field == 10
        assert instance.number_field == 3.14
        assert instance.boolean_field is False

    def test_any_type_field(self):
        """
        GWT:
        Given a schema with an 'any' type field.
        When generate() is called.
        Then the model accepts any value for that field.
        """
        generator = SchemaGenerator()
        schema_def = {"data": "any"}
        Model = generator.generate("FlexModel", schema_def)

        # Should accept various types
        instance1 = Model(data="string")
        assert instance1.data == "string"

        instance2 = Model(data=123)
        assert instance2.data == 123

        instance3 = Model(data={"nested": "dict"})
        assert instance3.data == {"nested": "dict"}


# =============================================================================
# GWT BEHAVIORAL TESTS - LIST TYPES
# =============================================================================


class TestSchemaGeneratorListTypes:
    """Test list type parsing."""

    def test_generate_list_string_field(self):
        """
        GWT:
        Given a schema with list[string] field.
        When generate() is called.
        Then the model accepts a list of strings.
        """
        generator = SchemaGenerator()
        schema_def = {"tags": "list[string]"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model(tags=["a", "b", "c"])
        assert instance.tags == ["a", "b", "c"]

    def test_generate_list_int_field(self):
        """
        GWT:
        Given a schema with list[int] field.
        When generate() is called.
        Then the model accepts a list of integers.
        """
        generator = SchemaGenerator()
        schema_def = {"scores": "list[int]"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model(scores=[100, 90, 85])
        assert instance.scores == [100, 90, 85]

    def test_generate_list_float_field(self):
        """
        GWT:
        Given a schema with list[float] field.
        When generate() is called.
        Then the model accepts a list of floats.
        """
        generator = SchemaGenerator()
        schema_def = {"values": "list[float]"}
        Model = generator.generate("FloatListModel", schema_def)

        instance = Model(values=[1.1, 2.2, 3.3])
        assert instance.values == [1.1, 2.2, 3.3]

    def test_generate_list_bool_field(self):
        """
        GWT:
        Given a schema with list[bool] field.
        When generate() is called.
        Then the model accepts a list of booleans.
        """
        generator = SchemaGenerator()
        schema_def = {"flags": "list[bool]"}
        Model = generator.generate("BoolListModel", schema_def)

        instance = Model(flags=[True, False, True])
        assert instance.flags == [True, False, True]

    def test_empty_list_accepted(self):
        """
        GWT:
        Given a schema with a list field.
        When instantiated with an empty list.
        Then the empty list is accepted.
        """
        generator = SchemaGenerator()
        schema_def = {"items": "list[string]"}
        Model = generator.generate("EmptyListModel", schema_def)

        instance = Model(items=[])
        assert instance.items == []


# =============================================================================
# GWT BEHAVIORAL TESTS - OPTIONAL FIELDS
# =============================================================================


class TestSchemaGeneratorOptionalFields:
    """Test optional field handling."""

    def test_optional_marker_suffix(self):
        """
        GWT:
        Given a schema field ending with '?'.
        When generate() is called.
        Then the field is optional (can be None).
        """
        generator = SchemaGenerator()
        schema_def = {"nickname": "string?"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model()  # No args - optional field should default to None
        assert instance.nickname is None

        instance2 = Model(nickname="Bob")
        assert instance2.nickname == "Bob"

    def test_optional_type_wrapper(self):
        """
        GWT:
        Given a schema with 'optional[type]' syntax.
        When generate() is called.
        Then the field is optional (can be None).
        """
        generator = SchemaGenerator()
        schema_def = {"middle_name": "optional[string]"}
        Model = generator.generate("TestModel", schema_def)

        instance = Model()
        assert instance.middle_name is None

    def test_dict_required_field(self):
        """
        GWT:
        Given a schema with dict notation and required: False.
        When generate() is called.
        Then the field is optional.
        """
        generator = SchemaGenerator()
        schema_def = {"email": {"type": "string", "required": False}}
        Model = generator.generate("TestModel", schema_def)

        instance = Model()
        assert instance.email is None

    def test_dict_required_true(self):
        """
        GWT:
        Given a schema with dict notation and required: True.
        When generate() is called.
        Then the field is required.
        """
        generator = SchemaGenerator()
        schema_def = {"email": {"type": "string", "required": True}}
        Model = generator.generate("RequiredDictModel", schema_def)

        with pytest.raises(ValidationError):
            Model()  # Missing required field

    def test_optional_list_field(self):
        """
        GWT:
        Given a schema with 'list[string]?' field.
        When generate() is called.
        Then the list field is optional.
        """
        generator = SchemaGenerator()
        schema_def = {"tags": "list[string]?"}
        Model = generator.generate("OptionalListModel", schema_def)

        instance = Model()
        assert instance.tags is None

        instance2 = Model(tags=["a", "b"])
        assert instance2.tags == ["a", "b"]

    def test_mixed_required_optional_fields(self):
        """
        GWT:
        Given a schema with both required and optional fields.
        When generate() is called.
        Then only required fields must be provided.
        """
        generator = SchemaGenerator()
        schema_def = {"name": "string", "age": "int?", "email": "string"}
        Model = generator.generate("MixedModel", schema_def)

        # Must provide required fields
        with pytest.raises(ValidationError):
            Model()

        # Can omit optional field
        instance = Model(name="Alice", email="alice@example.com")
        assert instance.name == "Alice"
        assert instance.age is None
        assert instance.email == "alice@example.com"


# =============================================================================
# GWT BEHAVIORAL TESTS - NESTED OBJECTS
# =============================================================================


class TestSchemaGeneratorNestedObjects:
    """Test nested object schema handling."""

    def test_nested_object(self):
        """
        GWT:
        Given a schema with a nested object definition.
        When generate() is called.
        Then the model has a nested Pydantic model.
        """
        generator = SchemaGenerator()
        schema_def = {"author": {"name": "string", "email": "string"}}
        Model = generator.generate("TestModel", schema_def)

        instance = Model(author={"name": "Alice", "email": "alice@example.com"})
        assert instance.author.name == "Alice"
        assert instance.author.email == "alice@example.com"

    def test_deeply_nested_object(self):
        """
        GWT:
        Given a schema with multiple levels of nesting.
        When generate() is called.
        Then all nesting levels are properly generated.
        """
        generator = SchemaGenerator()
        schema_def = {
            "company": {
                "name": "string",
                "address": {"street": "string", "city": "string"},
            }
        }
        Model = generator.generate("DeepNestedModel", schema_def)

        instance = Model(
            company={
                "name": "Acme",
                "address": {"street": "123 Main St", "city": "Boston"},
            }
        )
        assert instance.company.name == "Acme"
        assert instance.company.address.street == "123 Main St"
        assert instance.company.address.city == "Boston"

    def test_nested_object_with_list(self):
        """
        GWT:
        Given a nested object containing a list field.
        When generate() is called.
        Then the nested list is properly typed.
        """
        generator = SchemaGenerator()
        schema_def = {"team": {"name": "string", "members": "list[string]"}}
        Model = generator.generate("TeamModel", schema_def)

        instance = Model(team={"name": "Engineering", "members": ["Alice", "Bob"]})
        assert instance.team.name == "Engineering"
        assert instance.team.members == ["Alice", "Bob"]

    def test_nested_object_cached(self):
        """
        GWT:
        Given a schema with nested objects.
        When generate() is called.
        Then nested models are also cached.
        """
        generator = SchemaGenerator()
        schema_def = {"author": {"name": "string"}}
        generator.generate("CachedNestedModel", schema_def)

        # Nested model should be in cache
        model_names = generator.list_models()
        assert "CachedNestedModel" in model_names
        # Nested models have generated names
        assert any("NestedModel" in name for name in model_names)


# =============================================================================
# GWT BEHAVIORAL TESTS - VALIDATION & ERRORS
# =============================================================================


class TestSchemaGeneratorValidation:
    """Test schema validation and error handling."""

    def test_empty_model_name_raises(self):
        """
        GWT:
        Given an empty model name.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        with pytest.raises(SchemaDefinitionError, match="Model name cannot be empty"):
            generator.generate("", {"field": "string"})

    def test_empty_schema_raises(self):
        """
        GWT:
        Given an empty schema definition.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        with pytest.raises(
            SchemaDefinitionError, match="Schema definition cannot be empty"
        ):
            generator.generate("TestModel", {})

    def test_too_many_fields_raises(self):
        """
        GWT:
        Given a schema exceeding MAX_SCHEMA_FIELDS.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {f"field_{i}": "string" for i in range(MAX_SCHEMA_FIELDS + 1)}
        with pytest.raises(SchemaDefinitionError, match="exceeds maximum"):
            generator.generate("TestModel", schema_def)

    def test_long_field_name_raises(self):
        """
        GWT:
        Given a field name exceeding MAX_FIELD_NAME_LENGTH.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        long_name = "a" * (MAX_FIELD_NAME_LENGTH + 1)
        schema_def = {long_name: "string"}
        with pytest.raises(SchemaDefinitionError, match="exceeds maximum length"):
            generator.generate("TestModel", schema_def)

    def test_unknown_type_raises(self):
        """
        GWT:
        Given an unknown type name.
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {"field": "unknown_type"}
        with pytest.raises(SchemaDefinitionError, match="Unknown type"):
            generator.generate("TestModel", schema_def)

    def test_invalid_type_def_raises(self):
        """
        GWT:
        Given an invalid type definition (not str or dict).
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {"field": 12345}  # Invalid - not a string or dict
        with pytest.raises(SchemaDefinitionError, match="Invalid type definition"):
            generator.generate("TestModel", schema_def)

    def test_invalid_list_syntax_raises(self):
        """
        GWT:
        Given malformed list syntax (missing closing bracket).
        When generate() is called.
        Then SchemaDefinitionError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {"items": "list[string"}  # Missing ]
        with pytest.raises(SchemaDefinitionError, match="Unknown type"):
            generator.generate("BadListModel", schema_def)

    def test_whitespace_in_type_name_handled(self):
        """
        GWT:
        Given a type name with surrounding whitespace.
        When generate() is called.
        Then the type is recognized after stripping.
        """
        generator = SchemaGenerator()
        schema_def = {"field": "  string  "}
        Model = generator.generate("WhitespaceModel", schema_def)

        instance = Model(field="test")
        assert instance.field == "test"


# =============================================================================
# GWT BEHAVIORAL TESTS - CACHING
# =============================================================================


class TestSchemaGeneratorCaching:
    """Test model caching functionality."""

    def test_model_cached_after_generation(self):
        """
        GWT:
        Given a generated model.
        When get_model() is called.
        Then the same model class is returned.
        """
        generator = SchemaGenerator()
        Model = generator.generate("CachedModel", {"field": "string"})

        cached = generator.get_model("CachedModel")
        assert cached is Model

    def test_list_models(self):
        """
        GWT:
        Given multiple generated models.
        When list_models() is called.
        Then all model names are returned.
        """
        generator = SchemaGenerator()
        generator.generate("Model1", {"a": "string"})
        generator.generate("Model2", {"b": "int"})

        names = generator.list_models()
        assert "Model1" in names
        assert "Model2" in names

    def test_clear_cache(self):
        """
        GWT:
        Given a cached model.
        When clear_cache() is called.
        Then the model is no longer retrievable.
        """
        generator = SchemaGenerator()
        generator.generate("TempModel", {"field": "string"})
        assert generator.get_model("TempModel") is not None

        generator.clear_cache()
        assert generator.get_model("TempModel") is None

    def test_cache_isolation_between_instances(self):
        """
        GWT:
        Given two SchemaGenerator instances.
        When models are generated.
        Then each instance has its own cache.
        """
        gen1 = SchemaGenerator()
        gen2 = SchemaGenerator()

        gen1.generate("ModelA", {"field": "string"})

        assert gen1.get_model("ModelA") is not None
        assert gen2.get_model("ModelA") is None

    def test_same_name_different_schema_overwrites(self):
        """
        GWT:
        Given a model name that was already generated.
        When generate() is called with a different schema.
        Then the cached model is replaced.
        """
        generator = SchemaGenerator()

        Model1 = generator.generate("DuplicateName", {"field1": "string"})
        Model2 = generator.generate("DuplicateName", {"field2": "int"})

        cached = generator.get_model("DuplicateName")
        assert cached is Model2
        assert cached is not Model1


# =============================================================================
# GWT BEHAVIORAL TESTS - CONVENIENCE FUNCTIONS
# =============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_generate_schema_from_yaml(self):
        """
        GWT:
        Given a YAML-style dict schema.
        When generate_schema_from_yaml() is called.
        Then a valid Pydantic model is returned.
        """
        schema = {"title": "string", "count": "int"}
        Model = generate_schema_from_yaml("Article", schema)

        instance = Model(title="Test", count=5)
        assert instance.title == "Test"
        assert instance.count == 5

    def test_attach_schema_to_artifact(self):
        """
        GWT:
        Given artifact metadata and a schema model.
        When attach_schema_to_artifact() is called.
        Then metadata includes schema info.
        """
        schema_def = {"name": "string"}
        Model = generate_schema_from_yaml("TestSchema", schema_def)

        metadata = {"existing_key": "value"}
        updated = attach_schema_to_artifact(metadata, Model)

        assert "existing_key" in updated
        assert "_target_schema" in updated
        assert updated["_target_schema"]["target_model_name"] == "TestSchema"
        assert "target_model_schema" in updated["_target_schema"]

    def test_attach_schema_preserves_original_metadata(self):
        """
        GWT:
        Given existing artifact metadata.
        When attach_schema_to_artifact() is called.
        Then original metadata is not mutated.
        """
        schema_def = {"field": "string"}
        Model = generate_schema_from_yaml("PreserveTest", schema_def)

        original = {"key1": "value1", "key2": "value2"}
        updated = attach_schema_to_artifact(original, Model)

        # Original should be unchanged
        assert "_target_schema" not in original
        # Updated should have schema
        assert "_target_schema" in updated

    def test_attach_schema_json_schema_valid(self):
        """
        GWT:
        Given a generated model.
        When attach_schema_to_artifact() is called.
        Then the JSON schema is a valid dict.
        """
        schema_def = {"name": "string", "age": "int"}
        Model = generate_schema_from_yaml("JsonSchemaTest", schema_def)

        updated = attach_schema_to_artifact({}, Model)
        json_schema = updated["_target_schema"]["target_model_schema"]

        assert isinstance(json_schema, dict)
        assert "properties" in json_schema
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]


# =============================================================================
# GWT BEHAVIORAL TESTS - MODEL VALIDATION
# =============================================================================


class TestModelValidation:
    """Test that generated models properly validate data."""

    def test_required_field_validation(self):
        """
        GWT:
        Given a schema with a required field.
        When instantiating without that field.
        Then ValidationError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {"required_field": "string"}
        Model = generator.generate("StrictModel", schema_def)

        with pytest.raises(ValidationError):
            Model()  # Missing required field

    def test_type_validation(self):
        """
        GWT:
        Given a schema with an int field.
        When instantiating with a non-coercible value.
        Then ValidationError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {"count": "int"}
        Model = generator.generate("TypedModel", schema_def)

        with pytest.raises(ValidationError):
            Model(count="not_a_number")

    def test_date_validation(self):
        """
        GWT:
        Given a schema with a date field.
        When instantiating with an invalid date string.
        Then ValidationError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {"published": "date"}
        Model = generator.generate("DateModel", schema_def)

        with pytest.raises(ValidationError):
            Model(published="not-a-date")

    def test_list_element_validation(self):
        """
        GWT:
        Given a schema with list[int] field.
        When instantiating with non-integer elements.
        Then ValidationError is raised.
        """
        generator = SchemaGenerator()
        schema_def = {"numbers": "list[int]"}
        Model = generator.generate("ListValidationModel", schema_def)

        with pytest.raises(ValidationError):
            Model(numbers=["not", "integers"])

    def test_bool_coercion(self):
        """
        GWT:
        Given a schema with a bool field.
        When instantiating with truthy/falsy values.
        Then values are coerced to boolean.
        """
        generator = SchemaGenerator()
        schema_def = {"flag": "bool"}
        Model = generator.generate("BoolCoerceModel", schema_def)

        instance = Model(flag=1)
        assert instance.flag is True

        instance2 = Model(flag=0)
        assert instance2.flag is False


# =============================================================================
# GWT BEHAVIORAL TESTS - BASE CLASS CUSTOMIZATION
# =============================================================================


class TestBaseClassCustomization:
    """Test custom base class support."""

    def test_custom_base_class(self):
        """
        GWT:
        Given a custom base class.
        When generate() is called with that base class.
        Then the generated model inherits from it.
        """

        class CustomBase(BaseModel):
            def custom_method(self) -> str:
                return "custom"

        generator = SchemaGenerator()
        Model = generator.generate(
            "CustomModel", {"field": "string"}, base_class=CustomBase
        )

        instance = Model(field="test")
        assert instance.custom_method() == "custom"
        assert isinstance(instance, CustomBase)

    def test_default_base_class(self):
        """
        GWT:
        Given no base class specified.
        When generate() is called.
        Then the model inherits from BaseModel.
        """
        generator = SchemaGenerator()
        Model = generator.generate("DefaultBaseModel", {"field": "string"})

        instance = Model(field="test")
        assert isinstance(instance, BaseModel)


# =============================================================================
# GWT BEHAVIORAL TESTS - INITIALIZATION OPTIONS
# =============================================================================


class TestInitializationOptions:
    """Test SchemaGenerator initialization options."""

    def test_allow_optional_default(self):
        """
        GWT:
        Given SchemaGenerator with default settings.
        When allow_optional is checked.
        Then it defaults to True.
        """
        generator = SchemaGenerator()
        assert generator._allow_optional is True

    def test_allow_optional_false(self):
        """
        GWT:
        Given SchemaGenerator with allow_optional=False.
        When a schema is generated.
        Then the generator is initialized correctly.
        """
        generator = SchemaGenerator(allow_optional=False)
        assert generator._allow_optional is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complex_schema(self):
        """
        GWT:
        Given a complex schema with multiple types and nesting.
        When generate() is called.
        Then the model handles all field types correctly.
        """
        generator = SchemaGenerator()
        schema_def = {
            "title": "string",
            "published": "date",
            "score": "float",
            "is_active": "bool",
            "tags": "list[string]",
            "view_count": "int?",
            "author": {"name": "string", "email": "string?"},
        }
        Model = generator.generate("ComplexModel", schema_def)

        instance = Model(
            title="Test Article",
            published=date.today(),
            score=4.5,
            is_active=True,
            tags=["tech", "ai"],
            author={"name": "Alice"},
        )

        assert instance.title == "Test Article"
        assert instance.published == date.today()
        assert instance.score == 4.5
        assert instance.is_active is True
        assert instance.tags == ["tech", "ai"]
        assert instance.view_count is None
        assert instance.author.name == "Alice"
        assert instance.author.email is None

    def test_model_serialization(self):
        """
        GWT:
        Given a generated model instance.
        When model_dump() is called.
        Then the data is properly serialized.
        """
        generator = SchemaGenerator()
        schema_def = {"name": "string", "count": "int", "tags": "list[string]"}
        Model = generator.generate("SerializeModel", schema_def)

        instance = Model(name="Test", count=42, tags=["a", "b"])
        data = instance.model_dump()

        assert data == {"name": "Test", "count": 42, "tags": ["a", "b"]}

    def test_model_json_schema_generation(self):
        """
        GWT:
        Given a generated model.
        When model_json_schema() is called.
        Then a valid JSON schema is returned.
        """
        generator = SchemaGenerator()
        schema_def = {"name": "string", "age": "int?"}
        Model = generator.generate("JsonSchemaModel", schema_def)

        json_schema = Model.model_json_schema()

        assert "properties" in json_schema
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]

    def test_model_from_dict(self):
        """
        GWT:
        Given a dictionary matching the schema.
        When model_validate() is called.
        Then the model is properly instantiated.
        """
        generator = SchemaGenerator()
        schema_def = {"name": "string", "score": "int"}
        Model = generator.generate("FromDictModel", schema_def)

        data = {"name": "Test", "score": 100}
        instance = Model.model_validate(data)

        assert instance.name == "Test"
        assert instance.score == 100
