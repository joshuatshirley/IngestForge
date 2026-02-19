"""Tests for Schema-Driven Table Component.

Schema-Driven Table Component implementation tests.
Tests dynamic columns, type-aware rendering, validation, and sorting."""

from __future__ import annotations

from datetime import datetime

import pytest

from ingestforge.viz.table_schema import (
    CellFormatter,
    ColumnDefinition,
    FieldType,
    SortDirection,
    TableDataSource,
    TableSchema,
    ValidationError,
    ValidationResult,
    create_table_from_blueprint,
    create_table_schema,
    infer_field_type,
    MAX_COLUMNS,
    MAX_ROWS,
)


# ---------------------------------------------------------------------------
# ValidationError Tests
# ---------------------------------------------------------------------------


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_error_creation(self) -> None:
        """Test creating a validation error."""
        error = ValidationError(
            field_name="age",
            message="Value out of range",
            value=150,
            row_index=5,
        )
        assert error.field_name == "age"
        assert error.message == "Value out of range"
        assert error.value == 150
        assert error.row_index == 5

    def test_error_to_dict(self) -> None:
        """Test error dictionary conversion."""
        error = ValidationError(field_name="name", message="Required")
        d = error.to_dict()
        assert d["field_name"] == "name"
        assert d["message"] == "Required"


# ---------------------------------------------------------------------------
# ValidationResult Tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.error_count == 0

    def test_invalid_result(self) -> None:
        """Test creating an invalid result."""
        errors = [
            ValidationError(field_name="f1", message="Error 1"),
            ValidationError(field_name="f2", message="Error 2"),
        ]
        result = ValidationResult(valid=False, errors=errors)
        assert result.valid is False
        assert result.error_count == 2

    def test_result_to_dict(self) -> None:
        """Test result dictionary conversion."""
        result = ValidationResult(valid=True, row_index=10)
        d = result.to_dict()
        assert d["valid"] is True
        assert d["row_index"] == 10


# ---------------------------------------------------------------------------
# ColumnDefinition Tests
# ---------------------------------------------------------------------------


class TestColumnDefinition:
    """Tests for ColumnDefinition dataclass."""

    def test_column_creation(self) -> None:
        """Test creating a column definition."""
        col = ColumnDefinition(
            field_name="user_name",
            display_label="User Name",
            field_type=FieldType.STRING,
            sortable=True,
        )
        assert col.field_name == "user_name"
        assert col.display_label == "User Name"
        assert col.field_type == FieldType.STRING
        assert col.sortable is True

    def test_column_defaults(self) -> None:
        """Test column default values."""
        col = ColumnDefinition(
            field_name="test",
            display_label="Test",
        )
        assert col.field_type == FieldType.STRING
        assert col.sortable is True
        assert col.required is False
        assert col.description is None

    def test_column_to_dict(self) -> None:
        """Test column dictionary conversion."""
        col = ColumnDefinition(
            field_name="age",
            display_label="Age",
            field_type=FieldType.INTEGER,
        )
        d = col.to_dict()
        assert d["field_name"] == "age"
        assert d["field_type"] == "integer"

    def test_column_tooltip_with_description(self) -> None:
        """Test tooltip generation with description."""
        col = ColumnDefinition(
            field_name="price",
            display_label="Price",
            description="Product price in USD",
        )
        tooltip = col.get_tooltip()
        assert "Product price in USD" in tooltip

    def test_column_tooltip_with_constraints(self) -> None:
        """Test tooltip generation with constraints."""
        col = ColumnDefinition(
            field_name="age",
            display_label="Age",
            field_type=FieldType.INTEGER,
            required=True,
            min_value=0,
            max_value=120,
        )
        tooltip = col.get_tooltip()
        assert "Required field" in tooltip
        assert "0" in tooltip
        assert "120" in tooltip


# ---------------------------------------------------------------------------
# CellFormatter Tests
# ---------------------------------------------------------------------------


class TestCellFormatter:
    """Tests for CellFormatter class."""

    def test_format_string(self) -> None:
        """Test string formatting."""
        result = CellFormatter.format_string("Hello World")
        assert result == "Hello World"

    def test_format_string_none(self) -> None:
        """Test string formatting with None."""
        result = CellFormatter.format_value(None, FieldType.STRING)
        assert result == ""

    def test_format_integer(self) -> None:
        """Test integer formatting with thousands separator."""
        result = CellFormatter.format_integer(1234567)
        assert result == "1,234,567"

    def test_format_integer_invalid(self) -> None:
        """Test integer formatting with invalid value."""
        result = CellFormatter.format_integer("not a number")
        assert result == "not a number"

    def test_format_float(self) -> None:
        """Test float formatting with decimal places."""
        result = CellFormatter.format_float(1234.5678)
        assert result == "1,234.57"

    def test_format_boolean_true(self) -> None:
        """Test boolean formatting for True."""
        result = CellFormatter.format_boolean(True)
        assert result == "Yes"

    def test_format_boolean_false(self) -> None:
        """Test boolean formatting for False."""
        result = CellFormatter.format_boolean(False)
        assert result == "No"

    def test_format_boolean_string(self) -> None:
        """Test boolean formatting for string values."""
        assert CellFormatter.format_boolean("true") == "Yes"
        assert CellFormatter.format_boolean("false") == "No"

    def test_format_date_datetime(self) -> None:
        """Test date formatting with datetime object."""
        dt = datetime(2024, 6, 15, 10, 30, 0)
        result = CellFormatter.format_date(dt)
        assert result == "2024-06-15"

    def test_format_date_string(self) -> None:
        """Test date formatting with string."""
        result = CellFormatter.format_date("2024-06-15T10:30:00")
        assert result == "2024-06-15"

    def test_format_datetime(self) -> None:
        """Test datetime formatting."""
        dt = datetime(2024, 6, 15, 10, 30, 45)
        result = CellFormatter.format_datetime(dt)
        assert result == "2024-06-15 10:30:45"

    def test_format_list(self) -> None:
        """Test list formatting."""
        result = CellFormatter.format_list(["a", "b", "c"])
        assert result == "a, b, c"

    def test_format_list_long(self) -> None:
        """Test list formatting with many items."""
        result = CellFormatter.format_list(list(range(20)))
        assert "+10 more" in result

    def test_format_object(self) -> None:
        """Test object formatting."""
        result = CellFormatter.format_object({"key": "value"})
        assert "key: value" in result

    def test_format_value_dispatch(self) -> None:
        """Test format_value dispatches to correct formatter."""
        assert CellFormatter.format_value(123, FieldType.INTEGER) == "123"
        assert CellFormatter.format_value(True, FieldType.BOOLEAN) == "Yes"


# ---------------------------------------------------------------------------
# GWT-1: Dynamic Column Generation Tests
# ---------------------------------------------------------------------------


class TestGWT1DynamicColumnGeneration:
    """GWT-1: Dynamic column generation tests."""

    def test_schema_from_dict(self) -> None:
        """GWT-1: Schema created from dictionary."""
        data = {
            "columns": [
                {"name": "id", "type": "integer", "label": "ID"},
                {"name": "name", "type": "string", "label": "Name"},
            ]
        }
        schema = TableSchema.from_dict(data)

        assert schema.column_count == 2
        assert schema.field_names == ["id", "name"]

    def test_schema_from_blueprint_dict(self) -> None:
        """GWT-1: Schema created from blueprint dictionary."""
        blueprint = {
            "target_schema": {
                "columns": [
                    {"name": "case_id", "type": "string"},
                    {"name": "amount", "type": "float"},
                ]
            }
        }
        schema = TableSchema.from_blueprint(blueprint)

        assert schema.column_count == 2
        assert "case_id" in schema.field_names

    def test_schema_empty_blueprint(self) -> None:
        """GWT-1: Empty schema from blueprint without target_schema."""
        blueprint = {}
        schema = TableSchema.from_blueprint(blueprint)

        assert schema.column_count == 0
        assert schema.schema_name == "default"

    def test_schema_get_column(self) -> None:
        """GWT-1: Get column by field name."""
        schema = create_table_schema(
            [
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
            ]
        )

        col = schema.get_column("name")
        assert col is not None
        assert col.field_type == FieldType.STRING

    def test_schema_get_column_not_found(self) -> None:
        """GWT-1: Get column returns None for unknown field."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])

        col = schema.get_column("unknown")
        assert col is None


# ---------------------------------------------------------------------------
# GWT-2: Type-Aware Rendering Tests
# ---------------------------------------------------------------------------


class TestGWT2TypeAwareRendering:
    """GWT-2: Type-aware rendering tests."""

    def test_data_source_format_cell_string(self) -> None:
        """GWT-2: Format string cell."""
        schema = create_table_schema(
            [
                {"name": "name", "type": "string"},
            ]
        )
        ds = TableDataSource(schema=schema, rows=[{"name": "Alice"}])

        result = ds.format_cell(0, "name")
        assert result == "Alice"

    def test_data_source_format_cell_integer(self) -> None:
        """GWT-2: Format integer cell."""
        schema = create_table_schema(
            [
                {"name": "count", "type": "integer"},
            ]
        )
        ds = TableDataSource(schema=schema, rows=[{"count": 1000}])

        result = ds.format_cell(0, "count")
        assert result == "1,000"

    def test_data_source_format_cell_float(self) -> None:
        """GWT-2: Format float cell."""
        schema = create_table_schema(
            [
                {"name": "price", "type": "float"},
            ]
        )
        ds = TableDataSource(schema=schema, rows=[{"price": 99.999}])

        result = ds.format_cell(0, "price")
        assert result == "100.00"

    def test_data_source_format_cell_boolean(self) -> None:
        """GWT-2: Format boolean cell."""
        schema = create_table_schema(
            [
                {"name": "active", "type": "boolean"},
            ]
        )
        ds = TableDataSource(schema=schema, rows=[{"active": True}])

        result = ds.format_cell(0, "active")
        assert result == "Yes"

    def test_data_source_format_cell_list(self) -> None:
        """GWT-2: Format list cell."""
        schema = create_table_schema(
            [
                {"name": "tags", "type": "list"},
            ]
        )
        ds = TableDataSource(schema=schema, rows=[{"tags": ["a", "b", "c"]}])

        result = ds.format_cell(0, "tags")
        assert "a, b, c" in result


# ---------------------------------------------------------------------------
# GWT-3: Field Metadata Display Tests
# ---------------------------------------------------------------------------


class TestGWT3FieldMetadataDisplay:
    """GWT-3: Field metadata display tests."""

    def test_column_has_description(self) -> None:
        """GWT-3: Column has description."""
        col = ColumnDefinition(
            field_name="ssn",
            display_label="SSN",
            description="Social Security Number",
        )
        assert col.description == "Social Security Number"

    def test_tooltip_includes_description(self) -> None:
        """GWT-3: Tooltip includes description."""
        col = ColumnDefinition(
            field_name="amount",
            display_label="Amount",
            description="Transaction amount in USD",
        )
        tooltip = col.get_tooltip()
        assert "Transaction amount in USD" in tooltip

    def test_tooltip_includes_required(self) -> None:
        """GWT-3: Tooltip shows required status."""
        col = ColumnDefinition(
            field_name="id",
            display_label="ID",
            required=True,
        )
        tooltip = col.get_tooltip()
        assert "Required" in tooltip

    def test_tooltip_includes_range(self) -> None:
        """GWT-3: Tooltip shows numeric range."""
        col = ColumnDefinition(
            field_name="age",
            display_label="Age",
            field_type=FieldType.INTEGER,
            min_value=0,
            max_value=150,
        )
        tooltip = col.get_tooltip()
        assert "0" in tooltip
        assert "150" in tooltip

    def test_tooltip_includes_max_length(self) -> None:
        """GWT-3: Tooltip shows max length."""
        col = ColumnDefinition(
            field_name="code",
            display_label="Code",
            max_length=10,
        )
        tooltip = col.get_tooltip()
        assert "Max length: 10" in tooltip


# ---------------------------------------------------------------------------
# GWT-4: Sortable Columns Tests
# ---------------------------------------------------------------------------


class TestGWT4SortableColumns:
    """GWT-4: Sortable columns tests."""

    def test_sort_by_string_ascending(self) -> None:
        """GWT-4: Sort by string column ascending."""
        schema = create_table_schema(
            [
                {"name": "name", "type": "string", "sortable": True},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"name": "Charlie"}, {"name": "Alice"}, {"name": "Bob"}],
        )

        ds.sort_by("name", SortDirection.ASCENDING)

        assert ds.rows[0]["name"] == "Alice"
        assert ds.rows[1]["name"] == "Bob"
        assert ds.rows[2]["name"] == "Charlie"

    def test_sort_by_string_descending(self) -> None:
        """GWT-4: Sort by string column descending."""
        schema = create_table_schema(
            [
                {"name": "name", "type": "string", "sortable": True},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"name": "Alice"}, {"name": "Charlie"}, {"name": "Bob"}],
        )

        ds.sort_by("name", SortDirection.DESCENDING)

        assert ds.rows[0]["name"] == "Charlie"

    def test_sort_by_integer(self) -> None:
        """GWT-4: Sort by integer column."""
        schema = create_table_schema(
            [
                {"name": "age", "type": "integer", "sortable": True},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"age": 30}, {"age": 20}, {"age": 25}],
        )

        ds.sort_by("age", SortDirection.ASCENDING)

        assert ds.rows[0]["age"] == 20
        assert ds.rows[1]["age"] == 25
        assert ds.rows[2]["age"] == 30

    def test_sort_toggle_direction(self) -> None:
        """GWT-4: Toggle sort direction on same column."""
        schema = create_table_schema(
            [
                {"name": "id", "type": "integer", "sortable": True},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"id": 1}, {"id": 3}, {"id": 2}],
        )

        ds.sort_by("id")  # First sort: ascending
        assert ds.current_sort == ("id", SortDirection.ASCENDING)

        ds.sort_by("id")  # Second sort: descending
        assert ds.current_sort == ("id", SortDirection.DESCENDING)

    def test_sort_non_sortable_column(self) -> None:
        """GWT-4: Non-sortable column is ignored."""
        schema = create_table_schema(
            [
                {"name": "notes", "type": "string", "sortable": False},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"notes": "B"}, {"notes": "A"}],
        )

        ds.sort_by("notes")

        # Order unchanged
        assert ds.rows[0]["notes"] == "B"

    def test_sort_handles_nulls(self) -> None:
        """GWT-4: Sort handles null values (nulls last)."""
        schema = create_table_schema(
            [
                {"name": "value", "type": "integer", "sortable": True},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"value": 2}, {"value": None}, {"value": 1}],
        )

        ds.sort_by("value", SortDirection.ASCENDING)

        assert ds.rows[0]["value"] == 1
        assert ds.rows[1]["value"] == 2
        assert ds.rows[2]["value"] is None


# ---------------------------------------------------------------------------
# GWT-5: Schema Validation Integration Tests
# ---------------------------------------------------------------------------


class TestGWT5SchemaValidationIntegration:
    """GWT-5: Schema validation integration tests."""

    def test_validate_valid_row(self) -> None:
        """GWT-5: Valid row passes validation."""
        schema = create_table_schema(
            [
                {"name": "name", "type": "string", "required": True},
                {"name": "age", "type": "integer"},
            ]
        )

        result = schema.validate_row({"name": "Alice", "age": 30})

        assert result.valid is True
        assert result.error_count == 0

    def test_validate_missing_required(self) -> None:
        """GWT-5: Missing required field fails validation."""
        schema = create_table_schema(
            [
                {"name": "id", "type": "integer", "required": True},
            ]
        )

        result = schema.validate_row({})

        assert result.valid is False
        assert result.error_count == 1
        assert result.errors[0].field_name == "id"

    def test_validate_type_mismatch(self) -> None:
        """GWT-5: Type mismatch fails validation."""
        schema = create_table_schema(
            [
                {"name": "count", "type": "integer"},
            ]
        )

        result = schema.validate_row({"count": "not a number"})

        assert result.valid is False
        assert "Expected integer" in result.errors[0].message

    def test_validate_range_below_min(self) -> None:
        """GWT-5: Value below minimum fails validation."""
        schema = TableSchema(
            columns=[
                ColumnDefinition(
                    field_name="age",
                    display_label="Age",
                    field_type=FieldType.INTEGER,
                    min_value=0,
                ),
            ]
        )

        result = schema.validate_row({"age": -5})

        assert result.valid is False
        assert "below minimum" in result.errors[0].message

    def test_validate_range_above_max(self) -> None:
        """GWT-5: Value above maximum fails validation."""
        schema = TableSchema(
            columns=[
                ColumnDefinition(
                    field_name="percentage",
                    display_label="Percentage",
                    field_type=FieldType.FLOAT,
                    max_value=100,
                ),
            ]
        )

        result = schema.validate_row({"percentage": 150})

        assert result.valid is False
        assert "exceeds maximum" in result.errors[0].message

    def test_validate_length_exceeded(self) -> None:
        """GWT-5: String exceeding max length fails validation."""
        schema = TableSchema(
            columns=[
                ColumnDefinition(
                    field_name="code",
                    display_label="Code",
                    max_length=5,
                ),
            ]
        )

        result = schema.validate_row({"code": "TOOLONG"})

        assert result.valid is False
        assert "exceeds maximum" in result.errors[0].message

    def test_validate_all_rows(self) -> None:
        """GWT-5: Validate all rows in data source."""
        schema = create_table_schema(
            [
                {"name": "id", "type": "integer", "required": True},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"id": 1}, {"id": None}, {"id": 3}],
        )

        results = ds.validate_all()

        assert results[0].valid is True
        assert results[1].valid is False
        assert results[2].valid is True

    def test_get_invalid_rows(self) -> None:
        """GWT-5: Get list of invalid rows."""
        schema = create_table_schema(
            [
                {"name": "value", "type": "integer", "required": True},
            ]
        )
        ds = TableDataSource(
            schema=schema,
            rows=[{"value": 1}, {}, {"value": 3}],
        )

        invalid = ds.get_invalid_rows()

        assert len(invalid) == 1
        assert invalid[0][0] == 1  # Row index


# ---------------------------------------------------------------------------
# TableDataSource Tests
# ---------------------------------------------------------------------------


class TestTableDataSource:
    """Tests for TableDataSource class."""

    def test_data_source_creation(self) -> None:
        """Test creating a data source."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])
        ds = TableDataSource(schema=schema)

        assert ds.row_count == 0
        assert ds.schema == schema

    def test_add_row(self) -> None:
        """Test adding a row."""
        schema = create_table_schema([{"name": "name", "type": "string"}])
        ds = TableDataSource(schema=schema)

        result = ds.add_row({"name": "Test"})

        assert result is True
        assert ds.row_count == 1

    def test_add_rows(self) -> None:
        """Test adding multiple rows."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])
        ds = TableDataSource(schema=schema)

        count = ds.add_rows([{"id": 1}, {"id": 2}, {"id": 3}])

        assert count == 3
        assert ds.row_count == 3

    def test_clear(self) -> None:
        """Test clearing rows."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])
        ds = TableDataSource(schema=schema, rows=[{"id": 1}, {"id": 2}])

        ds.clear()

        assert ds.row_count == 0

    def test_filter_by(self) -> None:
        """Test filtering rows."""
        schema = create_table_schema([{"name": "value", "type": "integer"}])
        ds = TableDataSource(
            schema=schema,
            rows=[{"value": 1}, {"value": 5}, {"value": 3}],
        )

        filtered = ds.filter_by(lambda row: row["value"] > 2)

        assert len(filtered) == 2

    def test_to_dict(self) -> None:
        """Test data source dictionary conversion."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])
        ds = TableDataSource(schema=schema, rows=[{"id": 1}])

        d = ds.to_dict()

        assert "schema" in d
        assert "rows" in d
        assert d["row_count"] == 1


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_table_schema(self) -> None:
        """Test create_table_schema function."""
        schema = create_table_schema(
            [
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
            ]
        )

        assert schema.column_count == 2

    def test_create_table_from_blueprint(self) -> None:
        """Test create_table_from_blueprint function."""
        blueprint = {
            "target_schema": {"columns": [{"name": "field1", "type": "string"}]}
        }

        ds = create_table_from_blueprint(blueprint)

        assert ds.schema.column_count == 1

    def test_infer_field_type_string(self) -> None:
        """Test inferring string type."""
        assert infer_field_type("hello") == FieldType.STRING

    def test_infer_field_type_integer(self) -> None:
        """Test inferring integer type."""
        assert infer_field_type(42) == FieldType.INTEGER

    def test_infer_field_type_float(self) -> None:
        """Test inferring float type."""
        assert infer_field_type(3.14) == FieldType.FLOAT

    def test_infer_field_type_boolean(self) -> None:
        """Test inferring boolean type."""
        assert infer_field_type(True) == FieldType.BOOLEAN

    def test_infer_field_type_list(self) -> None:
        """Test inferring list type."""
        assert infer_field_type([1, 2, 3]) == FieldType.LIST

    def test_infer_field_type_dict(self) -> None:
        """Test inferring object type."""
        assert infer_field_type({"key": "value"}) == FieldType.OBJECT

    def test_infer_field_type_none(self) -> None:
        """Test inferring unknown type for None."""
        assert infer_field_type(None) == FieldType.UNKNOWN


# ---------------------------------------------------------------------------
# JPL Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule2_max_columns_constant(self) -> None:
        """JPL Rule #2: MAX_COLUMNS is bounded."""
        assert MAX_COLUMNS == 100
        assert MAX_COLUMNS > 0

    def test_rule2_max_rows_constant(self) -> None:
        """JPL Rule #2: MAX_ROWS is bounded."""
        assert MAX_ROWS == 10000
        assert MAX_ROWS > 0

    def test_rule5_schema_none_assertion(self) -> None:
        """JPL Rule #5: from_dict asserts data not None."""
        with pytest.raises(AssertionError):
            TableSchema.from_dict(None)  # type: ignore

    def test_rule5_blueprint_none_assertion(self) -> None:
        """JPL Rule #5: from_blueprint asserts blueprint not None."""
        with pytest.raises(AssertionError):
            TableSchema.from_blueprint(None)  # type: ignore

    def test_rule5_data_source_none_schema(self) -> None:
        """JPL Rule #5: TableDataSource asserts schema not None."""
        with pytest.raises(AssertionError):
            TableDataSource(schema=None)  # type: ignore

    def test_rule9_field_type_enum(self) -> None:
        """JPL Rule #9: FieldType has complete values."""
        values = {ft.value for ft in FieldType}
        required = {"string", "integer", "float", "boolean", "date", "list"}
        assert required.issubset(values)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_schema(self) -> None:
        """Test schema with no columns."""
        schema = TableSchema()
        assert schema.column_count == 0
        assert schema.field_names == []

    def test_format_cell_invalid_row(self) -> None:
        """Test formatting cell with invalid row index."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])
        ds = TableDataSource(schema=schema, rows=[{"id": 1}])

        result = ds.format_cell(999, "id")
        assert result == ""

    def test_format_cell_invalid_column(self) -> None:
        """Test formatting cell with invalid column."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])
        ds = TableDataSource(schema=schema, rows=[{"id": 1}])

        result = ds.format_cell(0, "unknown")
        assert result == ""

    def test_sort_unknown_column(self) -> None:
        """Test sorting by unknown column is ignored."""
        schema = create_table_schema([{"name": "id", "type": "integer"}])
        ds = TableDataSource(schema=schema, rows=[{"id": 2}, {"id": 1}])

        ds.sort_by("unknown")

        # Order unchanged
        assert ds.rows[0]["id"] == 2

    def test_required_fields_property(self) -> None:
        """Test required_fields property."""
        schema = TableSchema(
            columns=[
                ColumnDefinition(field_name="id", display_label="ID", required=True),
                ColumnDefinition(
                    field_name="name", display_label="Name", required=False
                ),
                ColumnDefinition(
                    field_name="email", display_label="Email", required=True
                ),
            ]
        )

        assert set(schema.required_fields) == {"id", "email"}

    def test_validate_row_null_value_not_required(self) -> None:
        """Test validation passes for null non-required field."""
        schema = create_table_schema(
            [
                {"name": "optional", "type": "string", "required": False},
            ]
        )

        result = schema.validate_row({"optional": None})

        assert result.valid is True

    def test_schema_to_dict(self) -> None:
        """Test schema serialization."""
        schema = create_table_schema(
            [
                {"name": "id", "type": "integer"},
            ]
        )
        schema.schema_name = "test_schema"

        d = schema.to_dict()

        assert d["schema_name"] == "test_schema"
        assert d["column_count"] == 1
