"""Schema-Driven Table Component.

Schema-Driven Table Component.
Follows NASA JPL Power of Ten rules.

Dynamically generates table columns from vertical Blueprint target_schema
for domain-specific data visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_COLUMNS = 100
MAX_ROWS = 10000
MAX_FIELD_NAME_LENGTH = 128
MAX_DISPLAY_LABEL_LENGTH = 256
MAX_DESCRIPTION_LENGTH = 1024
MAX_CELL_VALUE_LENGTH = 10000


class FieldType(Enum):
    """Supported field types for table columns.

    GWT-2: Type-aware rendering.
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"
    OBJECT = "object"
    UNKNOWN = "unknown"


class SortDirection(Enum):
    """Sort direction for columns."""

    ASCENDING = "asc"
    DESCENDING = "desc"


@dataclass
class ValidationError:
    """Validation error for a cell or row.

    GWT-5: Schema validation integration.
    """

    field_name: str
    message: str
    value: Any = None
    row_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_name": self.field_name,
            "message": self.message,
            "value": str(self.value)[:100] if self.value is not None else None,
            "row_index": self.row_index,
        }


@dataclass
class ValidationResult:
    """Result of row validation.

    GWT-5: Schema validation integration.
    """

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    row_index: Optional[int] = None

    @property
    def error_count(self) -> int:
        """Number of validation errors."""
        return len(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "error_count": self.error_count,
            "row_index": self.row_index,
        }


@dataclass
class ColumnDefinition:
    """Individual column configuration.

    GWT-1: Dynamic column generation.
    GWT-3: Field metadata display.
    Rule #9: Complete type hints.
    """

    field_name: str
    display_label: str
    field_type: FieldType = FieldType.STRING
    sortable: bool = True
    required: bool = False
    description: Optional[str] = None
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate column definition.

        Rule #5: Assert preconditions.
        """
        assert (
            len(self.field_name) <= MAX_FIELD_NAME_LENGTH
        ), f"field_name exceeds {MAX_FIELD_NAME_LENGTH} characters"
        assert (
            len(self.display_label) <= MAX_DISPLAY_LABEL_LENGTH
        ), f"display_label exceeds {MAX_DISPLAY_LABEL_LENGTH} characters"
        if self.description:
            assert (
                len(self.description) <= MAX_DESCRIPTION_LENGTH
            ), f"description exceeds {MAX_DESCRIPTION_LENGTH} characters"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_name": self.field_name,
            "display_label": self.display_label,
            "field_type": self.field_type.value,
            "sortable": self.sortable,
            "required": self.required,
            "description": self.description,
        }

    def get_tooltip(self) -> str:
        """Generate tooltip text with description and constraints.

        GWT-3: Field metadata display.
        """
        parts: List[str] = []

        if self.description:
            parts.append(self.description)

        if self.required:
            parts.append("Required field")

        if self.min_value is not None or self.max_value is not None:
            range_str = (
                f"Range: {self.min_value or '-inf'} to {self.max_value or 'inf'}"
            )
            parts.append(range_str)

        if self.max_length is not None:
            parts.append(f"Max length: {self.max_length}")

        return " | ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Cell Formatters
# ---------------------------------------------------------------------------


class CellFormatter:
    """Type-aware cell value formatting.

    GWT-2: Type-aware rendering.
    Rule #4: Methods < 60 lines.
    """

    @staticmethod
    def format_value(value: Any, field_type: FieldType) -> str:
        """Format value based on field type.

        Args:
            value: Raw cell value.
            field_type: Expected field type.

        Returns:
            Formatted string representation.
        """
        if value is None:
            return ""

        formatters = {
            FieldType.STRING: CellFormatter.format_string,
            FieldType.INTEGER: CellFormatter.format_integer,
            FieldType.FLOAT: CellFormatter.format_float,
            FieldType.BOOLEAN: CellFormatter.format_boolean,
            FieldType.DATE: CellFormatter.format_date,
            FieldType.DATETIME: CellFormatter.format_datetime,
            FieldType.LIST: CellFormatter.format_list,
            FieldType.OBJECT: CellFormatter.format_object,
            FieldType.UNKNOWN: CellFormatter.format_string,
        }

        formatter = formatters.get(field_type, CellFormatter.format_string)
        return formatter(value)

    @staticmethod
    def format_string(value: Any) -> str:
        """Format string value."""
        result = str(value)
        if len(result) > MAX_CELL_VALUE_LENGTH:
            return result[: MAX_CELL_VALUE_LENGTH - 3] + "..."
        return result

    @staticmethod
    def format_integer(value: Any) -> str:
        """Format integer value with thousands separator."""
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def format_float(value: Any) -> str:
        """Format float value with 2 decimal places."""
        try:
            return f"{float(value):,.2f}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def format_boolean(value: Any) -> str:
        """Format boolean value."""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, str):
            return "Yes" if value.lower() in ("true", "yes", "1") else "No"
        return "Yes" if value else "No"

    @staticmethod
    def format_date(value: Any) -> str:
        """Format date value as YYYY-MM-DD."""
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str):
            return value[:10] if len(value) >= 10 else value
        return str(value)

    @staticmethod
    def format_datetime(value: Any) -> str:
        """Format datetime value as YYYY-MM-DD HH:MM:SS."""
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    @staticmethod
    def format_list(value: Any) -> str:
        """Format list value as comma-separated."""
        if isinstance(value, (list, tuple)):
            items = [str(v) for v in value[:10]]  # Limit items shown
            result = ", ".join(items)
            if len(value) > 10:
                result += f" (+{len(value) - 10} more)"
            return result
        return str(value)

    @staticmethod
    def format_object(value: Any) -> str:
        """Format object value as JSON-like string."""
        if isinstance(value, dict):
            items = [f"{k}: {v}" for k, v in list(value.items())[:5]]
            result = "{" + ", ".join(items)
            if len(value) > 5:
                result += f", ... (+{len(value) - 5} keys)"
            return result + "}"
        return str(value)


# ---------------------------------------------------------------------------
# Table Schema
# ---------------------------------------------------------------------------


@dataclass
class TableSchema:
    """Schema definition for data table.

    GWT-1: Dynamic column generation.
    Rule #9: Complete type hints.
    """

    columns: List[ColumnDefinition] = field(default_factory=list)
    schema_name: str = ""
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate schema.

        Rule #5: Assert preconditions.
        """
        assert (
            len(self.columns) <= MAX_COLUMNS
        ), f"columns exceeds maximum of {MAX_COLUMNS}"

    @property
    def column_count(self) -> int:
        """Number of columns in schema."""
        return len(self.columns)

    @property
    def field_names(self) -> List[str]:
        """List of all field names."""
        return [col.field_name for col in self.columns]

    @property
    def required_fields(self) -> List[str]:
        """List of required field names."""
        return [col.field_name for col in self.columns if col.required]

    def get_column(self, field_name: str) -> Optional[ColumnDefinition]:
        """Get column definition by field name."""
        for col in self.columns:
            if col.field_name == field_name:
                return col
        return None

    def validate_row(self, row: Dict[str, Any], row_index: int = 0) -> ValidationResult:
        """Validate a data row against the schema.

        GWT-5: Schema validation integration.
        Rule #4: Function < 60 lines.

        Args:
            row: Data row to validate.
            row_index: Index of row for error reporting.

        Returns:
            ValidationResult with any errors found.
        """
        errors: List[ValidationError] = []

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in row or row[field_name] is None:
                errors.append(
                    ValidationError(
                        field_name=field_name,
                        message="Required field is missing",
                        row_index=row_index,
                    )
                )

        # Check field constraints
        for col in self.columns:
            if col.field_name not in row:
                continue

            value = row[col.field_name]
            if value is None:
                continue

            # Type validation
            type_error = self._validate_type(col, value)
            if type_error:
                type_error.row_index = row_index
                errors.append(type_error)
                continue

            # Range validation
            range_error = self._validate_range(col, value)
            if range_error:
                range_error.row_index = row_index
                errors.append(range_error)

            # Length validation
            length_error = self._validate_length(col, value)
            if length_error:
                length_error.row_index = row_index
                errors.append(length_error)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            row_index=row_index,
        )

    def _validate_type(
        self, col: ColumnDefinition, value: Any
    ) -> Optional[ValidationError]:
        """Validate value type matches column type."""
        type_checks = {
            FieldType.INTEGER: lambda v: isinstance(v, int)
            or (isinstance(v, str) and v.isdigit()),
            FieldType.FLOAT: lambda v: isinstance(v, (int, float))
            or (isinstance(v, str) and self._is_float_str(v)),
            FieldType.BOOLEAN: lambda v: isinstance(v, bool)
            or (
                isinstance(v, str)
                and v.lower() in ("true", "false", "yes", "no", "1", "0")
            ),
            FieldType.LIST: lambda v: isinstance(v, (list, tuple)),
            FieldType.OBJECT: lambda v: isinstance(v, dict),
        }

        checker = type_checks.get(col.field_type)
        if checker and not checker(value):
            return ValidationError(
                field_name=col.field_name,
                message=f"Expected {col.field_type.value}, got {type(value).__name__}",
                value=value,
            )
        return None

    def _validate_range(
        self, col: ColumnDefinition, value: Any
    ) -> Optional[ValidationError]:
        """Validate numeric value is within range."""
        if col.field_type not in (FieldType.INTEGER, FieldType.FLOAT):
            return None

        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return None

        if col.min_value is not None and num_value < col.min_value:
            return ValidationError(
                field_name=col.field_name,
                message=f"Value {num_value} is below minimum {col.min_value}",
                value=value,
            )

        if col.max_value is not None and num_value > col.max_value:
            return ValidationError(
                field_name=col.field_name,
                message=f"Value {num_value} exceeds maximum {col.max_value}",
                value=value,
            )

        return None

    def _validate_length(
        self, col: ColumnDefinition, value: Any
    ) -> Optional[ValidationError]:
        """Validate string length constraint."""
        if col.max_length is None:
            return None

        if isinstance(value, str) and len(value) > col.max_length:
            return ValidationError(
                field_name=col.field_name,
                message=f"Length {len(value)} exceeds maximum {col.max_length}",
                value=value,
            )

        return None

    @staticmethod
    def _is_float_str(s: str) -> bool:
        """Check if string represents a float."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "columns": [col.to_dict() for col in self.columns],
            "column_count": self.column_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSchema":
        """Create TableSchema from dictionary.

        Rule #5: Assert preconditions.
        """
        assert data is not None, "data cannot be None"

        columns: List[ColumnDefinition] = []
        for col_data in data.get("columns", data.get("fields", []))[:MAX_COLUMNS]:
            field_type = FieldType.STRING
            type_str = col_data.get("type", col_data.get("field_type", "string"))
            for ft in FieldType:
                if ft.value == type_str:
                    field_type = ft
                    break

            col = ColumnDefinition(
                field_name=str(col_data.get("name", col_data.get("field_name", "")))[
                    :MAX_FIELD_NAME_LENGTH
                ],
                display_label=str(
                    col_data.get(
                        "label", col_data.get("display_label", col_data.get("name", ""))
                    )
                )[:MAX_DISPLAY_LABEL_LENGTH],
                field_type=field_type,
                sortable=bool(col_data.get("sortable", True)),
                required=bool(col_data.get("required", False)),
                description=col_data.get("description"),
                min_value=col_data.get("min_value", col_data.get("minimum")),
                max_value=col_data.get("max_value", col_data.get("maximum")),
                max_length=col_data.get("max_length", col_data.get("maxLength")),
            )
            columns.append(col)

        return cls(
            columns=columns,
            schema_name=str(data.get("schema_name", data.get("name", "")))[
                :MAX_FIELD_NAME_LENGTH
            ],
            schema_version=str(data.get("schema_version", data.get("version", "1.0"))),
        )

    @classmethod
    def from_blueprint(cls, blueprint: Any) -> "TableSchema":
        """Create TableSchema from VerticalBlueprint.

        GWT-1: Dynamic column generation.
        Rule #5: Assert preconditions.

        Args:
            blueprint: VerticalBlueprint with target_schema.

        Returns:
            TableSchema for the blueprint.
        """
        assert blueprint is not None, "blueprint cannot be None"

        # Handle dict-like blueprints
        if isinstance(blueprint, dict):
            target_schema = blueprint.get("target_schema", {})
        else:
            target_schema = getattr(blueprint, "target_schema", {})

        if not target_schema:
            return cls(schema_name="default")

        return cls.from_dict(target_schema)


# ---------------------------------------------------------------------------
# Table Data Source
# ---------------------------------------------------------------------------


@dataclass
class TableDataSource:
    """Data management for schema-driven table.

    GWT-4: Sortable columns.
    Rule #9: Complete type hints.
    """

    schema: TableSchema
    rows: List[Dict[str, Any]] = field(default_factory=list)
    _sort_column: Optional[str] = None
    _sort_direction: SortDirection = SortDirection.ASCENDING

    def __post_init__(self) -> None:
        """Validate data source.

        Rule #5: Assert preconditions.
        """
        assert self.schema is not None, "schema cannot be None"
        assert len(self.rows) <= MAX_ROWS, f"rows exceeds maximum of {MAX_ROWS}"

    @property
    def row_count(self) -> int:
        """Number of rows in data source."""
        return len(self.rows)

    @property
    def current_sort(self) -> Tuple[Optional[str], SortDirection]:
        """Get current sort column and direction."""
        return (self._sort_column, self._sort_direction)

    def add_row(self, row: Dict[str, Any]) -> bool:
        """Add a row to the data source.

        Args:
            row: Data row to add.

        Returns:
            True if added, False if at capacity.
        """
        if len(self.rows) >= MAX_ROWS:
            return False
        self.rows.append(row)
        return True

    def add_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Add multiple rows to the data source.

        Args:
            rows: Data rows to add.

        Returns:
            Number of rows actually added.
        """
        added = 0
        for row in rows:
            if self.add_row(row):
                added += 1
            else:
                break
        return added

    def clear(self) -> None:
        """Clear all rows."""
        self.rows.clear()
        self._sort_column = None

    def sort_by(
        self,
        column: str,
        direction: Optional[SortDirection] = None,
    ) -> None:
        """Sort rows by column.

        GWT-4: Sortable columns.
        Rule #4: Function < 60 lines.

        Args:
            column: Field name to sort by.
            direction: Sort direction. If None, toggles current direction.
        """
        col_def = self.schema.get_column(column)
        if col_def is None or not col_def.sortable:
            return

        # Toggle direction if same column and no direction specified
        if direction is None:
            if self._sort_column == column:
                direction = (
                    SortDirection.DESCENDING
                    if self._sort_direction == SortDirection.ASCENDING
                    else SortDirection.ASCENDING
                )
            else:
                direction = SortDirection.ASCENDING

        self._sort_column = column
        self._sort_direction = direction

        reverse = direction == SortDirection.DESCENDING

        def sort_key(row: Dict[str, Any]) -> Any:
            value = row.get(column)
            if value is None:
                return (1, "")  # Nulls last
            if col_def.field_type in (FieldType.INTEGER, FieldType.FLOAT):
                try:
                    return (0, float(value))
                except (ValueError, TypeError):
                    return (1, str(value))
            return (0, str(value).lower())

        self.rows.sort(key=sort_key, reverse=reverse)

    def filter_by(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
    ) -> List[Dict[str, Any]]:
        """Filter rows by predicate.

        Args:
            predicate: Function returning True for rows to include.

        Returns:
            Filtered list of rows.
        """
        return [row for row in self.rows if predicate(row)]

    def validate_all(self) -> List[ValidationResult]:
        """Validate all rows against schema.

        GWT-5: Schema validation integration.

        Returns:
            List of validation results, one per row.
        """
        return [
            self.schema.validate_row(row, i)
            for i, row in enumerate(self.rows[:MAX_ROWS])
        ]

    def get_invalid_rows(self) -> List[Tuple[int, ValidationResult]]:
        """Get indices and results for invalid rows.

        Returns:
            List of (index, ValidationResult) for invalid rows.
        """
        results = self.validate_all()
        return [(i, result) for i, result in enumerate(results) if not result.valid]

    def format_cell(self, row_index: int, column: str) -> str:
        """Format a cell value for display.

        GWT-2: Type-aware rendering.

        Args:
            row_index: Index of the row.
            column: Field name of the column.

        Returns:
            Formatted cell value string.
        """
        if row_index < 0 or row_index >= len(self.rows):
            return ""

        col_def = self.schema.get_column(column)
        if col_def is None:
            return ""

        value = self.rows[row_index].get(column)
        return CellFormatter.format_value(value, col_def.field_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema": self.schema.to_dict(),
            "rows": self.rows[:MAX_ROWS],
            "row_count": self.row_count,
            "sort_column": self._sort_column,
            "sort_direction": self._sort_direction.value
            if self._sort_direction
            else None,
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_table_schema(columns: List[Dict[str, Any]]) -> TableSchema:
    """Convenience function to create a TableSchema from column definitions.

    Args:
        columns: List of column definition dictionaries.

    Returns:
        Configured TableSchema.
    """
    return TableSchema.from_dict({"columns": columns})


def create_table_from_blueprint(blueprint: Any) -> TableDataSource:
    """Convenience function to create a TableDataSource from a blueprint.

    Args:
        blueprint: VerticalBlueprint or dict with target_schema.

    Returns:
        Configured TableDataSource.
    """
    schema = TableSchema.from_blueprint(blueprint)
    return TableDataSource(schema=schema)


def infer_field_type(value: Any) -> FieldType:
    """Infer field type from a sample value.

    Args:
        value: Sample value to infer type from.

    Returns:
        Inferred FieldType.
    """
    if value is None:
        return FieldType.UNKNOWN
    if isinstance(value, bool):
        return FieldType.BOOLEAN
    if isinstance(value, int):
        return FieldType.INTEGER
    if isinstance(value, float):
        return FieldType.FLOAT
    if isinstance(value, datetime):
        return FieldType.DATETIME
    if isinstance(value, list):
        return FieldType.LIST
    if isinstance(value, dict):
        return FieldType.OBJECT
    return FieldType.STRING
