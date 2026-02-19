"""Tests for Post-Extraction Schema Validation.

Comprehensive GWT tests for SchemaValidator, ValidationReport,
and IFValidationProcessor.
"""

import pytest
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from ingestforge.core.pipeline.schema_validator import (
    ErrorType,
    FieldValidationError,
    SchemaValidationReport,
    SchemaValidator,
    IFValidationProcessor,
    validate_against_schema,
    validate_batch,
    create_validation_processor,
    MAX_ERRORS,
    MAX_BATCH_SIZE,
    MAX_FIELD_PATH_LENGTH,
)
from ingestforge.core.pipeline.artifacts import IFTextArtifact


# ---------------------------------------------------------------------------
# Test Schemas
# ---------------------------------------------------------------------------


class SimpleSchema(BaseModel):
    """Simple test schema."""

    name: str
    age: int
    active: bool = True


class RequiredFieldsSchema(BaseModel):
    """Schema with required fields."""

    title: str
    author: str
    year: int


class NestedSchema(BaseModel):
    """Schema with nested objects."""

    id: str
    metadata: dict
    tags: List[str] = Field(default_factory=list)


class DateSchema(BaseModel):
    """Schema with date fields."""

    name: str
    created: date
    updated: Optional[datetime] = None


class StrictSchema(BaseModel):
    """Schema with constraints."""

    code: str = Field(min_length=3, max_length=10)
    count: int = Field(ge=0, le=100)


# ---------------------------------------------------------------------------
# GWT-1: Schema Validation Tests
# ---------------------------------------------------------------------------


class TestGWT1SchemaValidation:
    """GWT-1: Schema validation with Pydantic."""

    def test_valid_data_passes_validation(self) -> None:
        """Given valid data, When validated, Then report is_valid=True."""
        validator = SchemaValidator()
        data = {"name": "Test", "age": 25, "active": True}

        report = validator.validate(data, SimpleSchema)

        assert report.is_valid is True
        assert report.error_count == 0
        assert report.validated_count == 1

    def test_invalid_type_fails_validation(self) -> None:
        """Given invalid type, When validated, Then report has type error."""
        validator = SchemaValidator(coerce_types=False)
        data = {"name": "Test", "age": "not_an_int", "active": True}

        report = validator.validate(data, SimpleSchema)

        assert report.is_valid is False
        assert report.error_count > 0
        assert any(e.error_type == ErrorType.TYPE_MISMATCH for e in report.errors)

    def test_missing_field_fails_validation(self) -> None:
        """Given missing required field, When validated, Then report has missing error."""
        validator = SchemaValidator()
        data = {"name": "Test"}  # Missing 'age'

        report = validator.validate(data, SimpleSchema)

        assert report.is_valid is False
        assert any(e.error_type == ErrorType.REQUIRED_MISSING for e in report.errors)

    def test_extra_fields_ignored(self) -> None:
        """Given extra fields, When validated, Then validation passes."""
        validator = SchemaValidator()
        data = {"name": "Test", "age": 25, "extra_field": "ignored"}

        report = validator.validate(data, SimpleSchema)

        assert report.is_valid is True

    def test_nested_schema_validation(self) -> None:
        """Given nested data, When validated, Then nested fields checked."""
        validator = SchemaValidator()
        data = {"id": "123", "metadata": {"key": "value"}, "tags": ["a", "b"]}

        report = validator.validate(data, NestedSchema)

        assert report.is_valid is True

    def test_constraint_violation_reported(self) -> None:
        """Given constraint violation, When validated, Then error reported."""
        validator = SchemaValidator()
        data = {"code": "AB", "count": 50}  # code too short

        report = validator.validate(data, StrictSchema)

        assert report.is_valid is False
        assert any(
            e.error_type == ErrorType.CONSTRAINT_VIOLATION for e in report.errors
        )

    def test_null_data_raises_assertion(self) -> None:
        """Given None data, When validated, Then assertion raised."""
        validator = SchemaValidator()

        with pytest.raises(AssertionError):
            validator.validate(None, SimpleSchema)  # type: ignore

    def test_null_schema_raises_assertion(self) -> None:
        """Given None schema, When validated, Then assertion raised."""
        validator = SchemaValidator()

        with pytest.raises(AssertionError):
            validator.validate({}, None)  # type: ignore


# ---------------------------------------------------------------------------
# GWT-2: Type Coercion Tests
# ---------------------------------------------------------------------------


class TestGWT2TypeCoercion:
    """GWT-2: Type coercion for compatible types."""

    def test_string_to_int_coercion(self) -> None:
        """Given string '123', When coercing to int, Then succeeds."""
        validator = SchemaValidator(coerce_types=True)
        data = {"name": "Test", "age": "25", "active": True}

        report = validator.validate(data, SimpleSchema)

        assert report.is_valid is True
        assert report.coerced_count >= 1

    def test_string_to_float_coercion(self) -> None:
        """Given string '3.14', When coercing to float, Then succeeds."""
        validator = SchemaValidator(coerce_types=True)
        result = validator._attempt_coercion("3.14", float)

        assert result == 3.14

    def test_string_to_bool_true_coercion(self) -> None:
        """Given string 'true', When coercing to bool, Then returns True."""
        validator = SchemaValidator(coerce_types=True)

        assert validator._attempt_coercion("true", bool) is True
        assert validator._attempt_coercion("1", bool) is True
        assert validator._attempt_coercion("yes", bool) is True

    def test_string_to_bool_false_coercion(self) -> None:
        """Given string 'false', When coercing to bool, Then returns False."""
        validator = SchemaValidator(coerce_types=True)

        assert validator._attempt_coercion("false", bool) is False
        assert validator._attempt_coercion("0", bool) is False

    def test_string_to_date_coercion(self) -> None:
        """Given ISO date string, When coercing to date, Then succeeds."""
        validator = SchemaValidator(coerce_types=True)
        result = validator._attempt_coercion("2024-01-15", date)

        assert result == date(2024, 1, 15)

    def test_string_to_datetime_coercion(self) -> None:
        """Given ISO datetime string, When coercing to datetime, Then succeeds."""
        validator = SchemaValidator(coerce_types=True)
        result = validator._attempt_coercion("2024-01-15T10:30:00", datetime)

        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_date_to_datetime_coercion(self) -> None:
        """Given date, When coercing to datetime, Then succeeds."""
        validator = SchemaValidator(coerce_types=True)
        result = validator._attempt_coercion(date(2024, 1, 15), datetime)

        assert result.date() == date(2024, 1, 15)

    def test_datetime_to_date_coercion(self) -> None:
        """Given datetime, When coercing to date, Then extracts date."""
        validator = SchemaValidator(coerce_types=True)
        result = validator._attempt_coercion(datetime(2024, 1, 15, 10, 30), date)

        assert result == date(2024, 1, 15)

    def test_invalid_coercion_returns_none(self) -> None:
        """Given incompatible value, When coercing, Then returns None."""
        validator = SchemaValidator(coerce_types=True)

        assert validator._attempt_coercion("not_a_number", int) is None
        assert validator._attempt_coercion("invalid", date) is None

    def test_coercion_disabled(self) -> None:
        """Given coercion disabled, When type mismatch, Then fails."""
        validator = SchemaValidator(coerce_types=False)
        data = {"name": "Test", "age": "25", "active": True}

        report = validator.validate(data, SimpleSchema)

        assert report.is_valid is False

    def test_coercion_with_date_schema(self) -> None:
        """Given string dates, When validating DateSchema, Then coerced."""
        validator = SchemaValidator(coerce_types=True)
        data = {"name": "Test", "created": "2024-01-15"}

        report = validator.validate(data, DateSchema)

        assert report.is_valid is True


# ---------------------------------------------------------------------------
# GWT-3: Required Field Check Tests
# ---------------------------------------------------------------------------


class TestGWT3RequiredFieldCheck:
    """GWT-3: Required field checking."""

    def test_all_required_fields_present(self) -> None:
        """Given all required fields, When checking, Then no errors."""
        validator = SchemaValidator()
        data = {"title": "Test", "author": "Author", "year": 2024}

        report = validator.validate(data, RequiredFieldsSchema)

        assert report.is_valid is True
        assert report.error_count == 0

    def test_missing_single_required_field(self) -> None:
        """Given one missing field, When checking, Then one error."""
        validator = SchemaValidator()
        errors = validator.check_required_fields(
            {"title": "Test", "author": "Author"}, ["title", "author", "year"]
        )

        assert len(errors) == 1
        assert errors[0].field_path == "year"
        assert errors[0].error_type == ErrorType.REQUIRED_MISSING

    def test_missing_multiple_required_fields(self) -> None:
        """Given multiple missing fields, When checking, Then multiple errors."""
        validator = SchemaValidator()
        errors = validator.check_required_fields(
            {"title": "Test"}, ["title", "author", "year"]
        )

        assert len(errors) == 2
        field_paths = {e.field_path for e in errors}
        assert "author" in field_paths
        assert "year" in field_paths

    def test_none_value_treated_as_missing(self) -> None:
        """Given None value, When checking required, Then treated as missing."""
        validator = SchemaValidator()
        errors = validator.check_required_fields(
            {"title": "Test", "author": None, "year": 2024}, ["title", "author", "year"]
        )

        assert len(errors) == 1
        assert errors[0].field_path == "author"

    def test_empty_string_not_treated_as_missing(self) -> None:
        """Given empty string, When checking required, Then not missing."""
        validator = SchemaValidator()
        errors = validator.check_required_fields(
            {"title": "", "author": "Author", "year": 2024}, ["title", "author", "year"]
        )

        assert len(errors) == 0

    def test_required_fields_bounded(self) -> None:
        """Given many required fields, When checking, Then bounded."""
        validator = SchemaValidator()
        required = [f"field_{i}" for i in range(200)]
        data = {}

        errors = validator.check_required_fields(data, required)

        assert len(errors) <= MAX_ERRORS


# ---------------------------------------------------------------------------
# GWT-4: Validation Report Tests
# ---------------------------------------------------------------------------


class TestGWT4ValidationReport:
    """GWT-4: Comprehensive validation report."""

    def test_report_tracks_error_count(self) -> None:
        """Given multiple errors, When reported, Then count accurate."""
        report = SchemaValidationReport(is_valid=False)

        for i in range(5):
            report.add_error(
                FieldValidationError(
                    field_path=f"field_{i}",
                    error_type=ErrorType.TYPE_MISMATCH,
                    message=f"Error {i}",
                )
            )

        assert report.error_count == 5

    def test_report_tracks_warning_count(self) -> None:
        """Given multiple warnings, When reported, Then count accurate."""
        report = SchemaValidationReport(is_valid=True)

        for i in range(3):
            report.add_warning(
                FieldValidationError(
                    field_path=f"field_{i}",
                    error_type=ErrorType.CONSTRAINT_VIOLATION,
                    message=f"Warning {i}",
                )
            )

        assert report.warning_count == 3

    def test_report_calculates_success_rate(self) -> None:
        """Given validation counts, When calculated, Then rate correct."""
        report = SchemaValidationReport(
            is_valid=True, validated_count=8, failed_count=2
        )

        assert report.success_rate == 0.8

    def test_report_success_rate_empty(self) -> None:
        """Given no validations, When calculated, Then rate is 1.0."""
        report = SchemaValidationReport(is_valid=True)

        assert report.success_rate == 1.0

    def test_report_to_dict(self) -> None:
        """Given report, When to_dict, Then contains all fields."""
        report = SchemaValidationReport(
            is_valid=False,
            validated_count=5,
            failed_count=2,
        )
        report.add_error(
            FieldValidationError(
                field_path="test",
                error_type=ErrorType.TYPE_MISMATCH,
                message="Test error",
            )
        )

        result = report.to_dict()

        assert "is_valid" in result
        assert "error_count" in result
        assert "success_rate" in result
        assert "errors" in result
        assert len(result["errors"]) == 1

    def test_report_bounds_errors(self) -> None:
        """Given many errors, When added, Then bounded to MAX_ERRORS."""
        report = SchemaValidationReport(is_valid=False)

        for i in range(MAX_ERRORS + 50):
            report.add_error(
                FieldValidationError(
                    field_path=f"field_{i}",
                    error_type=ErrorType.TYPE_MISMATCH,
                    message=f"Error {i}",
                )
            )

        assert report.error_count <= MAX_ERRORS

    def test_field_error_truncates_long_message(self) -> None:
        """Given long message, When created, Then truncated."""
        long_message = "x" * 1000
        error = FieldValidationError(
            field_path="test", error_type=ErrorType.TYPE_MISMATCH, message=long_message
        )

        assert len(error.message) <= 503  # 500 + "..."

    def test_field_error_validates_path_length(self) -> None:
        """Given long path, When created, Then assertion raised."""
        long_path = "x" * (MAX_FIELD_PATH_LENGTH + 10)

        with pytest.raises(AssertionError):
            FieldValidationError(
                field_path=long_path, error_type=ErrorType.TYPE_MISMATCH, message="Test"
            )

    def test_field_error_to_dict(self) -> None:
        """Given field error, When to_dict, Then all fields present."""
        error = FieldValidationError(
            field_path="test.field",
            error_type=ErrorType.TYPE_MISMATCH,
            message="Expected int",
            expected_type="int",
            actual_type="str",
        )

        result = error.to_dict()

        assert result["field_path"] == "test.field"
        assert result["error_type"] == "type_mismatch"
        assert result["expected_type"] == "int"


# ---------------------------------------------------------------------------
# GWT-5: Partial Validation Mode Tests
# ---------------------------------------------------------------------------


class TestGWT5PartialValidationMode:
    """GWT-5: Partial validation mode."""

    def test_partial_mode_continues_after_error(self) -> None:
        """Given partial mode, When error occurs, Then continues."""
        validator = SchemaValidator(partial_mode=True)
        items = [
            {"name": "Valid", "age": 25, "active": True},
            {"name": "Invalid", "age": "not_int"},  # Invalid
            {"name": "Also Valid", "age": 30, "active": False},
        ]

        report = validator.validate_batch(items, SimpleSchema)

        assert report.validated_count == 2
        assert report.failed_count == 1
        assert report.is_valid is True  # Partial mode: valid if any succeeded

    def test_strict_mode_stops_on_error(self) -> None:
        """Given strict mode, When error occurs, Then stops."""
        validator = SchemaValidator(partial_mode=False)
        items = [
            {"name": "Valid", "age": 25, "active": True},
            {"name": "Invalid", "age": "not_int"},  # Invalid
            {"name": "Not Reached", "age": 30, "active": False},
        ]

        report = validator.validate_batch(items, SimpleSchema)

        assert report.is_valid is False
        assert report.validated_count == 1
        assert report.failed_count == 1

    def test_partial_mode_aggregates_errors(self) -> None:
        """Given multiple errors, When partial mode, Then all collected."""
        validator = SchemaValidator(partial_mode=True, coerce_types=False)
        items = [
            {"name": "A", "age": "invalid1"},
            {"name": "B", "age": "invalid2"},
            {"name": "C", "age": "invalid3"},
        ]

        report = validator.validate_batch(items, SimpleSchema)

        assert report.failed_count == 3
        assert report.error_count >= 3

    def test_partial_mode_invalid_if_all_fail(self) -> None:
        """Given all items fail, When partial mode, Then is_valid=False."""
        validator = SchemaValidator(partial_mode=True, coerce_types=False)
        items = [
            {"name": "A", "age": "invalid"},
            {"name": "B", "age": "invalid"},
        ]

        report = validator.validate_batch(items, SimpleSchema)

        assert report.is_valid is False
        assert report.validated_count == 0

    def test_batch_bounds_items(self) -> None:
        """Given large batch, When validating, Then bounded."""
        validator = SchemaValidator(partial_mode=True)
        items = [
            {"name": f"Item{i}", "age": i, "active": True}
            for i in range(MAX_BATCH_SIZE + 100)
        ]

        report = validator.validate_batch(items, SimpleSchema)

        total = report.validated_count + report.failed_count
        assert total <= MAX_BATCH_SIZE

    def test_batch_error_paths_include_index(self) -> None:
        """Given batch errors, When reported, Then paths include index."""
        validator = SchemaValidator(partial_mode=True, coerce_types=False)
        items = [
            {"name": "Valid", "age": 25, "active": True},
            {"name": "Invalid", "age": "bad"},
        ]

        report = validator.validate_batch(items, SimpleSchema)

        assert any("[1]." in e.field_path for e in report.errors)


# ---------------------------------------------------------------------------
# IFValidationProcessor Tests
# ---------------------------------------------------------------------------


class TestIFValidationProcessor:
    """Tests for IFValidationProcessor pipeline integration."""

    def test_processor_id(self) -> None:
        """Given processor, When accessing id, Then returns correct value."""
        processor = IFValidationProcessor(schema=SimpleSchema)

        assert processor.processor_id == "schema-validator"

    def test_processor_capabilities(self) -> None:
        """Given processor, When accessing capabilities, Then includes validate."""
        processor = IFValidationProcessor(schema=SimpleSchema)

        assert "validate" in processor.capabilities
        assert "validate.schema" in processor.capabilities

    def test_processor_is_available(self) -> None:
        """Given processor, When checking availability, Then returns True."""
        processor = IFValidationProcessor(schema=SimpleSchema)

        assert processor.is_available() is True

    def test_processor_validates_artifact(self) -> None:
        """Given valid artifact, When processed, Then validation_status=True."""
        processor = IFValidationProcessor(schema=SimpleSchema)
        artifact = IFTextArtifact(
            artifact_id="test-001",
            content="Test content",
            metadata={"name": "Test", "age": 25, "active": True},
        )

        result = processor.process(artifact)

        assert result.metadata["validation_status"] is True
        assert result.metadata["validation_error_count"] == 0

    def test_processor_reports_errors(self) -> None:
        """Given invalid artifact, When processed, Then errors in metadata."""
        processor = IFValidationProcessor(schema=SimpleSchema, coerce_types=False)
        artifact = IFTextArtifact(
            artifact_id="test-001",
            content="Test content",
            metadata={"name": "Test", "age": "not_int"},
        )

        result = processor.process(artifact)

        assert result.metadata["validation_status"] is False
        assert result.metadata["validation_error_count"] > 0
        assert "validation_errors" in result.metadata

    def test_processor_updates_lineage(self) -> None:
        """Given artifact, When processed, Then lineage updated."""
        processor = IFValidationProcessor(schema=SimpleSchema)
        artifact = IFTextArtifact(
            artifact_id="test-001",
            content="Test",
            lineage_depth=1,
            provenance=["previous-processor"],
        )

        result = processor.process(artifact)

        assert result.lineage_depth == 2
        assert "schema-validator" in result.provenance
        assert result.parent_id == "test-001"


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_validate_against_schema(self) -> None:
        """Given data and schema, When using convenience function, Then validates."""
        data = {"name": "Test", "age": 25, "active": True}

        report = validate_against_schema(data, SimpleSchema)

        assert report.is_valid is True

    def test_validate_batch_function(self) -> None:
        """Given batch and schema, When using convenience function, Then validates."""
        items = [
            {"name": "A", "age": 1, "active": True},
            {"name": "B", "age": 2, "active": False},
        ]

        report = validate_batch(items, SimpleSchema)

        assert report.validated_count == 2

    def test_create_validation_processor(self) -> None:
        """Given schema, When creating processor, Then configured correctly."""
        processor = create_validation_processor(SimpleSchema, partial_mode=True)

        assert processor.processor_id == "schema-validator"
        assert processor._validator._partial_mode is True


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_data_with_defaults(self) -> None:
        """Given empty data with defaults, When validated, Then uses defaults."""
        validator = SchemaValidator()
        data = {"name": "Test", "age": 25}  # 'active' has default

        report = validator.validate(data, SimpleSchema)

        assert report.is_valid is True

    def test_empty_batch(self) -> None:
        """Given empty batch, When validated, Then valid with zero counts."""
        validator = SchemaValidator()

        report = validator.validate_batch([], SimpleSchema)

        assert report.is_valid is True
        assert report.validated_count == 0

    def test_deeply_nested_field_path(self) -> None:
        """Given deep nesting, When error, Then path preserved."""
        error = FieldValidationError(
            field_path="root.level1.level2.level3.field",
            error_type=ErrorType.TYPE_MISMATCH,
            message="Nested error",
        )

        assert "level3" in error.field_path

    def test_error_type_enum_values(self) -> None:
        """Given error types, When accessing values, Then match expected."""
        assert ErrorType.TYPE_MISMATCH.value == "type_mismatch"
        assert ErrorType.REQUIRED_MISSING.value == "required_missing"
        assert ErrorType.COERCION_FAILED.value == "coercion_failed"

    def test_validate_field_with_none(self) -> None:
        """Given None value, When validating field, Then returns None (no error)."""
        validator = SchemaValidator()

        result = validator.validate_field(None, str, "test")

        assert result is None

    def test_get_type_name_generic(self) -> None:
        """Given generic type, When getting name, Then formatted correctly."""
        name = SchemaValidator._get_type_name(List[str])

        assert "list" in name.lower()
        assert "str" in name.lower()
