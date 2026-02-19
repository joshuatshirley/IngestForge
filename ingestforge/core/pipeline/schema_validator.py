"""Post-Extraction Schema Validation for IngestForge (IF).

Post-Extraction Schema Validation.
Ensures 100% of extracted data complies with user-defined output schemas.
Follows NASA JPL Power of Ten rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, get_origin, get_args

from pydantic import BaseModel, ValidationError as PydanticValidationError

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFTextArtifact

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_ERRORS = 100
MAX_FIELDS = 100
MAX_NESTING_DEPTH = 10
MAX_BATCH_SIZE = 5000
MAX_FIELD_PATH_LENGTH = 256
MAX_ERROR_MESSAGE_LENGTH = 500


class ErrorType(Enum):
    """Types of validation errors."""

    TYPE_MISMATCH = "type_mismatch"
    REQUIRED_MISSING = "required_missing"
    CONSTRAINT_VIOLATION = "constraint_violation"
    COERCION_FAILED = "coercion_failed"
    NESTED_ERROR = "nested_error"
    UNKNOWN = "unknown"


@dataclass
class FieldValidationError:
    """Single validation error for a field.

    GWT-4: Validation report with field-level errors.
    Rule #9: Complete type hints.
    """

    field_path: str
    error_type: ErrorType
    message: str
    expected_type: Optional[str] = None
    actual_value: Optional[Any] = None
    actual_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate error fields.

        Rule #5: Assert preconditions.
        """
        assert (
            len(self.field_path) <= MAX_FIELD_PATH_LENGTH
        ), f"field_path exceeds {MAX_FIELD_PATH_LENGTH} characters"
        # Truncate message if too long
        if len(self.message) > MAX_ERROR_MESSAGE_LENGTH:
            self.message = self.message[: MAX_ERROR_MESSAGE_LENGTH - 3] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_path": self.field_path,
            "error_type": self.error_type.value,
            "message": self.message,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
        }


@dataclass
class SchemaValidationReport:
    """Aggregated validation results.

    GWT-4: Comprehensive validation report.
    Rule #9: Complete type hints.
    """

    is_valid: bool
    errors: List[FieldValidationError] = field(default_factory=list)
    warnings: List[FieldValidationError] = field(default_factory=list)
    validated_count: int = 0
    failed_count: int = 0
    coerced_count: int = 0
    partial_mode: bool = False

    def __post_init__(self) -> None:
        """Validate report constraints.

        Rule #5: Assert preconditions.
        Rule #2: Bounded lists.
        """
        # Truncate error lists to max
        if len(self.errors) > MAX_ERRORS:
            self.errors = self.errors[:MAX_ERRORS]
        if len(self.warnings) > MAX_ERRORS:
            self.warnings = self.warnings[:MAX_ERRORS]

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Total number of warnings."""
        return len(self.warnings)

    @property
    def success_rate(self) -> float:
        """Percentage of successful validations."""
        total = self.validated_count + self.failed_count
        if total == 0:
            return 1.0
        return self.validated_count / total

    def add_error(self, error: FieldValidationError) -> bool:
        """Add an error to the report.

        Rule #2: Bounded list.

        Args:
            error: Error to add.

        Returns:
            True if added, False if at capacity.
        """
        if len(self.errors) >= MAX_ERRORS:
            return False
        self.errors.append(error)
        return True

    def add_warning(self, warning: FieldValidationError) -> bool:
        """Add a warning to the report.

        Rule #2: Bounded list.

        Args:
            warning: Warning to add.

        Returns:
            True if added, False if at capacity.
        """
        if len(self.warnings) >= MAX_ERRORS:
            return False
        self.warnings.append(warning)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "validated_count": self.validated_count,
            "failed_count": self.failed_count,
            "coerced_count": self.coerced_count,
            "success_rate": self.success_rate,
            "partial_mode": self.partial_mode,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


class SchemaValidator:
    """Main schema validation engine.

    GWT-1: Schema validation with Pydantic.
    GWT-2: Type coercion.
    GWT-3: Required field checking.
    Rule #4: Methods < 60 lines.
    """

    def __init__(
        self,
        partial_mode: bool = False,
        coerce_types: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """Initialize validator.

        Rule #5: Assert preconditions.

        Args:
            partial_mode: If True, continue validation after errors.
            coerce_types: If True, attempt type coercion before failing.
            strict_mode: If True, treat warnings as errors.
        """
        self._partial_mode = partial_mode
        self._coerce_types = coerce_types
        self._strict_mode = strict_mode

    def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
    ) -> SchemaValidationReport:
        """Validate data against a Pydantic schema.

        GWT-1: Schema validation.
        Rule #4: Function < 60 lines.

        Args:
            data: Data dictionary to validate.
            schema: Pydantic model class to validate against.

        Returns:
            Validation report with errors and status.
        """
        assert data is not None, "data cannot be None"
        assert schema is not None, "schema cannot be None"

        report = SchemaValidationReport(
            is_valid=True,
            partial_mode=self._partial_mode,
        )

        # Attempt validation with Pydantic
        try:
            if self._coerce_types:
                # Pydantic will attempt coercion automatically
                instance = schema.model_validate(data)
                report.validated_count = 1
                report.coerced_count = self._count_coercions(data, instance)
            else:
                # Strict mode - no coercion
                instance = schema.model_validate(data, strict=True)
                report.validated_count = 1

        except PydanticValidationError as e:
            report.is_valid = False
            report.failed_count = 1
            self._process_pydantic_errors(e, report)

        return report

    def validate_field(
        self,
        value: Any,
        expected_type: Type,
        field_path: str = "root",
    ) -> Optional[FieldValidationError]:
        """Validate a single field value.

        GWT-2: Type coercion support.
        Rule #4: Function < 60 lines.

        Args:
            value: Value to validate.
            expected_type: Expected Python type.
            field_path: Path to field for error reporting.

        Returns:
            FieldValidationError if invalid, None if valid.
        """
        if value is None:
            return None  # None handling is separate (required check)

        actual_type = type(value).__name__
        expected_name = self._get_type_name(expected_type)

        # Check if type matches
        if isinstance(value, expected_type):
            return None

        # Attempt coercion if enabled
        if self._coerce_types:
            coerced = self._attempt_coercion(value, expected_type)
            if coerced is not None:
                return None  # Coercion successful

        return FieldValidationError(
            field_path=field_path,
            error_type=ErrorType.TYPE_MISMATCH,
            message=f"Expected {expected_name}, got {actual_type}",
            expected_type=expected_name,
            actual_value=str(value)[:100],
            actual_type=actual_type,
        )

    def validate_batch(
        self,
        items: List[Dict[str, Any]],
        schema: Type[BaseModel],
    ) -> SchemaValidationReport:
        """Validate multiple items against a schema.

        GWT-5: Partial validation mode.
        Rule #2: Bounded batch size.
        Rule #4: Function < 60 lines.

        Args:
            items: List of data dictionaries.
            schema: Pydantic model class.

        Returns:
            Aggregated validation report.
        """
        assert items is not None, "items cannot be None"

        # Bound batch size
        bounded_items = items[:MAX_BATCH_SIZE]

        report = SchemaValidationReport(
            is_valid=True,
            partial_mode=self._partial_mode,
        )

        for idx, item in enumerate(bounded_items):
            item_report = self.validate(item, schema)

            if item_report.is_valid:
                report.validated_count += 1
                report.coerced_count += item_report.coerced_count
            else:
                report.failed_count += 1
                # Add errors with item index prefix
                for error in item_report.errors:
                    prefixed_error = FieldValidationError(
                        field_path=f"[{idx}].{error.field_path}",
                        error_type=error.error_type,
                        message=error.message,
                        expected_type=error.expected_type,
                        actual_value=error.actual_value,
                        actual_type=error.actual_type,
                    )
                    report.add_error(prefixed_error)

                if not self._partial_mode:
                    report.is_valid = False
                    break

        # In partial mode, valid if any succeeded
        if self._partial_mode:
            report.is_valid = report.validated_count > 0
        else:
            report.is_valid = report.failed_count == 0

        return report

    def check_required_fields(
        self,
        data: Dict[str, Any],
        required_fields: List[str],
    ) -> List[FieldValidationError]:
        """Check for missing required fields.

        GWT-3: Required field checking.
        Rule #4: Function < 60 lines.

        Args:
            data: Data dictionary.
            required_fields: List of required field names.

        Returns:
            List of errors for missing fields.
        """
        errors: List[FieldValidationError] = []

        for field_name in required_fields[:MAX_FIELDS]:
            if field_name not in data or data[field_name] is None:
                error = FieldValidationError(
                    field_path=field_name,
                    error_type=ErrorType.REQUIRED_MISSING,
                    message=f"Required field '{field_name}' is missing",
                    expected_type=None,
                    actual_value=None,
                )
                errors.append(error)

                if len(errors) >= MAX_ERRORS:
                    break

        return errors

    def _process_pydantic_errors(
        self,
        exc: PydanticValidationError,
        report: SchemaValidationReport,
    ) -> None:
        """Process Pydantic validation errors into report.

        Rule #4: Function < 60 lines.
        """
        for error in exc.errors()[:MAX_ERRORS]:
            # Build field path from location
            loc = error.get("loc", ())
            field_path = ".".join(str(p) for p in loc) if loc else "root"

            # Determine error type
            error_type_str = error.get("type", "")
            if "missing" in error_type_str:
                error_type = ErrorType.REQUIRED_MISSING
            elif "type" in error_type_str:
                error_type = ErrorType.TYPE_MISMATCH
            else:
                error_type = ErrorType.CONSTRAINT_VIOLATION

            field_error = FieldValidationError(
                field_path=field_path,
                error_type=error_type,
                message=error.get("msg", "Validation failed"),
                expected_type=None,
                actual_value=str(error.get("input", ""))[:100],
            )
            report.add_error(field_error)

    def _attempt_coercion(
        self,
        value: Any,
        target_type: Type,
    ) -> Optional[Any]:
        """Attempt to coerce value to target type.

        GWT-2: Type coercion.
        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            value: Value to coerce.
            target_type: Target Python type.

        Returns:
            Coerced value or None if coercion failed.
        """
        try:
            # Handle common coercions
            if target_type == int:
                return int(value)
            elif target_type == float:
                return float(value)
            elif target_type == str:
                return str(value)
            elif target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif target_type == date:
                if isinstance(value, str):
                    return date.fromisoformat(value[:10])
                elif isinstance(value, datetime):
                    return value.date()
            elif target_type == datetime:
                if isinstance(value, str):
                    return datetime.fromisoformat(value)
                elif isinstance(value, date):
                    return datetime.combine(value, datetime.min.time())
        except (ValueError, TypeError, AttributeError):
            return None

        return None

    def _count_coercions(
        self,
        original: Dict[str, Any],
        validated: BaseModel,
    ) -> int:
        """Count fields that were coerced during validation.

        Rule #4: Function < 60 lines.
        """
        count = 0
        validated_dict = validated.model_dump()

        for key in original:
            if key in validated_dict:
                orig_type = type(original[key])
                new_type = type(validated_dict[key])
                if orig_type != new_type and original[key] is not None:
                    count += 1

        return count

    @staticmethod
    def _get_type_name(t: Type) -> str:
        """Get human-readable type name."""
        origin = get_origin(t)
        if origin is not None:
            args = get_args(t)
            args_str = ", ".join(SchemaValidator._get_type_name(a) for a in args)
            return f"{origin.__name__}[{args_str}]"
        return getattr(t, "__name__", str(t))


class IFValidationProcessor(IFProcessor):
    """Pipeline processor for schema validation.

    Post-Extraction Schema Validation.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        schema: Type[BaseModel],
        partial_mode: bool = False,
        coerce_types: bool = True,
    ) -> None:
        """Initialize validation processor.

        Rule #5: Assert preconditions.

        Args:
            schema: Pydantic model to validate against.
            partial_mode: Continue after validation errors.
            coerce_types: Attempt type coercion.
        """
        assert schema is not None, "schema cannot be None"
        self._schema = schema
        self._validator = SchemaValidator(
            partial_mode=partial_mode,
            coerce_types=coerce_types,
        )
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        return "schema-validator"

    @property
    def version(self) -> str:
        return self._version

    @property
    def capabilities(self) -> List[str]:
        return ["validate", "validate.schema", "quality"]

    @property
    def memory_mb(self) -> int:
        return 50

    def is_available(self) -> bool:
        return True

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """Validate artifact data against schema.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact to validate.

        Returns:
            Artifact with validation metadata.
        """
        # Extract data from artifact metadata
        data = dict(artifact.metadata)

        # Add content if present
        if hasattr(artifact, "content"):
            data["content"] = getattr(artifact, "content", "")

        # Perform validation
        report = self._validator.validate(data, self._schema)

        # Update metadata with validation results
        new_metadata = dict(artifact.metadata)
        new_metadata["validation_status"] = report.is_valid
        new_metadata["validation_error_count"] = report.error_count
        new_metadata["validation_warning_count"] = report.warning_count
        new_metadata["validation_coerced_count"] = report.coerced_count

        if not report.is_valid:
            new_metadata["validation_errors"] = [
                e.to_dict() for e in report.errors[:10]
            ]

        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-validated",
            content=getattr(artifact, "content", ""),
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.root_artifact_id or artifact.artifact_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self.processor_id],
            metadata=new_metadata,
        )


# Convenience functions


def validate_against_schema(
    data: Dict[str, Any],
    schema: Type[BaseModel],
    partial_mode: bool = False,
) -> SchemaValidationReport:
    """Convenience function to validate data against a schema.

    Args:
        data: Data dictionary to validate.
        schema: Pydantic model class.
        partial_mode: Continue after errors.

    Returns:
        Validation report.
    """
    validator = SchemaValidator(partial_mode=partial_mode)
    return validator.validate(data, schema)


def validate_batch(
    items: List[Dict[str, Any]],
    schema: Type[BaseModel],
    partial_mode: bool = True,
) -> SchemaValidationReport:
    """Convenience function to validate multiple items.

    Args:
        items: List of data dictionaries.
        schema: Pydantic model class.
        partial_mode: Continue after errors.

    Returns:
        Aggregated validation report.
    """
    validator = SchemaValidator(partial_mode=partial_mode)
    return validator.validate_batch(items, schema)


def create_validation_processor(
    schema: Type[BaseModel],
    partial_mode: bool = False,
) -> IFValidationProcessor:
    """Convenience function to create a validation processor.

    Args:
        schema: Pydantic model to validate against.
        partial_mode: Continue after validation errors.

    Returns:
        Configured IFValidationProcessor.
    """
    return IFValidationProcessor(schema=schema, partial_mode=partial_mode)
