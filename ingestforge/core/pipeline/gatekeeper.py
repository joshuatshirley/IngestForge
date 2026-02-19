"""
Registry Gatekeeper for IngestForge (IF).

Integrity - Registry Gatekeeper.
Validates processor integrity before registration.
Follows NASA JPL Power of Ten rules.
"""

import hashlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

from ingestforge.core.pipeline.interfaces import IFProcessor

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_VALIDATION_CHECKS = 32
MAX_HASH_CACHE_SIZE = 256

# Required IFProcessor methods and their expected signatures
REQUIRED_METHODS: Dict[str, Dict[str, Any]] = {
    "process": {"params": ["artifact"], "return": "IFArtifact"},
    "is_available": {"params": [], "return": "bool"},
}

REQUIRED_PROPERTIES: List[str] = ["processor_id", "version"]


@dataclass(frozen=True)
class ValidationCheck:
    """
    Definition of a validation check.

    Integrity - Registry Gatekeeper.
    Rule #9: Complete type hints.

    Attributes:
        check_id: Unique identifier for this check.
        description: Human-readable description.
        severity: "error" (blocks registration) or "warning" (logs only).
    """

    check_id: str
    description: str
    severity: str = "error"


@dataclass(frozen=True)
class ValidationError:
    """
    Represents a validation error.

    Integrity - Registry Gatekeeper.
    Rule #7: Explicit error reporting.

    Attributes:
        check_id: The check that failed.
        message: Human-readable error message.
        severity: "error" or "warning".
        details: Additional context.
    """

    check_id: str
    message: str
    severity: str = "error"
    details: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of processor validation.

    Integrity - Registry Gatekeeper.
    Rule #7: Explicit return values.

    Attributes:
        valid: True if all error-level checks passed.
        errors: List of validation errors.
        warnings: List of validation warnings.
        processor_hash: Computed code hash if valid.
    """

    valid: bool
    errors: Tuple[ValidationError, ...]
    warnings: Tuple[ValidationError, ...]
    processor_hash: Optional[str] = None


class ProcessorValidator:
    """
    Validates processor structure against IFProcessor interface.

    Integrity - Registry Gatekeeper.
    Rule #2: Bounded validation checks.
    Rule #7: Explicit validation results.
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self._checks: List[ValidationCheck] = [
            ValidationCheck("required_methods", "Check required methods exist"),
            ValidationCheck("required_properties", "Check required properties exist"),
            ValidationCheck("method_signatures", "Check method signatures"),
            ValidationCheck("return_types", "Check return type annotations"),
            ValidationCheck("inheritance", "Check IFProcessor inheritance"),
        ]

    def validate(self, processor: Any) -> ValidationResult:
        """
        Validate a processor against the IFProcessor interface.

        Args:
            processor: The processor instance or class to validate.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Check 1: Inheritance
        if not isinstance(processor, IFProcessor):
            errors.append(
                ValidationError(
                    check_id="inheritance",
                    message=f"{type(processor).__name__} does not inherit from IFProcessor",
                    severity="error",
                )
            )
            # Can't continue without proper inheritance
            return ValidationResult(
                valid=False, errors=tuple(errors), warnings=tuple(warnings)
            )

        # Check 2: Required methods
        self._check_required_methods(processor, errors)

        # Check 3: Required properties
        self._check_required_properties(processor, errors)

        # Check 4: Method signatures
        self._check_method_signatures(processor, errors, warnings)

        # Check 5: Return type annotations
        self._check_return_types(processor, errors, warnings)

        return ValidationResult(
            valid=len(errors) == 0, errors=tuple(errors), warnings=tuple(warnings)
        )

    def _check_required_methods(
        self, processor: IFProcessor, errors: List[ValidationError]
    ) -> None:
        """Check that required methods exist."""
        for method_name in REQUIRED_METHODS:
            if not hasattr(processor, method_name):
                errors.append(
                    ValidationError(
                        check_id="required_methods",
                        message=f"Missing required method: {method_name}",
                        severity="error",
                    )
                )
            elif not callable(getattr(processor, method_name)):
                errors.append(
                    ValidationError(
                        check_id="required_methods",
                        message=f"'{method_name}' is not callable",
                        severity="error",
                    )
                )

    def _check_required_properties(
        self, processor: IFProcessor, errors: List[ValidationError]
    ) -> None:
        """Check that required properties exist and are accessible."""
        for prop_name in REQUIRED_PROPERTIES:
            if not hasattr(processor, prop_name):
                errors.append(
                    ValidationError(
                        check_id="required_properties",
                        message=f"Missing required property: {prop_name}",
                        severity="error",
                    )
                )
            else:
                try:
                    value = getattr(processor, prop_name)
                    if value is None:
                        errors.append(
                            ValidationError(
                                check_id="required_properties",
                                message=f"Property '{prop_name}' returned None",
                                severity="error",
                            )
                        )
                except Exception as e:
                    errors.append(
                        ValidationError(
                            check_id="required_properties",
                            message=f"Property '{prop_name}' raised exception: {e}",
                            severity="error",
                        )
                    )

    def _check_method_signatures(
        self,
        processor: IFProcessor,
        errors: List[ValidationError],
        warnings: List[ValidationError],
    ) -> None:
        """Check method signatures match expected parameters."""
        for method_name, spec in REQUIRED_METHODS.items():
            method = getattr(processor, method_name, None)
            if method is None or not callable(method):
                continue  # Already caught in required_methods check

            try:
                sig = inspect.signature(method)
                params = [p for p in sig.parameters if p != "self"]
                expected_params = spec["params"]

                if len(params) < len(expected_params):
                    errors.append(
                        ValidationError(
                            check_id="method_signatures",
                            message=f"Method '{method_name}' missing parameters: expected {expected_params}",
                            severity="error",
                        )
                    )
            except (ValueError, TypeError):
                warnings.append(
                    ValidationError(
                        check_id="method_signatures",
                        message=f"Could not inspect signature of '{method_name}'",
                        severity="warning",
                    )
                )

    def _check_return_types(
        self,
        processor: IFProcessor,
        errors: List[ValidationError],
        warnings: List[ValidationError],
    ) -> None:
        """Check return type annotations are present."""
        proc_class = type(processor)
        for method_name, spec in REQUIRED_METHODS.items():
            method = getattr(proc_class, method_name, None)
            if method is None:
                continue

            try:
                hints = get_type_hints(method)
                if "return" not in hints:
                    warnings.append(
                        ValidationError(
                            check_id="return_types",
                            message=f"Method '{method_name}' lacks return type annotation",
                            severity="warning",
                        )
                    )
            except Exception:
                # Type hints may not be resolvable in all contexts
                pass


class IntegrityChecker:
    """
    Computes and verifies processor code hashes.

    Integrity - Registry Gatekeeper.
    Rule #2: Bounded hash cache.
    Rule #7: Explicit results.
    """

    def __init__(self) -> None:
        """Initialize the integrity checker."""
        self._hash_cache: Dict[str, str] = {}
        self._known_hashes: Dict[str, str] = {}

    def compute_hash(self, processor: IFProcessor) -> str:
        """
        Compute SHA-256 hash of processor source code.

        Args:
            processor: The processor to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        proc_id = processor.processor_id

        # Check cache first
        if proc_id in self._hash_cache:
            return self._hash_cache[proc_id]

        # Get source code
        proc_class = type(processor)
        try:
            source = inspect.getsource(proc_class)
        except (OSError, TypeError):
            # Built-in or dynamically generated - use class name + version
            source = f"{proc_class.__name__}:{processor.version}"

        # Compute hash
        hash_value = hashlib.sha256(source.encode("utf-8")).hexdigest()

        # Cache with bounded size
        if len(self._hash_cache) < MAX_HASH_CACHE_SIZE:
            self._hash_cache[proc_id] = hash_value

        return hash_value

    def register_known_hash(self, processor_id: str, hash_value: str) -> None:
        """
        Register a known-good hash for a processor.

        Args:
            processor_id: The processor identifier.
            hash_value: The expected hash value.
        """
        self._known_hashes[processor_id] = hash_value

    def verify_integrity(self, processor: IFProcessor) -> Tuple[bool, Optional[str]]:
        """
        Verify processor integrity against known hash.

        Args:
            processor: The processor to verify.

        Returns:
            Tuple of (is_valid, message). If no known hash, returns (True, None).
        """
        proc_id = processor.processor_id
        current_hash = self.compute_hash(processor)

        if proc_id not in self._known_hashes:
            return True, None  # No known hash to compare

        expected = self._known_hashes[proc_id]
        if current_hash == expected:
            return True, None

        return (
            False,
            f"Hash mismatch: expected {expected[:16]}..., got {current_hash[:16]}...",
        )

    def clear_cache(self) -> None:
        """Clear the hash cache."""
        self._hash_cache.clear()


class DependencyChecker:
    """
    Validates processor dependencies are available.

    Integrity - Registry Gatekeeper.
    Rule #7: Explicit validation results.
    """

    def check_dependencies(self, processor: IFProcessor) -> List[ValidationError]:
        """
        Check if processor dependencies are available.

        Args:
            processor: The processor to check.

        Returns:
            List of dependency validation errors.
        """
        errors: List[ValidationError] = []

        # Check if is_available returns True
        try:
            if not processor.is_available():
                errors.append(
                    ValidationError(
                        check_id="dependencies",
                        message=f"Processor '{processor.processor_id}' reports unavailable",
                        severity="error",
                        details="is_available() returned False",
                    )
                )
        except Exception as e:
            errors.append(
                ValidationError(
                    check_id="dependencies",
                    message=f"is_available() raised exception: {e}",
                    severity="error",
                )
            )

        return errors


class RegistryGatekeeper:
    """
    Combines validation and integrity checking for processor registration.

    Integrity - Registry Gatekeeper.
    Rule #2: Bounded checks.
    Rule #7: Explicit results.
    Rule #9: Complete type hints.
    """

    def __init__(
        self, check_integrity: bool = True, check_dependencies: bool = True
    ) -> None:
        """
        Initialize the gatekeeper.

        Args:
            check_integrity: Whether to verify code hashes.
            check_dependencies: Whether to check dependencies.
        """
        self._validator = ProcessorValidator()
        self._integrity_checker = IntegrityChecker()
        self._dependency_checker = DependencyChecker()
        self._check_integrity = check_integrity
        self._check_dependencies = check_dependencies

    def validate(self, processor: Any) -> ValidationResult:
        """
        Perform full validation of a processor.

        Args:
            processor: The processor to validate.

        Returns:
            ValidationResult with all checks performed.
        """
        # Start with structure validation
        result = self._validator.validate(processor)
        errors = list(result.errors)
        warnings = list(result.warnings)

        # If structure validation failed, return early
        if not result.valid:
            return ValidationResult(
                valid=False, errors=tuple(errors), warnings=tuple(warnings)
            )

        # Check dependencies if enabled
        if self._check_dependencies:
            dep_errors = self._dependency_checker.check_dependencies(processor)
            errors.extend(dep_errors)

        # Compute hash and check integrity if enabled
        processor_hash: Optional[str] = None
        if self._check_integrity:
            processor_hash = self._integrity_checker.compute_hash(processor)
            is_valid, message = self._integrity_checker.verify_integrity(processor)
            if not is_valid and message:
                warnings.append(
                    ValidationError(
                        check_id="integrity", message=message, severity="warning"
                    )
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            processor_hash=processor_hash,
        )

    def register_known_hash(self, processor_id: str, hash_value: str) -> None:
        """Register a known-good hash for integrity checking."""
        self._integrity_checker.register_known_hash(processor_id, hash_value)

    @property
    def integrity_checker(self) -> IntegrityChecker:
        """Access the integrity checker for hash management."""
        return self._integrity_checker


def create_gatekeeper(
    check_integrity: bool = True, check_dependencies: bool = True
) -> RegistryGatekeeper:
    """
    Factory function to create a configured gatekeeper.

    Args:
        check_integrity: Whether to verify code hashes.
        check_dependencies: Whether to check dependencies.

    Returns:
        Configured RegistryGatekeeper instance.
    """
    return RegistryGatekeeper(
        check_integrity=check_integrity, check_dependencies=check_dependencies
    )
