"""
Unit tests for Integrity - Registry Gatekeeper.

Tests follow GWT (Given-When-Then) behavioral specification format.
Validates NASA JPL Power of Ten rule compliance.
"""

from unittest.mock import MagicMock, patch

from ingestforge.core.pipeline.gatekeeper import (
    MAX_VALIDATION_CHECKS,
    MAX_HASH_CACHE_SIZE,
    ValidationCheck,
    ValidationError,
    ValidationResult,
    ProcessorValidator,
    IntegrityChecker,
    DependencyChecker,
    RegistryGatekeeper,
    create_gatekeeper,
    REQUIRED_METHODS,
    REQUIRED_PROPERTIES,
)
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact


# =============================================================================
# Test Fixtures
# =============================================================================


class ValidProcessor(IFProcessor):
    """A fully valid IFProcessor implementation."""

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        return True

    @property
    def processor_id(self) -> str:
        return "valid-processor"

    @property
    def version(self) -> str:
        return "1.0.0"


class UnavailableProcessor(IFProcessor):
    """Processor that reports unavailable."""

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        return False

    @property
    def processor_id(self) -> str:
        return "unavailable-processor"

    @property
    def version(self) -> str:
        return "1.0.0"


class ExceptionProcessor(IFProcessor):
    """Processor whose is_available raises."""

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        raise RuntimeError("Dependency check failed")

    @property
    def processor_id(self) -> str:
        return "exception-processor"

    @property
    def version(self) -> str:
        return "1.0.0"


class MissingVersionProcessor(IFProcessor):
    """Processor missing version property (will fail validation)."""

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        return True

    @property
    def processor_id(self) -> str:
        return "missing-version"

    # Intentionally missing version property - but we need to satisfy ABC
    @property
    def version(self) -> str:
        return None  # Returns None to simulate missing


class NonePropertyProcessor(IFProcessor):
    """Processor with None property values."""

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        return True

    @property
    def processor_id(self) -> str:
        return None  # Invalid - returns None

    @property
    def version(self) -> str:
        return "1.0.0"


class NotAProcessor:
    """Not an IFProcessor - should fail inheritance check."""

    def process(self, artifact):
        return artifact

    def is_available(self):
        return True

    @property
    def processor_id(self):
        return "fake-processor"

    @property
    def version(self):
        return "1.0.0"


# =============================================================================
# GWT Scenario 1: Valid Processor Registration
# =============================================================================


class TestValidProcessorRegistration:
    """Tests for GWT Scenario 1: Valid Processor Registration."""

    def test_given_valid_processor_when_validated_then_success(self):
        """
        Given a processor implementing all required IFProcessor methods,
        When registration is attempted,
        Then the processor is registered successfully.
        """
        # Given
        processor = ValidProcessor()
        gatekeeper = RegistryGatekeeper()

        # When
        result = gatekeeper.validate(processor)

        # Then
        assert result.valid is True
        assert len(result.errors) == 0

    def test_given_valid_processor_when_validated_then_hash_computed(self):
        """
        Given a valid processor,
        When validation completes,
        Then a code hash is computed.
        """
        # Given
        processor = ValidProcessor()
        gatekeeper = RegistryGatekeeper(check_integrity=True)

        # When
        result = gatekeeper.validate(processor)

        # Then
        assert result.valid is True
        assert result.processor_hash is not None
        assert len(result.processor_hash) == 64  # SHA-256 hex


# =============================================================================
# GWT Scenario 2: Missing Required Method Rejection
# =============================================================================


class TestMissingRequiredMethodRejection:
    """Tests for GWT Scenario 2: Missing Required Method Rejection."""

    def test_given_non_processor_when_validated_then_rejected(self):
        """
        Given an object not inheriting from IFProcessor,
        When validation is attempted,
        Then registration is rejected with inheritance error.
        """
        # Given
        not_processor = NotAProcessor()
        gatekeeper = RegistryGatekeeper()

        # When
        result = gatekeeper.validate(not_processor)

        # Then
        assert result.valid is False
        assert len(result.errors) >= 1
        assert any(e.check_id == "inheritance" for e in result.errors)

    def test_given_none_property_when_validated_then_rejected(self):
        """
        Given a processor with None property values,
        When validation is attempted,
        Then registration is rejected.
        """
        # Given
        processor = NonePropertyProcessor()
        validator = ProcessorValidator()

        # When
        result = validator.validate(processor)

        # Then
        assert result.valid is False
        assert any(
            e.check_id == "required_properties" and "None" in e.message
            for e in result.errors
        )


# =============================================================================
# GWT Scenario 3: Invalid Return Type Rejection
# =============================================================================


class TestInvalidReturnTypeRejection:
    """Tests for GWT Scenario 3: Invalid Return Type Rejection."""

    def test_valid_processor_has_return_types(self):
        """
        Given a processor with correct return type annotations,
        When validation is performed,
        Then no return type warnings are generated.
        """
        # Given
        processor = ValidProcessor()
        validator = ProcessorValidator()

        # When
        result = validator.validate(processor)

        # Then
        # Valid processor should pass
        assert result.valid is True


# =============================================================================
# GWT Scenario 4: Hash-Based Integrity Check
# =============================================================================


class TestHashBasedIntegrityCheck:
    """Tests for GWT Scenario 4: Hash-Based Integrity Check."""

    def test_given_known_hash_when_matches_then_valid(self):
        """
        Given a processor with a known code hash,
        When the hash matches,
        Then integrity check passes.
        """
        # Given
        processor = ValidProcessor()
        checker = IntegrityChecker()
        original_hash = checker.compute_hash(processor)
        checker.register_known_hash(processor.processor_id, original_hash)

        # When
        is_valid, message = checker.verify_integrity(processor)

        # Then
        assert is_valid is True
        assert message is None

    def test_given_known_hash_when_mismatch_then_flagged(self):
        """
        Given a processor with a known code hash,
        When the processor code has been modified (hash differs),
        Then the gatekeeper detects the change and flags it.
        """
        # Given
        processor = ValidProcessor()
        checker = IntegrityChecker()
        checker.register_known_hash(processor.processor_id, "fake-hash-12345")

        # When
        is_valid, message = checker.verify_integrity(processor)

        # Then
        assert is_valid is False
        assert message is not None
        assert "mismatch" in message.lower()

    def test_given_no_known_hash_when_verified_then_passes(self):
        """
        Given a processor without a registered hash,
        When integrity is verified,
        Then it passes (nothing to compare against).
        """
        # Given
        processor = ValidProcessor()
        checker = IntegrityChecker()

        # When
        is_valid, message = checker.verify_integrity(processor)

        # Then
        assert is_valid is True
        assert message is None


# =============================================================================
# GWT Scenario 5: Dependency Validation
# =============================================================================


class TestDependencyValidation:
    """Tests for GWT Scenario 5: Dependency Validation."""

    def test_given_unavailable_processor_when_validated_then_fails(self):
        """
        Given a processor declaring it is unavailable,
        When validation is performed,
        Then registration fails with dependency error.
        """
        # Given
        processor = UnavailableProcessor()
        gatekeeper = RegistryGatekeeper(check_dependencies=True)

        # When
        result = gatekeeper.validate(processor)

        # Then
        assert result.valid is False
        assert any(e.check_id == "dependencies" for e in result.errors)

    def test_given_exception_in_availability_when_validated_then_fails(self):
        """
        Given a processor whose is_available raises,
        When validation is performed,
        Then registration fails with dependency error.
        """
        # Given
        processor = ExceptionProcessor()
        gatekeeper = RegistryGatekeeper(check_dependencies=True)

        # When
        result = gatekeeper.validate(processor)

        # Then
        assert result.valid is False
        assert any(
            e.check_id == "dependencies" and "exception" in e.message.lower()
            for e in result.errors
        )

    def test_given_dependencies_disabled_when_unavailable_then_passes(self):
        """
        Given dependency checking is disabled,
        When an unavailable processor is validated,
        Then validation passes (dependency not checked).
        """
        # Given
        processor = UnavailableProcessor()
        gatekeeper = RegistryGatekeeper(check_dependencies=False)

        # When
        result = gatekeeper.validate(processor)

        # Then
        assert result.valid is True


# =============================================================================
# JPL Rule #2: Fixed Bounds
# =============================================================================


class TestJPLRule2FixedBounds:
    """Tests for JPL Rule #2: Fixed upper bounds."""

    def test_max_validation_checks_defined(self):
        """Verify MAX_VALIDATION_CHECKS constant is defined."""
        assert MAX_VALIDATION_CHECKS == 32
        assert MAX_VALIDATION_CHECKS > 0

    def test_max_hash_cache_size_defined(self):
        """Verify MAX_HASH_CACHE_SIZE constant is defined."""
        assert MAX_HASH_CACHE_SIZE == 256
        assert MAX_HASH_CACHE_SIZE > 0

    def test_hash_cache_bounded(self):
        """Test that hash cache respects size limit."""
        checker = IntegrityChecker()

        # Create many processors (more than cache size)
        for i in range(MAX_HASH_CACHE_SIZE + 50):
            mock_proc = MagicMock(spec=IFProcessor)
            mock_proc.processor_id = f"proc-{i}"
            mock_proc.version = "1.0.0"

            # Force source lookup to fail, using fallback
            with patch("inspect.getsource", side_effect=OSError):
                checker.compute_hash(mock_proc)

        # Cache should not exceed limit
        assert len(checker._hash_cache) <= MAX_HASH_CACHE_SIZE


# =============================================================================
# JPL Rule #7: Explicit Return Values
# =============================================================================


class TestJPLRule7ReturnValues:
    """Tests for JPL Rule #7: Explicit return values."""

    def test_validation_result_has_explicit_fields(self):
        """Verify ValidationResult has explicit fields."""
        result = ValidationResult(valid=True, errors=(), warnings=())
        assert hasattr(result, "valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "processor_hash")

    def test_validation_error_has_explicit_fields(self):
        """Verify ValidationError has explicit fields."""
        error = ValidationError(check_id="test", message="Test error", severity="error")
        assert hasattr(error, "check_id")
        assert hasattr(error, "message")
        assert hasattr(error, "severity")

    def test_verify_integrity_returns_tuple(self):
        """Verify verify_integrity returns explicit tuple."""
        checker = IntegrityChecker()
        processor = ValidProcessor()

        result = checker.verify_integrity(processor)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)


# =============================================================================
# JPL Rule #9: Type Hints
# =============================================================================


class TestJPLRule9TypeHints:
    """Tests for JPL Rule #9: Complete type hints."""

    def test_validation_check_has_type_hints(self):
        """Verify ValidationCheck has type annotations."""
        annotations = ValidationCheck.__dataclass_fields__
        assert "check_id" in annotations
        assert "description" in annotations
        assert "severity" in annotations

    def test_gatekeeper_validate_returns_typed_result(self):
        """Verify gatekeeper.validate returns typed result."""
        gatekeeper = RegistryGatekeeper()
        processor = ValidProcessor()

        result = gatekeeper.validate(processor)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.valid, bool)
        assert isinstance(result.errors, tuple)


# =============================================================================
# ProcessorValidator Tests
# =============================================================================


class TestProcessorValidator:
    """Tests for ProcessorValidator class."""

    def test_validates_required_methods_present(self):
        """Test validator checks for required methods."""
        processor = ValidProcessor()
        validator = ProcessorValidator()

        result = validator.validate(processor)

        assert result.valid is True

    def test_required_methods_list(self):
        """Test REQUIRED_METHODS contains expected entries."""
        assert "process" in REQUIRED_METHODS
        assert "is_available" in REQUIRED_METHODS

    def test_required_properties_list(self):
        """Test REQUIRED_PROPERTIES contains expected entries."""
        assert "processor_id" in REQUIRED_PROPERTIES
        assert "version" in REQUIRED_PROPERTIES


# =============================================================================
# IntegrityChecker Tests
# =============================================================================


class TestIntegrityChecker:
    """Tests for IntegrityChecker class."""

    def test_compute_hash_returns_hex_string(self):
        """Test hash computation returns valid hex string."""
        checker = IntegrityChecker()
        processor = ValidProcessor()

        hash_value = checker.compute_hash(processor)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_compute_hash_cached(self):
        """Test hash computation is cached."""
        checker = IntegrityChecker()
        processor = ValidProcessor()

        hash1 = checker.compute_hash(processor)
        hash2 = checker.compute_hash(processor)

        assert hash1 == hash2
        assert processor.processor_id in checker._hash_cache

    def test_clear_cache(self):
        """Test cache clearing."""
        checker = IntegrityChecker()
        processor = ValidProcessor()
        checker.compute_hash(processor)

        checker.clear_cache()

        assert len(checker._hash_cache) == 0


# =============================================================================
# DependencyChecker Tests
# =============================================================================


class TestDependencyChecker:
    """Tests for DependencyChecker class."""

    def test_available_processor_passes(self):
        """Test available processor passes dependency check."""
        checker = DependencyChecker()
        processor = ValidProcessor()

        errors = checker.check_dependencies(processor)

        assert len(errors) == 0

    def test_unavailable_processor_fails(self):
        """Test unavailable processor fails dependency check."""
        checker = DependencyChecker()
        processor = UnavailableProcessor()

        errors = checker.check_dependencies(processor)

        assert len(errors) == 1
        assert errors[0].check_id == "dependencies"


# =============================================================================
# RegistryGatekeeper Tests
# =============================================================================


class TestRegistryGatekeeper:
    """Tests for RegistryGatekeeper class."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        processor = ValidProcessor()
        gatekeeper = RegistryGatekeeper()

        result = gatekeeper.validate(processor)

        assert result.valid is True
        assert result.processor_hash is not None
        assert len(result.errors) == 0

    def test_integrity_checker_accessible(self):
        """Test integrity_checker property provides access."""
        gatekeeper = RegistryGatekeeper()

        checker = gatekeeper.integrity_checker

        assert isinstance(checker, IntegrityChecker)

    def test_register_known_hash(self):
        """Test registering known hash."""
        gatekeeper = RegistryGatekeeper()

        gatekeeper.register_known_hash("test-proc", "abc123")

        assert "test-proc" in gatekeeper.integrity_checker._known_hashes


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateGatekeeper:
    """Tests for create_gatekeeper factory function."""

    def test_creates_default_gatekeeper(self):
        """Test factory creates default gatekeeper."""
        gatekeeper = create_gatekeeper()

        assert isinstance(gatekeeper, RegistryGatekeeper)
        assert gatekeeper._check_integrity is True
        assert gatekeeper._check_dependencies is True

    def test_creates_with_options(self):
        """Test factory respects options."""
        gatekeeper = create_gatekeeper(check_integrity=False, check_dependencies=False)

        assert gatekeeper._check_integrity is False
        assert gatekeeper._check_dependencies is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestGatekeeperIntegration:
    """Integration tests for gatekeeper workflow."""

    def test_validate_multiple_processors(self):
        """Test validating multiple processors."""
        gatekeeper = RegistryGatekeeper()
        processors = [ValidProcessor(), ValidProcessor(), ValidProcessor()]

        results = [gatekeeper.validate(p) for p in processors]

        assert all(r.valid for r in results)

    def test_hash_integrity_across_validation(self):
        """Test hash remains consistent across validations."""
        processor = ValidProcessor()
        gatekeeper = RegistryGatekeeper()

        result1 = gatekeeper.validate(processor)
        result2 = gatekeeper.validate(processor)

        assert result1.processor_hash == result2.processor_hash

    def test_warnings_collected_separately(self):
        """Test warnings are separate from errors."""
        processor = ValidProcessor()
        gatekeeper = RegistryGatekeeper()

        result = gatekeeper.validate(processor)

        # Even with warnings, valid should be True if no errors
        assert result.valid is True
        assert isinstance(result.warnings, tuple)


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestAdditionalCoverage:
    """Additional tests to improve code coverage."""

    def test_processor_with_non_callable_method_attribute(self):
        """Test processor where method attribute is not callable."""

        class BadProcessor(IFProcessor):
            process = "not_a_method"  # Not callable

            def is_available(self) -> bool:
                return True

            @property
            def processor_id(self) -> str:
                return "bad-processor"

            @property
            def version(self) -> str:
                return "1.0.0"

        processor = BadProcessor()
        validator = ProcessorValidator()

        result = validator.validate(processor)

        assert result.valid is False
        assert any(
            e.check_id == "required_methods" and "not callable" in e.message
            for e in result.errors
        )

    def test_processor_with_all_required_properties(self):
        """Test processor with all required properties properly implemented."""
        processor = ValidProcessor()
        validator = ProcessorValidator()

        result = validator.validate(processor)

        assert result.valid is True
        # Both processor_id and version should be accessible
        assert processor.processor_id == "valid-processor"
        assert processor.version == "1.0.0"

    def test_processor_method_signature_inspection_fails(self):
        """Test when signature inspection fails."""

        class BuiltinMethodProcessor(IFProcessor):
            def __init__(self):
                # Replace process with a built-in that can't be inspected
                pass

            def process(self, artifact: IFArtifact) -> IFArtifact:
                return artifact

            def is_available(self) -> bool:
                return True

            @property
            def processor_id(self) -> str:
                return "builtin-processor"

            @property
            def version(self) -> str:
                return "1.0.0"

        processor = BuiltinMethodProcessor()
        validator = ProcessorValidator()

        # Patch signature to raise
        with patch("inspect.signature", side_effect=ValueError("Cannot inspect")):
            result = validator.validate(processor)

        # Should have a warning about signature inspection
        assert any(
            w.check_id == "method_signatures" and "Could not inspect" in w.message
            for w in result.warnings
        )

    def test_processor_missing_parameters(self):
        """Test processor with method missing required parameters."""

        class MissingParamProcessor(IFProcessor):
            def process(self) -> IFArtifact:  # Missing artifact parameter
                pass

            def is_available(self) -> bool:
                return True

            @property
            def processor_id(self) -> str:
                return "missing-param"

            @property
            def version(self) -> str:
                return "1.0.0"

        processor = MissingParamProcessor()
        validator = ProcessorValidator()

        result = validator.validate(processor)

        assert result.valid is False
        assert any(
            e.check_id == "method_signatures" and "missing parameters" in e.message
            for e in result.errors
        )

    def test_integrity_checker_builtin_fallback(self):
        """Test IntegrityChecker fallback when getsource fails."""
        checker = IntegrityChecker()
        processor = ValidProcessor()

        with patch("inspect.getsource", side_effect=OSError("No source")):
            hash_value = checker.compute_hash(processor)

        # Should still return a valid hash
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_validation_check_dataclass(self):
        """Test ValidationCheck dataclass properties."""
        check = ValidationCheck(
            check_id="test_check",
            description="A test validation check",
            severity="warning",
        )
        assert check.check_id == "test_check"
        assert check.description == "A test validation check"
        assert check.severity == "warning"

    def test_validation_error_with_details(self):
        """Test ValidationError with details field."""
        error = ValidationError(
            check_id="test",
            message="Test error",
            severity="error",
            details="Additional context",
        )
        assert error.details == "Additional context"

    def test_gatekeeper_integrity_disabled(self):
        """Test gatekeeper with integrity checking disabled."""
        processor = ValidProcessor()
        gatekeeper = RegistryGatekeeper(check_integrity=False)

        result = gatekeeper.validate(processor)

        assert result.valid is True
        assert result.processor_hash is None  # No hash computed

    def test_integrity_warning_on_hash_mismatch(self):
        """Test integrity warning is added on hash mismatch."""
        processor = ValidProcessor()
        gatekeeper = RegistryGatekeeper(check_integrity=True)

        # Register a fake hash that won't match
        gatekeeper.register_known_hash(processor.processor_id, "fake_hash_123")

        result = gatekeeper.validate(processor)

        # Still valid but should have warning
        assert result.valid is True
        assert any(
            w.check_id == "integrity" and "mismatch" in w.message.lower()
            for w in result.warnings
        )
