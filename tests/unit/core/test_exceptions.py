"""
Tests for Exception Hierarchy.

This module tests the custom exception classes used throughout IngestForge.
All exceptions should inherit from IngestForgeError.

Test Strategy
-------------
- Focus on exception creation and inheritance
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test custom attributes for exceptions that have them
- Test catchability (all should be catchable as IngestForgeError)

Organization
------------
- TestBaseException: IngestForgeError
- TestSecurityExceptions: SecurityError, PathTraversalError, SSRFError
- TestProcessingExceptions: ProcessingError, ExtractionError, ChunkingError, etc.
- TestLLMExceptions: LLMError, RateLimitError, ConfigurationError, ContextLengthError
- TestStorageExceptions: StorageError, IndexError
- TestInfrastructureExceptions: RetryError, TimeoutError
- TestValidationExceptions: ValidationError, ConfigValidationError
"""

import pytest

from ingestforge.core.exceptions import (
    # Base
    IngestForgeError,
    # Security
    SecurityError,
    PathTraversalError,
    SSRFError,
    # Processing
    ProcessingError,
    ExtractionError,
    ChunkingError,
    EnrichmentError,
    EmbeddingError,
    # LLM
    LLMError,
    RateLimitError,
    ConfigurationError,
    ContextLengthError,
    # Storage
    StorageError,
    IndexError,
    # Infrastructure
    RetryError,
    TimeoutError,
    # Validation
    ValidationError,
    ConfigValidationError,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestBaseException:
    """Tests for IngestForgeError base exception.

    Rule #4: Focused test class - tests only IngestForgeError
    """

    def test_create_base_exception(self):
        """Test creating base IngestForgeError."""
        error = IngestForgeError("test error")

        assert str(error) == "test error"

    def test_raise_and_catch_base_exception(self):
        """Test raising and catching IngestForgeError."""
        with pytest.raises(IngestForgeError) as exc_info:
            raise IngestForgeError("test")

        assert "test" in str(exc_info.value)

    def test_base_is_exception(self):
        """Test that IngestForgeError inherits from Exception."""
        error = IngestForgeError("test")

        assert isinstance(error, Exception)


class TestSecurityExceptions:
    """Tests for security-related exceptions.

    Rule #4: Focused test class - tests security exceptions only
    """

    def test_security_error_inheritance(self):
        """Test SecurityError inherits from IngestForgeError."""
        error = SecurityError("security issue")

        assert isinstance(error, IngestForgeError)

    def test_path_traversal_error(self):
        """Test PathTraversalError creation and catching."""
        with pytest.raises(PathTraversalError) as exc_info:
            raise PathTraversalError("Path traversal detected: ../../../etc/passwd")

        assert "traversal" in str(exc_info.value)

    def test_path_traversal_catchable_as_security_error(self):
        """Test PathTraversalError can be caught as SecurityError."""
        with pytest.raises(SecurityError):
            raise PathTraversalError("test")

    def test_ssrf_error(self):
        """Test SSRFError creation and catching."""
        with pytest.raises(SSRFError) as exc_info:
            raise SSRFError("URL resolves to private IP")

        assert "private" in str(exc_info.value)

    def test_ssrf_catchable_as_security_error(self):
        """Test SSRFError can be caught as SecurityError."""
        with pytest.raises(SecurityError):
            raise SSRFError("test")


class TestProcessingExceptions:
    """Tests for processing-related exceptions.

    Rule #4: Focused test class - tests processing exceptions only
    """

    def test_processing_error_inheritance(self):
        """Test ProcessingError inherits from IngestForgeError."""
        error = ProcessingError("processing failed")

        assert isinstance(error, IngestForgeError)

    def test_extraction_error(self):
        """Test ExtractionError creation."""
        error = ExtractionError("PDF extraction failed")

        assert "extraction" in str(error).lower()

    def test_extraction_catchable_as_processing_error(self):
        """Test ExtractionError can be caught as ProcessingError."""
        with pytest.raises(ProcessingError):
            raise ExtractionError("test")

    def test_chunking_error(self):
        """Test ChunkingError creation."""
        error = ChunkingError("Text too short to chunk")

        assert "chunk" in str(error).lower()

    def test_enrichment_error(self):
        """Test EnrichmentError creation."""
        error = EnrichmentError("Entity extraction failed")

        assert "extraction" in str(error).lower()

    def test_embedding_error_inheritance(self):
        """Test EmbeddingError inherits from EnrichmentError."""
        error = EmbeddingError("GPU memory exhausted")

        assert isinstance(error, EnrichmentError)
        assert isinstance(error, ProcessingError)


class TestLLMExceptions:
    """Tests for LLM-related exceptions.

    Rule #4: Focused test class - tests LLM exceptions only
    """

    def test_llm_error_inheritance(self):
        """Test LLMError inherits from IngestForgeError."""
        error = LLMError("API call failed")

        assert isinstance(error, IngestForgeError)

    def test_rate_limit_error_basic(self):
        """Test RateLimitError without retry_after."""
        error = RateLimitError("Rate limit exceeded")

        assert str(error) == "Rate limit exceeded"
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after attribute."""
        error = RateLimitError("Rate limit exceeded", retry_after=60.0)

        assert error.retry_after == 60.0

    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError("API key missing")

        assert "API key" in str(error)

    def test_context_length_error_basic(self):
        """Test ContextLengthError without token counts."""
        error = ContextLengthError("Input too long")

        assert error.max_tokens is None
        assert error.actual_tokens is None

    def test_context_length_error_with_tokens(self):
        """Test ContextLengthError with token counts."""
        error = ContextLengthError(
            "Input exceeds context window", max_tokens=4096, actual_tokens=5000
        )

        assert error.max_tokens == 4096
        assert error.actual_tokens == 5000


class TestStorageExceptions:
    """Tests for storage-related exceptions.

    Rule #4: Focused test class - tests storage exceptions only
    """

    def test_storage_error_inheritance(self):
        """Test StorageError inherits from IngestForgeError."""
        error = StorageError("Database connection failed")

        assert isinstance(error, IngestForgeError)

    def test_index_error_inheritance(self):
        """Test IndexError inherits from StorageError."""
        error = IndexError("BM25 index corrupted")

        assert isinstance(error, StorageError)
        assert isinstance(error, IngestForgeError)


class TestInfrastructureExceptions:
    """Tests for infrastructure-related exceptions.

    Rule #4: Focused test class - tests infrastructure exceptions only
    """

    def test_retry_error_basic(self):
        """Test RetryError without attributes."""
        error = RetryError("All retries exhausted")

        assert error.attempts is None
        assert error.last_exception is None

    def test_retry_error_with_attributes(self):
        """Test RetryError with attempts and last_exception."""
        original = ValueError("original error")
        error = RetryError(
            "Failed after 3 attempts", attempts=3, last_exception=original
        )

        assert error.attempts == 3
        assert error.last_exception is original

    def test_timeout_error_basic(self):
        """Test TimeoutError without attributes."""
        error = TimeoutError("Operation timed out")

        assert error.timeout is None
        assert error.operation is None

    def test_timeout_error_with_attributes(self):
        """Test TimeoutError with timeout and operation."""
        error = TimeoutError(
            "LLM request timed out", timeout=30.0, operation="LLM API call"
        )

        assert error.timeout == 30.0
        assert error.operation == "LLM API call"


class TestValidationExceptions:
    """Tests for validation-related exceptions.

    Rule #4: Focused test class - tests validation exceptions only
    """

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from IngestForgeError."""
        error = ValidationError("Input validation failed")

        assert isinstance(error, IngestForgeError)

    def test_config_validation_error_basic(self):
        """Test ConfigValidationError without attributes."""
        error = ConfigValidationError("Invalid configuration")

        assert error.field is None
        assert error.value is None

    def test_config_validation_error_with_attributes(self):
        """Test ConfigValidationError with field and value."""
        error = ConfigValidationError(
            "Invalid chunk size", field="chunk_size", value=-1
        )

        assert error.field == "chunk_size"
        assert error.value == -1

    def test_config_validation_catchable_as_validation_error(self):
        """Test ConfigValidationError can be caught as ValidationError."""
        with pytest.raises(ValidationError):
            raise ConfigValidationError("test")


class TestExceptionHierarchy:
    """Tests for overall exception hierarchy behavior.

    Rule #4: Focused test class - tests hierarchy behavior only
    """

    def test_all_exceptions_catchable_as_base(self):
        """Test that all exceptions can be caught as IngestForgeError."""
        exceptions = [
            PathTraversalError("test"),
            SSRFError("test"),
            ExtractionError("test"),
            ChunkingError("test"),
            EmbeddingError("test"),
            RateLimitError("test"),
            ConfigurationError("test"),
            StorageError("test"),
            RetryError("test"),
            TimeoutError("test"),
            ValidationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, IngestForgeError)

    def test_exception_messages_preserved(self):
        """Test that exception messages are preserved."""
        message = "Custom error message with details"
        error = ProcessingError(message)

        assert str(error) == message


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Base exception: 3 tests (creation, catching, inheritance)
    - Security exceptions: 5 tests (SecurityError, PathTraversalError, SSRFError)
    - Processing exceptions: 6 tests (all processing exception types)
    - LLM exceptions: 5 tests (basic + custom attributes)
    - Storage exceptions: 2 tests (StorageError, IndexError)
    - Infrastructure exceptions: 4 tests (RetryError, TimeoutError with attributes)
    - Validation exceptions: 4 tests (ValidationError, ConfigValidationError)
    - Hierarchy tests: 2 tests (catchability, message preservation)

    Total: 31 tests

Design Decisions:
    1. Focus on exception creation and inheritance
    2. Test custom attributes (retry_after, max_tokens, attempts, etc.)
    3. Test catchability at different hierarchy levels
    4. Don't test every possible message - test representative cases
    5. Simple, clear tests that verify exception behavior
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - Exception creation with messages
    - Exception inheritance hierarchy
    - Custom attribute storage (retry_after, max_tokens, etc.)
    - Catchability as parent exception types
    - All exceptions catchable as IngestForgeError

Justification:
    - Exceptions are simple data structures
    - Key behavior is inheritance and attributes
    - Don't need to test Exception base class (stdlib)
    - Focus on what developers actually use
    - Ensure all custom exceptions work as expected
"""
