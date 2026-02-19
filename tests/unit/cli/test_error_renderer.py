"""
Tests for UX-004: Helpful Error Messages.

This module tests the ErrorRenderer and error handling utilities.

Test Strategy
-------------
- Test error mapping from exceptions to helpful messages
- Test sensitive info sanitization (paths, API keys)
- Test root cause extraction from nested exceptions
- Test verbose mode toggle for tracebacks
- Follows NASA JPL Rule #1 (Simple Control Flow)

Organization
------------
- TestSanitization: Path and message sanitization
- TestRootCause: Nested exception handling
- TestErrorInfo: Exception to info mapping
- TestErrorRenderer: Panel rendering
"""


from ingestforge.core.exceptions import (
    IngestForgeError,
    ProcessingError,
    ExtractionError,
    LLMError,
    RateLimitError,
    ConfigurationError,
    StorageError,
    DependencyError,
    ConfigValidationError,
    sanitize_path,
    sanitize_message,
    get_root_cause,
    get_error_info,
)
from ingestforge.cli.console import (
    set_verbose_mode,
    is_verbose_mode,
)


class TestSanitizePath:
    """Tests for path sanitization (Rule #7: sanitize sensitive info)."""

    def test_sanitize_empty_path(self):
        """Test that empty path returns empty."""
        assert sanitize_path("") == ""
        assert sanitize_path(None) is None

    def test_sanitize_windows_user_path(self):
        """Test Windows user directory is sanitized."""
        path = r"C:\Users\johndoe\Documents\secret.txt"
        result = sanitize_path(path)

        assert "johndoe" not in result
        assert "<user-home>" in result

    def test_sanitize_unix_home_path(self):
        """Test Unix home directory is sanitized."""
        path = "/home/johndoe/.config/ingestforge/config.yaml"
        result = sanitize_path(path)

        assert "johndoe" not in result
        assert "<user-home>" in result

    def test_sanitize_mac_users_path(self):
        """Test Mac /Users directory is sanitized."""
        path = "/Users/johndoe/Library/Application Support/ingestforge"
        result = sanitize_path(path)

        assert "johndoe" not in result
        assert "<user-home>" in result

    def test_sanitize_long_key_in_path(self):
        """Test that long alphanumeric strings (potential keys) are sanitized."""
        # 32+ char string that could be an API key
        path = "/var/data/key_placeholder_for_sanitization_testing_12345/file.txt"
        result = sanitize_path(path)

        assert "key_placeholder_for_sanitization_testing_12345" not in result

    def test_preserve_safe_paths(self):
        """Test that safe paths are preserved."""
        path = "/var/log/ingestforge/app.log"
        result = sanitize_path(path)

        # Should not modify paths without user info
        assert "ingestforge" in result
        assert "app.log" in result


class TestSanitizeMessage:
    """Tests for error message sanitization (Rule #7)."""

    def test_sanitize_empty_message(self):
        """Test that empty message returns empty."""
        assert sanitize_message("") == ""
        assert sanitize_message(None) is None

    def test_sanitize_openai_api_key(self):
        """Test OpenAI API key pattern is sanitized."""
        message = (
            "Error: OPENAI_API_KEY=sk-abcdef123456789012345678901234567890 is invalid"
        )
        result = sanitize_message(message)

        assert "sk-abcdef123456789012345678901234567890" not in result
        assert "OPENAI_API_KEY" in result  # Keep the key name

    def test_sanitize_anthropic_api_key(self):
        """Test Anthropic API key pattern is sanitized."""
        message = "Error: ANTHROPIC_API_KEY: sk-ant-abc123def456ghi789jkl012 failed"
        result = sanitize_message(message)

        assert "sk-ant-abc123def456ghi789jkl012" not in result

    def test_sanitize_bearer_token(self):
        """Test Bearer token is sanitized."""
        message = (
            "Authorization failed: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdef"
        )
        result = sanitize_message(message)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "Bearer" in result

    def test_sanitize_basic_auth_url(self):
        """Test basic auth credentials in URL are sanitized."""
        message = "Connection to https://user:password123@api.example.com failed"
        result = sanitize_message(message)

        assert "password123" not in result
        assert "user" not in result or "<user>" in result

    def test_sanitize_long_hex_string(self):
        """Test long hex strings (potential hashes/keys) are sanitized."""
        # 40+ char hex string
        message = (
            "Hash mismatch: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8"
        )
        result = sanitize_message(message)

        # The long hex should be replaced
        assert "<hash>" in result or "a1b2c3d4e5f6a7b8c9d0" not in result

    def test_preserve_normal_message(self):
        """Test that normal messages without sensitive data are preserved."""
        message = "File processing failed: invalid format"
        result = sanitize_message(message)

        assert result == message

    def test_sanitize_file_paths_in_message(self):
        """Test that file paths in messages are sanitized."""
        message = "Cannot read C:\\Users\\johndoe\\Documents\\secret.pdf"
        result = sanitize_message(message)

        assert "johndoe" not in result


class TestGetRootCause:
    """Tests for root cause extraction from nested exceptions."""

    def test_single_exception(self):
        """Test that single exception returns itself."""
        exc = ValueError("test error")

        result = get_root_cause(exc)

        assert result is exc

    def test_chained_exception_with_cause(self):
        """Test extraction of root cause from __cause__ chain."""
        root = ValueError("root error")
        middle = RuntimeError("middle error")
        middle.__cause__ = root
        top = ProcessingError("top error")
        top.__cause__ = middle

        result = get_root_cause(top)

        assert result is root

    def test_chained_exception_with_context(self):
        """Test extraction of root cause from __context__ chain."""
        root = ValueError("root error")
        middle = RuntimeError("middle error")
        middle.__context__ = root
        top = ProcessingError("top error")
        top.__context__ = middle

        result = get_root_cause(top)

        assert result is root

    def test_prefer_cause_over_context(self):
        """Test that __cause__ is preferred over __context__."""
        context_root = KeyError("context root")
        cause_root = ValueError("cause root")

        exc = RuntimeError("test")
        exc.__context__ = context_root
        exc.__cause__ = cause_root

        result = get_root_cause(exc)

        assert result is cause_root

    def test_avoid_infinite_loop(self):
        """Test that circular references don't cause infinite loops."""
        exc1 = ValueError("error 1")
        exc2 = RuntimeError("error 2")

        # Create circular reference
        exc1.__cause__ = exc2
        exc2.__cause__ = exc1

        # Should not hang
        result = get_root_cause(exc1)

        # Should return one of them
        assert result in (exc1, exc2)


class TestGetErrorInfo:
    """Tests for exception to error info mapping."""

    def test_ingestforge_error_info(self):
        """Test that IngestForgeError attributes are extracted."""
        error = ProcessingError("test error")

        info = get_error_info(error)

        assert "error_code" in info
        assert info["error_code"].startswith("IF-")
        assert "why_it_happened" in info
        assert "how_to_fix" in info
        assert isinstance(info["how_to_fix"], list)

    def test_rate_limit_error_info(self):
        """Test RateLimitError has specific info."""
        error = RateLimitError("rate limit exceeded")

        info = get_error_info(error)

        assert info["error_code"] == "IF-LLM-001"
        assert "rate limit" in info["why_it_happened"].lower()

    def test_standard_file_not_found_error(self):
        """Test standard FileNotFoundError gets mapped."""
        import builtins

        error = builtins.FileNotFoundError("file.txt")

        info = get_error_info(error)

        assert info["error_code"] == "IF-FILE-001"
        assert len(info["how_to_fix"]) > 0

    def test_standard_permission_error(self):
        """Test standard PermissionError gets mapped."""
        error = PermissionError("access denied")

        info = get_error_info(error)

        assert info["error_code"] == "IF-FILE-002"

    def test_module_not_found_error(self):
        """Test ModuleNotFoundError gets mapped."""
        error = ModuleNotFoundError("No module named 'missing'")

        info = get_error_info(error)

        assert info["error_code"] == "IF-DEP-001"
        assert "pip install" in " ".join(info["how_to_fix"]).lower()

    def test_unknown_error_fallback(self):
        """Test unknown exception types get fallback info."""

        class CustomError(Exception):
            pass

        error = CustomError("custom error")

        info = get_error_info(error)

        assert info["error_code"] == "IF-ERR-999"
        assert len(info["how_to_fix"]) > 0


class TestErrorRendererVerboseMode:
    """Tests for verbose mode toggle."""

    def test_verbose_mode_default_off(self):
        """Test that verbose mode is off by default."""
        # Reset to default
        set_verbose_mode(False)

        assert is_verbose_mode() is False

    def test_verbose_mode_toggle_on(self):
        """Test enabling verbose mode."""
        set_verbose_mode(True)

        assert is_verbose_mode() is True

        # Clean up
        set_verbose_mode(False)

    def test_verbose_mode_toggle_off(self):
        """Test disabling verbose mode."""
        set_verbose_mode(True)
        set_verbose_mode(False)

        assert is_verbose_mode() is False


class TestIngestForgeErrorAttributes:
    """Tests for IngestForgeError class attributes."""

    def test_default_attributes(self):
        """Test default error attributes."""
        error = IngestForgeError("test error")

        assert hasattr(error, "error_code")
        assert hasattr(error, "why_it_happened")
        assert hasattr(error, "how_to_fix")

    def test_custom_attributes(self):
        """Test overriding default attributes."""
        error = IngestForgeError(
            "custom error",
            error_code="IF-CUSTOM-001",
            why_it_happened="Custom reason",
            how_to_fix=["Fix step 1", "Fix step 2"],
        )

        assert error.error_code == "IF-CUSTOM-001"
        assert error.why_it_happened == "Custom reason"
        assert len(error.how_to_fix) == 2

    def test_user_message_property(self):
        """Test user_message property returns string representation."""
        error = IngestForgeError("test message")

        assert error.user_message == "test message"

    def test_message_sanitization(self):
        """Test that message is sanitized on creation."""
        error = IngestForgeError(
            "Error with OPENAI_API_KEY=sk-test12345678901234567890123456789012345678"
        )

        # The message should be sanitized
        assert "sk-test12345678901234567890123456789012345678" not in str(error)

    def test_subclass_attributes(self):
        """Test that subclasses have their own default attributes."""
        proc_error = ProcessingError("processing failed")
        llm_error = LLMError("llm failed")

        assert proc_error.error_code != llm_error.error_code
        assert proc_error.why_it_happened != llm_error.why_it_happened


class TestSpecificExceptionInfo:
    """Tests for specific exception type helpful info."""

    def test_extraction_error_info(self):
        """Test ExtractionError has extraction-specific help."""
        error = ExtractionError("PDF extraction failed")

        assert "IF-PROC-001" == error.error_code
        assert "extract" in error.why_it_happened.lower()
        # Should mention OCR or format conversion in fixes
        fixes = " ".join(error.how_to_fix).lower()
        assert "tesseract" in fixes or "format" in fixes

    def test_configuration_error_info(self):
        """Test ConfigurationError has config-specific help."""
        error = ConfigurationError("API key missing")

        assert "IF-LLM-002" == error.error_code
        fixes = " ".join(error.how_to_fix).lower()
        # Should mention setting API keys
        assert "api key" in fixes or "api_key" in fixes

    def test_storage_error_info(self):
        """Test StorageError has storage-specific help."""
        error = StorageError("Database locked")

        assert error.error_code.startswith("IF-STOR")
        fixes = " ".join(error.how_to_fix).lower()
        # Should mention disk space or rebuild
        assert "disk" in fixes or "rebuild" in fixes

    def test_dependency_error_info(self):
        """Test DependencyError has install instructions."""
        error = DependencyError("pdfplumber not installed")

        assert "IF-DEP-001" == error.error_code
        fixes = " ".join(error.how_to_fix).lower()
        assert "pip install" in fixes

    def test_config_validation_error_attributes(self):
        """Test ConfigValidationError preserves field and value."""
        error = ConfigValidationError(
            "Invalid chunk_size",
            field="chunk_size",
            value=-100,
        )

        assert error.field == "chunk_size"
        assert error.value == -100


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Path sanitization: 6 tests (Windows, Unix, Mac paths, keys)
    - Message sanitization: 7 tests (API keys, tokens, URLs, hashes)
    - Root cause extraction: 5 tests (single, chained, circular)
    - Error info mapping: 6 tests (custom, standard, fallback)
    - Verbose mode: 3 tests (default, on, off)
    - IngestForgeError attributes: 5 tests (default, custom, sanitization)
    - Specific exception info: 5 tests (extraction, config, storage, dep)

    Total: 37 tests

Design Decisions:
    1. Focus on sensitive info sanitization (Rule #7)
    2. Test error mapping completeness
    3. Verify root cause extraction for nested exceptions
    4. Test verbose mode toggle
    5. Verify specific exception types have helpful info

Behaviors Tested:
    - Path sanitization removes user directories and keys
    - Message sanitization removes API keys, tokens, hashes
    - Root cause extraction handles chains and cycles
    - Error info provides actionable fix suggestions
    - Verbose mode controls traceback display
"""
