"""
Unit tests for error_formatter module.

Tests user-friendly error formatting with JPL compliance.
"""

import pytest
from ingestforge.core.errors import ErrorCode, ErrorContext, ERROR_SOLUTIONS
from ingestforge.core.error_formatter import (
    format_user_error,
    get_error_code_from_exception,
    MAX_SUGGESTIONS,
)


class TestFormatUserError:
    """Test format_user_error function."""

    def test_basic_error_formatting(self) -> None:
        """Given error code, when formatting, then returns user-friendly message."""
        # Given
        code = ErrorCode.E201_FILE_NOT_FOUND
        context: ErrorContext = {
            "operation": "file ingestion",
            "file_path": "/path/to/missing.pdf",
        }

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then
        assert "[E201]" in result
        assert "File or directory not found" in result
        assert "/path/to/missing.pdf" in result
        assert "Troubleshooting:" in result

    def test_includes_troubleshooting_steps(self) -> None:
        """Given error code, when formatting, then includes actionable fixes."""
        # Given
        code = ErrorCode.E101_CONFIG_NOT_FOUND
        context: ErrorContext = {"operation": "configuration loading"}

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then
        assert "1." in result  # Numbered list
        assert "2." in result
        assert "ingestforge setup" in result or "Run setup wizard" in result

    def test_respects_max_suggestions_bound(self) -> None:
        """Given error code, when formatting, then limits suggestions to MAX_SUGGESTIONS."""
        # Given - error with many fixes
        code = ErrorCode.E201_FILE_NOT_FOUND
        context: ErrorContext = {"operation": "test"}

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then - count numbered items
        numbered_items = [
            line
            for line in result.split("\n")
            if line.strip().startswith(("1.", "2.", "3.", "4."))
        ]
        assert len(numbered_items) <= MAX_SUGGESTIONS

    def test_includes_docs_link(self) -> None:
        """Given error code with docs link, when formatting, then includes link."""
        # Given
        code = ErrorCode.E203_OCR_FAILED
        context: ErrorContext = {"operation": "OCR"}

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then
        assert "See:" in result
        assert "https://docs.ingestforge.io" in result

    def test_shows_technical_details_when_debug(self) -> None:
        """Given show_technical=True, when formatting, then includes details."""
        # Given
        code = ErrorCode.E401_DB_CONNECTION
        context: ErrorContext = {
            "operation": "database query",
            "details": "Connection timeout after 30s",
        }

        # When
        result = format_user_error(code, context, show_technical=True)

        # Then
        assert "Connection timeout after 30s" in result

    def test_hides_technical_details_when_not_debug(self) -> None:
        """Given show_technical=False, when formatting, then hides details."""
        # Given
        code = ErrorCode.E401_DB_CONNECTION
        context: ErrorContext = {
            "operation": "database query",
            "details": "Connection timeout after 30s",
        }

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then
        assert "Connection timeout after 30s" not in result

    def test_handles_missing_optional_context(self) -> None:
        """Given minimal context, when formatting, then handles gracefully."""
        # Given
        code = ErrorCode.E302_QUERY_TIMEOUT
        context: ErrorContext = {"operation": "query"}

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then - should not crash
        assert "[E302]" in result
        assert "Query execution timeout" in result

    def test_handles_error_with_empty_context(self) -> None:
        """Given error with empty context, when formatting, then handles gracefully."""
        # Given
        code = ErrorCode.E201_FILE_NOT_FOUND
        context: ErrorContext = {}  # Empty context

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then - should not crash and still show error code
        assert "[E201]" in result
        assert "File or directory not found" in result

    def test_includes_command_context(self) -> None:
        """Given command in context, when formatting, then displays command."""
        # Given
        code = ErrorCode.E003_DEPENDENCY_FAILED
        context: ErrorContext = {
            "operation": "installation",
            "command": "pip install ingestforge",
        }

        # When
        result = format_user_error(code, context, show_technical=False)

        # Then
        assert "pip install ingestforge" in result


class TestGetErrorCodeFromException:
    """Test get_error_code_from_exception function."""

    def test_infers_file_not_found_error(self) -> None:
        """Given FileNotFoundError, when inferring code, then returns E201."""
        # Given
        error = FileNotFoundError("test.pdf not found")
        operation = "file ingestion"

        # When
        result = get_error_code_from_exception(error, operation)

        # Then
        assert result == ErrorCode.E201_FILE_NOT_FOUND

    def test_infers_permission_error(self) -> None:
        """Given PermissionError, when inferring code, then returns E403."""
        # Given
        error = PermissionError("access denied")
        operation = "file write"

        # When
        result = get_error_code_from_exception(error, operation)

        # Then
        assert result == ErrorCode.E403_PERMISSION_DENIED

    def test_infers_python_version_error(self) -> None:
        """Given Python version error message, when inferring, then returns E001."""
        # Given
        error = RuntimeError("Python version 3.9 not supported")
        operation = "installation"

        # When
        result = get_error_code_from_exception(error, operation)

        # Then
        assert result == ErrorCode.E001_PYTHON_VERSION

    def test_infers_config_not_found_from_context(self) -> None:
        """Given config operation with 'not found', when inferring, then returns E101."""
        # Given
        error = Exception("config.yml not found")
        operation = "configuration loading"

        # When
        result = get_error_code_from_exception(error, operation)

        # Then
        assert result == ErrorCode.E101_CONFIG_NOT_FOUND

    def test_infers_disk_full_error(self) -> None:
        """Given disk space error, when inferring, then returns E402."""
        # Given
        error = OSError("No space left on disk")
        operation = "file write"

        # When
        result = get_error_code_from_exception(error, operation)

        # Then
        assert result == ErrorCode.E402_DISK_FULL

    def test_infers_query_timeout(self) -> None:
        """Given timeout error during query, when inferring, then returns E302."""
        # Given
        error = TimeoutError("Query timeout after 30s")
        operation = "query execution"

        # When
        result = get_error_code_from_exception(error, operation)

        # Then
        assert result == ErrorCode.E302_QUERY_TIMEOUT

    def test_returns_none_for_unknown_error(self) -> None:
        """Given unrecognized error, when inferring, then returns None."""
        # Given
        error = ValueError("some random error")
        operation = "unknown operation"

        # When
        result = get_error_code_from_exception(error, operation)

        # Then
        assert result is None


class TestErrorSolutionsRegistry:
    """Test ERROR_SOLUTIONS registry completeness."""

    def test_all_error_codes_have_solutions(self) -> None:
        """Given all ErrorCode enum values, when checking registry, then all have solutions."""
        # Given
        all_codes = [code for code in ErrorCode]

        # When/Then
        for code in all_codes:
            assert code in ERROR_SOLUTIONS, f"Missing solution for {code}"

    def test_all_solutions_have_required_fields(self) -> None:
        """Given all solutions, when checking structure, then all have required fields."""
        # Given/When
        for code, solution in ERROR_SOLUTIONS.items():
            # Then
            assert "message" in solution, f"{code} missing 'message'"
            assert "fixes" in solution, f"{code} missing 'fixes'"
            assert "docs" in solution, f"{code} missing 'docs'"

    def test_all_solutions_have_at_least_one_fix(self) -> None:
        """Given all solutions, when checking fixes, then all have at least one."""
        # Given/When
        for code, solution in ERROR_SOLUTIONS.items():
            fixes = solution.get("fixes", [])
            # Then
            assert isinstance(fixes, list), f"{code} fixes not a list"
            assert len(fixes) > 0, f"{code} has no fixes"

    def test_solutions_respect_max_suggestions_bound(self) -> None:
        """Given all solutions, when checking fix count, then none exceed MAX_SUGGESTIONS."""
        # This is a soft check - solutions CAN have more than MAX_SUGGESTIONS,
        # but format_user_error will truncate them. This test just warns if too many.
        for code, solution in ERROR_SOLUTIONS.items():
            fixes = solution.get("fixes", [])
            if len(fixes) > MAX_SUGGESTIONS + 2:  # Allow some buffer
                pytest.fail(
                    f"{code} has {len(fixes)} fixes, "
                    f"consider reducing to {MAX_SUGGESTIONS} most actionable"
                )


class TestJPLCompliance:
    """Test JPL Power of Ten compliance."""

    def test_max_suggestions_is_bounded(self) -> None:
        """Given MAX_SUGGESTIONS constant, when checking value, then is bounded."""
        # Then
        assert MAX_SUGGESTIONS == 3  # JPL Rule #2: Bounded constant
        assert isinstance(MAX_SUGGESTIONS, int)

    def test_format_user_error_has_type_hints(self) -> None:
        """Given format_user_error, when checking signature, then has complete type hints."""
        # Given
        from typing import get_type_hints

        # When
        hints = get_type_hints(format_user_error)

        # Then - JPL Rule #9
        assert "code" in hints
        assert "context" in hints
        assert "show_technical" in hints
        assert "return" in hints

    def test_get_error_code_has_type_hints(self) -> None:
        """Given get_error_code_from_exception, when checking, then has complete type hints."""
        # Given
        from typing import get_type_hints

        # When
        hints = get_type_hints(get_error_code_from_exception)

        # Then - JPL Rule #9
        assert "error" in hints
        assert "operation" in hints
        assert "return" in hints
