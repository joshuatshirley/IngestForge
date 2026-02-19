"""
Unit tests for errors module.

Tests ErrorCode enum and ERROR_SOLUTIONS registry.
"""

from ingestforge.core.errors import (
    ErrorCode,
    ErrorContext,
    ERROR_SOLUTIONS,
    SafeErrorMessage,
    sanitize_error,
    MAX_SUGGESTIONS,
)


class TestErrorCode:
    """Test ErrorCode enum."""

    def test_error_code_values_are_strings(self) -> None:
        """Given ErrorCode enum, when accessing values, then all are strings."""
        # Given/When
        for code in ErrorCode:
            # Then
            assert isinstance(code.value, str)
            assert code.value.startswith("E")

    def test_installation_error_codes_start_with_e0(self) -> None:
        """Given installation error codes, when checking values, then start with E0."""
        # Given
        install_codes = [
            ErrorCode.E001_PYTHON_VERSION,
            ErrorCode.E002_NODE_NOT_FOUND,
            ErrorCode.E003_DEPENDENCY_FAILED,
            ErrorCode.E004_INSTALL_TIMEOUT,
        ]

        # When/Then
        for code in install_codes:
            assert code.value.startswith("E00")

    def test_config_error_codes_start_with_e1(self) -> None:
        """Given config error codes, when checking values, then start with E1."""
        # Given
        config_codes = [
            ErrorCode.E101_CONFIG_NOT_FOUND,
            ErrorCode.E102_CONFIG_INVALID,
            ErrorCode.E103_MODEL_NOT_FOUND,
            ErrorCode.E104_API_KEY_MISSING,
            ErrorCode.E105_STORAGE_PATH_INVALID,
        ]

        # When/Then
        for code in config_codes:
            assert code.value.startswith("E10")

    def test_ingestion_error_codes_start_with_e2(self) -> None:
        """Given ingestion error codes, when checking values, then start with E2."""
        # Given
        ingest_codes = [
            ErrorCode.E201_FILE_NOT_FOUND,
            ErrorCode.E202_UNSUPPORTED_FORMAT,
            ErrorCode.E203_OCR_FAILED,
            ErrorCode.E204_PARSING_ERROR,
            ErrorCode.E205_FILE_TOO_LARGE,
        ]

        # When/Then
        for code in ingest_codes:
            assert code.value.startswith("E20")

    def test_query_error_codes_start_with_e3(self) -> None:
        """Given query error codes, when checking values, then start with E3."""
        # Given
        query_codes = [
            ErrorCode.E301_COLLECTION_EMPTY,
            ErrorCode.E302_QUERY_TIMEOUT,
            ErrorCode.E303_INVALID_QUERY,
            ErrorCode.E304_EMBEDDINGS_FAILED,
        ]

        # When/Then
        for code in query_codes:
            assert code.value.startswith("E30")

    def test_storage_error_codes_start_with_e4(self) -> None:
        """Given storage error codes, when checking values, then start with E4."""
        # Given
        storage_codes = [
            ErrorCode.E401_DB_CONNECTION,
            ErrorCode.E402_DISK_FULL,
            ErrorCode.E403_PERMISSION_DENIED,
            ErrorCode.E404_MIGRATION_FAILED,
        ]

        # When/Then
        for code in storage_codes:
            assert code.value.startswith("E40")

    def test_error_codes_are_unique(self) -> None:
        """Given all ErrorCode values, when checking, then all are unique."""
        # Given
        all_codes = [code.value for code in ErrorCode]

        # When
        unique_codes = set(all_codes)

        # Then
        assert len(all_codes) == len(unique_codes)

    def test_at_least_20_error_codes_defined(self) -> None:
        """Given ErrorCode enum, when counting members, then at least 20 exist."""
        # Given/When
        code_count = len([code for code in ErrorCode])

        # Then - AC requires 20+ error codes
        assert code_count >= 20


class TestErrorSolutionsRegistry:
    """Test ERROR_SOLUTIONS registry."""

    def test_registry_is_dict(self) -> None:
        """Given ERROR_SOLUTIONS, when checking type, then is dict."""
        # Then
        assert isinstance(ERROR_SOLUTIONS, dict)

    def test_all_codes_map_to_solutions(self) -> None:
        """Given all ErrorCode values, when checking registry, then all have entries."""
        # Given
        all_codes = [code for code in ErrorCode]

        # When/Then
        for code in all_codes:
            assert code in ERROR_SOLUTIONS

    def test_solution_structure_is_correct(self) -> None:
        """Given solution entries, when checking structure, then match expected format."""
        # Given
        expected_keys = {"message", "fixes", "docs"}

        # When/Then
        for code, solution in ERROR_SOLUTIONS.items():
            assert set(solution.keys()) == expected_keys, f"{code} has wrong keys"

    def test_solution_messages_are_strings(self) -> None:
        """Given solution messages, when checking type, then all are strings."""
        # Given/When/Then
        for code, solution in ERROR_SOLUTIONS.items():
            message = solution.get("message")
            assert isinstance(message, str), f"{code} message not a string"
            assert len(message) > 0, f"{code} message is empty"

    def test_solution_fixes_are_lists(self) -> None:
        """Given solution fixes, when checking type, then all are lists."""
        # Given/When/Then
        for code, solution in ERROR_SOLUTIONS.items():
            fixes = solution.get("fixes")
            assert isinstance(fixes, list), f"{code} fixes not a list"

    def test_solution_docs_are_strings(self) -> None:
        """Given solution docs, when checking type, then all are strings."""
        # Given/When/Then
        for code, solution in ERROR_SOLUTIONS.items():
            docs = solution.get("docs")
            assert isinstance(docs, str), f"{code} docs not a string"
            assert docs.startswith("https://"), f"{code} docs not a URL"

    def test_fixes_contain_actionable_steps(self) -> None:
        """Given fix lists, when checking content, then contain actionable verbs."""
        # Given
        actionable_verbs = {
            "install",
            "check",
            "run",
            "verify",
            "set",
            "try",
            "use",
            "get",
            "fix",
            "create",
            "download",
            "increase",
            "split",
            "process",
            "backup",
            "start",
            "add",
            "regenerate",
            "validate",
            "move",
            "rollback",
        }

        # When/Then
        for code, solution in ERROR_SOLUTIONS.items():
            fixes = solution.get("fixes", [])
            assert len(fixes) > 0, f"{code} has no fixes"

            # At least one fix should start with an actionable verb
            has_actionable = any(
                any(verb in fix.lower() for verb in actionable_verbs) for fix in fixes
            )
            assert has_actionable, f"{code} fixes not actionable"


class TestErrorContext:
    """Test ErrorContext TypedDict."""

    def test_error_context_accepts_all_fields(self) -> None:
        """Given ErrorContext, when creating with all fields, then accepts."""
        # Given/When
        context: ErrorContext = {
            "operation": "test",
            "file_path": "/path/to/file",
            "details": "some details",
            "command": "ingestforge test",
        }

        # Then
        assert context["operation"] == "test"
        assert context["file_path"] == "/path/to/file"
        assert context["details"] == "some details"
        assert context["command"] == "ingestforge test"

    def test_error_context_accepts_minimal_fields(self) -> None:
        """Given ErrorContext, when creating with minimal fields, then accepts."""
        # Given/When
        context: ErrorContext = {
            "operation": "test",
        }

        # Then
        assert context["operation"] == "test"

    def test_error_context_none_values_allowed(self) -> None:
        """Given ErrorContext, when using None for optional fields, then accepts."""
        # Given/When
        context: ErrorContext = {
            "operation": "test",
            "file_path": None,
            "details": None,
            "command": None,
        }

        # Then
        assert context["file_path"] is None


class TestSafeErrorMessage:
    """Test SafeErrorMessage class (existing functionality)."""

    def test_sanitize_removes_paths(self) -> None:
        """Given error with path, when sanitizing, then removes path."""
        # Given
        error = Exception("Failed to read C:\\Users\\test\\file.pdf")

        # When
        result = SafeErrorMessage.sanitize(error, "file read")

        # Then
        assert "C:\\Users\\test\\file.pdf" not in result
        assert "file read" in result

    def test_sanitize_message_removes_sensitive_patterns(self) -> None:
        """Given message with sensitive data, when sanitizing, then removes it."""
        # Given
        message = "Error: api_key=sk-1234567890 failed"

        # When
        result = SafeErrorMessage.sanitize_message(message)

        # Then
        assert "sk-1234567890" not in result
        assert "[REDACTED]" in result

    def test_sanitize_error_convenience_function(self) -> None:
        """Given error, when using convenience function, then sanitizes."""
        # Given
        error = Exception("test error")

        # When
        result = sanitize_error(error, "test operation")

        # Then
        assert isinstance(result, str)
        assert "test operation" in result


class TestJPLCompliance:
    """Test JPL Power of Ten compliance."""

    def test_max_suggestions_constant_is_bounded(self) -> None:
        """Given MAX_SUGGESTIONS, when checking value, then is bounded constant."""
        # Then - JPL Rule #2
        assert MAX_SUGGESTIONS == 3
        assert isinstance(MAX_SUGGESTIONS, int)

    def test_error_code_has_type_hints(self) -> None:
        """Given ErrorCode, when checking base classes, then uses str and Enum."""
        # Then - JPL Rule #9
        assert issubclass(ErrorCode, str)
        from enum import Enum

        assert issubclass(ErrorCode, Enum)
