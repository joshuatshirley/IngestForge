"""Input validation utilities for CLI commands.

This module provides reusable validators following Commandment #7
(Check Parameters & Returns). All validators raise typer.BadParameter
with clear, actionable error messages.

Eliminates ~40 lines of duplicate validation code.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer


class InputValidator:
    """Reusable input validators for CLI commands.

    All validation methods are static and raise typer.BadParameter
    on validation failure with clear error messages.

    Example:
        InputValidator.validate_project_name(name)
        path = InputValidator.validate_file_path(file_path)
        InputValidator.validate_k_value(k)
    """

    # Reserved names (Windows + common reserved)
    RESERVED_NAMES = frozenset(
        {
            # Windows reserved
            "con",
            "prn",
            "aux",
            "nul",
            "com1",
            "com2",
            "com3",
            "com4",
            "com5",
            "com6",
            "com7",
            "com8",
            "com9",
            "lpt1",
            "lpt2",
            "lpt3",
            "lpt4",
            "lpt5",
            "lpt6",
            "lpt7",
            "lpt8",
            "lpt9",
            # Common reserved
            "test",
            "tmp",
            "temp",
        }
    )

    @staticmethod
    def validate_project_name(name: str) -> None:
        """Validate project name meets requirements.

        Requirements:
        - Non-empty
        - Alphanumeric, hyphens, underscores only
        - Max 100 characters
        - Not a reserved name

        Args:
            name: Project name to validate

        Raises:
            typer.BadParameter: If name is invalid

        Example:
            InputValidator.validate_project_name("my-project")  # OK
            InputValidator.validate_project_name("")  # Raises
        """
        # Check non-empty (Commandment #7: Check inputs)
        if not name or not name.strip():
            raise typer.BadParameter("Project name cannot be empty")

        # Check valid characters
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise typer.BadParameter(
                "Project name can only contain:\n"
                "  - Letters (a-z, A-Z)\n"
                "  - Numbers (0-9)\n"
                "  - Hyphens (-)\n"
                "  - Underscores (_)"
            )

        # Check length
        if len(name) > 100:
            raise typer.BadParameter(
                f"Project name too long: {len(name)} characters (max 100)"
            )

        # Check reserved names (Commandment #5: Assertion density)
        if name.lower() in InputValidator.RESERVED_NAMES:
            raise typer.BadParameter(f"'{name}' is a reserved name and cannot be used")

    @staticmethod
    def validate_file_path(
        path: Path, must_exist: bool = True, must_be_file: bool = True
    ) -> Path:
        """Validate file path meets requirements.

        Args:
            path: File path to validate
            must_exist: Whether file must exist
            must_be_file: Whether path must be a file (not directory)

        Returns:
            Resolved absolute path

        Raises:
            typer.BadParameter: If path is invalid

        Example:
            path = InputValidator.validate_file_path(Path("doc.txt"))
            path = InputValidator.validate_file_path(
                Path("new.txt"),
                must_exist=False
            )
        """
        # Resolve to absolute path
        resolved = path.resolve()

        # Check existence if required
        if must_exist and not resolved.exists():
            raise typer.BadParameter(f"File not found: {resolved}")

        # Check is file (not directory)
        if must_be_file and resolved.exists() and not resolved.is_file():
            raise typer.BadParameter(f"Not a file: {resolved}\n(is a directory)")

        return resolved

    @staticmethod
    def validate_directory_path(
        path: Path, must_exist: bool = True, must_be_empty: bool = False
    ) -> Path:
        """Validate directory path meets requirements.

        Args:
            path: Directory path to validate
            must_exist: Whether directory must exist
            must_be_empty: Whether directory must be empty (if exists)

        Returns:
            Resolved absolute path

        Raises:
            typer.BadParameter: If path is invalid

        Example:
            path = InputValidator.validate_directory_path(Path("data/"))
            path = InputValidator.validate_directory_path(
                Path("new_dir/"),
                must_exist=False
            )
        """
        # Resolve to absolute path
        resolved = path.resolve()

        # Check existence if required
        if must_exist and not resolved.exists():
            raise typer.BadParameter(f"Directory not found: {resolved}")

        # Check is directory (not file)
        if resolved.exists() and not resolved.is_dir():
            raise typer.BadParameter(f"Not a directory: {resolved}\n(is a file)")

        # Check emptiness if required
        if must_be_empty and resolved.exists():
            contents = list(resolved.iterdir())
            if contents:
                raise typer.BadParameter(
                    f"Directory not empty: {resolved}\n"
                    f"Contains {len(contents)} items"
                )

        return resolved

    @staticmethod
    def validate_k_value(k: int, min_k: int = 1, max_k: int = 100) -> None:
        """Validate k (number of results) parameter.

        Args:
            k: Number of results to return
            min_k: Minimum allowed value (default: 1)
            max_k: Maximum allowed value (default: 100)

        Raises:
            typer.BadParameter: If k is invalid

        Example:
            InputValidator.validate_k_value(10)  # OK
            InputValidator.validate_k_value(0)   # Raises (too low)
            InputValidator.validate_k_value(200) # Raises (too high)
        """
        # Check minimum (Commandment #5: Assertions)
        if k < min_k:
            raise typer.BadParameter(f"k must be at least {min_k}, got {k}")

        # Check maximum
        if k > max_k:
            raise typer.BadParameter(
                f"k must be at most {max_k}, got {k}\n"
                f"(requesting too many results may impact performance)"
            )

    @staticmethod
    def validate_positive_integer(value: int, name: str = "value") -> None:
        """Validate value is a positive integer.

        Args:
            value: Integer to validate
            name: Parameter name for error messages

        Raises:
            typer.BadParameter: If value is not positive

        Example:
            InputValidator.validate_positive_integer(10, "batch_size")
            InputValidator.validate_positive_integer(-5, "count")  # Raises
        """
        if value <= 0:
            raise typer.BadParameter(f"{name} must be positive, got {value}")

    @staticmethod
    def validate_non_negative_integer(value: int, name: str = "value") -> None:
        """Validate value is a non-negative integer (>= 0).

        Args:
            value: Integer to validate
            name: Parameter name for error messages

        Raises:
            typer.BadParameter: If value is negative

        Example:
            InputValidator.validate_non_negative_integer(0, "offset")  # OK
            InputValidator.validate_non_negative_integer(-1, "count")  # Raises
        """
        if value < 0:
            raise typer.BadParameter(f"{name} must be non-negative, got {value}")

    @staticmethod
    def validate_percentage(value: float, name: str = "percentage") -> None:
        """Validate value is a valid percentage (0.0 to 100.0).

        Args:
            value: Percentage value to validate
            name: Parameter name for error messages

        Raises:
            typer.BadParameter: If value is outside valid range

        Example:
            InputValidator.validate_percentage(50.0, "threshold")  # OK
            InputValidator.validate_percentage(150.0, "rate")      # Raises
        """
        if not (0.0 <= value <= 100.0):
            raise typer.BadParameter(
                f"{name} must be between 0.0 and 100.0, got {value}"
            )

    @staticmethod
    def validate_non_empty_string(value: str, name: str = "value") -> None:
        """Validate string is non-empty (after stripping whitespace).

        Args:
            value: String to validate
            name: Parameter name for error messages

        Raises:
            typer.BadParameter: If string is empty or whitespace-only

        Example:
            InputValidator.validate_non_empty_string("query", "question")
            InputValidator.validate_non_empty_string("  ", "query")  # Raises
        """
        if not value or not value.strip():
            raise typer.BadParameter(f"{name} cannot be empty")
