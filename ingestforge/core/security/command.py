"""
Safe command execution for preventing shell injection.

Provides CommandValidator class and safe_run function to prevent shell injection
vulnerabilities by validating commands before execution without shell=True.

Threat Model
------------
IngestForge allows users to define scheduled workflows and pipelines that
execute commands. Without validation, attackers could inject shell metacharacters:

    Attack: Schedule command "ingestforge ingest; rm -rf /"
    Defense: CommandValidator blocks shell metacharacters

Components
----------
**CommandValidator**
    Validates command strings before execution:
    - Parses commands using shlex (no shell interpretation)
    - Checks executables against whitelist
    - Blocks shell metacharacters in arguments

**safe_run**
    Convenience function for safe subprocess execution:
    - Validates command with CommandValidator
    - Executes with shell=False
    - Returns subprocess.CompletedProcess

Usage Pattern
-------------
Instead of dangerous shell=True subprocess calls:

    # DANGEROUS - allows shell injection
    subprocess.run(user_command, shell=True)

Use safe_run:

    # SAFE - validates and executes without shell
    from ingestforge.core.security.command import safe_run
    result = safe_run(user_command, timeout=3600)
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional


class CommandInjectionError(Exception):
    """Raised when potential command injection detected."""

    pass


# Whitelist of allowed executables for workflows
# Can be extended by passing custom allowed set to CommandValidator
ALLOWED_EXECUTABLES: FrozenSet[str] = frozenset(
    [
        # IngestForge commands
        "ingestforge",
        # Python ecosystem
        "python",
        "python3",
        "pip",
        "pip3",
        # Common safe utilities
        "echo",
        "date",
        "sleep",
    ]
)


@dataclass
class SafeCommand:
    """Validated command ready for execution."""

    executable: str
    args: List[str]

    def as_list(self) -> List[str]:
        """Return command as list for subprocess."""
        return [self.executable] + self.args


class CommandValidator:
    """
    Validate commands for safe execution without shell=True.

    This class:
    1. Parses command strings using shlex (safe tokenization)
    2. Checks executable against whitelist
    3. Blocks dangerous shell metacharacters in arguments

    Example:
        >>> validator = CommandValidator()
        >>> cmd = validator.validate("ingestforge ingest ./docs")
        >>> # Returns SafeCommand

        >>> validator.validate("rm -rf /")
        >>> # Raises CommandInjectionError - rm not in whitelist

        >>> validator.validate("ingestforge ingest; whoami")
        >>> # Raises CommandInjectionError - semicolon blocked
    """

    # Shell metacharacters that could enable command injection
    SHELL_METACHARACTERS: FrozenSet[str] = frozenset(
        [
            ";",  # Command separator
            "|",  # Pipe
            "&",  # Background/AND
            "$",  # Variable expansion
            "`",  # Command substitution
            "(",  # Subshell
            ")",  # Subshell
            "{",  # Brace expansion
            "}",  # Brace expansion
            "<",  # Input redirection
            ">",  # Output redirection
            "\n",  # Newline (command separator)
            "\r",  # Carriage return
            "\x00",  # Null byte
        ]
    )

    def __init__(
        self,
        allowed: Optional[FrozenSet[str]] = None,
        block_metacharacters: bool = True,
    ) -> None:
        """
        Initialize command validator.

        Args:
            allowed: Set of allowed executable names. Defaults to ALLOWED_EXECUTABLES.
            block_metacharacters: If True, blocks shell metacharacters in args.
        """
        self._allowed = allowed or ALLOWED_EXECUTABLES
        self._block_metacharacters = block_metacharacters

    def validate(self, command: str) -> SafeCommand:
        """
        Validate a command string and return SafeCommand.

        Args:
            command: Command string to validate.

        Returns:
            SafeCommand object ready for execution.

        Raises:
            CommandInjectionError: If command is unsafe.
        """
        # Parse using shlex (safe tokenization without shell interpretation)
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            raise CommandInjectionError(f"Invalid command syntax: {e}") from e

        if not tokens:
            raise CommandInjectionError("Empty command")

        executable, args = tokens[0], tokens[1:]

        # Check executable against whitelist
        # Handle both bare name and full path
        basename = Path(executable).name
        # Remove .exe extension for Windows compatibility
        if basename.lower().endswith(".exe"):
            basename = basename[:-4]

        if basename not in self._allowed:
            raise CommandInjectionError(
                f"Executable not allowed: {executable!r}. "
                f"Allowed: {sorted(self._allowed)}"
            )

        # Check args for shell metacharacters
        if self._block_metacharacters:
            for arg in args:
                self._check_metacharacters(arg)

        return SafeCommand(executable=executable, args=args)

    def _check_metacharacters(self, arg: str) -> None:
        """
        Check argument for dangerous shell metacharacters.

        Args:
            arg: Argument string to check.

        Raises:
            CommandInjectionError: If metacharacter found.
        """
        for char in self.SHELL_METACHARACTERS:
            if char in arg:
                # Provide readable name for control characters
                char_repr = repr(char) if char.isprintable() else f"0x{ord(char):02x}"
                raise CommandInjectionError(
                    f"Shell metacharacter {char_repr} in argument: {arg!r}"
                )


def safe_run(
    command: str,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    allowed: Optional[FrozenSet[str]] = None,
) -> "subprocess.CompletedProcess[str]":
    """
    Execute a command safely without shell=True.

    This function:
    1. Validates command with CommandValidator
    2. Executes with shell=False
    3. Captures output and returns CompletedProcess

    Args:
        command: Command string to execute.
        timeout: Optional timeout in seconds.
        cwd: Optional working directory.
        allowed: Optional custom whitelist of executables.

    Returns:
        subprocess.CompletedProcess with stdout, stderr, returncode.

    Raises:
        CommandInjectionError: If command is unsafe.
        subprocess.TimeoutExpired: If command times out.

    Example:
        >>> result = safe_run("ingestforge ingest ./docs", timeout=3600)
        >>> if result.returncode == 0:
        ...     print("Success:", result.stdout)
        ... else:
        ...     print("Failed:", result.stderr)
    """
    validator = CommandValidator(allowed=allowed)
    safe_cmd = validator.validate(command)

    return subprocess.run(
        safe_cmd.as_list(),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        shell=False,
    )
