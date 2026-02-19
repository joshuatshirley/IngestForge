"""Naming Convention Linter Implementation.

Naming Convention Linter
Epic: EP-26 (Security & Compliance)

Provides naming convention linting to enforce code style standards
across the codebase for CI/CD integration.

JPL Power of Ten Compliance:
- Rule #1: No recursion (uses iterative directory traversal)
- Rule #2: Fixed upper bounds (MAX_VIOLATIONS, MAX_FILE_SIZE, MAX_DEPTH)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import ast
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_VIOLATIONS = 500  # Maximum violations per scan
MAX_FILE_SIZE = 5_000_000  # Maximum file size to lint (5MB)
MAX_DEPTH = 20  # Maximum directory depth
MAX_FILES_PER_SCAN = 5_000  # Maximum files per scan


class ViolationSeverity(Enum):
    """Naming violation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ViolationCategory(Enum):
    """Categories of naming violations."""

    FUNCTION = "function"
    CLASS = "class"
    CONSTANT = "constant"
    VARIABLE = "variable"
    MODULE = "module"
    ARGUMENT = "argument"


# Naming convention patterns
SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
PASCAL_CASE_PATTERN = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
UPPER_CASE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
PRIVATE_PREFIX_PATTERN = re.compile(r"^_+")


@dataclass(frozen=True)
class NamingViolation:
    """Immutable naming violation.

    Rule #9: Complete type hints.
    """

    violation_id: str
    category: ViolationCategory
    severity: ViolationSeverity
    rule_id: str
    name: str
    expected_pattern: str
    file_path: str
    line_number: int
    message: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "violation_id": self.violation_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "rule_id": self.rule_id,
            "name": self.name,
            "expected_pattern": self.expected_pattern,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "message": self.message,
            "recommendation": self.recommendation,
        }


@dataclass
class LintReport:
    """Lint scan report.

    Rule #9: Complete type hints.
    """

    report_id: str
    scan_path: str
    violations: List[NamingViolation] = field(default_factory=list)
    files_scanned: int = 0
    scan_duration_ms: float = 0.0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    truncated: bool = False

    @property
    def error_count(self) -> int:
        """Count of error severity violations."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning severity violations."""
        return sum(
            1 for v in self.violations if v.severity == ViolationSeverity.WARNING
        )

    @property
    def info_count(self) -> int:
        """Count of info severity violations."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.INFO)

    @property
    def exit_code(self) -> int:
        """Get CI exit code based on violations.

        Returns:
            0 = clean, 1 = warnings only, 2 = errors
        """
        if self.error_count > 0:
            return 2
        if self.warning_count > 0:
            return 1
        return 0

    def add_violation(self, violation: NamingViolation) -> bool:
        """Add violation to report.

        Args:
            violation: Violation to add.

        Returns:
            True if added, False if at capacity.

        Rule #2: Enforce MAX_VIOLATIONS.
        """
        if len(self.violations) >= MAX_VIOLATIONS:
            self.truncated = True
            return False
        self.violations.append(violation)
        return True

    def complete(self, duration_ms: float) -> None:
        """Mark report as completed.

        Args:
            duration_ms: Scan duration in milliseconds.
        """
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.scan_duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "report_id": self.report_id,
            "scan_path": self.scan_path,
            "summary": {
                "total_violations": len(self.violations),
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
                "exit_code": self.exit_code,
            },
            "files_scanned": self.files_scanned,
            "scan_duration_ms": self.scan_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "truncated": self.truncated,
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass
class NamingRule:
    """Naming convention rule.

    Rule #9: Complete type hints.
    """

    rule_id: str
    title: str
    category: ViolationCategory
    severity: ViolationSeverity
    pattern: re.Pattern[str]
    message_template: str
    recommendation: str
    enabled: bool = True


# Module-level rule definitions (JPL Rule #4: Keep functions < 60 lines)
_FUNCTION_RULES = [
    NamingRule(
        rule_id="NAME001",
        title="Function naming",
        category=ViolationCategory.FUNCTION,
        severity=ViolationSeverity.ERROR,
        pattern=SNAKE_CASE_PATTERN,
        message_template="Function '{name}' should be snake_case",
        recommendation="Rename to snake_case (e.g., my_function)",
    ),
]

_CLASS_RULES = [
    NamingRule(
        rule_id="NAME002",
        title="Class naming",
        category=ViolationCategory.CLASS,
        severity=ViolationSeverity.ERROR,
        pattern=PASCAL_CASE_PATTERN,
        message_template="Class '{name}' should be PascalCase",
        recommendation="Rename to PascalCase (e.g., MyClass)",
    ),
]

_CONSTANT_RULES = [
    NamingRule(
        rule_id="NAME003",
        title="Constant naming",
        category=ViolationCategory.CONSTANT,
        severity=ViolationSeverity.WARNING,
        pattern=UPPER_CASE_PATTERN,
        message_template="Constant '{name}' should be UPPER_CASE",
        recommendation="Rename to UPPER_CASE (e.g., MAX_VALUE)",
    ),
]

_VARIABLE_RULES = [
    NamingRule(
        rule_id="NAME004",
        title="Variable naming",
        category=ViolationCategory.VARIABLE,
        severity=ViolationSeverity.WARNING,
        pattern=SNAKE_CASE_PATTERN,
        message_template="Variable '{name}' should be snake_case",
        recommendation="Rename to snake_case (e.g., my_variable)",
    ),
]

_ARGUMENT_RULES = [
    NamingRule(
        rule_id="NAME005",
        title="Argument naming",
        category=ViolationCategory.ARGUMENT,
        severity=ViolationSeverity.INFO,
        pattern=SNAKE_CASE_PATTERN,
        message_template="Argument '{name}' should be snake_case",
        recommendation="Rename to snake_case (e.g., my_arg)",
    ),
]


def _strip_private_prefix(name: str) -> str:
    """Strip leading underscores for pattern matching.

    Args:
        name: Name to process.

    Returns:
        Name without leading underscores.
    """
    return PRIVATE_PREFIX_PATTERN.sub("", name)


def _is_dunder(name: str) -> bool:
    """Check if name is a dunder (double underscore) method.

    Args:
        name: Name to check.

    Returns:
        True if dunder method.
    """
    return name.startswith("__") and name.endswith("__")


class NamingLinter:
    """Naming convention linter for CI/CD integration.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        rules: Optional[List[NamingRule]] = None,
        max_file_size: int = MAX_FILE_SIZE,
    ) -> None:
        """Initialize linter.

        Args:
            rules: Custom naming rules. Uses defaults if None.
            max_file_size: Maximum file size to lint.

        Rule #5: Assert preconditions.
        """
        assert max_file_size > 0, "max_file_size must be positive"

        self._rules = rules or self._get_default_rules()
        self._max_file_size = min(max_file_size, MAX_FILE_SIZE)

    def _get_default_rules(self) -> List[NamingRule]:
        """Get default naming rules.

        Returns:
            List of default naming rules.

        Rule #4: Function < 60 lines - uses module-level rule definitions.
        """
        return (
            _FUNCTION_RULES
            + _CLASS_RULES
            + _CONSTANT_RULES
            + _VARIABLE_RULES
            + _ARGUMENT_RULES
        )

    def _check_file_eligible(self, file_path: Path) -> Optional[str]:
        """Check if file is eligible for linting and return content.

        Args:
            file_path: Path to check.

        Returns:
            File content if eligible, None otherwise.

        Rule #4: Helper to keep lint_file < 60 lines.
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        if not file_path.is_file():
            return None

        if file_path.suffix.lower() != ".py":
            return None

        try:
            if file_path.stat().st_size > self._max_file_size:
                logger.info(f"Skipping large file: {file_path}")
                return None
        except OSError:
            return None

        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Cannot read {file_path}: {e}")
            return None

    def _check_name(
        self,
        name: str,
        rule: NamingRule,
        file_path: str,
        line_number: int,
    ) -> Optional[NamingViolation]:
        """Check a name against a rule.

        Args:
            name: Name to check.
            rule: Rule to apply.
            file_path: Source file path.
            line_number: Line number.

        Returns:
            NamingViolation if violated, None otherwise.

        Rule #4: Helper to keep lint_file < 60 lines.
        """
        if not rule.enabled:
            return None

        # Skip dunder methods
        if _is_dunder(name):
            return None

        # Strip private prefix for matching
        check_name = _strip_private_prefix(name)
        if not check_name:
            return None

        if rule.pattern.match(check_name):
            return None

        return NamingViolation(
            violation_id=str(uuid.uuid4()),
            category=rule.category,
            severity=rule.severity,
            rule_id=rule.rule_id,
            name=name,
            expected_pattern=rule.pattern.pattern,
            file_path=file_path,
            line_number=line_number,
            message=rule.message_template.format(name=name),
            recommendation=rule.recommendation,
        )

    def _get_rule_for_category(
        self, category: ViolationCategory
    ) -> Optional[NamingRule]:
        """Get first enabled rule for category.

        Args:
            category: Category to find rule for.

        Returns:
            NamingRule or None.
        """
        for rule in self._rules:
            if rule.category == category and rule.enabled:
                return rule
        return None

    def _lint_ast_node(
        self,
        node: ast.AST,
        file_path: str,
        violations: List[NamingViolation],
    ) -> None:
        """Lint a single AST node.

        Args:
            node: AST node to lint.
            file_path: Source file path.
            violations: List to append violations to.

        Rule #4: Helper to keep lint_file < 60 lines.
        """
        if isinstance(node, ast.FunctionDef):
            rule = self._get_rule_for_category(ViolationCategory.FUNCTION)
            if rule:
                v = self._check_name(node.name, rule, file_path, node.lineno)
                if v:
                    violations.append(v)

        elif isinstance(node, ast.ClassDef):
            rule = self._get_rule_for_category(ViolationCategory.CLASS)
            if rule:
                v = self._check_name(node.name, rule, file_path, node.lineno)
                if v:
                    violations.append(v)

        elif isinstance(node, ast.Assign):
            self._lint_assignment(node, file_path, violations)

    def _lint_assignment(
        self,
        node: ast.Assign,
        file_path: str,
        violations: List[NamingViolation],
    ) -> None:
        """Lint assignment for constants/variables.

        Args:
            node: Assignment node.
            file_path: Source file path.
            violations: List to append violations to.

        Rule #4: Helper function.
        """
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            name = target.id
            # Check if it looks like a constant (all caps or has underscore)
            if name.isupper() or (name.upper() == name and "_" in name):
                rule = self._get_rule_for_category(ViolationCategory.CONSTANT)
            else:
                rule = self._get_rule_for_category(ViolationCategory.VARIABLE)

            if rule:
                v = self._check_name(name, rule, file_path, node.lineno)
                if v:
                    violations.append(v)

    def lint_file(self, file_path: Path) -> List[NamingViolation]:
        """Lint a single Python file.

        Args:
            file_path: Path to file to lint.

        Returns:
            List of naming violations.

        Rule #4: Function < 60 lines (uses helper methods).
        Rule #7: Return explicit result.
        """
        violations: List[NamingViolation] = []

        content = self._check_file_eligible(file_path)
        if content is None:
            return violations

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return violations

        for node in ast.walk(tree):
            self._lint_ast_node(node, str(file_path), violations)
            if len(violations) >= MAX_VIOLATIONS:
                break

        return violations

    def _collect_files_iteratively(
        self,
        directory: Path,
        exclude_patterns: List[str],
        recursive: bool,
    ) -> List[Path]:
        """Collect files to lint using iterative traversal.

        Args:
            directory: Root directory.
            exclude_patterns: Patterns to exclude.
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of file paths to lint.

        Rule #1: No recursion (uses iterative approach).
        Rule #4: Helper to keep lint_directory < 60 lines.
        """
        files_to_lint: List[Path] = []
        dirs_to_process = [directory]
        depth = 0

        while dirs_to_process and depth < MAX_DEPTH:
            current_dirs = dirs_to_process[:]
            dirs_to_process = []
            depth += 1

            for current_dir in current_dirs:
                try:
                    for item in current_dir.iterdir():
                        if self._should_exclude(item, exclude_patterns):
                            continue

                        if item.is_file() and item.suffix == ".py":
                            files_to_lint.append(item)
                            if len(files_to_lint) >= MAX_FILES_PER_SCAN:
                                return files_to_lint
                        elif item.is_dir() and recursive:
                            dirs_to_process.append(item)

                except PermissionError:
                    continue

        return files_to_lint

    def _should_exclude(self, path: Path, patterns: List[str]) -> bool:
        """Check if path should be excluded.

        Args:
            path: Path to check.
            patterns: Exclusion patterns.

        Returns:
            True if path should be excluded.
        """
        path_str = str(path)
        for pattern in patterns:
            if pattern.replace("**", "").replace("*", "") in path_str:
                return True
        return False

    def lint_directory(
        self,
        directory: Path,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
    ) -> LintReport:
        """Lint a directory for naming violations.

        Args:
            directory: Directory to lint.
            recursive: Whether to lint subdirectories.
            exclude_patterns: Glob patterns to exclude.

        Returns:
            LintReport with all violations.

        Rule #4: Function < 60 lines (uses helper methods).
        Rule #5: Assert preconditions.
        Rule #7: Return explicit result.
        """
        import time

        start = time.perf_counter()

        assert directory.exists(), f"Directory does not exist: {directory}"
        assert directory.is_dir(), f"Not a directory: {directory}"

        report = LintReport(
            report_id=str(uuid.uuid4()),
            scan_path=str(directory),
        )

        default_excludes = [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.venv/**",
            "**/env/**",
            "**/.tox/**",
        ]
        all_excludes = (exclude_patterns or []) + default_excludes

        files_to_lint = self._collect_files_iteratively(
            directory, all_excludes, recursive
        )

        for file_path in files_to_lint:
            violations = self.lint_file(file_path)
            report.files_scanned += 1

            for violation in violations:
                if not report.add_violation(violation):
                    break

        elapsed_ms = (time.perf_counter() - start) * 1000
        report.complete(elapsed_ms)

        return report

    def get_rules(self) -> List[NamingRule]:
        """Get current rules.

        Returns:
            List of naming rules.
        """
        return list(self._rules)

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule was found and enabled.
        """
        for rule in self._rules:
            if rule.rule_id == rule_id:
                rule.enabled = True
                return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if rule was found and disabled.
        """
        for rule in self._rules:
            if rule.rule_id == rule_id:
                rule.enabled = False
                return True
        return False


def create_linter(
    rules: Optional[List[NamingRule]] = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> NamingLinter:
    """Factory function to create a naming linter.

    Args:
        rules: Optional custom rules.
        max_file_size: Maximum file size to lint.

    Returns:
        Configured NamingLinter instance.
    """
    return NamingLinter(rules=rules, max_file_size=max_file_size)
