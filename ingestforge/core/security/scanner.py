"""Security Scanner Implementation.

Security Shield CI Pipeline
Epic: EP-26 (Security & Compliance)

Provides security scanning to detect vulnerabilities, unsafe patterns,
and compliance issues in source code for CI/CD integration.

JPL Power of Ten Compliance:
- Rule #1: No recursion (uses iterative directory traversal)
- Rule #2: Fixed upper bounds (MAX_FINDINGS, MAX_FILE_SIZE, MAX_DEPTH)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_FINDINGS = 1000  # Maximum findings per scan
MAX_FILE_SIZE = 10_000_000  # Maximum file size to scan (10MB)
MAX_DEPTH = 20  # Maximum directory depth
MAX_LINE_LENGTH = 10_000  # Maximum line length to scan
MAX_FILES_PER_SCAN = 10_000  # Maximum files per scan


class Severity(Enum):
    """Security finding severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(Enum):
    """Categories of security findings."""

    SECRETS = "secrets"  # Hardcoded secrets, API keys
    INJECTION = "injection"  # SQL, command, XSS injection
    CRYPTO = "crypto"  # Weak cryptography
    AUTH = "auth"  # Authentication issues
    CONFIG = "config"  # Security misconfigurations
    DATA = "data"  # Data exposure
    DEPENDENCIES = "dependencies"  # Vulnerable dependencies


@dataclass(frozen=True)
class SecurityFinding:
    """Immutable security finding.

    Rule #9: Complete type hints.
    """

    finding_id: str
    category: FindingCategory
    severity: Severity
    title: str
    description: str
    file_path: str
    line_number: int
    line_content: str
    recommendation: str
    rule_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "finding_id": self.finding_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content[:100],  # Truncate for display
            "recommendation": self.recommendation,
            "rule_id": self.rule_id,
            "metadata": self.metadata,
        }


@dataclass
class SecurityReport:
    """Security scan report.

    Rule #9: Complete type hints.
    """

    report_id: str
    scan_path: str
    findings: List[SecurityFinding] = field(default_factory=list)
    files_scanned: int = 0
    scan_duration_ms: float = 0.0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        """Count of critical severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        """Count of medium severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        """Count of low severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.LOW)

    @property
    def info_count(self) -> int:
        """Count of info severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.INFO)

    @property
    def has_critical(self) -> bool:
        """Check if any critical findings exist."""
        return self.critical_count > 0

    @property
    def has_high(self) -> bool:
        """Check if any high severity findings exist."""
        return self.high_count > 0

    @property
    def exit_code(self) -> int:
        """Get CI exit code based on findings.

        Returns:
            0 = clean/info only, 1 = warnings (low/medium), 2 = errors (high/critical)
        """
        if self.has_critical or self.has_high:
            return 2
        if self.medium_count > 0 or self.low_count > 0:
            return 1
        return 0

    def add_finding(self, finding: SecurityFinding) -> bool:
        """Add finding to report.

        Args:
            finding: Finding to add.

        Returns:
            True if added, False if at capacity.

        Rule #2: Enforce MAX_FINDINGS.
        """
        if len(self.findings) >= MAX_FINDINGS:
            self.truncated = True
            return False
        self.findings.append(finding)
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
                "total_findings": len(self.findings),
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "info": self.info_count,
                "exit_code": self.exit_code,
            },
            "files_scanned": self.files_scanned,
            "scan_duration_ms": self.scan_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "truncated": self.truncated,
            "findings": [f.to_dict() for f in self.findings],
            "metadata": self.metadata,
        }


@dataclass
class SecurityRule:
    """Security scanning rule.

    Rule #9: Complete type hints.
    """

    rule_id: str
    title: str
    description: str
    category: FindingCategory
    severity: Severity
    pattern: Pattern[str]
    recommendation: str
    file_extensions: List[str] = field(default_factory=list)
    enabled: bool = True


# Module-level rule definitions (JPL Rule #4: Keep functions < 60 lines)
# Rules are defined at module level to keep _get_default_rules() concise

_SECRETS_RULES = [
    SecurityRule(
        rule_id="SEC001",
        title="Hardcoded API Key",
        description="Potential API key or secret found in source code",
        category=FindingCategory.SECRETS,
        severity=Severity.HIGH,
        pattern=re.compile(
            r'(?i)(api[_-]?key|apikey|secret[_-]?key|auth[_-]?token)\s*[=:]\s*["\'][\w\-]{16,}["\']',
            re.IGNORECASE,
        ),
        recommendation="Store secrets in environment variables or a secrets manager",
    ),
    SecurityRule(
        rule_id="SEC002",
        title="Hardcoded Password",
        description="Potential hardcoded password found",
        category=FindingCategory.SECRETS,
        severity=Severity.CRITICAL,
        pattern=re.compile(
            r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{4,}["\']', re.IGNORECASE
        ),
        recommendation="Never hardcode passwords. Use environment variables or secrets managers",
    ),
    SecurityRule(
        rule_id="SEC003",
        title="AWS Access Key",
        description="AWS access key ID detected",
        category=FindingCategory.SECRETS,
        severity=Severity.CRITICAL,
        pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
        recommendation="Rotate this key immediately and use IAM roles instead",
    ),
    SecurityRule(
        rule_id="SEC004",
        title="Private Key",
        description="Private key content detected",
        category=FindingCategory.SECRETS,
        severity=Severity.CRITICAL,
        pattern=re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
        recommendation="Never commit private keys. Use a secrets manager",
    ),
]

_INJECTION_RULES = [
    SecurityRule(
        rule_id="SEC010",
        title="SQL Injection Risk",
        description="String concatenation in SQL query detected",
        category=FindingCategory.INJECTION,
        severity=Severity.HIGH,
        pattern=re.compile(
            r'(?i)(execute|cursor\.execute|query)\s*\(\s*["\'].*\%s.*["\'].*\%',
            re.IGNORECASE,
        ),
        recommendation="Use parameterized queries or prepared statements",
        file_extensions=[".py", ".php", ".rb"],
    ),
    SecurityRule(
        rule_id="SEC011",
        title="SQL String Formatting",
        description="F-string or format() in SQL query",
        category=FindingCategory.INJECTION,
        severity=Severity.HIGH,
        pattern=re.compile(r"(?i)(select|insert|update|delete).*\{.*\}"),
        recommendation="Use parameterized queries instead of string formatting",
        file_extensions=[".py"],
    ),
    SecurityRule(
        rule_id="SEC020",
        title="innerHTML Assignment",
        description="Direct innerHTML assignment may lead to XSS",
        category=FindingCategory.INJECTION,
        severity=Severity.MEDIUM,
        pattern=re.compile(r"\.innerHTML\s*="),
        recommendation="Use textContent or sanitize HTML before assignment",
        file_extensions=[".js", ".ts", ".jsx", ".tsx"],
    ),
    SecurityRule(
        rule_id="SEC021",
        title="eval() Usage",
        description="eval() can execute arbitrary code",
        category=FindingCategory.INJECTION,
        severity=Severity.HIGH,
        pattern=re.compile(r"\beval\s*\("),
        recommendation="Avoid eval(). Use safer alternatives like JSON.parse()",
        file_extensions=[".js", ".ts", ".py"],
    ),
]

_CONFIG_RULES = [
    SecurityRule(
        rule_id="SEC030",
        title="Weak Hash Algorithm",
        description="MD5 or SHA1 used for hashing",
        category=FindingCategory.CRYPTO,
        severity=Severity.MEDIUM,
        pattern=re.compile(r"(?i)\b(md5|sha1)\s*\("),
        recommendation="Use SHA-256 or stronger hash algorithms",
    ),
    SecurityRule(
        rule_id="SEC040",
        title="Debug Mode Enabled",
        description="Debug mode may be enabled in production",
        category=FindingCategory.CONFIG,
        severity=Severity.MEDIUM,
        pattern=re.compile(r"(?i)(debug\s*[=:]\s*true|DEBUG\s*=\s*True)"),
        recommendation="Ensure debug mode is disabled in production",
        file_extensions=[".py", ".js", ".json", ".yaml", ".yml", ".env"],
    ),
    SecurityRule(
        rule_id="SEC041",
        title="Insecure SSL/TLS Setting",
        description="SSL verification may be disabled",
        category=FindingCategory.CONFIG,
        severity=Severity.HIGH,
        pattern=re.compile(
            r"(?i)(verify\s*[=:]\s*false|ssl[_-]?verify\s*[=:]\s*false)"
        ),
        recommendation="Always verify SSL certificates in production",
    ),
]


class SecurityScanner:
    """Security scanner for CI/CD integration.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        rules: Optional[List[SecurityRule]] = None,
        max_file_size: int = MAX_FILE_SIZE,
    ) -> None:
        """Initialize scanner.

        Args:
            rules: Custom security rules. Uses defaults if None.
            max_file_size: Maximum file size to scan.

        Rule #5: Assert preconditions.
        """
        assert max_file_size > 0, "max_file_size must be positive"

        self._rules = rules or self._get_default_rules()
        self._max_file_size = min(max_file_size, MAX_FILE_SIZE)
        self._scanned_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rb",
            ".php",
            ".cs",
            ".cpp",
            ".c",
            ".h",
            ".rs",
            ".yaml",
            ".yml",
            ".json",
            ".xml",
            ".env",
            ".ini",
            ".conf",
            ".sh",
            ".bash",
            ".zsh",
            ".ps1",
            ".sql",
        }

    def _get_default_rules(self) -> List[SecurityRule]:
        """Get default security rules.

        Returns:
            List of default security rules.

        Rule #4: Function < 60 lines - uses module-level rule definitions.
        """
        return _SECRETS_RULES + _INJECTION_RULES + _CONFIG_RULES

    def _check_file_eligible(self, file_path: Path) -> Optional[str]:
        """Check if file is eligible for scanning and return content.

        Args:
            file_path: Path to check.

        Returns:
            File content if eligible, None otherwise.

        Rule #4: Helper to keep scan_file() < 60 lines.
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        if not file_path.is_file():
            return None

        if file_path.suffix.lower() not in self._scanned_extensions:
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

    def _scan_line_for_findings(
        self,
        line: str,
        line_num: int,
        file_path: Path,
        findings: List[SecurityFinding],
    ) -> bool:
        """Scan a single line for rule matches.

        Args:
            line: Line content to scan.
            line_num: Line number in file.
            file_path: Path to file being scanned.
            findings: List to append findings to.

        Returns:
            True if at capacity, False otherwise.

        Rule #4: Helper to keep scan_file() < 60 lines.
        """
        for rule in self._rules:
            if not rule.enabled:
                continue

            if (
                rule.file_extensions
                and file_path.suffix.lower() not in rule.file_extensions
            ):
                continue

            if rule.pattern.search(line):
                finding = SecurityFinding(
                    finding_id=str(uuid.uuid4()),
                    category=rule.category,
                    severity=rule.severity,
                    title=rule.title,
                    description=rule.description,
                    file_path=str(file_path),
                    line_number=line_num,
                    line_content=line.strip(),
                    recommendation=rule.recommendation,
                    rule_id=rule.rule_id,
                )
                findings.append(finding)

                if len(findings) >= MAX_FINDINGS:
                    return True
        return False

    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for security issues.

        Args:
            file_path: Path to file to scan.

        Returns:
            List of security findings.

        Rule #4: Function < 60 lines (uses helper methods).
        Rule #7: Return explicit result.
        """
        findings: List[SecurityFinding] = []

        content = self._check_file_eligible(file_path)
        if content is None:
            return findings

        lines = content.split("\n")
        for line_num, line in enumerate(lines[:10000], start=1):
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH]

            if self._scan_line_for_findings(line, line_num, file_path, findings):
                break

        return findings

    def _collect_files_iteratively(
        self,
        directory: Path,
        exclude_patterns: List[str],
        recursive: bool,
    ) -> List[Path]:
        """Collect files to scan using iterative traversal.

        Args:
            directory: Root directory.
            exclude_patterns: Patterns to exclude.
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of file paths to scan.

        Rule #1: No recursion (uses iterative approach).
        Rule #4: Helper to keep scan_directory() < 60 lines.
        """
        files_to_scan: List[Path] = []
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

                        if item.is_file():
                            files_to_scan.append(item)
                            if len(files_to_scan) >= MAX_FILES_PER_SCAN:
                                return files_to_scan
                        elif item.is_dir() and recursive:
                            dirs_to_process.append(item)

                except PermissionError:
                    continue

        return files_to_scan

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
    ) -> SecurityReport:
        """Scan a directory for security issues.

        Args:
            directory: Directory to scan.
            recursive: Whether to scan subdirectories.
            exclude_patterns: Glob patterns to exclude.

        Returns:
            SecurityReport with all findings.

        Rule #4: Function < 60 lines (uses helper methods).
        Rule #5: Assert preconditions.
        Rule #7: Return explicit result.
        """
        import time

        start = time.perf_counter()

        assert directory.exists(), f"Directory does not exist: {directory}"
        assert directory.is_dir(), f"Not a directory: {directory}"

        report = SecurityReport(
            report_id=str(uuid.uuid4()),
            scan_path=str(directory),
        )

        # Build exclusion patterns
        default_excludes = [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.venv/**",
            "**/env/**",
        ]
        all_excludes = (exclude_patterns or []) + default_excludes

        # Collect and scan files
        files_to_scan = self._collect_files_iteratively(
            directory, all_excludes, recursive
        )

        for file_path in files_to_scan:
            findings = self.scan_file(file_path)
            report.files_scanned += 1

            for finding in findings:
                if not report.add_finding(finding):
                    break

        elapsed_ms = (time.perf_counter() - start) * 1000
        report.complete(elapsed_ms)

        return report

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

    def get_rules(self) -> List[SecurityRule]:
        """Get current rules.

        Returns:
            List of security rules.
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


def create_scanner(
    rules: Optional[List[SecurityRule]] = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> SecurityScanner:
    """Factory function to create a security scanner.

    Args:
        rules: Optional custom rules.
        max_file_size: Maximum file size to scan.

    Returns:
        Configured SecurityScanner instance.
    """
    return SecurityScanner(rules=rules, max_file_size=max_file_size)
