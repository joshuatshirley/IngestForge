"""Dependency Integrity Auditor.

BUG002: Dependency Integrity Audit
Epic: EP-26 (Security & Compliance)

Audits dependency declarations against actual imports to ensure
reproducible builds and identify missing/unused dependencies.

JPL Power of Ten Compliance:
- Rule #1: No recursion (iterative scanning)
- Rule #2: Fixed upper bounds (MAX_FILES, MAX_IMPORTS)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_FILES_TO_SCAN = 2000
MAX_IMPORTS_PER_FILE = 200
MAX_TOTAL_IMPORTS = 10000

# Standard library modules (Python 3.10+)
STDLIB_MODULES: Set[str] = {
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "atexit",
    "base64",
    "bdb",
    "binascii",
    "binhex",
    "bisect",
    "builtins",
    "bz2",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "cProfile",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "doctest",
    "email",
    "encodings",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "graphlib",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "idlelib",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "numbers",
    "operator",
    "optparse",
    "os",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "spwd",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "tomllib",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "typing_extensions",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "winsound",
    "wsgiref",
    "xdrlib",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "_thread",
    "__future__",
}


@dataclass
class DependencyIssue:
    """A single dependency issue."""

    issue_type: str  # "missing", "unused", "version_mismatch"
    package_name: str
    details: str
    files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type,
            "package_name": self.package_name,
            "details": self.details,
            "files": self.files[:10],  # Limit file list
        }


@dataclass
class DependencyReport:
    """Complete dependency audit report."""

    report_id: str
    timestamp: str
    declared_packages: Set[str] = field(default_factory=set)
    imported_packages: Set[str] = field(default_factory=set)
    missing: List[DependencyIssue] = field(default_factory=list)
    unused: List[DependencyIssue] = field(default_factory=list)
    files_scanned: int = 0
    audit_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "declared_count": len(self.declared_packages),
            "imported_count": len(self.imported_packages),
            "missing": [m.to_dict() for m in self.missing],
            "unused": [u.to_dict() for u in self.unused],
            "files_scanned": self.files_scanned,
            "audit_duration_ms": self.audit_duration_ms,
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "missing_count": len(self.missing),
            "unused_count": len(self.unused),
            "status": "clean" if not self.missing else "issues_found",
        }

    @property
    def exit_code(self) -> int:
        """Get exit code for CI integration."""
        if self.missing:
            return 2  # Missing dependencies are errors
        if self.unused:
            return 1  # Unused dependencies are warnings
        return 0


class DependencyAuditor:
    """Dependency integrity auditor.

    Rule #9: Complete type hints.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        """Initialize auditor.

        Args:
            project_root: Root directory of the project.

        Rule #5: Assert preconditions.
        """
        self._project_root = project_root or Path.cwd()
        assert (
            self._project_root.exists()
        ), f"Project root does not exist: {self._project_root}"

    def audit(self) -> DependencyReport:
        """Run full dependency audit.

        Returns:
            DependencyReport with audit results.

        Rule #7: Return explicit result.
        """
        start_time = time.perf_counter()

        report = DependencyReport(
            report_id=f"deps-{int(time.time())}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Get declared dependencies
        report.declared_packages = self._parse_requirements()

        # Get imported packages
        imports_by_package, files_scanned = self._scan_imports()
        report.imported_packages = set(imports_by_package.keys())
        report.files_scanned = files_scanned

        # Find missing (imported but not declared)
        report.missing = self._find_missing(
            imports_by_package, report.declared_packages
        )

        # Find unused (declared but not imported)
        report.unused = self._find_unused(
            report.declared_packages, report.imported_packages
        )

        report.audit_duration_ms = (time.perf_counter() - start_time) * 1000
        return report

    def _parse_requirements(self) -> Set[str]:
        """Parse declared dependencies from requirements.txt and pyproject.toml.

        Returns:
            Set of declared package names (normalized).
        """
        declared: Set[str] = set()

        # Parse requirements.txt
        req_file = self._project_root / "requirements.txt"
        if req_file.exists():
            declared.update(self._parse_requirements_txt(req_file))

        # Parse pyproject.toml
        pyproject = self._project_root / "pyproject.toml"
        if pyproject.exists():
            declared.update(self._parse_pyproject_toml(pyproject))

        return declared

    def _parse_requirements_txt(self, filepath: Path) -> Set[str]:
        """Parse requirements.txt file.

        Args:
            filepath: Path to requirements.txt.

        Returns:
            Set of package names.
        """
        packages: Set[str] = set()
        try:
            content = filepath.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # Extract package name (before version specifiers)
                pkg = self._extract_package_name(line)
                if pkg:
                    packages.add(self._normalize_package_name(pkg))
        except Exception as e:
            logger.warning(f"Failed to parse requirements.txt: {e}")
        return packages

    def _parse_pyproject_toml(self, filepath: Path) -> Set[str]:
        """Parse pyproject.toml dependencies.

        Args:
            filepath: Path to pyproject.toml.

        Returns:
            Set of package names.
        """
        packages: Set[str] = set()
        try:
            content = filepath.read_text(encoding="utf-8")
            # Simple parsing - look for dependencies section
            in_deps = False
            for line in content.splitlines():
                if "dependencies" in line and "=" in line:
                    in_deps = True
                    continue
                if in_deps:
                    if line.strip().startswith("]"):
                        in_deps = False
                        continue
                    # Extract package from line like '    "requests>=2.28",'
                    if '"' in line:
                        start = line.find('"') + 1
                        end = line.find('"', start)
                        if end > start:
                            pkg = self._extract_package_name(line[start:end])
                            if pkg:
                                packages.add(self._normalize_package_name(pkg))
        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml: {e}")
        return packages

    def _extract_package_name(self, spec: str) -> Optional[str]:
        """Extract package name from requirement specifier.

        Args:
            spec: Requirement specifier (e.g., "requests>=2.28").

        Returns:
            Package name or None.
        """
        # Remove version specifiers
        for sep in [">=", "<=", "==", "!=", ">", "<", "~=", "[", ";"]:
            if sep in spec:
                spec = spec.split(sep)[0]
        return spec.strip() if spec.strip() else None

    def _normalize_package_name(self, name: str) -> str:
        """Normalize package name for comparison.

        Args:
            name: Package name.

        Returns:
            Normalized name (lowercase, underscores to hyphens).
        """
        return name.lower().replace("_", "-").replace(".", "-")

    def _scan_imports(self) -> Tuple[Dict[str, List[str]], int]:
        """Scan Python files for imports.

        Returns:
            Tuple of (imports dict, files_scanned count).

        Rule #2: Respects MAX_FILES_TO_SCAN.
        Rule #7: Return explicit result.
        """
        imports: Dict[str, List[str]] = {}
        files_scanned = 0

        src_dir = self._project_root / "ingestforge"
        if not src_dir.exists():
            return imports, 0

        # Iterative file collection (JPL Rule #1: No recursion)
        py_files = self._collect_python_files(src_dir)

        for filepath in py_files[:MAX_FILES_TO_SCAN]:
            file_imports = self._extract_imports_from_file(filepath)
            files_scanned += 1

            for imp in file_imports[:MAX_IMPORTS_PER_FILE]:
                pkg = self._get_top_level_package(imp)
                if pkg and not self._is_internal_or_stdlib(pkg):
                    normalized = self._normalize_package_name(pkg)
                    if normalized not in imports:
                        imports[normalized] = []
                    if len(imports[normalized]) < 10:  # Limit examples
                        imports[normalized].append(
                            str(filepath.relative_to(self._project_root))
                        )

        return imports, files_scanned

    def _collect_python_files(self, directory: Path) -> List[Path]:
        """Collect Python files iteratively.

        Args:
            directory: Directory to scan.

        Returns:
            List of Python file paths.

        Rule #1: No recursion - uses iterative approach.
        """
        files: List[Path] = []
        stack = [directory]

        while stack and len(files) < MAX_FILES_TO_SCAN:
            current = stack.pop()
            try:
                for item in current.iterdir():
                    if item.is_file() and item.suffix == ".py":
                        files.append(item)
                    elif item.is_dir() and not item.name.startswith((".", "__")):
                        stack.append(item)
            except PermissionError:
                continue

        return files

    def _extract_imports_from_file(self, filepath: Path) -> List[str]:
        """Extract import statements from a Python file.

        Args:
            filepath: Path to Python file.

        Returns:
            List of imported module names.
        """
        imports: List[str] = []
        try:
            content = filepath.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception:
            pass  # Skip files that can't be parsed

        return imports

    def _get_top_level_package(self, module_name: str) -> str:
        """Get top-level package from module path.

        Args:
            module_name: Full module path (e.g., "requests.auth").

        Returns:
            Top-level package name (e.g., "requests").
        """
        return module_name.split(".")[0]

    def _is_internal_or_stdlib(self, package: str) -> bool:
        """Check if package is internal or stdlib.

        Args:
            package: Package name.

        Returns:
            True if internal or stdlib.
        """
        if package == "ingestforge":
            return True
        if package in STDLIB_MODULES:
            return True
        return False

    def _find_missing(
        self, imports: Dict[str, List[str]], declared: Set[str]
    ) -> List[DependencyIssue]:
        """Find missing dependencies.

        Args:
            imports: Dict of imported packages to files.
            declared: Set of declared packages.

        Returns:
            List of missing dependency issues.
        """
        missing: List[DependencyIssue] = []

        for pkg, files in imports.items():
            if pkg not in declared and not self._is_known_alias(pkg, declared):
                missing.append(
                    DependencyIssue(
                        issue_type="missing",
                        package_name=pkg,
                        details="Imported but not declared in requirements",
                        files=files,
                    )
                )

        return missing

    def _find_unused(
        self, declared: Set[str], imported: Set[str]
    ) -> List[DependencyIssue]:
        """Find unused dependencies.

        Args:
            declared: Set of declared packages.
            imported: Set of imported packages.

        Returns:
            List of unused dependency issues.
        """
        unused: List[DependencyIssue] = []

        for pkg in declared:
            if pkg not in imported and not self._is_known_alias(pkg, imported):
                # Skip common dev/test dependencies
                if not self._is_dev_dependency(pkg):
                    unused.append(
                        DependencyIssue(
                            issue_type="unused",
                            package_name=pkg,
                            details="Declared but not imported (may be optional)",
                            files=[],
                        )
                    )

        return unused

    def _is_known_alias(self, package: str, packages: Set[str]) -> bool:
        """Check if package has a known import alias.

        Args:
            package: Package name to check.
            packages: Set of packages to check against.

        Returns:
            True if package is an alias of something in packages.
        """
        # Common package name mismatches
        aliases = {
            "PIL": "pillow",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "google": "google-api-python-client",
        }
        return aliases.get(package, package) in packages

    def _is_dev_dependency(self, package: str) -> bool:
        """Check if package is likely a dev dependency.

        Args:
            package: Package name.

        Returns:
            True if likely dev dependency.
        """
        dev_packages = {
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-xdist",
            "black",
            "isort",
            "flake8",
            "mypy",
            "pylint",
            "ruff",
            "sphinx",
            "mkdocs",
            "pre-commit",
            "tox",
            "coverage",
        }
        return self._normalize_package_name(package) in dev_packages


def create_dependency_auditor(
    project_root: Optional[Path] = None,
) -> DependencyAuditor:
    """Factory function to create a dependency auditor.

    Args:
        project_root: Root directory of the project.

    Returns:
        Configured DependencyAuditor instance.
    """
    return DependencyAuditor(project_root=project_root)
