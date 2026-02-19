"""Base class for monitor commands.

Provides shared functionality for monitoring operations.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from ingestforge.cli.core.command_base import BaseCommand


class MonitorCommand(BaseCommand):
    """Base class for monitor commands."""

    def get_system_status(self, project: Path) -> Dict[str, Any]:
        """Get system status information.

        Args:
            project: Project directory

        Returns:
            System status dictionary
        """
        ingestforge_dir = project / ".ingestforge"

        if not ingestforge_dir.exists():
            return {
                "initialized": False,
                "status": "not_initialized",
            }

        # Check critical components
        config_exists = (ingestforge_dir / "config.json").exists()
        storage_exists = (ingestforge_dir / "storage").exists()

        return {
            "initialized": True,
            "config_exists": config_exists,
            "storage_exists": storage_exists,
            "status": "healthy" if config_exists and storage_exists else "partial",
        }

    def get_storage_metrics(self, project: Path) -> Dict[str, Any]:
        """Get storage metrics.

        Args:
            project: Project directory

        Returns:
            Storage metrics dictionary
        """
        storage_dir = project / ".ingestforge" / "storage"

        if not storage_dir.exists():
            return {
                "exists": False,
                "size": 0,
                "files": 0,
            }

        # Count files and calculate size
        file_count = 0
        total_size = 0

        for item in storage_dir.rglob("*"):
            if item.is_file():
                file_count += 1
                total_size += item.stat().st_size

        return {
            "exists": True,
            "size": total_size,
            "size_formatted": self._format_size(total_size),
            "files": file_count,
        }

    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory metrics.

        Returns:
            Memory metrics dictionary
        """
        import sys

        # Get basic memory info
        return {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        }

    def check_dependencies(self) -> Dict[str, bool]:
        """Check required dependencies.

        Returns:
            Dictionary of dependency availability
        """
        deps = {}

        # Check core dependencies
        try:
            import typer

            deps["typer"] = True
        except ImportError:
            deps["typer"] = False

        try:
            import rich

            deps["rich"] = True
        except ImportError:
            deps["rich"] = False

        return deps

    def get_log_files(self, project: Path) -> List[Path]:
        """Get log files.

        Args:
            project: Project directory

        Returns:
            List of log file paths
        """
        log_dir = project / ".ingestforge" / "logs"

        if not log_dir.exists():
            return []

        return sorted(
            log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True
        )

    def read_log_file(self, log_file: Path, lines: int = 100) -> List[str]:
        """Read log file lines.

        Args:
            log_file: Log file path
            lines: Number of lines to read

        Returns:
            List of log lines
        """
        try:
            content = log_file.read_text(encoding="utf-8")
            all_lines = content.splitlines()

            # Return last N lines
            return all_lines[-lines:] if len(all_lines) > lines else all_lines

        except Exception:
            return []

    def _classify_log_line(self, line: str) -> str:
        """
        Classify a single log line by severity.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            line: Log line to classify

        Returns:
            Classification: "error", "warning", "info", or "other"
        """
        assert line is not None, "Line cannot be None"
        assert isinstance(line, str), "Line must be string"
        line_lower = line.lower()
        if "error" in line_lower:
            return "error"

        if "warning" in line_lower or "warn" in line_lower:
            return "warning"

        if "info" in line_lower:
            return "info"

        return "other"

    def analyze_logs(self, log_lines: List[str]) -> Dict[str, Any]:
        """
        Analyze log lines and count by severity.

        Rule #1: Zero nesting - helper extracts classification logic
        Rule #2: Fixed upper bound for safety
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            log_lines: Log lines to analyze

        Returns:
            Analysis results with counts by severity
        """
        assert log_lines is not None, "Log lines cannot be None"
        assert isinstance(log_lines, list), "Log lines must be list"
        MAX_LINES: int = 1_000_000  # Hard limit
        lines_processed: int = 0

        # Initialize counters
        counts = {
            "error": 0,
            "warning": 0,
            "info": 0,
            "other": 0,
        }
        for line in log_lines:
            lines_processed += 1
            if lines_processed > MAX_LINES:
                logger.warning(f"Safety limit: processed {MAX_LINES} log lines")
                break
            classification = self._classify_log_line(line)
            counts[classification] += 1
        assert all(
            count >= 0 for count in counts.values()
        ), "All counts must be non-negative"

        return {
            "total_lines": len(log_lines),
            "errors": counts["error"],
            "warnings": counts["warning"],
            "info": counts["info"],
        }

    def run_health_checks(self, project: Path) -> List[Dict[str, Any]]:
        """Run health checks.

        Args:
            project: Project directory

        Returns:
            List of health check results
        """
        checks = []

        # Check project initialization
        checks.append(self._check_initialization(project))

        # Check configuration
        checks.append(self._check_configuration(project))

        # Check storage
        checks.append(self._check_storage(project))

        # Check dependencies
        checks.append(self._check_dependencies_health())

        return checks

    def _check_initialization(self, project: Path) -> Dict[str, Any]:
        """Check if project is initialized.

        Args:
            project: Project directory

        Returns:
            Check result
        """
        ingestforge_dir = project / ".ingestforge"
        is_initialized = ingestforge_dir.exists()

        return {
            "name": "Project Initialization",
            "status": "pass" if is_initialized else "fail",
            "message": "Project initialized"
            if is_initialized
            else "Project not initialized",
        }

    def _check_configuration(self, project: Path) -> Dict[str, Any]:
        """Check configuration.

        Args:
            project: Project directory

        Returns:
            Check result
        """
        config_file = project / ".ingestforge" / "config.json"
        exists = config_file.exists()

        return {
            "name": "Configuration",
            "status": "pass" if exists else "warn",
            "message": "Configuration exists"
            if exists
            else "Using default configuration",
        }

    def _check_storage(self, project: Path) -> Dict[str, Any]:
        """Check storage.

        Args:
            project: Project directory

        Returns:
            Check result
        """
        storage_dir = project / ".ingestforge" / "storage"
        exists = storage_dir.exists()

        return {
            "name": "Storage",
            "status": "pass" if exists else "warn",
            "message": "Storage initialized" if exists else "Storage not initialized",
        }

    def _check_dependencies_health(self) -> Dict[str, Any]:
        """Check dependencies health.

        Returns:
            Check result
        """
        deps = self.check_dependencies()
        all_available = all(deps.values())

        return {
            "name": "Dependencies",
            "status": "pass" if all_available else "fail",
            "message": "All dependencies available"
            if all_available
            else "Some dependencies missing",
        }

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0

        return f"{size_bytes:.2f} TB"

    def create_monitor_panel(self, title: str, content: List[str]) -> str:
        """Create monitor panel.

        Args:
            title: Panel title
            content: Panel content lines

        Returns:
            Panel object
        """
        from rich.panel import Panel

        return Panel("\n".join(content), title=title, border_style="cyan")
