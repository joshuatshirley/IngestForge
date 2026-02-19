"""Base class for maintenance commands.

Provides shared functionality for maintenance operations.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import shutil

from ingestforge.cli.core.command_base import BaseCommand
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class MaintenanceCommand(BaseCommand):
    """Base class for maintenance commands."""

    def get_temp_dirs(self, project: Path) -> List[Path]:
        """Get temporary directories in project.

        Args:
            project: Project directory

        Returns:
            List of temporary directory paths
        """
        temp_paths = [
            project / ".ingestforge" / "temp",
            project / ".ingestforge" / "cache",
            project / "__pycache__",
        ]

        return [p for p in temp_paths if p.exists()]

    def get_log_files(self, project: Path) -> List[Path]:
        """Get log files in project.

        Args:
            project: Project directory

        Returns:
            List of log file paths
        """
        log_dir = project / ".ingestforge" / "logs"

        if not log_dir.exists():
            return []

        return list(log_dir.glob("*.log"))

    def calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory.

        Args:
            directory: Directory to measure

        Returns:
            Size in bytes
        """
        total = 0

        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except Exception as e:
            logger.debug(f"Failed to calculate directory size for {directory}: {e}")

        return total

    def format_size(self, size_bytes: int) -> str:
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

    def _delete_file(self, file_path: Path) -> bool:
        """
        Delete a single file.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: Path to file to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        assert file_path is not None, "File path cannot be None"
        assert isinstance(file_path, Path), "File path must be Path object"

        try:
            file_path.unlink()
            return True
        except Exception as e:
            logger.debug(f"Failed to delete file {file_path}: {e}")
            return False

    def _delete_directory(self, dir_path: Path) -> bool:
        """
        Delete a directory and its contents.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            dir_path: Path to directory to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        assert dir_path is not None, "Directory path cannot be None"
        assert isinstance(dir_path, Path), "Directory path must be Path object"

        try:
            shutil.rmtree(dir_path)
            return True
        except Exception as e:
            logger.debug(f"Failed to delete directory {dir_path}: {e}")
            return False

    def _delete_item(self, item: Path) -> bool:
        """
        Delete a file or directory.

        Rule #1: Early returns eliminate if/elif nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            item: Path to item to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        assert item is not None, "Item cannot be None"
        assert isinstance(item, Path), "Item must be Path object"
        if item.is_file():
            return self._delete_file(item)
        if item.is_dir():
            return self._delete_directory(item)

        # Unknown item type
        logger.debug(f"Unknown item type: {item}")
        return False

    def delete_directory_contents(self, directory: Path) -> int:
        """
        Delete directory contents.

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #2: Fixed upper bound for safety
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            directory: Directory to clean

        Returns:
            Number of items deleted
        """
        assert directory is not None, "Directory cannot be None"
        assert isinstance(directory, Path), "Directory must be Path object"
        if not directory.exists():
            logger.debug(f"Directory does not exist: {directory}")
            return 0
        MAX_ITEMS: int = 100_000  # Hard limit
        items_processed: int = 0
        count: int = 0

        try:
            for item in directory.iterdir():
                items_processed += 1
                if items_processed > MAX_ITEMS:
                    logger.error(f"Safety limit: processed {MAX_ITEMS} items, stopping")
                    break
                if self._delete_item(item):
                    count += 1

        except Exception as e:
            logger.error(f"Failed to clean directory {directory}: {e}")
        assert count >= 0, "Deleted count must be non-negative"

        return count

    def create_backup_archive(self, source: Path, destination: Path) -> None:
        """Create backup archive.

        Args:
            source: Source directory
            destination: Destination archive path

        Raises:
            ValueError: If backup fails
        """
        try:
            # Create parent directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Create archive (remove .zip extension as make_archive adds it)
            archive_base = str(destination.with_suffix(""))
            shutil.make_archive(archive_base, "zip", source)

        except Exception as e:
            raise ValueError(f"Backup failed: {e}")

    def extract_backup_archive(self, archive: Path, destination: Path) -> None:
        """Extract backup archive.

        Args:
            archive: Archive file path
            destination: Destination directory

        Raises:
            ValueError: If extraction fails
        """
        try:
            # Create destination if needed
            destination.mkdir(parents=True, exist_ok=True)

            # Extract archive
            shutil.unpack_archive(str(archive), str(destination))

        except Exception as e:
            raise ValueError(f"Restore failed: {e}")

    def get_storage_info(self, project: Path) -> Dict[str, Any]:
        """Get storage information.

        Args:
            project: Project directory

        Returns:
            Storage information dictionary
        """
        storage_dir = project / ".ingestforge" / "storage"

        if not storage_dir.exists():
            return {
                "exists": False,
                "size": 0,
                "file_count": 0,
            }

        file_count = sum(1 for _ in storage_dir.rglob("*") if _.is_file())
        size = self.calculate_directory_size(storage_dir)

        return {
            "exists": True,
            "size": size,
            "size_formatted": self.format_size(size),
            "file_count": file_count,
            "path": str(storage_dir),
        }

    def create_maintenance_summary(
        self, results: Dict[str, Any], operation: str
    ) -> str:
        """Create maintenance operation summary.

        Args:
            results: Operation results
            operation: Operation name

        Returns:
            Summary text
        """
        from rich.panel import Panel

        lines = [
            f"[bold]{operation} Results[/bold]",
            "",
        ]

        if "items_cleaned" in results:
            lines.append(f"Items cleaned: {results['items_cleaned']}")

        if "space_freed" in results:
            lines.append(f"Space freed: {results['space_freed']}")

        if "backup_path" in results:
            lines.append(f"Backup: {results['backup_path']}")

        if "restore_path" in results:
            lines.append(f"Restored to: {results['restore_path']}")

        if "optimizations" in results:
            lines.append("")
            lines.append("Optimizations:")
            for opt in results["optimizations"]:
                lines.append(f"  • {opt}")

        if results.get("errors"):
            lines.append("")
            lines.append("[yellow]Warnings:[/yellow]")
            for error in results["errors"][:3]:
                lines.append(f"  • {error}")

        return Panel("\n".join(lines), border_style="cyan")

    def save_maintenance_report(
        self, output: Path, results: Dict[str, Any], operation: str
    ) -> None:
        """Save maintenance operation report.

        Args:
            output: Output file path
            results: Operation results
            operation: Operation name
        """
        lines = [
            f"# Maintenance Report: {operation}",
            "",
            f"**Date:** {self._get_timestamp()}",
            "",
            "## Summary",
            "",
        ]

        for key, value in results.items():
            if key != "errors":
                lines.append(f"- **{key}:** {value}")

        if results.get("errors"):
            lines.extend(
                [
                    "",
                    "## Warnings",
                    "",
                ]
            )
            for error in results["errors"]:
                lines.append(f"- {error}")

        report_text = "\n".join(lines)

        try:
            output.write_text(report_text, encoding="utf-8")
            self.print_success(f"Report saved: {output}")
        except Exception as e:
            self.print_error(f"Failed to save report: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp string.

        Returns:
            Formatted timestamp
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
