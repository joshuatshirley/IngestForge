"""Backup command - Backup project data.

Creates backup archives of project data.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import typer

from ingestforge.cli.maintenance.base import MaintenanceCommand


class BackupCommand(MaintenanceCommand):
    """Backup project data."""

    def execute(
        self,
        output: Path,
        project: Optional[Path] = None,
        include_config: bool = True,
        include_storage: bool = True,
        include_logs: bool = False,
    ) -> int:
        """Create backup archive.

        Args:
            output: Output backup file path (.zip)
            project: Project directory
            include_config: Include configuration
            include_storage: Include storage data
            include_logs: Include log files

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate output path
            if not output.suffix == ".zip":
                output = output.with_suffix(".zip")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Check what exists
            ingestforge_dir = project_path / ".ingestforge"
            if not ingestforge_dir.exists():
                raise ValueError("No IngestForge data found in project")

            # Create backup
            self.print_info(f"Creating backup: {output}")
            results = self._create_backup(
                project_path,
                output,
                include_config,
                include_storage,
                include_logs,
            )

            # Display results
            summary = self.create_maintenance_summary(results, "Backup")
            self.console.print(summary)

            return 0

        except Exception as e:
            return self.handle_error(e, "Backup failed")

    def _backup_items(
        self,
        ingestforge_dir: Path,
        temp_path: Path,
        include_config: bool,
        include_storage: bool,
        include_logs: bool,
    ) -> int:
        """Backup selected items to temp directory.

        Rule #4: No large functions - Extracted from _create_backup

        Returns:
            Number of items backed up
        """
        import shutil

        items_backed_up = 0

        # Backup config
        if include_config:
            config_file = ingestforge_dir / "config.json"
            if config_file.exists():
                shutil.copy2(config_file, temp_path / "config.json")
                items_backed_up += 1

        # Backup storage
        if include_storage:
            storage_dir = ingestforge_dir / "storage"
            if storage_dir.exists():
                shutil.copytree(
                    storage_dir,
                    temp_path / "storage",
                    dirs_exist_ok=True,
                )
                items_backed_up += 1

        # Backup logs
        if include_logs:
            logs_dir = ingestforge_dir / "logs"
            if logs_dir.exists():
                shutil.copytree(
                    logs_dir,
                    temp_path / "logs",
                    dirs_exist_ok=True,
                )
                items_backed_up += 1

        return items_backed_up

    def _build_backup_result(
        self, output: Path, items_backed_up: int
    ) -> dict[str, Any]:
        """Build backup result dictionary.

        Rule #4: No large functions - Extracted from _create_backup
        """
        backup_size = output.stat().st_size
        return {
            "backup_path": str(output),
            "backup_size": self.format_size(backup_size),
            "items_backed_up": items_backed_up,
            "errors": [],
        }

    def _create_backup(
        self,
        project: Path,
        output: Path,
        include_config: bool,
        include_storage: bool,
        include_logs: bool,
    ) -> dict[str, Any]:
        """Create backup archive.

        Rule #4: No large functions - Refactored to <60 lines

        Args:
            project: Project directory
            output: Output file path
            include_config: Include config
            include_storage: Include storage
            include_logs: Include logs

        Returns:
            Backup results
        """
        ingestforge_dir = project / ".ingestforge"

        # Create temporary backup directory
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / ".ingestforge"
            temp_path.mkdir(parents=True, exist_ok=True)

            # Backup items to temp directory
            items_backed_up = self._backup_items(
                ingestforge_dir,
                temp_path,
                include_config,
                include_storage,
                include_logs,
            )

            # Create archive
            self.create_backup_archive(temp_path.parent, output)

        return self._build_backup_result(output, items_backed_up)


# Typer command wrapper
def command(
    output: Path = typer.Argument(..., help="Output backup file (.zip)"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    include_config: bool = typer.Option(
        True, "--config/--no-config", help="Include configuration"
    ),
    include_storage: bool = typer.Option(
        True, "--storage/--no-storage", help="Include storage data"
    ),
    include_logs: bool = typer.Option(False, "--logs", help="Include log files"),
) -> None:
    """Backup project data.

    Creates a backup archive of project data including
    configuration, storage, and optionally logs.

    Default: Backs up config and storage (excludes logs)

    Examples:
        # Create backup
        ingestforge maintenance backup backup.zip

        # Backup everything including logs
        ingestforge maintenance backup full_backup.zip --logs

        # Backup config only
        ingestforge maintenance backup config_backup.zip --no-storage

        # Backup storage only
        ingestforge maintenance backup storage_backup.zip --no-config

        # Specific project
        ingestforge maintenance backup backup.zip -p /path/to/project
    """
    cmd = BackupCommand()
    exit_code = cmd.execute(
        output, project, include_config, include_storage, include_logs
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
