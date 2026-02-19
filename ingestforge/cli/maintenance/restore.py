"""Restore command - Restore from backup.

Restores project data from backup archives.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import typer

from ingestforge.cli.maintenance.base import MaintenanceCommand


class RestoreCommand(MaintenanceCommand):
    """Restore from backup."""

    def execute(
        self,
        backup_file: Path,
        project: Optional[Path] = None,
        force: bool = False,
    ) -> int:
        """Restore from backup archive.

        Args:
            backup_file: Backup file path (.zip)
            project: Project directory
            force: Overwrite existing data

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate backup file
            self.validate_file_path(backup_file, must_exist=True)

            if backup_file.suffix != ".zip":
                raise ValueError("Backup file must be .zip")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Check if data exists
            ingestforge_dir = project_path / ".ingestforge"
            if ingestforge_dir.exists() and not force:
                if not typer.confirm("IngestForge data exists. Overwrite?"):
                    self.print_info("Restore cancelled")
                    return 0

            # Restore backup
            self.print_info(f"Restoring from: {backup_file}")
            results = self._restore_backup(backup_file, project_path)

            # Display results
            summary = self.create_maintenance_summary(results, "Restore")
            self.console.print(summary)

            return 0

        except Exception as e:
            return self.handle_error(e, "Restore failed")

    def _restore_backup(self, backup_file: Path, project: Path) -> dict[str, Any]:
        """Restore from backup archive.

        Args:
            backup_file: Backup file path
            project: Project directory

        Returns:
            Restore results
        """
        import tempfile
        import shutil

        ingestforge_dir = project / ".ingestforge"

        # Extract to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract backup
            self.extract_backup_archive(backup_file, temp_path)

            # Find .ingestforge directory in backup
            backup_ingestforge = temp_path / ".ingestforge"

            if not backup_ingestforge.exists():
                raise ValueError("Invalid backup: no .ingestforge directory found")

            # Remove existing data if present
            if ingestforge_dir.exists():
                shutil.rmtree(ingestforge_dir)

            # Restore data
            shutil.copytree(backup_ingestforge, ingestforge_dir)

        # Count restored items
        items_restored = sum(
            1 for item in ingestforge_dir.iterdir() if item.is_file() or item.is_dir()
        )

        return {
            "restore_path": str(project),
            "items_restored": items_restored,
            "backup_file": str(backup_file),
            "errors": [],
        }


# Typer command wrapper
def command(
    backup_file: Path = typer.Argument(..., help="Backup file to restore (.zip)"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing data"),
) -> None:
    """Restore from backup.

    Restores project data from a backup archive.
    Will prompt for confirmation if data exists.

    WARNING: This will overwrite existing project data!

    Examples:
        # Restore from backup (with confirmation)
        ingestforge maintenance restore backup.zip

        # Restore without confirmation
        ingestforge maintenance restore backup.zip --force

        # Restore to specific project
        ingestforge maintenance restore backup.zip -p /path/to/project
    """
    cmd = RestoreCommand()
    exit_code = cmd.execute(backup_file, project, force)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
