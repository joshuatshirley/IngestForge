"""Corpus sharing commands for portable knowledge base transfer.

Provides pack and unpack commands for creating and restoring
portable corpus packages."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from ingestforge.core.export import (
    CorpusPackager,
    CorpusImporter,
    PackageManifest,
    get_package_info,
)
from ingestforge.cli.export.base import ExportCommand


class CorpusExportCommand(ExportCommand):
    """Command for creating portable corpus packages."""

    def execute(
        self,
        output: Path,
        project: Optional[Path] = None,
        include_embeddings: bool = True,
        include_state: bool = True,
    ) -> int:
        """Create portable corpus package.

        Args:
            output: Output .zip file path
            project: Project directory (default: current)
            include_embeddings: Include embedding vectors
            include_state: Include pipeline state

        Returns:
            0 on success, 1 on error
        """
        try:
            # Resolve project directory
            project_dir = project or Path.cwd()
            if not project_dir.exists():
                self.print_error(f"Project directory not found: {project_dir}")
                return 1

            # Ensure .zip extension
            if not output.suffix.lower() == ".zip":
                output = output.with_suffix(".zip")

            # Create packager
            packager = CorpusPackager(
                project_dir=project_dir,
                include_embeddings=include_embeddings,
                include_state=include_state,
            )

            # Package with progress
            self.print_info("Collecting files...")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Packaging...", total=0)

                def update_progress(current: int, total: int, name: str) -> None:
                    progress.update(task, total=total, completed=current)
                    progress.update(task, description=f"Packaging {name}")

                result = packager.package(output, progress_callback=update_progress)

            if not result.success:
                self.print_error(f"Packaging failed: {result.error}")
                return 1

            # Display success summary
            self._display_summary(result.output_path, result.manifest)
            return 0

        except Exception as e:
            return self.handle_error(e, "Corpus packaging failed")

    def _display_summary(
        self, output_path: Path, manifest: Optional[PackageManifest]
    ) -> None:
        """Display packaging summary.

        Args:
            output_path: Created package path
            manifest: Package manifest
        """
        self.console.print()

        if manifest:
            table = Table(title="Package Summary", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Output", str(output_path))
            table.add_row("Files", str(manifest.file_count))
            table.add_row("Size", self._format_size(manifest.total_size_bytes))
            table.add_row("Storage", manifest.storage_type)
            table.add_row("Embeddings", "Yes" if manifest.includes_embeddings else "No")
            table.add_row("State", "Yes" if manifest.includes_state else "No")

            self.console.print(table)

        self.print_success(f"Package created: {output_path}")

    def _format_size(self, bytes_count: int) -> str:
        """Format byte count as human-readable size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"


class CorpusImportCommand(ExportCommand):
    """Command for restoring corpus packages."""

    def execute(
        self,
        package: Path,
        target: Optional[Path] = None,
        verify: bool = True,
        force: bool = False,
    ) -> int:
        """Import corpus package.

        Args:
            package: Path to .zip package
            target: Target directory (default: current)
            verify: Verify checksums after import
            force: Overwrite existing files

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate package exists
            if not package.exists():
                self.print_error(f"Package not found: {package}")
                return 1

            # Preview package
            manifest = get_package_info(package)
            if not manifest:
                self.print_error("Invalid package: Could not read manifest")
                return 1

            self._display_package_info(manifest)

            # Resolve target directory
            target_dir = target or Path.cwd()

            # Create importer
            importer = CorpusImporter(
                verify_checksums=verify,
                overwrite_existing=force,
            )

            # Validate
            self.print_info("Validating package...")
            validation = importer.validate(package)

            if validation.warnings:
                for warning in validation.warnings:
                    self.print_warning(warning)

            if not validation.valid:
                for error in validation.errors:
                    self.print_error(error)
                return 1

            # Import with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Importing...", total=0)

                def update_progress(current: int, total: int, name: str) -> None:
                    progress.update(task, total=total, completed=current)
                    progress.update(task, description=f"Extracting {name}")

                result = importer.import_package(
                    package, target_dir, progress_callback=update_progress
                )

            if not result.success:
                self.print_error(f"Import failed: {result.error}")
                return 1

            # Show warnings
            if result.warnings:
                self.console.print()
                for warning in result.warnings:
                    self.print_warning(warning)

            self.print_success(
                f"Imported {result.files_imported} files to {target_dir}"
            )
            return 0

        except Exception as e:
            return self.handle_error(e, "Corpus import failed")

    def _display_package_info(self, manifest: PackageManifest) -> None:
        """Display package info before import."""
        panel = Panel(
            f"[cyan]Files:[/cyan] {manifest.file_count}\n"
            f"[cyan]Storage:[/cyan] {manifest.storage_type}\n"
            f"[cyan]Created:[/cyan] {manifest.created_at[:19] if manifest.created_at else 'Unknown'}\n"
            f"[cyan]Platform:[/cyan] {manifest.source_platform}",
            title="Package Info",
            border_style="blue",
        )
        self.console.print(panel)


# Typer command wrappers


def pack_command(
    output: Path = typer.Argument(..., help="Output .zip file path"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    no_embeddings: bool = typer.Option(
        False, "--no-embeddings", help="Exclude embedding vectors"
    ),
    no_state: bool = typer.Option(
        False, "--no-state", help="Exclude pipeline state files"
    ),
) -> None:
    """Create portable corpus package.

    Packages your knowledge base into a ZIP file that can be
    shared with others or restored on another machine.

    The package includes:
    - .ingest/ directory (config and metadata)
    - .data/ directory (storage backend data)
    - Configuration files

    Examples:
        # Create package in current directory
        ingestforge export pack corpus.zip

        # Create package from specific project
        ingestforge export pack corpus.zip --project ./my_project

        # Create lightweight package (no embeddings)
        ingestforge export pack corpus.zip --no-embeddings
    """
    cmd = CorpusExportCommand()
    exit_code = cmd.execute(
        output=output,
        project=project,
        include_embeddings=not no_embeddings,
        include_state=not no_state,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def unpack_command(
    package: Path = typer.Argument(..., help="Package .zip file path"),
    target: Optional[Path] = typer.Option(
        None, "--target", "-t", help="Target directory"
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Skip checksum verification"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
) -> None:
    """Import corpus package.

    Restores a corpus package to a target directory.
    Validates package integrity and handles cross-platform
    path normalization.

    Examples:
        # Import to current directory
        ingestforge export unpack corpus.zip

        # Import to specific directory
        ingestforge export unpack corpus.zip --target ./restored

        # Force overwrite existing files
        ingestforge export unpack corpus.zip --force
    """
    cmd = CorpusImportCommand()
    exit_code = cmd.execute(
        package=package,
        target=target,
        verify=not no_verify,
        force=force,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def info_command(
    package: Path = typer.Argument(..., help="Package .zip file path"),
) -> None:
    """Show corpus package information.

    Displays metadata about a corpus package without extracting it.

    Examples:
        ingestforge export info corpus.zip
    """
    console = Console()

    if not package.exists():
        console.print(f"[red]Error:[/red] Package not found: {package}")
        raise typer.Exit(code=1)

    manifest = get_package_info(package)
    if not manifest:
        console.print("[red]Error:[/red] Invalid package or missing manifest")
        raise typer.Exit(code=1)

    table = Table(title="Package Information", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", manifest.version)
    table.add_row(
        "Created", manifest.created_at[:19] if manifest.created_at else "Unknown"
    )
    table.add_row("Platform", manifest.source_platform)
    table.add_row("Files", str(manifest.file_count))
    table.add_row("Storage Type", manifest.storage_type)
    table.add_row(
        "Includes Embeddings", "Yes" if manifest.includes_embeddings else "No"
    )
    table.add_row("Includes State", "Yes" if manifest.includes_state else "No")

    if manifest.checksums:
        table.add_row("Verified Files", str(len(manifest.checksums)))

    console.print(table)
