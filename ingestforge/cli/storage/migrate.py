"""Migrate command - Migrate between storage backends.

Migrates data from one storage backend to another with
progress tracking and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, TaskID

from ingestforge.cli.storage.base import StorageCommand
from ingestforge.storage.migration import MigrationProgress, StorageMigrator


class MigrateCommand(StorageCommand):
    """Migrate between storage backends."""

    def __init__(self) -> None:
        """Initialize migrate command."""
        super().__init__()
        self._progress: Optional[Progress] = None
        self._task: Optional[TaskID] = None

    def execute(
        self,
        source_backend: str,
        target_backend: str,
        source_connection: Optional[str] = None,
        target_connection: Optional[str] = None,
        project: Optional[Path] = None,
        batch_size: int = 100,
        verify: bool = True,
        format: str = "table",
    ) -> int:
        """Migrate storage data.

        Args:
            source_backend: Source backend type (jsonl/chromadb/postgres)
            target_backend: Target backend type
            source_connection: Source connection string (for postgres)
            target_connection: Target connection string (for postgres)
            project: Project directory
            batch_size: Chunks per batch
            verify: Verify after migration
            format: Output format

        Returns:
            0 on success, 1 on error
        """
        try:
            # Get source and target storage
            source = self._create_storage(source_backend, source_connection, project)
            target = self._create_storage(target_backend, target_connection, project)

            self.print_info(f"Migrating from {source_backend} to {target_backend}")

            # Create migrator with progress callback
            migrator = StorageMigrator(
                source, target, batch_size, self._progress_callback
            )

            # Run migration with progress display
            with Progress() as progress:
                self._progress = progress
                self._task = progress.add_task("Migrating...", total=100)
                result = migrator.migrate()

            # Verify if requested
            if verify and result.success:
                self.print_info("Verifying migration...")
                result.success = migrator.verify()

            # Display result
            self._display_result(result, format)

            return 0 if result.success else 1

        except Exception as e:
            return self.handle_error(e, "Migration failed")

    def _create_storage(
        self,
        backend: str,
        connection: Optional[str],
        project: Optional[Path],
    ):
        """Create storage backend instance.

        Args:
            backend: Backend type
            connection: Connection string
            project: Project directory

        Returns:
            Storage backend instance
        """
        config = self.get_config(project)

        if backend == "jsonl":
            from ingestforge.storage.jsonl import JSONLRepository

            return JSONLRepository(config.data_path)

        if backend == "chromadb":
            from ingestforge.storage.chromadb import ChromaDBRepository

            return ChromaDBRepository(persist_directory=config.chromadb_path)

        if backend == "postgres":
            from ingestforge.storage.postgres import PostgresRepository

            conn_str = connection or config.storage.postgres.connection_string
            if not conn_str:
                raise ValueError("PostgreSQL connection string required")
            return PostgresRepository(conn_str)

        raise ValueError(f"Unknown backend: {backend}")

    def _progress_callback(self, progress: MigrationProgress) -> None:
        """Update progress display.

        Args:
            progress: Progress data
        """
        if self._progress and self._task is not None:
            self._progress.update(
                self._task,
                completed=progress.percentage,
                description=f"Migrating {progress.current_document}...",
            )

    def _display_result(self, result, format: str) -> None:
        """Display migration result.

        Args:
            result: MigrationResult
            format: Output format
        """
        self.console.print()

        if format == "json":
            json_str = json.dumps(result.to_dict(), indent=2)
            self.console.print(json_str)
            return

        # Table display
        if result.success:
            self.print_success("Migration completed successfully")
        else:
            self.print_error("Migration failed")

        table = self.create_stats_table(
            {
                "Chunks Migrated": result.chunks_migrated,
                "Chunks Failed": result.chunks_failed,
                "Documents Migrated": result.documents_migrated,
                "Duration": f"{result.duration_seconds:.1f}s",
                "Source Backend": result.source_backend,
                "Target Backend": result.target_backend,
            }
        )
        self.console.print(table)

        if result.errors:
            self.console.print()
            self.print_warning(f"{len(result.errors)} errors occurred")
            for error in result.errors[:5]:
                self.console.print(f"  - {error}")
            if len(result.errors) > 5:
                self.console.print(f"  ... and {len(result.errors) - 5} more")


def command(
    source: str = typer.Argument(..., help="Source backend (jsonl/chromadb/postgres)"),
    target: str = typer.Argument(..., help="Target backend (jsonl/chromadb/postgres)"),
    source_connection: Optional[str] = typer.Option(
        None, "--source-conn", help="Source PostgreSQL connection string"
    ),
    target_connection: Optional[str] = typer.Option(
        None, "--target-conn", help="Target PostgreSQL connection string"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Chunks per batch"),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Verify after migration"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table/json)"
    ),
) -> None:
    """Migrate data between storage backends.

    Transfers all chunks and documents from source to target backend
    with progress tracking and optional verification.

    Supported backends:
    - jsonl: File-based JSONL storage
    - chromadb: ChromaDB vector database
    - postgres: PostgreSQL with pgvector

    Examples:
        # Migrate from JSONL to PostgreSQL
        ingestforge storage migrate jsonl postgres \\
            --target-conn "postgresql://user:pass@localhost/db"

        # Migrate from ChromaDB to PostgreSQL
        ingestforge storage migrate chromadb postgres \\
            --target-conn "postgresql://localhost/ingestforge"

        # Migrate without verification
        ingestforge storage migrate jsonl postgres --no-verify

        # Custom batch size
        ingestforge storage migrate jsonl postgres -b 500
    """
    cmd = MigrateCommand()
    exit_code = cmd.execute(
        source,
        target,
        source_connection,
        target_connection,
        project,
        batch_size,
        verify,
        format,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
