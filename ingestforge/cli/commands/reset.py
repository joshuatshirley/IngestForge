"""Reset command - Clear all processed data (DEAD-D04).

This command wipes all data and resets the project to initial state.
Requires explicit confirmation to prevent accidental data loss.

NASA JPL Commandments compliance:
- Rule #1: Simple control flow
- Rule #4: Functions <60 lines
- Rule #7: Validate confirmation parameter
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel

from ingestforge.cli.core import IngestForgeCommand


class ResetCommand(IngestForgeCommand):
    """Reset all processed data in the project."""

    def execute(
        self,
        project: Optional[Path] = None,
        force: bool = False,
    ) -> int:
        """Reset all processed data.

        Args:
            project: Project directory (default: current)
            force: Skip confirmation prompt

        Returns:
            0 on success, 1 on error
        """
        try:
            ctx = self.initialize_context(project, require_storage=True)
            config = ctx["config"]
            storage = ctx["storage"]

            # Show warning
            self._show_warning(config)

            # Get confirmation
            if not force:
                confirmed = self._get_confirmation()
                if not confirmed:
                    self.console.print("[yellow]Reset cancelled.[/yellow]")
                    return 0

            # Perform reset
            self._perform_reset(ctx)

            self.console.print(
                "[bold green]Reset complete.[/bold green] "
                "All processed data has been cleared."
            )
            return 0

        except Exception as e:
            return self.handle_error(e, "Reset failed")

    def _show_warning(self, config: object) -> None:
        """Display warning about data loss.

        Args:
            config: Project configuration
        """
        warning = f"""[bold red]WARNING: This will permanently delete:[/bold red]

- All processed chunks
- All embeddings
- All index data
- Processing state and history

[bold]Project:[/bold] {config.project.name}
[bold]Data Directory:[/bold] {config.project.data_dir}

This action cannot be undone."""

        self.console.print(
            Panel(
                warning,
                title="[bold red]Reset Project[/bold red]",
                border_style="red",
            )
        )

    def _get_confirmation(self) -> bool:
        """Get user confirmation for reset.

        Returns:
            True if user confirms, False otherwise
        """
        response = typer.prompt(
            "Type 'RESET' to confirm (or anything else to cancel)",
            default="",
        )
        return response.upper() == "RESET"

    def _perform_reset(self, ctx: dict) -> None:
        """Perform the actual reset.

        Args:
            ctx: Context dict with pipeline and storage
        """
        from ingestforge.core.pipeline import Pipeline

        config = ctx["config"]
        project_path = ctx["project_path"]

        # Create pipeline and call reset
        pipeline = Pipeline(config=config, base_path=project_path)
        pipeline.reset(confirm=True)


def reset_command(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory (default: current directory)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt (dangerous)",
    ),
) -> None:
    """Reset project and clear all processed data.

    This command permanently deletes all processed chunks, embeddings,
    and index data. Use with caution!

    Examples:
        # Interactive reset with confirmation
        ingestforge reset

        # Force reset without confirmation
        ingestforge reset --force
    """
    cmd = ResetCommand()
    raise typer.Exit(cmd.execute(project=project, force=force))
