"""Analysis Management CLI Commands.

Provides commands to manage stored analysis results:
- list: List stored analyses with filters
- search: Semantic search across analyses
- show: Display a specific analysis
- delete: Remove an analysis
- refresh: Re-run analysis with current LLM
- for-document: Get all analyses for a document"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
console = Console()

# =============================================================================
# Helper Functions
# =============================================================================


def _get_analysis_storage() -> Any:
    """Get analysis storage instance.

    Returns:
        AnalysisStorage: The analysis storage backend instance.
    """
    from ingestforge.storage.analysis import get_analysis_storage

    config_path: Path = Path.cwd() / "ingestforge.yaml"
    persist_dir: Path = Path.cwd() / ".data" / "chromadb"

    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            persist_path: str = (
                config.get("storage", {})
                .get("chromadb", {})
                .get("persist_directory", ".data/chromadb")
            )
            persist_dir = Path.cwd() / persist_path
        except Exception:
            pass

    return get_analysis_storage(persist_dir)


def _format_analysis_row(record: Any) -> Tuple[str, str, str, str, str]:
    """Format analysis record for table display."""
    from datetime import datetime

    # Format created_at
    try:
        dt = datetime.fromisoformat(record.created_at)
        created = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        created = record.created_at[:16] if record.created_at else "Unknown"

    # Truncate title/content for display
    title = record.title or record.content[:50]
    if len(title) > 50:
        title = title[:47] + "..."

    return (
        record.analysis_id,
        record.analysis_type,
        title,
        f"{record.confidence:.0%}",
        created,
    )


# =============================================================================
# Commands
# =============================================================================


def list_command(
    analysis_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by analysis type (theme, argument, etc.)"
    ),
    library: Optional[str] = typer.Option(
        None, "--library", "-l", help="Filter by library"
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of results"),
) -> None:
    """List stored analyses.

    Shows all stored analysis results with optional filters.

    Examples:
        # List all analyses
        ingestforge analysis list

        # List only theme analyses
        ingestforge analysis list --type theme

        # List analyses in a library
        ingestforge analysis list --library research
    """
    try:
        storage = _get_analysis_storage()
        records = storage.list_analyses(
            analysis_type=analysis_type,
            library=library,
            limit=limit,
        )

        if not records:
            console.print("[yellow]No analyses found[/yellow]")
            return

        # Create table
        table = Table(title=f"Stored Analyses ({len(records)} total)")
        table.add_column("ID", style="cyan", width=18)
        table.add_column("Type", style="magenta", width=12)
        table.add_column("Title", style="white")
        table.add_column("Confidence", style="green", width=10)
        table.add_column("Created", style="dim", width=16)

        for record in records:
            row = _format_analysis_row(record)
            table.add_row(*row)

        console.print()
        console.print(table)

        # Show statistics
        stats = storage.get_statistics()
        console.print()
        console.print(f"[dim]Total: {stats['total_analyses']} analyses[/dim]")

        if stats.get("by_type"):
            types_str = ", ".join(f"{t}: {c}" for t, c in stats["by_type"].items())
            console.print(f"[dim]By type: {types_str}[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing analyses: {e}[/red]")
        logger.error(f"List analyses failed: {e}")
        raise typer.Exit(code=1)


def search_command(
    query: str = typer.Argument(..., help="Search query"),
    analysis_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by analysis type"
    ),
    library: Optional[str] = typer.Option(
        None, "--library", "-l", help="Filter by library"
    ),
    limit: int = typer.Option(10, "-k", "--limit", help="Number of results"),
) -> None:
    """Semantic search across stored analyses.

    Search for analyses by content similarity.

    Examples:
        # Search for theme-related analyses
        ingestforge analysis search "redemption and forgiveness"

        # Search within theme analyses only
        ingestforge analysis search "power corruption" --type theme

        # Limit results
        ingestforge analysis search "character development" -k 5
    """
    try:
        storage = _get_analysis_storage()
        records = storage.search_analyses(
            query=query,
            k=limit,
            analysis_type=analysis_type,
            library=library,
        )

        if not records:
            console.print("[yellow]No matching analyses found[/yellow]")
            return

        console.print()
        console.print(f"[bold]Search Results for: [cyan]{query}[/cyan][/bold]")
        console.print()

        for i, record in enumerate(records, 1):
            # Display each result as a panel
            content_preview = record.content[:300]
            if len(record.content) > 300:
                content_preview += "..."

            panel_content = [
                f"[bold]{record.title or 'Untitled'}[/bold]",
                f"[dim]Type: {record.analysis_type} | "
                f"Confidence: {record.confidence:.0%} | "
                f"Source: {record.source_document}[/dim]",
                "",
                content_preview,
            ]

            panel = Panel(
                "\n".join(panel_content),
                title=f"[cyan]{i}. {record.analysis_id}[/cyan]",
                border_style="blue",
            )
            console.print(panel)

    except Exception as e:
        console.print(f"[red]Error searching analyses: {e}[/red]")
        logger.error(f"Search analyses failed: {e}")
        raise typer.Exit(code=1)


def show_command(
    analysis_id: str = typer.Argument(..., help="Analysis ID to display"),
) -> None:
    """Display a specific analysis.

    Shows full details of a stored analysis.

    Examples:
        ingestforge analysis show analysis_abc123def456
    """
    try:
        storage = _get_analysis_storage()
        record = storage.get_analysis(analysis_id)

        if not record:
            console.print(f"[red]Analysis not found: {analysis_id}[/red]")
            raise typer.Exit(code=1)

        # Build display content
        lines = [
            f"# {record.title or 'Analysis'}",
            "",
            f"**ID:** {record.analysis_id}",
            f"**Type:** {record.analysis_type}",
            f"**Source Document:** {record.source_document}",
            f"**Confidence:** {record.confidence:.0%}",
            f"**Created:** {record.created_at}",
            f"**Library:** {record.library}",
            "",
            "---",
            "",
            "## Content",
            "",
            record.content,
            "",
        ]

        # Add metadata if present
        if record.metadata:
            lines.extend(
                [
                    "---",
                    "",
                    "## Metadata",
                    "",
                ]
            )
            import json

            lines.append(f"```json\n{json.dumps(record.metadata, indent=2)}\n```")

        # Add source chunks if present
        if record.source_chunks:
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    "## Source Chunks",
                    "",
                ]
            )
            for chunk_id in record.source_chunks[:10]:
                lines.append(f"- {chunk_id}")

        console.print()
        panel = Panel(
            Markdown("\n".join(lines)),
            title=f"[bold cyan]Analysis: {record.analysis_id}[/bold cyan]",
            border_style="cyan",
        )
        console.print(panel)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error displaying analysis: {e}[/red]")
        logger.error(f"Show analysis failed: {e}")
        raise typer.Exit(code=1)


def delete_command(
    analysis_id: str = typer.Argument(..., help="Analysis ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a stored analysis.

    Permanently removes an analysis from storage.

    Examples:
        # Delete with confirmation
        ingestforge analysis delete analysis_abc123def456

        # Delete without confirmation
        ingestforge analysis delete analysis_abc123def456 --force
    """
    try:
        storage = _get_analysis_storage()

        # Get analysis first to show details
        record = storage.get_analysis(analysis_id)
        if not record:
            console.print(f"[red]Analysis not found: {analysis_id}[/red]")
            raise typer.Exit(code=1)

        # Confirm deletion
        if not force:
            console.print("[yellow]Analysis to delete:[/yellow]")
            console.print(f"  ID: {record.analysis_id}")
            console.print(f"  Type: {record.analysis_type}")
            console.print(f"  Title: {record.title or record.content[:50]}")
            console.print(f"  Source: {record.source_document}")
            console.print()

            confirm = typer.confirm("Are you sure you want to delete this analysis?")
            if not confirm:
                console.print("[dim]Cancelled[/dim]")
                return

        # Delete
        success = storage.delete_analysis(analysis_id)
        if success:
            console.print(f"[green]Deleted analysis: {analysis_id}[/green]")
        else:
            console.print("[red]Failed to delete analysis[/red]")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error deleting analysis: {e}[/red]")
        logger.error(f"Delete analysis failed: {e}")
        raise typer.Exit(code=1)


def _get_refresh_command(analysis_type: str, source_document: str) -> str:
    """Get refresh command for analysis type.

    Args:
        analysis_type: Type of analysis (theme, argument, etc.)
        source_document: Source document path

    Returns:
        Command string to refresh the analysis
    """
    # Dictionary dispatch to avoid nested if/elif (JPL-003)
    commands = {
        "theme": f'ingestforge lit themes "{source_document}" --store',
        "argument": f'ingestforge argument analyze "{source_document}" --store',
        "comprehension": f'ingestforge comprehension explain "{source_document}" --store',
    }
    return commands.get(
        analysis_type, "Re-run the original analysis command with --store flag"
    )


def refresh_command(
    analysis_id: str = typer.Argument(..., help="Analysis ID to refresh"),
) -> None:
    """Re-run an analysis with the current LLM.

    Note: Full refresh requires re-running the original analysis command.
    This command provides guidance on how to refresh.

    Examples:
        ingestforge analysis refresh analysis_abc123def456
    """
    try:
        storage = _get_analysis_storage()
        record = storage.get_analysis(analysis_id)

        if not record:
            console.print(f"[red]Analysis not found: {analysis_id}[/red]")
            raise typer.Exit(code=1)

        console.print()
        console.print("[bold]Analysis Refresh[/bold]")
        console.print()
        console.print(f"To refresh this {record.analysis_type} analysis:")
        console.print()

        # Use helper to get command (JPL-003: Reduced nesting)
        refresh_cmd = _get_refresh_command(record.analysis_type, record.source_document)
        console.print(f"  {refresh_cmd}")

        console.print()
        console.print("[dim]The old analysis will remain until you delete it.[/dim]")
        console.print(
            f"[dim]To delete: ingestforge analysis delete {analysis_id}[/dim]"
        )

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Refresh analysis failed: {e}")
        raise typer.Exit(code=1)


def for_document_command(
    doc_path: str = typer.Argument(..., help="Document path or name"),
) -> None:
    """Get all analyses for a document.

    Shows all stored analyses related to a specific document.

    Examples:
        ingestforge analysis for-document "Hamlet"
        ingestforge analysis for-document "/path/to/document.pdf"
    """
    try:
        storage = _get_analysis_storage()
        records = storage.get_analyses_for_document(doc_path)

        if not records:
            console.print(f"[yellow]No analyses found for: {doc_path}[/yellow]")
            return

        # Create table
        table = Table(title=f"Analyses for: {doc_path}")
        table.add_column("ID", style="cyan", width=18)
        table.add_column("Type", style="magenta", width=12)
        table.add_column("Title", style="white")
        table.add_column("Confidence", style="green", width=10)
        table.add_column("Created", style="dim", width=16)

        for record in records:
            row = _format_analysis_row(record)
            table.add_row(*row)

        console.print()
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"For-document failed: {e}")
        raise typer.Exit(code=1)


def stats_command() -> None:
    """Show analysis storage statistics.

    Displays counts and breakdowns of stored analyses.

    Examples:
        ingestforge analysis stats
    """
    try:
        storage = _get_analysis_storage()
        stats = storage.get_statistics()

        console.print()
        console.print("[bold]Analysis Storage Statistics[/bold]")
        console.print()

        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Analyses", str(stats["total_analyses"]))

        if stats.get("by_type"):
            table.add_row("", "")  # Spacer
            for atype, count in sorted(stats["by_type"].items()):
                table.add_row(f"  {atype}", str(count))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Stats failed: {e}")
        raise typer.Exit(code=1)


# =============================================================================
# Typer App Registration
# =============================================================================

app = typer.Typer(
    name="analysis",
    help="Manage stored analysis results",
    add_completion=False,
)

app.command("list")(list_command)
app.command("search")(search_command)
app.command("show")(show_command)
app.command("delete")(delete_command)
app.command("refresh")(refresh_command)
app.command("for-document")(for_document_command)
app.command("stats")(stats_command)


@app.callback()
def main() -> None:
    """Manage stored analysis results.

    Analysis results from theme detection, argument analysis, and other
    deep analysis commands can be stored in the vector database for
    later searching and retrieval.

    Commands:
        list         - List stored analyses
        search       - Semantic search across analyses
        show         - Display a specific analysis
        delete       - Remove an analysis
        refresh      - Re-run analysis guidance
        for-document - Get analyses for a document
        stats        - Show storage statistics

    Examples:
        # List all stored analyses
        ingestforge analysis list

        # Search for specific content
        ingestforge analysis search "redemption theme"

        # View a specific analysis
        ingestforge analysis show analysis_abc123

        # Delete an analysis
        ingestforge analysis delete analysis_abc123

    To store an analysis, use the --store flag with analysis commands:
        ingestforge lit themes "Hamlet" --store
    """
    pass
