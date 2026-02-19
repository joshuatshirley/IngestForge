"""Manual Graph Link CLI command.

Manual Graph Linker
Epic: EP-12 (Knowledge Graph Foundry)

Enables users to manually create semantic links between entities
via the command line.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ingestforge.core.logging import get_logger
from ingestforge.enrichment.manual_linker import (
    ManualGraphLinker,
    ManualLinkRequest,
    create_manual_linker,
)
from ingestforge.enrichment.semantic_linker import LinkType

logger = get_logger(__name__)
console = Console()

# JPL Rule #2: Fixed upper bounds
MAX_LIST_RESULTS = 100


app = typer.Typer(help="Manual graph linking commands")


def _get_linker() -> ManualGraphLinker:
    """Get or create manual linker instance.

    Returns:
        ManualGraphLinker instance.
    """
    return create_manual_linker()


def _display_link_types() -> None:
    """Display available link types."""
    console.print("\n[bold]Available Link Types:[/bold]")
    for link_type in LinkType:
        console.print(f"  - {link_type.value}")
    console.print("  - (or any custom relation string)")


@app.command("add")
def add_link(
    source: str = typer.Argument(..., help="Source entity name or ID"),
    target: str = typer.Argument(..., help="Target entity name or ID"),
    relation: str = typer.Option(
        "relates_to",
        "--relation",
        "-r",
        help="Relationship type (e.g., 'works_at', 'knows', 'cites')",
    ),
    confidence: float = typer.Option(
        1.0, "--confidence", "-c", min=0.0, max=1.0, help="Confidence score (0.0 - 1.0)"
    ),
    evidence: str = typer.Option(
        "", "--evidence", "-e", help="Evidence or reason for the link"
    ),
    show_types: bool = typer.Option(
        False, "--show-types", help="Show available link types and exit"
    ),
) -> None:
    """Add a manual semantic link between two entities.

    Creates a link from SOURCE to TARGET with the specified RELATION type.
    Manual links are marked with 'source: manual' for auditability.

    Examples:
        # Simple link
        ingestforge graph link add "Tim Cook" "Apple Inc" -r works_at

        # Link with confidence and evidence
        ingestforge graph link add "Doc1" "Doc2" -r cites -c 0.9 -e "Citation found"

        # Show available link types
        ingestforge graph link add --show-types
    """
    if show_types:
        _display_link_types()
        return

    try:
        linker = _get_linker()

        request = ManualLinkRequest(
            source_entity=source,
            target_entity=target,
            relation=relation,
            confidence=confidence,
            evidence=evidence,
        )

        result = linker.add_link(request)

        if result.success:
            console.print(
                f"[green]Link created:[/green] {source} --[{relation}]--> {target}"
            )
            console.print(f"[dim]Link ID: {result.link_id}[/dim]")
            console.print(f"[dim]Confidence: {confidence}[/dim]")
            if evidence:
                console.print(f"[dim]Evidence: {evidence}[/dim]")
        else:
            console.print(f"[red]Failed:[/red] {result.message}")
            raise typer.Exit(code=1)

    except AssertionError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to add link: {e}")
        raise typer.Exit(code=1)


@app.command("remove")
def remove_link(
    link_id: str = typer.Argument(..., help="Link ID to remove"),
) -> None:
    """Remove a manual link by its ID.

    Examples:
        ingestforge graph link remove manual_TimCook_AppleInc_works_at_1
    """
    try:
        linker = _get_linker()
        result = linker.remove_link(link_id)

        if result.success:
            console.print(f"[green]Link removed:[/green] {link_id}")
        else:
            console.print(f"[red]Failed:[/red] {result.message}")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to remove link: {e}")
        raise typer.Exit(code=1)


@app.command("list")
def list_links(
    entity: Optional[str] = typer.Option(
        None, "--entity", "-e", help="Filter links by entity (as source or target)"
    ),
    limit: int = typer.Option(
        MAX_LIST_RESULTS, "--limit", "-l", help="Maximum links to display"
    ),
) -> None:
    """List manual links.

    Examples:
        # List all manual links
        ingestforge graph link list

        # List links for specific entity
        ingestforge graph link list -e "Tim Cook"
    """
    try:
        linker = _get_linker()

        if entity:
            links = linker.get_links_for_entity(entity)
        else:
            links = linker.get_manual_links()

        # JPL Rule #2: Enforce bounds
        links = links[:limit]

        if not links:
            console.print("[yellow]No manual links found[/yellow]")
            return

        # Create table
        table = Table(title="Manual Links")
        table.add_column("Source", style="cyan")
        table.add_column("Relation", style="green")
        table.add_column("Target", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Evidence", style="dim")

        for link in links:
            relation = link.metadata.get("relation_string", link.link_type.value)
            evidence = link.evidence[0] if link.evidence else ""
            evidence_short = evidence[:30] + "..." if len(evidence) > 30 else evidence

            table.add_row(
                link.source_artifact_id,
                relation,
                link.target_artifact_id,
                f"{link.confidence:.2f}",
                evidence_short,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(links)} link(s)[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to list links: {e}")
        raise typer.Exit(code=1)


@app.command("stats")
def show_stats() -> None:
    """Show statistics about manual links.

    Examples:
        ingestforge graph link stats
    """
    try:
        linker = _get_linker()
        stats = linker.get_statistics()

        console.print("\n[bold]Manual Link Statistics[/bold]")
        console.print(f"  Total manual links: {stats['total_manual_links']}")
        console.print(f"  Max links allowed: {stats['max_links_allowed']}")
        console.print(f"  Links remaining: {stats['links_remaining']}")

        if stats["link_type_counts"]:
            console.print("\n[bold]Links by Type:[/bold]")
            for link_type, count in stats["link_type_counts"].items():
                console.print(f"  - {link_type}: {count}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to get stats: {e}")
        raise typer.Exit(code=1)


@app.command("types")
def show_types() -> None:
    """Show available link types.

    Examples:
        ingestforge graph link types
    """
    _display_link_types()


# Export command for integration
command = app
