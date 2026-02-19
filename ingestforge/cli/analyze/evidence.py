"""Evidence linking CLI command.

Links claims to supporting/refuting evidence in the knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.core import CLIInitializer
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
console = Console()
MAX_CLAIM_LENGTH = 2000
MAX_EVIDENCE = 20


def command(
    claim: str = typer.Argument(..., help="Claim to find evidence for"),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum evidence items to show"
    ),
    support_threshold: float = typer.Option(
        0.6, "--support", "-s", help="Support score threshold (0.0-1.0)"
    ),
    refute_threshold: float = typer.Option(
        0.6, "--refute", "-r", help="Refutation score threshold (0.0-1.0)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    show_neutral: bool = typer.Option(
        False, "--neutral", "-n", help="Also show neutral evidence"
    ),
) -> None:
    """Link claims to supporting/refuting evidence.

    Searches the knowledge base for evidence that supports or
    refutes a given claim using semantic similarity and
    contradiction detection.

    Examples:
        ingestforge analyze evidence "The Earth is round"
        ingestforge analyze evidence "Climate change is real" --limit 20
        ingestforge analyze evidence "Python is faster than C" --neutral
    """
    if not claim.strip():
        console.print("[red]Error: Claim cannot be empty[/red]")
        raise typer.Exit(1)

    claim = claim[:MAX_CLAIM_LENGTH]

    if not 0.0 <= support_threshold <= 1.0:
        console.print("[red]Error: Support threshold must be between 0.0 and 1.0[/red]")
        raise typer.Exit(1)

    if not 0.0 <= refute_threshold <= 1.0:
        console.print("[red]Error: Refute threshold must be between 0.0 and 1.0[/red]")
        raise typer.Exit(1)

    try:
        # Load config and storage
        config, storage = CLIInitializer.load_config_and_storage(project)

        console.print(f"[blue]Searching evidence for: {claim}[/blue]")

        # Run evidence linking
        from ingestforge.enrichment.evidence_linker import EvidenceLinker

        linker = EvidenceLinker(
            support_threshold=support_threshold,
            refute_threshold=refute_threshold,
        )

        result = linker.link_evidence(
            claim=claim,
            storage=storage,
            top_k=min(limit, MAX_EVIDENCE),
        )

        # Display results
        _display_results(result, show_neutral)

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: pip install sentence-transformers")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Evidence linking failed")
        raise typer.Exit(1)


def _display_results(result: "EvidenceLinkResult", show_neutral: bool) -> None:
    """Display evidence linking results."""
    from ingestforge.enrichment.evidence_linker import SupportType

    # Summary panel
    summary = (
        f"[green]Supporting: {result.total_support}[/green] | "
        f"[red]Refuting: {result.total_refute}[/red] | "
        f"[yellow]Neutral: {result.total_neutral}[/yellow]"
    )

    console.print(Panel(summary, title="Evidence Summary", border_style="blue"))

    if result.key_entities:
        console.print(
            f"\n[dim]Key entities: {', '.join(result.key_entities[:5])}[/dim]"
        )

    # Filter evidence
    evidence_to_show = []
    for e in result.linked_evidence:
        if e.support_type == SupportType.NEUTRAL and not show_neutral:
            continue
        evidence_to_show.append(e)

    if not evidence_to_show:
        console.print("\n[yellow]No relevant evidence found[/yellow]")
        return

    # Evidence table
    table = Table(title=f"\nLinked Evidence ({len(evidence_to_show)} items)")
    table.add_column("Type", style="bold", width=10)
    table.add_column("Conf", width=6)
    table.add_column("Evidence", width=60)
    table.add_column("Source", width=20)

    for e in evidence_to_show:
        # Color by type
        if e.support_type == SupportType.SUPPORTS:
            type_str = "[green]SUPPORT[/green]"
        elif e.support_type == SupportType.REFUTES:
            type_str = "[red]REFUTE[/red]"
        else:
            type_str = "[yellow]NEUTRAL[/yellow]"

        # Truncate evidence text
        evidence_text = (
            e.evidence_text[:100] + "..."
            if len(e.evidence_text) > 100
            else e.evidence_text
        )

        # Truncate source
        source = e.source[:20] + "..." if len(e.source) > 20 else e.source

        table.add_row(
            type_str,
            f"{e.confidence:.2f}",
            evidence_text,
            source,
        )

    console.print(table)
