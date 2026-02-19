"""Contradiction detection CLI command.

Detects semantic contradictions between claims in the knowledge base."""

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
MAX_CLAIMS = 100
MAX_TOPIC_LENGTH = 200


def command(
    topic: str = typer.Argument(
        ..., help="Topic or document to check for contradictions"
    ),
    threshold: float = typer.Option(
        0.6, "--threshold", "-t", help="Contradiction score threshold (0.0-1.0)"
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum contradictions to show"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Detect contradictions in knowledge base content.

    Finds claims that semantically contradict each other using
    embedding similarity and negation pattern detection.

    Examples:
        ingestforge analyze contradictions "climate change"
        ingestforge analyze contradictions "project scope" --threshold 0.7
    """
    if not topic.strip():
        console.print("[red]Error: Topic cannot be empty[/red]")
        raise typer.Exit(1)

    topic = topic[:MAX_TOPIC_LENGTH]

    if not 0.0 <= threshold <= 1.0:
        console.print("[red]Error: Threshold must be between 0.0 and 1.0[/red]")
        raise typer.Exit(1)

    try:
        # Load config and storage
        config, storage = CLIInitializer.load_config_and_storage(project)

        # Search for relevant chunks
        console.print(f"[blue]Searching for claims about: {topic}[/blue]")
        results = storage.search(topic, limit=MAX_CLAIMS)

        if not results:
            console.print("[yellow]No content found for topic[/yellow]")
            return

        # Extract claims from chunks
        claims = [r.content for r in results[:MAX_CLAIMS]]
        console.print(f"[green]Found {len(claims)} chunks to analyze[/green]")

        # Run contradiction detection
        from ingestforge.enrichment.contradiction import ContradictionDetector

        detector = ContradictionDetector(similarity_threshold=0.7)

        console.print("[blue]Analyzing for contradictions...[/blue]")
        contradictions = detector.find_all_contradictions(claims, min_score=threshold)

        # Display results
        _display_contradictions(contradictions[:limit])

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: pip install sentence-transformers")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Contradiction detection failed")
        raise typer.Exit(1)


def _display_contradictions(contradictions: list) -> None:
    """Display contradiction results."""
    if not contradictions:
        console.print(
            Panel(
                "[green]No contradictions detected[/green]",
                title="Results",
                border_style="green",
            )
        )
        return

    table = Table(title=f"Found {len(contradictions)} Contradictions")
    table.add_column("Score", style="red", width=8)
    table.add_column("Claim 1", style="cyan", width=40)
    table.add_column("Claim 2", style="yellow", width=40)

    for c in contradictions:
        claim1 = c.claim1[:80] + "..." if len(c.claim1) > 80 else c.claim1
        claim2 = c.claim2[:80] + "..." if len(c.claim2) > 80 else c.claim2
        table.add_row(f"{c.score:.2f}", claim1, claim2)

    console.print(table)
