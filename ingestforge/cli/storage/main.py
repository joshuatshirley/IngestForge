"""Storage subcommands.

Provides tools for storage management:
- stats: Show storage statistics
- migrate: Migrate between backends
- health: Check storage health"""

from __future__ import annotations

import typer

from ingestforge.cli.storage import stats, migrate, health

# Create storage subcommand application
app = typer.Typer(
    name="storage",
    help="Storage management",
    add_completion=False,
)

# Register storage commands
app.command("stats")(stats.command)
app.command("migrate")(migrate.command)
app.command("health")(health.command)


@app.callback()
def main() -> None:
    """Storage management for IngestForge.

    Manage storage backends for your RAG application:
    - View storage statistics
    - Migrate between backends
    - Check storage health

    Supported backends:
    - jsonl: File-based JSONL storage (simple, portable)
    - chromadb: ChromaDB vector database (vector search)
    - postgres: PostgreSQL with pgvector (production-grade)

    Use cases:
    - Monitor storage usage
    - Migrate from development to production
    - Health monitoring for alerting
    - Capacity planning

    Examples:
        # Show storage stats
        ingestforge storage stats

        # Check health
        ingestforge storage health

        # Migrate to PostgreSQL
        ingestforge storage migrate jsonl postgres \\
            --target-conn "postgresql://localhost/ingestforge"

        # Get stats as JSON
        ingestforge storage stats --format json

    For help on specific commands:
        ingestforge storage <command> --help
    """
    pass
