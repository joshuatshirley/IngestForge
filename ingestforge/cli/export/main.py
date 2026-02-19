"""Export subcommands.

Provides tools for exporting knowledge base content:
- markdown: Export to Markdown format
- json: Export to JSON format
- folder-export: Export complete study package
- pdf: Export to PDF format
- context: Export context for RAG applications
- outline: Export with thesis-evidence structure
- pack: Create portable corpus package
- unpack: Restore corpus package
- info: Show package information
- knowledge-graph: Export interactive knowledge graph visualization

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.export import (
    markdown,
    json,
    folder,
    pdf,
    context,
    outline,
    corpus,
    knowledge_graph,
)

# Create export subcommand application
app = typer.Typer(
    name="export",
    help="Export knowledge base content",
    add_completion=False,
)

# Register export commands
app.command("markdown")(markdown.command)
app.command("json")(json.command)
app.command("folder-export")(folder.command)
app.command("pdf")(pdf.command)
app.command("context")(context.command)
app.command("outline")(outline.command)

# Corpus sharing commands
app.command("pack")(corpus.pack_command)
app.command("unpack")(corpus.unpack_command)
app.command("info")(corpus.info_command)

# Knowledge graph export
app.command("knowledge-graph")(knowledge_graph.command)


@app.callback()
def main() -> None:
    """Export tools for IngestForge.

    Export your knowledge base to various formats:
    - Markdown: Human-readable documentation
    - JSON: Structured data for programmatic access

    Features:
    - Filter by search query
    - Limit number of chunks
    - Preserve metadata
    - Multiple output formats

    Use cases:
    - Documentation generation
    - Data backup and archival
    - Integration with other tools
    - Content analysis

    Examples:
        # Export to Markdown
        ingestforge export markdown docs.md

        # Export to JSON
        ingestforge export json data.json

        # Export filtered content
        ingestforge export markdown ml_docs.md --query "machine learning"

        # Export with limit
        ingestforge export json sample.json --limit 100

    For help on specific commands:
        ingestforge export <command> --help
    """
    pass
