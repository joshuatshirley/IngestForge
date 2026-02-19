"""Citation subcommands.

Provides tools for extracting and formatting citations:
- extract: Extract citations from documents
- format: Format citations in various styles
- graph: Visualize citation networks
- validate: Check citation completeness
- bibliography: Generate complete bibliography

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.citation import extract, format, graph, validate, bibliography

# Create citation subcommand application
app = typer.Typer(
    name="citation",
    help="Citation extraction and formatting tools",
    add_completion=False,
)

# Register citation commands
app.command("extract")(extract.command)
app.command("format")(format.command)
app.command("graph")(graph.command)
app.command("validate")(validate.command)
app.command("bibliography")(bibliography.command)


@app.callback()
def main() -> None:
    """Citation tools for IngestForge.

    Extract and format citations from your knowledge base
    for academic writing and bibliography generation.

    Features:
    - Extract citations from document metadata
    - Format in standard academic styles (APA, MLA, Chicago)
    - Generate BibTeX for LaTeX documents
    - Filter by source
    - Export formatted bibliographies

    Use cases:
    - Academic paper writing
    - Bibliography generation
    - Reference management
    - Citation verification

    Examples:
        # Extract all citations
        ingestforge citation extract

        # Format in APA style
        ingestforge citation format apa --output bibliography.md

        # Generate BibTeX
        ingestforge citation format bibtex -o references.bib

        # Filter specific source
        ingestforge citation extract --source "research_paper.pdf"

    For help on specific commands:
        ingestforge citation <command> --help
    """
    pass
