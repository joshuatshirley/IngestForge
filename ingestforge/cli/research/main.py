"""Research subcommands.

Provides tools for analyzing and verifying knowledge base quality:
- audit: Audit knowledge base quality and coverage
- verify: Verify source citations and references
- summarize: Multi-agent paper summarization (RES-004)

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.research import audit, verify, summarize

# Create research subcommand application
app = typer.Typer(
    name="research",
    help="Research and verification tools",
    add_completion=False,
)

# Register research commands
app.command("audit")(audit.command)
app.command("verify")(verify.command)
app.command("summarize")(summarize.command)


@app.callback()
def main() -> None:
    """Research and verification tools for IngestForge.

    Analyze and verify knowledge base quality:
    - Quality metrics and coverage analysis
    - Citation verification
    - Source traceability
    - Gap identification
    - Multi-agent paper summarization

    All commands work with the ingested knowledge base.

    Examples:
        # Audit knowledge base
        ingestforge research audit --detailed

        # Verify citations
        ingestforge research verify --show-missing

        # Summarize a paper with multi-agent analysis
        ingestforge research summarize paper.pdf

        # Generate reports
        ingestforge research audit -o audit.md
        ingestforge research verify -o verify.md

    For help on specific commands:
        ingestforge research <command> --help
    """
    pass
