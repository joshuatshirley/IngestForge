"""Writing support command group - Academic writing assistance.

This module provides the main CLI interface for writing assistance tools:
- draft: Generate drafts with citations
- outline: Create structured outlines
- paraphrase: Paraphrase and rewrite text
- rewrite: Rewrite text with tone adjustment
- cite: Citation management (insert, format, check)
- quote: Find quotable passages
- thesis: Evaluate thesis statements"""

from __future__ import annotations

import typer

from ingestforge.cli.writing import quote, thesis, draft, paraphrase, outline, cite

# Create writing command group
app = typer.Typer(
    name="writing",
    help="Academic writing assistance tools",
    no_args_is_help=True,
)

# Create cite subgroup
cite_app = typer.Typer(
    name="cite",
    help="Citation management tools",
    no_args_is_help=True,
)

# Register main commands
app.command(name="quote")(quote.command)
app.command(name="thesis")(thesis.command)
app.command(name="draft")(draft.command)
app.command(name="paraphrase")(paraphrase.command)
app.command(name="outline")(outline.command)
app.command(name="rewrite")(paraphrase.rewrite_command)
app.command(name="simplify")(paraphrase.simplify_command)

# Register cite subcommands
cite_app.command(name="insert")(cite.insert_command)
cite_app.command(name="format")(cite.format_command)
cite_app.command(name="check")(cite.check_command)

# Add cite subgroup to main app
app.add_typer(cite_app, name="cite")


@app.callback()
def main() -> None:
    """Academic writing assistance tools.

    IngestForge provides a comprehensive suite of writing tools
    to help with academic and professional writing tasks.

    Features:
    - Draft generation with citations
    - Outline creation from source material
    - Text paraphrasing and rewriting
    - Citation management (APA, MLA, Chicago, IEEE, Harvard)
    - Thesis statement evaluation
    - Quote finding

    Examples:
        # Generate a draft
        ingestforge writing draft "Introduction to neural networks" --style academic

        # Create an outline
        ingestforge writing outline "Machine Learning Thesis" --depth 3

        # Paraphrase text
        ingestforge writing paraphrase "The quick brown fox" --style formal

        # Rewrite with tone
        ingestforge writing rewrite input.txt --tone professional

        # Insert citations
        ingestforge writing cite insert draft.md --style apa

        # Format bibliography
        ingestforge writing cite format references.bib --style mla

        # Verify citations
        ingestforge writing cite check paper.md

    For help on specific commands:
        ingestforge writing <command> --help
    """
    pass
