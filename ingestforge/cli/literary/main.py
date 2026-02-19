"""Literary analysis subcommands.

Provides tools for analyzing literary works:
- themes: Extract and analyze major themes
- character: Analyze character development
- symbols: Identify symbolic patterns
- arc: Analyze story structure
- outline: Generate structural outline

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.literary import themes, character, symbols, outline, arc

# Create literary subcommand application
app = typer.Typer(
    name="lit",
    help="Literary analysis tools",
    add_completion=False,
)

# Register literary analysis commands
app.command("themes")(themes.command)
app.command("character")(character.command)
app.command("symbols")(symbols.command)
app.command("arc")(arc.command)
app.command("outline")(outline.command)


@app.callback()
def main() -> None:
    """Literary analysis tools for IngestForge.

    Analyze literary works to extract:
    - Themes and motifs
    - Character development and relationships
    - Symbolic patterns
    - Story structure and narrative arc
    - Narrative outline

    All commands require documents about the literary work to be
    ingested into the knowledge base first.

    Examples:
        # Analyze themes
        ingestforge lit themes "Hamlet"

        # Analyze character
        ingestforge lit character "Macbeth" --character "Lady Macbeth"

        # Identify symbols
        ingestforge lit symbols "The Great Gatsby"

        # Analyze story arc
        ingestforge lit arc "The Odyssey" --structure hero-journey

        # Generate outline
        ingestforge lit outline "Pride and Prejudice" --detailed

    For help on specific commands:
        ingestforge lit <command> --help
    """
    pass
