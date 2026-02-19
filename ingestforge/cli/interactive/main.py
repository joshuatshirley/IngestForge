"""Interactive subcommands.

Provides interactive query tools:
- ask: Interactive REPL for conversational queries

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.interactive import ask, shell, menu, gui_menu

# Create interactive subcommand application
app = typer.Typer(
    name="interactive",
    help="Interactive query tools",
    add_completion=False,
)

# Register interactive commands
app.command("ask")(ask.command)
app.command("shell")(shell.command)
app.command("menu")(menu.command)


@app.command("gui")
def gui_command() -> None:
    """Launch the IngestForge graphical user interface.

    Opens a professional desktop GUI with:
    - Document import with drag-and-drop
    - Real-time processing visualization
    - Query interface with AI answers
    - Literary analysis tools
    - LLM configuration

    Examples:
        ingestforge interactive gui
    """
    gui_menu.main()


@app.callback()
def main() -> None:
    """Interactive tools for IngestForge.

    Conversational query interface with context awareness:
    - REPL mode for natural conversations
    - Maintained conversation history
    - Real-time knowledge base search
    - Easy exploration and discovery

    Perfect for:
    - Exploring new knowledge bases
    - Research and investigation
    - Learning and discovery
    - Quick information retrieval

    Examples:
        # Start interactive session
        ingestforge interactive ask

        # Without conversation history
        ingestforge interactive ask --no-history

    For help on specific commands:
        ingestforge interactive <command> --help
    """
    pass
