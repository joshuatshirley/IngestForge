"""Index subcommands.

Provides tools for index management:
- list: List all indexes
- info: Show index information
- rebuild: Rebuild indexes
- delete: Delete index

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.index import list as list_cmd, info, rebuild, delete

# Create index subcommand application
app = typer.Typer(
    name="index",
    help="Index management",
    add_completion=False,
)

# Register index commands
app.command("list")(list_cmd.command)
app.command("info")(info.command)
app.command("rebuild")(rebuild.command)
app.command("delete")(delete.command)


@app.callback()
def main() -> None:
    """Index management for IngestForge.

    Manage search indexes for your RAG application:
    - List all indexes
    - View index information
    - Rebuild indexes
    - Delete indexes

    Features:
    - Index size tracking
    - Rebuild from storage
    - Safe deletion with confirmation
    - Multiple index support

    Use cases:
    - View index status
    - Rebuild after embedding changes
    - Clean up unused indexes
    - Monitor index growth

    Examples:
        # List all indexes
        ingestforge index list

        # Show index information
        ingestforge index info documents

        # Rebuild index
        ingestforge index rebuild documents

        # Delete index
        ingestforge index delete old_index

        # Force operations
        ingestforge index rebuild documents --force
        ingestforge index delete old_index --force

    For help on specific commands:
        ingestforge index <command> --help
    """
    pass
