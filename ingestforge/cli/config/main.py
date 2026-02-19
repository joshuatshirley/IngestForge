"""Config subcommands.

Provides tools for configuration management:
- show: Display current configuration
- set: Set configuration values
- reset: Reset to defaults
- validate: Validate configuration

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.config import show, set, reset, validate, list as list_cmd

# Create config subcommand application
app = typer.Typer(
    name="config",
    help="Configuration management",
    add_completion=False,
)

# Register config commands
app.command("show")(show.command)
app.command("set")(set.command)
app.command("reset")(reset.command)
app.command("validate")(validate.command)
app.command("list")(list_cmd.command)


@app.callback()
def main() -> None:
    """Configuration management for IngestForge.

    Manage application configuration settings including:
    - LLM provider and model selection
    - Embedding model configuration
    - Storage backend settings
    - Chunking strategies
    - Retrieval parameters
    - Processing options

    Features:
    - View current configuration
    - Set individual values
    - Reset to defaults
    - Validate configuration
    - Auto-fix common issues

    Configuration sections:
    - llm: Language model settings
    - embedding: Embedding model settings
    - storage: Storage backend configuration
    - chunking: Document chunking settings
    - retrieval: Retrieval parameters
    - processing: Processing options

    Examples:
        # Show configuration summary
        ingestforge config show

        # Show full configuration
        ingestforge config show --format full

        # Set LLM model
        ingestforge config set llm.model gpt-4

        # Set chunk size
        ingestforge config set chunking.size 1500

        # Reset configuration
        ingestforge config reset

        # Validate configuration
        ingestforge config validate

        # Fix configuration issues
        ingestforge config validate --fix

    For help on specific commands:
        ingestforge config <command> --help
    """
    pass
