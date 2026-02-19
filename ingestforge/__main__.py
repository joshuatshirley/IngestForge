"""Package entry point.

Allows running the CLI as: python -m ingestforge

This module handles the entry point cleanly.
"""

from ingestforge.cli.main import cli_main

if __name__ == "__main__":
    cli_main()
