"""CLI package entry point.

Allows running the CLI as: python -m ingestforge.cli

This module handles the entry point cleanly to avoid RuntimeWarning
about module already being in sys.modules.
"""

from ingestforge.cli.main import cli_main

if __name__ == "__main__":
    cli_main()
