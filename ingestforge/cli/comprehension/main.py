"""Comprehension command group - Understand and explain concepts."""

import typer

from ingestforge.cli.comprehension import explain, compare, connect

app = typer.Typer(
    name="comprehension",
    help="Understand and explain concepts from your knowledge base",
    no_args_is_help=True,
)

# Register subcommands
app.command(name="explain")(explain.command)
app.command(name="compare")(compare.command)
app.command(name="connect")(connect.command)
