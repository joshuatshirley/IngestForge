"""Argument building command group - Critical thinking tools."""

import typer

from ingestforge.cli.argument import debate, support, counter, conflicts, gaps

app = typer.Typer(
    name="argument",
    help="Build and analyze arguments from your knowledge base",
    no_args_is_help=True,
)

# Register subcommands
app.command(name="debate")(debate.command)
app.command(name="support")(support.command)
app.command(name="counter")(counter.command)
app.command(name="conflicts")(conflicts.command)
app.command(name="gaps")(gaps.command)
