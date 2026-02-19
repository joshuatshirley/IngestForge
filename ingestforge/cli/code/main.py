"""Code analysis subcommands.

Provides tools for analyzing code and generating documentation:
- analyze: Analyze code structure and patterns
- map: Generate code maps and dependency graphs
- explain: Explain code from files
- document: Auto-generate code documentation

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.code import analyze, map, explain, document

# Create code subcommand application
app = typer.Typer(
    name="code",
    help="Code analysis and documentation tools",
    add_completion=False,
)

# Register code analysis commands
app.command("analyze")(analyze.command)
app.command("map")(map.command)
app.command("explain")(explain.command)
app.command("document")(document.command)


@app.callback()
def main() -> None:
    """Code analysis tools for IngestForge.

    Analyze code structure, identify patterns, and generate
    documentation from your codebase.

    Features:
    - Structural analysis
    - Pattern identification
    - Code quality insights
    - Visual code maps
    - Multiple output formats

    Use cases:
    - Technical documentation
    - Code reviews
    - Onboarding materials
    - Architecture diagrams

    Examples:
        # Analyze code structure
        ingestforge code analyze src/main.py

        # Generate code map
        ingestforge code map src/ --format markdown

        # Focus on specific patterns
        ingestforge code analyze app/ --pattern "error handling"

        # Export visualizations
        ingestforge code map project/ --format mermaid -o diagram.md

    For help on specific commands:
        ingestforge code <command> --help
    """
    pass
