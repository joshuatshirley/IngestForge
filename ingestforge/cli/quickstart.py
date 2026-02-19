"""Quickstart Command - Interactive walkthrough for new users.

Guides users through their first IngestForge workflow:
1. Project initialization
2. Document ingestion
3. First query
4. Study material generation"""

from __future__ import annotations

import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from ingestforge.cli.console import tip

console = Console()

# =============================================================================
# STEP DEFINITIONS
# =============================================================================

WELCOME_MESSAGE = """
# Welcome to IngestForge!

This quickstart will guide you through:
1. Initializing your first project
2. Ingesting a document
3. Querying your knowledge base
4. Generating study materials

Let's get started!
"""

STEP_INIT = """
## Step 1: Initialize Your Project

Every IngestForge project needs a workspace to store:
- Document embeddings
- Search indexes
- Configuration files

We'll create a project folder in your current directory.
"""

STORAGE_PROMPT = """
### Storage Mode

Choose how to store your knowledge base:

**Default (ChromaDB)** - Full-featured vector database
- Best for: Desktop/laptop with 4GB+ RAM
- Features: Hybrid search, semantic similarity, fast queries

**Mobile (JSONL)** - Lightweight file-based storage
- Best for: Low-memory devices, portability
- Features: Works anywhere, no dependencies, smaller footprint
"""

STEP_INGEST = """
## Step 2: Ingest Documents

Now let's add some content to your knowledge base.

You can ingest:
- PDF files
- Word documents (.docx)
- Markdown files
- HTML/web pages
- Plain text files
"""

STEP_QUERY = """
## Step 3: Query Your Knowledge Base

Now you can ask questions about your documents!

IngestForge uses hybrid search (keyword + semantic)
combined with AI to give you accurate, sourced answers.
"""

STEP_COMPLETE = """
## Quickstart Complete!

You've successfully:
- Created an IngestForge project
- Ingested your first document(s)
- Queried the knowledge base

### Next Steps:

- `ingestforge query "your question"` - Ask more questions
- `ingestforge study flashcards "topic"` - Generate flashcards
- `ingestforge study quiz "topic"` - Create a quiz
- `ingestforge export markdown output.md` - Export findings
- `ingestforge --help` - See all commands
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def display_step(content: str) -> None:
    """Display a step with markdown formatting."""
    console.print()
    panel = Panel(
        Markdown(content),
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def run_command_quiet(args: list[str]) -> tuple[int, str]:
    """Run an IngestForge command and capture output.

    Args:
        args: Command arguments

    Returns:
        Tuple of (return_code, output_text)
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "ingestforge.cli.main"] + args,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    return result.returncode, result.stdout + result.stderr


def run_command_interactive(args: list[str]) -> int:
    """Run an IngestForge command interactively.

    Args:
        args: Command arguments

    Returns:
        Return code
    """
    import subprocess

    console.print(f"\n[dim]Running: ingestforge {' '.join(args)}[/dim]\n")
    result = subprocess.run(
        [sys.executable, "-m", "ingestforge.cli.main"] + args,
        cwd=Path.cwd(),
    )
    return result.returncode


def check_project_exists() -> bool:
    """Check if a project already exists."""
    return (Path.cwd() / ".ingestforge").exists()


def get_sample_files() -> list[Path]:
    """Get sample files in current directory."""
    extensions = {".pdf", ".docx", ".md", ".txt", ".html", ".epub"}
    files = []

    for ext in extensions:
        files.extend(Path.cwd().glob(f"*{ext}"))

    return sorted(files)[:10]  # Limit to 10


# =============================================================================
# QUICKSTART STEPS
# =============================================================================


def step_welcome() -> bool:
    """Display welcome and get confirmation to proceed."""
    display_step(WELCOME_MESSAGE)

    return Confirm.ask(
        "\n[cyan]Ready to begin?[/cyan]",
        default=True,
    )


def prompt_storage_mode() -> bool:
    """Prompt user for storage mode preference.

    Returns:
        True for mobile mode, False for default ChromaDB
    """
    display_step(STORAGE_PROMPT)

    choice = Prompt.ask(
        "\n[cyan]Select storage mode[/cyan]",
        choices=["default", "mobile"],
        default="default",
    )

    return choice == "mobile"


def step_init() -> bool:
    """Initialize project step.

    Returns:
        True if successful, False otherwise
    """
    display_step(STEP_INIT)

    # Check if already initialized
    if check_project_exists():
        console.print("[green]Project already initialized![/green]")
        tip("You can have multiple libraries within one project")
        return True

    # Get project name
    project_name = Prompt.ask(
        "\n[cyan]Project name[/cyan]",
        default="my_research",
    )

    if not project_name.strip():
        console.print("[red]Project name cannot be empty[/red]")
        return False

    # Ask about storage mode
    use_mobile = prompt_storage_mode()

    # Build init command
    init_args = ["init", project_name]
    if use_mobile:
        init_args.append("--mobile")
        console.print("\n[dim]Using mobile storage mode (JSONL)[/dim]")
    else:
        console.print("\n[dim]Using default storage (ChromaDB)[/dim]")

    # Run init command
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Initializing project...", total=None)
        code, output = run_command_quiet(init_args)

    if code != 0:
        console.print(f"[red]Failed to initialize project:[/red]\n{output}")
        return False

    console.print("[green]Project initialized successfully![/green]")
    return True


def step_ingest() -> bool:
    """Ingest documents step.

    Returns:
        True if successful, False otherwise
    """
    display_step(STEP_INGEST)

    # Look for sample files
    sample_files = get_sample_files()

    if sample_files:
        console.print("\n[yellow]Found documents in current directory:[/yellow]")
        for i, f in enumerate(sample_files[:5], 1):
            size_kb = f.stat().st_size / 1024
            console.print(f"  {i}. {f.name} ({size_kb:.1f} KB)")

        if len(sample_files) > 5:
            console.print(f"  ... and {len(sample_files) - 5} more")

    # Get path from user
    console.print()
    path_input = Prompt.ask(
        "[cyan]Path to document or folder[/cyan]",
        default=str(sample_files[0]) if sample_files else "",
    )

    if not path_input.strip():
        console.print("[yellow]Skipping document ingestion[/yellow]")
        tip("You can ingest later with: ingestforge ingest <path>")
        return True

    # Validate path
    path = Path(path_input.strip())
    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        return False

    # Build command
    args = ["ingest", str(path)]
    if path.is_dir():
        if Confirm.ask("Include subdirectories?", default=True):
            args.append("--recursive")

    # Run ingest
    console.print()
    code = run_command_interactive(args)

    if code != 0:
        console.print("[yellow]Ingestion had warnings - check output above[/yellow]")
        # Don't fail, continue to query step

    return True


def step_query() -> bool:
    """Query knowledge base step.

    Returns:
        True if successful
    """
    display_step(STEP_QUERY)

    # Get query from user
    query = Prompt.ask(
        "\n[cyan]Ask a question about your documents[/cyan]",
        default="What are the main topics covered?",
    )

    if not query.strip():
        console.print("[yellow]Skipping query[/yellow]")
        return True

    # Run query
    console.print()
    code = run_command_interactive(["query", query.strip()])

    return True  # Always continue


def step_complete() -> None:
    """Display completion message."""
    display_step(STEP_COMPLETE)

    # Offer next actions
    console.print()
    next_action = Prompt.ask(
        "[cyan]What would you like to do next?[/cyan]",
        choices=["query", "flashcards", "quiz", "menu", "exit"],
        default="exit",
    )

    if next_action == "exit":
        console.print("\n[cyan]Happy researching![/cyan]\n")
        return

    if next_action == "menu":
        # Launch interactive menu
        run_command_interactive([])
        return

    if next_action == "query":
        query = Prompt.ask("[cyan]Enter your question[/cyan]")
        if query:
            run_command_interactive(["query", query])
        return

    if next_action in ("flashcards", "quiz"):
        topic = Prompt.ask("[cyan]Enter topic[/cyan]")
        if topic:
            run_command_interactive(["study", next_action, topic])
        return


# =============================================================================
# MAIN COMMAND
# =============================================================================


def quickstart_command(
    skip_init: bool = typer.Option(
        False, "--skip-init", help="Skip project initialization"
    ),
    skip_ingest: bool = typer.Option(
        False, "--skip-ingest", help="Skip document ingestion"
    ),
) -> None:
    """Interactive quickstart wizard for new IngestForge users.

    Walks you through your first complete IngestForge workflow:

    1. **Initialize** - Create a project workspace
    2. **Ingest** - Add documents to your knowledge base
    3. **Query** - Ask questions about your content
    4. **Next Steps** - Explore study tools and exports

    Examples:
        # Full quickstart
        ingestforge quickstart

        # Skip initialization (project exists)
        ingestforge quickstart --skip-init

        # Skip to query (documents already ingested)
        ingestforge quickstart --skip-init --skip-ingest
    """
    try:
        # Welcome
        if not step_welcome():
            console.print("\n[dim]Quickstart cancelled. Run again anytime![/dim]\n")
            raise typer.Exit(0)

        # Step 1: Initialize
        if not skip_init:
            if not step_init():
                raise typer.Exit(1)
        else:
            console.print("[dim]Skipping initialization...[/dim]")

        # Step 2: Ingest
        if not skip_ingest:
            if not step_ingest():
                raise typer.Exit(1)
        else:
            console.print("[dim]Skipping ingestion...[/dim]")

        # Step 3: Query
        step_query()

        # Complete
        step_complete()

    except KeyboardInterrupt:
        console.print("\n\n[dim]Quickstart interrupted. Run again anytime![/dim]\n")
        raise typer.Exit(0)


# Alias for backwards compatibility
command = quickstart_command

if __name__ == "__main__":
    typer.run(quickstart_command)
