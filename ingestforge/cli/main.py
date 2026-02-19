"""IngestForge CLI - Main application entry point.

This is the main CLI application that registers all commands.

BEFORE: 4,996 lines with 50+ inline command implementations
AFTER: ~150 lines with clean command registration

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Any
from functools import wraps
import typer

from ingestforge.core.logging import configure_logging
from ingestforge.core.errors import ErrorContext
from ingestforge.core.error_formatter import (
    format_user_error,
    get_error_code_from_exception,
)

from ingestforge.cli.commands import (
    status_command,
    init_command,
    query_command,
    ingest_command,
    preview_command,
    mark_command,
    tag_command,
    tags_command,
    bookmark_command,
    bookmarks_command,
    annotate_command,
    annotations_app,
    link_command,
    links_command,
    unlink_command,
    auth_wizard_command,
    reset_command,
    api_app,
    demo_command,
)
from ingestforge.cli.commands.agent import agent_command
from ingestforge.cli.commands.analysis_mgmt import app as analysis_app
from ingestforge.cli.nexus.main import app as nexus_app
from ingestforge.cli.commands.sync import app as sync_app
from ingestforge.cli.quickstart import quickstart_command
from ingestforge.cli.doctor import doctor_command
from ingestforge.cli.setup_wizard import setup_command
from ingestforge.cli.literary import literary_app
from ingestforge.cli.research import research_app
from ingestforge.cli.study import study_app
from ingestforge.cli.comprehension import comprehension_app
from ingestforge.cli.argument import argument_app
from ingestforge.cli.writing import writing_app
from ingestforge.cli.discovery import discovery_app
from ingestforge.cli.interactive import get_interactive_app
from ingestforge.cli.export import export_app
from ingestforge.cli.code import code_app
from ingestforge.cli.citation import citation_app
from ingestforge.cli.analyze import analyze_app
from ingestforge.cli.workflow import workflow_app
from ingestforge.cli.transform import transform_app
from ingestforge.cli.config import config_app
from ingestforge.cli.maintenance import maintenance_app
from ingestforge.cli.monitor import monitor_app
from ingestforge.cli.index import index_app
from ingestforge.cli.storage import storage_app
from ingestforge.cli.audit import audit_command as audit_app
from ingestforge.cli.security import security_command as security_app
from ingestforge.cli.lint import lint_command as lint_app
from ingestforge.cli.version import version_command as version_app
from ingestforge.cli.registry import registry_command as registry_app


def _handle_cli_error(e: Exception, operation_name: str, show_debug: bool) -> None:
    """
    Handle CLI error with user-friendly formatting.

    JPL Rule #4: <40 lines.
    JPL Rule #7: No return value (side effects only).
    JPL Rule #9: Full type hints.

    Args:
        e: The exception that occurred
        operation_name: Human-readable operation name
        show_debug: Whether to show technical details
    """
    # Try to infer error code
    error_code = get_error_code_from_exception(e, operation_name)

    if error_code:
        # Format user-friendly error
        context: ErrorContext = {
            "operation": operation_name,
            "details": str(e) if show_debug else None,
        }
        error_msg = format_user_error(error_code, context, show_debug)
        typer.echo(error_msg, err=True)
    else:
        # Fallback to generic error
        typer.echo(f"✗ Error during {operation_name}: {type(e).__name__}", err=True)
        if show_debug:
            typer.echo(f"  Details: {e}", err=True)
            import traceback

            traceback.print_exc()
        else:
            typer.echo("  Run with --debug for more information", err=True)

    # Log the error
    logger = logging.getLogger(__name__)
    logger.error(f"[{operation_name}] {type(e).__name__}: {e}", exc_info=True)


def safe_cli_command(
    operation_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to wrap CLI commands with user-friendly error handling.

    JPL Rule #4: <50 lines.
    JPL Rule #7: Returns explicit callable.
    JPL Rule #9: Full type hints.

    Args:
        operation_name: Human-readable operation name for error context

    Returns:
        Decorator function that wraps the command with error handling
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                show_debug = kwargs.get("debug", False)
                _handle_cli_error(e, operation_name, show_debug)
                raise typer.Exit(code=1)

        return wrapper

    return decorator


# Create main Typer application
app = typer.Typer(
    name="ingestforge",
    help="Document processing and RAG framework for research",
    add_completion=True,
    pretty_exceptions_enable=True,
)


def _setup_file_logging() -> None:
    """Configure file-based logging for error tracking.

    Rule #4: Function <60 lines
    Rule #7: Check parameters

    Creates logs directory and configures file handler.
    """
    # Find project directory (.ingestforge)
    cwd = Path.cwd()
    ingestforge_dir = cwd / ".ingestforge"

    # Only enable file logging if project is initialized
    if not ingestforge_dir.exists():
        return

    # Create logs directory
    logs_dir = ingestforge_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure with file output
    log_file = logs_dir / "ingestforge.log"
    configure_logging(
        level="INFO",
        log_file=log_file,
        console=True,
    )


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
) -> None:
    """IngestForge - Document processing and RAG framework."""
    # Setup file logging on every invocation
    _setup_file_logging()

    if version:
        from ingestforge import __version__

        typer.echo(f"IngestForge {__version__}")
        raise typer.Exit()

    # If no command provided, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# Register core commands
# UX-010: Organize commands into help panels for better discoverability
app.command("status", rich_help_panel="Core")(status_command)
app.command("init", rich_help_panel="Core")(init_command)
app.command("query", rich_help_panel="Core")(query_command)
app.command("ingest", rich_help_panel="Core")(ingest_command)
app.command("preview", rich_help_panel="Core")(preview_command)
app.command("quickstart", rich_help_panel="System")(quickstart_command)
app.command("setup", rich_help_panel="System")(setup_command)
app.command("demo", rich_help_panel="System")(demo_command)
app.command("doctor", rich_help_panel="System")(doctor_command)
app.command("auth-wizard", rich_help_panel="System")(auth_wizard_command)
app.command("mark", rich_help_panel="Organization")(mark_command)
app.command("tag", rich_help_panel="Organization")(tag_command)
app.command("tags", rich_help_panel="Organization")(tags_command)
app.command("bookmark", rich_help_panel="Organization")(bookmark_command)
app.command("bookmarks", rich_help_panel="Organization")(bookmarks_command)
app.command("annotate", rich_help_panel="Organization")(annotate_command)
app.command("link", rich_help_panel="Organization")(link_command)
app.command("links", rich_help_panel="Organization")(links_command)
app.command("unlink", rich_help_panel="Organization")(unlink_command)
app.command("reset", rich_help_panel="System")(reset_command)
app.add_typer(api_app, name="api", rich_help_panel="System")
app.add_typer(nexus_app, name="nexus", rich_help_panel="Security")


# Register click-based agent command group
# Agent uses Click for complex option handling - wrap and delegate
@app.command(
    "agent",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    rich_help_panel="Research",
)
def agent_command_wrapper(
    ctx: typer.Context,
    help_flag: bool = typer.Option(
        False, "--help", "-h", is_eager=True, help="Show help"
    ),
) -> None:
    """Autonomous agent for research tasks (ReAct reasoning).

    Subcommands:
        run      - Run autonomous research agent with LLM
        tools    - List available agent tools
        status   - Show agent status and capabilities
        test-llm - Test LLM connectivity

    Examples:
        ingestforge agent run "Research topic X"
        ingestforge agent tools
        ingestforge agent status
    """
    from ingestforge.cli.commands.agent import agent_command as click_agent

    # Build args list - include the subcommand and all extra args
    args = ctx.args if ctx.args else ["--help"]

    # Invoke the click group
    try:
        click_agent.main(args=args, standalone_mode=False)
    except SystemExit as e:
        if e.code != 0:
            raise typer.Exit(e.code)


# GUI command
@app.command("gui", rich_help_panel="System")
def gui_command() -> None:
    """Launch the IngestForge graphical user interface.

    Opens a professional desktop GUI with:
    - Document import with drag-and-drop
    - Real-time processing visualization
    - Query interface with AI answers
    - Literary analysis tools
    - LLM configuration

    Examples:
        ingestforge gui
    """
    from ingestforge.cli.interactive.gui_menu import main as gui_main

    gui_main()


# Register subcommand groups
# UX-010: Organize subcommands into help panels
app.add_typer(literary_app, name="lit", rich_help_panel="Study")
app.add_typer(research_app, name="research", rich_help_panel="Research")
app.add_typer(study_app, name="study", rich_help_panel="Study")
app.add_typer(comprehension_app, name="comprehension", rich_help_panel="Study")
app.add_typer(argument_app, name="argument", rich_help_panel="Study")
app.add_typer(writing_app, name="writing", rich_help_panel="Analysis")
app.add_typer(discovery_app, name="discovery", rich_help_panel="Research")
app.add_typer(get_interactive_app(), name="interactive", rich_help_panel="System")
app.add_typer(export_app, name="export", rich_help_panel="Data")
app.add_typer(code_app, name="code", rich_help_panel="Analysis")
app.add_typer(citation_app, name="citation", rich_help_panel="Research")
app.add_typer(analyze_app, name="analyze", rich_help_panel="Analysis")
app.add_typer(workflow_app, name="workflow", rich_help_panel="System")
app.add_typer(transform_app, name="transform", rich_help_panel="Data")
app.add_typer(config_app, name="config", rich_help_panel="System")
app.add_typer(maintenance_app, name="maintenance", rich_help_panel="System")
app.add_typer(monitor_app, name="monitor", rich_help_panel="System")
app.add_typer(index_app, name="index", rich_help_panel="Data")
app.add_typer(storage_app, name="storage", rich_help_panel="Data")
app.add_typer(sync_app, name="sync", rich_help_panel="Data")
app.add_typer(analysis_app, name="analysis", rich_help_panel="Data")
app.add_typer(audit_app, name="audit", rich_help_panel="System")
app.add_typer(security_app, name="security", rich_help_panel="System")
app.add_typer(lint_app, name="lint", rich_help_panel="System")
app.add_typer(version_app, name="version", rich_help_panel="System")
app.add_typer(registry_app, name="registry", rich_help_panel="System")
app.add_typer(annotations_app, name="annotations", rich_help_panel="Organization")

# Add Click-based agent command group
# Note: agent uses Click directly for complex option handling
import typer.core

typer.core.rich = None  # Suppress rich for this command
_click_app = typer.main.get_command(app)
_click_app.add_command(agent_command, name="agent")


# Version callback (Commandment #4: Small function)
def version_callback(value: bool) -> None:
    """Show version and exit.

    Args:
        value: True if --version flag provided
    """
    if value:
        from ingestforge import __version__

        typer.echo(f"IngestForge version {__version__}")
        raise typer.Exit()


def verbose_callback(value: bool) -> None:
    """Enable verbose/debug output.

    UX-004: Also enables full traceback display for errors.

    Args:
        value: True if --verbose flag provided
    """
    if value:
        # Set root logger to DEBUG level
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # Also set ingestforge-specific loggers
        logging.getLogger("ingestforge").setLevel(logging.DEBUG)

        # UX-004: Enable verbose mode for error display (show tracebacks)
        from ingestforge.cli.console import set_verbose_mode

        set_verbose_mode(True)


def check_environment() -> List[Tuple[str, bool, str]]:
    """Check for required external tools and dependencies.

    Returns:
        List of tuples: (tool_name, is_available, message)
    """
    checks: List[Tuple[str, bool, str]] = []

    # Check for pdftotext (from poppler)
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        checks.append(("pdftotext", True, f"Found at {pdftotext}"))
    else:
        checks.append(
            ("pdftotext", False, "Not found - PDF text extraction may be limited")
        )

    # Check for OCR engines (tesseract or easyocr)
    tesseract = shutil.which("tesseract")
    easyocr_available = _is_easyocr_installed()

    if tesseract:
        checks.append(("ocr", True, f"Tesseract at {tesseract}"))
    elif easyocr_available:
        checks.append(("ocr", True, "EasyOCR available (pure Python)"))
    else:
        checks.append(
            ("ocr", False, "No OCR engine - install tesseract or: pip install easyocr")
        )

    return checks


def _is_easyocr_installed() -> bool:
    """Check if EasyOCR is installed without importing it."""
    import importlib.util

    return importlib.util.find_spec("easyocr") is not None


def print_environment_warnings(verbose: bool = False) -> None:
    """Print warnings for missing optional tools.

    Args:
        verbose: If True, print all checks including successful ones
    """
    checks = check_environment()

    for tool_name, is_available, message in checks:
        if not is_available:
            typer.echo(f"[Warning] {tool_name}: {message}", err=True)
        elif verbose:
            typer.echo(f"[OK] {tool_name}: {message}")


def offline_callback(value: bool) -> None:
    """Enable offline/air-gap mode.

    SEC-002: Blocks all network access except whitelisted connections.
    Useful for processing sensitive documents without network leakage.

    Args:
        value: True if --offline flag provided
    """
    if value:
        from ingestforge.core.security.network_lock import enable_offline_mode

        enable_offline_mode(allow_localhost=True)
        typer.echo("[Offline Mode] Network access restricted")


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        callback=verbose_callback,
        is_eager=True,
        help="Enable verbose/debug output for all commands",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        callback=offline_callback,
        is_eager=True,
        help="Enable air-gap mode (block network access)",
    ),
    no_remind: bool = typer.Option(
        False,
        "--no-remind",
        help="Suppress due cards reminder at startup",
    ),
) -> None:
    """IngestForge - Document processing and RAG framework.

    A mission-critical Python framework for processing documents,
    generating embeddings, and building RAG applications.

    Core Commands:
        quickstart - Interactive walkthrough for new users
        init       - Initialize new project
        ingest     - Process documents
        query      - Search and answer questions
        status     - Show project status
        doctor     - Check system requirements

    Examples:
        # New users - start here!
        ingestforge quickstart

        # Check your setup
        ingestforge doctor

        # Initialize project
        ingestforge init my_project

        # Process documents
        ingestforge ingest documents/ --recursive

        # Query knowledge base
        ingestforge query "What is IngestForge?"

        # Run with verbose output
        ingestforge --verbose ingest documents/

        # Suppress due cards reminder
        ingestforge --no-remind status

        # Run in offline/air-gap mode
        ingestforge --offline ingest documents/

    For help on a specific command:
        ingestforge <command> --help
    """
    # Check environment and warn about missing optional tools
    print_environment_warnings(verbose=verbose)

    # SRS-004: Show due cards notification if applicable
    # Only show if a subcommand is being invoked
    if ctx.invoked_subcommand is not None and not no_remind:
        from ingestforge.study.due_check import show_due_notification

        show_due_notification()


def _set_process_title() -> None:
    """Set process title for easier identification in task managers.

    Uses setproctitle if available, otherwise silently skips.
    On Windows, this helps identify IngestForge in Task Manager.
    """
    try:
        import setproctitle

        setproctitle.setproctitle("IngestForge")
    except ImportError:
        pass  # Optional dependency - skip if not installed


def cli_main() -> None:
    """Entry point for console_scripts.

    This function is called when running 'ingestforge' command.
    """
    _set_process_title()
    app()


if __name__ == "__main__":
    cli_main()
