"""Doctor Command - System requirements checker and troubleshooter.

Validates environment setup and provides actionable fix suggestions:
- Python version compatibility
- Required and optional dependencies
- LLM provider configuration
- Storage backend health
- OCR tool availability"""

from __future__ import annotations

import os
import sys
import shutil
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Union, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# =============================================================================
# CHECK RESULT DATACLASS
# =============================================================================


@dataclass
class CheckResult:
    """Result of a diagnostic check.

    Attributes:
        name: Name of the check
        status: 'ok', 'warning', or 'error'
        message: Human-readable status message
        fix: Suggested fix for issues (optional)
        details: Additional details (optional)
    """

    name: str
    status: str  # 'ok', 'warning', 'error'
    message: str
    fix: Optional[str] = None
    details: Optional[str] = None


@dataclass
class DiagnosticTotals:
    """Tracks cumulative diagnostic check totals.

    Rule #4: Extracted to reduce doctor_command below 60 lines.
    """

    ok: int = 0
    warnings: int = 0
    errors: int = 0

    def add(self, ok: int, warn: int, err: int) -> None:
        """Add check results to totals."""
        self.ok += ok
        self.warnings += warn
        self.errors += err


# =============================================================================
# INDIVIDUAL CHECKS
# =============================================================================


def check_python_version() -> CheckResult:
    """Check Python version compatibility.

    Rule #4: Function <60 lines
    """
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major < 3:
        return CheckResult(
            name="Python Version",
            status="error",
            message=f"Python {version_str} - Python 3.10+ required",
            fix="Install Python 3.10 or later from python.org",
        )

    if version.minor < 10:
        return CheckResult(
            name="Python Version",
            status="warning",
            message=f"Python {version_str} - 3.10+ recommended",
            fix="Consider upgrading to Python 3.10+ for best compatibility",
        )

    return CheckResult(
        name="Python Version",
        status="ok",
        message=f"Python {version_str}",
    )


def check_core_dependencies() -> list[CheckResult]:
    """Check required core dependencies.

    Rule #4: Function <60 lines
    """
    results = []

    # Core required packages
    core_packages = [
        ("typer", "typer", "CLI framework"),
        ("rich", "rich", "Terminal formatting"),
        ("chromadb", "chromadb", "Vector storage"),
        ("sentence_transformers", "sentence-transformers", "Embeddings"),
    ]

    for import_name, pip_name, description in core_packages:
        result = _check_package(import_name, pip_name, description, required=True)
        results.append(result)

    return results


def check_optional_dependencies() -> list[CheckResult]:
    """Check optional dependencies.

    Rule #4: Function <60 lines
    """
    results = []

    # Optional packages with features they enable
    optional_packages = [
        ("anthropic", "anthropic", "Claude LLM support"),
        ("openai", "openai", "OpenAI/GPT support"),
        ("google.generativeai", "google-generativeai", "Gemini LLM support"),
        ("ollama", "ollama", "Ollama local LLM support"),
        ("pypdf", "pypdf", "PDF text extraction"),
        ("docx", "python-docx", "Word document support"),
        ("ebooklib", "ebooklib", "EPUB support"),
        ("trafilatura", "trafilatura", "Web content extraction"),
        ("questionary", "questionary", "Interactive menus"),
    ]

    for import_name, pip_name, description in optional_packages:
        result = _check_package(import_name, pip_name, description, required=False)
        results.append(result)

    return results


def _check_package(
    import_name: str, pip_name: str, description: str, required: bool
) -> CheckResult:
    """Check if a Python package is installed.

    Args:
        import_name: Module name for import
        pip_name: Package name for pip install
        description: Human-readable description
        required: Whether this is a required package

    Returns:
        CheckResult with status
    """
    try:
        __import__(import_name)
        return CheckResult(
            name=description,
            status="ok",
            message=f"{pip_name} installed",
        )
    except ImportError:
        status = "error" if required else "warning"
        return CheckResult(
            name=description,
            status=status,
            message=f"{pip_name} not found",
            fix=f"pip install {pip_name}",
        )


def check_ocr_tools() -> list[CheckResult]:
    """Check OCR tool availability.

    Rule #4: Function <60 lines
    """
    results = []

    # Tesseract OCR
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        results.append(
            CheckResult(
                name="Tesseract OCR",
                status="ok",
                message=f"Found at {tesseract_path}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="Tesseract OCR",
                status="warning",
                message="Not found - scanned PDF support limited",
                fix=_get_tesseract_install_command(),
            )
        )

    # EasyOCR (Python package)
    try:
        import easyocr

        results.append(
            CheckResult(
                name="EasyOCR",
                status="ok",
                message="Installed (GPU-accelerated OCR)",
            )
        )
    except ImportError:
        results.append(
            CheckResult(
                name="EasyOCR",
                status="warning",
                message="Not installed",
                fix="pip install easyocr",
            )
        )

    return results


def _get_tesseract_install_command() -> str:
    """Get platform-specific Tesseract install command."""
    system = platform.system().lower()

    if system == "darwin":
        return "brew install tesseract"
    elif system == "linux":
        return "sudo apt install tesseract-ocr"
    elif system == "windows":
        return "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
    else:
        return "Install Tesseract OCR for your platform"


def check_llm_configuration() -> list[CheckResult]:
    """Check LLM provider configuration.

    Rule #4: Function <60 lines
    """
    results = []

    # Check environment variables for API keys
    llm_providers = [
        ("ANTHROPIC_API_KEY", "Claude (Anthropic)"),
        ("OPENAI_API_KEY", "OpenAI/GPT"),
        ("GOOGLE_API_KEY", "Google Gemini"),
    ]

    found_any = False
    for env_var, provider in llm_providers:
        if os.environ.get(env_var):
            found_any = True
            # Mask the key for security
            key = os.environ[env_var]
            masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            results.append(
                CheckResult(
                    name=provider,
                    status="ok",
                    message=f"API key configured ({masked})",
                )
            )
        else:
            results.append(
                CheckResult(
                    name=provider,
                    status="warning",
                    message="API key not set",
                    fix=f"Set {env_var} environment variable",
                )
            )

    # Check for local LLM (llama.cpp)
    llama_result = _check_llamacpp()
    results.append(llama_result)

    # Add overall status
    if not found_any and llama_result.status != "ok":
        results.insert(
            0,
            CheckResult(
                name="LLM Provider",
                status="error",
                message="No LLM configured - AI features unavailable",
                fix="Set an API key or configure local llama.cpp model",
            ),
        )

    return results


def _check_llamacpp() -> CheckResult:
    """Check llama.cpp local LLM setup."""
    # Check for config file
    config_path = Path.cwd() / "ingestforge.yaml"
    if not config_path.exists():
        return CheckResult(
            name="Local LLM (llama.cpp)",
            status="warning",
            message="No config file found",
            fix="Run 'ingestforge init' to create configuration",
        )

    # Check for models directory
    models_dir = Path.cwd() / ".data" / "models"
    if not models_dir.exists():
        return CheckResult(
            name="Local LLM (llama.cpp)",
            status="warning",
            message="No models directory",
            fix="Create .data/models/ and add GGUF model files",
        )

    # Look for GGUF models
    models = list(models_dir.glob("*.gguf"))
    if models:
        return CheckResult(
            name="Local LLM (llama.cpp)",
            status="ok",
            message=f"{len(models)} model(s) found",
            details=", ".join(m.name for m in models[:3]),
        )

    return CheckResult(
        name="Local LLM (llama.cpp)",
        status="warning",
        message="No GGUF models found",
        fix="Download a GGUF model to .data/models/",
    )


def check_storage_health() -> CheckResult:
    """Check storage backend health via StorageFactory.

    Rule #4: Function <60 lines
    """
    from ingestforge.core.config_loaders import load_config
    from ingestforge.storage.factory import StorageFactory

    try:
        config = load_config()
        health = StorageFactory.check_health(config)

        if health["healthy"]:
            return CheckResult(
                name="Storage Backend",
                status="ok",
                message=f"{health['backend'].upper()} healthy",
                details=str(health["details"]),
            )

        return CheckResult(
            name="Storage Backend",
            status="error",
            message=f"{health['backend'].upper()} issue: {health['error']}",
            fix=f"Check {health['backend']} configuration or run 'ingestforge doctor --fix'",
        )
    except Exception as e:
        return CheckResult(
            name="Storage Backend",
            status="warning",
            message="Unable to verify health",
            details=str(e),
            fix="Run 'ingestforge init' to ensure project is set up",
        )


def check_disk_space() -> CheckResult:
    """Check available disk space."""
    try:
        import shutil

        total, used, free = shutil.disk_usage(Path.cwd())
        free_gb = free / (1024**3)

        if free_gb < 1:
            return CheckResult(
                name="Disk Space",
                status="error",
                message=f"Only {free_gb:.1f} GB free",
                fix="Free up disk space - IngestForge needs space for embeddings",
            )

        if free_gb < 5:
            return CheckResult(
                name="Disk Space",
                status="warning",
                message=f"{free_gb:.1f} GB free (5+ GB recommended)",
            )

        return CheckResult(
            name="Disk Space",
            status="ok",
            message=f"{free_gb:.1f} GB free",
        )
    except Exception:
        return CheckResult(
            name="Disk Space",
            status="warning",
            message="Unable to check disk space",
        )


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def display_results(results: list[CheckResult], category: str) -> tuple[int, int, int]:
    """Display check results in a table.

    Args:
        results: List of CheckResults
        category: Category name for the table

    Returns:
        Tuple of (ok_count, warning_count, error_count)
    """
    table = Table(title=category, show_header=True, header_style="bold cyan")
    table.add_column("Check", style="white", width=25)
    table.add_column("Status", width=10)
    table.add_column("Message", style="dim", width=40)

    ok_count = 0
    warning_count = 0
    error_count = 0

    for result in results:
        # Status styling
        if result.status == "ok":
            status = "[green]OK[/green]"
            ok_count += 1
        elif result.status == "warning":
            status = "[yellow]WARN[/yellow]"
            warning_count += 1
        else:
            status = "[red]ERROR[/red]"
            error_count += 1

        table.add_row(result.name, status, result.message)

    console.print(table)
    console.print()

    # Show fixes for non-ok results
    fixes_shown = False
    for result in results:
        if result.fix and result.status != "ok":
            if not fixes_shown:
                console.print("[bold yellow]Suggested fixes:[/bold yellow]")
                fixes_shown = True
            console.print(f"  - {result.name}: [cyan]{result.fix}[/cyan]")

    if fixes_shown:
        console.print()

    return ok_count, warning_count, error_count


def display_summary(total_ok: int, total_warnings: int, total_errors: int) -> None:
    """Display overall summary."""
    total = total_ok + total_warnings + total_errors

    if total_errors > 0:
        status = "[red]Issues Found[/red]"
        border = "red"
        message = f"Fix {total_errors} error(s) to ensure proper functionality."
    elif total_warnings > 0:
        status = "[yellow]Mostly Healthy[/yellow]"
        border = "yellow"
        message = (
            f"Consider addressing {total_warnings} warning(s) for best experience."
        )
    else:
        status = "[green]All Systems Go![/green]"
        border = "green"
        message = "IngestForge is ready to use."

    summary = f"""
Status: {status}

Checks passed: [green]{total_ok}[/green]
Warnings: [yellow]{total_warnings}[/yellow]
Errors: [red]{total_errors}[/red]

{message}
"""

    panel = Panel(
        summary.strip(),
        title="[bold]Diagnostic Summary[/bold]",
        border_style=border,
    )
    console.print(panel)


# =============================================================================
# HELPER FOR MAIN COMMAND
# =============================================================================


def _run_diagnostic_check(
    checker: Callable[[], Union[CheckResult, List[CheckResult]]],
    category: str,
    totals: DiagnosticTotals,
) -> None:
    """Run a diagnostic check and accumulate results.

    Rule #1: Helper to reduce nesting in doctor_command.
    Rule #4: Keeps main function under 60 lines.

    Args:
        checker: Function that returns CheckResult or list of CheckResult
        category: Display category name
        totals: DiagnosticTotals to accumulate into
    """
    result = checker()
    results = result if isinstance(result, list) else [result]
    ok, warn, err = display_results(results, category)
    totals.add(ok, warn, err)


# =============================================================================
# MAIN COMMAND
# =============================================================================


def _run_healing_session(dry_run: bool = False) -> None:
    """
    Run a background healing session to fix stale artifacts.

    Background Healer Worker.
    Rule #4: < 60 lines.

    Args:
        dry_run: If True, only report what would be healed.
    """
    from ingestforge.core.config_loaders import load_config
    from ingestforge.storage.factory import StorageFactory
    from ingestforge.core.maintenance.healer import BackgroundHealer, HealingConfig

    console.print()
    console.print("[bold cyan]IngestForge Artifact Healer[/bold cyan]")
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]HEALING[/green]"
    console.print(f"Mode: {mode}")
    console.print()

    try:
        config = load_config()
        storage = StorageFactory.create(config)

        healing_config = HealingConfig(dry_run=dry_run)
        healer = BackgroundHealer(storage, healing_config)

        console.print("[dim]Scanning for stale artifacts...[/dim]")
        result = healer.run_session()

        # Display results table
        table = Table(title="Healing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Session ID", result.session_id)
        table.add_row("Stale Found", str(result.stale_found))
        table.add_row("Healed", f"[green]{result.healed}[/green]")
        table.add_row("Failed", f"[red]{result.failed}[/red]" if result.failed else "0")
        table.add_row("Skipped", str(result.skipped))
        table.add_row("Success Rate", f"{result.success_rate:.1f}%")

        console.print(table)

        # Show failed items if any
        if result.failed > 0:
            console.print()
            console.print("[bold red]Failed Healings:[/bold red]")
            for r in result.results:
                if not r.success:
                    console.print(f"  - {r.chunk_id}: {r.error_message}")

    except Exception as e:
        console.print(f"[red]Healing failed: {e}[/red]")
        raise typer.Exit(1)


def doctor_command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    fix: bool = typer.Option(
        False, "--fix", help="Attempt automatic fixes where possible"
    ),
    heal: bool = typer.Option(
        False, "--heal", help="Scan and heal stale artifacts in storage"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be healed without making changes"
    ),
) -> None:
    """Run system diagnostics and check configuration.

    Rule #4: Refactored to <60 lines using _run_diagnostic_check helper.

    Validates your environment and provides actionable suggestions
    to fix any issues found.

    Examples:
        ingestforge doctor          # Run diagnostics
        ingestforge doctor -v       # Verbose output
        ingestforge doctor --heal   # Heal stale artifacts
        ingestforge doctor --heal --dry-run  # Preview healing
    """
    # Handle heal mode separately
    if heal or dry_run:
        _run_healing_session(dry_run=dry_run)
        return

    console.print()
    console.print("[bold cyan]IngestForge System Diagnostics[/bold cyan]")
    console.print("[dim]Checking your environment...[/dim]")
    console.print()

    totals = DiagnosticTotals()

    # Run all diagnostic checks
    _run_diagnostic_check(check_python_version, "Python Environment", totals)
    _run_diagnostic_check(check_core_dependencies, "Core Dependencies", totals)

    if verbose:
        _run_diagnostic_check(
            check_optional_dependencies, "Optional Dependencies", totals
        )

    _run_diagnostic_check(check_ocr_tools, "OCR Tools", totals)
    _run_diagnostic_check(check_llm_configuration, "LLM Configuration", totals)
    _run_diagnostic_check(check_storage_health, "Storage", totals)
    _run_diagnostic_check(check_disk_space, "System Resources", totals)

    # Display summary and exit
    display_summary(totals.ok, totals.warnings, totals.errors)

    if totals.errors > 0:
        raise typer.Exit(1)


# Alias for backwards compatibility
command = doctor_command

if __name__ == "__main__":
    typer.run(doctor_command)
