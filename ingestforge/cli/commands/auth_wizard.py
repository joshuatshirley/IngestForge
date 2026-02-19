"""API Key Setup Wizard.

Provides interactive setup for LLM provider API keys.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

# Key patterns for validation
KEY_PATTERNS = {
    "ANTHROPIC_API_KEY": ("sk-ant-", "Anthropic/Claude"),
    "OPENAI_API_KEY": ("sk-", "OpenAI/GPT"),
    "GOOGLE_API_KEY": ("", "Google Gemini"),  # No specific prefix
}


def mask_key(key: str) -> str:
    """Mask API key for display."""
    if len(key) < 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def validate_key(key: str, key_type: str) -> bool:
    """Basic validation of API key format."""
    if not key or len(key) < 10:
        return False
    prefix, _ = KEY_PATTERNS.get(key_type, ("", "Unknown"))
    if prefix and not key.startswith(prefix):
        return False
    return True


def save_to_env(project_dir: Path, key_name: str, key_value: str) -> None:
    """Save API key to .env file."""
    env_file = project_dir / ".env"

    # Read existing content
    existing_lines = []
    if env_file.exists():
        with open(env_file, "r") as f:
            existing_lines = [
                l for l in f.readlines() if not l.startswith(f"{key_name}=")
            ]

    # Add new key
    existing_lines.append(f"{key_name}={key_value}\n")

    # Write back
    with open(env_file, "w") as f:
        f.writelines(existing_lines)


def _setup_single_provider(project_dir: Path, key_name: str) -> Optional[str]:
    """Setup a single provider's API key.

    JPL-003: Extracted to reduce nesting in main function.

    Args:
        project_dir: Project directory path
        key_name: Environment variable name for the key

    Returns:
        Provider name if configured, None if skipped
    """
    _, provider_name = KEY_PATTERNS[key_name]
    console.print(f"\n[bold]{provider_name}[/bold]")

    key_value = Prompt.ask(f"Enter {key_name}", password=True)

    # Early return: empty key (JPL-003: Guard clause)
    if not key_value:
        console.print("[yellow]Skipped[/yellow]")
        return None

    # Early return: invalid key without confirmation (JPL-003: Guard clause)
    if not validate_key(key_value, key_name):
        console.print("[red]Invalid key format[/red]")
        if not Confirm.ask("Save anyway?", default=False):
            return None

    save_to_env(project_dir, key_name, key_value)
    console.print(f"[green]✓ Saved {mask_key(key_value)}[/green]")
    return provider_name


def _get_providers_to_setup(provider_choice: str) -> list[str]:
    """Get list of provider keys to setup based on choice.

    JPL-003: Extracted to reduce function size.

    Args:
        provider_choice: User's choice (claude, openai, gemini, or all)

    Returns:
        List of provider key names to setup
    """
    # Dictionary dispatch pattern (JPL-003)
    choice_map = {
        "all": list(KEY_PATTERNS.keys()),
        "claude": ["ANTHROPIC_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "gemini": ["GOOGLE_API_KEY"],
    }
    return choice_map.get(provider_choice, [])


def _display_welcome_banner() -> None:
    """Display welcome banner for auth wizard.

    JPL-003: Extracted to reduce function size.
    """
    console.print(
        Panel(
            "[bold]IngestForge API Key Setup Wizard[/bold]\n\n"
            "This wizard helps you configure API keys for LLM providers.",
            title="🔑 Setup",
            border_style="blue",
        )
    )


def _display_available_providers() -> None:
    """Display list of available providers.

    JPL-003: Extracted to reduce function size.
    """
    console.print("\n[bold]Available Providers:[/bold]")
    for key_name, (_, provider) in KEY_PATTERNS.items():
        console.print(f"  • {provider}")


def auth_wizard_command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Interactive API key setup wizard.

    JPL-003: Refactored to <60 lines via helper functions.

    Guides you through setting up LLM provider API keys.
    Keys are saved to .env file in your project directory.

    Examples:
        ingestforge auth-wizard
        ingestforge auth-wizard -p /path/to/project
    """
    project_dir = project or Path.cwd()

    _display_welcome_banner()
    _display_available_providers()

    # Ask which provider
    provider_choice = Prompt.ask(
        "\nWhich provider would you like to configure?",
        choices=["claude", "openai", "gemini", "all"],
        default="claude",
    )

    providers_to_setup = _get_providers_to_setup(provider_choice)

    configured = []
    for key_name in providers_to_setup:
        result = _setup_single_provider(project_dir, key_name)
        if result:
            configured.append(result)

    # Summary
    if configured:
        console.print(
            Panel(
                f"[green]Configured: {', '.join(configured)}[/green]\n\n"
                f"Keys saved to: {project_dir / '.env'}",
                title="✓ Complete",
                border_style="green",
            )
        )
    else:
        console.print("[yellow]No keys configured[/yellow]")


# Export for registration
command = auth_wizard_command
