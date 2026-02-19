"""Version CLI Commands.

Automated Semantic Versioning
Epic: EP-26 (Security & Compliance)

Provides CLI commands for semantic version management.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ingestforge.core.versioning import (
    BumpType,
    VersionManager,
    create_version_manager,
    parse_version,
)

console = Console()
version_command = typer.Typer(help="Version management commands")


@version_command.command("show")
def show_version(
    project_root: Optional[Path] = typer.Option(
        None,
        "--root",
        "-r",
        help="Project root directory",
    ),
) -> None:
    """Display current version.

    Rule #4: Function < 60 lines.
    """
    manager = _create_manager(project_root)
    current = manager.get_current_version()

    if current:
        _display_version_info(current)
    else:
        console.print("[yellow]No version found[/yellow]")
        raise typer.Exit(1)


@version_command.command("bump")
def bump_version(
    bump_type: str = typer.Argument(
        ...,
        help="Bump type: major, minor, or patch",
    ),
    project_root: Optional[Path] = typer.Option(
        None,
        "--root",
        "-r",
        help="Project root directory",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be done without making changes",
    ),
) -> None:
    """Bump version by specified type.

    Rule #4: Function < 60 lines.
    """
    # Validate bump type
    bump_type_lower = bump_type.lower()
    if bump_type_lower not in ("major", "minor", "patch"):
        console.print(f"[red]Invalid bump type: {bump_type}[/red]")
        console.print("Valid types: major, minor, patch")
        raise typer.Exit(2)

    bump_enum = _get_bump_enum(bump_type_lower)
    manager = _create_manager(project_root)
    current = manager.get_current_version()

    if not current:
        console.print("[yellow]No current version found, starting at 0.1.0[/yellow]")

    if dry_run:
        _display_dry_run(manager, bump_enum)
        return

    new_version = manager.bump_version(bump_enum)
    if new_version:
        console.print(f"[green]Version bumped: {current} -> {new_version}[/green]")
    else:
        console.print("[red]Failed to bump version[/red]")
        raise typer.Exit(1)


@version_command.command("validate")
def validate_version(
    version_string: str = typer.Argument(
        ...,
        help="Version string to validate",
    ),
) -> None:
    """Validate a version string against SemVer 2.0.0.

    Rule #4: Function < 60 lines.
    """
    parsed = parse_version(version_string)

    if parsed:
        console.print(f"[green]Valid SemVer: {parsed}[/green]")
        _display_version_components(parsed)
    else:
        console.print(f"[red]Invalid version string: {version_string}[/red]")
        console.print("Expected format: MAJOR.MINOR.PATCH[-prerelease][+build]")
        raise typer.Exit(1)


@version_command.command("next")
def next_version(
    project_root: Optional[Path] = typer.Option(
        None,
        "--root",
        "-r",
        help="Project root directory",
    ),
    since_tag: Optional[str] = typer.Option(
        None,
        "--since",
        "-s",
        help="Analyze commits since this tag",
    ),
) -> None:
    """Preview next version based on commit analysis.

    Rule #4: Function < 60 lines.
    """
    manager = _create_manager(project_root)
    current = manager.get_current_version()

    if since_tag:
        bump_type = manager.analyze_commits(since_tag)
    else:
        bump_type = manager.analyze_commits()

    if bump_type == BumpType.NONE:
        console.print(
            "[yellow]No conventional commits found, defaulting to patch[/yellow]"
        )
        bump_type = BumpType.PATCH

    if current:
        next_ver = current.bump(bump_type)
        _display_next_version_info(current, next_ver, bump_type)
    else:
        console.print("[yellow]No current version found[/yellow]")
        raise typer.Exit(1)


# Helper functions (JPL Rule #4: Keep functions small)


def _create_manager(project_root: Optional[Path]) -> VersionManager:
    """Create version manager instance.

    Args:
        project_root: Optional project root path.

    Returns:
        Configured VersionManager.
    """
    if project_root:
        return create_version_manager(project_root=project_root)
    return create_version_manager()


def _get_bump_enum(bump_type: str) -> BumpType:
    """Convert string to BumpType enum.

    Args:
        bump_type: Lowercase bump type string.

    Returns:
        BumpType enum value.
    """
    mapping = {
        "major": BumpType.MAJOR,
        "minor": BumpType.MINOR,
        "patch": BumpType.PATCH,
    }
    return mapping.get(bump_type, BumpType.PATCH)


def _display_version_info(version: "SemanticVersion") -> None:
    """Display version information in a table.

    Args:
        version: SemanticVersion to display.
    """

    table = Table(title="Current Version")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", str(version))
    table.add_row("Major", str(version.major))
    table.add_row("Minor", str(version.minor))
    table.add_row("Patch", str(version.patch))

    if version.prerelease:
        table.add_row("Prerelease", version.prerelease)
    if version.build_metadata:
        table.add_row("Build", version.build_metadata)

    console.print(table)


def _display_version_components(version: "SemanticVersion") -> None:
    """Display version components.

    Args:
        version: SemanticVersion to display.
    """

    table = Table(title="Version Components")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Major", str(version.major))
    table.add_row("Minor", str(version.minor))
    table.add_row("Patch", str(version.patch))

    if version.prerelease:
        table.add_row("Prerelease", version.prerelease)
    if version.build_metadata:
        table.add_row("Build Metadata", version.build_metadata)

    console.print(table)


def _display_dry_run(manager: VersionManager, bump_type: BumpType) -> None:
    """Display dry run information.

    Args:
        manager: VersionManager instance.
        bump_type: Type of bump to preview.
    """
    current = manager.get_current_version()
    if current:
        new_version = current.bump(bump_type)
        console.print(f"[cyan]Dry run:[/cyan] {current} -> {new_version}")
    else:
        from ingestforge.core.versioning import SemanticVersion

        default = SemanticVersion(0, 1, 0)
        new_version = default.bump(bump_type)
        console.print(f"[cyan]Dry run:[/cyan] (none) -> {new_version}")


def _display_next_version_info(
    current: "SemanticVersion",
    next_ver: "SemanticVersion",
    bump_type: BumpType,
) -> None:
    """Display next version information.

    Args:
        current: Current version.
        next_ver: Next version.
        bump_type: Detected bump type.
    """
    table = Table(title="Next Version Preview")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Current Version", str(current))
    table.add_row("Recommended Bump", bump_type.value.upper())
    table.add_row("Next Version", str(next_ver))

    console.print(table)
