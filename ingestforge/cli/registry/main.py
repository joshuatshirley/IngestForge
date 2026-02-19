"""Registry CLI Commands.

Registry-Driven Discovery.
Provides CLI commands to list, inspect, and manage registered processors.

NASA JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds
- Rule #4: Functions < 60 lines
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
console = Console()

# JPL Rule #2: Fixed upper bounds
MAX_DISPLAY_PROCESSORS = 256
MAX_CAPABILITIES_PER_LINE = 5

registry_command = typer.Typer(
    name="registry",
    help="Manage processor registry ().",
    no_args_is_help=True,
)


def _get_registry() -> Any:
    """
    Get the IFRegistry singleton.

    Rule #7: Returns registry or raises.
    """
    from ingestforge.core.pipeline.registry import IFRegistry

    return IFRegistry()


def _format_capabilities(
    caps: List[str], max_per_line: int = MAX_CAPABILITIES_PER_LINE
) -> str:
    """
    Format capabilities list for display.

    Rule #4: Function < 60 lines.
    """
    if not caps:
        return "-"
    if len(caps) <= max_per_line:
        return ", ".join(caps)
    # Truncate with ellipsis
    return ", ".join(caps[:max_per_line]) + f" (+{len(caps) - max_per_line})"


@registry_command.command("list")
def list_processors(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    capability: Optional[str] = typer.Option(
        None, "--capability", "-c", help="Filter by capability"
    ),
    mime_type: Optional[str] = typer.Option(
        None, "--mime", "-m", help="Filter by MIME type"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
) -> None:
    """
    List all registered processors.

    AC#4: CLI command to list discovered processors.
    Rule #4: Function < 60 lines.
    """
    registry = _get_registry()

    processors_data: List[Dict[str, Any]] = []

    # Collect processor data
    for proc_id, proc in list(registry._id_map.items())[:MAX_DISPLAY_PROCESSORS]:
        # Apply filters
        if capability and capability not in proc.capabilities:
            continue

        entry = {
            "processor_id": proc.processor_id,
            "version": proc.version,
            "capabilities": list(proc.capabilities),
            "memory_mb": proc.memory_mb,
            "available": proc.is_available(),
        }

        if verbose:
            entry["class"] = type(proc).__name__
            entry["module"] = type(proc).__module__

        processors_data.append(entry)

    # JSON output
    if json_output:
        output = {
            "processors": processors_data,
            "total": len(processors_data),
        }
        console.print(json.dumps(output, indent=2))
        return

    # Table output
    if not processors_data:
        console.print("[yellow]No processors registered[/yellow]")
        return

    table = Table(title="Registered Processors")
    table.add_column("ID", style="cyan")
    table.add_column("Version")
    table.add_column("Capabilities")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Available", justify="center")

    if verbose:
        table.add_column("Class")

    for proc in processors_data:
        row = [
            proc["processor_id"],
            proc["version"],
            _format_capabilities(proc["capabilities"]),
            str(proc["memory_mb"]),
            "[green]Yes[/green]" if proc["available"] else "[red]No[/red]",
        ]
        if verbose:
            row.append(proc.get("class", "-"))
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Total: {len(processors_data)} processors[/dim]")


@registry_command.command("capabilities")
def list_capabilities(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """
    List all registered capabilities.

    Shows capability-to-processor mapping.
    Rule #4: Function < 60 lines.
    """
    registry = _get_registry()

    # Collect capabilities from processors
    cap_map: Dict[str, List[str]] = {}
    for proc_id, proc in registry._id_map.items():
        for cap in proc.capabilities:
            if cap not in cap_map:
                cap_map[cap] = []
            cap_map[cap].append(proc_id)

    # Add enricher capabilities
    for cap in registry.get_all_enricher_capabilities():
        if cap not in cap_map:
            cap_map[cap] = []
        # Get enricher class names
        cls_names = registry._enricher_capability_index.get(cap, [])
        for cls_name in cls_names:
            if cls_name not in cap_map[cap]:
                cap_map[cap].append(f"[enricher] {cls_name}")

    if json_output:
        console.print(json.dumps(cap_map, indent=2))
        return

    if not cap_map:
        console.print("[yellow]No capabilities registered[/yellow]")
        return

    table = Table(title="Registered Capabilities")
    table.add_column("Capability", style="cyan")
    table.add_column("Processors")
    table.add_column("Count", justify="right")

    for cap in sorted(cap_map.keys()):
        procs = cap_map[cap]
        table.add_row(
            cap,
            ", ".join(procs[:3]) + (f" (+{len(procs) - 3})" if len(procs) > 3 else ""),
            str(len(procs)),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(cap_map)} capabilities[/dim]")


@registry_command.command("enrichers")
def list_enrichers(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """
    List all registered enrichers.

    Shows enricher factories registered via @register_enricher.
    Rule #4: Function < 60 lines.
    """
    registry = _get_registry()

    enrichers_data: List[Dict[str, Any]] = []

    for cls_name, entry in registry._enricher_factories.items():
        enrichers_data.append(
            {
                "class_name": cls_name,
                "capabilities": entry.capabilities,
                "priority": entry.priority,
            }
        )

    # Sort by priority (descending)
    enrichers_data.sort(key=lambda e: -e["priority"])

    if json_output:
        output = {
            "enrichers": enrichers_data,
            "total": len(enrichers_data),
        }
        console.print(json.dumps(output, indent=2))
        return

    if not enrichers_data:
        console.print("[yellow]No enrichers registered[/yellow]")
        return

    table = Table(title="Registered Enrichers")
    table.add_column("Class", style="cyan")
    table.add_column("Capabilities")
    table.add_column("Priority", justify="right")

    for enricher in enrichers_data:
        table.add_row(
            enricher["class_name"],
            ", ".join(enricher["capabilities"]),
            str(enricher["priority"]),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(enrichers_data)} enrichers[/dim]")


@registry_command.command("health")
def check_health() -> None:
    """
    Check registry health status.

    Verifies registry is properly initialized.
    """
    registry = _get_registry()

    is_healthy = registry.is_healthy()
    proc_count = len(registry._id_map)
    enricher_count = len(registry._enricher_factories)
    cap_count = len(registry._capability_index)

    if is_healthy:
        console.print("[green]Registry is healthy[/green]")
    else:
        console.print("[red]Registry is NOT healthy[/red]")
        sys.exit(1)

    console.print(f"  Processors: {proc_count}")
    console.print(f"  Enrichers: {enricher_count}")
    console.print(f"  Capabilities: {cap_count}")
    console.print(f"  Process ID: {registry.get_process_id()}")


@registry_command.command("discover")
def trigger_discovery(
    plugin_dir: Optional[str] = typer.Option(
        None, "--plugin-dir", "-p", help="Additional plugin directory"
    ),
) -> None:
    """
    Manually trigger processor discovery.

    AC#3: Plugin directory scanning.
    Rule #4: Function < 60 lines.
    """
    from ingestforge.core.pipeline.registry import discover_plugins

    count_before = len(_get_registry()._id_map)

    # Discover from plugin directory if specified
    if plugin_dir:
        discovered = discover_plugins(plugin_dir)
        console.print(
            f"[green]Discovered {discovered} processors from {plugin_dir}[/green]"
        )
    else:
        # Re-trigger auto-discovery
        from ingestforge.core.pipeline.registry import _auto_discover_processors

        _auto_discover_processors()
        console.print("[green]Auto-discovery triggered[/green]")

    count_after = len(_get_registry()._id_map)
    console.print(f"  Processors before: {count_before}")
    console.print(f"  Processors after: {count_after}")
    console.print(f"  New processors: {count_after - count_before}")
