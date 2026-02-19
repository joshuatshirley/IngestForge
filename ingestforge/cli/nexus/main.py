"""Nexus Management CLI.

Task 256: Global Silence and Peer Revocation.
JPL Rule #4: CLI methods < 60 lines.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from ingestforge.storage.nexus_registry import NexusRegistry

app = typer.Typer(help="Manage Workspace Nexus federation and global isolation.")
console = Console()


def _get_registry() -> NexusRegistry:
    """Get registry singleton."""
    return NexusRegistry(Path(".data/nexus"))


@app.command("kill")
def kill_peer(
    peer_id: str = typer.Option(None, "--peer", help="Peer ID to revoke"),
    all_peers: bool = typer.Option(
        False, "--all", help="Enable Global Silence mode (Isolation)"
    ),
) -> None:
    """Emergency revocation of peers or global isolation."""
    registry = _get_registry()

    if all_peers:
        if registry.set_silence(True):
            console.print(
                "[bold red]GLOBAL SILENCE ACTIVATED.[/bold red] All federated traffic blocked."
            )
        else:
            console.print(
                "[bold yellow]Failed to activate global silence.[/bold yellow]"
            )
        return

    if peer_id:
        from ingestforge.core.security.nexus_acl import NexusACLManager

        acl = NexusACLManager(Path(".data/nexus"))
        acl.revoke_all_access(peer_id)
        console.print(f"[bold green]Peer {peer_id} revoked globally.[/bold green]")
    else:
        console.print("[yellow]Please specify a Peer ID or --all.[/yellow]")


@app.command("restore")
def restore_federation() -> None:
    """Disable Global Silence mode."""
    registry = _get_registry()
    if registry.set_silence(False):
        console.print(
            "[bold green]Federation restored.[/bold green] Network traffic resumed."
        )
    else:
        console.print("[red]Failed to restore federation state.[/red]")


@app.command("list")
def list_peers() -> None:
    """List all registered Nexus peers."""
    registry = _get_registry()
    peers = registry.list_active_peers()
    silenced = registry.is_silenced()

    if silenced:
        console.print("[bold red]NETWORK STATUS: SILENCED[/bold red]")

    table = Table(title="Nexus Peer Registry")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")

    for peer in peers:
        table.add_row(peer.id, peer.name, peer.status)

    console.print(table)
