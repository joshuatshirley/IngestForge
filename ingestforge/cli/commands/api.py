"""API command - Manage the IngestForge REST API server.

TICKET-504: CLI command to start the FastAPI server with configurable
host, port, and reload options.
Added subcommands for status and stop."""

from __future__ import annotations


import typer
import requests

from ingestforge.cli.core import IngestForgeCommand
from ingestforge.cli.console import get_console


class ApiCommand(IngestForgeCommand):
    """Start the IngestForge REST API server."""

    def _validate_inputs(self, host: str, port: int) -> bool:
        """Validate host and port inputs."""
        if not (1 <= port <= 65535):
            self.console.print(f"[red]Error: Invalid port number {port}.[/red]")
            return False
        if not host:
            self.console.print("[red]Error: Invalid host.[/red]")
            return False
        return True

    def execute(
        self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False
    ) -> int:
        """Start the API server."""
        try:
            if not self._validate_inputs(host, port):
                return 1

            self.console.print("\n[cyan]Starting IngestForge API Server[/cyan]")
            self.console.print(f"  Host: {host}")
            self.console.print(f"  Port: {port}")
            self.console.print(f"  Reload: {reload}")
            self.console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

            # Import and run server (lazy import)
            from ingestforge.api.main import run_server

            run_server(host=host, port=port, reload=reload)
            return 0

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Server stopped[/yellow]")
            return 0
        except Exception as e:
            return self.handle_error(e, "Failed to start API server")


# Create Typer app for API subcommands
api_app = typer.Typer(help="Manage IngestForge API server")


@api_app.command("start")
def start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the API server."""
    cmd = ApiCommand()
    exit_code = cmd.execute(host=host, port=port, reload=reload)
    raise typer.Exit(code=exit_code)


@api_app.command("status")
def status(
    host: str = typer.Option("localhost", "--host", "-h", help="API host"),
    port: int = typer.Option(8000, "--port", "-p", help="API port"),
) -> None:
    """Check if the API server is running."""
    url = f"http://{host}:{port}/v1/health"
    console = get_console()
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            console.print(
                f"[green]API Server is running at http://{host}:{port}[/green]"
            )
            console.print("Health check: [bold]OK[/bold]")
        else:
            console.print(
                f"[yellow]API Server responded with status {response.status_code}[/yellow]"
            )
    except requests.RequestException:
        console.print(f"[red]API Server is NOT running at http://{host}:{port}[/red]")
        raise typer.Exit(code=1)


@api_app.command("stop")
def stop() -> None:
    """Stop the API server (instructions)."""
    console = get_console()
    console.print(
        "\n[yellow]To stop the API server, please press Ctrl+C in the terminal where it is running.[/yellow]"
    )
    console.print(
        "[dim]Note: Background process management is not yet implemented for Windows.[/dim]\n"
    )


@api_app.command("docs")
def docs(
    host: str = typer.Option("localhost", "--host", "-h", help="API host"),
    port: int = typer.Option(8000, "--port", "-p", help="API port"),
) -> None:
    """Open API documentation in browser."""
    url = f"http://{host}:{port}/docs"
    import webbrowser

    console = get_console()
    console.print(f"Opening documentation: {url}")
    webbrowser.open(url)
