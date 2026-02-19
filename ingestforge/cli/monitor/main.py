"""Monitor subcommands.

Provides tools for system monitoring and diagnostics:
- health: System health check
- metrics: Show system metrics
- logs: View and analyze logs
- diagnostics: Run diagnostics

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.monitor import health, metrics, logs, diagnostics

# Create monitor subcommand application
app = typer.Typer(
    name="monitor",
    help="System monitoring and diagnostics",
    add_completion=False,
)

# Register monitor commands
app.command("health")(health.command)
app.command("metrics")(metrics.command)
app.command("logs")(logs.command)
app.command("diagnostics")(diagnostics.command)


@app.callback()
def main() -> None:
    """System monitoring and diagnostics for IngestForge.

    Monitor and diagnose your IngestForge system:
    - Check system health
    - View metrics and statistics
    - Analyze logs
    - Run comprehensive diagnostics

    Features:
    - Health checks with pass/fail status
    - Real-time metrics
    - Log analysis with pattern detection
    - Comprehensive diagnostic reports
    - Color-coded output

    Use cases:
    - Production monitoring
    - Troubleshooting issues
    - Performance analysis
    - System verification
    - Pre-deployment checks

    Examples:
        # Quick health check
        ingestforge monitor health

        # View system metrics
        ingestforge monitor metrics

        # Analyze recent logs
        ingestforge monitor logs --analyze

        # Run full diagnostics
        ingestforge monitor diagnostics

        # Generate diagnostic report
        ingestforge monitor diagnostics -o report.md

        # Monitor specific project
        ingestforge monitor health -p /path/to/project

    For help on specific commands:
        ingestforge monitor <command> --help
    """
    pass
