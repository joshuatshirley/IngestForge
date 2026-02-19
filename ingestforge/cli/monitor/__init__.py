"""Monitor command group - System monitoring and diagnostics.

Provides commands for system monitoring and diagnostics:
- health: System health check
- metrics: Show system metrics
- logs: View and analyze logs
- diagnostics: Run diagnostics

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations


from ingestforge.cli.monitor.main import app as monitor_app

__all__ = ["monitor_app"]
