"""Config command group - Configuration management.

Provides commands for managing application configuration:
- show: Display current configuration
- set: Set configuration values
- reset: Reset to defaults
- validate: Validate configuration

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations


from ingestforge.cli.config.main import app as config_app

__all__ = ["config_app"]
