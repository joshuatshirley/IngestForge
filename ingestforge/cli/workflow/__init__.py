"""Workflow automation commands package.

Provides workflow and batch operation tools:
- batch: Run batch operations on multiple files
- pipeline: Execute multi-step pipelines with YAML support
- schedule: Manage scheduled workflow execution

Usage:
    from ingestforge.cli.workflow import workflow_app
"""

from __future__ import annotations

from ingestforge.cli.workflow.main import app as workflow_app

__all__ = ["workflow_app"]
