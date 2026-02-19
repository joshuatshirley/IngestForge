"""Workflow automation subcommands.

Provides tools for workflow automation and batch operations:
- batch: Run batch operations on multiple files
- pipeline: Execute multi-step pipelines with YAML support
- schedule: Manage scheduled workflow execution

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.workflow import batch, pipeline, schedule

# Create workflow subcommand application
app = typer.Typer(
    name="workflow",
    help="Workflow automation tools",
    add_completion=False,
)

# Register workflow commands
app.command("batch")(batch.command)
app.command("pipeline")(pipeline.command)
app.command("schedule")(schedule.command)


@app.callback()
def main() -> None:
    """Workflow automation tools for IngestForge.

    Automate repetitive tasks and execute multi-step workflows:
    - Batch operations on multiple files
    - Multi-step pipeline execution with YAML support
    - Scheduled workflow management
    - Automated processing workflows

    Features:
    - Process multiple files at once
    - Chain multiple operations with conditions
    - Schedule recurring workflows with cron
    - Variable substitution in pipelines
    - Dry-run mode for testing
    - Progress reporting and detailed reports

    Use cases:
    - Bulk document processing
    - Automated ingestion pipelines
    - Batch analysis operations
    - Scheduled nightly ingestion
    - Quality assurance workflows

    Examples:
        # Batch ingest all PDFs
        ingestforge workflow batch ingest documents/ --pattern "*.pdf"

        # Run full pipeline
        ingestforge workflow pipeline full data/

        # Run YAML-defined pipeline
        ingestforge workflow pipeline my-pipeline.yaml data/

        # Dry run to test pipeline
        ingestforge workflow pipeline full data/ --dry-run

        # Create scheduled workflow
        ingestforge workflow schedule create nightly --cron "0 2 * * *" --command "ingestforge ingest ./docs"

        # List scheduled workflows
        ingestforge workflow schedule list

        # Run schedule immediately
        ingestforge workflow schedule run nightly

    For help on specific commands:
        ingestforge workflow <command> --help
    """
    pass
