"""Ingestion Report command - Display redaction statistics (SEC-001.3).

Shows PII redaction statistics for ingested documents:
- Total items redacted per type
- Documents with most redactions
- Whitelisted items skipped

NASA JPL Commandments compliance:
- Rule #1: Linear control flow
- Rule #2: Fixed upper bounds
- Rule #4: Functions <60 lines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.core import IngestForgeCommand

MAX_DOCUMENTS_IN_REPORT = 100
MAX_TOP_DOCUMENTS = 10


@dataclass
class DocumentRedactionStats:
    """Redaction statistics for a single document.

    Attributes:
        document_id: Unique document identifier
        source_name: Original filename or URL
        total_redactions: Total number of redactions made
        by_type: Count per PII type
        skipped: Number of whitelisted items skipped
    """

    document_id: str
    source_name: str
    total_redactions: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    skipped: int = 0


@dataclass
class RedactionReport:
    """Aggregate redaction report across all documents.

    Attributes:
        total_documents: Number of documents processed
        total_redactions: Total redactions across all docs
        by_type: Aggregate counts per PII type
        total_skipped: Total whitelisted items skipped
        documents: Per-document statistics
    """

    total_documents: int = 0
    total_redactions: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    total_skipped: int = 0
    documents: List[DocumentRedactionStats] = field(default_factory=list)

    @property
    def top_documents(self) -> List[DocumentRedactionStats]:
        """Get documents with most redactions."""
        sorted_docs = sorted(
            self.documents,
            key=lambda d: d.total_redactions,
            reverse=True,
        )
        return sorted_docs[:MAX_TOP_DOCUMENTS]


class IngestReportCommand(IngestForgeCommand):
    """Display PII redaction statistics for ingested documents."""

    def execute(
        self,
        project: Optional[Path] = None,
        document_id: Optional[str] = None,
    ) -> int:
        """Display redaction report.

        Args:
            project: Project directory (default: current)
            document_id: Optional specific document to show

        Returns:
            0 on success, 1 on error
        """
        try:
            ctx = self.initialize_context(project, require_storage=True)
            storage = ctx["storage"]

            # Get redaction stats from storage metadata
            report = self._gather_redaction_stats(storage, document_id)

            # Display report
            if document_id:
                self._display_document_report(report, document_id)
            else:
                self._display_summary_report(report)

            return 0

        except Exception as e:
            return self.handle_error(e, "Report generation failed")

    def _gather_redaction_stats(
        self,
        storage: object,
        document_id: Optional[str],
    ) -> RedactionReport:
        """Gather redaction statistics from storage.

        Args:
            storage: Storage backend
            document_id: Optional filter for specific document

        Returns:
            Aggregated report
        """
        report = RedactionReport()

        # Get chunks with redaction metadata
        try:
            if hasattr(storage, "get_all_chunks"):
                chunks = storage.get_all_chunks()
            else:
                chunks = []
        except Exception:
            chunks = []

        # Group by document and aggregate
        doc_stats: Dict[str, DocumentRedactionStats] = {}

        for chunk in chunks[: MAX_DOCUMENTS_IN_REPORT * 100]:
            metadata = getattr(chunk, "metadata", {}) or {}

            # Skip if no redaction data
            if "redaction_stats" not in metadata:
                continue

            doc_id = metadata.get("document_id", "unknown")
            source = metadata.get("source", doc_id)

            # Filter by document_id if specified
            if document_id and doc_id != document_id:
                continue

            # Get or create doc stats
            if doc_id not in doc_stats:
                doc_stats[doc_id] = DocumentRedactionStats(
                    document_id=doc_id,
                    source_name=source,
                )

            stats = doc_stats[doc_id]
            redaction_data = metadata["redaction_stats"]

            # Aggregate stats
            chunk_total = redaction_data.get("total", 0)
            stats.total_redactions += chunk_total
            report.total_redactions += chunk_total

            chunk_skipped = redaction_data.get("skipped", 0)
            stats.skipped += chunk_skipped
            report.total_skipped += chunk_skipped

            # Aggregate by type
            for pii_type, count in redaction_data.get("by_type", {}).items():
                stats.by_type[pii_type] = stats.by_type.get(pii_type, 0) + count
                report.by_type[pii_type] = report.by_type.get(pii_type, 0) + count

        report.documents = list(doc_stats.values())
        report.total_documents = len(doc_stats)

        return report

    def _display_summary_report(self, report: RedactionReport) -> None:
        """Display summary report for all documents.

        Args:
            report: Aggregated report
        """
        # Overview panel
        overview = f"""[bold]Total Documents:[/bold] {report.total_documents}
[bold]Total Redactions:[/bold] {report.total_redactions}
[bold]Whitelisted Skipped:[/bold] {report.total_skipped}"""

        self.console.print(
            Panel(
                overview,
                title="[bold cyan]Redaction Summary[/bold cyan]",
                border_style="cyan",
            )
        )

        # By-type breakdown table
        if report.by_type:
            type_table = Table(title="Redactions by Type")
            type_table.add_column("PII Type", style="cyan")
            type_table.add_column("Count", style="green", justify="right")
            type_table.add_column("Percentage", justify="right")

            total = max(report.total_redactions, 1)
            for pii_type, count in sorted(
                report.by_type.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                pct = (count / total) * 100
                type_table.add_row(pii_type.upper(), str(count), f"{pct:.1f}%")

            self.console.print(type_table)

        # Top documents table
        if report.top_documents:
            doc_table = Table(title=f"Top {len(report.top_documents)} Documents")
            doc_table.add_column("Document", style="cyan", max_width=40)
            doc_table.add_column("Redactions", style="green", justify="right")
            doc_table.add_column("Skipped", justify="right")

            for doc in report.top_documents:
                doc_table.add_row(
                    doc.source_name[:40],
                    str(doc.total_redactions),
                    str(doc.skipped),
                )

            self.console.print(doc_table)

        # No data message
        if report.total_documents == 0:
            self.console.print(
                "[yellow]No redaction data found. "
                "Enable redaction in config.yaml and re-ingest documents.[/yellow]"
            )

    def _display_document_report(
        self,
        report: RedactionReport,
        document_id: str,
    ) -> None:
        """Display report for a specific document.

        Args:
            report: Report containing document data
            document_id: Document to display
        """
        # Find the document
        doc = next(
            (d for d in report.documents if d.document_id == document_id),
            None,
        )

        if not doc:
            self.console.print(
                f"[yellow]No redaction data found for document: {document_id}[/yellow]"
            )
            return

        # Document panel
        overview = f"""[bold]Document:[/bold] {doc.source_name}
[bold]ID:[/bold] {doc.document_id}
[bold]Total Redactions:[/bold] {doc.total_redactions}
[bold]Skipped (Whitelisted):[/bold] {doc.skipped}"""

        self.console.print(
            Panel(
                overview,
                title="[bold cyan]Document Redaction Report[/bold cyan]",
                border_style="cyan",
            )
        )

        # By-type breakdown
        if doc.by_type:
            type_table = Table(title="Redactions by Type")
            type_table.add_column("PII Type", style="cyan")
            type_table.add_column("Count", style="green", justify="right")

            for pii_type, count in sorted(
                doc.by_type.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                type_table.add_row(pii_type.upper(), str(count))

            self.console.print(type_table)


# CLI registration
app = typer.Typer(help="Redaction ingestion reports")


@app.command("summary")
def summary_command(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory",
    ),
) -> None:
    """Show redaction summary across all documents."""
    cmd = IngestReportCommand()
    raise typer.Exit(cmd.execute(project=project))


@app.command("document")
def document_command(
    document_id: str = typer.Argument(help="Document ID to show"),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project directory",
    ),
) -> None:
    """Show redaction report for a specific document."""
    cmd = IngestReportCommand()
    raise typer.Exit(cmd.execute(project=project, document_id=document_id))
