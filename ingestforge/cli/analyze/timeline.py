"""Timeline command - Build security incident timelines (CYBER-004).

Sorts disparate log sources into a unified security timeline with links
to evidence chunks. Supports Markdown, JSON, and ASCII output formats.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from ingestforge.cli.analyze.base import AnalyzeCommand


class TimelineCommand(AnalyzeCommand):
    """Build security incident timelines from log sources."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        output_format: str = "md",
        correlate: bool = False,
    ) -> int:
        """Build incident timeline from stored events.

        Args:
            project: Project directory
            output: Output file for timeline
            start: Start date (ISO format or YYYY-MM-DD)
            end: End date (ISO format or YYYY-MM-DD)
            output_format: Output format (md, json, ascii)
            correlate: Whether to show correlated event groups

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs
            self._validate_format(output_format)
            start_dt = self._parse_date(start, "start")
            end_dt = self._parse_date(end, "end")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Retrieve chunks
            chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not chunks:
                self._handle_no_chunks()
                return 0

            # Build timeline
            timeline_data = self._build_timeline(chunks, start_dt, end_dt)

            # Display results
            self._display_timeline(timeline_data, output_format)

            # Show correlations if requested
            if correlate:
                self._display_correlations(timeline_data)

            # Save to file if requested
            if output:
                self._save_timeline(output, timeline_data, output_format)

            return 0

        except Exception as e:
            return self.handle_error(e, "Timeline building failed")

    def _validate_format(self, output_format: str) -> None:
        """Validate output format option."""
        valid_formats = {"md", "json", "ascii"}
        if output_format not in valid_formats:
            raise typer.BadParameter(
                f"Format must be one of: {', '.join(valid_formats)}"
            )

    def _parse_date(self, date_str: Optional[str], name: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None

        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        raise typer.BadParameter(
            f"Invalid {name} date format. Use YYYY-MM-DD or ISO format."
        )

    def _handle_no_chunks(self) -> None:
        """Handle case where no chunks found."""
        self.print_warning("Knowledge base is empty")
        self.print_info("Ingest log files to build a timeline")

    def _build_timeline(
        self,
        chunks: List[Any],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> Dict[str, Any]:
        """Build timeline from chunks.

        Args:
            chunks: List of chunks from storage
            start: Start date filter
            end: End date filter

        Returns:
            Timeline data dictionary
        """
        from ingestforge.analysis.timeline_builder import TimelineBuilder

        builder = TimelineBuilder()

        # Add chunks to timeline
        for chunk in chunks:
            chunk_dict = self._chunk_to_dict(chunk)
            builder.add_chunk(chunk_dict)

        # Build timeline with date range
        entries = builder.build(start=start, end=end)

        return {
            "builder": builder,
            "entries": entries,
            "start": start,
            "end": end,
        }

    def _chunk_to_dict(self, chunk: Any) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        if isinstance(chunk, dict):
            return chunk

        # Build dict from object attributes
        chunk_dict: Dict[str, Any] = {}

        # Extract content
        text = self.extract_chunk_text(chunk)
        chunk_dict["content"] = text

        # Extract metadata
        metadata = self.extract_chunk_metadata(chunk)
        chunk_dict["metadata"] = metadata

        # Extract chunk ID
        for attr in ("chunk_id", "id"):
            if hasattr(chunk, attr):
                chunk_dict["chunk_id"] = getattr(chunk, attr)
                break

        return chunk_dict

    def _display_timeline(
        self, timeline_data: Dict[str, Any], output_format: str
    ) -> None:
        """Display timeline in specified format."""
        builder = timeline_data["builder"]
        entries = timeline_data["entries"]

        self.console.print()
        self.print_info(f"Timeline: {len(entries)} events")
        self.console.print()

        if output_format == "md":
            self._display_markdown(builder)
        elif output_format == "json":
            self._display_json(builder)
        else:
            self._display_ascii(builder)

    def _display_markdown(self, builder: Any) -> None:
        """Display Markdown output."""
        md_content = builder.to_markdown()
        panel = Panel(
            Markdown(md_content),
            title="[bold cyan]Security Timeline[/bold cyan]",
            border_style="cyan",
        )
        self.console.print(panel)

    def _display_json(self, builder: Any) -> None:
        """Display JSON output."""
        json_content = builder.to_json()
        from rich.syntax import Syntax

        syntax = Syntax(json_content, "json", theme="monokai", line_numbers=True)
        self.console.print(syntax)

    def _display_ascii(self, builder: Any) -> None:
        """Display ASCII table output."""
        ascii_content = builder.to_ascii_table()
        self.console.print(ascii_content)

    def _display_correlations(self, timeline_data: Dict[str, Any]) -> None:
        """Display correlated event groups."""
        builder = timeline_data["builder"]
        groups = builder.correlate_events()

        if not groups:
            self.print_info("No correlated event groups found")
            return

        self.console.print()
        self.console.print("[bold]Correlated Event Groups:[/bold]")
        self.console.print()

        table = Table(title="Event Correlations")
        table.add_column("Type", style="cyan")
        table.add_column("Events", style="green")
        table.add_column("Confidence", style="yellow")

        for group in groups[:10]:  # Limit to 10 groups
            table.add_row(
                group.correlation_type,
                str(len(group.entries)),
                f"{group.confidence:.0%}",
            )

        self.console.print(table)

    def _save_timeline(
        self,
        output: Path,
        timeline_data: Dict[str, Any],
        output_format: str,
    ) -> None:
        """Save timeline to file."""
        builder = timeline_data["builder"]

        try:
            if output_format == "json":
                content = builder.to_json()
            elif output_format == "ascii":
                content = builder.to_ascii_table()
            else:
                content = builder.to_markdown()

            output.write_text(content, encoding="utf-8")
            self.print_success(f"Timeline saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save timeline: {e}")


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for timeline"
    ),
    start: Optional[str] = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD or ISO format)"
    ),
    end: Optional[str] = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD or ISO format)"
    ),
    output_format: str = typer.Option(
        "md", "--format", "-f", help="Output format: md, json, ascii"
    ),
    correlate: bool = typer.Option(
        False, "--correlate", "-c", help="Show correlated event groups"
    ),
) -> None:
    """Build security incident timeline from stored events.

    Sorts log events chronologically and generates a unified timeline
    with links to evidence. Supports multiple output formats.

    Examples:
        # Build timeline with Markdown output
        ingestforge analyze timeline

        # Build timeline for specific date range
        ingestforge analyze timeline --start 2024-01-01 --end 2024-01-31

        # Export as JSON
        ingestforge analyze timeline --format json --output timeline.json

        # Show correlated events
        ingestforge analyze timeline --correlate

        # ASCII table for CLI viewing
        ingestforge analyze timeline --format ascii
    """
    cmd = TimelineCommand()
    exit_code = cmd.execute(project, output, start, end, output_format, correlate)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
