"""Timeline command - Map chronological events and development."""

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from ingestforge.cli.discovery.base import DiscoveryCommand


class TimelineCommand(DiscoveryCommand):
    """Generate chronological timelines."""

    def execute(
        self, topic: str, project: Optional[Path] = None, output: Optional[Path] = None
    ) -> int:
        try:
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self.get_llm_client(ctx)
            if not llm_client:
                return 1

            chunks = self.search_context(ctx["storage"], topic, k=30)
            context = (
                "\n".join([getattr(c, "text", str(c))[:200] for c in chunks[:10]])
                if chunks
                else ""
            )

            prompt = f"""Create chronological timeline for: "{topic}"

Context:
{context}

Return JSON:
{{
  "topic": "{topic}",
  "events": [
    {{"date": "YYYY or YYYY-MM", "event": "...", "significance": "...", "category": "..."}},
    ...
  ],
  "periods": [
    {{"name": "period name", "start": "YYYY", "end": "YYYY", "description": "..."}},
    ...
  ],
  "key_milestones": ["milestone 1", ...]
}}

Order events chronologically (oldest to newest)."""

            response = self.generate_with_llm(llm_client, prompt, "timeline")
            data = self.parse_json(response) or {"events": []}

            self._display_timeline(data, topic)

            if output:
                self.save_json_output(output, data, f"Timeline saved to: {output}")

            return 0
        except Exception as e:
            return self.handle_error(e, "Timeline generation failed")

    def _display_timeline(self, data: dict, topic: str) -> None:
        """Display timeline."""
        self.console.print()
        self.console.print(f"[bold cyan]Timeline: {topic}[/bold cyan]\n")

        events = data.get("events", [])

        table = Table(title="Chronological Events", show_lines=True)
        table.add_column("Date", width=12)
        table.add_column("Event", width=40)
        table.add_column("Significance", width=30)

        for event in events:
            date = event.get("date", "")
            evt = event.get("event", "")
            sig = event.get("significance", "")

            table.add_row(date, evt, sig)

        self.console.print(table)

        milestones = data.get("key_milestones", [])
        if milestones:
            self.console.print()
            self.console.print("[bold yellow]Key Milestones:[/bold yellow]")
            for m in milestones[:5]:
                self.console.print(f"  • {m}")


def command(
    topic: str = typer.Argument(..., help="Topic to create timeline for"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Generate chronological timeline of events.

    Examples:
        ingestforge discovery timeline "World War II"
        ingestforge discovery timeline "Evolution of AI" -o timeline.json
    """
    exit_code = TimelineCommand().execute(topic, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
