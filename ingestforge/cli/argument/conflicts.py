"""Conflicts command - Detect contradictions.

Identifies conflicting information and inconsistencies.

Follows Commandments #4 (Small Functions), #7 (Check Parameters), and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.argument.base import ArgumentCommand


class ConflictsCommand(ArgumentCommand):
    """Detect contradictions and conflicts."""

    def execute(
        self,
        topic: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Detect conflicts in knowledge base."""
        try:
            if not topic or len(topic) < 2:
                self.print_error("Topic too short")
                return 1

            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            chunks = self.search_claim_context(ctx["storage"], topic, k=50)

            if not chunks:
                self.print_warning(f"No context found for: '{topic}'")
                return 0

            conflicts_data = self._generate_conflicts(llm_client, topic, chunks)

            if not conflicts_data:
                self.print_error("Failed to detect conflicts")
                return 1

            self._display_conflicts(conflicts_data, topic)

            if output:
                self.save_json_output(
                    output, conflicts_data, f"Conflicts saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Conflict detection failed")

    def _generate_conflicts(
        self, llm_client: Any, topic: str, chunks: list[Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate conflict analysis."""
        context = self.format_context_for_prompt(chunks, max_length=6000)
        prompt = f"""Identify contradictions and conflicts about: "{topic}"

Context:
{context}

Generate conflict analysis in JSON:
{{
  "topic": "{topic}",
  "conflicts": [
    {{"claim1": "...", "claim2": "...", "severity": "major|moderate|minor", "resolution": "..."}},
    ...
  ],
  "inconsistencies": ["inconsistency 1", ...],
  "unresolved_questions": ["question 1", ...]
}}

Return ONLY valid JSON."""

        response = self.generate_with_llm(llm_client, prompt, "conflict analysis")
        return self.parse_json_response(response)

    def _display_conflicts(self, conflicts_data: Dict[str, Any], topic: str) -> None:
        """Display conflicts."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold yellow]Conflicts: {topic}[/bold yellow]", border_style="yellow"
            )
        )
        self.console.print()

        conflicts = conflicts_data.get("conflicts", [])

        if not conflicts:
            self.print_success("No major conflicts detected!")
            return

        table = Table(title="Detected Conflicts", show_lines=True)
        table.add_column("Claim 1", style="cyan", width=30)
        table.add_column("Claim 2", style="magenta", width=30)
        table.add_column("Severity", style="red", width=10)
        table.add_column("Resolution", style="green", width=20)

        for conflict in conflicts:
            claim1 = conflict.get("claim1", "")
            claim2 = conflict.get("claim2", "")
            severity = conflict.get("severity", "moderate")
            resolution = conflict.get("resolution", "Unclear")
            table.add_row(claim1, claim2, severity, resolution)

        self.console.print(table)


def command(
    topic: str = typer.Argument(..., help="Topic to check for conflicts"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
) -> None:
    """Detect contradictions and conflicting information.

    Examples:
        ingestforge argument conflicts "COVID-19 treatments"
        ingestforge argument conflicts "Economic policy effects" -o conflicts.json
    """
    cmd = ConflictsCommand()
    exit_code = cmd.execute(topic, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
