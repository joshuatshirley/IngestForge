"""Gaps command - Identify coverage gaps.

Analyzes missing information and suggests research directions.

Follows Commandments #4 (Small Functions), #7 (Check Parameters), and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.argument.base import ArgumentCommand


class GapsCommand(ArgumentCommand):
    """Identify knowledge gaps and missing information."""

    def execute(
        self,
        topic: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Identify gaps in knowledge base."""
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

            gaps_data = self._generate_gaps(llm_client, topic, chunks)

            if not gaps_data:
                self.print_error("Failed to identify gaps")
                return 1

            self._display_gaps(gaps_data, topic)

            if output:
                self.save_json_output(
                    output, gaps_data, f"Gaps analysis saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Gap analysis failed")

    def _generate_gaps(
        self, llm_client: Any, topic: str, chunks: list[Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate gap analysis."""
        context = self.format_context_for_prompt(chunks, max_length=6000)
        prompt = f"""Identify knowledge gaps about: "{topic}"

Context:
{context}

Generate gap analysis in JSON:
{{
  "topic": "{topic}",
  "coverage_gaps": [
    {{"gap": "...", "priority": "high|medium|low", "suggested_research": "..."}},
    ...
  ],
  "missing_perspectives": ["perspective 1", ...],
  "unanswered_questions": ["question 1", ...],
  "completeness_score": 0-100
}}

Return ONLY valid JSON."""

        response = self.generate_with_llm(llm_client, prompt, "gap analysis")
        return self.parse_json_response(response)

    def _display_gaps(self, gaps_data: Dict[str, Any], topic: str) -> None:
        """Display gaps."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold blue]Knowledge Gaps: {topic}[/bold blue]", border_style="blue"
            )
        )
        self.console.print()

        completeness = gaps_data.get("completeness_score", 50)
        self.print_info(f"Completeness Score: {completeness}/100")
        self.console.print()

        gaps = gaps_data.get("coverage_gaps", [])

        table = Table(title="Coverage Gaps", show_lines=True)
        table.add_column("Gap", style="yellow", width=35)
        table.add_column("Priority", style="red", width=10)
        table.add_column("Suggested Research", style="green", width=35)

        for gap in gaps:
            gap_desc = gap.get("gap", "")
            priority = gap.get("priority", "medium")
            research = gap.get("suggested_research", "")
            table.add_row(gap_desc, priority, research)

        self.console.print(table)

        questions = gaps_data.get("unanswered_questions", [])
        if questions:
            self.console.print()
            self.console.print("[bold cyan]Unanswered Questions:[/bold cyan]")
            for q in questions[:5]:
                self.console.print(f"  • {q}")


def command(
    topic: str = typer.Argument(..., help="Topic to analyze for gaps"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
) -> None:
    """Identify knowledge gaps and suggest research directions.

    Examples:
        ingestforge argument gaps "Machine Learning"
        ingestforge argument gaps "Ancient Rome" -o gaps.json
    """
    cmd = GapsCommand()
    exit_code = cmd.execute(topic, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
