"""Counter command - Generate counterarguments.

Identifies opposing views and generates rebuttals.

Follows Commandments #4 (Small Functions), #7 (Check Parameters), and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.argument.base import ArgumentCommand


class CounterCommand(ArgumentCommand):
    """Generate counterarguments and rebuttals."""

    def execute(
        self,
        claim: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Generate counterarguments for a claim."""
        try:
            self.validate_claim(claim)
            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            chunks = self.search_claim_context(ctx["storage"], claim, k=40)

            if not chunks:
                self.print_warning(f"No context found for: '{claim}'")
                return 0

            counter_data = self._generate_counter(llm_client, claim, chunks)

            if not counter_data:
                self.print_error("Failed to generate counterarguments")
                return 1

            self._display_counter(counter_data, claim)

            if output:
                self.save_json_output(
                    output, counter_data, f"Counter analysis saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Counter analysis failed")

    def _generate_counter(
        self, llm_client: Any, claim: str, chunks: list[Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate counter analysis."""
        context = self.format_context_for_prompt(chunks)
        prompt = f"""Identify counterarguments and generate rebuttals for: "{claim}"

Context:
{context}

Generate counter analysis in JSON:
{{
  "claim": "{claim}",
  "counterarguments": [
    {{"argument": "...", "strength": "strong|moderate|weak", "rebuttal": "..."}},
    ...
  ],
  "weaknesses": ["weakness 1", ...],
  "alternative_views": ["view 1", ...]
}}

Return ONLY valid JSON."""

        response = self.generate_with_llm(llm_client, prompt, "counter analysis")
        return self.parse_json_response(response)

    def _display_counter(self, counter_data: Dict[str, Any], claim: str) -> None:
        """Display counter analysis."""
        self.console.print()
        self.console.print(
            Panel(f"[bold red]Counterarguments: {claim}[/bold red]", border_style="red")
        )
        self.console.print()

        counters = counter_data.get("counterarguments", [])

        table = Table(title="Counterarguments & Rebuttals", show_lines=True)
        table.add_column("Counterargument", style="red", width=35)
        table.add_column("Rebuttal", style="green", width=35)
        table.add_column("Strength", style="yellow", width=10)

        for counter in counters:
            arg = counter.get("argument", "")
            rebuttal = counter.get("rebuttal", "")
            strength = counter.get("strength", "moderate")
            table.add_row(arg, rebuttal, strength)

        self.console.print(table)


def command(
    claim: str = typer.Argument(..., help="Claim to analyze"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
) -> None:
    """Generate counterarguments and rebuttals.

    Examples:
        ingestforge argument counter "Remote work increases productivity"
        ingestforge argument counter "Nuclear energy is safe" -o counter.json
    """
    cmd = CounterCommand()
    exit_code = cmd.execute(claim, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
