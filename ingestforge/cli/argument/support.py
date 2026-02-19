"""Support command - Find evidence for claims.

Identifies supporting evidence and assesses strength.

Follows Commandments #4 (Small Functions), #7 (Check Parameters), and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.argument.base import ArgumentCommand


class SupportCommand(ArgumentCommand):
    """Find supporting evidence for claims."""

    def execute(
        self,
        claim: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Find evidence supporting a claim."""
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

            support_data = self._generate_support(llm_client, claim, chunks)

            if not support_data:
                self.print_error("Failed to generate support analysis")
                return 1

            self._display_support(support_data, claim)

            if output:
                self.save_json_output(
                    output, support_data, f"Support analysis saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Support analysis failed")

    def _generate_support(
        self, llm_client: Any, claim: str, chunks: list[Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate support analysis."""
        context = self.format_context_for_prompt(chunks)
        prompt = f"""Find evidence supporting this claim: "{claim}"

Context:
{context}

Generate support analysis in JSON:
{{
  "claim": "{claim}",
  "supporting_evidence": [
    {{"evidence": "...", "source": "...", "strength": "strong|moderate|weak", "type": "empirical|theoretical|anecdotal"}},
    ...
  ],
  "overall_strength": "strong|moderate|weak",
  "confidence_level": "high|medium|low",
  "gaps": ["missing evidence 1", ...]
}}

Return ONLY valid JSON."""

        response = self.generate_with_llm(llm_client, prompt, "support analysis")
        return self.parse_json_response(response)

    def _display_support(self, support_data: Dict[str, Any], claim: str) -> None:
        """Display support analysis."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold green]Supporting Evidence: {claim}[/bold green]",
                border_style="green",
            )
        )
        self.console.print()

        evidence_list = support_data.get("supporting_evidence", [])

        table = Table(title="Evidence", show_lines=True)
        table.add_column("Evidence", style="white", width=50)
        table.add_column("Strength", style="yellow", width=12)
        table.add_column("Type", style="cyan", width=15)

        for ev in evidence_list:
            evidence = ev.get("evidence", "")
            strength = ev.get("strength", "moderate")
            ev_type = ev.get("type", "empirical")
            table.add_row(evidence, strength, ev_type)

        self.console.print(table)

        overall = support_data.get("overall_strength", "moderate")
        confidence = support_data.get("confidence_level", "medium")
        self.console.print()
        self.print_info(f"Overall Strength: {overall} | Confidence: {confidence}")


def command(
    claim: str = typer.Argument(..., help="Claim to find support for"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)"
    ),
) -> None:
    """Find supporting evidence for a claim.

    Examples:
        ingestforge argument support "Exercise improves mental health"
        ingestforge argument support "Renewable energy is cost-effective" -o support.json
    """
    cmd = SupportCommand()
    exit_code = cmd.execute(claim, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
