"""Scholars command - Identify key researchers and authors."""

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from ingestforge.cli.discovery.base import DiscoveryCommand


class ScholarsCommand(DiscoveryCommand):
    """Identify key researchers and authors."""

    def execute(
        self,
        field: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        limit: int = 10,
    ) -> int:
        try:
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self.get_llm_client(ctx)
            if not llm_client:
                return 1

            chunks = self.search_context(ctx["storage"], field, k=15)
            context = (
                "\n".join([getattr(c, "text", str(c))[:150] for c in chunks[:5]])
                if chunks
                else ""
            )

            prompt = f"""Identify key researchers and scholars in: "{field}"

Context:
{context}

Return JSON:
{{
  "field": "{field}",
  "scholars": [
    {{"name": "...", "affiliation": "...", "contributions": ["..."], "notable_works": ["..."], "impact": "high|medium"}},
    ...
  ],
  "research_groups": ["group 1", ...],
  "conferences": ["conference 1", ...]
}}

Limit to {limit} most influential scholars."""

            response = self.generate_with_llm(
                llm_client, prompt, "scholar identification"
            )
            data = self.parse_json(response) or {"scholars": []}

            self._display_scholars(data, field)

            if output:
                self.save_json_output(output, data, f"Scholars saved to: {output}")

            return 0
        except Exception as e:
            return self.handle_error(e, "Scholar discovery failed")

    def _display_scholars(self, data: dict, field: str) -> None:
        """Display scholars."""
        self.console.print()
        self.console.print(f"[bold cyan]Key Scholars: {field}[/bold cyan]\n")

        scholars = data.get("scholars", [])

        table = Table(title="Influential Researchers", show_lines=True)
        table.add_column("Name", width=20)
        table.add_column("Affiliation", width=25)
        table.add_column("Key Contributions", width=35)

        for scholar in scholars:
            name = scholar.get("name", "")
            affiliation = scholar.get("affiliation", "")
            contributions = scholar.get("contributions", [])
            contrib_text = ", ".join(contributions[:2])

            table.add_row(name, affiliation, contrib_text)

        self.console.print(table)


def command(
    field: str = typer.Argument(..., help="Research field or topic"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of scholars"),
) -> None:
    """Identify key researchers and scholars in a field.

    Examples:
        ingestforge discovery scholars "Natural Language Processing"
        ingestforge discovery scholars "Quantum Computing" --limit 15
    """
    exit_code = ScholarsCommand().execute(field, project, output, limit)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
