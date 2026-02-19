"""Educational resources command - Find learning materials."""

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from ingestforge.cli.discovery.base import DiscoveryCommand


class EducationalCommand(DiscoveryCommand):
    """Find educational resources."""

    def execute(
        self, topic: str, project: Optional[Path] = None, output: Optional[Path] = None
    ) -> int:
        try:
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self.get_llm_client(ctx)
            if not llm_client:
                return 1

            prompt = f"""Suggest educational resources for: "{topic}"

Return JSON:
{{
  "topic": "{topic}",
  "resources": [
    {{"title": "...", "platform": "MIT OCW|Khan Academy|Coursera|edX|YouTube", "type": "course|video|lecture|tutorial", "level": "beginner|intermediate|advanced", "url_search": "..."}},
    ...
  ],
  "learning_path": ["step 1", ...],
  "prerequisites": ["prereq 1", ...]
}}

Focus on: MIT OCW, Khan Academy, Coursera, edX, YouTube."""

            response = self.generate_with_llm(
                llm_client, prompt, "educational resources"
            )
            data = self.parse_json(response) or {"resources": []}

            self._display_resources(data, topic)

            if output:
                self.save_json_output(
                    output, data, f"Educational resources saved to: {output}"
                )

            return 0
        except Exception as e:
            return self.handle_error(e, "Educational discovery failed")

    def _display_resources(self, data: dict, topic: str) -> None:
        """Display educational resources."""
        self.console.print()
        self.console.print(f"[bold cyan]Educational Resources: {topic}[/bold cyan]\n")

        resources = data.get("resources", [])

        table = Table(title="Recommended Resources")
        table.add_column("Title", width=30)
        table.add_column("Platform", width=15)
        table.add_column("Type", width=12)
        table.add_column("Level", width=12)

        for res in resources:
            table.add_row(
                res.get("title", ""),
                res.get("platform", ""),
                res.get("type", ""),
                res.get("level", ""),
            )

        self.console.print(table)

        path = data.get("learning_path", [])
        if path:
            self.console.print()
            self.console.print("[bold yellow]Suggested Learning Path:[/bold yellow]")
            for idx, step in enumerate(path, 1):
                self.console.print(f"  {idx}. {step}")


def command(
    topic: str = typer.Argument(..., help="Topic to find resources for"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Find educational resources and learning materials.

    Examples:
        ingestforge discovery educational "Linear Algebra"
        ingestforge discovery educational "Python programming" -o resources.json
    """
    exit_code = EducationalCommand().execute(topic, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
