"""Thesis command - Evaluate thesis statements."""

from pathlib import Path
from typing import Optional
import typer
from rich.panel import Panel
from ingestforge.cli.writing.base import WritingCommand


class ThesisCommand(WritingCommand):
    """Evaluate thesis statements."""

    def execute(
        self, thesis: str, project: Optional[Path] = None, output: Optional[Path] = None
    ) -> int:
        try:
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self.get_llm_client(ctx)
            if not llm_client:
                return 1

            chunks = self.search_context(ctx["storage"], thesis, k=20)
            context = self.format_context(chunks) if chunks else ""

            prompt = f"""Evaluate this thesis statement: "{thesis}"

Context:
{context}

Return JSON:
{{
  "thesis": "{thesis}",
  "strength": "strong|moderate|weak",
  "is_arguable": true|false,
  "specificity": "high|medium|low",
  "improvements": ["suggestion 1", ...],
  "revised_versions": ["version 1", ...]
}}"""

            response = self.generate_with_llm(llm_client, prompt, "evaluation")
            data = self.parse_json(response) or {}

            self.console.print()
            self.console.print(
                Panel(
                    f"[bold]Thesis Evaluation[/bold]\n\n{thesis}", border_style="cyan"
                )
            )
            self.console.print()

            self.print_info(f"Strength: {data.get('strength', 'unknown')}")
            self.print_info(f"Arguable: {data.get('is_arguable', 'unknown')}")
            self.print_info(f"Specificity: {data.get('specificity', 'unknown')}")

            improvements = data.get("improvements", [])
            if improvements:
                self.console.print("\n[bold yellow]Suggestions:[/bold yellow]")
                for imp in improvements:
                    self.console.print(f"  • {imp}")

            revised = data.get("revised_versions", [])
            if revised:
                self.console.print("\n[bold green]Revised Versions:[/bold green]")
                for idx, rev in enumerate(revised, 1):
                    self.console.print(f"  {idx}. {rev}")

            if output:
                self.save_json_output(output, data, f"Evaluation saved to: {output}")

            return 0
        except Exception as e:
            return self.handle_error(e, "Thesis evaluation failed")


def command(
    thesis: str = typer.Argument(..., help="Thesis statement to evaluate"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Evaluate and improve thesis statements."""
    exit_code = ThesisCommand().execute(thesis, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
