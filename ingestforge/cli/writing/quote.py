"""Quote command - Find quotable passages."""

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from ingestforge.cli.writing.base import WritingCommand


class QuoteCommand(WritingCommand):
    """Find quotable passages."""

    def execute(
        self, topic: str, project: Optional[Path] = None, output: Optional[Path] = None
    ) -> int:
        try:
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self.get_llm_client(ctx)
            if not llm_client:
                return 1

            chunks = self.search_context(ctx["storage"], topic, k=30)
            if not chunks:
                self.print_warning(f"No context found for: '{topic}'")
                return 0

            context = self.format_context(chunks)
            prompt = f"""Find 5-10 notable quotable passages about: "{topic}"

Context:
{context}

Return JSON:
{{
  "quotes": [
    {{"quote": "...", "source": "...", "context": "...", "usage": "..."}},
    ...
  ]
}}"""

            response = self.generate_with_llm(llm_client, prompt, "quotes")
            data = self.parse_json(response) or {"quotes": []}

            self.console.print()
            table = Table(title=f"Quotable Passages: {topic}")
            table.add_column("Quote", width=50)
            table.add_column("Usage", width=30)

            for q in data.get("quotes", []):
                table.add_row(q.get("quote", ""), q.get("usage", ""))

            self.console.print(table)

            if output:
                self.save_json_output(output, data, f"Quotes saved to: {output}")

            return 0
        except Exception as e:
            return self.handle_error(e, "Quote finding failed")


def command(
    topic: str = typer.Argument(..., help="Topic to find quotes about"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Find quotable passages about a topic."""
    exit_code = QuoteCommand().execute(topic, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
