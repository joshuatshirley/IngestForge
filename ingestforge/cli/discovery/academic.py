"""Academic search command - Find academic papers and research."""

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from rich.panel import Panel
from ingestforge.cli.discovery.base import DiscoveryCommand


class AcademicSearchCommand(DiscoveryCommand):
    """Search academic papers and research."""

    def execute(
        self,
        query: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        limit: int = 10,
    ) -> int:
        try:
            ctx = self.initialize_context(project, require_storage=True)
            llm_client = self.get_llm_client(ctx)
            if not llm_client:
                return 1

            # Get context to understand domain
            chunks = self.search_context(ctx["storage"], query, k=10)
            context = (
                "\n".join([getattr(c, "text", str(c))[:200] for c in chunks[:3]])
                if chunks
                else ""
            )

            prompt = f"""Suggest academic research papers and sources for: "{query}"

Context from knowledge base:
{context}

Return JSON:
{{
  "query": "{query}",
  "suggested_papers": [
    {{"title": "...", "authors": "...", "venue": "...", "year": 2024, "relevance": "high|medium|low", "search_terms": "..."}},
    ...
  ],
  "search_strategies": ["strategy 1", ...],
  "databases": ["arXiv", "Semantic Scholar", "Google Scholar", ...]
}}

Focus on: arXiv preprints, peer-reviewed papers, conference proceedings.
Limit to {limit} most relevant papers."""

            response = self.generate_with_llm(
                llm_client, prompt, "academic suggestions"
            )
            data = self.parse_json(response) or {"suggested_papers": []}

            self._display_results(data, query)

            if output:
                self.save_json_output(
                    output, data, f"Academic search saved to: {output}"
                )

            return 0
        except Exception as e:
            return self.handle_error(e, "Academic search failed")

    def _display_results(self, data: dict, query: str) -> None:
        """Display academic search results."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]Academic Search: {query}[/bold cyan]", border_style="cyan"
            )
        )
        self.console.print()

        papers = data.get("suggested_papers", [])

        table = Table(title="Suggested Research Papers", show_lines=True)
        table.add_column("Title", width=35)
        table.add_column("Authors", width=20)
        table.add_column("Venue/Year", width=15)
        table.add_column("Relevance", width=10)

        for paper in papers:
            title = paper.get("title", "")
            authors = paper.get("authors", "")
            venue = paper.get("venue", "")
            year = paper.get("year", "")
            relevance = paper.get("relevance", "medium")

            venue_year = f"{venue} ({year})" if venue and year else venue or str(year)
            table.add_row(title, authors, venue_year, relevance)

        self.console.print(table)

        strategies = data.get("search_strategies", [])
        if strategies:
            self.console.print()
            self.console.print("[bold yellow]Search Strategies:[/bold yellow]")
            for s in strategies:
                self.console.print(f"  • {s}")

        databases = data.get("databases", [])
        if databases:
            self.console.print()
            self.print_info(f"Recommended databases: {', '.join(databases)}")


def command(
    query: str = typer.Argument(..., help="Research topic or query"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of papers to suggest"),
) -> None:
    """Search for academic papers and research.

    Suggests relevant papers from arXiv, Semantic Scholar, and other academic sources.

    Examples:
        ingestforge discovery academic "machine learning interpretability"
        ingestforge discovery academic "quantum computing" --limit 20
        ingestforge discovery academic "climate models" -o papers.json
    """
    exit_code = AcademicSearchCommand().execute(query, project, output, limit)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
