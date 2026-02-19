"""Bibliography command - Generate complete bibliography from citations."""

from pathlib import Path
from typing import Optional, Dict, List, Any
import typer
from ingestforge.cli.citation.base import CitationCommand


class BibliographyCommand(CitationCommand):
    """Generate complete bibliography from knowledge base."""

    def execute(
        self,
        style: str = "apa",
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        group_by: Optional[str] = None,
    ) -> int:
        """Generate bibliography from all citations."""
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Get all chunks
            all_chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not all_chunks:
                self.print_warning("No content in knowledge base")
                return 0

            # Extract citations
            citations = self._extract_citations(all_chunks)

            if not citations:
                self.print_warning("No citations found")
                return 0

            # Remove duplicates
            unique_citations = self._deduplicate_citations(citations)

            # Format citations
            formatted = self._format_bibliography(unique_citations, style)

            # Group if requested
            if group_by:
                formatted = self._group_citations(formatted, group_by)

            # Display bibliography
            self._display_bibliography(formatted, style, group_by)

            # Save if requested
            if output:
                self._save_bibliography(output, formatted, style, group_by)

            return 0

        except Exception as e:
            return self.handle_error(e, "Bibliography generation failed")

    def _extract_citations(self, chunks: list) -> List[Dict[str, Any]]:
        """Extract citations from chunks."""
        citations = []

        for chunk in chunks:
            metadata = self.extract_chunk_metadata(chunk)
            source = metadata.get("source", "Unknown")
            doc_type = metadata.get("type", "document")

            # Check for citation metadata
            if "citations" in metadata:
                for cite in metadata["citations"]:
                    citations.append(
                        {
                            "citation": cite,
                            "source": source,
                            "type": doc_type,
                            "metadata": metadata,
                        }
                    )
            else:
                # Use source as citation if no explicit citations
                citations.append(
                    {
                        "citation": source,
                        "source": source,
                        "type": doc_type,
                        "metadata": metadata,
                    }
                )

        return citations

    def _deduplicate_citations(self, citations: List[Dict]) -> List[Dict[str, Any]]:
        """Remove duplicate citations."""
        seen = set()
        unique = []

        for cite in citations:
            # Use citation text as key
            key = str(cite["citation"]).strip().lower()

            if key not in seen:
                seen.add(key)
                unique.append(cite)

        return unique

    def _format_bibliography(
        self, citations: List[Dict], style: str
    ) -> List[Dict[str, Any]]:
        """Format citations in specified style."""
        formatted = []

        for cite in citations:
            citation_str = str(cite["citation"])

            # Format based on style
            formatted_cite = self.format_citation(citation_str, style)

            formatted.append(
                {
                    "formatted": formatted_cite,
                    "original": citation_str,
                    "source": cite["source"],
                    "type": cite.get("type", "document"),
                }
            )

        # Sort alphabetically by formatted citation
        formatted.sort(key=lambda x: x["formatted"].lower())

        return formatted

    def _group_citations(
        self, citations: List[Dict], group_by: str
    ) -> Dict[str, List[Dict]]:
        """Group citations by specified field."""
        groups: Dict[str, List[Dict]] = {}

        for cite in citations:
            if group_by == "type":
                key = cite.get("type", "Other")
            elif group_by == "source":
                key = cite.get("source", "Unknown")
            else:
                key = "All Citations"

            if key not in groups:
                groups[key] = []

            groups[key].append(cite)

        return groups

    def _display_bibliography(
        self,
        formatted: Any,
        style: str,
        group_by: Optional[str],
    ) -> None:
        """Display bibliography."""
        self.console.print()
        self.console.print(f"[bold cyan]Bibliography ({style.upper()})[/bold cyan]\n")

        if isinstance(formatted, dict):
            # Grouped display
            for group_name, items in formatted.items():
                self.console.print(f"\n[bold yellow]{group_name}[/bold yellow]")
                for idx, item in enumerate(items, 1):
                    self.console.print(f"{idx}. {item['formatted']}")
        else:
            # Simple list display
            for idx, item in enumerate(formatted, 1):
                self.console.print(f"{idx}. {item['formatted']}")

        # Summary
        count = (
            sum(len(items) for items in formatted.values())
            if isinstance(formatted, dict)
            else len(formatted)
        )
        self.console.print(f"\n[dim]Total entries: {count}[/dim]")

    def _save_bibliography(
        self,
        output: Path,
        formatted: Any,
        style: str,
        group_by: Optional[str],
    ) -> None:
        """Save bibliography to file."""
        lines = []
        lines.append(f"# Bibliography ({style.upper()})\n")

        if isinstance(formatted, dict):
            # Grouped output
            for group_name, items in formatted.items():
                lines.append(f"\n## {group_name}\n")
                for idx, item in enumerate(items, 1):
                    lines.append(f"{idx}. {item['formatted']}\n")
        else:
            # Simple list output
            for idx, item in enumerate(formatted, 1):
                lines.append(f"{idx}. {item['formatted']}\n")

        # Write to file
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(lines), encoding="utf-8")

        self.print_success(f"Bibliography saved to: {output}")


def command(
    style: str = typer.Argument("apa", help="Citation style (apa, mla, chicago)"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
    group_by: Optional[str] = typer.Option(
        None, "--group-by", help="Group by 'type' or 'source'"
    ),
) -> None:
    """Generate complete bibliography from all citations.

    Examples:
        ingestforge citation bibliography apa
        ingestforge citation bibliography mla -o bibliography.md
        ingestforge citation bibliography chicago --group-by type
    """
    cmd = BibliographyCommand()
    exit_code = cmd.execute(style, project, output, group_by)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
