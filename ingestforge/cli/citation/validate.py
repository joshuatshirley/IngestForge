"""Validate command - Check citation completeness and formatting."""

from pathlib import Path
from typing import Optional, Dict, List, Any
import typer
from rich.table import Table
from rich.panel import Panel
from ingestforge.cli.citation.base import CitationCommand


class ValidateCommand(CitationCommand):
    """Validate citations for completeness and correctness."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        strict: bool = False,
    ) -> int:
        """Validate citations in knowledge base."""
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Get all chunks
            all_chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not all_chunks:
                self.print_warning("No content in knowledge base")
                return 0

            # Extract and validate citations
            citations = self._extract_citations(all_chunks)

            if not citations:
                self.print_warning("No citations found to validate")
                return 0

            # Perform validation
            validation_results = self._validate_citations(citations, strict)

            # Display results
            self._display_validation_results(validation_results)

            # Save if requested
            if output:
                self.save_json_output(
                    output,
                    validation_results,
                    f"Validation results saved to: {output}",
                )

            # Return error code if validation failed
            if validation_results["has_errors"]:
                return 1

            return 0

        except Exception as e:
            return self.handle_error(e, "Citation validation failed")

    def _extract_citations(self, chunks: list) -> List[Dict[str, Any]]:
        """Extract citations from chunks."""
        citations = []

        for chunk in chunks:
            metadata = self.extract_chunk_metadata(chunk)
            source = metadata.get("source", "Unknown")

            # Check for citation metadata
            if "citations" in metadata:
                for cite in metadata["citations"]:
                    citations.append(
                        {"source": source, "citation": cite, "metadata": metadata}
                    )

        return citations

    def _validate_citations(
        self, citations: List[Dict], strict: bool
    ) -> Dict[str, Any]:
        """Validate citation completeness."""
        results = {
            "total_citations": len(citations),
            "valid": [],
            "warnings": [],
            "errors": [],
            "has_errors": False,
        }

        for idx, cite_data in enumerate(citations):
            cite = cite_data["citation"]
            source = cite_data["source"]

            # Validate based on citation type
            validation = self._validate_single_citation(cite, source, strict)

            if validation["status"] == "valid":
                results["valid"].append(validation)
            elif validation["status"] == "warning":
                results["warnings"].append(validation)
            else:
                results["errors"].append(validation)
                results["has_errors"] = True

        return results

    def _validate_single_citation(
        self, cite: Any, source: str, strict: bool
    ) -> Dict[str, Any]:
        """Validate a single citation."""
        issues = []

        # Convert to string if needed
        cite_str = str(cite)

        # Required fields check
        has_author = any(
            indicator in cite_str.lower()
            for indicator in ["author", "et al", "ed.", "eds."]
        )
        has_year = any(char.isdigit() for char in cite_str)
        has_title = len(cite_str) > 10

        if not has_author:
            issues.append("Missing author information")
        if not has_year:
            issues.append("Missing year/date")
        if not has_title:
            issues.append("Citation too short, may be incomplete")

        # Determine status
        if not issues:
            status = "valid"
        elif strict:
            status = "error"
        else:
            status = "warning"

        return {
            "citation": cite_str[:100],
            "source": source,
            "status": status,
            "issues": issues,
        }

    def _display_validation_results(self, results: Dict) -> None:
        """Display validation results."""
        self.console.print()
        self.console.print("[bold cyan]Citation Validation Results[/bold cyan]\n")

        # Summary
        total = results["total_citations"]
        valid_count = len(results["valid"])
        warning_count = len(results["warnings"])
        error_count = len(results["errors"])

        summary = f"""Total Citations: {total}
✓ Valid: {valid_count}
⚠ Warnings: {warning_count}
✗ Errors: {error_count}"""

        color = "green" if error_count == 0 else "red"
        self.console.print(Panel(summary, border_style=color, title="Summary"))

        # Display errors
        if results["errors"]:
            self.console.print()
            self.console.print("[bold red]Errors:[/bold red]")
            self._display_validation_table(results["errors"], "red")

        # Display warnings
        if results["warnings"]:
            self.console.print()
            self.console.print("[bold yellow]Warnings:[/bold yellow]")
            self._display_validation_table(results["warnings"], "yellow")

    def _display_validation_table(self, items: List[Dict], color: str) -> None:
        """Display validation issues in table format."""
        table = Table(show_lines=True)
        table.add_column("Citation", width=40)
        table.add_column("Source", width=25)
        table.add_column("Issues", width=35)

        for item in items[:20]:  # Limit display
            citation = item.get("citation", "")
            source = item.get("source", "")
            issues = ", ".join(item.get("issues", []))

            table.add_row(citation, source, f"[{color}]{issues}[/{color}]")

        self.console.print(table)


def command(
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors"),
) -> None:
    """Validate citations for completeness and formatting.

    Examples:
        ingestforge citation validate
        ingestforge citation validate --strict
        ingestforge citation validate -o validation.json
    """
    cmd = ValidateCommand()
    exit_code = cmd.execute(project, output, strict)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
