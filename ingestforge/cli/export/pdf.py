"""PDF command - Export knowledge base to PDF format."""

from pathlib import Path
from typing import Optional
import typer
from ingestforge.cli.export.base import ExportCommand


class PDFCommand(ExportCommand):
    """Export knowledge base to PDF format."""

    def execute(
        self,
        output: Path,
        query: Optional[str] = None,
        limit: int = 100,
        project: Optional[Path] = None,
    ) -> int:
        """Export to PDF format."""
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Get chunks
            if query:
                chunks = self.search_context(ctx["storage"], query, k=limit)
            else:
                chunks = self.get_all_chunks_from_storage(ctx["storage"])
                chunks = chunks[:limit]

            if not chunks:
                self.print_warning("No content to export")
                return 0

            # Generate markdown content first
            markdown_content = self._generate_markdown_content(chunks)

            # Try to convert to PDF
            success = self._convert_to_pdf(markdown_content, output)

            if success:
                self.print_success(f"PDF exported to: {output}")
                return 0
            else:
                # Fallback: save as markdown with instructions
                md_path = output.with_suffix(".md")
                md_path.write_text(markdown_content, encoding="utf-8")
                self.print_warning(
                    f"PDF conversion not available. Saved as Markdown: {md_path}"
                )
                self.print_info(
                    "Install pandoc or wkhtmltopdf for PDF export: pip install pypandoc"
                )
                return 1

        except Exception as e:
            return self.handle_error(e, "PDF export failed")

    def _generate_markdown_content(self, chunks: list) -> str:
        """Generate markdown content from chunks."""
        lines = ["# Knowledge Base Export\n\n"]

        for idx, chunk in enumerate(chunks, 1):
            metadata = self.extract_chunk_metadata(chunk)
            text = getattr(chunk, "text", str(chunk))
            source = metadata.get("source", "Unknown")

            lines.append(f"## Chunk {idx}: {source}\n\n")
            lines.append(f"{text}\n\n")
            lines.append("---\n\n")

        return "".join(lines)

    def _convert_to_pdf(self, markdown: str, output: Path) -> bool:
        """Convert markdown to PDF."""
        # Try different conversion methods
        converters = [
            self._try_pypandoc,
            self._try_markdown2pdf,
            self._try_weasyprint,
        ]

        for converter in converters:
            if converter(markdown, output):
                return True

        return False

    def _try_pypandoc(self, markdown: str, output: Path) -> bool:
        """Try pypandoc for conversion."""
        try:
            import pypandoc  # type: ignore

            pypandoc.convert_text(
                markdown,
                "pdf",
                format="md",
                outputfile=str(output),
                extra_args=["--pdf-engine=xelatex"],
            )
            return True
        except (ImportError, Exception):
            return False

    def _try_markdown2pdf(self, markdown: str, output: Path) -> bool:
        """Try markdown2pdf for conversion."""
        try:
            from markdown2pdf import md2pdf  # type: ignore

            md2pdf(str(output), md_content=markdown)
            return True
        except (ImportError, Exception):
            return False

    def _try_weasyprint(self, markdown: str, output: Path) -> bool:
        """Try weasyprint for conversion."""
        try:
            import markdown as md  # type: ignore
            from weasyprint import HTML  # type: ignore

            html = md.markdown(markdown)
            HTML(string=html).write_pdf(str(output))
            return True
        except (ImportError, Exception):
            return False


def command(
    output: Path = typer.Argument(..., help="Output PDF file"),
    query: Optional[str] = typer.Option(None, "-q", help="Filter by query"),
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum chunks to export"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
) -> None:
    """Export knowledge base to PDF format.

    Note: Requires additional dependencies for PDF generation:
    - pypandoc (recommended): pip install pypandoc
    - OR markdown2pdf: pip install markdown2pdf
    - OR weasyprint: pip install weasyprint

    If PDF conversion is not available, exports as Markdown instead.

    Examples:
        ingestforge export pdf output.pdf
        ingestforge export pdf filtered.pdf -q "machine learning"
        ingestforge export pdf summary.pdf --limit 50
    """
    cmd = PDFCommand()
    exit_code = cmd.execute(output, query, limit, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
