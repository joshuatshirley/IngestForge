"""Context command - Export context for RAG applications."""

from pathlib import Path
from typing import Any, Optional
import typer
from ingestforge.cli.export.base import ExportCommand


class ContextCommand(ExportCommand):
    """Export retrieved context for RAG applications."""

    def execute(
        self,
        query: str,
        output: Path,
        k: int = 10,
        project: Optional[Path] = None,
        include_metadata: bool = True,
    ) -> int:
        """Export context for a query."""
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Search for relevant chunks
            chunks = self.search_context(ctx["storage"], query, k=k)

            if not chunks:
                self.print_warning(f"No context found for query: {query}")
                return 0

            # Build context data
            context_data = self._build_context_data(chunks, query, include_metadata)

            # Save context
            self._save_context(output, context_data)

            return 0

        except Exception as e:
            return self.handle_error(e, "Context export failed")

    def _build_context_data(
        self, chunks: list, query: str, include_metadata: bool
    ) -> dict[str, Any]:
        """Build context data structure."""
        context_items = []

        for idx, chunk in enumerate(chunks):
            metadata = self.extract_chunk_metadata(chunk)
            text = getattr(chunk, "text", str(chunk))

            item = {
                "rank": idx + 1,
                "text": text,
            }

            if include_metadata:
                item["metadata"] = {
                    "source": metadata.get("source", "Unknown"),
                    "type": metadata.get("type", "document"),
                }

                # Include relevance score if available
                if hasattr(chunk, "score"):
                    item["relevance_score"] = float(chunk.score)

            context_items.append(item)

        return {
            "query": query,
            "context": context_items,
            "total_chunks": len(context_items),
        }

    def _save_context(self, output: Path, context_data: dict) -> None:
        """Save context to file."""
        output.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        self.save_json_output(
            output,
            context_data,
            f"Context exported to: {output}",
        )

        # Display summary
        self.print_info(f"Query: {context_data['query']}")
        self.print_info(f"Chunks: {context_data['total_chunks']}")


def command(
    query: str = typer.Argument(..., help="Query to retrieve context for"),
    output: Path = typer.Option(..., "-o", help="Output file"),
    k: int = typer.Option(10, "--top-k", "-k", help="Number of chunks to retrieve"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    include_metadata: bool = typer.Option(
        True, "--metadata/--no-metadata", help="Include metadata"
    ),
) -> None:
    """Export context for RAG applications.

    Retrieves relevant chunks for a query and exports them in a format
    suitable for use in RAG (Retrieval-Augmented Generation) pipelines.

    Examples:
        ingestforge export context "machine learning" -o context.json
        ingestforge export context "Python tutorials" -k 20 -o context.json
        ingestforge export context "AI ethics" --no-metadata -o context.json
    """
    cmd = ContextCommand()
    exit_code = cmd.execute(query, output, k, project, include_metadata)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
