"""JSON export command - Export knowledge base to JSON.

Exports chunks to structured JSON format for programmatic access.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import typer

from ingestforge.cli.export.base import ExportCommand
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class JSONExportCommand(ExportCommand):
    """Export knowledge base to JSON format."""

    def execute(
        self,
        output: Path,
        project: Optional[Path] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        pretty: bool = True,
    ) -> int:
        """Export knowledge base to JSON.

        Args:
            output: Output JSON file
            project: Project directory
            query: Optional search query to filter chunks
            limit: Optional limit on number of chunks
            pretty: Use pretty formatting (indented)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate parameters (Commandment #7)
            self.validate_output_path(output)

            if limit is not None and limit < 1:
                raise typer.BadParameter("Limit must be positive")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Retrieve chunks
            chunks = self.search_filtered_chunks(ctx["storage"], query, limit)

            if not chunks:
                self._handle_no_chunks(query)
                return 0

            # Convert to JSON structure
            json_data = self._convert_to_json(chunks, query)

            # Save to file
            self._save_json_file(output, json_data, pretty)

            # Display summary
            self._display_summary(len(chunks), output)

            return 0

        except Exception as e:
            return self.handle_error(e, "JSON export failed")

    def _handle_no_chunks(self, query: Optional[str]) -> None:
        """Handle case where no chunks found.

        Args:
            query: Optional search query
        """
        if query:
            self.print_warning(f"No chunks found matching: '{query}'")
        else:
            self.print_warning("Knowledge base is empty")

        self.print_info("Try ingesting some documents first")

    def _convert_to_json(self, chunks: list, query: Optional[str]) -> Dict[str, Any]:
        """Convert chunks to JSON structure.

        Args:
            chunks: List of chunks
            query: Optional search query

        Returns:
            JSON-serializable dictionary
        """
        from datetime import datetime

        timestamp = datetime.now().isoformat()

        json_data = {
            "metadata": {
                "generated": timestamp,
                "total_chunks": len(chunks),
                "filter_query": query,
            },
            "chunks": self._convert_chunks(chunks),
        }

        return json_data

    def _convert_chunks(self, chunks: list) -> List[Dict[str, Any]]:
        """Convert chunks to JSON-serializable format.

        Args:
            chunks: List of chunks

        Returns:
            List of chunk dictionaries
        """
        json_chunks = []

        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "id": idx,
                "text": self.extract_chunk_text(chunk),
                "metadata": self._serialize_metadata(
                    self.extract_chunk_metadata(chunk)
                ),
            }

            json_chunks.append(chunk_data)

        return json_chunks

    def _serialize_primitive(self, value: Any) -> bool:
        """
        Check if value is a JSON-serializable primitive.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            value: Value to check

        Returns:
            True if primitive type
        """
        return isinstance(value, (str, int, float, bool, type(None)))

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize a single value to JSON-compatible type.

        Rule #1: Early returns eliminate nested if/elif chain
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            value: Value to serialize

        Returns:
            Serialized value
        """
        if self._serialize_primitive(value):
            return value
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, dict):
            return self._serialize_metadata(value)
        # Convert to string for complex types (datetime, Path, etc.)
        return str(value)

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize metadata to JSON-compatible types.

        Rule #1: Zero nesting - helper extracts type checking logic
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            metadata: Metadata dictionary

        Returns:
            Serialized metadata dictionary
        """
        assert metadata is not None, "Metadata cannot be None"
        assert isinstance(metadata, dict), "Metadata must be dictionary"
        MAX_ITEMS: int = 10_000  # Prevent infinite recursion
        items_processed: int = 0

        serialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            items_processed += 1
            if items_processed > MAX_ITEMS:
                logger.warning(f"Safety limit: processed {MAX_ITEMS} metadata items")
                break
            serialized[key] = self._serialize_value(value)
        assert isinstance(serialized, dict), "Result must be dictionary"

        return serialized

    def _save_json_file(self, output: Path, data: Dict[str, Any], pretty: bool) -> None:
        """Save JSON data to file.

        Args:
            output: Output file path
            data: JSON data
            pretty: Use pretty formatting
        """
        indent = 2 if pretty else None

        json_content = json.dumps(data, indent=indent, ensure_ascii=False, default=str)

        self.save_export_file(output, json_content, encoding="utf-8")

    def _display_summary(self, chunk_count: int, output: Path) -> None:
        """Display export summary.

        Args:
            chunk_count: Number of chunks exported
            output: Output file path
        """
        self.console.print()
        self.print_info(f"Exported {chunk_count} chunks")
        self.print_info(f"Output file: {output}")

        # Show file size
        try:
            file_size = output.stat().st_size
            size_mb = file_size / (1024 * 1024)
            self.print_info(f"File size: {size_mb:.2f} MB")
        except Exception as e:
            logger.debug(f"Failed to get file size for {output}: {e}")


# Typer command wrapper
def command(
    output: Path = typer.Argument(..., help="Output JSON file"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Filter chunks by search query"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of chunks"
    ),
    compact: bool = typer.Option(
        False, "--compact", help="Use compact formatting (no indentation)"
    ),
) -> None:
    """Export knowledge base to JSON format.

    Creates a structured JSON file containing your knowledge
    base chunks with full metadata.

    Perfect for:
    - Programmatic access to data
    - Data analysis and processing
    - Integration with other tools
    - Backup and archival

    Examples:
        # Export entire knowledge base
        ingestforge export json output.json

        # Export filtered chunks
        ingestforge export json ml_docs.json --query "machine learning"

        # Limit number of chunks
        ingestforge export json sample.json --limit 100

        # Compact format (smaller file size)
        ingestforge export json compact.json --compact

        # Specific project
        ingestforge export json docs.json -p /path/to/project
    """
    cmd = JSONExportCommand()
    pretty = not compact
    exit_code = cmd.execute(output, project, query, limit, pretty)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
