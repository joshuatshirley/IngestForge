"""Base class for export commands.

Provides common functionality for exporting content.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class ExportCommand(IngestForgeCommand):
    """Base class for export commands."""

    def get_all_chunks_from_storage(self, storage: Any) -> list[Any]:
        """Retrieve all chunks from storage.

        Args:
            storage: ChunkRepository instance

        Returns:
            List of all chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: self._retrieve_all_chunks(storage),
            "Retrieving chunks from storage...",
            "Chunks retrieved",
        )

    def _retrieve_all_chunks(self, storage: Any) -> list[Any]:
        """Retrieve all chunks (internal helper).

        Args:
            storage: ChunkRepository instance

        Returns:
            List of chunks
        """
        # Different storage backends have different APIs
        if hasattr(storage, "get_all_chunks"):
            return storage.get_all_chunks()
        elif hasattr(storage, "list_all"):
            return storage.list_all()
        else:
            # Fallback: search with empty query
            return storage.search("", k=10000)

    def search_filtered_chunks(
        self, storage: Any, query: Optional[str], limit: Optional[int]
    ) -> list[Any]:
        """Search for filtered chunks.

        Args:
            storage: ChunkRepository instance
            query: Optional search query for filtering
            limit: Optional limit on number of chunks

        Returns:
            List of chunks
        """
        if query:
            # Search with query
            k = limit if limit else 1000
            return ProgressManager.run_with_spinner(
                lambda: storage.search(query, k=k),
                f"Searching for chunks matching '{query}'...",
                "Search complete",
            )
        else:
            # Get all chunks
            chunks = self.get_all_chunks_from_storage(storage)

            # Apply limit if specified
            if limit and len(chunks) > limit:
                return chunks[:limit]

            return chunks

    def extract_chunk_text(self, chunk: Any) -> str:
        """Extract text from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Chunk text content
        """
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        elif hasattr(chunk, "text"):
            return chunk.text
        else:
            return str(chunk)

    def extract_chunk_metadata(self, chunk: Any) -> Dict[str, Any]:
        """Extract metadata from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Metadata dictionary
        """
        if isinstance(chunk, dict):
            return chunk.get("metadata", {})
        elif hasattr(chunk, "metadata"):
            metadata = chunk.metadata
            if isinstance(metadata, dict):
                return metadata
            else:
                # Convert object to dict
                return vars(metadata) if metadata else {}
        else:
            return {}

    def group_chunks_by_source(self, chunks: list) -> Dict[str, List[Any]]:
        """Group chunks by source document.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary mapping source to list of chunks
        """
        grouped: Dict[str, List[Any]] = {}

        for chunk in chunks:
            metadata = self.extract_chunk_metadata(chunk)
            source = metadata.get("source", "unknown")

            if source not in grouped:
                grouped[source] = []

            grouped[source].append(chunk)

        return grouped

    def validate_output_path(self, output: Path) -> None:
        """Validate output file path.

        Args:
            output: Output file path

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        # Check if parent directory exists
        if not output.parent.exists():
            raise typer.BadParameter(
                f"Parent directory does not exist: {output.parent}"
            )

        # Check if file already exists (warn but allow)
        if output.exists():
            self.print_warning(f"File already exists and will be overwritten: {output}")

    def save_export_file(
        self, output: Path, content: str, encoding: str = "utf-8"
    ) -> None:
        """Save exported content to file.

        Args:
            output: Output file path
            content: Content to save
            encoding: File encoding
        """
        try:
            output.write_text(content, encoding=encoding)
            self.print_success(f"Exported to: {output}")

        except Exception as e:
            self.print_error(f"Failed to save file: {e}")
            raise
