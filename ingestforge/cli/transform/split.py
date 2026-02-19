"""Split command - Split and partition documents/chunks.

Splits documents into chunks and partitions chunk collections."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.table import Table

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.cli.transform.base import TransformCommand


class SplitBy(str, Enum):
    """Splitting/partitioning strategy options."""

    DOCUMENT = "document"
    TOPIC = "topic"
    SIZE = "size"
    SOURCE = "source"


@dataclass
class SplitResult:
    """Result of a split operation."""

    input_count: int = 0
    output_partitions: int = 0
    partition_sizes: Dict[str, int] = field(default_factory=dict)
    exported_files: List[Path] = field(default_factory=list)
    splits: Dict[str, List[ChunkRecord]] = field(default_factory=dict)


class ChunkSplitter:
    """Split and partition chunk collections.

    Supports splitting by document, topic, size, or source.
    Can export partitions to separate files.
    """

    def __init__(
        self,
        min_chunks: int = 1,
        max_chunks: int = 10000,
    ) -> None:
        """Initialize chunk splitter.

        Args:
            min_chunks: Minimum chunks per partition
            max_chunks: Maximum chunks per partition
        """
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks

    def split_by_document(
        self, chunks: List[ChunkRecord]
    ) -> Dict[str, List[ChunkRecord]]:
        """Split chunks by document.

        Args:
            chunks: Chunks to split

        Returns:
            Dictionary mapping document ID to chunks
        """
        result: Dict[str, List[ChunkRecord]] = defaultdict(list)

        for chunk in chunks:
            doc_id = chunk.document_id or "unknown"
            result[doc_id].append(chunk)

        return dict(result)

    def split_by_topic(
        self, chunks: List[ChunkRecord], min_chunks: int = 10
    ) -> Dict[str, List[ChunkRecord]]:
        """Split chunks by topic using simple keyword clustering.

        Uses concept/entity overlap for basic topic grouping.
        For advanced topic modeling, use the analysis module.

        Args:
            chunks: Chunks to split
            min_chunks: Minimum chunks per topic

        Returns:
            Dictionary mapping topic to chunks
        """
        # Group by dominant concept/entity
        topic_groups: Dict[str, List[ChunkRecord]] = defaultdict(list)

        for chunk in chunks:
            topic = self._extract_primary_topic(chunk)
            topic_groups[topic].append(chunk)

        # Merge small groups
        return self._merge_small_groups(topic_groups, min_chunks)

    def _extract_primary_topic(self, chunk: ChunkRecord) -> str:
        """Extract primary topic from chunk.

        Args:
            chunk: Chunk to analyze

        Returns:
            Topic string
        """
        # Use first concept if available
        if chunk.concepts:
            return chunk.concepts[0]

        # Use first entity if available
        if chunk.entities:
            return chunk.entities[0]

        # Use section title
        if chunk.section_title:
            return chunk.section_title

        # Default
        return "general"

    def _merge_small_groups(
        self,
        groups: Dict[str, List[ChunkRecord]],
        min_size: int,
    ) -> Dict[str, List[ChunkRecord]]:
        """Merge groups smaller than minimum size.

        Args:
            groups: Topic groups
            min_size: Minimum group size

        Returns:
            Merged groups
        """
        result: Dict[str, List[ChunkRecord]] = {}
        other_chunks: List[ChunkRecord] = []

        for topic, topic_chunks in groups.items():
            if len(topic_chunks) >= min_size:
                result[topic] = topic_chunks
            else:
                other_chunks.extend(topic_chunks)

        if other_chunks:
            result["other"] = other_chunks

        return result

    def split_by_size(
        self, chunks: List[ChunkRecord], max_per_split: int
    ) -> List[List[ChunkRecord]]:
        """Split chunks into groups of maximum size.

        Args:
            chunks: Chunks to split
            max_per_split: Maximum chunks per group

        Returns:
            List of chunk groups
        """
        if not chunks:
            return []

        result: List[List[ChunkRecord]] = []
        current_group: List[ChunkRecord] = []

        for chunk in chunks:
            current_group.append(chunk)

            if len(current_group) >= max_per_split:
                result.append(current_group)
                current_group = []

        if current_group:
            result.append(current_group)

        return result

    def split_by_source(
        self, chunks: List[ChunkRecord]
    ) -> Dict[str, List[ChunkRecord]]:
        """Split chunks by source file.

        Args:
            chunks: Chunks to split

        Returns:
            Dictionary mapping source to chunks
        """
        result: Dict[str, List[ChunkRecord]] = defaultdict(list)

        for chunk in chunks:
            source = chunk.source_file or "unknown"
            # Use just the filename
            source_name = Path(source).stem if source != "unknown" else source
            result[source_name].append(chunk)

        return dict(result)

    def export_splits(
        self,
        splits: Dict[str, List[ChunkRecord]],
        output_dir: Path,
        pattern: str = "{name}",
    ) -> List[Path]:
        """Export splits to separate files.

        Args:
            splits: Dictionary of splits
            output_dir: Output directory
            pattern: Filename pattern with {name} and {n} placeholders

        Returns:
            List of created file paths
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        created_files: List[Path] = []

        for idx, (name, chunk_list) in enumerate(splits.items()):
            # Create filename from pattern
            safe_name = self._sanitize_filename(name)
            filename = pattern.replace("{name}", safe_name).replace("{n}", str(idx))

            if not filename.endswith(".json") and not filename.endswith(".jsonl"):
                filename += ".jsonl"

            filepath = output_dir / filename
            self._save_chunks_to_file(filepath, chunk_list)
            created_files.append(filepath)

        return created_files

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string for use as filename.

        Args:
            name: Original name

        Returns:
            Sanitized filename
        """
        import re

        # Replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = sanitized.strip("_")

        return sanitized[:100] if len(sanitized) > 100 else sanitized

    def _save_chunks_to_file(self, filepath: Path, chunks: List[ChunkRecord]) -> None:
        """Save chunks to file.

        Args:
            filepath: Output path
            chunks: Chunks to save
        """
        import json

        data = [chunk.to_dict() for chunk in chunks]

        if filepath.suffix == ".jsonl":
            with filepath.open("w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            filepath.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )


class SplitCommand(TransformCommand):
    """Split documents into chunks or partition chunk collections."""

    def execute(
        self,
        source: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        by: str = "size",
        output_pattern: str = "{name}",
        chunk_size: int = 1000,
        overlap: int = 100,
        max_chunks: int = 1000,
        min_chunks: int = 10,
    ) -> int:
        """Execute split operation.

        Args:
            source: Source file or library
            project: Project directory
            output: Output file or directory
            by: Split strategy (document, topic, size, source)
            output_pattern: Pattern for output filenames
            chunk_size: Chunk size for text splitting
            overlap: Overlap for text splitting
            max_chunks: Maximum chunks per partition
            min_chunks: Minimum chunks per partition

        Returns:
            0 on success, 1 on error
        """
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Determine if splitting text or partitioning chunks
            source_path = Path(source)
            is_text_file = self._is_text_file(source_path)

            if is_text_file:
                return self._split_text_file(source_path, output, chunk_size, overlap)

            # Partition existing chunks
            return self._partition_chunks(
                ctx, source, output, by, output_pattern, max_chunks, min_chunks
            )

        except Exception as e:
            return self.handle_error(e, "Split operation failed")

    def _is_text_file(self, path: Path) -> bool:
        """Check if path is a plain text file.

        Args:
            path: File path

        Returns:
            True if text file
        """
        text_extensions = {".txt", ".md", ".rst", ".html", ".htm"}
        return path.exists() and path.suffix.lower() in text_extensions

    def _split_text_file(
        self,
        source_path: Path,
        output: Optional[Path],
        chunk_size: int,
        overlap: int,
    ) -> int:
        """Split a text file into chunks.

        Args:
            source_path: Source file path
            output: Output path
            chunk_size: Chunk size
            overlap: Overlap size

        Returns:
            Exit code
        """
        # Validate
        self.validate_file_path(source_path, must_exist=True)
        self.validate_chunk_size(chunk_size)
        self.validate_overlap(overlap, chunk_size)

        # Read and split
        content = source_path.read_text(encoding="utf-8")
        content = self.clean_text_simple(content)
        content = self.normalize_whitespace(content)

        chunks = self.split_into_chunks(content, chunk_size, overlap)

        # Create chunk data
        chunk_data = []
        for idx, chunk_text in enumerate(chunks):
            metadata = self.extract_metadata_simple(chunk_text)
            chunk_data.append(
                {
                    "index": idx,
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

        # Display results
        self._display_text_split_results(source_path, content, chunk_data)

        # Save
        if output:
            self._save_text_chunks(output, chunk_data)

        return 0

    def _partition_chunks(
        self,
        ctx: Dict[str, Any],
        source: str,
        output: Optional[Path],
        by: str,
        output_pattern: str,
        max_chunks: int,
        min_chunks: int,
    ) -> int:
        """Partition existing chunk collection.

        Args:
            ctx: Command context
            source: Source library or file
            output: Output directory
            by: Split strategy
            output_pattern: Output filename pattern
            max_chunks: Maximum chunks per partition
            min_chunks: Minimum chunks per partition

        Returns:
            Exit code
        """
        chunks = self._load_chunks(ctx, source)

        if not chunks:
            self.print_warning(f"No chunks found in source: {source}")
            return 0

        splitter = ChunkSplitter(min_chunks=min_chunks, max_chunks=max_chunks)
        result = self._perform_split(splitter, chunks, by, max_chunks, min_chunks)

        # Display results
        self._display_partition_results(result)

        # Export if output specified
        if output:
            output_dir = (
                output if output.is_dir() or not output.suffix else output.parent
            )
            result.exported_files = splitter.export_splits(
                result.splits, output_dir, output_pattern
            )
            self.print_success(
                f"Exported {len(result.exported_files)} files to: {output_dir}"
            )

        return 0

    def _perform_split(
        self,
        splitter: ChunkSplitter,
        chunks: List[ChunkRecord],
        by: str,
        max_chunks: int,
        min_chunks: int,
    ) -> SplitResult:
        """Perform the split operation.

        Args:
            splitter: ChunkSplitter instance
            chunks: Chunks to split
            by: Split strategy
            max_chunks: Maximum per partition
            min_chunks: Minimum per partition

        Returns:
            SplitResult
        """
        result = SplitResult(input_count=len(chunks))

        # Dispatch to appropriate method
        handlers = {
            "document": lambda: splitter.split_by_document(chunks),
            "topic": lambda: splitter.split_by_topic(chunks, min_chunks),
            "source": lambda: splitter.split_by_source(chunks),
        }

        if by == "size":
            size_splits = splitter.split_by_size(chunks, max_chunks)
            result.splits = {f"partition_{i}": s for i, s in enumerate(size_splits)}
        elif by in handlers:
            result.splits = handlers[by]()
        else:
            # Default: by document
            result.splits = splitter.split_by_document(chunks)

        result.output_partitions = len(result.splits)
        result.partition_sizes = {k: len(v) for k, v in result.splits.items()}

        return result

    def _load_chunks(self, ctx: Dict[str, Any], source: str) -> List[ChunkRecord]:
        """Load chunks from source.

        Args:
            ctx: Command context
            source: Library name or file path

        Returns:
            List of chunks
        """
        storage = ctx.get("storage")

        # Check if source is a library
        if storage:
            libraries = storage.get_libraries()
            if source in libraries:
                results = storage.search("", top_k=10000, library_filter=source)
                return [
                    storage.get_chunk(r.chunk_id)
                    for r in results
                    if storage.get_chunk(r.chunk_id)
                ]

        # Check if source is a file
        source_path = Path(source)
        if source_path.exists() and source_path.suffix in {".json", ".jsonl"}:
            return self._load_chunks_from_file(source_path)

        return []

    def _load_chunks_from_file(self, path: Path) -> List[ChunkRecord]:
        """Load chunks from JSON/JSONL file.

        Rule #1: Max 3 nesting levels via helper extraction.

        Args:
            path: File path

        Returns:
            List of chunks
        """
        import json

        chunks = []
        try:
            if path.suffix == ".jsonl":
                chunks = self._read_jsonl_chunks(path)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    chunks = [ChunkRecord.from_dict(item) for item in data]
        except Exception as e:
            self.print_error(f"Error loading file: {e}")

        return chunks

    def _read_jsonl_chunks(self, path: Path) -> List[ChunkRecord]:
        """Read chunks from JSONL file.

        Rule #1: Extracted to reduce nesting in _load_chunks_from_file.

        Args:
            path: JSONL file path

        Returns:
            List of chunks
        """
        import json

        chunks = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                chunks.append(ChunkRecord.from_dict(data))
        return chunks

    def _display_text_split_results(
        self, source_path: Path, content: str, chunk_data: List[Dict]
    ) -> None:
        """Display text split results.

        Args:
            source_path: Source file path
            content: Original content
            chunk_data: Split chunk data
        """
        results = {
            "input_count": 1,
            "output_count": len(chunk_data),
            "successful": 1,
            "failed": 0,
            "errors": [],
            "details": [
                f"File: {source_path.name}",
                f"Original size: {len(content)} chars",
                f"Chunks created: {len(chunk_data)}",
            ],
        }
        summary = self.create_transform_summary(results, "Split")
        self.console.print(summary)

    def _display_partition_results(self, result: SplitResult) -> None:
        """Display partition results.

        Args:
            result: Split result
        """
        self.console.print()

        table = Table(title="Partition Results")
        table.add_column("Partition", style="cyan")
        table.add_column("Chunks", style="green")

        for name, count in result.partition_sizes.items():
            display_name = name if len(name) < 40 else name[:37] + "..."
            table.add_row(display_name, str(count))

        self.console.print(table)

        self.console.print()
        self.print_info(f"Input chunks: {result.input_count}")
        self.print_info(f"Partitions created: {result.output_partitions}")

    def _save_text_chunks(self, output: Path, chunk_data: List[Dict]) -> None:
        """Save text chunks to file.

        Args:
            output: Output path
            chunk_data: Chunk data to save
        """
        import json

        try:
            with output.open("w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            self.print_success(f"Chunks saved: {output}")
        except Exception as e:
            self.print_error(f"Failed to save chunks: {e}")


def command(
    source: str = typer.Argument(..., help="Source file, library, or chunk file"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file or directory"
    ),
    by: str = typer.Option(
        "size", "--by", "-b", help="Split strategy: document, topic, size, source"
    ),
    output_pattern: str = typer.Option(
        "{name}",
        "--output-pattern",
        help="Output filename pattern with {name} and {n} placeholders",
    ),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", "-c", help="Chunk size for text splitting"
    ),
    overlap: int = typer.Option(100, "--overlap", help="Overlap for text splitting"),
    max_chunks: int = typer.Option(
        1000, "--max-chunks", help="Maximum chunks per partition"
    ),
    min_chunks: int = typer.Option(
        10, "--min-chunks", help="Minimum chunks per partition"
    ),
) -> None:
    """Split documents or partition chunk collections.

    This command operates in two modes:
    1. Text splitting: Split text files into chunks
    2. Chunk partitioning: Split existing chunk collections

    Split Strategies (for partitioning):
    - document: Group by document ID
    - topic: Group by topics/concepts
    - size: Equal-sized partitions
    - source: Group by source file

    Examples:
        # Split text file into chunks
        ingestforge transform split document.txt -c 1000 -o chunks.json

        # Partition by document
        ingestforge transform split docs --by document -o partitions/

        # Partition by topic
        ingestforge transform split library --by topic --min-chunks 10 -o topics/

        # Partition by size
        ingestforge transform split chunks.jsonl --by size --max-chunks 1000 -o split_{n}.jsonl

        # Partition by source file
        ingestforge transform split library --by source -o sources/
    """
    cmd = SplitCommand()
    exit_code = cmd.execute(
        source,
        project,
        output,
        by,
        output_pattern,
        chunk_size,
        overlap,
        max_chunks,
        min_chunks,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
