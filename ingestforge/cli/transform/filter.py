"""Filter command - Filter chunks by criteria.

Filters chunks based on various criteria like length, entities, topics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.table import Table

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.cli.transform.base import TransformCommand


@dataclass
class FilterResult:
    """Result of a filter operation."""

    original_count: int
    filtered_count: int
    removed_count: int
    filters_applied: List[str] = field(default_factory=list)
    chunks: List[ChunkRecord] = field(default_factory=list)

    @property
    def retention_rate(self) -> float:
        """Calculate retention rate as percentage."""
        if self.original_count == 0:
            return 0.0
        return (self.filtered_count / self.original_count) * 100


class ChunkFilter:
    """Filter chunks by various criteria.

    Supports combining multiple filters with AND logic.
    All filter methods return filtered lists, enabling method chaining.
    """

    def __init__(self) -> None:
        """Initialize chunk filter."""
        self._filters: List[Callable[[List[ChunkRecord]], List[ChunkRecord]]] = []
        self._filter_names: List[str] = []

    def add_length_filter(
        self, min_len: Optional[int] = None, max_len: Optional[int] = None
    ) -> "ChunkFilter":
        """Add length filter.

        Args:
            min_len: Minimum character length (inclusive)
            max_len: Maximum character length (inclusive)

        Returns:
            Self for method chaining
        """
        if min_len is None and max_len is None:
            return self

        def length_filter(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
            return self.filter_by_length(chunks, min_len, max_len)

        self._filters.append(length_filter)
        self._filter_names.append(f"length({min_len}-{max_len})")
        return self

    def add_entity_filter(
        self, entity_type: Optional[str] = None, has_entities: bool = False
    ) -> "ChunkFilter":
        """Add entity filter.

        Args:
            entity_type: Specific entity type to require
            has_entities: If True, only keep chunks with any entities

        Returns:
            Self for method chaining
        """
        if not entity_type and not has_entities:
            return self

        def entity_filter(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
            return self.filter_by_entity(chunks, entity_type, has_entities)

        self._filters.append(entity_filter)
        name = f"entity({entity_type})" if entity_type else "has_entities"
        self._filter_names.append(name)
        return self

    def add_topic_filter(self, topic: str, threshold: float = 0.5) -> "ChunkFilter":
        """Add topic filter.

        Args:
            topic: Topic keyword or phrase to search for
            threshold: Relevance threshold (0-1)

        Returns:
            Self for method chaining
        """
        if not topic:
            return self

        def topic_filter(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
            return self.filter_by_topic(chunks, topic, threshold)

        self._filters.append(topic_filter)
        self._filter_names.append(f"topic({topic})")
        return self

    def add_source_filter(self, pattern: str) -> "ChunkFilter":
        """Add source file pattern filter.

        Args:
            pattern: Regex pattern for source file matching

        Returns:
            Self for method chaining
        """
        if not pattern:
            return self

        def source_filter(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
            return self.filter_by_source(chunks, pattern)

        self._filters.append(source_filter)
        self._filter_names.append(f"source({pattern})")
        return self

    def add_date_filter(
        self, start: Optional[date] = None, end: Optional[date] = None
    ) -> "ChunkFilter":
        """Add date range filter.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Self for method chaining
        """
        if start is None and end is None:
            return self

        def date_filter(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
            return self.filter_by_date(chunks, start, end)

        self._filters.append(date_filter)
        self._filter_names.append(f"date({start}-{end})")
        return self

    def add_quality_filter(self, min_score: float = 0.0) -> "ChunkFilter":
        """Add quality score filter.

        Args:
            min_score: Minimum quality score (0-1)

        Returns:
            Self for method chaining
        """
        if min_score <= 0:
            return self

        def quality_filter(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
            return self.filter_by_quality(chunks, min_score)

        self._filters.append(quality_filter)
        self._filter_names.append(f"quality(>={min_score})")
        return self

    def apply(self, chunks: List[ChunkRecord]) -> FilterResult:
        """Apply all configured filters to chunks.

        Args:
            chunks: List of chunks to filter

        Returns:
            FilterResult with filtered chunks and statistics
        """
        original_count = len(chunks)
        filtered = chunks

        for filter_func in self._filters:
            filtered = filter_func(filtered)

        return FilterResult(
            original_count=original_count,
            filtered_count=len(filtered),
            removed_count=original_count - len(filtered),
            filters_applied=self._filter_names.copy(),
            chunks=filtered,
        )

    def filter_by_length(
        self,
        chunks: List[ChunkRecord],
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
    ) -> List[ChunkRecord]:
        """Filter chunks by character length.

        Args:
            chunks: Chunks to filter
            min_len: Minimum length (inclusive)
            max_len: Maximum length (inclusive)

        Returns:
            Filtered chunks
        """
        result: List[ChunkRecord] = []

        for chunk in chunks:
            char_count = chunk.char_count or len(chunk.content)

            if min_len is not None and char_count < min_len:
                continue
            if max_len is not None and char_count > max_len:
                continue

            result.append(chunk)

        return result

    def filter_by_entity(
        self,
        chunks: List[ChunkRecord],
        entity_type: Optional[str] = None,
        has_entities: bool = False,
    ) -> List[ChunkRecord]:
        """Filter chunks by entity presence.

        Args:
            chunks: Chunks to filter
            entity_type: Required entity type (e.g., PERSON, ORG)
            has_entities: If True, only keep chunks with any entities

        Returns:
            Filtered chunks
        """
        result: List[ChunkRecord] = []

        for chunk in chunks:
            entities = chunk.entities or []

            # Check has_entities condition
            if has_entities and not entities:
                continue

            # Check specific entity type
            if entity_type and not self._has_entity_type(entities, entity_type):
                continue

            result.append(chunk)

        return result

    def _has_entity_type(self, entities: List[str], entity_type: str) -> bool:
        """Check if entities list contains specified type.

        Args:
            entities: List of entity strings
            entity_type: Type to search for

        Returns:
            True if entity type found
        """
        type_upper = entity_type.upper()
        for entity in entities:
            if type_upper in entity.upper():
                return True
        return False

    def filter_by_topic(
        self,
        chunks: List[ChunkRecord],
        topic: str,
        threshold: float = 0.5,
    ) -> List[ChunkRecord]:
        """Filter chunks by topic relevance.

        Uses keyword matching to estimate topic relevance.
        For semantic filtering, use the semantic retriever instead.

        Args:
            chunks: Chunks to filter
            topic: Topic keyword or phrase
            threshold: Minimum relevance score (0-1)

        Returns:
            Filtered chunks
        """
        result: List[ChunkRecord] = []
        topic_words = set(topic.lower().split())

        for chunk in chunks:
            relevance = self._calculate_topic_relevance(chunk.content, topic_words)
            if relevance >= threshold:
                result.append(chunk)

        return result

    def _calculate_topic_relevance(self, content: str, topic_words: set) -> float:
        """Calculate topic relevance score.

        Args:
            content: Chunk content
            topic_words: Set of topic keywords

        Returns:
            Relevance score (0-1)
        """
        if not topic_words:
            return 0.0

        content_lower = content.lower()
        matches = sum(1 for word in topic_words if word in content_lower)

        return matches / len(topic_words)

    def filter_by_source(
        self, chunks: List[ChunkRecord], pattern: str
    ) -> List[ChunkRecord]:
        """Filter chunks by source file pattern.

        Args:
            chunks: Chunks to filter
            pattern: Regex pattern to match source files

        Returns:
            Filtered chunks
        """
        result: List[ChunkRecord] = []
        compiled = re.compile(pattern, re.IGNORECASE)

        for chunk in chunks:
            source = chunk.source_file or ""
            if compiled.search(source):
                result.append(chunk)

        return result

    def filter_by_date(
        self,
        chunks: List[ChunkRecord],
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[ChunkRecord]:
        """Filter chunks by ingestion date.

        Args:
            chunks: Chunks to filter
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Filtered chunks
        """
        result: List[ChunkRecord] = []

        for chunk in chunks:
            chunk_date = self._parse_chunk_date(chunk.ingested_at)
            if chunk_date is None:
                continue

            if start is not None and chunk_date < start:
                continue
            if end is not None and chunk_date > end:
                continue

            result.append(chunk)

        return result

    def _parse_chunk_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse chunk date string to date object.

        Args:
            date_str: Date string in various formats

        Returns:
            Date object or None if parsing fails
        """
        if not date_str:
            return None

        # Try ISO format first, then common formats
        formats_to_try = [None, "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"]
        for fmt in formats_to_try:
            try:
                if fmt is None:
                    # ISO format
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    return dt.date()
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        return None

    def filter_by_quality(
        self, chunks: List[ChunkRecord], min_score: float
    ) -> List[ChunkRecord]:
        """Filter chunks by quality score.

        Args:
            chunks: Chunks to filter
            min_score: Minimum quality score (0-1)

        Returns:
            Filtered chunks
        """
        result: List[ChunkRecord] = []

        for chunk in chunks:
            if chunk.quality_score >= min_score:
                result.append(chunk)

        return result


class FilterCommand(TransformCommand):
    """Filter chunks by various criteria."""

    def execute(
        self,
        source: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        has_entities: bool = False,
        entity_type: Optional[str] = None,
        topic: Optional[str] = None,
        threshold: float = 0.5,
        source_pattern: Optional[str] = None,
        min_quality: float = 0.0,
    ) -> int:
        """Execute filter operation.

        Args:
            source: Source library or file
            project: Project directory
            output: Output file for filtered chunks
            min_length: Minimum chunk length
            max_length: Maximum chunk length
            has_entities: Only keep chunks with entities
            entity_type: Required entity type
            topic: Topic to filter by
            threshold: Topic relevance threshold
            source_pattern: Source file pattern
            min_quality: Minimum quality score

        Returns:
            0 on success, 1 on error
        """
        try:
            ctx = self.initialize_context(project, require_storage=True)
            chunks = self._load_chunks(ctx, source)

            if not chunks:
                self.print_warning(f"No chunks found in source: {source}")
                return 0

            # Build and apply filter
            filter_obj = self._build_filter(
                min_length,
                max_length,
                has_entities,
                entity_type,
                topic,
                threshold,
                source_pattern,
                min_quality,
            )
            result = filter_obj.apply(chunks)

            # Display results
            self._display_results(result)

            # Save output if specified
            if output:
                self._save_filtered_chunks(output, result.chunks)

            return 0

        except Exception as e:
            return self.handle_error(e, "Filter operation failed")

    def _load_chunks(self, ctx: Dict[str, Any], source: str) -> List[ChunkRecord]:
        """Load chunks from source.

        Args:
            ctx: Command context with storage
            source: Library name or path

        Returns:
            List of chunks
        """
        storage = ctx.get("storage")
        if storage is None:
            self.print_error("Storage not initialized")
            return []

        # Check if source is a library name
        libraries = storage.get_libraries()
        if source in libraries:
            return self._load_from_library(storage, source)

        # Try loading from file
        source_path = Path(source)
        if source_path.exists():
            return self._load_from_file(source_path)

        self.print_warning(f"Source not found: {source}")
        return []

    def _load_from_library(self, storage: Any, library: str) -> List[ChunkRecord]:
        """Load all chunks from a library.

        Args:
            storage: Storage backend
            library: Library name

        Returns:
            List of chunks
        """
        # Get all chunks from library via search
        results = storage.search("", top_k=10000, library_filter=library)
        chunks = []

        for result in results:
            chunk = storage.get_chunk(result.chunk_id)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _load_from_file(self, path: Path) -> List[ChunkRecord]:
        """Load chunks from JSON/JSONL file.

        Rule #1: Max 3 nesting levels via early return and helper.

        Args:
            path: File path

        Returns:
            List of chunks
        """
        import json

        chunks = []
        try:
            if path.suffix == ".jsonl":
                chunks = self._load_jsonl_file(path)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    chunks = [ChunkRecord.from_dict(item) for item in data]
        except Exception as e:
            self.print_error(f"Error loading file: {e}")

        return chunks

    def _load_jsonl_file(self, path: Path) -> List[ChunkRecord]:
        """Load chunks from JSONL file.

        Rule #1: Extracted to reduce nesting in _load_from_file.

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

    def _build_filter(
        self,
        min_length: Optional[int],
        max_length: Optional[int],
        has_entities: bool,
        entity_type: Optional[str],
        topic: Optional[str],
        threshold: float,
        source_pattern: Optional[str],
        min_quality: float,
    ) -> ChunkFilter:
        """Build filter with all specified criteria.

        Args:
            min_length: Minimum length filter
            max_length: Maximum length filter
            has_entities: Has entities filter
            entity_type: Entity type filter
            topic: Topic filter
            threshold: Topic threshold
            source_pattern: Source pattern filter
            min_quality: Quality score filter

        Returns:
            Configured ChunkFilter
        """
        chunk_filter = ChunkFilter()

        if min_length is not None or max_length is not None:
            chunk_filter.add_length_filter(min_length, max_length)

        if has_entities or entity_type:
            chunk_filter.add_entity_filter(entity_type, has_entities)

        if topic:
            chunk_filter.add_topic_filter(topic, threshold)

        if source_pattern:
            chunk_filter.add_source_filter(source_pattern)

        if min_quality > 0:
            chunk_filter.add_quality_filter(min_quality)

        return chunk_filter

    def _display_results(self, result: FilterResult) -> None:
        """Display filter results.

        Args:
            result: Filter result
        """
        self.console.print()

        table = Table(title="Filter Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Original chunks", str(result.original_count))
        table.add_row("Filtered chunks", str(result.filtered_count))
        table.add_row("Removed chunks", str(result.removed_count))
        table.add_row("Retention rate", f"{result.retention_rate:.1f}%")

        self.console.print(table)

        if result.filters_applied:
            self.console.print()
            self.print_info("Filters applied:")
            for filter_name in result.filters_applied:
                self.console.print(f"  - {filter_name}")

    def _save_filtered_chunks(self, output: Path, chunks: List[ChunkRecord]) -> None:
        """Save filtered chunks to file.

        Rule #1: Max 3 nesting levels via helper extraction.

        Args:
            output: Output file path
            chunks: Chunks to save
        """
        import json

        try:
            data = [chunk.to_dict() for chunk in chunks]

            if output.suffix == ".jsonl":
                self._write_jsonl(output, data)
            else:
                output.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            self.print_success(f"Saved {len(chunks)} chunks to: {output}")

        except Exception as e:
            self.print_error(f"Failed to save chunks: {e}")

    def _write_jsonl(self, output: Path, data: List[Dict[str, Any]]) -> None:
        """Write data to JSONL file.

        Rule #1: Extracted to reduce nesting in _save_filtered_chunks.

        Args:
            output: Output file path
            data: List of dictionaries to write
        """
        import json

        with output.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def command(
    source: str = typer.Argument(..., help="Source library or file"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for filtered chunks"
    ),
    min_length: Optional[int] = typer.Option(
        None, "--min-length", help="Minimum chunk length in characters"
    ),
    max_length: Optional[int] = typer.Option(
        None, "--max-length", help="Maximum chunk length in characters"
    ),
    has_entities: bool = typer.Option(
        False, "--has-entities", help="Only chunks with entities"
    ),
    entity_type: Optional[str] = typer.Option(
        None, "--entity-type", help="Required entity type (PERSON, ORG, etc.)"
    ),
    topic: Optional[str] = typer.Option(
        None, "--topic", "-t", help="Topic to filter by"
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Topic relevance threshold (0-1)"
    ),
    source_pattern: Optional[str] = typer.Option(
        None, "--source-pattern", help="Source file regex pattern"
    ),
    min_quality: float = typer.Option(
        0.0, "--min-quality", help="Minimum quality score (0-1)"
    ),
) -> None:
    """Filter chunks by various criteria.

    Filters chunks from a library or file based on length, entities,
    topic, source, and quality. Multiple filters are combined with AND logic.

    Features:
    - Length filtering (min/max characters)
    - Entity presence filtering
    - Topic relevance filtering
    - Source file pattern matching
    - Quality score thresholds

    Examples:
        # Filter by length
        ingestforge transform filter docs --min-length 100 --max-length 1000

        # Filter by entities
        ingestforge transform filter docs --has-entities --entity-type PERSON

        # Filter by topic
        ingestforge transform filter docs --topic "machine learning" --threshold 0.7

        # Combined filters
        ingestforge transform filter docs --min-length 200 --topic AI -o filtered.json

        # Filter from file
        ingestforge transform filter chunks.jsonl --min-quality 0.5 -o quality.jsonl
    """
    cmd = FilterCommand()
    exit_code = cmd.execute(
        source,
        project,
        output,
        min_length,
        max_length,
        has_entities,
        entity_type,
        topic,
        threshold,
        source_pattern,
        min_quality,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
