"""Merge command - Merge and deduplicate chunks.

Merges multiple libraries or files and provides deduplication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from rich.table import Table

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.cli.transform.base import TransformCommand


class DedupeStrategy(str, Enum):
    """Deduplication strategy options."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


class ConflictResolution(str, Enum):
    """Conflict resolution strategy options."""

    KEEP_FIRST = "keep_first"
    KEEP_LATEST = "keep_latest"
    MERGE_METADATA = "merge_metadata"


@dataclass
class MergeResult:
    """Result of a merge operation."""

    source_counts: Dict[str, int] = field(default_factory=dict)
    total_input: int = 0
    total_output: int = 0
    duplicates_removed: int = 0
    conflicts_resolved: int = 0
    chunks: List[ChunkRecord] = field(default_factory=list)

    @property
    def dedup_rate(self) -> float:
        """Calculate deduplication rate as percentage."""
        if self.total_input == 0:
            return 0.0
        return (self.duplicates_removed / self.total_input) * 100


class ChunkMerger:
    """Merge and deduplicate chunks from multiple sources.

    Supports merging libraries with configurable deduplication
    and conflict resolution strategies.
    """

    def __init__(
        self,
        dedupe_strategy: DedupeStrategy = DedupeStrategy.EXACT,
        conflict_resolution: ConflictResolution = ConflictResolution.KEEP_FIRST,
        similarity_threshold: float = 0.9,
    ) -> None:
        """Initialize chunk merger.

        Args:
            dedupe_strategy: Strategy for detecting duplicates
            conflict_resolution: How to resolve duplicate conflicts
            similarity_threshold: Threshold for fuzzy/semantic matching
        """
        self.dedupe_strategy = dedupe_strategy
        self.conflict_resolution = conflict_resolution
        self.similarity_threshold = similarity_threshold
        self._seen_hashes: Set[str] = set()
        self._embedding_cache: Dict[str, List[float]] = {}

    def merge_libraries(
        self,
        sources: Dict[str, List[ChunkRecord]],
        deduplicate: bool = True,
    ) -> MergeResult:
        """Merge multiple sources into one.

        Args:
            sources: Dictionary mapping source name to chunks
            deduplicate: Whether to remove duplicates

        Returns:
            MergeResult with merged chunks and statistics
        """
        result = MergeResult()
        all_chunks: List[ChunkRecord] = []

        # Collect chunks from all sources
        for source_name, chunks in sources.items():
            result.source_counts[source_name] = len(chunks)
            result.total_input += len(chunks)
            all_chunks.extend(chunks)

        # Deduplicate if requested
        if deduplicate:
            merged = self.deduplicate(all_chunks)
            result.duplicates_removed = len(all_chunks) - len(merged)
        else:
            merged = all_chunks

        result.total_output = len(merged)
        result.chunks = merged

        return result

    def deduplicate(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """Remove duplicate chunks.

        Args:
            chunks: Chunks to deduplicate

        Returns:
            Deduplicated chunks
        """
        # Choose strategy
        handlers = {
            DedupeStrategy.EXACT: self._dedupe_exact,
            DedupeStrategy.FUZZY: self._dedupe_fuzzy,
            DedupeStrategy.SEMANTIC: self._dedupe_semantic,
        }

        handler = handlers.get(self.dedupe_strategy, self._dedupe_exact)
        return handler(chunks)

    def _dedupe_exact(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """Deduplicate using exact content hash.

        Args:
            chunks: Chunks to deduplicate

        Returns:
            Deduplicated chunks
        """
        result: List[ChunkRecord] = []
        seen: Set[str] = set()

        for chunk in chunks:
            content_hash = self._hash_content(chunk.content)

            if content_hash in seen:
                continue

            seen.add(content_hash)
            result.append(chunk)

        return result

    def _dedupe_fuzzy(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """Deduplicate using fuzzy matching (Jaccard similarity).

        Args:
            chunks: Chunks to deduplicate

        Returns:
            Deduplicated chunks
        """
        result: List[ChunkRecord] = []
        kept_words: List[Set[str]] = []

        for chunk in chunks:
            words = set(chunk.content.lower().split())
            is_duplicate = self._is_fuzzy_duplicate(words, kept_words)

            if is_duplicate:
                continue

            kept_words.append(words)
            result.append(chunk)

        return result

    def _is_fuzzy_duplicate(self, words: Set[str], kept_words: List[Set[str]]) -> bool:
        """Check if words match any kept set.

        Args:
            words: Word set to check
            kept_words: List of previously kept word sets

        Returns:
            True if duplicate found
        """
        for kept in kept_words:
            similarity = self._jaccard_similarity(words, kept)
            if similarity >= self.similarity_threshold:
                return True
        return False

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between sets.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Similarity score (0-1)
        """
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _dedupe_semantic(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """Deduplicate using semantic similarity.

        Falls back to fuzzy if embeddings unavailable.

        Args:
            chunks: Chunks to deduplicate

        Returns:
            Deduplicated chunks
        """
        # Check if chunks have embeddings
        has_embeddings = all(chunk.embedding for chunk in chunks)

        if not has_embeddings:
            # Fall back to fuzzy
            return self._dedupe_fuzzy(chunks)

        result: List[ChunkRecord] = []
        kept_embeddings: List[List[float]] = []

        for chunk in chunks:
            embedding = chunk.embedding
            if embedding is None:
                result.append(chunk)
                continue

            is_duplicate = self._is_semantic_duplicate(embedding, kept_embeddings)

            if is_duplicate:
                continue

            kept_embeddings.append(embedding)
            result.append(chunk)

        return result

    def _is_semantic_duplicate(
        self, embedding: List[float], kept_embeddings: List[List[float]]
    ) -> bool:
        """Check if embedding matches any kept embedding.

        Args:
            embedding: Embedding to check
            kept_embeddings: List of previously kept embeddings

        Returns:
            True if duplicate found
        """

        for kept in kept_embeddings:
            similarity = self._cosine_similarity(embedding, kept)
            if similarity >= self.similarity_threshold:
                return True
        return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0-1)
        """
        import math

        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _hash_content(self, content: str) -> str:
        """Create hash of content for exact matching.

        Args:
            content: Content to hash

        Returns:
            Hash string
        """
        normalized = content.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def combine_chunks(
        self, chunks: List[ChunkRecord], strategy: str = "sequential"
    ) -> List[ChunkRecord]:
        """Combine related chunks.

        Args:
            chunks: Chunks to combine
            strategy: Combination strategy (sequential, by_document)

        Returns:
            Combined chunks
        """
        if strategy == "by_document":
            return self._combine_by_document(chunks)
        return chunks  # Default: keep as-is

    def _combine_by_document(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """Group and combine chunks by document.

        Args:
            chunks: Chunks to combine

        Returns:
            Combined chunks (one per document)
        """
        by_doc: Dict[str, List[ChunkRecord]] = {}

        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id not in by_doc:
                by_doc[doc_id] = []
            by_doc[doc_id].append(chunk)

        result: List[ChunkRecord] = []
        for doc_id, doc_chunks in by_doc.items():
            combined = self._merge_document_chunks(doc_chunks)
            result.append(combined)

        return result

    def _merge_document_chunks(self, chunks: List[ChunkRecord]) -> ChunkRecord:
        """Merge chunks from same document.

        Args:
            chunks: Chunks to merge

        Returns:
            Single merged chunk
        """
        if not chunks:
            raise ValueError("No chunks to merge")

        if len(chunks) == 1:
            return chunks[0]

        # Sort by index
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)

        # Combine content
        content = "\n\n".join(c.content for c in sorted_chunks)

        # Use first chunk as base
        base = sorted_chunks[0]

        return ChunkRecord(
            chunk_id=f"{base.document_id}_merged",
            document_id=base.document_id,
            content=content,
            section_title=base.section_title,
            chunk_type="merged",
            source_file=base.source_file,
            word_count=len(content.split()),
            char_count=len(content),
            chunk_index=0,
            total_chunks=1,
            library=base.library,
            entities=self._merge_lists([c.entities for c in sorted_chunks]),
            concepts=self._merge_lists([c.concepts for c in sorted_chunks]),
        )

    def _merge_lists(self, lists: List[List[str]]) -> List[str]:
        """Merge multiple lists, removing duplicates.

        Args:
            lists: Lists to merge

        Returns:
            Merged list
        """
        seen: Set[str] = set()
        result: List[str] = []

        for lst in lists:
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    result.append(item)

        return result

    def resolve_conflicts(
        self, chunk1: ChunkRecord, chunk2: ChunkRecord
    ) -> ChunkRecord:
        """Resolve conflict between duplicate chunks.

        Args:
            chunk1: First chunk
            chunk2: Second chunk

        Returns:
            Resolved chunk
        """
        handlers = {
            ConflictResolution.KEEP_FIRST: lambda: chunk1,
            ConflictResolution.KEEP_LATEST: lambda: self._keep_latest(chunk1, chunk2),
            ConflictResolution.MERGE_METADATA: lambda: self._merge_metadata(
                chunk1, chunk2
            ),
        }

        handler = handlers.get(self.conflict_resolution, lambda: chunk1)
        return handler()

    def _keep_latest(self, chunk1: ChunkRecord, chunk2: ChunkRecord) -> ChunkRecord:
        """Keep the chunk with later ingestion date.

        Args:
            chunk1: First chunk
            chunk2: Second chunk

        Returns:
            Latest chunk
        """
        date1 = chunk1.ingested_at or ""
        date2 = chunk2.ingested_at or ""

        return chunk2 if date2 > date1 else chunk1

    def _merge_metadata(self, chunk1: ChunkRecord, chunk2: ChunkRecord) -> ChunkRecord:
        """Merge metadata from both chunks.

        Args:
            chunk1: First chunk
            chunk2: Second chunk

        Returns:
            Chunk with merged metadata
        """
        # Use first chunk content, merge metadata
        merged_entities = list(set(chunk1.entities + chunk2.entities))
        merged_concepts = list(set(chunk1.concepts + chunk2.concepts))
        merged_metadata = {**chunk1.metadata, **chunk2.metadata}

        return ChunkRecord(
            chunk_id=chunk1.chunk_id,
            document_id=chunk1.document_id,
            content=chunk1.content,
            section_title=chunk1.section_title,
            chunk_type=chunk1.chunk_type,
            source_file=chunk1.source_file,
            word_count=chunk1.word_count,
            char_count=chunk1.char_count,
            chunk_index=chunk1.chunk_index,
            total_chunks=chunk1.total_chunks,
            library=chunk1.library,
            entities=merged_entities,
            concepts=merged_concepts,
            quality_score=max(chunk1.quality_score, chunk2.quality_score),
            metadata=merged_metadata,
        )


class MergeCommand(TransformCommand):
    """Merge and deduplicate chunks."""

    def execute(
        self,
        sources: List[str],
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        target: Optional[str] = None,
        deduplicate: bool = False,
        similarity: float = 0.9,
        strategy: str = "exact",
        conflict: str = "keep_first",
    ) -> int:
        """Execute merge operation.

        Args:
            sources: Source libraries or files to merge
            project: Project directory
            output: Output file path
            target: Target library name
            deduplicate: Whether to deduplicate
            similarity: Similarity threshold for deduplication
            strategy: Deduplication strategy
            conflict: Conflict resolution strategy

        Returns:
            0 on success, 1 on error
        """
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Load chunks from all sources
            source_chunks = self._load_sources(ctx, sources)

            if not source_chunks:
                self.print_warning("No chunks found in any source")
                return 0

            # Create merger
            dedupe_strategy = (
                DedupeStrategy(strategy) if strategy else DedupeStrategy.EXACT
            )
            conflict_resolution = (
                ConflictResolution(conflict)
                if conflict
                else ConflictResolution.KEEP_FIRST
            )

            merger = ChunkMerger(
                dedupe_strategy=dedupe_strategy,
                conflict_resolution=conflict_resolution,
                similarity_threshold=similarity,
            )

            # Merge
            result = merger.merge_libraries(source_chunks, deduplicate)

            # Display results
            self._display_results(result)

            # Save output
            if output:
                self._save_chunks(output, result.chunks)

            if target:
                self._save_to_library(ctx, target, result.chunks)

            return 0

        except Exception as e:
            return self.handle_error(e, "Merge operation failed")

    def _load_sources(
        self, ctx: Dict[str, Any], sources: List[str]
    ) -> Dict[str, List[ChunkRecord]]:
        """Load chunks from all sources.

        Args:
            ctx: Command context
            sources: List of source names/paths

        Returns:
            Dictionary mapping source to chunks
        """
        storage = ctx.get("storage")
        result: Dict[str, List[ChunkRecord]] = {}

        for source in sources:
            chunks = self._load_single_source(storage, source)
            if chunks:
                result[source] = chunks

        return result

    def _load_single_source(self, storage: Any, source: str) -> List[ChunkRecord]:
        """Load chunks from a single source.

        Args:
            storage: Storage backend
            source: Source name or path

        Returns:
            List of chunks
        """
        # Check if source is a library
        if storage:
            libraries = storage.get_libraries()
            if source in libraries:
                return self._load_from_library(storage, source)

        # Check if source is a file
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
        results = storage.search("", top_k=10000, library_filter=library)
        chunks = []

        for result in results:
            chunk = storage.get_chunk(result.chunk_id)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _load_from_file(self, path: Path) -> List[ChunkRecord]:
        """Load chunks from file.

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
                chunks = self._load_jsonl_chunks(path)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    chunks = [ChunkRecord.from_dict(item) for item in data]
        except Exception as e:
            self.print_error(f"Error loading file: {e}")

        return chunks

    def _load_jsonl_chunks(self, path: Path) -> List[ChunkRecord]:
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

    def _display_results(self, result: MergeResult) -> None:
        """Display merge results.

        Args:
            result: Merge result
        """
        self.console.print()

        # Source table
        source_table = Table(title="Sources")
        source_table.add_column("Source", style="cyan")
        source_table.add_column("Chunks", style="green")

        for source, count in result.source_counts.items():
            source_table.add_row(source, str(count))

        self.console.print(source_table)

        # Summary table
        self.console.print()
        summary_table = Table(title="Merge Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total input chunks", str(result.total_input))
        summary_table.add_row("Output chunks", str(result.total_output))
        summary_table.add_row("Duplicates removed", str(result.duplicates_removed))
        summary_table.add_row("Deduplication rate", f"{result.dedup_rate:.1f}%")

        self.console.print(summary_table)

    def _save_chunks(self, output: Path, chunks: List[ChunkRecord]) -> None:
        """Save chunks to file.

        Rule #1: Max 3 nesting levels via helper extraction.

        Args:
            output: Output path
            chunks: Chunks to save
        """
        import json

        try:
            data = [chunk.to_dict() for chunk in chunks]

            if output.suffix == ".jsonl":
                self._write_jsonl_chunks(output, data)
            else:
                output.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            self.print_success(f"Saved {len(chunks)} chunks to: {output}")

        except Exception as e:
            self.print_error(f"Failed to save chunks: {e}")

    def _write_jsonl_chunks(self, output: Path, data: List[Dict[str, Any]]) -> None:
        """Write chunk data to JSONL file.

        Rule #1: Extracted to reduce nesting in _save_chunks.

        Args:
            output: Output file path
            data: List of chunk dictionaries to write
        """
        import json

        with output.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _save_to_library(
        self, ctx: Dict[str, Any], library: str, chunks: List[ChunkRecord]
    ) -> None:
        """Save chunks to a library.

        Args:
            ctx: Command context
            library: Target library name
            chunks: Chunks to save
        """
        storage = ctx.get("storage")
        if not storage:
            self.print_error("Storage not available")
            return

        # Update library for all chunks
        for chunk in chunks:
            chunk.library = library

        count = storage.add_chunks(chunks)
        self.print_success(f"Added {count} chunks to library: {library}")


def command(
    sources: List[str] = typer.Argument(..., help="Source libraries or files to merge"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for merged chunks"
    ),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target library name"
    ),
    deduplicate: bool = typer.Option(
        False, "--deduplicate", "-d", help="Remove duplicate chunks"
    ),
    similarity: float = typer.Option(
        0.9, "--similarity", help="Similarity threshold for fuzzy/semantic dedup"
    ),
    strategy: str = typer.Option(
        "exact", "--strategy", help="Dedup strategy: exact, fuzzy, semantic"
    ),
    conflict: str = typer.Option(
        "keep_first",
        "--conflict",
        help="Conflict resolution: keep_first, keep_latest, merge_metadata",
    ),
) -> None:
    """Merge and deduplicate chunks from multiple sources.

    Combines chunks from multiple libraries or files into a single output.
    Supports various deduplication strategies and conflict resolution.

    Deduplication strategies:
    - exact: Remove exact content matches (fastest)
    - fuzzy: Remove similar content using Jaccard similarity
    - semantic: Remove semantically similar content using embeddings

    Conflict resolution:
    - keep_first: Keep the first occurrence
    - keep_latest: Keep the chunk with latest timestamp
    - merge_metadata: Keep first content, merge metadata

    Examples:
        # Merge two libraries
        ingestforge transform merge library1 library2 --target merged

        # Merge with deduplication
        ingestforge transform merge lib1 lib2 --deduplicate --similarity 0.9

        # Merge files with semantic dedup
        ingestforge transform merge data1.jsonl data2.jsonl -d --strategy semantic -o merged.jsonl

        # Merge and save to library
        ingestforge transform merge old_docs new_docs -d -t combined
    """
    cmd = MergeCommand()
    exit_code = cmd.execute(
        sources, project, output, target, deduplicate, similarity, strategy, conflict
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
