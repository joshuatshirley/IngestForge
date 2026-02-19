"""
JSONL file-based storage backend.

Simple file-based storage for development and small datasets.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import (
    ChunkInput,
    ChunkRepository,
    SearchResult,
    sanitize_tag,
    normalize_to_chunk_record,
    MAX_TAGS_PER_CHUNK,
)


class _Logger:
    """Lazy logger holder.

    Rule #6: Encapsulates logger state in smallest scope.
    Avoids slow startup from rich library import.
    """

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


class JSONLRepository(ChunkRepository):
    """
    JSONL file-based chunk storage.

    Stores chunks in a JSONL file with in-memory indexing for search.
    Best for development and small to medium datasets.
    """

    def __init__(self, data_path: Path) -> None:
        """
        Initialize JSONL repository.

        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path
        self.chunks_file = data_path / "chunks" / "chunks.jsonl"
        self.index_file = data_path / "index" / "bm25_index.json"

        # Ensure directories exist
        self.chunks_file.parent.mkdir(parents=True, exist_ok=True)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory indexes
        self._chunks: Dict[str, ChunkRecord] = {}
        self._doc_index: Dict[str, List[str]] = {}  # document_id -> [chunk_ids]
        self._term_index: Dict[str, Dict[str, float]] = {}  # term -> {chunk_id: tf}

        # Load existing data
        self._load()

    def _read_file_lines(self) -> Generator[str, None, None]:
        """
        Read lines from chunks file with safety bounds.

        Rule #1: Extracted helper reduces nesting
        Rule #2: Fixed upper bound (MAX_LINES)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Yields:
            Individual lines from the file

        Raises:
            AssertionError: If chunks_file is not set or doesn't exist
        """
        assert self.chunks_file is not None, "chunks_file must be set"
        assert self.chunks_file.exists(), "chunks_file must exist"
        MAX_LINES: int = 10_000_000  # Hard limit: 10M lines
        lines_read: int = 0
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                lines_read += 1
                if lines_read > MAX_LINES:
                    _Logger.get().error(
                        f"Safety limit reached: read {MAX_LINES} lines, "
                        f"stopping to prevent runaway execution"
                    )
                    break

                yield line

    def _parse_chunk_line(self, line: str) -> Optional[ChunkRecord]:
        """
        Parse a single JSONL line into a ChunkRecord.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            line: JSONL line to parse

        Returns:
            ChunkRecord if valid, None if parse failed
        """
        assert line is not None, "Line cannot be None"
        assert isinstance(line, str), "Line must be string"
        stripped = line.strip()
        if not stripped:
            return None

        try:
            data = json.loads(stripped)
            chunk = ChunkRecord.from_dict(data)
            return chunk
        except json.JSONDecodeError as e:
            _Logger.get().debug(f"Failed to parse JSONL line: {e}")
            return None
        except Exception as e:
            _Logger.get().debug(f"Unexpected error parsing chunk: {e}")
            return None

    def _load(self) -> None:
        """
        Load chunks from file.

        Rule #1: Reduced to 2 nesting levels (try → for)
        Rule #2: Fixed upper bound delegated to _read_file_lines()
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Raises:
            AssertionError: If chunks_file is not set
        """
        assert self.chunks_file is not None, "chunks_file must be set"
        if not self.chunks_file.exists():
            _Logger.get().debug(f"No existing chunks file at {self.chunks_file}")
            return
        chunks_loaded: int = 0
        lines_processed: int = 0

        try:
            # Nesting reduced from 4 to 2 levels (try → for)
            for line in self._read_file_lines():
                lines_processed += 1
                chunk = self._parse_chunk_line(line)
                if chunk is not None:
                    self._add_to_index(chunk)
                    chunks_loaded += 1
            assert (
                len(self._chunks) == chunks_loaded
            ), f"Chunk count mismatch: {len(self._chunks)} != {chunks_loaded}"

            _Logger.get().info(
                f"Loaded {chunks_loaded} chunks from {self.chunks_file} "
                f"({lines_processed} lines processed)"
            )

        except Exception as e:
            _Logger.get().error(f"Failed to load chunks from {self.chunks_file}: {e}")

    def _save_chunk(self, chunk: ChunkRecord) -> None:
        """Append chunk to file."""
        with open(self.chunks_file, "a", encoding="utf-8") as f:
            data = chunk.to_dict()
            # Don't save embeddings to JSONL (too large)
            data["embedding"] = None
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _add_to_index(self, chunk: ChunkRecord) -> None:
        """Add chunk to in-memory indexes."""
        self._chunks[chunk.chunk_id] = chunk

        # Document index
        if chunk.document_id not in self._doc_index:
            self._doc_index[chunk.document_id] = []
        if chunk.chunk_id not in self._doc_index[chunk.document_id]:
            self._doc_index[chunk.document_id].append(chunk.chunk_id)

        # Term index (simple TF)
        terms = self._tokenize(chunk.content)
        term_counts: Dict[str, int] = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        for term, count in term_counts.items():
            if term not in self._term_index:
                self._term_index[term] = {}
            # TF with length normalization
            tf = count / max(len(terms), 1)
            self._term_index[term][chunk.chunk_id] = tf

    def _remove_from_index(self, chunk_id: str) -> None:
        """Remove chunk from indexes."""
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            return

        # Remove from document index
        if chunk.document_id in self._doc_index:
            self._doc_index[chunk.document_id] = [
                cid for cid in self._doc_index[chunk.document_id] if cid != chunk_id
            ]

        # Remove from term index
        for term_dict in self._term_index.values():
            term_dict.pop(chunk_id, None)

        # Remove chunk
        del self._chunks[chunk_id]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]

    def _rebuild_file(self) -> None:
        """Rebuild JSONL file from memory."""
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            for chunk in self._chunks.values():
                data = chunk.to_dict()
                data["embedding"] = None
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def add_chunk(self, chunk: ChunkInput) -> bool:
        """
        Add a chunk.

        g: Accepts both ChunkRecord and IFChunkArtifact.
        """
        try:
            # Normalize to ChunkRecord for storage
            record = normalize_to_chunk_record(chunk)
            self._add_to_index(record)
            self._save_chunk(record)
            return True
        except Exception as e:
            _Logger.get().error(f"Failed to add chunk: {e}")
            return False

    def add_chunks(self, chunks: List[ChunkInput]) -> int:
        """
        Add multiple chunks.

        g: Accepts lists containing ChunkRecord or IFChunkArtifact.
        """
        count = 0
        with open(self.chunks_file, "a", encoding="utf-8") as f:
            for item in chunks:
                try:
                    # Normalize to ChunkRecord for storage
                    record = normalize_to_chunk_record(item)
                    self._add_to_index(record)
                    data = record.to_dict()
                    data["embedding"] = None
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    chunk_id = getattr(item, "chunk_id", None) or getattr(
                        item, "artifact_id", "unknown"
                    )
                    _Logger.get().error(f"Failed to add chunk {chunk_id}: {e}")

        return count

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Get chunk by ID."""
        return self._chunks.get(chunk_id)

    def verify_chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk ID exists (Fast lookup)."""
        return chunk_id in self._chunks

    def get_chunks_by_document(self, document_id: str) -> List[ChunkRecord]:
        """Get all chunks for a document."""
        chunk_ids = self._doc_index.get(document_id, [])
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_all_chunks(self) -> List[ChunkRecord]:
        """Get all chunks in storage."""
        return list(self._chunks.values())

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        if chunk_id not in self._chunks:
            return False

        self._remove_from_index(chunk_id)
        self._rebuild_file()
        return True

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        chunk_ids = self._doc_index.get(document_id, []).copy()
        for chunk_id in chunk_ids:
            self._remove_from_index(chunk_id)

        if chunk_ids:
            self._rebuild_file()

        return len(chunk_ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_filter: Optional[str] = None,
        library_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using BM25-style scoring.

        Rule #1: Reduced nesting via extraction
        Rule #4: Function <60 lines (refactored from 61)
        Rule #9: Full type hints

        Args:
            query: Search query
            top_k: Number of results
            document_filter: Filter by document ID
            library_filter: Filter by library name
            tag_filter: Filter by tag (ORG-002)

        Returns:
            List of SearchResult
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Sanitize tag filter if provided
        clean_tag = None
        if tag_filter:
            clean_tag = sanitize_tag(tag_filter)

        # Calculate BM25 scores with filters
        scores = self._calculate_bm25_scores(
            query_terms, document_filter, library_filter, clean_tag
        )

        # Sort and build results
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return self._build_search_results(ranked)

    def _calculate_bm25_scores(
        self,
        query_terms: List[str],
        document_filter: Optional[str],
        library_filter: Optional[str],
        tag_filter: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate BM25-style scores for query terms.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query_terms: Tokenized query terms
            document_filter: Filter by document ID
            library_filter: Filter by library name
            tag_filter: Filter by tag (sanitized)

        Returns:
            Mapping of chunk_id to BM25 score
        """
        scores: Dict[str, float] = {}
        n_docs = len(self._chunks)

        for term in query_terms:
            if term not in self._term_index:
                continue

            term_docs = self._term_index[term]
            idf = 1.0 + (n_docs / (1 + len(term_docs)))

            for chunk_id, tf in term_docs.items():
                # Apply filters
                if not self._passes_filters(
                    chunk_id, document_filter, library_filter, tag_filter
                ):
                    continue

                scores[chunk_id] = scores.get(chunk_id, 0.0) + tf * idf

        return scores

    def _passes_filters(
        self,
        chunk_id: str,
        document_filter: Optional[str],
        library_filter: Optional[str],
        tag_filter: Optional[str] = None,
    ) -> bool:
        """
        Check if chunk passes document, library, and tag filters.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            chunk_id: Chunk ID to check
            document_filter: Filter by document ID
            library_filter: Filter by library name
            tag_filter: Filter by tag (sanitized)

        Returns:
            True if chunk passes all filters
        """
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            return False

        if document_filter and chunk.document_id != document_filter:
            return False

        if library_filter and chunk.library != library_filter:
            return False

        # Tag filter (ORG-002)
        if tag_filter:
            chunk_tags = getattr(chunk, "tags", None) or []
            if tag_filter not in chunk_tags:
                return False

        return True

    def _build_search_results(
        self, ranked: List[Tuple[str, float]]
    ) -> List[SearchResult]:
        """
        Build SearchResult objects from ranked chunk IDs and scores.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            ranked: List of (chunk_id, score) tuples

        Returns:
            List of SearchResult objects
        """
        results = []
        for chunk_id, score in ranked:
            chunk = self._chunks.get(chunk_id)
            if chunk:
                results.append(SearchResult.from_chunk(chunk, score))
        return results

    def count(self) -> int:
        """Get total chunk count."""
        return len(self._chunks)

    def clear(self) -> None:
        """Clear all data."""
        self._chunks.clear()
        self._doc_index.clear()
        self._term_index.clear()

        if self.chunks_file.exists():
            self.chunks_file.unlink()

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_chunks": len(self._chunks),
            "total_documents": len(self._doc_index),
            "index_terms": len(self._term_index),
            "storage_file": str(self.chunks_file),
            "file_size_bytes": (
                self.chunks_file.stat().st_size if self.chunks_file.exists() else 0
            ),
        }

    def get_libraries(self) -> List[str]:
        """
        Get list of unique library names in storage.

        Returns:
            List of library names, always including "default".
        """
        libraries = set()
        for chunk in self._chunks.values():
            libraries.add(chunk.library)
        libraries.add("default")
        return sorted(list(libraries))

    def count_by_library(self, library_name: str) -> int:
        """
        Count chunks in a specific library.

        Args:
            library_name: Library to count

        Returns:
            Number of chunks in the library
        """
        return sum(
            1 for chunk in self._chunks.values() if chunk.library == library_name
        )

    def delete_by_library(self, library_name: str) -> int:
        """
        Delete all chunks in a library.

        Args:
            library_name: Library to delete

        Returns:
            Number of chunks deleted
        """
        to_delete = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.library == library_name
        ]
        for chunk_id in to_delete:
            del self._chunks[chunk_id]
        self._save()
        return len(to_delete)

    def reassign_library(self, old_library: str, new_library: str) -> int:
        """
        Move all chunks from one library to another.

        Args:
            old_library: Source library name
            new_library: Target library name

        Returns:
            Number of chunks moved
        """
        count = 0
        for chunk in self._chunks.values():
            if chunk.library == old_library:
                chunk.library = new_library
                count += 1
        if count > 0:
            self._save()
        return count

    def move_document_to_library(self, document_id: str, new_library: str) -> int:
        """
        Move all chunks of a specific document to a different library.

        Args:
            document_id: Document to move
            new_library: Target library name

        Returns:
            Number of chunks updated
        """
        count = 0
        for chunk in self._chunks.values():
            if chunk.document_id == document_id:
                chunk.library = new_library
                count += 1
        if count > 0:
            self._save()
        return count

    def mark_read(self, chunk_id: str, status: bool = True) -> bool:
        """
        Mark a chunk as read or unread.

        Updates the is_read field for the specified chunk.
        Completes in <100ms for single chunk updates.

        Args:
            chunk_id: Unique identifier of the chunk to update
            status: True to mark as read, False to mark as unread

        Returns:
            True if successful, False if chunk not found

        Raises:
            ValueError: If chunk_id is empty or None
        """
        if not chunk_id:
            raise ValueError("chunk_id cannot be empty or None")
        if not isinstance(status, bool):
            raise ValueError(f"status must be bool, got {type(status).__name__}")

        # Check if chunk exists
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            _Logger.get().warning(f"Chunk not found: {chunk_id}")
            return False

        # Update is_read status
        chunk.is_read = status

        # Persist change to file
        self._rebuild_file()
        return True

    def get_unread_chunks(
        self,
        library_filter: Optional[str] = None,
    ) -> List[ChunkRecord]:
        """
        Get all chunks marked as unread.

        Args:
            library_filter: If provided, only return chunks from this library

        Returns:
            List of unread ChunkRecords
        """
        unread = []
        for chunk in self._chunks.values():
            # Skip read chunks
            if getattr(chunk, "is_read", False):
                continue

            # Apply library filter if specified
            if library_filter and chunk.library != library_filter:
                continue

            unread.append(chunk)

        return unread

    def _save(self) -> None:
        """Save all chunks to file (alias for _rebuild_file)."""
        self._rebuild_file()

    # Tagging methods (ORG-002)
    def add_tag(self, chunk_id: str, tag: str) -> bool:
        """
        Add a tag to a chunk.

        Rule #7: Input sanitization
        Rule #1: Early returns

        Args:
            chunk_id: Unique identifier of the chunk
            tag: Tag to add (will be sanitized: lowercase, alphanumeric, max 32 chars)

        Returns:
            True if tag was added, False if chunk not found or tag already exists

        Raises:
            ValueError: If chunk_id or tag is empty, or if chunk already has 50 tags
        """
        # Validate parameters
        if not chunk_id:
            raise ValueError("chunk_id cannot be empty or None")

        # Sanitize tag (may raise ValueError)
        clean_tag = sanitize_tag(tag)

        # Check if chunk exists
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            _Logger.get().warning(f"Chunk not found: {chunk_id}")
            return False

        # Ensure tags list exists
        if not hasattr(chunk, "tags") or chunk.tags is None:
            chunk.tags = []

        # Check if tag already exists
        if clean_tag in chunk.tags:
            _Logger.get().debug(f"Tag '{clean_tag}' already exists on chunk {chunk_id}")
            return False

        # Check max tags limit
        if len(chunk.tags) >= MAX_TAGS_PER_CHUNK:
            raise ValueError(
                f"Chunk {chunk_id} already has {MAX_TAGS_PER_CHUNK} tags (maximum)"
            )

        # Add tag and persist
        chunk.tags.append(clean_tag)
        self._rebuild_file()
        return True

    def remove_tag(self, chunk_id: str, tag: str) -> bool:
        """
        Remove a tag from a chunk.

        Args:
            chunk_id: Unique identifier of the chunk
            tag: Tag to remove (will be sanitized before lookup)

        Returns:
            True if tag was removed, False if chunk not found or tag doesn't exist

        Raises:
            ValueError: If chunk_id or tag is empty
        """
        # Validate parameters
        if not chunk_id:
            raise ValueError("chunk_id cannot be empty or None")

        # Sanitize tag (may raise ValueError)
        clean_tag = sanitize_tag(tag)

        # Check if chunk exists
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            _Logger.get().warning(f"Chunk not found: {chunk_id}")
            return False

        # Ensure tags list exists
        if not hasattr(chunk, "tags") or chunk.tags is None:
            chunk.tags = []
            return False

        # Check if tag exists
        if clean_tag not in chunk.tags:
            _Logger.get().debug(f"Tag '{clean_tag}' not found on chunk {chunk_id}")
            return False

        # Remove tag and persist
        chunk.tags.remove(clean_tag)
        self._rebuild_file()
        return True

    def get_chunks_by_tag(
        self,
        tag: str,
        library_filter: Optional[str] = None,
    ) -> List[ChunkRecord]:
        """
        Get all chunks with a specific tag.

        Args:
            tag: Tag to filter by (will be sanitized)
            library_filter: If provided, only return chunks from this library

        Returns:
            List of ChunkRecords with the specified tag (empty list if none found)
        """
        # Sanitize tag (may raise ValueError)
        clean_tag = sanitize_tag(tag)

        results = []
        for chunk in self._chunks.values():
            # Check library filter
            if library_filter and chunk.library != library_filter:
                continue

            # Check if chunk has the tag
            if hasattr(chunk, "tags") and chunk.tags and clean_tag in chunk.tags:
                results.append(chunk)

        return results

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags in the storage.

        Returns:
            Sorted list of unique tags
        """
        all_tags: set = set()
        for chunk in self._chunks.values():
            if hasattr(chunk, "tags") and chunk.tags:
                all_tags.update(chunk.tags)

        return sorted(list(all_tags))
