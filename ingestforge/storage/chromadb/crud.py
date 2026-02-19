"""
CRUD Operations Mixin for ChromaDB.

Handles create, read, update, delete operations for chunks.
"""

from typing import Any, Dict, List, Optional

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import (
    ChunkInput,
    sanitize_tag,
    normalize_to_chunk_record,
    MAX_TAGS_PER_CHUNK,
    chunk_has_tag,
    parse_tags_json,
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


class ChromaDBCRUDMixin:
    """
    Mixin providing CRUD operations for ChromaDB repository.

    Extracted from chromadb.py to reduce file size (Phase 4, Rule #4).
    """

    def add_chunk(self, chunk: ChunkInput) -> bool:
        """
        Add a chunk to content collection and questions to questions collection.

        g: Accepts both ChunkRecord and IFChunkArtifact.
        """
        try:
            # Normalize to ChunkRecord for storage
            record = normalize_to_chunk_record(chunk)
            metadata = self._chunk_to_metadata(record)

            # If chunk has embedding, use it directly
            if record.embedding:
                self.collection.add(
                    ids=[record.chunk_id],
                    embeddings=[record.embedding],
                    documents=[record.content],
                    metadatas=[metadata],
                )
            else:
                # Let ChromaDB generate embedding
                self.collection.add(
                    ids=[record.chunk_id],
                    documents=[record.content],
                    metadatas=[metadata],
                )

            # Add hypothetical questions to questions collection (if enabled)
            if self.enable_multi_vector and self.questions_collection:
                questions = getattr(record, "hypothetical_questions", None)
                if questions:
                    self._add_questions_for_chunk(record.chunk_id, questions, metadata)

            return True
        except Exception as e:
            _Logger.get().error(f"Failed to add chunk: {e}")
            return False

    def _add_questions_for_chunk(
        self,
        chunk_id: str,
        questions: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        """Add hypothetical questions to the questions collection.

        Each question maps back to its parent chunk_id for retrieval.
        """
        if not questions or not self.questions_collection:
            return

        try:
            # Create unique IDs for each question
            question_ids = [f"{chunk_id}_q{i}" for i in range(len(questions))]

            # Add parent chunk_id to metadata
            question_metadata = [
                {**metadata, "parent_chunk_id": chunk_id} for _ in questions
            ]

            self.questions_collection.add(
                ids=question_ids,
                documents=questions,
                metadatas=question_metadata,
            )
        except Exception as e:
            _Logger.get().warning(f"Failed to add questions for chunk {chunk_id}: {e}")

    def add_chunks(self, chunks: List[ChunkInput]) -> int:
        """
        Add multiple chunks.

        g: Accepts lists containing ChunkRecord or IFChunkArtifact.
        """
        if not chunks:
            return 0

        try:
            # Normalize all items to ChunkRecords
            records = [normalize_to_chunk_record(c) for c in chunks]

            ids = [r.chunk_id for r in records]
            documents = [r.content for r in records]
            metadatas = [self._chunk_to_metadata(r) for r in records]

            # Check if chunks have embeddings
            if records[0].embedding:
                embeddings = [r.embedding for r in records]
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

            return len(records)
        except Exception as e:
            _Logger.get().error(f"Failed to add chunks: {e}")
            return 0

    def _chunk_to_metadata(self, chunk: ChunkRecord) -> Dict[str, Any]:
        """Convert chunk to ChromaDB metadata."""
        import json

        metadata = {
            "document_id": chunk.document_id,
            "section_title": chunk.section_title or "",
            "chunk_type": chunk.chunk_type,
            "source_file": chunk.source_file,
            "word_count": chunk.word_count,
            "chunk_index": chunk.chunk_index,
            "quality_score": chunk.quality_score,
            "library": chunk.library,
            "ingested_at": chunk.ingested_at or "",
            "is_read": getattr(chunk, "is_read", False),
        }

        # Serialize source_location as JSON string (ChromaDB only supports simple types)
        if chunk.source_location:
            metadata["source_location_json"] = json.dumps(
                chunk.source_location.to_dict()
            )

        # Serialize tags as JSON string (ChromaDB only supports simple types)
        if hasattr(chunk, "tags") and chunk.tags:
            metadata["tags_json"] = json.dumps(chunk.tags)

        # Unstructured-style metadata fields
        # Bounding box coordinates
        if hasattr(chunk, "bbox") and chunk.bbox:
            metadata["bbox_x1"] = chunk.bbox[0]
            metadata["bbox_y1"] = chunk.bbox[1]
            metadata["bbox_x2"] = chunk.bbox[2]
            metadata["bbox_y2"] = chunk.bbox[3]

        # Table HTML (for table elements)
        if hasattr(chunk, "table_html") and chunk.table_html:
            metadata["table_html"] = chunk.table_html

        # Element type classification
        if hasattr(chunk, "element_type") and chunk.element_type:
            metadata["element_type"] = chunk.element_type

        return metadata

    def _metadata_to_chunk(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> ChunkRecord:
        """Convert ChromaDB result to ChunkRecord."""
        import json
        from ingestforge.core.provenance import SourceLocation

        chunk = ChunkRecord(
            chunk_id=chunk_id,
            document_id=metadata.get("document_id", ""),
            content=content,
            chunk_type=metadata.get("chunk_type", "content"),
            section_title=metadata.get("section_title", ""),
            source_file=metadata.get("source_file", ""),
            word_count=metadata.get("word_count", 0),
            chunk_index=metadata.get("chunk_index", 0),
            quality_score=metadata.get("quality_score", 0.0),
            library=metadata.get("library", "default"),
            ingested_at=metadata.get("ingested_at") or None,
            is_read=metadata.get("is_read", False),
        )

        # Deserialize tags if present
        if "tags_json" in metadata:
            try:
                chunk.tags = json.loads(metadata["tags_json"])
            except json.JSONDecodeError:
                chunk.tags = []

        # Restore Unstructured-style metadata fields
        # Bounding box
        if all(k in metadata for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]):
            chunk.bbox = (
                metadata["bbox_x1"],
                metadata["bbox_y1"],
                metadata["bbox_x2"],
                metadata["bbox_y2"],
            )

        # Table HTML
        if "table_html" in metadata:
            chunk.table_html = metadata["table_html"]

        # Element type
        if "element_type" in metadata:
            chunk.element_type = metadata["element_type"]

        # Deserialize source_location if present
        if "source_location_json" in metadata:
            try:
                source_dict = json.loads(metadata["source_location_json"])
                chunk.source_location = SourceLocation.from_dict(source_dict)
            except json.JSONDecodeError as e:
                _Logger.get().error(
                    f"Corrupt source_location JSON for chunk {chunk_id}: {e}. "
                    f"Raw JSON: {metadata.get('source_location_json', '')[:100]}"
                )
                # Create minimal placeholder to preserve some citation info
                from ingestforge.core.provenance import SourceType

                chunk.source_location = SourceLocation(
                    title=metadata.get("source_file", "Unknown"),
                    source_type=SourceType.UNKNOWN,
                )
            except Exception as e:
                _Logger.get().error(
                    f"Failed to deserialize source_location for chunk {chunk_id}: {e}"
                )
                # Preserve fallback citation using available metadata
                from ingestforge.core.provenance import SourceType

                chunk.source_location = SourceLocation(
                    title=metadata.get("source_file", "Unknown"),
                    source_type=SourceType.UNKNOWN,
                )

        return chunk

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Get chunk by ID."""
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )

            if result["ids"]:
                return self._metadata_to_chunk(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                )
            return None
        except Exception as e:
            _Logger.get().error(f"Failed to get chunk: {e}")
            return None

    def verify_chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk ID exists in the storage (Fast check)."""
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=[],  # Don't fetch data, just IDs
            )
            return bool(result["ids"])
        except Exception as e:
            _Logger.get().error(f"Failed to verify chunk existence: {e}")
            return False

    def get_chunks_by_document(self, document_id: str) -> List[ChunkRecord]:
        """Get all chunks for a document."""
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )

            chunks = []
            for i, chunk_id in enumerate(result["ids"]):
                chunk = self._metadata_to_chunk(
                    chunk_id,
                    result["documents"][i],
                    result["metadatas"][i],
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            _Logger.get().error(f"Failed to get document chunks: {e}")
            return []

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        try:
            self.collection.delete(ids=[chunk_id])
            return True
        except Exception as e:
            _Logger.get().error(f"Failed to delete chunk: {e}")
            return False

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        try:
            # Get chunks first to count them
            result = self.collection.get(
                where={"document_id": document_id},
                include=[],
            )

            if result["ids"]:
                self.collection.delete(ids=result["ids"])
                return len(result["ids"])
            return 0
        except Exception as e:
            _Logger.get().error(f"Failed to delete document: {e}")
            return 0

    def mark_read(self, chunk_id: str, status: bool = True) -> bool:
        """
        Mark a chunk as read or unread.

        Updates the is_read metadata field for the specified chunk.
        Completes in <100ms for single chunk updates.

        Args:
            chunk_id: Unique identifier of the chunk to update
            status: True to mark as read, False to mark as unread

        Returns:
            True if successful, False if chunk not found or update failed

        Raises:
            ValueError: If chunk_id is empty or None
        """
        if not chunk_id:
            raise ValueError("chunk_id cannot be empty or None")
        if not isinstance(status, bool):
            raise ValueError(f"status must be bool, got {type(status).__name__}")

        try:
            # Check if chunk exists
            result = self.collection.get(
                ids=[chunk_id],
                include=["metadatas"],
            )

            if not result["ids"]:
                _Logger.get().warning(f"Chunk not found: {chunk_id}")
                return False

            # Update metadata with is_read status
            existing_metadata = result["metadatas"][0] if result["metadatas"] else {}
            existing_metadata["is_read"] = status

            # ChromaDB update requires full upsert
            self.collection.update(
                ids=[chunk_id],
                metadatas=[existing_metadata],
            )

            return True
        except Exception as e:
            _Logger.get().error(f"Failed to mark chunk as read: {e}")
            return False

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
        try:
            # Build where clause for unread chunks
            where_clause: Dict[str, Any] = {"is_read": {"$ne": True}}

            if library_filter:
                where_clause = {
                    "$and": [
                        {"is_read": {"$ne": True}},
                        {"library": library_filter},
                    ]
                }

            result = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"],
            )

            chunks = []
            for i, chunk_id in enumerate(result["ids"]):
                chunk = self._metadata_to_chunk(
                    chunk_id,
                    result["documents"][i],
                    result["metadatas"][i],
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            _Logger.get().error(f"Failed to get unread chunks: {e}")
            return []

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
        import json

        # Validate parameters
        if not chunk_id:
            raise ValueError("chunk_id cannot be empty or None")

        # Sanitize tag (may raise ValueError)
        clean_tag = sanitize_tag(tag)

        try:
            # Get existing chunk
            result = self.collection.get(
                ids=[chunk_id],
                include=["metadatas"],
            )

            if not result["ids"]:
                _Logger.get().warning(f"Chunk not found: {chunk_id}")
                return False

            metadata = result["metadatas"][0] if result["metadatas"] else {}

            # Parse existing tags
            existing_tags = []
            if "tags_json" in metadata:
                try:
                    existing_tags = json.loads(metadata["tags_json"])
                except json.JSONDecodeError:
                    existing_tags = []

            # Check if tag already exists
            if clean_tag in existing_tags:
                _Logger.get().debug(
                    f"Tag '{clean_tag}' already exists on chunk {chunk_id}"
                )
                return False

            # Check max tags limit
            if len(existing_tags) >= MAX_TAGS_PER_CHUNK:
                raise ValueError(
                    f"Chunk {chunk_id} already has {MAX_TAGS_PER_CHUNK} tags (maximum)"
                )

            # Add tag and update
            existing_tags.append(clean_tag)
            metadata["tags_json"] = json.dumps(existing_tags)

            self.collection.update(
                ids=[chunk_id],
                metadatas=[metadata],
            )

            return True
        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            _Logger.get().error(f"Failed to add tag: {e}")
            return False

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
        import json

        # Validate parameters
        if not chunk_id:
            raise ValueError("chunk_id cannot be empty or None")

        # Sanitize tag (may raise ValueError)
        clean_tag = sanitize_tag(tag)

        try:
            # Get existing chunk
            result = self.collection.get(
                ids=[chunk_id],
                include=["metadatas"],
            )

            if not result["ids"]:
                _Logger.get().warning(f"Chunk not found: {chunk_id}")
                return False

            metadata = result["metadatas"][0] if result["metadatas"] else {}

            # Parse existing tags
            existing_tags = []
            if "tags_json" in metadata:
                try:
                    existing_tags = json.loads(metadata["tags_json"])
                except json.JSONDecodeError:
                    existing_tags = []

            # Check if tag exists
            if clean_tag not in existing_tags:
                _Logger.get().debug(f"Tag '{clean_tag}' not found on chunk {chunk_id}")
                return False

            # Remove tag and update
            existing_tags.remove(clean_tag)
            metadata["tags_json"] = json.dumps(existing_tags)

            self.collection.update(
                ids=[chunk_id],
                metadatas=[metadata],
            )

            return True
        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            _Logger.get().error(f"Failed to remove tag: {e}")
            return False

    def get_chunks_by_tag(
        self,
        tag: str,
        library_filter: Optional[str] = None,
    ) -> List[ChunkRecord]:
        """
        Get all chunks with a specific tag.

        Note: ChromaDB doesn't support JSON field queries, so we retrieve all
        chunks and filter in memory. For large datasets, consider a dedicated
        tag index.

        Args:
            tag: Tag to filter by (will be sanitized)
            library_filter: If provided, only return chunks from this library

        Returns:
            List of ChunkRecords with the specified tag (empty list if none found)
        """

        # Sanitize tag (may raise ValueError)
        clean_tag = sanitize_tag(tag)

        try:
            # Build where clause
            where = None
            if library_filter:
                where = {"library": library_filter}

            # Get all chunks (with optional library filter)
            result = self.collection.get(
                where=where,
                include=["documents", "metadatas"],
            )

            chunks = []
            for i, chunk_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i]
                if not chunk_has_tag(metadata, clean_tag):
                    continue

                chunk = self._metadata_to_chunk(
                    chunk_id,
                    result["documents"][i],
                    metadata,
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            _Logger.get().error(f"Failed to get chunks by tag: {e}")
            return []

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags in the storage.

        Rule #1: Refactored to use parse_tags_json helper.

        Returns:
            Sorted list of unique tags
        """
        try:
            # Get all chunks
            result = self.collection.get(
                include=["metadatas"],
            )

            all_tags: set = set()
            for metadata in result["metadatas"]:
                all_tags.update(parse_tags_json(metadata))

            return sorted(list(all_tags))
        except Exception as e:
            _Logger.get().error(f"Failed to get all tags: {e}")
            return []
