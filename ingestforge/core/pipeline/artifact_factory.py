"""
Artifact Factory for IngestForge (IF).

a: Create ArtifactFactory.
Converts raw strings, dictionaries, and legacy ChunkRecord objects
into typed IFArtifact instances for gradual migration.
Follows NASA JPL Power of Ten rules.
"""

import hashlib
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from ingestforge.core.pipeline.artifacts import (
    IFTextArtifact,
    IFChunkArtifact,
    IFFileArtifact,
)
from ingestforge.core.pipeline.interfaces import IFArtifact

if TYPE_CHECKING:
    from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_BATCH_CONVERSION = 1000  # Maximum records per batch conversion
MAX_CONTENT_SIZE = 10_000_000  # 10MB max content size


class ArtifactFactory:
    """
    Factory for creating IFArtifact instances from various sources.

    a: Create ArtifactFactory.
    Rule #2: Bounded batch operations.
    Rule #7: Explicit return types.
    Rule #9: Complete type hints.

    This class provides static methods for converting:
    - Raw strings → IFTextArtifact
    - File paths → IFFileArtifact
    - ChunkRecord → IFChunkArtifact
    - Dictionaries → IFChunkArtifact
    """

    @staticmethod
    def text_from_string(
        content: str,
        source_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_id: Optional[str] = None,
    ) -> IFTextArtifact:
        """
        Create an IFTextArtifact from a raw text string.

        Rule #7: Explicit return type.

        Args:
            content: The text content.
            source_path: Optional path to the source file.
            metadata: Optional metadata dictionary.
            artifact_id: Optional custom artifact ID.

        Returns:
            IFTextArtifact with content hash computed.
        """
        if len(content) > MAX_CONTENT_SIZE:
            logger.warning(f"Content exceeds {MAX_CONTENT_SIZE} bytes, truncating")
            content = content[:MAX_CONTENT_SIZE]

        meta = metadata.copy() if metadata else {}
        if source_path:
            meta["source_path"] = source_path

        return IFTextArtifact(
            artifact_id=artifact_id or str(uuid.uuid4()),
            content=content,
            metadata=meta,
        )

    @staticmethod
    def file_from_path(
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        artifact_id: Optional[str] = None,
        compute_hash: bool = True,
    ) -> IFFileArtifact:
        """
        Create an IFFileArtifact from a file path.

        Rule #7: Explicit return type.

        Args:
            path: Path to the file.
            metadata: Optional metadata dictionary.
            artifact_id: Optional custom artifact ID.
            compute_hash: Whether to compute file hash (default True).

        Returns:
            IFFileArtifact with file metadata.
        """
        file_path = Path(path) if isinstance(path, str) else path
        mime_type, _ = mimetypes.guess_type(str(file_path))

        meta = metadata.copy() if metadata else {}
        meta["file_name"] = file_path.name

        content_hash: Optional[str] = None
        if compute_hash and file_path.exists():
            content_hash = _compute_file_hash(file_path)

        return IFFileArtifact(
            artifact_id=artifact_id or str(uuid.uuid4()),
            file_path=file_path.absolute(),
            mime_type=mime_type or "application/octet-stream",
            content_hash=content_hash,
            metadata=meta,
        )

    @staticmethod
    def chunk_from_record(
        record: "ChunkRecord",
        parent: Optional[IFArtifact] = None,
        artifact_id: Optional[str] = None,
    ) -> IFChunkArtifact:
        """
        Create an IFChunkArtifact from a legacy ChunkRecord.

        Rule #7: Explicit return type.

        Args:
            record: The ChunkRecord to convert.
            parent: Optional parent artifact for lineage tracking.
            artifact_id: Optional custom artifact ID.

        Returns:
            IFChunkArtifact preserving all ChunkRecord data.
        """
        # Build metadata from ChunkRecord fields
        metadata: Dict[str, Any] = {
            "section_title": record.section_title,
            "chunk_type": record.chunk_type,
            "source_file": record.source_file,
            "word_count": record.word_count,
            "char_count": record.char_count,
            "library": record.library,
            "is_read": record.is_read,
        }

        # Add optional fields if present
        if record.section_hierarchy:
            metadata["section_hierarchy"] = record.section_hierarchy
        if record.page_start is not None:
            metadata["page_start"] = record.page_start
        if record.page_end is not None:
            metadata["page_end"] = record.page_end
        if record.source_location:
            metadata["source_location"] = str(record.source_location)
        if record.ingested_at:
            metadata["ingested_at"] = record.ingested_at
        if record.tags:
            metadata["tags"] = record.tags
        if record.author_id:
            metadata["author_id"] = record.author_id
        if record.author_name:
            metadata["author_name"] = record.author_name
        if record.entities:
            metadata["entities"] = record.entities
        if record.concepts:
            metadata["concepts"] = record.concepts
        if record.quality_score:
            metadata["quality_score"] = record.quality_score

        # Build lineage from parent if provided
        parent_id: Optional[str] = None
        root_artifact_id: Optional[str] = None
        lineage_depth: int = 0
        provenance: List[str] = []

        if parent:
            parent_id = parent.artifact_id
            root_artifact_id = parent.effective_root_id
            lineage_depth = parent.lineage_depth + 1
            provenance = list(parent.provenance) + ["artifact-factory"]

        return IFChunkArtifact(
            artifact_id=artifact_id or record.chunk_id or str(uuid.uuid4()),
            document_id=record.document_id,
            content=record.content,
            chunk_index=record.chunk_index,
            total_chunks=record.total_chunks,
            metadata=metadata,
            parent_id=parent_id,
            root_artifact_id=root_artifact_id,
            lineage_depth=lineage_depth,
            provenance=provenance,
        )

    @staticmethod
    def chunk_from_dict(
        data: Dict[str, Any],
        parent: Optional[IFArtifact] = None,
        artifact_id: Optional[str] = None,
    ) -> IFChunkArtifact:
        """
        Create an IFChunkArtifact from a dictionary.

        Rule #7: Explicit return type.

        Args:
            data: Dictionary with chunk data.
            parent: Optional parent artifact for lineage tracking.
            artifact_id: Optional custom artifact ID.

        Returns:
            IFChunkArtifact from dictionary data.
        """
        # Extract required fields
        content = str(data.get("content", ""))
        document_id = str(data.get("document_id", data.get("doc_id", "")))

        # Extract optional fields
        chunk_index = int(data.get("chunk_index", data.get("index", 0)))
        total_chunks = int(data.get("total_chunks", 1))

        # Build metadata from remaining fields
        excluded_keys = {
            "content",
            "document_id",
            "doc_id",
            "chunk_index",
            "index",
            "total_chunks",
            "chunk_id",
            "artifact_id",
        }
        metadata = {k: v for k, v in data.items() if k not in excluded_keys}

        # Build lineage from parent if provided
        parent_id: Optional[str] = None
        root_artifact_id: Optional[str] = None
        lineage_depth: int = 0
        provenance: List[str] = []

        if parent:
            parent_id = parent.artifact_id
            root_artifact_id = parent.effective_root_id
            lineage_depth = parent.lineage_depth + 1
            provenance = list(parent.provenance) + ["artifact-factory"]

        return IFChunkArtifact(
            artifact_id=artifact_id or str(data.get("chunk_id", uuid.uuid4())),
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            metadata=metadata,
            parent_id=parent_id,
            root_artifact_id=root_artifact_id,
            lineage_depth=lineage_depth,
            provenance=provenance,
        )

    @staticmethod
    def chunks_from_records(
        records: List["ChunkRecord"], parent: Optional[IFArtifact] = None
    ) -> List[IFChunkArtifact]:
        """
        Convert a batch of ChunkRecords to IFChunkArtifacts.

        Rule #2: Bounded batch size.
        Rule #7: Explicit return type.

        Args:
            records: List of ChunkRecords to convert.
            parent: Optional parent artifact for lineage tracking.

        Returns:
            List of IFChunkArtifact instances.

        Raises:
            ValueError: If batch size exceeds MAX_BATCH_CONVERSION.
        """
        if len(records) > MAX_BATCH_CONVERSION:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum {MAX_BATCH_CONVERSION}"
            )

        return [ArtifactFactory.chunk_from_record(record, parent) for record in records]

    @staticmethod
    def chunks_from_dicts(
        data_list: List[Dict[str, Any]], parent: Optional[IFArtifact] = None
    ) -> List[IFChunkArtifact]:
        """
        Convert a batch of dictionaries to IFChunkArtifacts.

        Rule #2: Bounded batch size.
        Rule #7: Explicit return type.

        Args:
            data_list: List of dictionaries to convert.
            parent: Optional parent artifact for lineage tracking.

        Returns:
            List of IFChunkArtifact instances.

        Raises:
            ValueError: If batch size exceeds MAX_BATCH_CONVERSION.
        """
        if len(data_list) > MAX_BATCH_CONVERSION:
            raise ValueError(
                f"Batch size {len(data_list)} exceeds maximum {MAX_BATCH_CONVERSION}"
            )

        return [ArtifactFactory.chunk_from_dict(data, parent) for data in data_list]


def _compute_file_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

    Rule #4: Function < 60 lines.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


# Convenience functions for common operations


def text_artifact(
    content: str, source_path: Optional[str] = None, **metadata: Any
) -> IFTextArtifact:
    """
    Convenience function to create IFTextArtifact.

    Args:
        content: Text content.
        source_path: Optional source file path.
        **metadata: Additional metadata.

    Returns:
        IFTextArtifact instance.
    """
    return ArtifactFactory.text_from_string(
        content=content,
        source_path=source_path,
        metadata=metadata if metadata else None,
    )


def file_artifact(path: Union[str, Path], **metadata: Any) -> IFFileArtifact:
    """
    Convenience function to create IFFileArtifact.

    Args:
        path: Path to file.
        **metadata: Additional metadata.

    Returns:
        IFFileArtifact instance.
    """
    return ArtifactFactory.file_from_path(
        path=path,
        metadata=metadata if metadata else None,
    )


def chunk_artifact(
    content: str,
    document_id: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
    parent: Optional[IFArtifact] = None,
    **metadata: Any,
) -> IFChunkArtifact:
    """
    Convenience function to create IFChunkArtifact directly.

    Args:
        content: Chunk content.
        document_id: Source document ID.
        chunk_index: Index in document.
        total_chunks: Total chunks in document.
        parent: Optional parent artifact.
        **metadata: Additional metadata.

    Returns:
        IFChunkArtifact instance.
    """
    data = {
        "content": content,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        **metadata,
    }
    return ArtifactFactory.chunk_from_dict(data, parent)
