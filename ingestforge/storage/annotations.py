"""Annotation Storage - Persistent notes attached to chunks.

Provides storage for user annotations that remain linked to chunks
even if the source content is re-ingested (via content hash mapping).

Architecture Position
---------------------
    CLI (outermost)
      |-- **Feature Modules** (you are here)
            |-- Shared (patterns, interfaces, utilities)
                  |-- Core (innermost)

Key Components
--------------
**Annotation**
    Dataclass representing a stored annotation:
    - annotation_id: Unique identifier
    - chunk_id: ID of the annotated chunk
    - content_hash: SHA256 hash of chunk content for re-ingestion mapping
    - text: User annotation text (up to 10,000 characters)
    - created_at: ISO timestamp
    - updated_at: ISO timestamp

**AnnotationManager**
    Storage class for annotations:
    - add(): Store annotation and return ID
    - get(): Retrieve by annotation ID
    - get_for_chunk(): Get all annotations for a chunk
    - get_by_content_hash(): Find annotations by content hash
    - update(): Update annotation text
    - delete(): Remove annotation
    - list_all(): List all annotations"""

from __future__ import annotations

import hashlib
import html
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Constants for validation
MAX_ANNOTATION_LENGTH = 10000
MIN_ANNOTATION_LENGTH = 1


class _Logger:
    """Lazy logger holder."""

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


# =============================================================================
# Annotation Dataclass
# =============================================================================


@dataclass
class Annotation:
    """User annotation attached to a chunk.

    Annotations persist even if the source chunk is re-ingested
    by mapping via content_hash.

    Attributes:
        annotation_id: Unique ID for this annotation
        chunk_id: ID of the chunk this annotation is attached to
        content_hash: SHA256 hash of chunk content for re-ingestion mapping
        text: User annotation text (up to 10,000 characters)
        created_at: ISO timestamp when created
        updated_at: ISO timestamp when last updated
    """

    annotation_id: str
    chunk_id: str
    content_hash: str
    text: str
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Set default timestamps if not provided."""
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "annotation_id": self.annotation_id,
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash,
            "text": self.text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create from dictionary."""
        return cls(
            annotation_id=data.get("annotation_id", ""),
            chunk_id=data.get("chunk_id", ""),
            content_hash=data.get("content_hash", ""),
            text=data.get("text", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a unique annotation ID."""
        return f"ann_{uuid.uuid4().hex[:12]}"

    def sanitize_for_html(self) -> str:
        """Return HTML-safe version of annotation text.

        Rule #7: Sanitize for HTML export to prevent injection attacks.

        Returns:
            HTML-escaped annotation text
        """
        return html.escape(self.text)

    def preview(self, max_length: int = 100) -> str:
        """Get a preview of the annotation text.

        Args:
            max_length: Maximum preview length

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(self.text) <= max_length:
            return self.text
        return self.text[: max_length - 3] + "..."


# =============================================================================
# Utility Functions
# =============================================================================


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for re-ingestion mapping.

    Args:
        content: Chunk content text

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sanitize_annotation_text(text: str) -> str:
    """Sanitize annotation text for storage.

    Rule #7: Check parameters

    Args:
        text: Raw annotation text

    Returns:
        Cleaned text (trimmed, normalized whitespace)

    Raises:
        ValueError: If text is empty or too long
    """
    if not text:
        raise ValueError("Annotation text cannot be empty")

    # Normalize whitespace but preserve intentional line breaks
    cleaned = text.strip()

    # Replace multiple spaces with single space (per line)
    lines = cleaned.split("\n")
    normalized_lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in lines]
    cleaned = "\n".join(normalized_lines)

    if len(cleaned) < MIN_ANNOTATION_LENGTH:
        raise ValueError("Annotation text cannot be empty")

    if len(cleaned) > MAX_ANNOTATION_LENGTH:
        raise ValueError(
            f"Annotation text exceeds maximum length of {MAX_ANNOTATION_LENGTH} "
            f"characters (got {len(cleaned)})"
        )

    return cleaned


# =============================================================================
# AnnotationManager Class
# =============================================================================


class AnnotationManager:
    """Manage persistent annotations stored in JSON.

    Stores annotations in .data/annotations.json with atomic writes
    to prevent data corruption.
    """

    DEFAULT_FILE = "annotations.json"

    def __init__(self, data_dir: Path) -> None:
        """Initialize annotation manager.

        Args:
            data_dir: Directory for data storage (e.g., .data/)
        """
        self.data_dir = data_dir
        self.file_path = data_dir / self.DEFAULT_FILE

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing annotations
        self._annotations: Dict[str, Annotation] = {}
        self._load()

    def _load(self) -> None:
        """Load annotations from disk."""
        if not self.file_path.exists():
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for ann_data in data.get("annotations", []):
                ann = Annotation.from_dict(ann_data)
                self._annotations[ann.annotation_id] = ann

            _Logger.get().debug(f"Loaded {len(self._annotations)} annotations")

        except json.JSONDecodeError as e:
            _Logger.get().error(f"Failed to parse annotations file: {e}")
        except Exception as e:
            _Logger.get().error(f"Failed to load annotations: {e}")

    def _save(self) -> None:
        """Save annotations to disk atomically.

        Uses write-to-temp-then-rename pattern for atomic writes.
        """
        temp_path = self.file_path.with_suffix(".json.tmp")

        try:
            data = {
                "version": 1,
                "annotations": [a.to_dict() for a in self._annotations.values()],
            }

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename (works on Unix, best-effort on Windows)
            temp_path.replace(self.file_path)
            _Logger.get().debug(f"Saved {len(self._annotations)} annotations")

        except Exception as e:
            _Logger.get().error(f"Failed to save annotations: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    def add(
        self,
        chunk_id: str,
        text: str,
        content_hash: str,
    ) -> Annotation:
        """Add a new annotation.

        Args:
            chunk_id: ID of the chunk to annotate
            text: Annotation text (up to 10,000 characters)
            content_hash: SHA256 hash of chunk content

        Returns:
            Created Annotation

        Raises:
            ValueError: If text is invalid
        """
        # Validate and sanitize
        cleaned_text = sanitize_annotation_text(text)

        # Create annotation
        annotation = Annotation(
            annotation_id=Annotation.generate_id(),
            chunk_id=chunk_id,
            content_hash=content_hash,
            text=cleaned_text,
        )

        # Store and persist
        self._annotations[annotation.annotation_id] = annotation
        self._save()

        _Logger.get().info(
            f"Created annotation {annotation.annotation_id} for chunk {chunk_id}"
        )

        return annotation

    def get(self, annotation_id: str) -> Optional[Annotation]:
        """Get annotation by ID.

        Args:
            annotation_id: Annotation identifier

        Returns:
            Annotation or None if not found
        """
        return self._annotations.get(annotation_id)

    def get_for_chunk(self, chunk_id: str) -> List[Annotation]:
        """Get all annotations for a specific chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            List of annotations for the chunk (sorted by created_at)
        """
        annotations = [a for a in self._annotations.values() if a.chunk_id == chunk_id]
        return sorted(annotations, key=lambda a: a.created_at)

    def get_by_content_hash(self, content_hash: str) -> List[Annotation]:
        """Find annotations by content hash.

        Used for remapping annotations after re-ingestion when
        chunk IDs may have changed but content remains the same.

        Args:
            content_hash: SHA256 hash of chunk content

        Returns:
            List of annotations matching the content hash
        """
        annotations = [
            a for a in self._annotations.values() if a.content_hash == content_hash
        ]
        return sorted(annotations, key=lambda a: a.created_at)

    def update(self, annotation_id: str, text: str) -> Optional[Annotation]:
        """Update annotation text.

        Args:
            annotation_id: Annotation to update
            text: New annotation text

        Returns:
            Updated Annotation or None if not found

        Raises:
            ValueError: If text is invalid
        """
        annotation = self._annotations.get(annotation_id)
        if not annotation:
            return None

        # Validate and sanitize
        cleaned_text = sanitize_annotation_text(text)

        # Update
        annotation.text = cleaned_text
        annotation.updated_at = datetime.now().isoformat()

        self._save()

        _Logger.get().info(f"Updated annotation {annotation_id}")

        return annotation

    def delete(self, annotation_id: str) -> bool:
        """Delete an annotation.

        Args:
            annotation_id: Annotation to delete

        Returns:
            True if deleted, False if not found
        """
        if annotation_id not in self._annotations:
            return False

        del self._annotations[annotation_id]
        self._save()

        _Logger.get().info(f"Deleted annotation {annotation_id}")

        return True

    def delete_for_chunk(self, chunk_id: str) -> int:
        """Delete all annotations for a chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            Number of annotations deleted
        """
        to_delete = [
            ann_id
            for ann_id, ann in self._annotations.items()
            if ann.chunk_id == chunk_id
        ]

        for ann_id in to_delete:
            del self._annotations[ann_id]

        if to_delete:
            self._save()
            _Logger.get().info(
                f"Deleted {len(to_delete)} annotations for chunk {chunk_id}"
            )

        return len(to_delete)

    def list_all(self, limit: int = 100) -> List[Annotation]:
        """List all annotations.

        Args:
            limit: Maximum number to return

        Returns:
            List of annotations (sorted by created_at descending)
        """
        annotations = list(self._annotations.values())
        annotations.sort(key=lambda a: a.created_at, reverse=True)
        return annotations[:limit]

    def count(self) -> int:
        """Get total number of annotations."""
        return len(self._annotations)

    def get_statistics(self) -> Dict[str, Any]:
        """Get annotation statistics.

        Returns:
            Dictionary with annotation stats
        """
        annotations = list(self._annotations.values())

        if not annotations:
            return {
                "total_annotations": 0,
                "unique_chunks": 0,
                "avg_length": 0,
                "total_characters": 0,
            }

        unique_chunks = len(set(a.chunk_id for a in annotations))
        total_chars = sum(len(a.text) for a in annotations)
        avg_length = total_chars / len(annotations)

        return {
            "total_annotations": len(annotations),
            "unique_chunks": unique_chunks,
            "avg_length": round(avg_length, 1),
            "total_characters": total_chars,
        }

    def remap_chunk_id(self, old_chunk_id: str, new_chunk_id: str) -> int:
        """Remap annotations from old chunk ID to new chunk ID.

        Used after re-ingestion when chunk IDs change but content
        remains the same.

        Args:
            old_chunk_id: Previous chunk ID
            new_chunk_id: New chunk ID

        Returns:
            Number of annotations remapped
        """
        count = 0
        for annotation in self._annotations.values():
            if annotation.chunk_id == old_chunk_id:
                annotation.chunk_id = new_chunk_id
                annotation.updated_at = datetime.now().isoformat()
                count += 1

        if count > 0:
            self._save()
            _Logger.get().info(
                f"Remapped {count} annotations from {old_chunk_id} to {new_chunk_id}"
            )

        return count


# =============================================================================
# Factory Function
# =============================================================================


def get_annotation_manager(
    data_dir: Optional[Path] = None,
) -> AnnotationManager:
    """Get annotation manager instance.

    Args:
        data_dir: Storage path (defaults to .data/)

    Returns:
        AnnotationManager instance
    """
    if data_dir is None:
        data_dir = Path.cwd() / ".data"

    return AnnotationManager(data_dir=data_dir)
