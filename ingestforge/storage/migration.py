"""Storage migration utility.

Migrate data between storage backends (JSONL, ChromaDB, PostgreSQL)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import ChunkRepository


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


@dataclass
class MigrationResult:
    """Result of a storage migration operation."""

    success: bool
    chunks_migrated: int
    chunks_failed: int
    documents_migrated: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    source_backend: str = ""
    target_backend: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "chunks_migrated": self.chunks_migrated,
            "chunks_failed": self.chunks_failed,
            "documents_migrated": self.documents_migrated,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "source_backend": self.source_backend,
            "target_backend": self.target_backend,
        }


@dataclass
class MigrationProgress:
    """Progress callback data."""

    current: int
    total: int
    percentage: float
    current_document: str
    chunks_in_document: int


class StorageMigrator:
    """Migrate data between storage backends.

    Supports batch processing, progress tracking, and verification.
    """

    def __init__(
        self,
        source: ChunkRepository,
        target: ChunkRepository,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[MigrationProgress], None]] = None,
    ) -> None:
        """Initialize migrator.

        Args:
            source: Source storage backend
            target: Target storage backend
            batch_size: Number of chunks to process per batch
            progress_callback: Optional callback for progress updates
        """
        self.source = source
        self.target = target
        self.batch_size = batch_size
        self.progress_callback = progress_callback

    def migrate(self) -> MigrationResult:
        """Migrate all data from source to target.

        Returns:
            MigrationResult with statistics
        """
        result = MigrationResult(
            success=False,
            chunks_migrated=0,
            chunks_failed=0,
            documents_migrated=0,
            start_time=datetime.now(),
            source_backend=self._get_backend_name(self.source),
            target_backend=self._get_backend_name(self.target),
        )

        try:
            self._perform_migration(result)
            result.success = result.chunks_failed == 0
        except Exception as e:
            _Logger.get().error(f"Migration failed: {e}")
            result.errors.append(str(e))

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        self._log_result(result)
        return result

    def _perform_migration(self, result: MigrationResult) -> None:
        """Perform the actual migration."""
        # Get all documents from source
        document_ids = self._get_document_ids()

        if not document_ids:
            _Logger.get().warning("No documents found in source")
            return

        total_docs = len(document_ids)
        _Logger.get().info(f"Migrating {total_docs} documents")

        for doc_idx, doc_id in enumerate(document_ids):
            self._migrate_document(doc_id, doc_idx, total_docs, result)

    def _get_document_ids(self) -> List[str]:
        """Get all document IDs from source."""
        # Try to get document IDs through available methods
        if hasattr(self.source, "get_document_ids"):
            return self.source.get_document_ids()

        # Fallback: iterate all chunks and collect document IDs
        if hasattr(self.source, "get_all_chunks"):
            chunks = self.source.get_all_chunks()
            doc_ids = set()
            for chunk in chunks:
                doc_ids.add(chunk.document_id)
            return list(doc_ids)

        # If we can't get documents, get stats
        if hasattr(self.source, "get_statistics"):
            stats = self.source.get_statistics()
            _Logger.get().warning(
                f"Cannot enumerate documents. Source has {stats.get('total_chunks', 0)} chunks"
            )

        return []

    def _migrate_document(
        self,
        document_id: str,
        doc_idx: int,
        total_docs: int,
        result: MigrationResult,
    ) -> None:
        """Migrate a single document."""
        try:
            chunks = self.source.get_chunks_by_document(document_id)
            if not chunks:
                return

            self._report_progress(doc_idx, total_docs, document_id, len(chunks))
            migrated = self._migrate_chunks_batch(chunks, result)

            if migrated > 0:
                result.documents_migrated += 1

        except Exception as e:
            _Logger.get().error(f"Failed to migrate document {document_id}: {e}")
            result.errors.append(f"Document {document_id}: {e}")

    def _migrate_chunks_batch(
        self,
        chunks: List[ChunkRecord],
        result: MigrationResult,
    ) -> int:
        """Migrate chunks in batches."""
        migrated = 0

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
                added = self.target.add_chunks(batch)
                result.chunks_migrated += added
                migrated += added
                result.chunks_failed += len(batch) - added
            except Exception as e:
                _Logger.get().error(f"Batch migration failed: {e}")
                result.chunks_failed += len(batch)
                result.errors.append(f"Batch error: {e}")

        return migrated

    def _report_progress(
        self,
        current: int,
        total: int,
        document_id: str,
        chunks_count: int,
    ) -> None:
        """Report migration progress."""
        if not self.progress_callback:
            return

        progress = MigrationProgress(
            current=current + 1,
            total=total,
            percentage=(current + 1) / total * 100,
            current_document=document_id,
            chunks_in_document=chunks_count,
        )
        self.progress_callback(progress)

    def _get_backend_name(self, backend: ChunkRepository) -> str:
        """Get friendly name for backend."""
        class_name = backend.__class__.__name__
        return class_name.replace("Repository", "").lower()

    def _log_result(self, result: MigrationResult) -> None:
        """Log migration result."""
        status = "SUCCESS" if result.success else "FAILED"
        _Logger.get().info(
            f"Migration {status}: "
            f"{result.chunks_migrated} chunks, "
            f"{result.documents_migrated} documents, "
            f"{result.duration_seconds:.1f}s"
        )
        if result.errors:
            _Logger.get().warning(f"Errors: {len(result.errors)}")

    def verify(self) -> bool:
        """Verify migration success by comparing counts.

        Returns:
            True if source and target have same chunk count
        """
        source_count = self.source.count()
        target_count = self.target.count()

        if source_count == target_count:
            _Logger.get().info(
                f"Verification PASSED: {source_count} chunks in both backends"
            )
            return True

        _Logger.get().error(
            f"Verification FAILED: source={source_count}, target={target_count}"
        )
        return False

    def verify_documents(self) -> Dict[str, bool]:
        """Verify each document was migrated correctly.

        Returns:
            Dict mapping document_id to verification result
        """
        results = {}
        document_ids = self._get_document_ids()

        for doc_id in document_ids:
            source_chunks = self.source.get_chunks_by_document(doc_id)
            target_chunks = self.target.get_chunks_by_document(doc_id)

            source_count = len(source_chunks)
            target_count = len(target_chunks)

            results[doc_id] = source_count == target_count

            if source_count != target_count:
                _Logger.get().warning(
                    f"Document {doc_id} mismatch: "
                    f"source={source_count}, target={target_count}"
                )

        return results


def migrate_storage(
    source: ChunkRepository,
    target: ChunkRepository,
    batch_size: int = 100,
    verify: bool = True,
) -> MigrationResult:
    """Convenience function to migrate storage.

    Args:
        source: Source storage backend
        target: Target storage backend
        batch_size: Chunks per batch
        verify: Whether to verify after migration

    Returns:
        MigrationResult
    """
    migrator = StorageMigrator(source, target, batch_size)
    result = migrator.migrate()

    if verify and result.success:
        result.success = migrator.verify()

    return result
