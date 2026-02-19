"""Tests for Storage Migration Utility.

This module tests the storage migration functionality.

Test Strategy
-------------
- Mock storage backends for unit tests
- Test migration flow and error handling
- Test verification logic
- Test progress tracking

Organization
------------
- TestMigrationResult: Result dataclass
- TestStorageMigrator: Migrator class
- TestMigrationFlow: Complete migration flow
- TestVerification: Verification logic
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock
from typing import List


from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.migration import (
    MigrationResult,
    MigrationProgress,
    StorageMigrator,
    migrate_storage,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(
    chunk_id: str,
    document_id: str = "doc_1",
    content: str = "Test content",
) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        word_count=len(content.split()),
    )


def make_mock_storage(chunks: List[ChunkRecord] = None):
    """Create a mock storage backend."""
    storage = MagicMock()
    chunks = chunks or []

    # Group chunks by document
    docs = {}
    for chunk in chunks:
        if chunk.document_id not in docs:
            docs[chunk.document_id] = []
        docs[chunk.document_id].append(chunk)

    storage.get_chunks_by_document = Mock(
        side_effect=lambda doc_id: docs.get(doc_id, [])
    )
    storage.get_all_chunks = Mock(return_value=chunks)
    storage.add_chunks = Mock(return_value=len(chunks))
    storage.count = Mock(return_value=len(chunks))

    return storage, docs


# ============================================================================
# Test Classes
# ============================================================================


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_create_result(self):
        """Test creating migration result."""
        result = MigrationResult(
            success=True,
            chunks_migrated=100,
            chunks_failed=0,
            documents_migrated=10,
            start_time=datetime.now(),
        )

        assert result.success is True
        assert result.chunks_migrated == 100
        assert result.documents_migrated == 10

    def test_to_dict(self):
        """Test converting result to dictionary."""
        start = datetime.now()
        result = MigrationResult(
            success=True,
            chunks_migrated=50,
            chunks_failed=2,
            documents_migrated=5,
            start_time=start,
            end_time=start,
            duration_seconds=1.5,
            source_backend="jsonl",
            target_backend="postgres",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["chunks_migrated"] == 50
        assert data["source_backend"] == "jsonl"
        assert data["target_backend"] == "postgres"

    def test_result_with_errors(self):
        """Test result with errors."""
        result = MigrationResult(
            success=False,
            chunks_migrated=80,
            chunks_failed=20,
            documents_migrated=8,
            start_time=datetime.now(),
            errors=["Error 1", "Error 2"],
        )

        assert result.success is False
        assert len(result.errors) == 2


class TestMigrationProgress:
    """Tests for MigrationProgress dataclass."""

    def test_create_progress(self):
        """Test creating progress object."""
        progress = MigrationProgress(
            current=5,
            total=10,
            percentage=50.0,
            current_document="doc_5",
            chunks_in_document=20,
        )

        assert progress.current == 5
        assert progress.percentage == 50.0
        assert progress.current_document == "doc_5"


class TestStorageMigrator:
    """Tests for StorageMigrator class."""

    def test_create_migrator(self):
        """Test creating migrator."""
        source = MagicMock()
        target = MagicMock()

        migrator = StorageMigrator(source, target, batch_size=50)

        assert migrator.source == source
        assert migrator.target == target
        assert migrator.batch_size == 50

    def test_migrate_empty_source(self):
        """Test migrating from empty source."""
        source, _ = make_mock_storage([])
        target, _ = make_mock_storage([])

        migrator = StorageMigrator(source, target)
        result = migrator.migrate()

        assert result.chunks_migrated == 0
        assert result.documents_migrated == 0

    def test_migrate_single_document(self):
        """Test migrating single document."""
        chunks = [
            make_chunk("chunk_1", "doc_1"),
            make_chunk("chunk_2", "doc_1"),
        ]
        source, _ = make_mock_storage(chunks)
        target = MagicMock()
        target.add_chunks = Mock(return_value=2)

        migrator = StorageMigrator(source, target)
        result = migrator.migrate()

        assert result.chunks_migrated == 2
        assert result.documents_migrated == 1

    def test_migrate_multiple_documents(self):
        """Test migrating multiple documents."""
        chunks = [
            make_chunk("chunk_1", "doc_1"),
            make_chunk("chunk_2", "doc_1"),
            make_chunk("chunk_3", "doc_2"),
        ]
        source, _ = make_mock_storage(chunks)
        target = MagicMock()
        target.add_chunks = Mock(side_effect=lambda x: len(x))

        migrator = StorageMigrator(source, target)
        result = migrator.migrate()

        assert result.chunks_migrated == 3
        assert result.documents_migrated == 2

    def test_migrate_with_batch_size(self):
        """Test migration respects batch size."""
        chunks = [make_chunk(f"chunk_{i}", "doc_1") for i in range(10)]
        source, _ = make_mock_storage(chunks)
        target = MagicMock()
        target.add_chunks = Mock(side_effect=lambda x: len(x))

        migrator = StorageMigrator(source, target, batch_size=3)
        result = migrator.migrate()

        # Should have made multiple batch calls
        assert target.add_chunks.call_count >= 3

    def test_migrate_handles_error(self):
        """Test migration handles errors gracefully."""
        chunks = [make_chunk("chunk_1", "doc_1")]
        source, _ = make_mock_storage(chunks)
        target = MagicMock()
        target.add_chunks = Mock(side_effect=Exception("Database error"))

        migrator = StorageMigrator(source, target)
        result = migrator.migrate()

        assert result.chunks_failed > 0
        assert len(result.errors) > 0

    def test_migrate_with_progress_callback(self):
        """Test migration calls progress callback."""
        chunks = [
            make_chunk("chunk_1", "doc_1"),
            make_chunk("chunk_2", "doc_2"),
        ]
        source, _ = make_mock_storage(chunks)
        target = MagicMock()
        target.add_chunks = Mock(return_value=1)

        progress_calls = []

        def callback(progress):
            progress_calls.append(progress)

        migrator = StorageMigrator(source, target, progress_callback=callback)
        migrator.migrate()

        assert len(progress_calls) == 2


class TestVerification:
    """Tests for migration verification."""

    def test_verify_success(self):
        """Test verification passes when counts match."""
        source = MagicMock()
        source.count = Mock(return_value=100)
        target = MagicMock()
        target.count = Mock(return_value=100)

        migrator = StorageMigrator(source, target)
        result = migrator.verify()

        assert result is True

    def test_verify_failure(self):
        """Test verification fails when counts mismatch."""
        source = MagicMock()
        source.count = Mock(return_value=100)
        target = MagicMock()
        target.count = Mock(return_value=90)

        migrator = StorageMigrator(source, target)
        result = migrator.verify()

        assert result is False

    def test_verify_documents(self):
        """Test document-level verification."""
        chunks = [
            make_chunk("chunk_1", "doc_1"),
            make_chunk("chunk_2", "doc_1"),
            make_chunk("chunk_3", "doc_2"),
        ]
        source, source_docs = make_mock_storage(chunks)
        target, _ = make_mock_storage(chunks)  # Same chunks in target

        migrator = StorageMigrator(source, target)
        results = migrator.verify_documents()

        assert results["doc_1"] is True
        assert results["doc_2"] is True


class TestMigrateStorageFunction:
    """Tests for migrate_storage convenience function."""

    def test_migrate_storage_basic(self):
        """Test migrate_storage function."""
        chunks = [make_chunk("chunk_1", "doc_1")]
        source, _ = make_mock_storage(chunks)
        target = MagicMock()
        target.add_chunks = Mock(return_value=1)
        target.count = Mock(return_value=1)

        result = migrate_storage(source, target, verify=False)

        assert result.chunks_migrated == 1

    def test_migrate_storage_with_verification(self):
        """Test migrate_storage with verification."""
        chunks = [make_chunk("chunk_1", "doc_1")]
        source, _ = make_mock_storage(chunks)
        target = MagicMock()
        target.add_chunks = Mock(return_value=1)
        target.count = Mock(return_value=1)

        # Source count also needs to return 1 for verification
        source.count = Mock(return_value=1)

        result = migrate_storage(source, target, verify=True)

        assert result.success is True


class TestBackendDetection:
    """Tests for backend name detection."""

    def test_get_backend_name(self):
        """Test getting backend name from class."""
        source = MagicMock()
        source.__class__.__name__ = "JSONLRepository"
        target = MagicMock()
        target.__class__.__name__ = "PostgresRepository"

        migrator = StorageMigrator(source, target)

        assert migrator._get_backend_name(source) == "jsonl"
        assert migrator._get_backend_name(target) == "postgres"


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - MigrationResult: 3 tests
    - MigrationProgress: 1 test
    - StorageMigrator: 7 tests
    - Verification: 3 tests
    - migrate_storage: 2 tests
    - Backend detection: 1 test

    Total: 17 tests

Design Decisions:
    1. Mock storage backends for isolation
    2. Test migration flow and error handling
    3. Test batch processing behavior
    4. Test verification logic
    5. Follows NASA JPL Rule #1 (Simple Control Flow)
    6. Follows NASA JPL Rule #4 (Small Focused Tests)
"""
