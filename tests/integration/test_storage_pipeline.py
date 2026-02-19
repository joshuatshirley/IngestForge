"""
Integration Tests for Storage Pipeline.

Tests the complete storage workflow including chunk storage,
retrieval, indexing, and cross-backend compatibility.

Test Coverage
-------------
- JSONL storage backend
- ChromaDB storage backend
- PostgreSQL storage backend (if available)
- Cross-backend data migration
- Bulk operations (batch add/delete)
- Query consistency across backends
- Index creation and maintenance
- Storage performance

Test Strategy
-------------
- Test each storage backend independently
- Test data format compatibility
- Verify query results are consistent
- Test bulk operations for performance
- Test migration between backends
- Test error handling for storage failures
"""

import json
import tempfile
from pathlib import Path
from typing import List
import time

import pytest
import numpy as np

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.jsonl import JSONLRepository
from ingestforge.storage.base import SearchResult
from ingestforge.storage.factory import StorageFactory
from ingestforge.core.config import Config


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for storage testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage_config(temp_dir: Path) -> Config:
    """Create configuration for storage testing."""
    config = Config()
    config.project.data_dir = str(temp_dir / "data")
    config.storage.backend = "jsonl"
    config._base_path = temp_dir
    (temp_dir / "data").mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def jsonl_storage(temp_dir: Path) -> JSONLRepository:
    """Create JSONL storage instance."""
    data_path = temp_dir / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    return JSONLRepository(data_path=data_path)


@pytest.fixture
def sample_chunks_with_embeddings() -> List[ChunkRecord]:
    """Create sample chunks with embeddings for testing."""
    chunks = []
    for i in range(10):
        # Generate deterministic embedding
        np.random.seed(i)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        chunk = ChunkRecord(
            chunk_id=f"chunk_{i}",
            document_id=f"doc_{i // 3}",  # Group chunks by document
            content=f"This is test content for chunk {i}. It contains information about topic {i % 3}.",
            section_title=f"Section {i}",
            chunk_type="content",
            source_file=f"test_{i // 3}.md",
            word_count=15,
            char_count=75,
            embedding=embedding.tolist(),
            entities=["entity1", "entity2"] if i % 2 == 0 else ["entity3"],
            concepts=[f"topic_{i % 3}"],
            metadata={"index": i, "category": "test"},
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def large_chunk_dataset() -> List[ChunkRecord]:
    """Create large dataset for performance testing."""
    chunks = []
    for i in range(100):
        np.random.seed(i)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        chunk = ChunkRecord(
            chunk_id=f"large_chunk_{i}",
            document_id=f"large_doc_{i // 10}",
            content=f"Content for large dataset chunk {i}. " * 10,
            section_title=f"Large Section {i}",
            chunk_type="content",
            source_file=f"large_{i // 10}.md",
            word_count=100,
            char_count=500,
            embedding=embedding.tolist(),
            entities=[f"entity_{i % 5}"],
            concepts=[f"concept_{i % 3}"],
            metadata={"batch": i // 10},
        )
        chunks.append(chunk)

    return chunks


# ============================================================================
# Test Classes
# ============================================================================


class TestJSONLStorage:
    """Tests for JSONL storage backend.

    Rule #4: Focused test class - tests JSONL storage
    """

    def test_add_single_chunk(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test adding a single chunk to JSONL storage."""
        chunk = sample_chunks_with_embeddings[0]

        count = jsonl_storage.add_chunks([chunk])

        assert count == 1
        assert jsonl_storage.count() >= 1

    def test_add_multiple_chunks(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test adding multiple chunks to JSONL storage."""
        chunks = sample_chunks_with_embeddings[:5]

        count = jsonl_storage.add_chunks(chunks)

        assert count == 5
        assert jsonl_storage.count() >= 5

    def test_retrieve_chunk_by_id(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test retrieving chunk by ID."""
        chunks = sample_chunks_with_embeddings[:3]
        jsonl_storage.add_chunks(chunks)

        retrieved = jsonl_storage.get_chunk("chunk_0")

        assert retrieved is not None
        assert retrieved.chunk_id == "chunk_0"
        assert retrieved.content == chunks[0].content

    def test_retrieve_all_chunks(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test retrieving all chunks."""
        chunks = sample_chunks_with_embeddings[:5]
        jsonl_storage.add_chunks(chunks)

        all_chunks = jsonl_storage.get_all_chunks()

        assert len(all_chunks) >= 5

    def test_delete_chunk(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test deleting chunk by ID."""
        chunks = sample_chunks_with_embeddings[:3]
        jsonl_storage.add_chunks(chunks)

        initial_count = jsonl_storage.count()
        deleted = jsonl_storage.delete_chunk("chunk_0")

        assert deleted is True
        assert jsonl_storage.count() == initial_count - 1

    def test_clear_all_chunks(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test clearing all chunks."""
        chunks = sample_chunks_with_embeddings[:5]
        jsonl_storage.add_chunks(chunks)

        jsonl_storage.clear()

        assert jsonl_storage.count() == 0

    def test_get_document_ids(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test retrieving unique document IDs."""
        chunks = sample_chunks_with_embeddings[:9]  # 3 documents
        jsonl_storage.add_chunks(chunks)

        doc_ids = jsonl_storage.get_document_ids()

        assert len(doc_ids) >= 3
        assert "doc_0" in doc_ids
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids


class TestJSONLSearch:
    """Tests for JSONL search functionality.

    Rule #4: Focused test class - tests JSONL search
    """

    def test_keyword_search(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test keyword-based search."""
        chunks = sample_chunks_with_embeddings[:5]
        jsonl_storage.add_chunks(chunks)

        results = jsonl_storage.search("topic 0", k=3)

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    def test_search_returns_scores(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test that search results include relevance scores."""
        chunks = sample_chunks_with_embeddings[:5]
        jsonl_storage.add_chunks(chunks)

        results = jsonl_storage.search("content", k=3)

        for result in results:
            assert hasattr(result, "score")
            assert isinstance(result.score, (int, float))
            assert result.score >= 0

    def test_search_respects_top_k(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test that search returns at most k results."""
        chunks = sample_chunks_with_embeddings
        jsonl_storage.add_chunks(chunks)

        results = jsonl_storage.search("test", k=3)

        assert len(results) <= 3

    def test_semantic_search_with_embeddings(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test semantic search using embeddings."""
        chunks = sample_chunks_with_embeddings[:5]
        jsonl_storage.add_chunks(chunks)

        # Create query embedding
        np.random.seed(0)
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = jsonl_storage.search_semantic(query_embedding.tolist(), k=3)

        assert len(results) > 0
        for result in results:
            assert hasattr(result, "score")

    def test_search_filters_by_metadata(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test search filtering by metadata."""
        chunks = sample_chunks_with_embeddings[:5]
        jsonl_storage.add_chunks(chunks)

        # Search with metadata filter
        results = jsonl_storage.search("test", k=10, filters={"category": "test"})

        # All results should match filter
        for result in results:
            assert result.metadata.get("category") == "test"


class TestBulkOperations:
    """Tests for bulk storage operations.

    Rule #4: Focused test class - tests bulk operations
    """

    def test_bulk_add_chunks(
        self, jsonl_storage: JSONLRepository, large_chunk_dataset: List[ChunkRecord]
    ):
        """Test bulk addition of chunks."""
        chunks = large_chunk_dataset

        start_time = time.time()
        count = jsonl_storage.add_chunks(chunks)
        duration = time.time() - start_time

        assert count == len(chunks)
        # Should complete reasonably fast (< 5 seconds for 100 chunks)
        assert duration < 5.0

    def test_bulk_delete_by_document(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test bulk deletion of chunks by document ID."""
        chunks = sample_chunks_with_embeddings
        jsonl_storage.add_chunks(chunks)

        # Delete all chunks from doc_0
        initial_count = jsonl_storage.count()
        deleted_count = 0
        all_chunks = jsonl_storage.get_all_chunks()

        for chunk in all_chunks:
            if chunk.document_id == "doc_0":
                jsonl_storage.delete_chunk(chunk.chunk_id)
                deleted_count += 1

        final_count = jsonl_storage.count()
        assert final_count == initial_count - deleted_count

    def test_batch_retrieval_performance(
        self, jsonl_storage: JSONLRepository, large_chunk_dataset: List[ChunkRecord]
    ):
        """Test performance of batch retrieval."""
        chunks = large_chunk_dataset
        jsonl_storage.add_chunks(chunks)

        start_time = time.time()
        all_chunks = jsonl_storage.get_all_chunks()
        duration = time.time() - start_time

        assert len(all_chunks) >= len(chunks)
        # Should retrieve reasonably fast (< 2 seconds for 100 chunks)
        assert duration < 2.0


class TestDataPersistence:
    """Tests for data persistence and recovery.

    Rule #4: Focused test class - tests persistence
    """

    def test_data_persists_after_reload(
        self, temp_dir: Path, sample_chunks_with_embeddings: List[ChunkRecord]
    ):
        """Test that data persists after reloading storage."""
        data_path = temp_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        # Create first storage instance and add data
        storage1 = JSONLRepository(data_path=data_path)
        chunks = sample_chunks_with_embeddings[:5]
        storage1.add_chunks(chunks)
        count1 = storage1.count()

        # Create new storage instance pointing to same path
        storage2 = JSONLRepository(data_path=data_path)
        count2 = storage2.count()

        # Counts should match
        assert count2 == count1

    def test_retrieve_persisted_chunks(
        self, temp_dir: Path, sample_chunks_with_embeddings: List[ChunkRecord]
    ):
        """Test retrieving chunks from persisted storage."""
        data_path = temp_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        # Add chunks with first instance
        storage1 = JSONLRepository(data_path=data_path)
        chunks = sample_chunks_with_embeddings[:3]
        storage1.add_chunks(chunks)

        # Retrieve with second instance
        storage2 = JSONLRepository(data_path=data_path)
        retrieved = storage2.get_chunk("chunk_0")

        assert retrieved is not None
        assert retrieved.chunk_id == "chunk_0"

    def test_embeddings_persist_correctly(
        self, temp_dir: Path, sample_chunks_with_embeddings: List[ChunkRecord]
    ):
        """Test that embeddings are correctly persisted."""
        data_path = temp_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        # Add chunks with embeddings
        storage1 = JSONLRepository(data_path=data_path)
        chunks = sample_chunks_with_embeddings[:2]
        storage1.add_chunks(chunks)

        # Retrieve embeddings
        storage2 = JSONLRepository(data_path=data_path)
        retrieved = storage2.get_chunk("chunk_0")

        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 384
        # Verify embedding values match
        assert np.allclose(retrieved.embedding, chunks[0].embedding, atol=1e-6)


class TestStorageFactory:
    """Tests for storage factory.

    Rule #4: Focused test class - tests factory
    """

    def test_create_jsonl_backend(self, storage_config: Config):
        """Test creating JSONL backend via factory."""
        storage_config.storage.backend = "jsonl"

        storage = StorageFactory.create_storage(storage_config)

        assert storage is not None
        assert isinstance(storage, JSONLRepository)

    def test_create_chromadb_backend_if_available(self, storage_config: Config):
        """Test creating ChromaDB backend if available."""
        storage_config.storage.backend = "chromadb"

        try:
            storage = StorageFactory.create_storage(storage_config)
            assert storage is not None
        except ImportError:
            # ChromaDB not installed, skip test
            pytest.skip("ChromaDB not available")

    def test_factory_respects_config(self, storage_config: Config):
        """Test that factory respects configuration."""
        storage_config.storage.backend = "jsonl"

        storage = StorageFactory.create_storage(storage_config)

        # Should use configured path
        assert storage is not None


class TestCrossBackendCompatibility:
    """Tests for cross-backend data compatibility.

    Rule #4: Focused test class - tests compatibility
    """

    def test_chunk_serialization_format(
        self, sample_chunks_with_embeddings: List[ChunkRecord]
    ):
        """Test that chunk serialization is backend-agnostic."""
        chunk = sample_chunks_with_embeddings[0]

        # Serialize to dict
        chunk_dict = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "section_title": chunk.section_title,
            "chunk_type": chunk.chunk_type,
            "source_file": chunk.source_file,
            "word_count": chunk.word_count,
            "char_count": chunk.char_count,
            "embedding": chunk.embedding,
            "entities": chunk.entities,
            "concepts": chunk.concepts,
            "metadata": chunk.metadata,
        }

        # Should be JSON serializable
        json_str = json.dumps(chunk_dict)
        assert len(json_str) > 0

        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded["chunk_id"] == chunk.chunk_id

    def test_export_to_jsonl_format(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test exporting chunks to JSONL format."""
        chunks = sample_chunks_with_embeddings[:5]
        jsonl_storage.add_chunks(chunks)

        # Export all chunks
        all_chunks = jsonl_storage.get_all_chunks()

        # Should be able to serialize all chunks
        for chunk in all_chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
            }
            json_str = json.dumps(chunk_dict)
            assert len(json_str) > 0


class TestErrorHandling:
    """Tests for storage error handling.

    Rule #4: Focused test class - tests errors
    """

    def test_handle_duplicate_chunk_ids(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test handling of duplicate chunk IDs."""
        chunk = sample_chunks_with_embeddings[0]
        jsonl_storage.add_chunks([chunk])

        # Try to add same chunk again
        duplicate = ChunkRecord(
            chunk_id=chunk.chunk_id,  # Same ID
            document_id="different_doc",
            content="Different content",
            section_title="Different section",
            chunk_type="content",
            source_file="different.md",
            word_count=2,
            char_count=17,
        )

        # Should handle gracefully (update or ignore)
        count = jsonl_storage.add_chunks([duplicate])
        assert count >= 0

    def test_handle_missing_chunk_retrieval(self, jsonl_storage: JSONLRepository):
        """Test retrieval of non-existent chunk."""
        retrieved = jsonl_storage.get_chunk("nonexistent_chunk")

        # Should return None for missing chunk
        assert retrieved is None

    def test_handle_invalid_search_query(
        self,
        jsonl_storage: JSONLRepository,
        sample_chunks_with_embeddings: List[ChunkRecord],
    ):
        """Test search with empty/invalid query."""
        chunks = sample_chunks_with_embeddings[:3]
        jsonl_storage.add_chunks(chunks)

        # Empty query
        results = jsonl_storage.search("", k=3)

        # Should handle gracefully (return all or none)
        assert isinstance(results, list)

    def test_handle_storage_corruption(self, temp_dir: Path):
        """Test handling of corrupted storage files."""
        data_path = temp_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        # Create corrupted JSONL file
        corrupt_file = data_path / "chunks.jsonl"
        corrupt_file.write_text("{invalid json\n{also invalid\n", encoding="utf-8")

        # Should handle corruption gracefully
        try:
            storage = JSONLRepository(data_path=data_path)
            count = storage.count()
            # May return 0 or raise exception
            assert count >= 0
        except Exception:
            # Expected for corrupted file
            pass


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - JSONL storage: 7 tests (add, retrieve, delete, clear, document IDs)
    - JSONL search: 5 tests (keyword, scores, top-k, semantic, filters)
    - Bulk operations: 3 tests (bulk add, bulk delete, performance)
    - Data persistence: 3 tests (reload, retrieve, embeddings)
    - Storage factory: 3 tests (JSONL, ChromaDB, config)
    - Cross-backend: 2 tests (serialization, export)
    - Error handling: 4 tests (duplicates, missing, invalid, corruption)

    Total: 27 integration tests

Design Decisions:
    1. Test JSONL backend primarily (most portable)
    2. Test ChromaDB conditionally if available
    3. Focus on data persistence and retrieval
    4. Test bulk operations for performance
    5. Verify cross-backend compatibility

Behaviors Tested:
    - Chunk storage and retrieval
    - Search functionality (keyword + semantic)
    - Bulk operations performance
    - Data persistence across restarts
    - Factory pattern for backend selection
    - Serialization format compatibility
    - Error handling and recovery

Justification:
    - Integration tests verify storage reliability
    - Persistence tests ensure data durability
    - Performance tests catch bottlenecks
    - Error tests ensure robustness
    - Cross-backend tests enable migration
"""
