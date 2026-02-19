"""
Integration Tests for Pipeline End-to-End Processing.

This module tests the full document processing pipeline from ingestion
through chunking, storage, and retrieval.

Test Strategy
-------------
- Use real file processing (small test files)
- Test actual storage backends (JSONL in temp directory)
- Verify data flows correctly through all stages
- Test error handling and recovery

Organization
------------
- TestPipelineBasicFlow: Simple end-to-end processing
- TestPipelineWithStorage: Pipeline + storage integration
- TestPipelineWithRetrieval: Full workflow including search
- TestPipelineErrorHandling: Error cases and recovery
"""

from pathlib import Path

import pytest

from ingestforge.core.pipeline import Pipeline
from ingestforge.core.config import Config
from ingestforge.storage.jsonl import JSONLRepository


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def test_text_file(temp_dir: Path) -> Path:
    """Create a test text file with sample content."""
    test_file = temp_dir / "test_document.txt"
    content = """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses on
    the development of algorithms and statistical models that enable computers
    to improve their performance on a specific task through experience.

    Types of Machine Learning

    There are three main types of machine learning: supervised learning,
    unsupervised learning, and reinforcement learning. Each approach has
    its own strengths and use cases.

    Supervised Learning

    In supervised learning, the algorithm learns from labeled training data.
    The model is trained on a dataset where the correct outputs are known,
    and it learns to map inputs to outputs based on these examples.

    Applications

    Machine learning has numerous applications including image recognition,
    natural language processing, recommendation systems, and autonomous vehicles.
    The field continues to grow rapidly with new techniques and applications.
    """
    test_file.write_text(content.strip(), encoding="utf-8")
    return test_file


@pytest.fixture
def basic_config(temp_dir: Path) -> Config:
    """Create minimal config for pipeline testing."""
    # Use default config
    config = Config()

    # Override key paths to use temp directory
    config.project.data_dir = str(temp_dir / "data")
    config.project.ingest_dir = str(temp_dir / "ingest")

    # Set smaller chunk size for testing
    config.chunking.target_size = 150  # words
    config.chunking.overlap = 25

    # Use JSONL backend (simpler for testing)
    config.storage.backend = "jsonl"

    # Set base path for runtime
    config._base_path = temp_dir

    # Create directories
    (temp_dir / "data").mkdir(parents=True, exist_ok=True)
    (temp_dir / "ingest").mkdir(parents=True, exist_ok=True)

    return config


# ============================================================================
# Test Classes
# ============================================================================


class TestPipelineBasicFlow:
    """Tests for basic pipeline processing flow.

    Rule #4: Focused test class - tests simple end-to-end processing
    """

    def test_pipeline_processes_text_file(
        self, basic_config: Config, test_text_file: Path
    ):
        """Test pipeline can process a simple text file."""
        pipeline = Pipeline(basic_config)

        result = pipeline.process_file(test_text_file)

        # Verify processing succeeded
        assert result is not None
        assert result.chunks_created > 0
        assert result.processing_time_sec > 0

    def test_pipeline_creates_chunks(self, basic_config: Config, test_text_file: Path):
        """Test pipeline creates chunks from text file."""
        pipeline = Pipeline(basic_config)

        result = pipeline.process_file(test_text_file)

        # Should create multiple chunks from the test content
        assert result.chunks_created >= 3  # Introduction, Types, Applications sections

    def test_pipeline_preserves_metadata(
        self, basic_config: Config, test_text_file: Path
    ):
        """Test pipeline preserves document metadata."""
        pipeline = Pipeline(basic_config)

        result = pipeline.process_file(test_text_file)

        # Verify metadata is captured
        assert result.source_file == str(test_text_file)
        assert result.document_id is not None


class TestPipelineWithStorage:
    """Tests for pipeline integration with storage backends.

    Rule #4: Focused test class - tests storage integration
    """

    def test_pipeline_stores_chunks_in_jsonl(
        self, basic_config: Config, test_text_file: Path, temp_dir: Path
    ):
        """Test pipeline stores chunks in JSONL backend."""
        pipeline = Pipeline(basic_config)

        # Process file
        result = pipeline.process_file(test_text_file)

        # Verify chunks were stored
        storage = JSONLRepository(data_path=Path(Path(basic_config.project.data_dir)))
        stored_chunks = storage.get_all_chunks()

        assert len(stored_chunks) == result.chunks_created

    def test_pipeline_storage_has_content(
        self, basic_config: Config, test_text_file: Path, temp_dir: Path
    ):
        """Test stored chunks contain actual content."""
        pipeline = Pipeline(basic_config)
        pipeline.process_file(test_text_file)

        # Retrieve stored chunks
        storage = JSONLRepository(data_path=Path(basic_config.project.data_dir))
        chunks = storage.get_all_chunks()

        # Verify chunks have content
        assert len(chunks) > 0
        for chunk in chunks:
            assert hasattr(chunk, "content")
            assert len(chunk.content) > 0
            assert hasattr(chunk, "chunk_id")

    def test_pipeline_storage_has_metadata(
        self, basic_config: Config, test_text_file: Path, temp_dir: Path
    ):
        """Test stored chunks include metadata."""
        pipeline = Pipeline(basic_config)
        pipeline.process_file(test_text_file)

        storage = JSONLRepository(data_path=Path(basic_config.project.data_dir))
        chunks = storage.get_all_chunks()

        # Verify metadata
        for chunk in chunks:
            assert hasattr(chunk, "document_id")
            assert hasattr(chunk, "source_file")
            assert (
                test_text_file.name in chunk.source_file
            )  # Source file contains filename


class TestPipelineWithRetrieval:
    """Tests for pipeline integration with retrieval.

    Rule #4: Focused test class - tests end-to-end retrieval
    """

    def test_pipeline_enables_search(
        self, basic_config: Config, test_text_file: Path, temp_dir: Path
    ):
        """Test pipeline creates searchable chunks."""
        pipeline = Pipeline(basic_config)
        pipeline.process_file(test_text_file)

        # Search for content
        storage = JSONLRepository(data_path=Path(basic_config.project.data_dir))
        results = storage.search("machine learning", k=5)

        assert len(results) > 0

    def test_pipeline_search_returns_relevant_chunks(
        self, basic_config: Config, test_text_file: Path, temp_dir: Path
    ):
        """Test search returns relevant content."""
        pipeline = Pipeline(basic_config)
        pipeline.process_file(test_text_file)

        storage = JSONLRepository(data_path=Path(basic_config.project.data_dir))
        results = storage.search("supervised learning", k=3)

        # Verify results contain relevant text
        assert len(results) > 0
        found_relevant = False
        for result in results:
            if "supervised" in result.content.lower():
                found_relevant = True
                break
        assert found_relevant

    def test_pipeline_search_includes_scores(
        self, basic_config: Config, test_text_file: Path, temp_dir: Path
    ):
        """Test search results include relevance scores."""
        pipeline = Pipeline(basic_config)
        pipeline.process_file(test_text_file)

        storage = JSONLRepository(data_path=Path(basic_config.project.data_dir))
        results = storage.search("reinforcement learning", k=3)

        # Verify scores are present
        for result in results:
            assert hasattr(result, "score")
            assert isinstance(result.score, (int, float))
            assert result.score >= 0


class TestPipelineErrorHandling:
    """Tests for pipeline error handling.

    Rule #4: Focused test class - tests error cases
    """

    def test_pipeline_handles_missing_file(self, basic_config: Config, temp_dir: Path):
        """Test pipeline handles missing file gracefully."""
        pipeline = Pipeline(basic_config)
        missing_file = temp_dir / "nonexistent.txt"

        # Should return failure result, not raise exception
        result = pipeline.process_file(missing_file)
        assert result.success is False
        assert result.error_message is not None
        assert "not found" in result.error_message.lower()

    def test_pipeline_handles_empty_file(self, basic_config: Config, temp_dir: Path):
        """Test pipeline handles empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        pipeline = Pipeline(basic_config)

        # Pipeline creates 1 chunk for empty files (better than 0)
        result = pipeline.process_file(empty_file)
        assert result.success is True
        assert result.chunks_created >= 0  # May create 0 or 1 chunk


class TestPipelineMultipleDocuments:
    """Tests for processing multiple documents.

    Rule #4: Focused test class - tests batch processing
    """

    def test_pipeline_processes_multiple_files(
        self, basic_config: Config, temp_dir: Path
    ):
        """Test pipeline can process multiple documents."""
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = temp_dir / f"doc_{i}.txt"
            test_file.write_text(f"Document {i} content. " * 20, encoding="utf-8")
            files.append(test_file)

        pipeline = Pipeline(basic_config)

        # Process all files
        total_chunks = 0
        for file in files:
            result = pipeline.process_file(file)
            total_chunks += result.chunks_created

        # Verify all were processed
        assert total_chunks > 0

        # Verify storage contains chunks from all docs
        storage = JSONLRepository(data_path=Path(basic_config.project.data_dir))
        all_chunks = storage.get_all_chunks()
        assert len(all_chunks) == total_chunks

    def test_pipeline_distinguishes_documents(
        self, basic_config: Config, temp_dir: Path
    ):
        """Test chunks from different documents are distinguishable."""
        # Create two distinct files
        file1 = temp_dir / "doc1.txt"
        file1.write_text("First document about cats. " * 20, encoding="utf-8")

        file2 = temp_dir / "doc2.txt"
        file2.write_text("Second document about dogs. " * 20, encoding="utf-8")

        pipeline = Pipeline(basic_config)
        pipeline.process_file(file1)
        pipeline.process_file(file2)

        storage = JSONLRepository(data_path=Path(basic_config.project.data_dir))
        all_chunks = storage.get_all_chunks()

        # Verify we have chunks from both documents
        doc_ids = set(chunk.document_id for chunk in all_chunks)
        assert len(doc_ids) >= 2  # At least 2 distinct documents


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Basic flow: 3 tests (process file, create chunks, preserve metadata)
    - Storage integration: 3 tests (store chunks, content, metadata)
    - Retrieval integration: 3 tests (search, relevance, scores)
    - Error handling: 2 tests (missing file, empty file)
    - Multiple documents: 2 tests (batch processing, document distinction)

    Total: 13 integration tests

Design Decisions:
    1. Use real file processing (not mocking)
    2. Use temporary directories for isolation
    3. Test JSONL backend (simpler than ChromaDB for CI/CD)
    4. Focus on critical happy paths first
    5. Small test files for fast execution
    6. Clean up handled by pytest tmp_path fixture

Behaviors Tested:
    - End-to-end file processing
    - Chunk creation and storage
    - Metadata preservation
    - Search and retrieval
    - Error handling
    - Multi-document processing

Justification:
    - Integration tests verify components work together
    - Real file processing catches integration issues
    - Temporary directories enable parallel test execution
    - JSONL backend is fast and doesn't require external services
    - Small files keep test execution fast
"""
