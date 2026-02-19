"""
Shared pytest fixtures and configuration for IngestForge tests.

This file is automatically discovered by pytest and provides fixtures
that can be used across all test files.

Fixture Organization
--------------------
- **temp_dir**: Temporary directory for file operations
- **mock_config**: Mock configuration objects
- **mock_llm**: Mock LLM clients for all providers
- **mock_storage**: Mock storage backends
- **sample_documents**: Document builders for various formats
- **sample_chunks**: ChunkRecord generators
- **sample_embeddings**: Embedding vector generators

All fixtures are designed to be reusable, generic, and composable.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np

# Mock cryptography to avoid PyO3 initialization errors in restricted environments
try:
    import sys
    from unittest.mock import MagicMock

    mock_crypto = MagicMock()
    sys.modules["cryptography"] = mock_crypto
    sys.modules["cryptography.hazmat"] = MagicMock()
    sys.modules["cryptography.hazmat.primitives"] = MagicMock()
    sys.modules["cryptography.hazmat.primitives.asymmetric"] = MagicMock()
    sys.modules["cryptography.hazmat.primitives.serialization"] = MagicMock()
    sys.modules["cryptography.hazmat.backends"] = MagicMock()
except Exception:
    pass

# Import core types
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import SearchResult

# j: Import artifact types for new fixtures
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFTextArtifact


# ============================================================================
# Path and Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path to temporary directory (cleaned up after test)

    Example:
        def test_file_creation(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("content")
            assert test_file.exists()
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def data_dir(temp_dir: Path) -> Path:
    """Create a data directory structure for testing.

    Creates:
        - data/chunks/
        - data/embeddings/
        - data/metadata/

    Returns:
        Path to data directory
    """
    data_path = temp_dir / "data"
    (data_path / "chunks").mkdir(parents=True, exist_ok=True)
    (data_path / "embeddings").mkdir(parents=True, exist_ok=True)
    (data_path / "metadata").mkdir(parents=True, exist_ok=True)
    return data_path


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock configuration object.

    Returns:
        Mock Config with all necessary attributes

    Example:
        def test_with_config(mock_config):
            assert mock_config.chunking.target_size == 1000
    """
    from ingestforge.core.config import Config

    config = Config()
    return config


@pytest.fixture
def minimal_config(temp_dir: Path) -> Mock:
    """Create a minimal config for tests that need specific paths.

    Returns:
        Config with paths pointing to temp_dir
    """
    from ingestforge.core.config import Config

    config = Config()
    config._base_path = temp_dir
    return config


# ============================================================================
# LLM Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_response() -> str:
    """Standard mock LLM response text.

    Returns:
        Sample LLM response string
    """
    return "This is a mock response from the LLM."


@pytest.fixture
def mock_llm_client(mock_llm_response: str) -> Mock:
    """Create a mock LLM client matching the LLMClient interface.

    Provides implementations for:
    - generate(prompt, config) -> str
    - generate_with_context(prompt, context, system_prompt, config) -> str
    - is_available() -> bool

    Returns:
        Mock LLMClient with standard responses

    Example:
        def test_llm_call(mock_llm_client):
            result = mock_llm_client.generate("test prompt")
            assert "mock response" in result
    """
    client = Mock()
    client.generate.return_value = mock_llm_response
    client.generate_with_context.return_value = mock_llm_response
    client.is_available.return_value = True
    client.model_name = "mock-model"
    return client


@pytest.fixture
def mock_claude_client(mock_llm_response: str) -> Mock:
    """Mock Claude LLM client."""
    client = Mock()
    client.generate.return_value = mock_llm_response
    client.generate_with_context.return_value = mock_llm_response
    client.is_available.return_value = True
    client.model_name = "claude-sonnet-4-20250514"
    client.api_key = "mock_api_key"
    return client


@pytest.fixture
def mock_openai_client(mock_llm_response: str) -> Mock:
    """Mock OpenAI LLM client."""
    client = Mock()
    client.generate.return_value = mock_llm_response
    client.generate_with_context.return_value = mock_llm_response
    client.is_available.return_value = True
    client.model_name = "gpt-4"
    client.api_key = "mock_api_key"
    return client


@pytest.fixture
def mock_ollama_client(mock_llm_response: str) -> Mock:
    """Mock Ollama LLM client."""
    client = Mock()
    client.generate.return_value = mock_llm_response
    client.generate_with_context.return_value = mock_llm_response
    client.is_available.return_value = True
    client.model_name = "llama2"
    client.base_url = "http://localhost:11434"
    return client


@pytest.fixture
def mock_gemini_client(mock_llm_response: str) -> Mock:
    """Mock Gemini LLM client."""
    client = Mock()
    client.generate.return_value = mock_llm_response
    client.generate_with_context.return_value = mock_llm_response
    client.is_available.return_value = True
    client.model_name = "gemini-pro"
    client.api_key = "mock_api_key"
    return client


@pytest.fixture
def mock_llamacpp_client(mock_llm_response: str) -> Mock:
    """Mock LlamaCPP LLM client."""
    client = Mock()
    client.generate.return_value = mock_llm_response
    client.generate_with_context.return_value = mock_llm_response
    client.is_available.return_value = True
    client.model_name = "llama-7b"
    client.model_path = "/path/to/model.gguf"
    return client


# ============================================================================
# Storage Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_storage() -> Mock:
    """Create a mock storage backend matching ChunkRepository interface.

    Provides implementations for:
    - add_chunks(chunks) -> int
    - get_chunk(chunk_id) -> ChunkRecord
    - search(query, top_k, **kwargs) -> List[SearchResult]
    - search_semantic(embedding, top_k, **kwargs) -> List[SearchResult]
    - delete_chunk(chunk_id) -> bool
    - clear() -> None
    - count() -> int

    Returns:
        Mock ChunkRepository with standard responses

    Example:
        def test_storage_add(mock_storage):
            mock_storage.add_chunks([chunk])
            mock_storage.add_chunks.assert_called_once()
    """
    storage = Mock()
    storage.add_chunks.return_value = 1
    storage.get_chunk.return_value = None
    storage.search.return_value = []
    storage.search_semantic.return_value = []
    storage.delete_chunk.return_value = False
    storage.clear.return_value = None
    storage.count.return_value = 0
    storage.get_all_chunks.return_value = []
    storage.get_document_ids.return_value = []
    storage.get_libraries.return_value = ["default"]
    return storage


@pytest.fixture
def in_memory_storage() -> Dict[str, ChunkRecord]:
    """Create an in-memory storage dictionary for testing.

    Returns:
        Empty dict for storing chunks by chunk_id

    Example:
        def test_in_memory(in_memory_storage):
            in_memory_storage["chunk_1"] = make_chunk("chunk_1")
            assert len(in_memory_storage) == 1
    """
    return {}


# ============================================================================
# Sample Data Fixtures - Chunks
# ============================================================================


@pytest.fixture
def make_chunk():
    """Factory fixture for creating test ChunkRecords.

    Returns:
        Function to create ChunkRecord instances

    Example:
        def test_chunks(make_chunk):
            chunk = make_chunk("chunk_1", content="Test content")
            assert chunk.chunk_id == "chunk_1"
    """

    def _make_chunk(
        chunk_id: str,
        document_id: str = "test_doc",
        content: str = "",
        section_title: str = "Test Section",
        chunk_type: str = "content",
        source_file: str = "test.txt",
        word_count: Optional[int] = None,
        char_count: Optional[int] = None,
        library: str = "default",
        embedding: Optional[List[float]] = None,
        entities: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChunkRecord:
        """Create a test ChunkRecord.

        Args:
            chunk_id: Unique chunk identifier
            document_id: Parent document identifier
            content: Chunk text content (auto-generated if empty)
            section_title: Section title for chunk
            chunk_type: Type of chunk (content, header, etc.)
            source_file: Source file name
            word_count: Word count (auto-calculated if None)
            char_count: Character count (auto-calculated if None)
            library: Library/collection name
            embedding: Embedding vector
            entities: Named entities
            concepts: Concept tags
            metadata: Additional metadata

        Returns:
            ChunkRecord instance
        """
        # Auto-generate content if not provided
        if not content:
            content = f"This is test content for {chunk_id}. It contains multiple sentences. This helps test chunking and retrieval."

        # Auto-calculate counts if not provided
        if word_count is None:
            word_count = len(content.split())
        if char_count is None:
            char_count = len(content)

        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            section_title=section_title,
            chunk_type=chunk_type,
            source_file=source_file,
            word_count=word_count,
            char_count=char_count,
            library=library,
            embedding=embedding,
            entities=entities or [],
            concepts=concepts or [],
            metadata=metadata or {},
        )

    return _make_chunk


@pytest.fixture
def sample_chunks(make_chunk) -> List[ChunkRecord]:
    """Generate a list of sample chunks for testing.

    Returns:
        List of 5 ChunkRecord instances with varied content

    Example:
        def test_batch_processing(sample_chunks):
            assert len(sample_chunks) == 5
    """
    return [
        make_chunk("chunk_1", content="Python is a programming language"),
        make_chunk("chunk_2", content="Machine learning uses statistical models"),
        make_chunk("chunk_3", content="Natural language processing analyzes text"),
        make_chunk("chunk_4", content="Vector databases store embeddings"),
        make_chunk("chunk_5", content="Semantic search finds relevant content"),
    ]


# ============================================================================
# Sample Data Fixtures - Artifacts (j)
# ============================================================================


@pytest.fixture
def make_chunk_artifact():
    """Factory fixture for creating test IFChunkArtifact instances.

    j: Preferred fixture for new tests.
    Use this instead of make_chunk for new artifact-aware tests.

    Returns:
        Function to create IFChunkArtifact instances

    Example:
        def test_artifacts(make_chunk_artifact):
            artifact = make_chunk_artifact("chunk_1", content="Test content")
            assert artifact.artifact_id == "chunk_1"
    """

    def _make_artifact(
        artifact_id: str = "",
        document_id: str = "test_doc",
        content: str = "",
        chunk_index: int = 0,
        total_chunks: int = 1,
        parent_id: Optional[str] = None,
        root_artifact_id: Optional[str] = None,
        lineage_depth: int = 0,
        provenance: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IFChunkArtifact:
        """Create a test IFChunkArtifact.

        Args:
            artifact_id: Unique artifact ID (auto-generated if empty)
            document_id: Parent document identifier
            content: Chunk content (auto-generated if empty)
            chunk_index: Index within document
            total_chunks: Total chunks in document
            parent_id: ID of parent artifact
            root_artifact_id: ID of root artifact in lineage chain
            lineage_depth: Depth in lineage chain
            provenance: List of processor IDs that created/modified this
            metadata: Additional metadata

        Returns:
            IFChunkArtifact instance
        """
        import uuid

        if not artifact_id:
            artifact_id = f"test-chunk-{uuid.uuid4().hex[:8]}"

        if not content:
            content = f"Test content for artifact {artifact_id}. Contains multiple sentences for testing."

        final_metadata = {
            "section_title": "Test Section",
            "chunk_type": "content",
            "source_file": "test.txt",
            "word_count": len(content.split()),
            "library": "default",
        }
        if metadata:
            final_metadata.update(metadata)

        return IFChunkArtifact(
            artifact_id=artifact_id,
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            parent_id=parent_id,
            root_artifact_id=root_artifact_id,
            lineage_depth=lineage_depth,
            provenance=provenance or [],
            metadata=final_metadata,
        )

    return _make_artifact


@pytest.fixture
def sample_chunk_artifact(make_chunk_artifact) -> IFChunkArtifact:
    """Single sample IFChunkArtifact for basic tests.

    j: Preferred fixture for simple artifact tests.

    Returns:
        IFChunkArtifact instance with default values
    """
    return make_chunk_artifact(
        artifact_id="sample-artifact",
        content="This is sample artifact content for unit tests.",
    )


@pytest.fixture
def sample_artifacts(make_chunk_artifact) -> List[IFChunkArtifact]:
    """Generate a list of sample artifacts for batch testing.

    j: Use this instead of sample_chunks for new tests.

    Returns:
        List of 5 IFChunkArtifact instances with varied content
    """
    return [
        make_chunk_artifact("artifact_1", content="Python is a programming language"),
        make_chunk_artifact(
            "artifact_2", content="Machine learning uses statistical models"
        ),
        make_chunk_artifact(
            "artifact_3", content="Natural language processing analyzes text"
        ),
        make_chunk_artifact("artifact_4", content="Vector databases store embeddings"),
        make_chunk_artifact(
            "artifact_5", content="Semantic search finds relevant content"
        ),
    ]


@pytest.fixture
def make_text_artifact():
    """Factory fixture for creating test IFTextArtifact instances.

    j: Use for text extraction stage testing.

    Returns:
        Function to create IFTextArtifact instances
    """

    def _make_artifact(
        artifact_id: str = "",
        content: str = "",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IFTextArtifact:
        """Create a test IFTextArtifact."""
        import uuid

        if not artifact_id:
            artifact_id = f"test-text-{uuid.uuid4().hex[:8]}"

        if not content:
            content = "This is sample extracted text content.\n\nIt has multiple paragraphs for testing."

        return IFTextArtifact(
            artifact_id=artifact_id,
            content=content,
            parent_id=parent_id,
            metadata=metadata or {},
        )

    return _make_artifact


@pytest.fixture
def sample_text_artifact(make_text_artifact) -> IFTextArtifact:
    """Single sample IFTextArtifact for basic tests.

    Returns:
        IFTextArtifact instance
    """
    return make_text_artifact(
        artifact_id="sample-text",
        content="Sample extracted text for testing purposes.",
    )


@pytest.fixture
def artifact_with_lineage(make_chunk_artifact) -> IFChunkArtifact:
    """Create an artifact with full lineage chain for testing.

    j: Tests lineage tracking features.

    Returns:
        IFChunkArtifact with parent_id, root_artifact_id, provenance
    """
    return make_chunk_artifact(
        artifact_id="child-artifact",
        content="Content with lineage tracking",
        parent_id="parent-text-artifact",
        root_artifact_id="root-file-artifact",
        lineage_depth=2,
        provenance=["file_loader", "text_extractor", "semantic_chunker"],
    )


# ============================================================================
# Sample Data Fixtures - Search Results
# ============================================================================


@pytest.fixture
def make_search_result():
    """Factory fixture for creating SearchResult instances.

    Returns:
        Function to create SearchResult instances

    Example:
        def test_results(make_search_result):
            result = make_search_result("chunk_1", score=0.95)
            assert result.score == 0.95
    """

    def _make_search_result(
        chunk_id: str,
        score: float = 0.85,
        content: str = "",
        document_id: str = "test_doc",
        section_title: str = "Test Section",
        chunk_type: str = "content",
        source_file: str = "test.txt",
        word_count: int = 10,
        metadata: Optional[Dict[str, Any]] = None,
        library: str = "default",
    ) -> SearchResult:
        """Create a test SearchResult.

        Args:
            chunk_id: Unique chunk identifier
            score: Relevance score (0-1)
            content: Chunk content
            document_id: Parent document identifier
            section_title: Section title
            chunk_type: Chunk type
            source_file: Source file name
            word_count: Word count
            metadata: Additional metadata
            library: Library/collection name

        Returns:
            SearchResult instance
        """
        if not content:
            content = f"Content for {chunk_id}"

        return SearchResult(
            chunk_id=chunk_id,
            score=score,
            content=content,
            document_id=document_id,
            section_title=section_title,
            chunk_type=chunk_type,
            source_file=source_file,
            word_count=word_count,
            metadata=metadata or {},
            library=library,
        )

    return _make_search_result


# ============================================================================
# Sample Data Fixtures - Embeddings
# ============================================================================


@pytest.fixture
def sample_embedding() -> List[float]:
    """Generate a sample embedding vector.

    Returns:
        384-dimensional embedding vector (all-MiniLM-L6-v2 size)

    Example:
        def test_embedding(sample_embedding):
            assert len(sample_embedding) == 384
    """
    # Generate deterministic embedding for testing
    np.random.seed(42)
    embedding = np.random.randn(384).astype(np.float32)
    # Normalize to unit length (typical for embeddings)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


@pytest.fixture
def make_embedding():
    """Factory fixture for creating embedding vectors.

    Returns:
        Function to create embedding vectors with optional seed

    Example:
        def test_embeddings(make_embedding):
            emb1 = make_embedding(seed=1)
            emb2 = make_embedding(seed=2)
            assert emb1 != emb2
    """

    def _make_embedding(
        dimensions: int = 384,
        seed: Optional[int] = None,
        normalize: bool = True,
    ) -> List[float]:
        """Create an embedding vector.

        Args:
            dimensions: Number of dimensions
            seed: Random seed for reproducibility
            normalize: Whether to normalize to unit length

        Returns:
            Embedding vector as list of floats
        """
        if seed is not None:
            np.random.seed(seed)

        embedding = np.random.randn(dimensions).astype(np.float32)

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.tolist()

    return _make_embedding


# ============================================================================
# Sample Data Fixtures - Text Files
# ============================================================================


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to created file
    """
    file_path = temp_dir / "sample.txt"
    file_path.write_text(
        "This is a sample text file.\n"
        "Line 2 contains more content.\n"
        "Line 3 has even more text.",
        encoding="utf-8",
    )
    return file_path


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create a sample markdown file for testing.

    Returns:
        Path to created markdown file
    """
    file_path = temp_dir / "sample.md"
    content = """# Main Title

This is an introduction paragraph with some content.

## Section 1

Content for section 1 goes here. It has multiple sentences.
This helps test section extraction.

### Subsection 1.1

Nested content here.

## Section 2

More content in section 2.

- Bullet point 1
- Bullet point 2
- Bullet point 3
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_json_file(temp_dir: Path) -> Path:
    """Create a sample JSON file for testing.

    Returns:
        Path to created JSON file
    """
    file_path = temp_dir / "sample.json"
    data = {
        "title": "Test Document",
        "content": "This is test content",
        "metadata": {
            "author": "Test Author",
            "date": "2024-01-01",
        },
        "chunks": [
            {"id": "chunk_1", "text": "First chunk"},
            {"id": "chunk_2", "text": "Second chunk"},
        ],
    }
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return file_path


# ============================================================================
# Mock Embedding Model Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding_model(make_embedding) -> Mock:
    """Create a mock embedding model (SentenceTransformer interface).

    Returns:
        Mock with encode() method returning sample embeddings

    Example:
        def test_embedding_generation(mock_embedding_model):
            emb = mock_embedding_model.encode("test text")
            assert len(emb) == 384
    """
    model = Mock()

    def mock_encode(text, convert_to_numpy=True, **kwargs):
        # Generate embedding based on text hash for consistency
        seed = hash(text) % 10000
        embedding = make_embedding(seed=seed)
        if convert_to_numpy:
            return np.array(embedding)
        return embedding

    model.encode.side_effect = mock_encode
    return model


# ============================================================================
# Pytest Configuration
# ============================================================================


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests.

    This prevents log pollution between tests.
    """
    import logging

    # Get root logger
    root = logging.getLogger()

    # Remove all handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Reset level
    root.setLevel(logging.WARNING)

    yield

    # Cleanup after test
    for handler in root.handlers[:]:
        root.removeHandler(handler)


def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require external API access"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require ML models"
    )


# ============================================================================
# Safe Cleanup Fixture
# ============================================================================


@pytest.fixture
def safe_cleanup(temp_dir: Path) -> Generator[Path, None, None]:
    """Fixture that ensures temp files are deleted even on test failure.

    Uses a dedicated temp directory and aggressively cleans up on teardown,
    including handling of locked files on Windows.

    Yields:
        Path to a clean temporary directory

    Example:
        def test_with_cleanup(safe_cleanup):
            test_file = safe_cleanup / "test.txt"
            test_file.write_text("data")
            # File will be deleted even if test fails
    """
    import shutil
    import time

    yield temp_dir

    # Aggressive cleanup on teardown
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            break
        except (PermissionError, OSError):
            # On Windows, files may be locked briefly
            time.sleep(0.1 * (attempt + 1))


# ============================================================================
# Mock Multi-Process Environment Fixture
# ============================================================================


@pytest.fixture
def mock_multiprocess_env(temp_dir: Path) -> Generator[Dict[str, Any], None, None]:
    """Fixture for testing StateManager locking in multi-process scenarios.

    Provides a simulated multi-process environment with:
    - Shared state file path
    - Lock file path
    - Helper functions for simulating concurrent access

    Yields:
        Dict with state_file, lock_file, and helper functions

    Example:
        def test_concurrent_access(mock_multiprocess_env):
            state_file = mock_multiprocess_env["state_file"]
            with mock_multiprocess_env["acquire_lock"]():
                # Simulate locked state
                pass
    """
    import threading
    from contextlib import contextmanager

    state_file = temp_dir / "state.json"
    lock_file = temp_dir / "state.lock"

    # Initialize state file
    state_file.write_text("{}", encoding="utf-8")
    lock_file.touch()

    # Lock tracking
    main_lock = threading.Lock()

    @contextmanager
    def acquire_lock():
        """Simulate acquiring a file lock."""
        with main_lock:
            yield

    @contextmanager
    def simulate_concurrent_writer(data: Dict[str, Any]):
        """Simulate another process writing to state file."""
        with acquire_lock():
            state_file.write_text(json.dumps(data), encoding="utf-8")
            yield

    def read_state() -> Dict[str, Any]:
        """Read current state."""
        return json.loads(state_file.read_text(encoding="utf-8"))

    def write_state(data: Dict[str, Any]) -> None:
        """Write state."""
        state_file.write_text(json.dumps(data), encoding="utf-8")

    yield {
        "state_file": state_file,
        "lock_file": lock_file,
        "acquire_lock": acquire_lock,
        "simulate_concurrent_writer": simulate_concurrent_writer,
        "read_state": read_state,
        "write_state": write_state,
        "temp_dir": temp_dir,
    }


# ============================================================================
# Mock spaCy Fixture
# ============================================================================


@pytest.fixture
def mock_spacy() -> Generator[Mock, None, None]:
    """Mock spaCy to avoid downloading large language models.

    Provides a mock spaCy module with:
    - load() function returning a mock Language object
    - Mock Doc, Token, and Span classes
    - Mock entity recognition

    Yields:
        Mock spaCy module

    Example:
        def test_ner(mock_spacy, monkeypatch):
            monkeypatch.setattr("spacy.load", mock_spacy.load)
            nlp = mock_spacy.load("en_core_web_sm")
            doc = nlp("Apple Inc. is based in California.")
    """

    mock_spacy_module = MagicMock()

    # Mock Token
    mock_token = MagicMock()
    mock_token.text = "word"
    mock_token.lemma_ = "word"
    mock_token.pos_ = "NOUN"
    mock_token.dep_ = "nsubj"
    mock_token.is_stop = False
    mock_token.is_alpha = True

    # Mock Span (for entities)
    mock_entity = MagicMock()
    mock_entity.text = "Apple Inc."
    mock_entity.label_ = "ORG"
    mock_entity.start = 0
    mock_entity.end = 2

    # Mock Doc
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_token])
    mock_doc.__len__ = lambda self: 1
    mock_doc.ents = [mock_entity]
    mock_doc.text = "Apple Inc. is based in California."
    mock_doc.sents = [mock_doc]  # Simplified: doc is its own sentence

    # Mock Language (nlp object)
    mock_nlp = MagicMock()
    mock_nlp.return_value = mock_doc
    mock_nlp.pipe_names = ["tok2vec", "tagger", "parser", "ner"]

    # Mock load function
    def mock_load(model_name: str, **kwargs):
        return mock_nlp

    mock_spacy_module.load = mock_load
    mock_spacy_module.blank = lambda lang: mock_nlp

    yield mock_spacy_module


# ============================================================================
# Mock Sentence Transformers Fixture
# ============================================================================


@pytest.fixture
def mock_sentence_transformers(make_embedding) -> Generator[Mock, None, None]:
    """Mock sentence-transformers to avoid downloading large models.

    Provides a mock SentenceTransformer class that returns consistent
    embeddings based on input text hash.

    Yields:
        Mock sentence_transformers module

    Example:
        def test_embeddings(mock_sentence_transformers, monkeypatch):
            monkeypatch.setattr(
                "sentence_transformers.SentenceTransformer",
                mock_sentence_transformers.SentenceTransformer
            )
            model = mock_sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
            embedding = model.encode("test text")
    """
    mock_st_module = MagicMock()

    class MockSentenceTransformer:
        """Mock SentenceTransformer class."""

        def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
            self.model_name = model_name
            self._dimension = 384

        def encode(
            self,
            sentences,
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            **kwargs,
        ):
            """Generate mock embeddings based on text hash."""
            if isinstance(sentences, str):
                sentences = [sentences]

            embeddings = []
            for text in sentences:
                seed = hash(text) % 10000
                emb = make_embedding(dimensions=self._dimension, seed=seed)
                embeddings.append(emb)

            result = np.array(embeddings)
            return result if convert_to_numpy else result.tolist()

        def get_sentence_embedding_dimension(self) -> int:
            """Return embedding dimension."""
            return self._dimension

    mock_st_module.SentenceTransformer = MockSentenceTransformer

    yield mock_st_module
