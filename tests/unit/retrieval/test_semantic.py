"""
Tests for Semantic Retriever.

This module tests semantic vector search using embedding similarity.

Test Strategy
-------------
- Focus on retrieval logic and embedding generation
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Mock external dependencies (sentence_transformers, storage)
- Test search workflow and fallback behavior

Organization
------------
- TestSemanticRetrieverInit: Initialization
- TestEmbedQuery: Query embedding generation
- TestSearch: Main search method and fallback
"""

from unittest.mock import Mock, patch
import numpy as np


from ingestforge.core.config import Config
from ingestforge.retrieval.semantic import SemanticRetriever
from ingestforge.storage.base import SearchResult


# ============================================================================
# Test Helpers
# ============================================================================


def make_search_result(chunk_id: str, score: float = 0.85) -> SearchResult:
    """Create a test SearchResult."""
    return SearchResult(
        chunk_id=chunk_id,
        content=f"Content for {chunk_id}",
        score=score,
        document_id="test_doc",
        section_title="Test Section",
        chunk_type="text",
        source_file="test.txt",
        word_count=10,
        metadata={},
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestSemanticRetrieverInit:
    """Tests for SemanticRetriever initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_semantic_retriever(self):
        """Test creating SemanticRetriever."""
        config = Config()
        storage = Mock()

        retriever = SemanticRetriever(config, storage)

        assert retriever.config is config
        assert retriever.storage is storage
        assert retriever._embedding_model is None

    def test_embedding_model_lazy_loading(self):
        """Test embedding model is lazy-loaded."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock SentenceTransformer from sentence_transformers module
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model

            # Access embedding_model property
            model = retriever.embedding_model

            # Should load and cache model
            assert model is mock_model
            assert retriever._embedding_model is mock_model
            mock_st.assert_called_once()


class TestEmbedQuery:
    """Tests for query embedding generation.

    Rule #4: Focused test class - tests embed_query only
    """

    def test_embed_query_generates_embedding(self):
        """Test embed_query generates embedding vector."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.encode.return_value = mock_embedding
        retriever._embedding_model = mock_model

        result = retriever.embed_query("test query")

        # Should return list of floats
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(x, float) for x in result)
        mock_model.encode.assert_called_once_with("test query", convert_to_numpy=True)

    def test_embed_query_calls_model_correctly(self):
        """Test embed_query calls model with correct parameters."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        retriever._embedding_model = mock_model

        retriever.embed_query("sample query text")

        # Should call encode with convert_to_numpy=True
        mock_model.encode.assert_called_once_with(
            "sample query text", convert_to_numpy=True
        )

    def test_embed_query_returns_correct_format(self):
        """Test embed_query returns embedding as list."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_embedding = np.array([1.0, 2.0, 3.0])
        mock_model.encode.return_value = mock_embedding
        retriever._embedding_model = mock_model

        result = retriever.embed_query("query")

        # Should convert numpy array to list
        assert result == [1.0, 2.0, 3.0]


class TestSearch:
    """Tests for semantic search method.

    Rule #4: Focused test class - tests search only
    """

    def test_search_with_query_embedding(self):
        """Test search generates query embedding and searches."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        retriever._embedding_model = mock_model

        # Mock storage search
        expected_results = [
            make_search_result("chunk_1", 0.95),
            make_search_result("chunk_2", 0.85),
        ]
        storage.search_semantic.return_value = expected_results

        results = retriever.search("test query", top_k=5)

        # Should call storage.search_semantic with embedding
        assert results == expected_results
        storage.search_semantic.assert_called_once()
        call_args = storage.search_semantic.call_args
        assert call_args[0][0] == [0.1, 0.2, 0.3]  # embedding
        assert call_args[1]["top_k"] == 5

    def test_search_without_query_embedding(self):
        """Test search without embedding (text search)."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock storage search
        expected_results = [make_search_result("chunk_1")]
        storage.search.return_value = expected_results

        results = retriever.search("test query", top_k=3, use_query_embedding=False)

        # Should call storage.search (text search)
        assert results == expected_results
        storage.search.assert_called_once_with(
            "test query", top_k=3, library_filter=None
        )
        # Should NOT call search_semantic
        storage.search_semantic.assert_not_called()

    def test_search_with_library_filter(self):
        """Test search with library filter."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        retriever._embedding_model = mock_model

        # Mock storage search
        storage.search_semantic.return_value = []

        retriever.search("query", top_k=5, library_filter="test_library")

        # Should pass library_filter to storage
        call_args = storage.search_semantic.call_args
        assert call_args[1]["library_filter"] == "test_library"

    def test_search_fallback_when_semantic_not_supported(self):
        """Test fallback to text search when semantic not supported."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        retriever._embedding_model = mock_model

        # Mock storage to raise NotImplementedError for semantic search
        storage.search_semantic.side_effect = NotImplementedError("Not supported")
        expected_results = [make_search_result("chunk_1")]
        storage.search.return_value = expected_results

        results = retriever.search("query", top_k=5)

        # Should fall back to text search
        assert results == expected_results
        storage.search.assert_called_once_with("query", top_k=5, library_filter=None)

    def test_search_passes_kwargs_to_storage(self):
        """Test search passes extra kwargs to storage."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        retriever._embedding_model = mock_model

        # Mock storage search
        storage.search_semantic.return_value = []

        retriever.search("query", top_k=5, custom_param="value", another_param=123)

        # Should pass kwargs to storage
        call_args = storage.search_semantic.call_args
        assert call_args[1]["custom_param"] == "value"
        assert call_args[1]["another_param"] == 123

    def test_search_respects_top_k_parameter(self):
        """Test search respects top_k parameter."""
        config = Config()
        storage = Mock()
        retriever = SemanticRetriever(config, storage)

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        retriever._embedding_model = mock_model

        # Mock storage search
        storage.search_semantic.return_value = []

        retriever.search("query", top_k=15)

        # Should pass top_k to storage
        call_args = storage.search_semantic.call_args
        assert call_args[1]["top_k"] == 15


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - SemanticRetriever init: 2 tests (creation, lazy loading)
    - embed_query: 3 tests (generates embedding, calls model, returns list)
    - search: 7 tests (with embedding, without, filter, fallback, kwargs, top_k)

    Total: 12 tests

Design Decisions:
    1. Focus on retrieval logic and embedding generation
    2. Mock external dependencies (sentence_transformers, storage)
    3. Test search workflow with and without embeddings
    4. Test fallback behavior when semantic search not supported
    5. Simple, clear tests that verify semantic retrieval works
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - SemanticRetriever initialization with config and storage
    - Lazy loading of embedding model (SentenceTransformer)
    - Query embedding generation with convert_to_numpy
    - Embedding conversion to list format
    - Semantic search with query embedding
    - Text search fallback when use_query_embedding=False
    - Library filter parameter passing
    - Fallback to text search when semantic not supported
    - Extra kwargs forwarding to storage
    - top_k parameter handling

Justification:
    - Semantic retrieval is critical for conceptual search
    - Embedding generation needs verification
    - Fallback behavior ensures graceful degradation
    - Mock external dependencies for unit testing
    - Simple tests verify semantic retrieval system works correctly
"""
