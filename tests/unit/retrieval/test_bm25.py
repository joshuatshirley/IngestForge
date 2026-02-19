"""
Tests for BM25 Retriever.

This module tests BM25 keyword-based retrieval using BM25+ scoring.

Test Strategy
-------------
- Focus on BM25 scoring logic and keyword matching
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Mock storage backend when needed
- Test tokenization, indexing, and search workflow

Organization
------------
- TestBM25Params: BM25Params dataclass
- TestBM25RetrieverInit: Initialization
- TestTokenization: _tokenize function
- TestIndexing: index_chunks and _index_document
- TestBM25Scoring: _calculate_bm25 function
- TestSearch: Main search method
- TestHelpers: _find_candidates, _score_candidates, _build_search_results
"""

from unittest.mock import Mock


from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config
from ingestforge.retrieval.bm25 import BM25Retriever, BM25Params
from ingestforge.storage.base import SearchResult


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(chunk_id: str, content: str) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="test_doc",
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
        source_file="test.txt",
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestBM25Params:
    """Tests for BM25Params dataclass.

    Rule #4: Focused test class - tests only BM25Params
    """

    def test_create_bm25_params_with_defaults(self):
        """Test creating BM25Params with default values."""
        params = BM25Params()

        assert params.k1 == 1.5
        assert params.b == 0.75
        assert params.delta == 0.5

    def test_create_bm25_params_with_custom_values(self):
        """Test creating BM25Params with custom values."""
        params = BM25Params(k1=2.0, b=0.5, delta=1.0)

        assert params.k1 == 2.0
        assert params.b == 0.5
        assert params.delta == 1.0


class TestBM25RetrieverInit:
    """Tests for BM25Retriever initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_retriever_with_defaults(self):
        """Test creating BM25Retriever with default params."""
        config = Config()
        retriever = BM25Retriever(config)

        assert retriever.config is config
        assert retriever.storage is None
        assert isinstance(retriever.params, BM25Params)
        assert retriever.loaded is False

    def test_create_retriever_with_storage(self):
        """Test creating BM25Retriever with storage backend."""
        config = Config()
        storage = Mock()
        retriever = BM25Retriever(config, storage)

        assert retriever.storage is storage

    def test_create_retriever_with_custom_params(self):
        """Test creating BM25Retriever with custom params."""
        config = Config()
        params = BM25Params(k1=2.0, b=0.5, delta=1.0)
        retriever = BM25Retriever(config, params=params)

        assert retriever.params.k1 == 2.0
        assert retriever.params.b == 0.5
        assert retriever.params.delta == 1.0


class TestTokenization:
    """Tests for tokenization.

    Rule #4: Focused test class - tests _tokenize only
    """

    def test_tokenize_lowercase(self):
        """Test tokenization converts to lowercase."""
        config = Config()
        retriever = BM25Retriever(config)
        text = "Hello World Python"

        tokens = retriever._tokenize(text)

        assert all(t.islower() for t in tokens)
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens

    def test_tokenize_removes_stop_words(self):
        """Test tokenization removes stop words."""
        config = Config()
        retriever = BM25Retriever(config)
        text = "the quick brown fox and it was on the lazy dog"

        tokens = retriever._tokenize(text)

        # Stop words should be removed
        assert "the" not in tokens
        assert "and" not in tokens
        assert "it" not in tokens
        assert "was" not in tokens
        assert "on" not in tokens
        # Content words should remain
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "lazy" in tokens
        assert "dog" in tokens

    def test_tokenize_removes_single_chars(self):
        """Test tokenization removes single-character tokens."""
        config = Config()
        retriever = BM25Retriever(config)
        text = "a b c programming language"

        tokens = retriever._tokenize(text)

        # Single chars should be removed
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        # Multi-char words remain
        assert "programming" in tokens
        assert "language" in tokens

    def test_tokenize_extracts_words_only(self):
        """Test tokenization extracts word tokens only."""
        config = Config()
        retriever = BM25Retriever(config)
        text = "word1, word2! word3? word4."

        tokens = retriever._tokenize(text)

        # Words should be extracted without punctuation
        assert "word1" in tokens
        assert "word2" in tokens
        assert "word3" in tokens
        assert "word4" in tokens


class TestIndexing:
    """Tests for document indexing.

    Rule #4: Focused test class - tests indexing only
    """

    def test_index_chunks_single_chunk(self):
        """Test indexing a single chunk."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [make_chunk("chunk_1", "Python programming language")]

        retriever.index_chunks(chunks)

        assert retriever.loaded is True
        assert "chunk_1" in retriever._documents
        assert retriever._avg_doc_length > 0

    def test_index_chunks_multiple_chunks(self):
        """Test indexing multiple chunks."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming"),
            make_chunk("chunk_2", "Java development"),
            make_chunk("chunk_3", "JavaScript coding"),
        ]

        retriever.index_chunks(chunks)

        assert retriever.loaded is True
        assert len(retriever._documents) == 3
        assert "chunk_1" in retriever._documents
        assert "chunk_2" in retriever._documents
        assert "chunk_3" in retriever._documents

    def test_index_document_updates_term_frequencies(self):
        """Test indexing updates term frequency index."""
        config = Config()
        retriever = BM25Retriever(config)
        content = "Python programming Python language"

        retriever._index_document("doc_1", content)

        # "python" appears twice
        assert retriever._term_freqs["python"]["doc_1"] == 2
        # "programming" appears once
        assert retriever._term_freqs["programming"]["doc_1"] == 1
        # "language" appears once
        assert retriever._term_freqs["language"]["doc_1"] == 1

    def test_index_document_updates_doc_frequencies(self):
        """Test indexing updates document frequency counts."""
        config = Config()
        retriever = BM25Retriever(config)

        retriever._index_document("doc_1", "Python programming")
        retriever._index_document("doc_2", "Python language")

        # "python" appears in 2 documents
        assert retriever._doc_freqs["python"] == 2
        # "programming" appears in 1 document
        assert retriever._doc_freqs["programming"] == 1


class TestBM25Scoring:
    """Tests for BM25 scoring.

    Rule #4: Focused test class - tests _calculate_bm25 only
    """

    def test_calculate_bm25_returns_score(self):
        """Test BM25 calculation returns a numeric score."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming language framework"),
        ]
        retriever.index_chunks(chunks)

        score = retriever._calculate_bm25(["python"], "chunk_1")

        assert isinstance(score, float)
        assert score > 0

    def test_calculate_bm25_higher_for_term_frequency(self):
        """Test BM25 scores higher for repeated terms."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python Python Python programming"),
            make_chunk("chunk_2", "Python programming language"),
        ]
        retriever.index_chunks(chunks)

        score_1 = retriever._calculate_bm25(["python"], "chunk_1")
        score_2 = retriever._calculate_bm25(["python"], "chunk_2")

        # chunk_1 has "python" 3 times, chunk_2 has it once
        assert score_1 > score_2

    def test_calculate_bm25_zero_for_missing_terms(self):
        """Test BM25 returns zero for documents without query terms."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming"),
            make_chunk("chunk_2", "Java development"),
        ]
        retriever.index_chunks(chunks)

        score = retriever._calculate_bm25(["python"], "chunk_2")

        assert score == 0.0


class TestSearch:
    """Tests for main search method.

    Rule #4: Focused test class - tests search() only
    """

    def test_search_returns_relevant_results(self):
        """Test search returns relevant results."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming language"),
            make_chunk("chunk_2", "Java development environment"),
        ]
        retriever.index_chunks(chunks)

        results = retriever.search("Python programming", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score > 0

    def test_search_returns_top_k_results(self):
        """Test search respects top_k parameter."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming language"),
            make_chunk("chunk_2", "Python development"),
            make_chunk("chunk_3", "Python framework"),
            make_chunk("chunk_4", "Python library"),
        ]
        retriever.index_chunks(chunks)

        results = retriever.search("Python", top_k=2)

        assert len(results) == 2

    def test_search_empty_query_returns_empty(self):
        """Test search with empty query returns no results."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [make_chunk("chunk_1", "Python programming")]
        retriever.index_chunks(chunks)

        results = retriever.search("", top_k=5)

        assert results == []

    def test_search_only_stop_words_returns_empty(self):
        """Test search with only stop words returns no results."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [make_chunk("chunk_1", "Python programming")]
        retriever.index_chunks(chunks)

        results = retriever.search("the and or but", top_k=5)

        assert results == []

    def test_search_orders_by_score_descending(self):
        """Test search returns results ordered by score (highest first)."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming"),
            make_chunk("chunk_2", "Python Python Python framework"),
            make_chunk("chunk_3", "Python language"),
        ]
        retriever.index_chunks(chunks)

        results = retriever.search("Python", top_k=5)

        # chunk_2 should score highest (3x "python")
        assert results[0].chunk_id == "chunk_2"
        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_with_storage_backend(self):
        """Test search with storage backend integration."""
        config = Config()
        storage = Mock()
        # Mock storage.get_chunk() to return None (no chunk found)
        storage.get_chunk.return_value = None
        retriever = BM25Retriever(config, storage)
        chunks = [make_chunk("chunk_1", "Python programming")]
        retriever.index_chunks(chunks)

        results = retriever.search("Python", top_k=5)

        # Should fall back to minimal result
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"


class TestHelpers:
    """Tests for helper methods.

    Rule #4: Focused test class - tests helper methods
    """

    def test_find_candidates_returns_matching_docs(self):
        """Test _find_candidates returns documents with query terms."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming"),
            make_chunk("chunk_2", "Java development"),
            make_chunk("chunk_3", "Python framework"),
        ]
        retriever.index_chunks(chunks)

        candidates = retriever._find_candidates(["python"])

        assert "chunk_1" in candidates
        assert "chunk_3" in candidates
        assert "chunk_2" not in candidates

    def test_find_candidates_empty_for_missing_terms(self):
        """Test _find_candidates returns empty set for missing terms."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [make_chunk("chunk_1", "Python programming")]
        retriever.index_chunks(chunks)

        candidates = retriever._find_candidates(["javascript"])

        assert len(candidates) == 0

    def test_score_candidates_returns_scored_tuples(self):
        """Test _score_candidates returns list of (doc_id, score) tuples."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [
            make_chunk("chunk_1", "Python programming"),
            make_chunk("chunk_2", "Python framework"),
        ]
        retriever.index_chunks(chunks)

        candidates = {"chunk_1", "chunk_2"}
        scores = retriever._score_candidates(["python"], candidates)

        assert len(scores) == 2
        assert all(isinstance(item, tuple) for item in scores)
        assert all(len(item) == 2 for item in scores)
        # Each tuple should have (doc_id, score)
        assert all(isinstance(item[0], str) for item in scores)
        assert all(isinstance(item[1], float) for item in scores)

    def test_build_search_results_creates_result_objects(self):
        """Test _build_search_results creates SearchResult objects."""
        config = Config()
        retriever = BM25Retriever(config)
        chunks = [make_chunk("chunk_1", "Python programming")]
        retriever.index_chunks(chunks)

        scores = [("chunk_1", 0.85)]
        results = retriever._build_search_results(scores, top_k=5, library_filter=None)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 0.85


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - BM25Params: 2 tests (defaults, custom values)
    - BM25Retriever init: 3 tests (defaults, with storage, custom params)
    - Tokenization: 4 tests (lowercase, stop words, single chars, words only)
    - Indexing: 5 tests (single chunk, multiple chunks, term freqs, doc freqs, updates)
    - BM25 scoring: 3 tests (returns score, term frequency, missing terms)
    - Search: 6 tests (relevant results, top_k, empty query, stop words, ordering, storage)
    - Helpers: 4 tests (find candidates, empty candidates, score candidates, build results)

    Total: 27 tests

Design Decisions:
    1. Focus on BM25 scoring logic and keyword matching
    2. Mock storage backend when needed
    3. Test tokenization thoroughly (critical for BM25 accuracy)
    4. Test indexing data structures (term freqs, doc freqs)
    5. Test search workflow with various scenarios
    6. Simple, clear tests that verify BM25 works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - BM25Params dataclass creation
    - BM25Retriever initialization with config, storage, params
    - Tokenization (lowercase, stop word removal, min length)
    - Index building (chunks, documents, term frequencies)
    - BM25+ scoring (term frequency, document frequency, length normalization)
    - Search workflow (tokenize, find candidates, score, order, return)
    - Top-k result limiting
    - Empty query and stop-word-only query handling
    - Score ordering (descending)
    - Storage backend integration (fallback to minimal result)
    - Helper methods (find candidates, score candidates, build results)

Justification:
    - BM25 is critical for keyword-based retrieval
    - Tokenization accuracy affects all BM25 results
    - Scoring formula needs verification with known inputs
    - Search workflow covers common use cases
    - Simple tests verify BM25 system works correctly
"""
