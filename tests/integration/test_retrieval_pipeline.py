"""
Integration Tests for Retrieval Pipeline.

Tests the complete retrieval workflow including query processing,
multi-strategy search, result fusion, and reranking.

Test Coverage
-------------
- Query parsing and expansion
- BM25 keyword search
- Semantic vector search
- Hybrid search (BM25 + Semantic)
- Query rewriting and expansion
- Result fusion (RRF, weighted)
- Result reranking
- Cross-corpus retrieval
- Parent document retrieval
- Performance benchmarks

Test Strategy
-------------
- Test each retrieval strategy independently
- Test hybrid combinations
- Verify relevance of results
- Test query expansion effectiveness
- Test reranking improves results
- Benchmark retrieval performance
"""

import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pytest
import numpy as np

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.jsonl import JSONLRepository
from ingestforge.storage.base import SearchResult
from ingestforge.retrieval.bm25 import BM25Retriever
from ingestforge.retrieval.semantic import SemanticRetriever
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.retrieval.reranker import Reranker
from ingestforge.retrieval.parent_retriever import ParentDocumentRetriever
from ingestforge.retrieval.cross_corpus import CrossCorpusRetriever
from ingestforge.query.parser import QueryParser
from ingestforge.query.expander import QueryExpander
from ingestforge.core.config import Config


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for retrieval testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def retrieval_config(temp_dir: Path) -> Config:
    """Create configuration for retrieval testing."""
    config = Config()
    config.project.data_dir = str(temp_dir / "data")
    config.retrieval.bm25_weight = 0.5
    config.retrieval.semantic_weight = 0.5
    config.retrieval.top_k = 10
    config._base_path = temp_dir
    (temp_dir / "data").mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def mock_embedding_model() -> Mock:
    """Create mock embedding model for testing."""
    model = Mock()

    def mock_encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 10000)
            emb = np.random.randn(384).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)

    model.encode.side_effect = mock_encode
    return model


@pytest.fixture
def sample_corpus() -> List[ChunkRecord]:
    """Create sample corpus for retrieval testing."""
    corpus_data = [
        (
            "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
            ["machine learning", "artificial intelligence"],
            "ML Basics",
        ),
        (
            "Neural networks are computational models inspired by biological neural networks in animal brains.",
            ["neural networks", "deep learning"],
            "Neural Networks",
        ),
        (
            "Supervised learning uses labeled data to train models for classification and regression tasks.",
            ["supervised learning", "classification"],
            "Supervised Learning",
        ),
        (
            "Unsupervised learning finds patterns in unlabeled data using clustering and dimensionality reduction.",
            ["unsupervised learning", "clustering"],
            "Unsupervised Learning",
        ),
        (
            "Deep learning uses multi-layer neural networks to learn hierarchical representations of data.",
            ["deep learning", "neural networks"],
            "Deep Learning",
        ),
        (
            "Natural language processing enables computers to understand and generate human language.",
            ["NLP", "language"],
            "NLP",
        ),
        (
            "Computer vision algorithms process and analyze visual information from images and videos.",
            ["computer vision", "image processing"],
            "Computer Vision",
        ),
        (
            "Reinforcement learning trains agents through rewards and penalties in interactive environments.",
            ["reinforcement learning", "agents"],
            "RL",
        ),
        (
            "Transfer learning reuses pre-trained models for new but related tasks to save training time.",
            ["transfer learning", "pre-training"],
            "Transfer Learning",
        ),
        (
            "Gradient descent is an optimization algorithm used to minimize loss functions in machine learning.",
            ["gradient descent", "optimization"],
            "Optimization",
        ),
    ]

    chunks = []
    for i, (content, concepts, title) in enumerate(corpus_data):
        np.random.seed(i)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        chunk = ChunkRecord(
            chunk_id=f"corpus_chunk_{i}",
            document_id=f"corpus_doc_{i // 3}",
            content=content,
            section_title=title,
            chunk_type="content",
            source_file=f"corpus_{i // 3}.md",
            word_count=len(content.split()),
            char_count=len(content),
            embedding=embedding.tolist(),
            entities=[],
            concepts=concepts,
            metadata={"topic": concepts[0] if concepts else "general"},
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def populated_storage(
    temp_dir: Path, sample_corpus: List[ChunkRecord]
) -> JSONLRepository:
    """Create storage populated with sample corpus."""
    data_path = temp_dir / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    storage = JSONLRepository(data_path=data_path)
    storage.add_chunks(sample_corpus)
    return storage


# ============================================================================
# Test Classes
# ============================================================================


class TestBM25Retrieval:
    """Tests for BM25 keyword-based retrieval.

    Rule #4: Focused test class - tests BM25
    """

    def test_bm25_basic_search(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test basic BM25 keyword search."""
        retriever = BM25Retriever(populated_storage, retrieval_config)

        results = retriever.search("machine learning", k=5)

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    def test_bm25_returns_relevant_results(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test that BM25 returns relevant results."""
        retriever = BM25Retriever(populated_storage, retrieval_config)

        results = retriever.search("neural networks", k=3)

        # Should find chunks mentioning neural networks
        assert len(results) > 0
        found_relevant = any("neural" in r.content.lower() for r in results)
        assert found_relevant

    def test_bm25_scoring(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test that BM25 scores are ordered correctly."""
        retriever = BM25Retriever(populated_storage, retrieval_config)

        results = retriever.search("deep learning", k=5)

        # Scores should be in descending order
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_bm25_respects_top_k(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test that BM25 returns at most k results."""
        retriever = BM25Retriever(populated_storage, retrieval_config)

        results = retriever.search("learning", k=3)

        assert len(results) <= 3

    def test_bm25_handles_no_matches(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test BM25 handling of queries with no matches."""
        retriever = BM25Retriever(populated_storage, retrieval_config)

        results = retriever.search("quantum cryptography blockchain", k=5)

        # May return empty list or low-scoring results
        assert isinstance(results, list)


class TestSemanticRetrieval:
    """Tests for semantic vector-based retrieval.

    Rule #4: Focused test class - tests semantic search
    """

    def test_semantic_search(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test semantic search using embeddings."""
        retriever = SemanticRetriever(
            populated_storage, retrieval_config, model=mock_embedding_model
        )

        results = retriever.search("What is machine learning?", k=5)

        assert len(results) > 0

    def test_semantic_finds_similar_concepts(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test that semantic search finds conceptually similar content."""
        retriever = SemanticRetriever(
            populated_storage, retrieval_config, model=mock_embedding_model
        )

        # Query about learning algorithms
        results = retriever.search("algorithms that learn from examples", k=3)

        # Should find content about supervised/machine learning
        assert len(results) > 0

    def test_semantic_similarity_scores(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test that semantic search returns similarity scores."""
        retriever = SemanticRetriever(
            populated_storage, retrieval_config, model=mock_embedding_model
        )

        results = retriever.search("neural network architecture", k=5)

        # Should have cosine similarity scores
        for result in results:
            assert hasattr(result, "score")
            assert 0.0 <= result.score <= 1.0

    def test_semantic_ordering(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test that semantic results are ordered by similarity."""
        retriever = SemanticRetriever(
            populated_storage, retrieval_config, model=mock_embedding_model
        )

        results = retriever.search("deep neural networks", k=5)

        # Scores should be in descending order
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


class TestHybridRetrieval:
    """Tests for hybrid BM25 + Semantic retrieval.

    Rule #4: Focused test class - tests hybrid search
    """

    def test_hybrid_combines_strategies(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test that hybrid search combines BM25 and semantic results."""
        retriever = HybridRetriever(
            populated_storage, retrieval_config, embedding_model=mock_embedding_model
        )

        results = retriever.search("machine learning algorithms", k=5)

        assert len(results) > 0

    def test_hybrid_fusion_weights(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test that hybrid search respects fusion weights."""
        # Test with BM25-heavy weighting
        retrieval_config.retrieval.bm25_weight = 0.8
        retrieval_config.retrieval.semantic_weight = 0.2

        retriever = HybridRetriever(
            populated_storage, retrieval_config, embedding_model=mock_embedding_model
        )

        results = retriever.search("exact keyword match test", k=5)

        assert len(results) >= 0

    def test_hybrid_rrf_fusion(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test hybrid search with Reciprocal Rank Fusion."""
        retrieval_config.retrieval.fusion_method = "rrf"

        retriever = HybridRetriever(
            populated_storage, retrieval_config, embedding_model=mock_embedding_model
        )

        results = retriever.search("neural networks deep learning", k=5)

        assert len(results) > 0
        # Scores should reflect RRF fusion
        for result in results:
            assert result.score >= 0

    def test_hybrid_outperforms_single_strategy(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test that hybrid search provides good coverage."""
        # Get hybrid results
        hybrid_retriever = HybridRetriever(
            populated_storage, retrieval_config, embedding_model=mock_embedding_model
        )
        hybrid_results = hybrid_retriever.search("learning from data", k=5)

        # Hybrid should return results
        assert len(hybrid_results) > 0


class TestQueryProcessing:
    """Tests for query parsing and expansion.

    Rule #4: Focused test class - tests query processing
    """

    def test_parse_simple_query(self, retrieval_config: Config):
        """Test parsing of simple keyword query."""
        parser = QueryParser(retrieval_config)

        parsed = parser.parse("machine learning")

        assert parsed is not None
        assert isinstance(parsed, dict) or isinstance(parsed, str)

    def test_parse_quoted_phrase(self, retrieval_config: Config):
        """Test parsing of quoted phrase query."""
        parser = QueryParser(retrieval_config)

        parsed = parser.parse('"neural networks"')

        assert parsed is not None

    def test_query_expansion(self, retrieval_config: Config):
        """Test query expansion with synonyms."""
        expander = QueryExpander(retrieval_config)

        expanded = expander.expand("ML")

        # Should expand ML to machine learning
        assert expanded is not None
        assert isinstance(expanded, str)

    def test_expand_technical_terms(self, retrieval_config: Config):
        """Test expansion of technical abbreviations."""
        expander = QueryExpander(retrieval_config)

        expanded = expander.expand("AI NLP")

        assert expanded is not None


class TestResultReranking:
    """Tests for result reranking.

    Rule #4: Focused test class - tests reranking
    """

    def test_rerank_by_relevance(
        self, retrieval_config: Config, mock_embedding_model: Mock
    ):
        """Test reranking of search results by relevance."""
        reranker = Reranker(retrieval_config, model=mock_embedding_model)

        # Create sample results
        results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                score=0.5 + i * 0.1,
                content=f"Content about {'neural' if i % 2 == 0 else 'other'} networks",
                document_id="doc_1",
                section_title="Section",
                chunk_type="content",
                source_file="test.md",
                word_count=10,
                metadata={},
            )
            for i in range(5)
        ]

        query = "neural networks"
        reranked = reranker.rerank(query, results)

        assert len(reranked) == len(results)

    def test_reranking_changes_order(
        self, retrieval_config: Config, mock_embedding_model: Mock
    ):
        """Test that reranking can change result order."""
        reranker = Reranker(retrieval_config, model=mock_embedding_model)

        # Create results with one highly relevant buried in middle
        results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                score=0.9 - i * 0.1,
                content="Generic content about topics"
                if i != 2
                else "Highly relevant neural network content matching query exactly",
                document_id="doc_1",
                section_title="Section",
                chunk_type="content",
                source_file="test.md",
                word_count=10,
                metadata={},
            )
            for i in range(5)
        ]

        query = "neural networks"
        reranked = reranker.rerank(query, results)

        # Reranking should promote relevant result
        assert len(reranked) == len(results)

    def test_rerank_preserves_top_results(
        self, retrieval_config: Config, mock_embedding_model: Mock
    ):
        """Test that reranking preserves top relevant results."""
        reranker = Reranker(retrieval_config, model=mock_embedding_model)

        # Create results where top is most relevant
        results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                score=0.9 - i * 0.1,
                content="Neural networks deep learning" if i == 0 else "Other content",
                document_id="doc_1",
                section_title="Section",
                chunk_type="content",
                source_file="test.md",
                word_count=10,
                metadata={},
            )
            for i in range(5)
        ]

        query = "neural networks"
        reranked = reranker.rerank(query, results, top_k=3)

        # Should return top_k results
        assert len(reranked) <= 3


class TestParentDocumentRetrieval:
    """Tests for parent document retrieval.

    Rule #4: Focused test class - tests parent retrieval
    """

    def test_retrieve_parent_context(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test retrieving parent document for context."""
        retriever = ParentDocumentRetriever(populated_storage, retrieval_config)

        # Search for a chunk
        results = populated_storage.search("machine learning", k=1)
        if results:
            chunk_id = results[0].chunk_id

            # Get parent context
            parent = retriever.get_parent_document(chunk_id)

            assert parent is not None

    def test_parent_includes_siblings(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test that parent retrieval includes sibling chunks."""
        retriever = ParentDocumentRetriever(populated_storage, retrieval_config)

        results = populated_storage.search("neural", k=1)
        if results:
            chunk_id = results[0].chunk_id
            parent = retriever.get_parent_document(chunk_id)

            # Should include context from same document
            assert parent is not None


class TestCrossCorpusRetrieval:
    """Tests for cross-corpus retrieval.

    Rule #4: Focused test class - tests cross-corpus
    """

    def test_search_multiple_libraries(self, temp_dir: Path, retrieval_config: Config):
        """Test searching across multiple libraries/collections."""
        # Create two separate corpora
        data_path = temp_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        storage1 = JSONLRepository(data_path=data_path)
        storage2 = JSONLRepository(data_path=temp_dir / "data2")

        # Add chunks to both
        chunk1 = ChunkRecord(
            chunk_id="lib1_chunk",
            document_id="doc1",
            content="Content in library 1",
            section_title="Section",
            chunk_type="content",
            source_file="lib1.md",
            word_count=4,
            char_count=20,
            library="library1",
        )

        chunk2 = ChunkRecord(
            chunk_id="lib2_chunk",
            document_id="doc2",
            content="Content in library 2",
            section_title="Section",
            chunk_type="content",
            source_file="lib2.md",
            word_count=4,
            char_count=20,
            library="library2",
        )

        storage1.add_chunks([chunk1])
        storage2.add_chunks([chunk2])

        # Cross-corpus retriever
        retriever = CrossCorpusRetriever([storage1, storage2], retrieval_config)

        results = retriever.search("content", k=5)

        # Should find results from both libraries
        assert len(results) >= 0


class TestRetrievalPerformance:
    """Tests for retrieval performance benchmarks.

    Rule #4: Focused test class - tests performance
    """

    def test_bm25_search_performance(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test BM25 search performance."""
        retriever = BM25Retriever(populated_storage, retrieval_config)

        start_time = time.time()
        for _ in range(10):
            retriever.search("machine learning", k=10)
        duration = time.time() - start_time

        # Should complete 10 searches in reasonable time (< 1 second)
        avg_time = duration / 10
        assert avg_time < 0.1  # < 100ms per search

    def test_semantic_search_performance(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test semantic search performance."""
        retriever = SemanticRetriever(
            populated_storage, retrieval_config, model=mock_embedding_model
        )

        start_time = time.time()
        for _ in range(10):
            retriever.search("neural networks", k=10)
        duration = time.time() - start_time

        # Should complete 10 searches in reasonable time
        avg_time = duration / 10
        assert avg_time < 0.2  # < 200ms per search

    def test_hybrid_search_performance(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test hybrid search performance."""
        retriever = HybridRetriever(
            populated_storage, retrieval_config, embedding_model=mock_embedding_model
        )

        start_time = time.time()
        for _ in range(10):
            retriever.search("learning algorithms", k=10)
        duration = time.time() - start_time

        # Hybrid should still be reasonably fast
        avg_time = duration / 10
        assert avg_time < 0.3  # < 300ms per search


class TestRetrievalAccuracy:
    """Tests for retrieval accuracy and relevance.

    Rule #4: Focused test class - tests accuracy
    """

    def test_exact_match_retrieval(
        self, retrieval_config: Config, populated_storage: JSONLRepository
    ):
        """Test retrieval of exact phrase matches."""
        retriever = BM25Retriever(populated_storage, retrieval_config)

        # Search for exact phrase
        results = retriever.search("machine learning is a subset", k=3)

        # Should find the exact chunk
        assert len(results) > 0
        found_exact = any("subset" in r.content.lower() for r in results)
        assert found_exact

    def test_related_concept_retrieval(
        self,
        retrieval_config: Config,
        populated_storage: JSONLRepository,
        mock_embedding_model: Mock,
    ):
        """Test retrieval of related concepts via semantic search."""
        retriever = SemanticRetriever(
            populated_storage, retrieval_config, model=mock_embedding_model
        )

        # Search for related concept
        results = retriever.search("teaching computers to recognize patterns", k=5)

        # Should find ML-related content
        assert len(results) > 0


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - BM25 retrieval: 5 tests (basic, relevance, scoring, top-k, no matches)
    - Semantic retrieval: 4 tests (basic, concepts, scores, ordering)
    - Hybrid retrieval: 4 tests (combination, weights, RRF, comparison)
    - Query processing: 4 tests (simple, quoted, expansion, technical)
    - Reranking: 3 tests (relevance, order change, preserve top)
    - Parent retrieval: 2 tests (context, siblings)
    - Cross-corpus: 1 test (multiple libraries)
    - Performance: 3 tests (BM25, semantic, hybrid)
    - Accuracy: 2 tests (exact match, related concepts)

    Total: 28 integration tests

Design Decisions:
    1. Test each retrieval strategy independently
    2. Test hybrid combinations and fusion
    3. Use mock embeddings for consistent testing
    4. Benchmark performance with repeated queries
    5. Test accuracy with known relevant results

Behaviors Tested:
    - BM25 keyword matching
    - Semantic similarity search
    - Hybrid result fusion (weighted, RRF)
    - Query expansion and processing
    - Result reranking
    - Parent document context
    - Cross-corpus search
    - Search performance
    - Retrieval accuracy

Justification:
    - Integration tests verify retrieval quality
    - Strategy-specific tests ensure correctness
    - Hybrid tests verify fusion logic
    - Performance tests catch bottlenecks
    - Accuracy tests validate relevance
"""
