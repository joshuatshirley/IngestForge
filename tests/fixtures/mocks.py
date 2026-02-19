"""
Reusable mock objects for IngestForge tests.

This module provides pre-configured mock objects and response
generators for common testing scenarios.

All mocks follow the interfaces defined in the application code
to ensure compatibility with actual implementations.

Usage Example
-------------
    from tests.fixtures.mocks import MockLLMResponses

    def test_summarization():
        responses = MockLLMResponses()
        summary = responses.get_summary()
        assert "summary" in summary.lower()
"""

from typing import Any, List, Optional
from unittest.mock import Mock
import numpy as np


# ============================================================================
# LLM Response Mocks
# ============================================================================


class MockLLMResponses:
    """Pre-defined LLM responses for common operations.

    Provides realistic-looking responses for:
    - Summarization
    - Question generation
    - Entity extraction
    - Answer generation
    - Content analysis
    """

    @staticmethod
    def get_summary(length: str = "medium") -> str:
        """Get a mock summary response.

        Args:
            length: "short", "medium", or "long"

        Returns:
            Mock summary text
        """
        summaries = {
            "short": "This document discusses key concepts in the field.",
            "medium": "This document provides an overview of several important concepts. It covers the main topics in detail and explains their relationships. The content is well-structured and informative.",
            "long": "This document provides a comprehensive overview of several important concepts in the field. It begins by introducing the fundamental principles and then explores each topic in detail. The content is well-structured with clear sections and examples. Key points are highlighted throughout, making it easy to understand the main ideas. The document concludes with practical applications and recommendations.",
        }
        return summaries.get(length, summaries["medium"])

    @staticmethod
    def get_questions(count: int = 3) -> List[str]:
        """Get mock hypothetical questions.

        Args:
            count: Number of questions to return

        Returns:
            List of question strings
        """
        all_questions = [
            "What are the main concepts discussed in this document?",
            "How does this topic relate to practical applications?",
            "What are the key benefits of this approach?",
            "What challenges might arise when implementing this?",
            "How does this compare to alternative methods?",
            "What are the prerequisites for understanding this topic?",
            "What are the next steps after learning this material?",
        ]
        return all_questions[:count]

    @staticmethod
    def get_entities() -> List[str]:
        """Get mock named entities.

        Returns:
            List of entity names
        """
        return [
            "Python",
            "Machine Learning",
            "Natural Language Processing",
            "Vector Database",
            "Semantic Search",
        ]

    @staticmethod
    def get_concepts() -> List[str]:
        """Get mock concept tags.

        Returns:
            List of concept strings
        """
        return [
            "programming",
            "data science",
            "artificial intelligence",
            "information retrieval",
            "text analysis",
        ]

    @staticmethod
    def get_answer(query: str = "test query") -> str:
        """Get mock answer to a query.

        Args:
            query: The query (used to customize response)

        Returns:
            Mock answer text
        """
        return f"Based on the provided context, the answer to '{query}' is that the document covers several relevant aspects. The main points include key concepts and their applications. This information helps address the question comprehensively."

    @staticmethod
    def get_analysis() -> str:
        """Get mock content analysis.

        Returns:
            Mock analysis text
        """
        return """This content is well-structured and informative. Key strengths include:
- Clear organization with logical flow
- Good use of examples to illustrate concepts
- Appropriate level of technical detail
- Comprehensive coverage of the topic

Areas for potential improvement:
- Could include more visual aids
- Additional cross-references would be helpful
"""


# ============================================================================
# Storage Mock Builders
# ============================================================================


class MockStorageBuilder:
    """Builder for creating configured mock storage backends.

    Example:
        storage = MockStorageBuilder()\\
            .with_chunks(sample_chunks)\\
            .with_search_results(results)\\
            .build()
    """

    def __init__(self):
        """Initialize storage builder."""
        self._mock = Mock()
        self._chunks = {}
        self._search_results = []

    def with_chunks(self, chunks: List[Any]) -> "MockStorageBuilder":
        """Add chunks to mock storage.

        Args:
            chunks: List of ChunkRecord objects

        Returns:
            Self for chaining
        """
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        return self

    def with_search_results(self, results: List[Any]) -> "MockStorageBuilder":
        """Configure search to return specific results.

        Args:
            results: List of SearchResult objects

        Returns:
            Self for chaining
        """
        self._search_results = results
        return self

    def build(self) -> Mock:
        """Build the configured mock storage.

        Returns:
            Configured Mock object
        """

        # Configure get_chunk
        def get_chunk(chunk_id: str):
            return self._chunks.get(chunk_id)

        self._mock.get_chunk.side_effect = get_chunk

        # Configure search methods
        self._mock.search.return_value = self._search_results
        self._mock.search_semantic.return_value = self._search_results

        # Configure other methods
        self._mock.add_chunks.return_value = len(self._chunks)
        self._mock.count.return_value = len(self._chunks)
        self._mock.get_all_chunks.return_value = list(self._chunks.values())
        self._mock.delete_chunk.return_value = True
        self._mock.clear.return_value = None

        return self._mock


# ============================================================================
# Embedding Mock Generators
# ============================================================================


class MockEmbeddingGenerator:
    """Mock embedding generator with deterministic outputs.

    Generates embeddings that are consistent for the same input text,
    allowing for reproducible test results.

    Example:
        generator = MockEmbeddingGenerator(dimensions=384)
        embedding = generator.generate("test text")
        assert len(embedding) == 384
    """

    def __init__(self, dimensions: int = 384):
        """Initialize embedding generator.

        Args:
            dimensions: Number of dimensions for embeddings
        """
        self.dimensions = dimensions

    def generate(self, text: str, normalize: bool = True) -> List[float]:
        """Generate deterministic embedding for text.

        Args:
            text: Input text
            normalize: Whether to normalize to unit length

        Returns:
            Embedding vector
        """
        # Use text hash as seed for consistency
        seed = hash(text) % 10000
        np.random.seed(seed)

        embedding = np.random.randn(self.dimensions).astype(np.float32)

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.tolist()

    def generate_batch(
        self, texts: List[str], normalize: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            normalize: Whether to normalize to unit length

        Returns:
            List of embedding vectors
        """
        return [self.generate(text, normalize) for text in texts]


# ============================================================================
# LLM Client Mock Builders
# ============================================================================


class MockLLMBuilder:
    """Builder for creating configured mock LLM clients.

    Example:
        llm = MockLLMBuilder()\\
            .with_responses(["response 1", "response 2"])\\
            .with_model_name("gpt-4")\\
            .build()
    """

    def __init__(self):
        """Initialize LLM builder."""
        self._mock = Mock()
        self._responses = ["This is a mock response."]
        self._response_index = 0
        self._model_name = "mock-model"
        self._available = True

    def with_responses(self, responses: List[str]) -> "MockLLMBuilder":
        """Set responses to return on generate() calls.

        Args:
            responses: List of response strings

        Returns:
            Self for chaining
        """
        self._responses = responses
        return self

    def with_model_name(self, name: str) -> "MockLLMBuilder":
        """Set model name.

        Args:
            name: Model name

        Returns:
            Self for chaining
        """
        self._model_name = name
        return self

    def with_availability(self, available: bool) -> "MockLLMBuilder":
        """Set availability status.

        Args:
            available: Whether model is available

        Returns:
            Self for chaining
        """
        self._available = available
        return self

    def build(self) -> Mock:
        """Build the configured mock LLM client.

        Returns:
            Configured Mock object
        """

        def generate_response(*args, **kwargs):
            # Cycle through responses
            response = self._responses[self._response_index % len(self._responses)]
            self._response_index += 1
            return response

        self._mock.generate.side_effect = generate_response
        self._mock.generate_with_context.side_effect = generate_response
        self._mock.is_available.return_value = self._available
        self._mock.model_name = self._model_name

        return self._mock


# ============================================================================
# Quick Mock Factories
# ============================================================================


def create_mock_llm(responses: Optional[List[str]] = None) -> Mock:
    """Quick factory for creating a mock LLM client.

    Args:
        responses: Optional list of responses (uses default if None)

    Returns:
        Configured Mock LLM client

    Example:
        llm = create_mock_llm(["response 1", "response 2"])
        result = llm.generate("prompt")
        assert result == "response 1"
    """
    builder = MockLLMBuilder()
    if responses:
        builder.with_responses(responses)
    return builder.build()


def create_mock_storage(chunks: Optional[List[Any]] = None) -> Mock:
    """Quick factory for creating a mock storage backend.

    Args:
        chunks: Optional list of chunks to pre-populate

    Returns:
        Configured Mock storage backend

    Example:
        storage = create_mock_storage(chunks=[chunk1, chunk2])
        retrieved = storage.get_chunk("chunk_1")
    """
    builder = MockStorageBuilder()
    if chunks:
        builder.with_chunks(chunks)
    return builder.build()


def create_mock_embedding_model(dimensions: int = 384) -> Mock:
    """Quick factory for creating a mock embedding model.

    Args:
        dimensions: Embedding dimensions

    Returns:
        Mock with encode() method

    Example:
        model = create_mock_embedding_model(dimensions=384)
        embedding = model.encode("text")
        assert len(embedding) == 384
    """
    generator = MockEmbeddingGenerator(dimensions)
    mock = Mock()

    def encode(text, convert_to_numpy=True, **kwargs):
        embedding = generator.generate(text)
        if convert_to_numpy:
            return np.array(embedding)
        return embedding

    mock.encode.side_effect = encode
    return mock


# ============================================================================
# Mock Response Data
# ============================================================================


class MockMetadataResponses:
    """Pre-defined metadata extraction responses.

    Provides realistic mock data for:
    - Dates
    - Numbers
    - URLs
    - Emails
    - Keywords
    """

    @staticmethod
    def get_dates() -> List[str]:
        """Get mock date strings."""
        return ["2024-01-01", "January 15, 2024", "03/20/2024"]

    @staticmethod
    def get_numbers() -> List[str]:
        """Get mock number strings."""
        return ["1,234", "25%", "87.5", "$99.99"]

    @staticmethod
    def get_urls() -> List[str]:
        """Get mock URLs."""
        return [
            "https://example.com",
            "http://test.org/page",
            "https://docs.example.com/api",
        ]

    @staticmethod
    def get_emails() -> List[str]:
        """Get mock email addresses."""
        return [
            "test@example.com",
            "support@test.org",
            "admin@company.com",
        ]

    @staticmethod
    def get_keywords() -> List[str]:
        """Get mock keywords."""
        return [
            "machine learning",
            "natural language",
            "processing",
            "embeddings",
            "retrieval",
        ]
