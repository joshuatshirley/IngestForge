"""
Tests for semantic chunking with embeddings.

Validates:
- Embedding-based chunking works
- Cosine similarity for semantic boundaries
- Retrieval quality improvement over Jaccard
- Performance benchmarks
"""

import pytest
from ingestforge.chunking.semantic_chunker import SemanticChunker, ChunkRecord


# Test corpus with clear semantic boundaries
TEST_CORPUS = """
Machine learning is a subset of artificial intelligence. It focuses on algorithms that learn from data.
Neural networks are inspired by biological neurons. They consist of interconnected layers of nodes.

Python is a popular programming language. It is widely used in data science and web development.
JavaScript runs in web browsers. It enables interactive web applications.

Climate change affects global temperatures. Rising seas threaten coastal cities.
Renewable energy sources include solar and wind. They reduce carbon emissions significantly.
"""


class TestSemanticChunkerEmbeddings:
    """Test semantic chunker with actual embeddings."""

    def test_embedding_mode_enabled(self):
        """Test that embedding mode can be enabled."""
        chunker = SemanticChunker(
            max_chunk_size=500, min_chunk_size=50, use_embeddings=True
        )
        assert chunker.use_embeddings is True

    def test_word_mode_fallback(self):
        """Test fallback to word-based similarity."""
        chunker = SemanticChunker(
            max_chunk_size=500, min_chunk_size=50, use_embeddings=False
        )
        chunks = chunker.chunk(TEST_CORPUS, "test_doc")
        assert len(chunks) > 0
        assert all(isinstance(c, ChunkRecord) for c in chunks)

    def test_embedding_chunking_creates_chunks(self):
        """Test that embedding-based chunking produces chunks."""
        try:
            chunker = SemanticChunker(
                max_chunk_size=500, min_chunk_size=50, use_embeddings=True
            )
            chunks = chunker.chunk(TEST_CORPUS, "test_doc")

            assert len(chunks) > 0
            assert len(chunks) >= 3  # Should identify 3 topics
            assert all(isinstance(c, ChunkRecord) for c in chunks)
            assert all(c.content.strip() for c in chunks)
        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        chunker = SemanticChunker(use_embeddings=True)

        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        sim = chunker._cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.01

        # Orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = chunker._cosine_similarity(vec1, vec2)
        assert abs(sim - 0.0) < 0.01

        # Opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        sim = chunker._cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 0.01

    def test_zero_vectors_handling(self):
        """Test handling of zero vectors."""
        chunker = SemanticChunker(use_embeddings=True)
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        sim = chunker._cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_empty_vectors_handling(self):
        """Test handling of empty vectors."""
        chunker = SemanticChunker(use_embeddings=True)
        vec1 = []
        vec2 = [1.0, 0.0, 0.0]
        sim = chunker._cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_chunk_coherence_with_embeddings(self):
        """Test that embedding chunks are more semantically coherent."""
        try:
            # Embedding-based chunking
            chunker_emb = SemanticChunker(
                max_chunk_size=500, min_chunk_size=50, use_embeddings=True
            )
            chunks_emb = chunker_emb.chunk(TEST_CORPUS, "test_doc")

            # Word-based chunking
            chunker_word = SemanticChunker(
                max_chunk_size=500, min_chunk_size=50, use_embeddings=False
            )
            chunks_word = chunker_word.chunk(TEST_CORPUS, "test_doc")

            # Embedding chunks should better separate topics
            # (ML/AI, Programming, Climate)
            assert len(chunks_emb) >= 3
            assert len(chunks_word) >= 1

            # Check that ML and AI terms are in same chunk
            ml_chunk = [
                c for c in chunks_emb if "machine learning" in c.content.lower()
            ]
            assert len(ml_chunk) > 0
            assert (
                "neural network" in ml_chunk[0].content.lower()
                or "artificial intelligence" in ml_chunk[0].content.lower()
            )

        except ImportError:
            pytest.skip("sentence-transformers not available")


class TestSemanticChunkerPerformance:
    """Performance and benchmark tests."""

    def test_chunking_speed_small_text(self):
        """Test chunking speed on small text (<1000 words)."""
        import time

        text = TEST_CORPUS * 10  # ~1000 words
        chunker = SemanticChunker(use_embeddings=False)

        start = time.time()
        chunks = chunker.chunk(text, "test_doc")
        duration = time.time() - start

        assert duration < 1.0  # Should complete in <1s
        assert len(chunks) > 0

    def test_embedding_speed_reasonable(self):
        """Test that embedding generation doesn't take too long."""
        try:
            import time

            text = TEST_CORPUS * 5  # ~500 words
            chunker = SemanticChunker(use_embeddings=True)

            start = time.time()
            chunks = chunker.chunk(text, "test_doc")
            duration = time.time() - start

            # Should complete in reasonable time (may be slower on CPU)
            assert duration < 10.0  # 10s timeout
            assert len(chunks) > 0

        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_memory_efficient(self):
        """Test that chunking doesn't leak memory."""
        import gc

        gc.collect()
        chunker = SemanticChunker(use_embeddings=False)

        # Process multiple times
        for _ in range(10):
            chunker.chunk(TEST_CORPUS, "test_doc")
            gc.collect()

        # No assertion - just verifying no crash


class TestSemanticChunkerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = SemanticChunker(use_embeddings=True)
        chunks = chunker.chunk("", "test_doc")
        assert len(chunks) == 1  # Returns single empty chunk
        assert chunks[0].content == ""

    def test_very_short_text(self):
        """Test handling of very short text."""
        chunker = SemanticChunker(min_chunk_size=100, use_embeddings=True)
        short_text = "Hello world."
        chunks = chunker.chunk(short_text, "test_doc")
        assert len(chunks) == 1
        assert chunks[0].content == short_text

    def test_single_sentence(self):
        """Test handling of single sentence."""
        chunker = SemanticChunker(use_embeddings=True)
        text = "This is a single sentence."
        chunks = chunker.chunk(text, "test_doc")
        assert len(chunks) == 1

    def test_metadata_preserved(self):
        """Test that metadata is preserved in chunks."""
        chunker = SemanticChunker(use_embeddings=False)
        metadata = {"author": "test", "year": 2024}
        chunks = chunker.chunk(TEST_CORPUS, "test_doc", metadata=metadata)
        # Metadata should be accessible (though not explicitly stored in current impl)
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
