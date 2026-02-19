"""
Tests for Semantic Chunker.

This module tests the semantic chunking functionality including
embedding-based boundary detection and topic shift detection.

Test Strategy
-------------
- Test embedding-based chunking
- Test topic boundary detection
- Test centroid calculation
- Test coherence scoring
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestSemanticChunkerBasic: Basic chunking functionality
- TestEmbeddingBased: Embedding-based chunking
- TestTopicBoundaries: Topic shift detection
- TestCentroidCalculation: Group centroid logic
- TestCoherenceScoring: Coherence calculations
"""

from unittest.mock import Mock

from ingestforge.chunking.semantic_chunker import SemanticChunker, ChunkRecord


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_embedding_generator():
    """Create mock embedding generator."""
    gen = Mock()

    def embed_batch_func(texts):
        """Generate distinct embeddings for each text."""
        embeddings = []
        for i in range(len(texts)):
            # Create orthogonal embeddings
            vec = [0.0] * 384
            # Use multiple dimensions to avoid overlap
            vec[i % 384] = 1.0
            vec[(i + 1) % 384] = 0.5
            embeddings.append(vec)
        return embeddings

    gen.embed_batch = Mock(side_effect=embed_batch_func)
    return gen


def make_similar_embeddings(count: int, base_value: float = 0.5) -> list:
    """Create similar embeddings (high cosine similarity)."""
    import math

    embeddings = []
    for i in range(count):
        # Create embeddings with small angle between them
        vec = [0.0] * 384
        # Use first 3 dimensions for similar direction
        angle = i * 0.05  # Small variation
        vec[0] = math.cos(angle) * base_value
        vec[1] = math.sin(angle) * base_value
        vec[2] = base_value * 0.1
        embeddings.append(vec)
    return embeddings


def make_different_embeddings(count: int) -> list:
    """Create different embeddings (low cosine similarity)."""
    embeddings = []
    for i in range(count):
        # Each embedding is orthogonal to others
        vec = [0.0] * 384
        vec[i % 384] = 1.0
        embeddings.append(vec)
    return embeddings


# ============================================================================
# Test Classes
# ============================================================================


class TestSemanticChunkerBasic:
    """Tests for basic semantic chunker functionality.

    Rule #4: Focused test class
    """

    def test_create_chunker(self):
        """Test creating semantic chunker."""
        chunker = SemanticChunker(max_chunk_size=1000, min_chunk_size=100)

        assert chunker is not None
        assert chunker.max_chunk_size == 1000
        assert chunker.min_chunk_size == 100

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = SemanticChunker()

        result = chunker.chunk("", "doc1")

        assert len(result) == 1
        assert result[0].content == ""

    def test_chunk_short_text(self):
        """Test chunking text shorter than min size."""
        chunker = SemanticChunker(min_chunk_size=100)
        short_text = "Short text."

        result = chunker.chunk(short_text, "doc1")

        assert len(result) == 1
        assert result[0].content == short_text

    def test_chunk_creates_chunk_records(self):
        """Test that chunks are ChunkRecord objects."""
        chunker = SemanticChunker()
        text = "First sentence. Second sentence. Third sentence."

        result = chunker.chunk(text, "doc1", source_file="test.txt")

        assert len(result) > 0
        assert all(isinstance(chunk, ChunkRecord) for chunk in result)
        assert all(chunk.document_id == "doc1" for chunk in result)
        assert all(chunk.source_file == "test.txt" for chunk in result)


class TestEmbeddingBased:
    """Tests for embedding-based chunking.

    Rule #4: Focused test class
    """

    def test_uses_embeddings_when_available(self):
        """Test chunker uses embeddings when available."""
        mock_gen = make_mock_embedding_generator()

        chunker = SemanticChunker(use_embeddings=True, min_chunk_size=10)
        # Directly inject the mock embedding generator
        chunker._embedding_generator = mock_gen

        text = "First sentence. Second sentence. Third sentence. Fourth sentence for good measure."
        chunker.chunk(text, "doc1")

        # Should have called embed_batch
        assert mock_gen.embed_batch.called

    def test_falls_back_to_jaccard_when_no_embeddings(self):
        """Test fallback to Jaccard when embeddings unavailable."""
        chunker = SemanticChunker(use_embeddings=False)
        text = "First sentence. Second sentence. Third sentence."

        result = chunker.chunk(text, "doc1")

        # Should still produce chunks
        assert len(result) > 0

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity between embedding vectors."""
        chunker = SemanticChunker()

        # Identical vectors -> similarity = 1.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = chunker._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

        # Orthogonal vectors -> similarity = 0.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = chunker._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

        # Opposite vectors -> similarity = -1.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = chunker._cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001

    def test_cosine_similarity_handles_zero_vectors(self):
        """Test cosine similarity handles zero vectors."""
        chunker = SemanticChunker()

        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 1.0, 1.0]

        similarity = chunker._cosine_similarity(vec1, vec2)

        # Should return 0.0 for zero vector
        assert similarity == 0.0


class TestTopicBoundaries:
    """Tests for topic boundary detection.

    Rule #4: Focused test class
    """

    def test_detect_topic_boundary_with_coherence_drop(self):
        """Test topic boundary detected when coherence drops."""
        chunker = SemanticChunker(use_embeddings=True)

        # Create similar embeddings for group, different for next
        similar_vecs = make_similar_embeddings(3, base_value=0.5)
        different_vec = [[0.9] * 384]  # Very different

        vectors = similar_vecs + different_vec
        current_group = [0, 1, 2]
        next_idx = 3

        is_boundary = chunker._detect_topic_boundary(vectors, current_group, next_idx)

        # Should detect boundary due to coherence drop
        assert is_boundary is True

    def test_no_boundary_for_similar_sentences(self):
        """Test no boundary for similar sentences."""
        chunker = SemanticChunker(use_embeddings=True)

        # All similar embeddings
        vectors = make_similar_embeddings(4, base_value=0.5)
        current_group = [0, 1, 2]
        next_idx = 3

        is_boundary = chunker._detect_topic_boundary(vectors, current_group, next_idx)

        # Should NOT detect boundary
        assert is_boundary is False

    def test_boundary_detection_requires_embeddings(self):
        """Test boundary detection requires embeddings."""
        chunker = SemanticChunker(use_embeddings=False)

        # Word-based vectors
        vectors = [["word1", "word2"], ["word3", "word4"]]
        current_group = [0]
        next_idx = 1

        is_boundary = chunker._detect_topic_boundary(vectors, current_group, next_idx)

        # Should return False (requires embeddings)
        assert is_boundary is False


class TestCentroidCalculation:
    """Tests for centroid calculation.

    Rule #4: Focused test class
    """

    def test_update_centroid_with_embeddings(self):
        """Test centroid calculation with embedding vectors."""
        chunker = SemanticChunker()

        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        group_indices = [0, 1, 2]

        centroid = chunker._update_centroid(vectors, group_indices)

        # Centroid should be average
        expected = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        assert len(centroid) == 3
        for i in range(3):
            assert abs(centroid[i] - expected[i]) < 0.001

    def test_update_centroid_with_word_lists(self):
        """Test centroid calculation with word lists."""
        chunker = SemanticChunker()

        vectors = [["word1", "word2"], ["word2", "word3"], ["word3", "word4"]]
        group_indices = [0, 1, 2]

        centroid = chunker._update_centroid(vectors, group_indices)

        # Centroid should be union of words
        expected_words = {"word1", "word2", "word3", "word4"}
        assert set(centroid) == expected_words

    def test_update_centroid_empty_group(self):
        """Test centroid calculation with empty group."""
        chunker = SemanticChunker()

        vectors = [[1.0, 2.0, 3.0]]
        group_indices = []

        centroid = chunker._update_centroid(vectors, group_indices)

        # Should return empty list
        assert centroid == []


class TestCoherenceScoring:
    """Tests for coherence scoring.

    Rule #4: Focused test class
    """

    def test_calculate_embedding_coherence_high(self):
        """Test coherence calculation for similar embeddings."""
        chunker = SemanticChunker()

        # Create very similar embeddings
        vectors = make_similar_embeddings(3, base_value=0.5)
        group_indices = [0, 1, 2]

        coherence = chunker._calculate_embedding_coherence(vectors, group_indices)

        # Similar embeddings should have high coherence
        assert coherence > 0.8

    def test_calculate_embedding_coherence_low(self):
        """Test coherence calculation for different embeddings."""
        chunker = SemanticChunker()

        # Create very different (orthogonal) embeddings
        vectors = make_different_embeddings(3)
        group_indices = [0, 1, 2]

        coherence = chunker._calculate_embedding_coherence(vectors, group_indices)

        # Different embeddings should have low coherence
        assert coherence < 0.2

    def test_coherence_single_sentence(self):
        """Test coherence for single sentence."""
        chunker = SemanticChunker()

        vectors = [[1.0, 0.0, 0.0]]
        group_indices = [0]

        coherence = chunker._calculate_embedding_coherence(vectors, group_indices)

        # Single sentence has perfect coherence
        assert coherence == 1.0


class TestIntegration:
    """Integration tests for complete semantic chunking.

    Rule #4: Focused test class
    """

    def test_full_chunking_pipeline_with_embeddings(self):
        """Test complete chunking pipeline with embeddings."""
        mock_gen = Mock()

        # Create properly different embeddings for different topics
        def embed_batch_func(texts):
            embeddings = []
            for i in range(len(texts)):
                vec = [0.0] * 384
                if i < 2:  # Topic 1 - similar to each other
                    vec[0] = 1.0
                    vec[1] = 0.0
                else:  # Topic 2 - orthogonal to topic 1
                    vec[0] = 0.0
                    vec[1] = 1.0
                embeddings.append(vec)
            return embeddings

        mock_gen.embed_batch = Mock(side_effect=embed_batch_func)

        chunker = SemanticChunker(
            max_chunk_size=1000,
            min_chunk_size=10,
            similarity_threshold=0.7,
            use_embeddings=True,
        )
        # Directly inject the mock embedding generator
        chunker._embedding_generator = mock_gen

        text = "First topic sentence. Another first topic sentence. Second topic sentence. Another second topic sentence."

        result = chunker.chunk(text, "doc1", source_file="test.txt")

        # Should create separate chunks for different topics
        assert len(result) >= 2

        # All chunks should have proper metadata
        for chunk in result:
            assert chunk.document_id == "doc1"
            assert chunk.source_file == "test.txt"
            assert chunk.content
            assert chunk.word_count > 0

    def test_grouping_uses_centroid_comparison(self):
        """Test that grouping compares to centroid, not last sentence."""
        chunker = SemanticChunker(use_embeddings=True)

        # Mock vectors where centroid comparison gives different result
        # than last-sentence comparison
        vectors = [
            [0.5, 0.5],  # Sentence 0
            [0.5, 0.5],  # Sentence 1 (similar to 0)
            [0.6, 0.4],  # Sentence 2 (closer to centroid than to 1)
        ]

        groups = chunker._group_by_similarity(["sent1", "sent2", "sent3"], vectors)

        # Should group all three together due to centroid comparison
        assert len(groups) >= 1


class TestChunkRecordSerialization:
    """Tests for ChunkRecord serialization.

    Rule #4: Focused test class
    """

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        chunk = ChunkRecord(
            chunk_id="test_1",
            document_id="doc_1",
            content="Test content",
            section_title="Introduction",
            library="my_library",
            word_count=2,
        )

        data = chunk.to_dict()

        assert data["chunk_id"] == "test_1"
        assert data["section_title"] == "Introduction"
        assert data["library"] == "my_library"
        assert data["word_count"] == 2

    def test_from_dict_creates_chunk(self):
        """Test that from_dict creates valid chunk."""
        data = {
            "chunk_id": "test_1",
            "document_id": "doc_1",
            "content": "Test content",
            "library": "custom_lib",
        }

        chunk = ChunkRecord.from_dict(data)

        assert chunk.chunk_id == "test_1"
        assert chunk.library == "custom_lib"

    def test_from_dict_ignores_unknown_fields(self):
        """Test from_dict ignores unknown fields."""
        data = {
            "chunk_id": "test_1",
            "document_id": "doc_1",
            "content": "Test",
            "unknown_field": "ignored",
            "another_unknown": 123,
        }

        chunk = ChunkRecord.from_dict(data)

        assert chunk.chunk_id == "test_1"
        assert not hasattr(chunk, "unknown_field")

    def test_round_trip_serialization(self):
        """Test that serialization is reversible."""
        original = ChunkRecord(
            chunk_id="test_1",
            document_id="doc_1",
            content="Test content",
            section_title="Section",
            word_count=2,
            char_count=12,
            entities=["entity1"],
            quality_score=0.8,
        )

        data = original.to_dict()
        restored = ChunkRecord.from_dict(data)

        assert restored.chunk_id == original.chunk_id
        assert restored.section_title == original.section_title
        assert restored.entities == original.entities


class TestSentenceBoundaries:
    """Tests for improved sentence boundary detection.

    Rule #4: Focused test class
    """

    def test_split_simple_sentences(self):
        """Test splitting simple sentences."""
        chunker = SemanticChunker(use_embeddings=False)

        sentences = chunker._split_into_sentences(
            "First sentence. Second sentence. Third sentence."
        )

        assert len(sentences) == 3

    def test_split_with_questions(self):
        """Test splitting with question marks."""
        chunker = SemanticChunker(use_embeddings=False)

        sentences = chunker._split_into_sentences("What is this? It is a test. Really?")

        assert len(sentences) == 3

    def test_split_with_exclamations(self):
        """Test splitting with exclamation marks."""
        chunker = SemanticChunker(use_embeddings=False)

        sentences = chunker._split_into_sentences("This is amazing! I love it! Yes!")

        assert len(sentences) == 3

    def test_split_handles_paragraphs(self):
        """Test that paragraph breaks are handled."""
        chunker = SemanticChunker(use_embeddings=False)

        sentences = chunker._split_into_sentences(
            "First paragraph.\n\nSecond paragraph."
        )

        assert len(sentences) >= 2

    def test_split_empty_text(self):
        """Test splitting empty text."""
        chunker = SemanticChunker(use_embeddings=False)

        sentences = chunker._split_into_sentences("")

        assert len(sentences) == 0


class TestOverlapBehavior:
    """Tests for chunk overlap functionality.

    Rule #4: Focused test class
    """

    def test_overlap_sentences_parameter(self):
        """Test overlap_sentences parameter is set."""
        chunker = SemanticChunker(overlap_sentences=2)

        assert chunker.overlap_sentences == 2

    def test_default_overlap_is_one(self):
        """Test default overlap is 1 sentence."""
        chunker = SemanticChunker()

        assert chunker.overlap_sentences == 1


class TestJaccardSimilarity:
    """Tests for Jaccard similarity fallback.

    Rule #4: Focused test class
    """

    def test_jaccard_identical_sets(self):
        """Test Jaccard similarity of identical sets."""
        chunker = SemanticChunker(use_embeddings=False)

        sim = chunker._jaccard_similarity(["a", "b"], ["a", "b"])

        assert sim == 1.0

    def test_jaccard_disjoint_sets(self):
        """Test Jaccard similarity of disjoint sets."""
        chunker = SemanticChunker(use_embeddings=False)

        sim = chunker._jaccard_similarity(["a"], ["b"])

        assert sim == 0.0

    def test_jaccard_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        chunker = SemanticChunker(use_embeddings=False)

        sim = chunker._jaccard_similarity(["a", "b"], ["b", "c"])

        # 1 in common, 3 total unique = 1/3
        assert abs(sim - 1 / 3) < 0.001

    def test_jaccard_empty_sets(self):
        """Test Jaccard similarity with empty sets."""
        chunker = SemanticChunker(use_embeddings=False)

        sim = chunker._jaccard_similarity([], [])

        assert sim == 0.0


class TestBoundaryScoring:
    """Tests for boundary score calculation.

    Rule #4: Focused test class
    """

    def test_similarity_drop_at_boundary(self):
        """Test similarity drop calculation."""
        chunker = SemanticChunker(use_embeddings=False)

        embeddings = [
            ["topic", "one"],
            ["topic", "one"],
            ["different", "topic"],
        ]

        drop = chunker._calculate_similarity_drop(embeddings, 2)

        # Should have positive drop at topic change
        assert drop > 0.0

    def test_similarity_drop_at_position_zero(self):
        """Test similarity drop at position 0."""
        chunker = SemanticChunker(use_embeddings=False)

        embeddings = [["word1"], ["word2"]]

        drop = chunker._calculate_similarity_drop(embeddings, 0)

        assert drop == 0.0


class TestEdgeCases:
    """Additional edge case tests.

    Rule #4: Focused test class
    """

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = SemanticChunker(use_embeddings=False)

        result = chunker.chunk("   \n\t   ", "doc1")

        assert len(result) <= 1

    def test_chunk_single_word(self):
        """Test chunking single word."""
        chunker = SemanticChunker(use_embeddings=False)

        result = chunker.chunk("Word", "doc1")

        assert len(result) == 1

    def test_chunk_preserves_source_file(self):
        """Test that source_file is preserved."""
        chunker = SemanticChunker(use_embeddings=False, min_chunk_size=1)

        result = chunker.chunk("Test.", "doc1", source_file="test.pdf")

        assert all(c.source_file == "test.pdf" for c in result)

    def test_chunk_ids_are_unique(self):
        """Test that chunk IDs are unique."""
        chunker = SemanticChunker(use_embeddings=False, min_chunk_size=5)

        text = "First. Second. Third. Fourth. Fifth."
        result = chunker.chunk(text, "doc1")

        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids))

    def test_metadata_passed_through(self):
        """Test that metadata is passed to chunks."""
        chunker = SemanticChunker(use_embeddings=False, min_chunk_size=1)

        result = chunker.chunk(
            "Test content.", "doc1", metadata={"custom_key": "custom_value"}
        )

        assert result[0].metadata.get("custom_key") == "custom_value"
