"""Tests for topic modeling enrichment.

Tests cover:
- TopicModeler initialization
- Topic extraction from chunks
- Chunk enrichment with topics
- Term extraction (bigrams, trigrams, single words)
- Term validation
- Stop word filtering
- Edge cases and boundary conditions
"""

import pytest
from ingestforge.enrichment.topics import (
    TopicModeler,
    extract_topics,
    enrich_with_topics,
)


class TestTopicModeler:
    """Test suite for TopicModeler class."""

    @pytest.fixture
    def modeler(self):
        """Create TopicModeler instance."""
        return TopicModeler()

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            {"text": "Machine learning is a powerful technology for data analysis"},
            {
                "text": "Deep learning models use neural networks for pattern recognition"
            },
            {
                "text": "Machine learning algorithms require training data for optimization"
            },
        ]

    # Initialization Tests
    def test_initialization_default(self):
        """Test modeler initializes with default parameters."""
        modeler = TopicModeler()
        assert modeler.min_term_freq == 2
        assert hasattr(modeler, "stop_words")
        assert isinstance(modeler.stop_words, set)

    def test_initialization_custom_freq(self):
        """Test modeler initializes with custom min_term_freq."""
        modeler = TopicModeler(min_term_freq=5)
        assert modeler.min_term_freq == 5

    def test_stop_words_loaded(self, modeler):
        """Test stop words are loaded."""
        assert len(modeler.stop_words) > 0
        assert "the" in modeler.stop_words
        assert "and" in modeler.stop_words

    # Topic Extraction Tests
    def test_extract_topics_returns_list(self, modeler, sample_chunks):
        """Test extract_topics returns list of dictionaries."""
        topics = modeler.extract_topics(sample_chunks)
        assert isinstance(topics, list)
        assert all(isinstance(t, dict) for t in topics)

    def test_extract_topics_structure(self, modeler, sample_chunks):
        """Test topic dictionary structure."""
        topics = modeler.extract_topics(sample_chunks, num_topics=5)
        if topics:
            topic = topics[0]
            assert "topic" in topic
            assert "frequency" in topic
            assert "weight" in topic

    def test_extract_topics_num_topics_limit(self, modeler, sample_chunks):
        """Test num_topics parameter limits results."""
        topics = modeler.extract_topics(sample_chunks, num_topics=3)
        assert len(topics) <= 3

    def test_extract_topics_frequency_sorting(self, modeler):
        """Test topics are sorted by frequency."""
        chunks = [
            {"text": "python python python"},
            {"text": "java python"},
        ]
        topics = modeler.extract_topics(chunks)
        if len(topics) >= 2:
            # First topic should have higher frequency than second
            assert topics[0]["frequency"] >= topics[1]["frequency"]

    def test_extract_topics_empty_chunks(self, modeler):
        """Test extract_topics handles empty chunk list."""
        topics = modeler.extract_topics([])
        assert isinstance(topics, list)
        assert len(topics) == 0

    def test_extract_topics_chunks_without_text(self, modeler):
        """Test extract_topics handles chunks without text key."""
        chunks = [{"content": "test"}]
        topics = modeler.extract_topics(chunks)
        assert isinstance(topics, list)

    def test_extract_topics_empty_text(self, modeler):
        """Test extract_topics handles empty text."""
        chunks = [{"text": ""}, {"text": ""}]
        topics = modeler.extract_topics(chunks)
        assert isinstance(topics, list)
        assert len(topics) == 0

    def test_extract_topics_min_frequency_filter(self, modeler):
        """Test min_term_freq filters infrequent terms."""
        modeler.min_term_freq = 3
        chunks = [
            {"text": "python python python"},
            {"text": "java"},  # frequency = 1
        ]
        topics = modeler.extract_topics(chunks)
        # 'java' should be filtered out
        topic_terms = [t["topic"] for t in topics]
        assert "java" not in topic_terms

    def test_extract_topics_weight_calculation(self, modeler):
        """Test weight is calculated correctly."""
        chunks = [{"text": "machine learning data science"}]
        topics = modeler.extract_topics(chunks)
        if topics:
            # Weight should be frequency / total_terms
            assert 0 <= topics[0]["weight"] <= 1

    # Term Extraction Tests
    def test_extract_terms_basic(self, modeler):
        """Test _extract_terms extracts terms."""
        terms = modeler._extract_terms("machine learning algorithms")
        assert isinstance(terms, list)
        assert len(terms) > 0

    def test_extract_terms_bigrams(self, modeler):
        """Test _extract_terms extracts bigrams."""
        terms = modeler._extract_terms("machine learning algorithms")
        # Should extract 'machine learning'
        assert any("machine learning" in t for t in terms)

    def test_extract_terms_trigrams(self, modeler):
        """Test _extract_terms extracts trigrams."""
        terms = modeler._extract_terms("deep learning neural networks")
        # Should extract 3-word phrases
        assert any(len(t.split()) == 3 for t in terms)

    def test_extract_terms_single_words(self, modeler):
        """Test _extract_terms extracts single words."""
        terms = modeler._extract_terms("python programming")
        # Should extract individual words
        assert "python" in terms or "programming" in terms

    def test_extract_terms_case_insensitive(self, modeler):
        """Test _extract_terms normalizes case."""
        terms = modeler._extract_terms("Python PROGRAMMING")
        # All terms should be lowercase
        assert all(t.islower() for t in terms)

    def test_extract_terms_min_word_length(self, modeler):
        """Test _extract_terms filters short words."""
        terms = modeler._extract_terms("a is an to test")
        # Short words (< 3 chars) should be filtered
        assert "a" not in terms
        assert "is" not in terms
        assert "an" not in terms
        assert "to" not in terms

    def test_extract_terms_special_characters(self, modeler):
        """Test _extract_terms handles special characters."""
        terms = modeler._extract_terms("test-case @mention #hashtag")
        # Should extract alphabetic terms
        assert isinstance(terms, list)

    def test_extract_terms_empty_text(self, modeler):
        """Test _extract_terms handles empty text."""
        terms = modeler._extract_terms("")
        assert terms == []

    # Term Validation Tests
    def test_is_valid_term_basic(self, modeler):
        """Test _is_valid_term validates terms."""
        assert modeler._is_valid_term("python") is True
        assert modeler._is_valid_term("machine learning") is True

    def test_is_valid_term_stop_words(self, modeler):
        """Test _is_valid_term filters stop words."""
        assert modeler._is_valid_term("the") is False
        assert modeler._is_valid_term("and") is False
        assert modeler._is_valid_term("the test") is False

    def test_is_valid_term_min_length(self, modeler):
        """Test _is_valid_term filters short terms."""
        assert modeler._is_valid_term("abc") is False
        assert modeler._is_valid_term("ab") is False
        assert modeler._is_valid_term("a") is False

    def test_is_valid_term_valid_length(self, modeler):
        """Test _is_valid_term accepts valid length terms."""
        assert modeler._is_valid_term("test") is True
        assert modeler._is_valid_term("testing") is True

    def test_is_valid_term_phrase_with_stop_word(self, modeler):
        """Test _is_valid_term filters phrases containing stop words."""
        assert modeler._is_valid_term("python and java") is False
        assert modeler._is_valid_term("learning the basics") is False

    # Chunk Enrichment Tests
    def test_enrich_chunk_adds_topics(self, modeler):
        """Test enrich_chunk adds topics to chunk."""
        chunk = {"text": "machine learning algorithms"}
        global_topics = [
            {"topic": "machine learning", "frequency": 5, "weight": 0.5},
            {"topic": "algorithms", "frequency": 3, "weight": 0.3},
        ]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert "topics" in enriched
        assert isinstance(enriched["topics"], list)

    def test_enrich_chunk_finds_matching_topics(self, modeler):
        """Test enrich_chunk identifies matching topics."""
        chunk = {"text": "machine learning is powerful"}
        global_topics = [
            {"topic": "machine learning", "frequency": 5, "weight": 0.5},
            {"topic": "deep learning", "frequency": 3, "weight": 0.3},
        ]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert "machine learning" in enriched["topics"]
        assert "deep learning" not in enriched["topics"]

    def test_enrich_chunk_sets_primary_topic(self, modeler):
        """Test enrich_chunk sets primary_topic."""
        chunk = {"text": "machine learning algorithms"}
        global_topics = [
            {"topic": "machine learning", "frequency": 5, "weight": 0.5},
        ]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert "primary_topic" in enriched
        assert enriched["primary_topic"] == "machine learning"

    def test_enrich_chunk_primary_topic_none_if_no_match(self, modeler):
        """Test primary_topic is None if no topics match."""
        chunk = {"text": "random text"}
        global_topics = [
            {"topic": "machine learning", "frequency": 5, "weight": 0.5},
        ]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert enriched["primary_topic"] is None

    def test_enrich_chunk_case_insensitive_matching(self, modeler):
        """Test enrich_chunk matches case-insensitively."""
        chunk = {"text": "MACHINE LEARNING"}
        global_topics = [
            {"topic": "machine learning", "frequency": 5, "weight": 0.5},
        ]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert "machine learning" in enriched["topics"]

    def test_enrich_chunk_empty_global_topics(self, modeler):
        """Test enrich_chunk handles empty global topics."""
        chunk = {"text": "test"}
        enriched = modeler.enrich_chunk(chunk, [])
        assert enriched["topics"] == []
        assert enriched["primary_topic"] is None

    def test_enrich_chunk_empty_text(self, modeler):
        """Test enrich_chunk handles empty text."""
        chunk = {"text": ""}
        global_topics = [
            {"topic": "test", "frequency": 5, "weight": 0.5},
        ]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert enriched["topics"] == []

    def test_enrich_chunk_no_text_key(self, modeler):
        """Test enrich_chunk handles missing text key."""
        chunk = {"content": "test"}
        global_topics = [
            {"topic": "test", "frequency": 5, "weight": 0.5},
        ]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert "topics" in enriched

    # Module-Level Function Tests
    def test_extract_topics_function(self, sample_chunks):
        """Test module-level extract_topics function."""
        topics = extract_topics(sample_chunks)
        assert isinstance(topics, list)

    def test_extract_topics_function_num_topics(self, sample_chunks):
        """Test extract_topics function respects num_topics."""
        topics = extract_topics(sample_chunks, num_topics=3)
        assert len(topics) <= 3

    def test_enrich_with_topics_function(self):
        """Test module-level enrich_with_topics function."""
        chunks = [{"text": "machine learning"}]
        topics = [{"topic": "machine learning", "frequency": 5, "weight": 0.5}]
        enriched = enrich_with_topics(chunks, topics)
        assert isinstance(enriched, list)
        assert len(enriched) == len(chunks)

    def test_enrich_with_topics_function_enriches_all(self):
        """Test enrich_with_topics enriches all chunks."""
        chunks = [
            {"text": "machine learning"},
            {"text": "deep learning"},
        ]
        topics = [
            {"topic": "machine learning", "frequency": 5, "weight": 0.5},
            {"topic": "deep learning", "frequency": 3, "weight": 0.3},
        ]
        enriched = enrich_with_topics(chunks, topics)
        assert all("topics" in c for c in enriched)
        assert all("primary_topic" in c for c in enriched)

    # Edge Cases
    def test_extract_topics_unicode_text(self, modeler):
        """Test topic extraction with unicode text."""
        chunks = [{"text": "café résumé naïve"}]
        topics = modeler.extract_topics(chunks)
        assert isinstance(topics, list)

    def test_extract_topics_numbers(self, modeler):
        """Test topic extraction with numbers."""
        chunks = [{"text": "python 3.9 version 2024"}]
        topics = modeler.extract_topics(chunks)
        assert isinstance(topics, list)

    def test_extract_topics_mixed_content(self, modeler):
        """Test topic extraction with mixed content."""
        chunks = [
            {"text": "Machine Learning"},
            {"text": "deep learning"},
            {"text": "NEURAL NETWORKS"},
        ]
        topics = modeler.extract_topics(chunks)
        assert isinstance(topics, list)

    def test_extract_topics_very_long_text(self, modeler):
        """Test topic extraction with very long text."""
        long_text = " ".join(["machine learning"] * 100)
        chunks = [{"text": long_text}]
        topics = modeler.extract_topics(chunks)
        assert isinstance(topics, list)

    def test_extract_topics_repeated_terms(self, modeler):
        """Test topic extraction handles repeated terms."""
        chunks = [
            {"text": "python python python"},
            {"text": "python python"},
        ]
        topics = modeler.extract_topics(chunks)
        if topics:
            # 'python' should appear with correct frequency
            assert topics[0]["frequency"] > 1

    def test_enrich_chunk_preserves_existing_fields(self, modeler):
        """Test enrich_chunk preserves existing chunk fields."""
        chunk = {"text": "machine learning", "id": "chunk_1", "source": "test.txt"}
        global_topics = [{"topic": "machine learning", "frequency": 5, "weight": 0.5}]
        enriched = modeler.enrich_chunk(chunk, global_topics)
        assert enriched["id"] == "chunk_1"
        assert enriched["source"] == "test.txt"

    def test_extract_terms_punctuation(self, modeler):
        """Test term extraction handles punctuation."""
        terms = modeler._extract_terms("Hello, world! This is a test.")
        assert isinstance(terms, list)
        # Punctuation should be removed
        assert not any("," in t for t in terms)
        assert not any("!" in t for t in terms)
        assert not any("." in t for t in terms)

    # Integration Tests
    def test_full_pipeline(self, modeler, sample_chunks):
        """Test full topic modeling pipeline."""
        # Extract topics
        topics = modeler.extract_topics(sample_chunks, num_topics=5)
        assert isinstance(topics, list)

        # Enrich chunks
        if topics:
            enriched = [modeler.enrich_chunk(c, topics) for c in sample_chunks]
            assert len(enriched) == len(sample_chunks)
            assert all("topics" in c for c in enriched)

    def test_min_term_freq_affects_results(self):
        """Test different min_term_freq values produce different results."""
        chunks = [
            {"text": "python programming language"},
            {"text": "python python"},
        ]

        modeler_freq_1 = TopicModeler(min_term_freq=1)
        modeler_freq_3 = TopicModeler(min_term_freq=3)

        topics_freq_1 = modeler_freq_1.extract_topics(chunks)
        topics_freq_3 = modeler_freq_3.extract_topics(chunks)

        # Lower frequency threshold should allow more topics
        assert len(topics_freq_1) >= len(topics_freq_3)

    def test_stop_words_affect_extraction(self, modeler):
        """Test stop words are effectively filtered."""
        chunks = [
            {"text": "the machine learning and deep learning"},
        ]
        topics = modeler.extract_topics(chunks)

        # Topics should not contain stop words
        topic_terms = [t["topic"] for t in topics]
        assert not any("the" in term for term in topic_terms)
        assert not any("and" in term for term in topic_terms)
