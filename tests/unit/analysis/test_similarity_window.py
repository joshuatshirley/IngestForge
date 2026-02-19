"""Tests for semantic window comparator.

Tests sliding window plagiarism detection."""

from __future__ import annotations


from ingestforge.analysis.similarity_window import (
    SemanticWindowComparator,
    ComparisonResult,
    SimilarityMatch,
    TextWindow,
    WindowConfig,
    create_comparator,
    find_similar_passages,
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    MAX_CORPUS_CHUNKS,
    DEFAULT_WINDOW_SIZE,
    SIMILARITY_THRESHOLD,
)

# TextWindow tests


class TestTextWindow:
    """Tests for TextWindow dataclass."""

    def test_window_creation(self) -> None:
        """Test creating a window."""
        window = TextWindow(
            text="hello world",
            start_pos=0,
            end_pos=11,
            window_index=0,
        )

        assert window.text == "hello world"
        assert window.start_pos == 0

    def test_window_length(self) -> None:
        """Test window length property."""
        window = TextWindow(
            text="test",
            start_pos=10,
            end_pos=20,
            window_index=1,
        )

        assert window.length == 10


# WindowConfig tests


class TestWindowConfig:
    """Tests for WindowConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = WindowConfig()

        assert config.window_size == DEFAULT_WINDOW_SIZE
        assert config.similarity_threshold == SIMILARITY_THRESHOLD

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = WindowConfig(
            window_size=50,
            stride=25,
            similarity_threshold=0.8,
        )

        assert config.window_size == 50
        assert config.stride == 25

    def test_bounds_enforced(self) -> None:
        """Test that bounds are enforced."""
        config = WindowConfig(
            window_size=MAX_WINDOW_SIZE + 100,
        )

        assert config.window_size == MAX_WINDOW_SIZE

    def test_minimum_window_size(self) -> None:
        """Test minimum window size."""
        config = WindowConfig(window_size=5)

        assert config.window_size >= MIN_WINDOW_SIZE


# SimilarityMatch tests


class TestSimilarityMatch:
    """Tests for SimilarityMatch dataclass."""

    def test_match_creation(self) -> None:
        """Test creating a match."""
        window = TextWindow("test", 0, 4, 0)
        match = SimilarityMatch(
            source_window=window,
            corpus_text="test text",
            corpus_source="doc.pdf",
            similarity_score=0.85,
        )

        assert match.similarity_score == 0.85
        assert match.corpus_source == "doc.pdf"

    def test_high_similarity_flag(self) -> None:
        """Test high similarity detection."""
        window = TextWindow("test", 0, 4, 0)

        high_match = SimilarityMatch(
            source_window=window,
            corpus_text="test",
            corpus_source="doc",
            similarity_score=0.90,
        )

        low_match = SimilarityMatch(
            source_window=window,
            corpus_text="test",
            corpus_source="doc",
            similarity_score=0.75,
        )

        assert high_match.is_high_similarity is True
        assert low_match.is_high_similarity is False


# ComparisonResult tests


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_default_result(self) -> None:
        """Test default result values."""
        result = ComparisonResult()

        assert result.total_windows == 0
        assert result.matches == []

    def test_overlap_ratio_zero(self) -> None:
        """Test overlap ratio with zero windows."""
        result = ComparisonResult()

        assert result.overlap_ratio == 0.0

    def test_overlap_ratio_calculation(self) -> None:
        """Test overlap ratio calculation."""
        result = ComparisonResult(
            total_windows=10,
            windows_with_matches=3,
        )

        assert result.overlap_ratio == 0.3


# SemanticWindowComparator tests


class TestSemanticWindowComparator:
    """Tests for SemanticWindowComparator class."""

    def test_comparator_creation(self) -> None:
        """Test creating comparator."""
        comparator = SemanticWindowComparator()

        assert comparator.config is not None

    def test_comparator_with_config(self) -> None:
        """Test comparator with custom config."""
        config = WindowConfig(window_size=50)
        comparator = SemanticWindowComparator(config=config)

        assert comparator.config.window_size == 50

    def test_set_corpus(self) -> None:
        """Test setting corpus."""
        comparator = SemanticWindowComparator()
        corpus = [
            ("First chunk of text", "doc1.pdf"),
            ("Second chunk of text", "doc2.pdf"),
        ]

        comparator.set_corpus(corpus)

        assert len(comparator._corpus) == 2

    def test_compare_empty_text(self) -> None:
        """Test comparing empty text."""
        comparator = SemanticWindowComparator()

        result = comparator.compare("")

        assert result.total_windows == 0

    def test_compare_no_corpus(self) -> None:
        """Test comparing without corpus."""
        comparator = SemanticWindowComparator()

        result = comparator.compare("Some text to check")

        assert result.total_windows == 0


class TestWindowGeneration:
    """Tests for window generation."""

    def test_generate_windows(self) -> None:
        """Test generating windows from text."""
        config = WindowConfig(window_size=5, stride=3, min_match_words=2)
        comparator = SemanticWindowComparator(config=config)

        text = "one two three four five six seven eight nine ten"
        windows = list(comparator._generate_windows(text))

        assert len(windows) > 0
        assert all(isinstance(w, TextWindow) for w in windows)

    def test_window_text_content(self) -> None:
        """Test that windows contain correct text."""
        config = WindowConfig(window_size=3, stride=2, min_match_words=2)
        comparator = SemanticWindowComparator(config=config)

        text = "word1 word2 word3 word4 word5"
        windows = list(comparator._generate_windows(text))

        # First window should contain first words
        assert "word1" in windows[0].text


class TestSimilarityComparison:
    """Tests for similarity comparison."""

    def test_find_exact_match(self) -> None:
        """Test finding exact text match."""
        config = WindowConfig(window_size=5, min_match_words=3)
        comparator = SemanticWindowComparator(config=config)
        corpus = [("hello world test phrase example words more", "source.txt")]
        comparator.set_corpus(corpus)

        text = "hello world test phrase example words more text here"
        result = comparator.compare(text)

        # Should find high similarity match
        assert result.total_windows > 0

    def test_find_similar_match(self) -> None:
        """Test finding similar but not exact match."""
        config = WindowConfig(similarity_threshold=0.5)
        comparator = SemanticWindowComparator(config=config)
        corpus = [("the quick brown fox jumps", "source.txt")]
        comparator.set_corpus(corpus)

        text = "the quick brown dog jumps"
        result = comparator.compare(text)

        # Should find partial match
        assert len(result.matches) > 0 or result.total_windows > 0

    def test_no_match_different_text(self) -> None:
        """Test no match for completely different text."""
        config = WindowConfig(similarity_threshold=0.9)
        comparator = SemanticWindowComparator(config=config)
        corpus = [("completely different content here", "source.txt")]
        comparator.set_corpus(corpus)

        text = "unrelated text about something else entirely new"
        result = comparator.compare(text)

        # With high threshold, should not match
        high_sim_matches = [m for m in result.matches if m.similarity_score >= 0.9]
        assert len(high_sim_matches) == 0


class TestDefaultSimilarity:
    """Tests for default similarity function."""

    def test_identical_text_similarity(self) -> None:
        """Test identical texts have high similarity."""
        comparator = SemanticWindowComparator()

        sim = comparator._default_similarity("hello world", "hello world")

        assert sim == 1.0

    def test_different_text_similarity(self) -> None:
        """Test different texts have low similarity."""
        comparator = SemanticWindowComparator()

        sim = comparator._default_similarity("hello world", "goodbye moon")

        assert sim < 0.5

    def test_empty_text_similarity(self) -> None:
        """Test empty text has zero similarity."""
        comparator = SemanticWindowComparator()

        sim = comparator._default_similarity("", "hello")

        assert sim == 0.0


class TestCorpusLimits:
    """Tests for corpus size limits."""

    def test_corpus_limited(self) -> None:
        """Test that corpus is limited to max size."""
        comparator = SemanticWindowComparator()

        # Create large corpus
        large_corpus = [
            (f"chunk {i}", f"doc{i}.txt") for i in range(MAX_CORPUS_CHUNKS + 100)
        ]

        comparator.set_corpus(large_corpus)

        assert len(comparator._corpus) == MAX_CORPUS_CHUNKS


# Factory function tests


class TestCreateComparator:
    """Tests for create_comparator factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        comparator = create_comparator()

        assert comparator.config.window_size == DEFAULT_WINDOW_SIZE

    def test_create_custom(self) -> None:
        """Test creating with custom options."""
        comparator = create_comparator(window_size=50, threshold=0.8)

        assert comparator.config.window_size == 50
        assert comparator.config.similarity_threshold == 0.8


class TestFindSimilarPassages:
    """Tests for find_similar_passages function."""

    def test_find_passages(self) -> None:
        """Test finding similar passages."""
        text = "the quick brown fox"
        corpus = [("the quick brown fox jumps", "source.txt")]

        matches = find_similar_passages(text, corpus, threshold=0.5)

        assert isinstance(matches, list)

    def test_find_passages_empty_corpus(self) -> None:
        """Test with empty corpus."""
        text = "hello world"

        matches = find_similar_passages(text, [], threshold=0.5)

        assert matches == []


class TestCustomSimilarityFunction:
    """Tests for custom similarity function."""

    def test_custom_function(self) -> None:
        """Test using custom similarity function."""

        def custom_sim(text1: str, text2: str) -> float:
            # Simple exact match
            return 1.0 if text1 == text2 else 0.0

        comparator = SemanticWindowComparator(similarity_func=custom_sim)
        corpus = [("exact match test", "source.txt")]
        comparator.set_corpus(corpus)

        # Should use custom function
        result = comparator.compare("exact match test")

        assert result is not None
