"""Tests for linguistic metric collector.

Tests readability scores, vocabulary diversity, and sentence metrics."""

from __future__ import annotations


from ingestforge.analysis.metrics import (
    MetricCollector,
    StyleMetrics,
    ReadabilityScores,
    VocabularyMetrics,
    SentenceMetrics,
    ReadabilityLevel,
    analyze_text,
    get_readability_summary,
    MAX_TEXT_LENGTH,
    MAX_WORDS_TO_ANALYZE,
)

# ReadabilityLevel tests


class TestReadabilityLevel:
    """Tests for ReadabilityLevel enum."""

    def test_levels_defined(self) -> None:
        """Test all readability levels are defined."""
        levels = [l.value for l in ReadabilityLevel]

        assert "very_easy" in levels
        assert "standard" in levels
        assert "very_difficult" in levels

    def test_level_count(self) -> None:
        """Test correct number of levels."""
        assert len(ReadabilityLevel) == 7


# ReadabilityScores tests


class TestReadabilityScores:
    """Tests for ReadabilityScores dataclass."""

    def test_default_scores(self) -> None:
        """Test default score values."""
        scores = ReadabilityScores()

        assert scores.flesch_reading_ease == 0.0
        assert scores.flesch_kincaid_grade == 0.0
        assert scores.level == ReadabilityLevel.STANDARD

    def test_custom_scores(self) -> None:
        """Test custom score values."""
        scores = ReadabilityScores(
            flesch_reading_ease=75.5,
            flesch_kincaid_grade=6.2,
            level=ReadabilityLevel.FAIRLY_EASY,
        )

        assert scores.flesch_reading_ease == 75.5
        assert scores.level == ReadabilityLevel.FAIRLY_EASY


# VocabularyMetrics tests


class TestVocabularyMetrics:
    """Tests for VocabularyMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default vocabulary values."""
        vocab = VocabularyMetrics()

        assert vocab.type_token_ratio == 0.0
        assert vocab.unique_words == 0
        assert vocab.total_words == 0

    def test_custom_values(self) -> None:
        """Test custom vocabulary values."""
        vocab = VocabularyMetrics(
            type_token_ratio=0.65,
            unique_words=50,
            total_words=100,
            average_word_length=5.2,
        )

        assert vocab.type_token_ratio == 0.65
        assert vocab.unique_words == 50


# SentenceMetrics tests


class TestSentenceMetrics:
    """Tests for SentenceMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default sentence values."""
        sent = SentenceMetrics()

        assert sent.total_sentences == 0
        assert sent.average_sentence_length == 0.0

    def test_custom_values(self) -> None:
        """Test custom sentence values."""
        sent = SentenceMetrics(
            total_sentences=10,
            average_sentence_length=15.5,
            max_sentence_length=30,
            min_sentence_length=5,
        )

        assert sent.total_sentences == 10
        assert sent.average_sentence_length == 15.5


# MetricCollector tests


class TestMetricCollector:
    """Tests for MetricCollector class."""

    def test_collector_creation(self) -> None:
        """Test creating collector."""
        collector = MetricCollector()

        assert collector is not None

    def test_analyze_empty_text(self) -> None:
        """Test analyzing empty text."""
        collector = MetricCollector()

        result = collector.analyze("")

        assert result.vocabulary.total_words == 0
        assert result.sentences.total_sentences == 0

    def test_analyze_simple_text(self) -> None:
        """Test analyzing simple text."""
        collector = MetricCollector()
        text = "The cat sat on the mat. It was a nice day."

        result = collector.analyze(text)

        assert result.vocabulary.total_words > 0
        assert result.sentences.total_sentences == 2

    def test_analyze_returns_style_metrics(self) -> None:
        """Test that analyze returns StyleMetrics."""
        collector = MetricCollector()
        text = "Hello world."

        result = collector.analyze(text)

        assert isinstance(result, StyleMetrics)
        assert isinstance(result.readability, ReadabilityScores)
        assert isinstance(result.vocabulary, VocabularyMetrics)


class TestSyllableCounting:
    """Tests for syllable counting."""

    def test_single_syllable(self) -> None:
        """Test single syllable words."""
        collector = MetricCollector()

        assert collector._count_syllables("cat") == 1
        assert collector._count_syllables("dog") == 1
        assert collector._count_syllables("the") == 1

    def test_two_syllables(self) -> None:
        """Test two syllable words."""
        collector = MetricCollector()

        assert collector._count_syllables("happy") == 2
        assert collector._count_syllables("water") == 2

    def test_three_syllables(self) -> None:
        """Test three syllable words."""
        collector = MetricCollector()

        assert collector._count_syllables("beautiful") == 3
        assert collector._count_syllables("computer") == 3

    def test_silent_e(self) -> None:
        """Test words ending in silent e."""
        collector = MetricCollector()

        # "make" should be 1 syllable
        assert collector._count_syllables("make") == 1


class TestWordTokenization:
    """Tests for word tokenization."""

    def test_tokenize_simple(self) -> None:
        """Test simple word tokenization."""
        collector = MetricCollector()

        words = collector._tokenize_words("Hello world")

        assert "hello" in words
        assert "world" in words

    def test_tokenize_with_punctuation(self) -> None:
        """Test tokenization ignores punctuation."""
        collector = MetricCollector()

        words = collector._tokenize_words("Hello, world! How are you?")

        assert "hello" in words
        assert len([w for w in words if w == "hello"]) == 1

    def test_tokenize_limits_words(self) -> None:
        """Test that tokenization respects limit."""
        collector = MetricCollector()
        text = " ".join(["word"] * (MAX_WORDS_TO_ANALYZE + 100))

        words = collector._tokenize_words(text)

        assert len(words) == MAX_WORDS_TO_ANALYZE


class TestSentenceTokenization:
    """Tests for sentence tokenization."""

    def test_tokenize_sentences(self) -> None:
        """Test sentence tokenization."""
        collector = MetricCollector()
        text = "First sentence. Second sentence! Third one?"

        sentences = collector._tokenize_sentences(text)

        assert len(sentences) == 3

    def test_empty_sentences_filtered(self) -> None:
        """Test that empty sentences are filtered."""
        collector = MetricCollector()
        text = "Hello.   . World."

        sentences = collector._tokenize_sentences(text)

        # Should only have non-empty sentences
        assert all(s.strip() for s in sentences)


class TestVocabularyCalculations:
    """Tests for vocabulary metric calculations."""

    def test_type_token_ratio(self) -> None:
        """Test type-token ratio calculation."""
        collector = MetricCollector()
        text = "the cat and the dog and the bird"  # 8 words, 5 unique

        result = collector.analyze(text)

        assert result.vocabulary.total_words == 8
        assert result.vocabulary.unique_words == 5
        assert 0.6 <= result.vocabulary.type_token_ratio <= 0.7

    def test_complex_word_ratio(self) -> None:
        """Test complex word ratio calculation."""
        collector = MetricCollector()
        text = "The beautiful representation demonstrated excellence."

        result = collector.analyze(text)

        # Most words are complex (3+ syllables)
        assert result.vocabulary.complex_word_ratio > 0.5


class TestReadabilityCalculations:
    """Tests for readability calculations."""

    def test_easy_text_readability(self) -> None:
        """Test easy text has high readability."""
        collector = MetricCollector()
        text = "The cat sat. The dog ran. I am here."

        result = collector.analyze(text)

        # Simple text should be easy to read
        assert result.readability.flesch_reading_ease > 60

    def test_complex_text_readability(self) -> None:
        """Test complex text has lower readability."""
        collector = MetricCollector()
        text = (
            "The implementation of sophisticated methodological "
            "considerations necessitates comprehensive evaluation "
            "of multifaceted organizational paradigms."
        )

        result = collector.analyze(text)

        # Complex text should be harder to read
        assert result.readability.flesch_reading_ease < 50

    def test_grade_level_reasonable(self) -> None:
        """Test grade level is reasonable."""
        collector = MetricCollector()
        text = "Hello world. This is a test."

        result = collector.analyze(text)

        # Grade level should be in reasonable range
        assert 0 <= result.readability.flesch_kincaid_grade <= 20


class TestReadabilityLevelMapping:
    """Tests for readability level mapping."""

    def test_very_easy_level(self) -> None:
        """Test very easy level assignment."""
        collector = MetricCollector()

        level = collector._get_readability_level(95)

        assert level == ReadabilityLevel.VERY_EASY

    def test_standard_level(self) -> None:
        """Test standard level assignment."""
        collector = MetricCollector()

        level = collector._get_readability_level(65)

        assert level == ReadabilityLevel.STANDARD

    def test_very_difficult_level(self) -> None:
        """Test very difficult level assignment."""
        collector = MetricCollector()

        level = collector._get_readability_level(20)

        assert level == ReadabilityLevel.VERY_DIFFICULT


# Factory function tests


class TestAnalyzeText:
    """Tests for analyze_text convenience function."""

    def test_analyze_text_simple(self) -> None:
        """Test analyzing simple text."""
        result = analyze_text("Hello world.")

        assert result.vocabulary.total_words == 2

    def test_analyze_text_empty(self) -> None:
        """Test analyzing empty text."""
        result = analyze_text("")

        assert result.vocabulary.total_words == 0


class TestGetReadabilitySummary:
    """Tests for get_readability_summary function."""

    def test_summary_keys(self) -> None:
        """Test summary contains expected keys."""
        metrics = analyze_text("This is a test sentence.")

        summary = get_readability_summary(metrics)

        assert "reading_level" in summary
        assert "flesch_ease" in summary
        assert "grade_level" in summary

    def test_summary_values(self) -> None:
        """Test summary values are strings."""
        metrics = analyze_text("Hello world.")

        summary = get_readability_summary(metrics)

        for value in summary.values():
            assert isinstance(value, str)


class TestLongTextHandling:
    """Tests for handling long text."""

    def test_truncates_long_text(self) -> None:
        """Test that very long text is truncated."""
        collector = MetricCollector()
        long_text = "word " * (MAX_TEXT_LENGTH + 1000)

        # Should not raise, should truncate
        result = collector.analyze(long_text)

        assert result.vocabulary.total_words > 0
        assert result.vocabulary.total_words <= MAX_WORDS_TO_ANALYZE
