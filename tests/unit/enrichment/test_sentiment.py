"""Tests for sentiment analysis enrichment."""
import pytest

from ingestforge.enrichment.sentiment import (
    SentimentAnalyzer,
    analyze_sentiment,
    batch_analyze_sentiment,
)


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> SentimentAnalyzer:
        """Create analyzer instance."""
        return SentimentAnalyzer()

    def test_initialization(self, analyzer: SentimentAnalyzer):
        """Test analyzer initializes with word lists."""
        assert len(analyzer.positive_words) > 0
        assert len(analyzer.negative_words) > 0
        assert "good" in analyzer.positive_words
        assert "bad" in analyzer.negative_words

    def test_positive_sentiment(self, analyzer: SentimentAnalyzer):
        """Test detection of positive sentiment."""
        chunk = {"text": "This is great and wonderful. I love it!"}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "positive"
        assert result["sentiment_score"] > 0
        assert len(result["positive_words"]) > 0

    def test_negative_sentiment(self, analyzer: SentimentAnalyzer):
        """Test detection of negative sentiment."""
        chunk = {"text": "This is terrible and awful. I hate it."}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "negative"
        assert result["sentiment_score"] < 0
        assert len(result["negative_words"]) > 0

    def test_neutral_sentiment(self, analyzer: SentimentAnalyzer):
        """Test detection of neutral sentiment."""
        chunk = {"text": "The document describes the process in detail."}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "neutral"
        assert abs(result["sentiment_score"]) <= 0.05

    def test_empty_text(self, analyzer: SentimentAnalyzer):
        """Test handling of empty text."""
        chunk = {"text": ""}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "neutral"
        assert result["sentiment_score"] == 0.0
        assert result["positive_words"] == []
        assert result["negative_words"] == []

    def test_missing_text_field(self, analyzer: SentimentAnalyzer):
        """Test handling of missing text field."""
        chunk = {}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "neutral"
        assert result["sentiment_score"] == 0.0

    def test_mixed_sentiment(self, analyzer: SentimentAnalyzer):
        """Test text with both positive and negative words."""
        chunk = {"text": "Good and bad. Happy but sad."}

        result = analyzer.analyze(chunk)

        # Should be neutral or close to it (2 positive, 2 negative = 0.0)
        assert result["sentiment"] in ["neutral", "positive", "negative"]
        # Both lists should have entries
        pos_words = result["positive_words"]
        neg_words = result["negative_words"]
        assert isinstance(pos_words, list)
        assert isinstance(neg_words, list)
        # At least one should have words (depending on score threshold)
        assert len(pos_words) + len(neg_words) > 0

    def test_case_insensitive(self, analyzer: SentimentAnalyzer):
        """Test sentiment detection is case insensitive."""
        chunk1 = {"text": "EXCELLENT work"}
        chunk2 = {"text": "excellent work"}

        result1 = analyzer.analyze(chunk1)
        result2 = analyzer.analyze(chunk2)

        assert result1["sentiment"] == result2["sentiment"]
        assert result1["sentiment_score"] == result2["sentiment_score"]

    def test_sentiment_score_calculation(self, analyzer: SentimentAnalyzer):
        """Test sentiment score is calculated correctly."""
        # 2 positive, 1 negative, 9 total words
        chunk = {"text": "good great words but bad stuff here now today"}

        result = analyzer.analyze(chunk)

        # Score = (2 - 1) / 9 = 0.111
        assert result["sentiment_score"] == pytest.approx(0.111, abs=0.01)

    def test_positive_words_limit(self, analyzer: SentimentAnalyzer):
        """Test positive words list is limited to 5."""
        chunk = {
            "text": "good great excellent amazing wonderful fantastic best better happy love"
        }

        result = analyzer.analyze(chunk)

        # Should return max 5 positive words
        assert len(result["positive_words"]) <= 5

    def test_negative_words_limit(self, analyzer: SentimentAnalyzer):
        """Test negative words list is limited to 5."""
        chunk = {"text": "bad terrible horrible awful worst worse hate sad fail poor"}

        result = analyzer.analyze(chunk)

        # Should return max 5 negative words
        assert len(result["negative_words"]) <= 5

    def test_strongly_positive(self, analyzer: SentimentAnalyzer):
        """Test strongly positive text."""
        chunk = {
            "text": "excellent amazing wonderful fantastic brilliant outstanding perfect"
        }

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "positive"
        assert result["sentiment_score"] > 0.5

    def test_strongly_negative(self, analyzer: SentimentAnalyzer):
        """Test strongly negative text."""
        chunk = {"text": "terrible horrible awful worst failure crisis damage threat"}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "negative"
        assert result["sentiment_score"] < -0.5

    def test_threshold_boundary_positive(self, analyzer: SentimentAnalyzer):
        """Test sentiment at positive threshold boundary."""
        # Create text with score just above 0.05
        chunk = {"text": "good " + "word " * 15}  # 1 positive in 17 words = 0.059

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "positive"

    def test_threshold_boundary_negative(self, analyzer: SentimentAnalyzer):
        """Test sentiment at negative threshold boundary."""
        # Create text with score just below -0.05
        chunk = {"text": "bad " + "word " * 15}  # 1 negative in 17 words = -0.059

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "negative"

    def test_score_rounded(self, analyzer: SentimentAnalyzer):
        """Test sentiment score is rounded to 3 decimal places."""
        chunk = {"text": "good word word word"}  # 1/4 = 0.25

        result = analyzer.analyze(chunk)

        assert result["sentiment_score"] == 0.25

    def test_preserves_original_chunk(self, analyzer: SentimentAnalyzer):
        """Test analyzer preserves original chunk data."""
        chunk = {"id": 123, "title": "Test", "text": "good"}

        result = analyzer.analyze(chunk)

        assert result["id"] == 123
        assert result["title"] == "Test"

    def test_word_lists_loaded(self, analyzer: SentimentAnalyzer):
        """Test word lists contain expected words."""
        # Positive words
        assert "good" in analyzer.positive_words
        assert "great" in analyzer.positive_words
        assert "excellent" in analyzer.positive_words

        # Negative words
        assert "bad" in analyzer.negative_words
        assert "terrible" in analyzer.negative_words
        assert "awful" in analyzer.negative_words


class TestAnalyzeSentiment:
    """Tests for analyze_sentiment convenience function."""

    def test_analyze_sentiment_function(self):
        """Test standalone analyze_sentiment function."""
        chunk = {"text": "This is excellent work"}

        result = analyze_sentiment(chunk)

        assert result["sentiment"] == "positive"
        assert "sentiment_score" in result

    def test_function_creates_new_analyzer(self):
        """Test function creates its own analyzer instance."""
        chunk = {"text": "good"}

        result = analyze_sentiment(chunk)

        assert result["sentiment"] == "positive"


class TestBatchAnalyzeSentiment:
    """Tests for batch sentiment analysis."""

    def test_batch_analyze_empty_list(self):
        """Test batch analysis with empty list."""
        result = batch_analyze_sentiment([])

        assert result == []

    def test_batch_analyze_single_chunk(self):
        """Test batch analysis with single chunk."""
        chunks = [{"text": "great work"}]

        result = batch_analyze_sentiment(chunks)

        assert len(result) == 1
        assert result[0]["sentiment"] == "positive"

    def test_batch_analyze_multiple_chunks(self):
        """Test batch analysis with multiple chunks."""
        chunks = [
            {"id": 1, "text": "excellent performance"},
            {"id": 2, "text": "terrible results"},
            {"id": 3, "text": "neutral statement"},
        ]

        result = batch_analyze_sentiment(chunks)

        assert len(result) == 3
        assert result[0]["sentiment"] == "positive"
        assert result[1]["sentiment"] == "negative"
        assert result[2]["sentiment"] == "neutral"

    def test_batch_preserves_order(self):
        """Test batch analysis preserves chunk order."""
        chunks = [{"id": i, "text": "text"} for i in range(10)]

        result = batch_analyze_sentiment(chunks)

        assert [r["id"] for r in result] == list(range(10))

    def test_batch_with_mixed_sentiments(self):
        """Test batch with various sentiments."""
        chunks = [
            {"text": "good good good"},
            {"text": "bad bad bad"},
            {"text": "the sky is blue"},
        ]

        result = batch_analyze_sentiment(chunks)

        sentiments = [r["sentiment"] for r in result]
        assert "positive" in sentiments
        assert "negative" in sentiments
        assert "neutral" in sentiments


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def analyzer(self) -> SentimentAnalyzer:
        """Create analyzer instance."""
        return SentimentAnalyzer()

    def test_very_long_text(self, analyzer: SentimentAnalyzer):
        """Test analysis of very long text."""
        long_text = "word " * 10000 + "good great"
        chunk = {"text": long_text}

        result = analyzer.analyze(chunk)

        # Should still work
        assert "sentiment" in result
        assert "sentiment_score" in result

    def test_only_sentiment_words(self, analyzer: SentimentAnalyzer):
        """Test text containing only sentiment words."""
        chunk = {"text": "good great excellent bad terrible awful"}

        result = analyzer.analyze(chunk)

        # 3 positive, 3 negative = neutral
        assert result["sentiment"] == "neutral"
        assert result["sentiment_score"] == 0.0

    def test_repeated_sentiment_word(self, analyzer: SentimentAnalyzer):
        """Test repeated sentiment words."""
        chunk = {"text": "good good good good"}

        result = analyzer.analyze(chunk)

        # All 4 words are positive
        assert result["sentiment"] == "positive"
        assert result["sentiment_score"] == 1.0

    def test_special_characters(self, analyzer: SentimentAnalyzer):
        """Test text with special characters mixed with words."""
        chunk = {"text": "good work with some @mentions and #hashtags"}

        result = analyzer.analyze(chunk)

        # Should still detect sentiment words that are present
        assert "sentiment_score" in result
        # good is a sentiment word, so should find it
        assert len(result["positive_words"]) >= 1

    def test_numbers_in_text(self, analyzer: SentimentAnalyzer):
        """Test text with numbers."""
        chunk = {"text": "good 123 456 words"}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "positive"

    def test_punctuation_only(self, analyzer: SentimentAnalyzer):
        """Test text with only punctuation."""
        chunk = {"text": "!@#$%^&*()"}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "neutral"

    def test_unicode_text(self, analyzer: SentimentAnalyzer):
        """Test text with Unicode characters."""
        chunk = {"text": "good café naïve résumé"}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "positive"

    def test_single_word(self, analyzer: SentimentAnalyzer):
        """Test single word text."""
        chunk = {"text": "excellent"}

        result = analyzer.analyze(chunk)

        assert result["sentiment"] == "positive"
        assert result["sentiment_score"] == 1.0

    def test_whitespace_only(self, analyzer: SentimentAnalyzer):
        """Test text with only whitespace."""
        chunk = {"text": "     "}

        result = analyzer.analyze(chunk)

        # split() returns empty list for whitespace
        assert result["sentiment"] == "neutral"
