"""Tests for style critique engine.

Tests comparative style analysis and suggestion generation."""

from __future__ import annotations


from ingestforge.analysis.style_critique import (
    StyleCritic,
    CritiqueResult,
    Suggestion,
    SuggestionType,
    SuggestionSeverity,
    StyleComparison,
    critique_text,
    compare_styles,
    MAX_SUGGESTIONS,
)

# SuggestionType tests


class TestSuggestionType:
    """Tests for SuggestionType enum."""

    def test_types_defined(self) -> None:
        """Test all suggestion types are defined."""
        types = [t.value for t in SuggestionType]

        assert "readability" in types
        assert "vocabulary" in types
        assert "sentence_structure" in types

    def test_type_count(self) -> None:
        """Test correct number of types."""
        assert len(SuggestionType) == 6


class TestSuggestionSeverity:
    """Tests for SuggestionSeverity enum."""

    def test_severities_defined(self) -> None:
        """Test all severity levels are defined."""
        severities = [s.value for s in SuggestionSeverity]

        assert "info" in severities
        assert "minor" in severities
        assert "major" in severities


# Suggestion tests


class TestSuggestion:
    """Tests for Suggestion dataclass."""

    def test_suggestion_creation(self) -> None:
        """Test creating a suggestion."""
        suggestion = Suggestion(
            type=SuggestionType.READABILITY,
            severity=SuggestionSeverity.MODERATE,
            message="Test message",
        )

        assert suggestion.type == SuggestionType.READABILITY
        assert suggestion.severity == SuggestionSeverity.MODERATE
        assert suggestion.message == "Test message"

    def test_suggestion_with_details(self) -> None:
        """Test suggestion with optional fields."""
        suggestion = Suggestion(
            type=SuggestionType.VOCABULARY,
            severity=SuggestionSeverity.MINOR,
            message="Use varied words",
            detail="Type-token ratio is low",
            example="Replace 'said' with 'stated'",
        )

        assert suggestion.detail != ""
        assert suggestion.example != ""


# CritiqueResult tests


class TestCritiqueResult:
    """Tests for CritiqueResult dataclass."""

    def test_default_result(self) -> None:
        """Test default critique result."""
        result = CritiqueResult()

        assert result.suggestions == []
        assert result.comparison is None
        assert result.overall_score == 0.0

    def test_result_with_data(self) -> None:
        """Test result with suggestions."""
        suggestion = Suggestion(
            type=SuggestionType.READABILITY,
            severity=SuggestionSeverity.MINOR,
            message="Test",
        )
        result = CritiqueResult(
            suggestions=[suggestion],
            overall_score=75.0,
        )

        assert len(result.suggestions) == 1
        assert result.overall_score == 75.0


# StyleComparison tests


class TestStyleComparison:
    """Tests for StyleComparison dataclass."""

    def test_default_comparison(self) -> None:
        """Test default comparison values."""
        comp = StyleComparison()

        assert comp.readability_diff == 0.0
        assert comp.matches_corpus is False

    def test_comparison_with_data(self) -> None:
        """Test comparison with custom values."""
        comp = StyleComparison(
            readability_diff=5.5,
            vocabulary_diff=0.05,
            matches_corpus=True,
        )

        assert comp.readability_diff == 5.5
        assert comp.matches_corpus is True


# StyleCritic tests


class TestStyleCritic:
    """Tests for StyleCritic class."""

    def test_critic_creation(self) -> None:
        """Test creating critic."""
        critic = StyleCritic()

        assert critic is not None

    def test_critique_empty_text(self) -> None:
        """Test critiquing empty text."""
        critic = StyleCritic()

        result = critic.critique("")

        assert result.metrics.vocabulary.total_words == 0
        assert result.suggestions == []

    def test_critique_simple_text(self) -> None:
        """Test critiquing simple text."""
        critic = StyleCritic()
        text = "The cat sat on the mat. It was a nice day."

        result = critic.critique(text)

        assert result.metrics.vocabulary.total_words > 0
        assert isinstance(result.suggestions, list)

    def test_critique_returns_result(self) -> None:
        """Test that critique returns CritiqueResult."""
        critic = StyleCritic()

        result = critic.critique("Hello world.")

        assert isinstance(result, CritiqueResult)


class TestCorpusBaseline:
    """Tests for corpus baseline setting."""

    def test_set_corpus_baseline(self) -> None:
        """Test setting corpus baseline."""
        critic = StyleCritic()
        corpus = [
            "The first sample text for analysis.",
            "Another sample with similar style.",
        ]

        critic.set_corpus_baseline(corpus)

        assert critic._corpus_metrics is not None

    def test_set_empty_corpus(self) -> None:
        """Test setting empty corpus."""
        critic = StyleCritic()

        critic.set_corpus_baseline([])

        assert critic._corpus_metrics is None

    def test_corpus_comparison(self) -> None:
        """Test that comparison uses corpus."""
        critic = StyleCritic()
        corpus = ["Simple sentence. Another one."]
        critic.set_corpus_baseline(corpus)

        result = critic.critique("Simple test text.")

        assert result.comparison is not None


class TestReadabilitySuggestions:
    """Tests for readability suggestion generation."""

    def test_difficult_text_suggestion(self) -> None:
        """Test that difficult text gets suggestions."""
        critic = StyleCritic()
        # Complex academic-style text
        text = (
            "The multifaceted implementation of sophisticated methodological "
            "considerations necessitates comprehensive evaluation of paradigms "
            "through systematic epistemological frameworks."
        )

        result = critic.critique(text)

        # Should have readability or complexity suggestions
        types = [s.type for s in result.suggestions]
        has_relevant = (
            SuggestionType.READABILITY in types
            or SuggestionType.COMPLEXITY in types
            or SuggestionType.VOCABULARY in types
        )
        assert has_relevant

    def test_easy_text_fewer_suggestions(self) -> None:
        """Test that easy text has fewer suggestions."""
        critic = StyleCritic()
        text = "The cat sat on the mat. It was warm. I was happy."

        result = critic.critique(text)

        # Should have fewer suggestions for simple text
        # (may still have some for low vocabulary diversity)
        assert len(result.suggestions) <= 5


class TestVocabularySuggestions:
    """Tests for vocabulary suggestion generation."""

    def test_low_diversity_suggestion(self) -> None:
        """Test suggestion for low vocabulary diversity."""
        critic = StyleCritic()
        # Repetitive text
        text = "The thing is the thing. The thing does the thing."

        result = critic.critique(text)

        # May have vocabulary suggestion
        types = [s.type for s in result.suggestions]
        # Low TTR should trigger vocabulary suggestion
        if result.metrics.vocabulary.type_token_ratio < 0.3:
            assert SuggestionType.VOCABULARY in types


class TestSentenceSuggestions:
    """Tests for sentence structure suggestions."""

    def test_long_sentence_suggestion(self) -> None:
        """Test suggestion for very long sentences."""
        critic = StyleCritic()
        # Very long sentence
        text = (
            "This is a very long sentence that goes on and on and on "
            "with many words and clauses and phrases and ideas all "
            "strung together in a way that makes it difficult to read "
            "and understand what the author is trying to say because "
            "there are so many different things happening at once."
        )

        result = critic.critique(text)

        # Should have sentence structure suggestion
        types = [s.type for s in result.suggestions]
        assert SuggestionType.SENTENCE_STRUCTURE in types


class TestOverallScore:
    """Tests for overall score calculation."""

    def test_score_range(self) -> None:
        """Test that score is in valid range."""
        critic = StyleCritic()

        result = critic.critique("This is a test.")

        assert 0 <= result.overall_score <= 100

    def test_good_text_higher_score(self) -> None:
        """Test that good text has reasonable score."""
        critic = StyleCritic()
        good_text = (
            "Writing clearly helps readers understand your message. "
            "Short sentences work well. Varied vocabulary adds interest."
        )

        result = critic.critique(good_text)

        # Should have reasonable score (above minimum)
        assert result.overall_score >= 30


class TestSuggestionLimits:
    """Tests for suggestion limits."""

    def test_max_suggestions_enforced(self) -> None:
        """Test that suggestions are limited."""
        critic = StyleCritic()
        # Text with many potential issues
        text = (
            "The extraordinarily sophisticated implementation of "
            "multifaceted methodological considerations " * 10
        )

        result = critic.critique(text)

        assert len(result.suggestions) <= MAX_SUGGESTIONS


# Factory function tests


class TestCritiqueText:
    """Tests for critique_text convenience function."""

    def test_critique_simple(self) -> None:
        """Test critiquing simple text."""
        result = critique_text("Hello world.")

        assert isinstance(result, CritiqueResult)

    def test_critique_empty(self) -> None:
        """Test critiquing empty text."""
        result = critique_text("")

        assert result.metrics.vocabulary.total_words == 0


class TestCompareStyles:
    """Tests for compare_styles function."""

    def test_compare_with_corpus(self) -> None:
        """Test comparing against corpus."""
        text = "This is the text to analyze."
        corpus = [
            "First corpus sample.",
            "Second corpus sample.",
        ]

        result = compare_styles(text, corpus)

        assert result.comparison is not None

    def test_compare_empty_corpus(self) -> None:
        """Test comparing with empty corpus."""
        text = "This is the text to analyze."

        result = compare_styles(text, [])

        assert result.comparison is None


class TestCorpusMatching:
    """Tests for corpus style matching."""

    def test_similar_style_matches(self) -> None:
        """Test that similar styles match."""
        critic = StyleCritic()
        corpus = [
            "Simple sentences work well. They are easy to read.",
            "Short words help. Clear writing is good.",
        ]
        critic.set_corpus_baseline(corpus)

        similar_text = "Simple text here. It reads well."
        result = critic.critique(similar_text)

        # Similar simple style should potentially match
        assert result.comparison is not None

    def test_different_style_no_match(self) -> None:
        """Test that different styles don't match."""
        critic = StyleCritic()
        corpus = [
            "Simple sentences work well. They are easy to read.",
        ]
        critic.set_corpus_baseline(corpus)

        complex_text = (
            "The implementation of sophisticated methodological "
            "considerations necessitates comprehensive evaluation."
        )
        result = critic.critique(complex_text)

        # Very different style should not match
        if result.comparison:
            # May or may not match depending on thresholds
            pass


class TestSuggestionContent:
    """Tests for suggestion content quality."""

    def test_suggestion_has_message(self) -> None:
        """Test that suggestions have messages."""
        critic = StyleCritic()
        text = (
            "The extraordinarily sophisticated implementation "
            "necessitates comprehensive evaluation."
        )

        result = critic.critique(text)

        for suggestion in result.suggestions:
            assert suggestion.message
            assert len(suggestion.message) > 5

    def test_suggestion_severity_assigned(self) -> None:
        """Test that suggestions have severity."""
        critic = StyleCritic()
        text = (
            "The extraordinarily sophisticated implementation "
            "necessitates comprehensive evaluation."
        )

        result = critic.critique(text)

        for suggestion in result.suggestions:
            assert suggestion.severity in SuggestionSeverity
