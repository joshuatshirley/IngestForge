"""Style Critique Engine for comparative tone analysis.

Compares writing style against corpus samples and provides
actionable suggestions for improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ingestforge.analysis.metrics import (
    MetricCollector,
    StyleMetrics,
    ReadabilityLevel,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_SUGGESTIONS = 10
MAX_CORPUS_SAMPLES = 50
MAX_COMPARISON_TEXT_LENGTH = 50000


class SuggestionType(str, Enum):
    """Types of style suggestions."""

    READABILITY = "readability"
    VOCABULARY = "vocabulary"
    SENTENCE_STRUCTURE = "sentence_structure"
    COMPLEXITY = "complexity"
    TONE = "tone"
    CONSISTENCY = "consistency"


class SuggestionSeverity(str, Enum):
    """Severity levels for suggestions."""

    INFO = "info"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


@dataclass
class Suggestion:
    """A style improvement suggestion."""

    type: SuggestionType
    severity: SuggestionSeverity
    message: str
    detail: str = ""
    example: str = ""


@dataclass
class StyleComparison:
    """Comparison between target and corpus style."""

    readability_diff: float = 0.0
    vocabulary_diff: float = 0.0
    sentence_length_diff: float = 0.0
    complexity_diff: float = 0.0
    matches_corpus: bool = False


@dataclass
class CritiqueResult:
    """Result of style critique analysis."""

    metrics: StyleMetrics = field(default_factory=StyleMetrics)
    suggestions: List[Suggestion] = field(default_factory=list)
    comparison: Optional[StyleComparison] = None
    overall_score: float = 0.0


class StyleCritic:
    """Comparative style analysis engine.

    Analyzes text style and compares against corpus samples
    to provide actionable improvement suggestions.
    """

    def __init__(self) -> None:
        """Initialize the style critic."""
        self._collector = MetricCollector()
        self._corpus_metrics: Optional[StyleMetrics] = None

    def set_corpus_baseline(self, corpus_texts: List[str]) -> None:
        """Set baseline style from corpus samples.

        Args:
            corpus_texts: List of corpus text samples
        """
        if not corpus_texts:
            return

        # Limit samples
        samples = corpus_texts[:MAX_CORPUS_SAMPLES]

        # Collect metrics from all samples and average
        all_metrics = [self._collector.analyze(t) for t in samples]
        self._corpus_metrics = self._average_metrics(all_metrics)

    def critique(self, text: str) -> CritiqueResult:
        """Analyze text and provide style critique.

        Args:
            text: Text to critique

        Returns:
            CritiqueResult with metrics and suggestions
        """
        if not text or not text.strip():
            return CritiqueResult()

        # Truncate if needed
        if len(text) > MAX_COMPARISON_TEXT_LENGTH:
            text = text[:MAX_COMPARISON_TEXT_LENGTH]

        # Analyze text
        metrics = self._collector.analyze(text)

        # Generate suggestions
        suggestions = self._generate_suggestions(metrics)

        # Compare to corpus if available
        comparison = None
        if self._corpus_metrics:
            comparison = self._compare_to_corpus(metrics)

        # Calculate overall score
        score = self._calculate_score(metrics, comparison)

        return CritiqueResult(
            metrics=metrics,
            suggestions=suggestions[:MAX_SUGGESTIONS],
            comparison=comparison,
            overall_score=score,
        )

    def _generate_suggestions(self, metrics: StyleMetrics) -> List[Suggestion]:
        """Generate improvement suggestions from metrics.

        Args:
            metrics: Calculated style metrics

        Returns:
            List of suggestions
        """
        suggestions: List[Suggestion] = []

        # Check readability
        suggestions.extend(self._check_readability(metrics))

        # Check vocabulary
        suggestions.extend(self._check_vocabulary(metrics))

        # Check sentence structure
        suggestions.extend(self._check_sentences(metrics))

        # Check complexity
        suggestions.extend(self._check_complexity(metrics))

        return suggestions

    def _check_readability(self, metrics: StyleMetrics) -> List[Suggestion]:
        """Check readability and generate suggestions.

        Args:
            metrics: Style metrics

        Returns:
            List of readability suggestions
        """
        suggestions: List[Suggestion] = []
        r = metrics.readability

        if r.level == ReadabilityLevel.VERY_DIFFICULT:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.READABILITY,
                    severity=SuggestionSeverity.MAJOR,
                    message="Text is very difficult to read",
                    detail=f"Flesch Reading Ease score is {r.flesch_reading_ease:.1f}. "
                    "Consider using shorter sentences and simpler words.",
                    example="Break long sentences into two. Replace complex terms with simpler alternatives.",
                )
            )
        elif r.level == ReadabilityLevel.DIFFICULT:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.READABILITY,
                    severity=SuggestionSeverity.MODERATE,
                    message="Text may be difficult for general audience",
                    detail=f"Grade level {r.flesch_kincaid_grade:.1f} suggests college-level reading.",
                )
            )

        if r.gunning_fog > 12:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.COMPLEXITY,
                    severity=SuggestionSeverity.MINOR,
                    message="High Gunning Fog Index",
                    detail=f"Score of {r.gunning_fog:.1f} indicates complex writing. "
                    "Reduce multi-syllable words where possible.",
                )
            )

        return suggestions

    def _check_vocabulary(self, metrics: StyleMetrics) -> List[Suggestion]:
        """Check vocabulary diversity and generate suggestions.

        Args:
            metrics: Style metrics

        Returns:
            List of vocabulary suggestions
        """
        suggestions: List[Suggestion] = []
        v = metrics.vocabulary

        # Low vocabulary diversity
        if v.type_token_ratio < 0.3 and v.total_words > 100:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.VOCABULARY,
                    severity=SuggestionSeverity.MINOR,
                    message="Consider using more varied vocabulary",
                    detail=f"Type-token ratio of {v.type_token_ratio:.2%} suggests word repetition.",
                    example="Use synonyms to add variety. 'said' → 'stated', 'mentioned', 'explained'",
                )
            )

        # High complexity
        if v.complex_word_ratio > 0.25:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.VOCABULARY,
                    severity=SuggestionSeverity.MODERATE,
                    message="High proportion of complex words",
                    detail=f"{v.complex_word_ratio:.1%} of words have 3+ syllables.",
                    example="Replace 'approximately' with 'about', 'utilize' with 'use'.",
                )
            )

        # Very long average word length
        if v.average_word_length > 6:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.VOCABULARY,
                    severity=SuggestionSeverity.MINOR,
                    message="Average word length is high",
                    detail=f"Average of {v.average_word_length:.1f} characters per word.",
                )
            )

        return suggestions

    def _check_sentences(self, metrics: StyleMetrics) -> List[Suggestion]:
        """Check sentence structure and generate suggestions.

        Args:
            metrics: Style metrics

        Returns:
            List of sentence structure suggestions
        """
        suggestions: List[Suggestion] = []
        s = metrics.sentences

        # Very long sentences
        if s.average_sentence_length > 25:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.SENTENCE_STRUCTURE,
                    severity=SuggestionSeverity.MODERATE,
                    message="Sentences are quite long",
                    detail=f"Average of {s.average_sentence_length:.1f} words per sentence. "
                    "Aim for 15-20 words on average.",
                    example="Break compound sentences with 'and' into two separate sentences.",
                )
            )

        # Very short sentences
        if s.average_sentence_length < 8 and s.total_sentences > 5:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.SENTENCE_STRUCTURE,
                    severity=SuggestionSeverity.MINOR,
                    message="Sentences are quite short",
                    detail=f"Average of {s.average_sentence_length:.1f} words. "
                    "Consider combining related ideas.",
                )
            )

        # High variance (inconsistent)
        if s.sentence_length_variance > 150:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.CONSISTENCY,
                    severity=SuggestionSeverity.MINOR,
                    message="Sentence lengths vary significantly",
                    detail="Mix of very short and very long sentences affects flow.",
                )
            )

        # Extremely long max sentence
        if s.max_sentence_length > 50:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.SENTENCE_STRUCTURE,
                    severity=SuggestionSeverity.MODERATE,
                    message=f"Found sentence with {s.max_sentence_length} words",
                    detail="Consider breaking this sentence into smaller parts.",
                )
            )

        return suggestions

    def _check_complexity(self, metrics: StyleMetrics) -> List[Suggestion]:
        """Check overall complexity and generate suggestions.

        Args:
            metrics: Style metrics

        Returns:
            List of complexity suggestions
        """
        suggestions: List[Suggestion] = []

        # Check paragraph density
        if metrics.paragraph_count == 1 and metrics.sentences.total_sentences > 10:
            suggestions.append(
                Suggestion(
                    type=SuggestionType.CONSISTENCY,
                    severity=SuggestionSeverity.MINOR,
                    message="Consider adding paragraph breaks",
                    detail="Single paragraph with many sentences can be hard to read.",
                )
            )

        return suggestions

    def _compare_to_corpus(self, metrics: StyleMetrics) -> StyleComparison:
        """Compare text metrics to corpus baseline.

        Args:
            metrics: Text metrics

        Returns:
            StyleComparison
        """
        if not self._corpus_metrics:
            return StyleComparison()

        corpus = self._corpus_metrics

        # Calculate differences
        readability_diff = (
            metrics.readability.flesch_reading_ease
            - corpus.readability.flesch_reading_ease
        )
        vocabulary_diff = (
            metrics.vocabulary.type_token_ratio - corpus.vocabulary.type_token_ratio
        )
        sentence_diff = (
            metrics.sentences.average_sentence_length
            - corpus.sentences.average_sentence_length
        )
        complexity_diff = (
            metrics.vocabulary.complex_word_ratio - corpus.vocabulary.complex_word_ratio
        )

        # Check if style matches corpus (within tolerance)
        matches = (
            abs(readability_diff) < 15
            and abs(vocabulary_diff) < 0.1
            and abs(sentence_diff) < 5
        )

        return StyleComparison(
            readability_diff=round(readability_diff, 2),
            vocabulary_diff=round(vocabulary_diff, 4),
            sentence_length_diff=round(sentence_diff, 2),
            complexity_diff=round(complexity_diff, 4),
            matches_corpus=matches,
        )

    def _calculate_score(
        self, metrics: StyleMetrics, comparison: Optional[StyleComparison]
    ) -> float:
        """Calculate overall style score.

        Args:
            metrics: Text metrics
            comparison: Corpus comparison if available

        Returns:
            Score from 0-100
        """
        score = 50.0  # Base score

        # Readability bonus/penalty
        r = metrics.readability
        if r.level in (ReadabilityLevel.STANDARD, ReadabilityLevel.FAIRLY_EASY):
            score += 20
        elif r.level == ReadabilityLevel.EASY:
            score += 15
        elif r.level in (ReadabilityLevel.DIFFICULT, ReadabilityLevel.VERY_DIFFICULT):
            score -= 15

        # Vocabulary diversity bonus
        v = metrics.vocabulary
        if 0.4 <= v.type_token_ratio <= 0.7:
            score += 10
        elif v.type_token_ratio < 0.3:
            score -= 5

        # Sentence length bonus
        s = metrics.sentences
        if 12 <= s.average_sentence_length <= 20:
            score += 10
        elif s.average_sentence_length > 30:
            score -= 10

        # Corpus match bonus
        if comparison and comparison.matches_corpus:
            score += 10

        return max(0, min(100, score))

    def _average_metrics(self, metrics_list: List[StyleMetrics]) -> StyleMetrics:
        """Average multiple StyleMetrics into one.

        Args:
            metrics_list: List of metrics to average

        Returns:
            Averaged StyleMetrics
        """
        if not metrics_list:
            return StyleMetrics()

        n = len(metrics_list)

        # Average readability
        avg_fre = sum(m.readability.flesch_reading_ease for m in metrics_list) / n
        avg_fkg = sum(m.readability.flesch_kincaid_grade for m in metrics_list) / n

        # Average vocabulary
        avg_ttr = sum(m.vocabulary.type_token_ratio for m in metrics_list) / n
        avg_complex = sum(m.vocabulary.complex_word_ratio for m in metrics_list) / n

        # Average sentences
        avg_sent_len = (
            sum(m.sentences.average_sentence_length for m in metrics_list) / n
        )

        from ingestforge.analysis.metrics import (
            ReadabilityScores,
            VocabularyMetrics,
            SentenceMetrics,
        )

        return StyleMetrics(
            readability=ReadabilityScores(
                flesch_reading_ease=avg_fre,
                flesch_kincaid_grade=avg_fkg,
            ),
            vocabulary=VocabularyMetrics(
                type_token_ratio=avg_ttr,
                complex_word_ratio=avg_complex,
            ),
            sentences=SentenceMetrics(
                average_sentence_length=avg_sent_len,
            ),
        )


def critique_text(text: str) -> CritiqueResult:
    """Convenience function to critique text.

    Args:
        text: Text to critique

    Returns:
        CritiqueResult with suggestions
    """
    critic = StyleCritic()
    return critic.critique(text)


def compare_styles(text: str, corpus_texts: List[str]) -> CritiqueResult:
    """Compare text style against corpus samples.

    Args:
        text: Text to analyze
        corpus_texts: Corpus samples for comparison

    Returns:
        CritiqueResult with comparison
    """
    critic = StyleCritic()
    critic.set_corpus_baseline(corpus_texts)
    return critic.critique(text)
