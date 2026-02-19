"""Linguistic Metric Collector for writing style analysis.

Calculates readability and complexity metrics including Flesch-Kincaid,
Type-Token Ratio, sentence complexity, and vocabulary diversity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_TEXT_LENGTH = 100000
MAX_WORDS_TO_ANALYZE = 10000
MAX_SENTENCES_TO_ANALYZE = 1000


class ReadabilityLevel(str, Enum):
    """Readability classification levels."""

    VERY_EASY = "very_easy"  # 90-100 (5th grade)
    EASY = "easy"  # 80-89 (6th grade)
    FAIRLY_EASY = "fairly_easy"  # 70-79 (7th grade)
    STANDARD = "standard"  # 60-69 (8th-9th grade)
    FAIRLY_DIFFICULT = "fairly_difficult"  # 50-59 (10th-12th grade)
    DIFFICULT = "difficult"  # 30-49 (college)
    VERY_DIFFICULT = "very_difficult"  # 0-29 (college graduate)


@dataclass
class ReadabilityScores:
    """Collection of readability scores."""

    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    automated_readability: float = 0.0
    coleman_liau: float = 0.0
    level: ReadabilityLevel = ReadabilityLevel.STANDARD


@dataclass
class VocabularyMetrics:
    """Vocabulary diversity and complexity metrics."""

    type_token_ratio: float = 0.0
    unique_words: int = 0
    total_words: int = 0
    average_word_length: float = 0.0
    complex_word_ratio: float = 0.0
    rare_word_ratio: float = 0.0


@dataclass
class SentenceMetrics:
    """Sentence structure metrics."""

    total_sentences: int = 0
    average_sentence_length: float = 0.0
    max_sentence_length: int = 0
    min_sentence_length: int = 0
    sentence_length_variance: float = 0.0


@dataclass
class StyleMetrics:
    """Complete style metrics for a text."""

    readability: ReadabilityScores = field(default_factory=ReadabilityScores)
    vocabulary: VocabularyMetrics = field(default_factory=VocabularyMetrics)
    sentences: SentenceMetrics = field(default_factory=SentenceMetrics)
    paragraph_count: int = 0
    character_count: int = 0
    syllable_count: int = 0


class MetricCollector:
    """Collects linguistic metrics from text.

    Calculates readability scores, vocabulary diversity,
    and sentence complexity metrics.
    """

    def __init__(self) -> None:
        """Initialize the metric collector."""
        self._common_words: set[str] = self._load_common_words()

    def analyze(self, text: str) -> StyleMetrics:
        """Analyze text and return complete style metrics.

        Args:
            text: Text to analyze

        Returns:
            StyleMetrics with all calculated metrics
        """
        if not text or not text.strip():
            return StyleMetrics()

        # Truncate if too long
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            logger.warning(f"Text truncated to {MAX_TEXT_LENGTH} characters")

        # Extract basic components
        words = self._tokenize_words(text)
        sentences = self._tokenize_sentences(text)
        paragraphs = self._count_paragraphs(text)

        # Calculate metrics
        vocabulary = self._calculate_vocabulary_metrics(words)
        sentence_metrics = self._calculate_sentence_metrics(sentences, words)
        syllable_count = self._count_syllables_total(words)
        readability = self._calculate_readability(
            words, sentences, syllable_count, len(text)
        )

        return StyleMetrics(
            readability=readability,
            vocabulary=vocabulary,
            sentences=sentence_metrics,
            paragraph_count=paragraphs,
            character_count=len(text),
            syllable_count=syllable_count,
        )

    def _tokenize_words(self, text: str) -> List[str]:
        """Extract words from text.

        Args:
            text: Input text

        Returns:
            List of words (lowercase)
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return words[:MAX_WORDS_TO_ANALYZE]

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Extract sentences from text.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        sentences = re.split(r"[.!?]+\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[:MAX_SENTENCES_TO_ANALYZE]

    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text.

        Args:
            text: Input text

        Returns:
            Number of paragraphs
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return len([p for p in paragraphs if p.strip()])

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word.

        Args:
            word: Word to count syllables for

        Returns:
            Syllable count
        """
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e"):
            count -= 1

        # Ensure at least 1 syllable
        return max(1, count)

    def _count_syllables_total(self, words: List[str]) -> int:
        """Count total syllables in word list.

        Args:
            words: List of words

        Returns:
            Total syllable count
        """
        return sum(self._count_syllables(w) for w in words)

    def _calculate_vocabulary_metrics(self, words: List[str]) -> VocabularyMetrics:
        """Calculate vocabulary diversity metrics.

        Args:
            words: List of words

        Returns:
            VocabularyMetrics
        """
        if not words:
            return VocabularyMetrics()

        unique = set(words)
        total = len(words)
        ttr = len(unique) / total if total > 0 else 0.0

        # Average word length
        avg_length = sum(len(w) for w in words) / total if total > 0 else 0.0

        # Complex words (3+ syllables)
        complex_count = sum(1 for w in words if self._count_syllables(w) >= 3)
        complex_ratio = complex_count / total if total > 0 else 0.0

        # Rare words (not in common words list)
        rare_count = sum(1 for w in words if w not in self._common_words)
        rare_ratio = rare_count / total if total > 0 else 0.0

        return VocabularyMetrics(
            type_token_ratio=ttr,
            unique_words=len(unique),
            total_words=total,
            average_word_length=avg_length,
            complex_word_ratio=complex_ratio,
            rare_word_ratio=rare_ratio,
        )

    def _calculate_sentence_metrics(
        self, sentences: List[str], words: List[str]
    ) -> SentenceMetrics:
        """Calculate sentence structure metrics.

        Args:
            sentences: List of sentences
            words: List of words (for average calculation)

        Returns:
            SentenceMetrics
        """
        if not sentences:
            return SentenceMetrics()

        total = len(sentences)
        avg = len(words) / total if total > 0 else 0.0

        # Calculate per-sentence lengths
        lengths = [len(self._tokenize_words(s)) for s in sentences]

        max_len = max(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0

        # Variance
        if len(lengths) > 1:
            mean = sum(lengths) / len(lengths)
            variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        else:
            variance = 0.0

        return SentenceMetrics(
            total_sentences=total,
            average_sentence_length=avg,
            max_sentence_length=max_len,
            min_sentence_length=min_len,
            sentence_length_variance=variance,
        )

    def _calculate_readability(
        self,
        words: List[str],
        sentences: List[str],
        syllables: int,
        char_count: int,
    ) -> ReadabilityScores:
        """Calculate readability scores.

        Args:
            words: List of words
            sentences: List of sentences
            syllables: Total syllable count
            char_count: Character count

        Returns:
            ReadabilityScores
        """
        if not words or not sentences:
            return ReadabilityScores()

        word_count = len(words)
        sentence_count = len(sentences)

        # Flesch Reading Ease
        fre = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (syllables / word_count)
        )
        fre = max(0, min(100, fre))

        # Flesch-Kincaid Grade Level
        fkg = (
            0.39 * (word_count / sentence_count)
            + 11.8 * (syllables / word_count)
            - 15.59
        )
        fkg = max(0, fkg)

        # Gunning Fog Index
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        fog = 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))

        # SMOG Index (requires 30+ sentences for accuracy)
        smog = 1.0430 * ((complex_words * (30 / sentence_count)) ** 0.5) + 3.1291

        # Automated Readability Index
        ari = (
            4.71 * (char_count / word_count)
            + 0.5 * (word_count / sentence_count)
            - 21.43
        )
        ari = max(0, ari)

        # Coleman-Liau Index
        letters_per_100 = (char_count / word_count) * 100
        sentences_per_100 = (sentence_count / word_count) * 100
        cli = 0.0588 * letters_per_100 - 0.296 * sentences_per_100 - 15.8
        cli = max(0, cli)

        # Determine level from Flesch Reading Ease
        level = self._get_readability_level(fre)

        return ReadabilityScores(
            flesch_reading_ease=round(fre, 2),
            flesch_kincaid_grade=round(fkg, 2),
            gunning_fog=round(fog, 2),
            smog_index=round(smog, 2),
            automated_readability=round(ari, 2),
            coleman_liau=round(cli, 2),
            level=level,
        )

    def _get_readability_level(self, score: float) -> ReadabilityLevel:
        """Map Flesch Reading Ease score to level.

        Args:
            score: Flesch Reading Ease score

        Returns:
            ReadabilityLevel
        """
        if score >= 90:
            return ReadabilityLevel.VERY_EASY
        if score >= 80:
            return ReadabilityLevel.EASY
        if score >= 70:
            return ReadabilityLevel.FAIRLY_EASY
        if score >= 60:
            return ReadabilityLevel.STANDARD
        if score >= 50:
            return ReadabilityLevel.FAIRLY_DIFFICULT
        if score >= 30:
            return ReadabilityLevel.DIFFICULT
        return ReadabilityLevel.VERY_DIFFICULT

    def _load_common_words(self) -> set[str]:
        """Load common English words for rare word detection.

        Returns:
            Set of common words
        """
        # Top 100 most common English words
        return {
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "i",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
            "this",
            "but",
            "his",
            "by",
            "from",
            "they",
            "we",
            "say",
            "her",
            "she",
            "or",
            "an",
            "will",
            "my",
            "one",
            "all",
            "would",
            "there",
            "their",
            "what",
            "so",
            "up",
            "out",
            "if",
            "about",
            "who",
            "get",
            "which",
            "go",
            "me",
            "when",
            "make",
            "can",
            "like",
            "time",
            "no",
            "just",
            "him",
            "know",
            "take",
            "people",
            "into",
            "year",
            "your",
            "good",
            "some",
            "could",
            "them",
            "see",
            "other",
            "than",
            "then",
            "now",
            "look",
            "only",
            "come",
            "its",
            "over",
            "think",
            "also",
            "back",
            "after",
            "use",
            "two",
            "how",
            "our",
            "work",
            "first",
            "well",
            "way",
            "even",
            "new",
            "want",
            "because",
            "any",
            "these",
            "give",
            "day",
            "most",
            "us",
        }


def analyze_text(text: str) -> StyleMetrics:
    """Convenience function to analyze text.

    Args:
        text: Text to analyze

    Returns:
        StyleMetrics with all calculated metrics
    """
    collector = MetricCollector()
    return collector.analyze(text)


def get_readability_summary(metrics: StyleMetrics) -> Dict[str, str]:
    """Get human-readable summary of readability.

    Args:
        metrics: Calculated style metrics

    Returns:
        Dictionary of metric names to descriptions
    """
    r = metrics.readability
    return {
        "reading_level": r.level.value.replace("_", " ").title(),
        "flesch_ease": f"{r.flesch_reading_ease:.1f}/100",
        "grade_level": f"Grade {r.flesch_kincaid_grade:.1f}",
        "vocabulary_diversity": f"{metrics.vocabulary.type_token_ratio:.2%}",
        "avg_sentence_length": f"{metrics.sentences.average_sentence_length:.1f} words",
    }
