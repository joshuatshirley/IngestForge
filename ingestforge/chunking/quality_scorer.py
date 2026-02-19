"""
Chunk quality scoring.

Evaluate chunk quality based on various metrics.
"""

import re
from dataclasses import dataclass
from typing import List

from ingestforge.chunking.semantic_chunker import ChunkRecord


@dataclass
class QualityMetrics:
    """Quality metrics for a chunk."""

    coherence_score: float  # Text coherence 0-1
    completeness_score: float  # Sentence completeness 0-1
    information_density: float  # Information per word 0-1
    readability_score: float  # Flesch-Kincaid simplified 0-1
    overall_score: float  # Weighted average


class QualityScorer:
    """
    Score chunk quality based on multiple factors.

    Metrics:
    - Coherence: Proper sentence structure
    - Completeness: Complete sentences, not fragments
    - Information density: Ratio of content words
    - Readability: Flesch-Kincaid inspired
    """

    def __init__(
        self,
        coherence_weight: float = 0.3,
        completeness_weight: float = 0.3,
        density_weight: float = 0.2,
        readability_weight: float = 0.2,
    ):
        self.coherence_weight = coherence_weight
        self.completeness_weight = completeness_weight
        self.density_weight = density_weight
        self.readability_weight = readability_weight

        # Common stop words for density calculation
        self.stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "and",
            "or",
            "but",
            "if",
            "than",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "we",
            "us",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "which",
            "what",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
            "not",
            "no",
        }

    def score(self, chunk: ChunkRecord) -> QualityMetrics:
        """
        Score a single chunk.

        Args:
            chunk: Chunk to score

        Returns:
            QualityMetrics with individual and overall scores
        """
        content = chunk.content

        coherence = self._score_coherence(content)
        completeness = self._score_completeness(content)
        density = self._score_information_density(content)
        readability = self._score_readability(content)

        overall = (
            self.coherence_weight * coherence
            + self.completeness_weight * completeness
            + self.density_weight * density
            + self.readability_weight * readability
        )

        return QualityMetrics(
            coherence_score=coherence,
            completeness_score=completeness,
            information_density=density,
            readability_score=readability,
            overall_score=overall,
        )

    def score_batch(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """
        Score a batch of chunks and update their quality_score field.

        Args:
            chunks: Chunks to score

        Returns:
            Same chunks with quality_score updated
        """
        for chunk in chunks:
            metrics = self.score(chunk)
            chunk.quality_score = metrics.overall_score
        return chunks

    def _score_coherence(self, text: str) -> float:
        """Score text coherence based on structure."""
        if not text.strip():
            return 0.0

        score = 1.0
        sentences = self._split_sentences(text)

        if not sentences:
            return 0.0

        # Check for proper capitalization
        caps_correct = sum(1 for s in sentences if s and s[0].isupper())
        score *= caps_correct / len(sentences)

        # Check for proper punctuation
        punct_correct = sum(1 for s in sentences if s and s[-1] in ".!?")
        score *= punct_correct / len(sentences)

        # Check for transition words (indicates connected thoughts)
        transition_words = {
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "additionally",
            "consequently",
            "nevertheless",
            "meanwhile",
            "specifically",
            "for example",
            "in addition",
            "as a result",
            "on the other hand",
        }
        text_lower = text.lower()
        has_transitions = any(tw in text_lower for tw in transition_words)
        if has_transitions:
            score = min(1.0, score * 1.1)

        return min(1.0, max(0.0, score))

    def _score_completeness(self, text: str) -> float:
        """Score sentence completeness."""
        if not text.strip():
            return 0.0

        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        complete_count = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check for subject-verb structure (simplified)
            words = sentence.split()
            if len(words) >= 3:  # Minimum for complete sentence
                complete_count += 1
            elif len(words) >= 1 and sentence[-1] in ".!?":
                # Short but punctuated
                complete_count += 0.5

        return min(1.0, complete_count / len(sentences))

    def _score_information_density(self, text: str) -> float:
        """Score information density (content words ratio)."""
        if not text.strip():
            return 0.0

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        content_words = [w for w in words if w not in self.stop_words]
        density = len(content_words) / len(words)

        # Normalize: typical range is 0.4-0.7 for good content
        # Map 0.4-0.7 to 0.5-1.0
        if density < 0.4:
            return density / 0.4 * 0.5
        elif density > 0.7:
            return 1.0
        else:
            return 0.5 + (density - 0.4) / 0.3 * 0.5

    def _score_readability(self, text: str) -> float:
        """
        Score readability (simplified Flesch-Kincaid).

        Higher score = more readable.
        """
        if not text.strip():
            return 0.0

        sentences = self._split_sentences(text)
        words = re.findall(r"\b\w+\b", text)

        if not sentences or not words:
            return 0.0

        # Average words per sentence
        avg_words = len(words) / len(sentences)

        # Average syllables per word (simplified)
        syllable_count = sum(self._count_syllables(w) for w in words)
        avg_syllables = syllable_count / len(words)

        # Simplified Flesch Reading Ease
        # Original: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        # Normalized to 0-1 (typical scores range from 0-100)
        score = 206.835 - 1.015 * avg_words - 84.6 * avg_syllables
        normalized = max(0, min(100, score)) / 100

        return normalized

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle abbreviations
        text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Jr|Sr|vs)\.", r"\1<DOT>", text)
        text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.replace("<DOT>", ".") for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
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
        if word.endswith("e") and count > 1:
            count -= 1

        return max(1, count)
