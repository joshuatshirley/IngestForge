"""Semantic Window Comparator for plagiarism detection.

Slides a window across text to find similar passages in corpus
using high-sensitivity semantic comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_WINDOW_SIZE = 500
MIN_WINDOW_SIZE = 20
MAX_CORPUS_CHUNKS = 10000
MAX_MATCHES_PER_WINDOW = 10
DEFAULT_WINDOW_SIZE = 100
DEFAULT_STRIDE = 50
SIMILARITY_THRESHOLD = 0.75


@dataclass
class TextWindow:
    """A window of text for comparison."""

    text: str
    start_pos: int
    end_pos: int
    window_index: int

    @property
    def length(self) -> int:
        """Get window length in characters."""
        return self.end_pos - self.start_pos


@dataclass
class SimilarityMatch:
    """A match between source window and corpus chunk."""

    source_window: TextWindow
    corpus_text: str
    corpus_source: str
    similarity_score: float
    matched_words: int = 0

    @property
    def is_high_similarity(self) -> bool:
        """Check if similarity is concerning."""
        return self.similarity_score >= 0.85


@dataclass
class WindowConfig:
    """Configuration for window comparison."""

    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    similarity_threshold: float = SIMILARITY_THRESHOLD
    min_match_words: int = 5

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        self.window_size = max(MIN_WINDOW_SIZE, min(self.window_size, MAX_WINDOW_SIZE))
        self.stride = max(1, min(self.stride, self.window_size))


@dataclass
class ComparisonResult:
    """Result of window-based similarity comparison."""

    total_windows: int = 0
    windows_with_matches: int = 0
    matches: List[SimilarityMatch] = field(default_factory=list)
    highest_similarity: float = 0.0
    flagged_count: int = 0

    @property
    def overlap_ratio(self) -> float:
        """Calculate ratio of windows with matches."""
        if self.total_windows == 0:
            return 0.0
        return self.windows_with_matches / self.total_windows


# Type alias for similarity function
SimilarityFunc = Callable[[str, str], float]


class SemanticWindowComparator:
    """Compares text windows against corpus using semantic similarity.

    Slides a window across the source text and compares each window
    against corpus chunks to detect potential plagiarism.
    """

    def __init__(
        self,
        config: Optional[WindowConfig] = None,
        similarity_func: Optional[SimilarityFunc] = None,
    ) -> None:
        """Initialize comparator.

        Args:
            config: Window configuration
            similarity_func: Custom similarity function (text1, text2) -> float
        """
        self.config = config or WindowConfig()
        self._similarity_func = similarity_func or self._default_similarity
        self._corpus: List[Tuple[str, str]] = []  # (text, source)

    def set_corpus(self, chunks: List[Tuple[str, str]]) -> None:
        """Set the corpus to compare against.

        Args:
            chunks: List of (text, source) tuples
        """
        self._corpus = chunks[:MAX_CORPUS_CHUNKS]
        logger.info(f"Corpus set with {len(self._corpus)} chunks")

    def compare(self, text: str) -> ComparisonResult:
        """Compare text against corpus using sliding windows.

        Args:
            text: Source text to check

        Returns:
            ComparisonResult with matches
        """
        if not text or not text.strip():
            return ComparisonResult()

        if not self._corpus:
            logger.warning("No corpus set for comparison")
            return ComparisonResult()

        # Generate windows
        windows = list(self._generate_windows(text))
        result = ComparisonResult(total_windows=len(windows))

        # Compare each window (Rule #3: Memory efficient)
        for window in windows:
            matches = self._find_matches(window)
            if matches:
                result.windows_with_matches += 1
                result.matches.extend(matches[:MAX_MATCHES_PER_WINDOW])
                self._update_result_metrics(result, matches)

        return result

    def _update_result_metrics(
        self,
        result: ComparisonResult,
        matches: List[SimilarityMatch],
    ) -> None:
        """Update result metrics with match data.

        Args:
            result: ComparisonResult to update
            matches: List of matches to process
        """
        for match in matches:
            if match.similarity_score > result.highest_similarity:
                result.highest_similarity = match.similarity_score
            if match.is_high_similarity:
                result.flagged_count += 1

    def _generate_windows(self, text: str) -> Iterator[TextWindow]:
        """Generate sliding windows from text.

        Args:
            text: Source text

        Yields:
            TextWindow objects
        """
        # Split into words for window generation
        words = text.split()
        total_words = len(words)

        window_size = self.config.window_size
        stride = self.config.stride
        window_index = 0

        for start in range(0, total_words, stride):
            end = min(start + window_size, total_words)
            window_words = words[start:end]

            if len(window_words) < self.config.min_match_words:
                continue

            window_text = " ".join(window_words)

            # Calculate character positions
            char_start = sum(len(w) + 1 for w in words[:start])
            char_end = char_start + len(window_text)

            yield TextWindow(
                text=window_text,
                start_pos=char_start,
                end_pos=char_end,
                window_index=window_index,
            )
            window_index += 1

            # Stop if we've reached the end
            if end >= total_words:
                break

    def _find_matches(self, window: TextWindow) -> List[SimilarityMatch]:
        """Find corpus matches for a window.

        Args:
            window: Text window to check

        Returns:
            List of matches above threshold
        """
        matches: List[SimilarityMatch] = []

        for corpus_text, corpus_source in self._corpus:
            similarity = self._similarity_func(window.text, corpus_text)

            if similarity >= self.config.similarity_threshold:
                # Count matching words
                window_words = set(window.text.lower().split())
                corpus_words = set(corpus_text.lower().split())
                matched = len(window_words & corpus_words)

                if matched >= self.config.min_match_words:
                    matches.append(
                        SimilarityMatch(
                            source_window=window,
                            corpus_text=corpus_text[:500],  # Truncate for memory
                            corpus_source=corpus_source,
                            similarity_score=similarity,
                            matched_words=matched,
                        )
                    )

        # Sort by similarity and limit
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:MAX_MATCHES_PER_WINDOW]

    def _default_similarity(self, text1: str, text2: str) -> float:
        """Default similarity using Jaccard index.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0-1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


def create_comparator(
    window_size: int = DEFAULT_WINDOW_SIZE,
    threshold: float = SIMILARITY_THRESHOLD,
) -> SemanticWindowComparator:
    """Factory function to create comparator.

    Args:
        window_size: Size of sliding window
        threshold: Similarity threshold

    Returns:
        Configured comparator
    """
    config = WindowConfig(
        window_size=window_size,
        similarity_threshold=threshold,
    )
    return SemanticWindowComparator(config=config)


def find_similar_passages(
    text: str,
    corpus: List[Tuple[str, str]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[SimilarityMatch]:
    """Convenience function to find similar passages.

    Args:
        text: Source text to check
        corpus: List of (text, source) tuples
        threshold: Similarity threshold

    Returns:
        List of matches
    """
    comparator = create_comparator(threshold=threshold)
    comparator.set_corpus(corpus)
    result = comparator.compare(text)
    return result.matches
