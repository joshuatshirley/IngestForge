"""Proximity-Based Ranking for Retrieval.

Ranks chunks by keyword proximity/density distance.
Chunks with query keywords appearing close together score higher.

NASA JPL Power of Ten Compliance:
- Rule #1: No recursion (linear iteration)
- Rule #2: Fixed upper bounds (MAX_* constants)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from ingestforge.core.logging import get_logger
from ingestforge.storage.base import SearchResult

logger = get_logger(__name__)

# =============================================================================
# JPL Rule #2: Fixed Upper Bounds
# =============================================================================

MAX_CANDIDATES = 100
MAX_QUERY_TERMS = 50
MAX_CONTENT_LENGTH = 100000
MAX_TERM_POSITIONS = 1000
MIN_TERM_LENGTH = 2
DEFAULT_PROXIMITY_WINDOW = 50  # Characters
MAX_PROXIMITY_WINDOW = 500


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ProximityScore:
    """Proximity score for a single chunk.

    Captures proximity metrics.
    Rule #9: Complete type hints.
    """

    chunk_id: str
    proximity_score: float  # 0.0 to 1.0, higher = terms closer
    term_coverage: float  # 0.0 to 1.0, fraction of query terms found
    min_span: int  # Minimum character span containing all terms
    avg_distance: float  # Average distance between term pairs
    cluster_count: int  # Number of term clusters found
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert 0.0 <= self.proximity_score <= 1.0, "proximity_score must be 0-1"
        assert 0.0 <= self.term_coverage <= 1.0, "term_coverage must be 0-1"
        assert self.min_span >= 0, "min_span cannot be negative"
        assert self.avg_distance >= 0, "avg_distance cannot be negative"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "proximity_score": self.proximity_score,
            "term_coverage": self.term_coverage,
            "min_span": self.min_span,
            "avg_distance": self.avg_distance,
            "cluster_count": self.cluster_count,
        }


@dataclass
class ProximityReport:
    """Report from proximity ranking operation.

    Rule #9: Complete type hints.
    """

    query: str
    input_count: int
    output_count: int
    avg_proximity_score: float
    time_ms: float = 0.0
    window_size: int = DEFAULT_PROXIMITY_WINDOW

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.input_count >= 0, "input_count cannot be negative"
        assert self.output_count >= 0, "output_count cannot be negative"
        assert self.time_ms >= 0, "time_ms cannot be negative"


# =============================================================================
# Proximity Ranker Class
# =============================================================================


class ProximityRanker:
    """Ranks chunks by keyword proximity/density.

    Proximity ranking stage.
    Chunks with query keywords appearing close together score higher.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_PROXIMITY_WINDOW,
        min_term_length: int = MIN_TERM_LENGTH,
    ) -> None:
        """Initialize proximity ranker.

        Args:
            window_size: Character window for proximity calculation.
            min_term_length: Minimum term length to consider.

        Rule #5: Assert preconditions.
        """
        assert window_size > 0, "window_size must be positive"
        assert (
            window_size <= MAX_PROXIMITY_WINDOW
        ), f"window_size max is {MAX_PROXIMITY_WINDOW}"
        assert min_term_length >= 1, "min_term_length must be >= 1"

        self._window_size = window_size
        self._min_term_length = min_term_length
        self._stopwords: Set[str] = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
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
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "as",
            "from",
        }

    @property
    def window_size(self) -> int:
        """Current proximity window size."""
        return self._window_size

    def rank(
        self,
        results: List[SearchResult],
        query: str,
        boost_weight: float = 0.3,
    ) -> Tuple[List[SearchResult], ProximityReport]:
        """Rank results by proximity and return boosted scores.

        Main proximity ranking method.
        Rule #2: Bounded by MAX_CANDIDATES.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            results: Search results to rank.
            query: Original query string.
            boost_weight: Weight of proximity boost (0.0 to 1.0).

        Returns:
            Tuple of (ranked results, report).
        """
        import time

        start_time = time.perf_counter()

        assert results is not None, "results cannot be None"
        assert query is not None, "query cannot be None"
        assert 0.0 <= boost_weight <= 1.0, "boost_weight must be 0-1"

        if not results or not query.strip():
            return results, self._empty_report(query, 0)

        # Extract query terms
        query_terms = self._extract_terms(query)
        if not query_terms:
            return results, self._empty_report(query, len(results))

        # Calculate proximity scores
        scored_results: List[Tuple[SearchResult, ProximityScore]] = []
        bounded_results = results[:MAX_CANDIDATES]

        for result in bounded_results:
            prox_score = self._calculate_proximity(result, query_terms)
            scored_results.append((result, prox_score))

        # Apply proximity boost to original scores
        boosted_results = self._apply_boost(scored_results, boost_weight)

        # Sort by boosted score
        boosted_results.sort(key=lambda r: r.score, reverse=True)

        # Calculate report metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        avg_score = sum(s.proximity_score for _, s in scored_results) / len(
            scored_results
        )

        report = ProximityReport(
            query=query,
            input_count=len(results),
            output_count=len(boosted_results),
            avg_proximity_score=avg_score,
            time_ms=elapsed_ms,
            window_size=self._window_size,
        )

        return boosted_results, report

    def score_chunk(
        self,
        content: str,
        query: str,
    ) -> ProximityScore:
        """Score a single chunk for proximity.

        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            content: Chunk content text.
            query: Query string.

        Returns:
            ProximityScore for the chunk.
        """
        assert content is not None, "content cannot be None"
        assert query is not None, "query cannot be None"

        query_terms = self._extract_terms(query)

        if not query_terms or not content.strip():
            return ProximityScore(
                chunk_id="",
                proximity_score=0.0,
                term_coverage=0.0,
                min_span=0,
                avg_distance=0.0,
                cluster_count=0,
            )

        return self._score_content(content, query_terms, "")

    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text.

        Rule #2: Bounded by MAX_QUERY_TERMS.
        Rule #4: Function < 60 lines.

        Args:
            text: Text to extract terms from.

        Returns:
            List of lowercase terms.
        """
        if not text:
            return []

        # Tokenize and filter
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        terms: List[str] = []
        for word in words:
            if len(word) >= self._min_term_length and word not in self._stopwords:
                terms.append(word)
                if len(terms) >= MAX_QUERY_TERMS:
                    break

        return terms

    def _calculate_proximity(
        self,
        result: SearchResult,
        query_terms: List[str],
    ) -> ProximityScore:
        """Calculate proximity score for a search result.

        Rule #4: Function < 60 lines.

        Args:
            result: Search result to score.
            query_terms: Query terms to find.

        Returns:
            ProximityScore for the result.
        """
        content = result.content[:MAX_CONTENT_LENGTH] if result.content else ""
        chunk_id = result.chunk_id if hasattr(result, "chunk_id") else ""

        return self._score_content(content, query_terms, chunk_id)

    def _score_content(
        self,
        content: str,
        query_terms: List[str],
        chunk_id: str,
    ) -> ProximityScore:
        """Score content for term proximity.

        Rule #4: Function < 60 lines.

        Args:
            content: Text content to score.
            query_terms: Terms to find.
            chunk_id: Chunk identifier.

        Returns:
            ProximityScore with metrics.
        """
        content_lower = content.lower()

        # Find all term positions
        term_positions: Dict[str, List[int]] = {}
        for term in query_terms:
            positions = self._find_positions(content_lower, term)
            if positions:
                term_positions[term] = positions

        # Calculate coverage
        found_terms = len(term_positions)
        total_terms = len(query_terms)
        coverage = found_terms / total_terms if total_terms > 0 else 0.0

        if found_terms == 0:
            return ProximityScore(
                chunk_id=chunk_id,
                proximity_score=0.0,
                term_coverage=0.0,
                min_span=0,
                avg_distance=0.0,
                cluster_count=0,
            )

        # Calculate proximity metrics
        min_span, avg_distance = self._calculate_span_metrics(term_positions)
        cluster_count = self._count_clusters(term_positions)

        # Compute final proximity score
        proximity_score = self._compute_proximity_score(
            coverage, min_span, avg_distance, cluster_count, len(content)
        )

        return ProximityScore(
            chunk_id=chunk_id,
            proximity_score=proximity_score,
            term_coverage=coverage,
            min_span=min_span,
            avg_distance=avg_distance,
            cluster_count=cluster_count,
        )

    def _find_positions(self, text: str, term: str) -> List[int]:
        """Find all positions of a term in text.

        Rule #2: Bounded by MAX_TERM_POSITIONS.
        Rule #4: Function < 60 lines.

        Args:
            text: Text to search.
            term: Term to find.

        Returns:
            List of character positions.
        """
        positions: List[int] = []
        start = 0

        while len(positions) < MAX_TERM_POSITIONS:
            idx = text.find(term, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1

        return positions

    def _calculate_span_metrics(
        self,
        term_positions: Dict[str, List[int]],
    ) -> Tuple[int, float]:
        """Calculate minimum span and average distance.

        Rule #4: Function < 60 lines.

        Args:
            term_positions: Map of term to positions.

        Returns:
            Tuple of (min_span, avg_distance).
        """
        if len(term_positions) < 2:
            return 0, 0.0

        # Get all positions flattened
        all_positions: List[Tuple[int, str]] = []
        for term, positions in term_positions.items():
            for pos in positions[:10]:  # Limit per term
                all_positions.append((pos, term))

        all_positions.sort(key=lambda x: x[0])

        if len(all_positions) < 2:
            return 0, 0.0

        # Find minimum span containing all terms
        min_span = float("inf")
        term_set = set(term_positions.keys())

        for i, (start_pos, _) in enumerate(all_positions):
            seen_terms: Set[str] = set()
            for j in range(i, len(all_positions)):
                pos, term = all_positions[j]
                seen_terms.add(term)
                if seen_terms == term_set:
                    span = pos - start_pos + len(term)
                    if span < min_span:
                        min_span = span
                    break

        if min_span == float("inf"):
            min_span = 0

        # Calculate average pairwise distance
        distances: List[int] = []
        for i in range(len(all_positions) - 1):
            dist = all_positions[i + 1][0] - all_positions[i][0]
            distances.append(dist)

        avg_distance = sum(distances) / len(distances) if distances else 0.0

        return int(min_span), avg_distance

    def _count_clusters(
        self,
        term_positions: Dict[str, List[int]],
    ) -> int:
        """Count term clusters within proximity window.

        Rule #4: Function < 60 lines.

        Args:
            term_positions: Map of term to positions.

        Returns:
            Number of clusters found.
        """
        if not term_positions:
            return 0

        # Flatten and sort positions
        all_positions = sorted(
            pos for positions in term_positions.values() for pos in positions[:10]
        )

        if len(all_positions) < 2:
            return 1 if all_positions else 0

        # Count clusters using window
        clusters = 1
        cluster_start = all_positions[0]

        for pos in all_positions[1:]:
            if pos - cluster_start > self._window_size:
                clusters += 1
                cluster_start = pos

        return clusters

    def _compute_proximity_score(
        self,
        coverage: float,
        min_span: int,
        avg_distance: float,
        cluster_count: int,
        content_length: int,
    ) -> float:
        """Compute final proximity score from metrics.

        Rule #4: Function < 60 lines.

        Args:
            coverage: Term coverage (0-1).
            min_span: Minimum span containing all terms.
            avg_distance: Average distance between terms.
            cluster_count: Number of term clusters.
            content_length: Length of content.

        Returns:
            Proximity score (0-1).
        """
        if content_length == 0 or coverage == 0:
            return 0.0

        # Score components
        coverage_score = coverage  # Higher coverage = better

        # Span score: smaller span relative to content = better
        span_ratio = min_span / content_length if min_span > 0 else 1.0
        span_score = max(0.0, 1.0 - span_ratio)

        # Distance score: smaller avg distance = better
        normalized_dist = avg_distance / self._window_size
        distance_score = max(0.0, 1.0 - min(normalized_dist, 1.0))

        # Cluster bonus: more clusters = terms appear in multiple places
        cluster_bonus = min(cluster_count / 5.0, 1.0) * 0.2

        # Weighted combination
        score = (
            coverage_score * 0.4
            + span_score * 0.3
            + distance_score * 0.2
            + cluster_bonus
        )

        return min(max(score, 0.0), 1.0)

    def _apply_boost(
        self,
        scored_results: List[Tuple[SearchResult, ProximityScore]],
        boost_weight: float,
    ) -> List[SearchResult]:
        """Apply proximity boost to result scores.

        Rule #4: Function < 60 lines.

        Args:
            scored_results: Results with proximity scores.
            boost_weight: Weight of proximity boost.

        Returns:
            Results with boosted scores.
        """
        boosted: List[SearchResult] = []

        for result, prox_score in scored_results:
            # Combine original score with proximity boost
            original_weight = 1.0 - boost_weight
            new_score = (
                result.score * original_weight
                + prox_score.proximity_score * boost_weight
            )

            # Create new result with boosted score
            boosted_result = SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                score=new_score,
                section_title=getattr(result, "section_title", ""),
                chunk_type=getattr(result, "chunk_type", "text"),
                source_file=getattr(result, "source_file", ""),
                word_count=getattr(result, "word_count", 0),
                metadata={
                    **(result.metadata or {}),
                    "proximity_score": prox_score.proximity_score,
                    "term_coverage": prox_score.term_coverage,
                    "original_score": result.score,
                },
                source_location=getattr(result, "source_location", None),
                page_start=getattr(result, "page_start", None),
                page_end=getattr(result, "page_end", None),
            )
            boosted.append(boosted_result)

        return boosted

    def _empty_report(self, query: str, count: int) -> ProximityReport:
        """Create empty report for edge cases."""
        return ProximityReport(
            query=query,
            input_count=count,
            output_count=count,
            avg_proximity_score=0.0,
            time_ms=0.0,
            window_size=self._window_size,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_proximity_ranker(
    window_size: int = DEFAULT_PROXIMITY_WINDOW,
) -> ProximityRanker:
    """Factory function to create proximity ranker.

    Args:
        window_size: Character window for proximity.

    Returns:
        Configured ProximityRanker.
    """
    return ProximityRanker(window_size=window_size)


def rank_by_proximity(
    results: List[SearchResult],
    query: str,
    boost_weight: float = 0.3,
) -> List[SearchResult]:
    """Convenience function to rank results by proximity.

    Args:
        results: Search results to rank.
        query: Query string.
        boost_weight: Weight of proximity boost.

    Returns:
        Results ranked by proximity-boosted scores.
    """
    ranker = ProximityRanker()
    ranked, _ = ranker.rank(results, query, boost_weight)
    return ranked
