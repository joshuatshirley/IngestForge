"""Score Normalization for Multi-Provider Search Results.

RRF Normalization Pass
Epic: Retrieval Enhancement

Provides score normalization across heterogeneous search providers
(BM25, semantic, external APIs) to enable fair fusion.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_RESULTS)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_RESULTS = 10_000
MIN_SCORE = 0.0
MAX_SCORE = 1.0
DEFAULT_RRF_K = 60


class NormalizationMethod(Enum):
    """Score normalization methods."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    RRF = "rrf"


@dataclass
class NormalizedScore:
    """Normalized score with provenance.

    Rule #9: Complete type hints.
    """

    item_id: str
    raw_score: float
    normalized_score: float
    provider: str
    method: NormalizationMethod


class ScoreNormalizer:
    """Normalize scores across search providers.

    Rule #9: Complete type hints.
    """

    def __init__(self, method: NormalizationMethod = NormalizationMethod.MIN_MAX):
        """Initialize normalizer.

        Args:
            method: Normalization method to use.
        """
        assert method is not None, "method cannot be None"
        self.method = method

    def normalize_min_max(
        self,
        scores: Dict[str, float],
        provider: str = "unknown",
    ) -> List[NormalizedScore]:
        """Normalize scores to [0,1] using min-max scaling.

        Args:
            scores: Dictionary of item_id -> raw_score.
            provider: Name of the score provider.

        Returns:
            List of NormalizedScore objects.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        assert scores is not None, "scores cannot be None"

        if not scores:
            return []

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        score_range = max_val - min_val

        results: List[NormalizedScore] = []

        for item_id, raw_score in scores.items():
            if score_range > 0:
                normalized = (raw_score - min_val) / score_range
            else:
                normalized = 1.0 if raw_score > 0 else 0.0

            normalized = max(MIN_SCORE, min(MAX_SCORE, normalized))

            results.append(
                NormalizedScore(
                    item_id=item_id,
                    raw_score=raw_score,
                    normalized_score=normalized,
                    provider=provider,
                    method=NormalizationMethod.MIN_MAX,
                )
            )

        return results

    def normalize_rrf(
        self,
        ranked_lists: List[List[str]],
        k: int = DEFAULT_RRF_K,
        provider: str = "fused",
    ) -> List[NormalizedScore]:
        """Compute RRF scores from multiple ranked lists.

        Args:
            ranked_lists: List of ranked item ID lists.
            k: RRF constant (default 60).
            provider: Name for the fused provider.

        Returns:
            List of NormalizedScore sorted by RRF score.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        assert ranked_lists is not None, "ranked_lists cannot be None"
        assert k > 0, "k must be positive"

        rrf_scores: Dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, item_id in enumerate(ranked_list[:MAX_RESULTS], 1):
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank)

        results: List[NormalizedScore] = []

        for item_id, rrf_score in rrf_scores.items():
            results.append(
                NormalizedScore(
                    item_id=item_id,
                    raw_score=rrf_score,
                    normalized_score=rrf_score,
                    provider=provider,
                    method=NormalizationMethod.RRF,
                )
            )

        results.sort(key=lambda x: -x.normalized_score)
        return results

    def normalize_z_score(
        self,
        scores: Dict[str, float],
        provider: str = "unknown",
    ) -> List[NormalizedScore]:
        """Normalize scores using z-score standardization.

        Args:
            scores: Dictionary of item_id -> raw_score.
            provider: Name of the score provider.

        Returns:
            List of NormalizedScore objects.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        assert scores is not None, "scores cannot be None"

        if not scores:
            return []

        values = list(scores.values())
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = variance**0.5

        results: List[NormalizedScore] = []

        for item_id, raw_score in scores.items():
            if std_dev > 0:
                z_score = (raw_score - mean_val) / std_dev
                # Sigmoid function to map z-score to [0,1]
                import math

                normalized = 1.0 / (1.0 + math.exp(-z_score))
            else:
                normalized = 0.5

            normalized = max(MIN_SCORE, min(MAX_SCORE, normalized))

            results.append(
                NormalizedScore(
                    item_id=item_id,
                    raw_score=raw_score,
                    normalized_score=normalized,
                    provider=provider,
                    method=NormalizationMethod.Z_SCORE,
                )
            )

        return results

    def normalize(
        self,
        scores: Dict[str, float],
        provider: str = "unknown",
    ) -> List[NormalizedScore]:
        """Normalize scores using configured method.

        Args:
            scores: Dictionary of item_id -> raw_score.
            provider: Name of the score provider.

        Returns:
            List of NormalizedScore objects.

        Rule #4: Function < 60 lines.
        """
        if self.method == NormalizationMethod.MIN_MAX:
            return self.normalize_min_max(scores, provider)
        elif self.method == NormalizationMethod.Z_SCORE:
            return self.normalize_z_score(scores, provider)
        else:
            raise ValueError(f"Unsupported method for dict input: {self.method}")

    def fuse_providers(
        self,
        provider_scores: Dict[str, Dict[str, float]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Fuse normalized scores from multiple providers.

        Args:
            provider_scores: Dict of provider_name -> {item_id: score}.
            weights: Optional weights per provider (default: equal).

        Returns:
            Fused scores dict {item_id: fused_score}.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        assert provider_scores is not None, "provider_scores cannot be None"

        if not provider_scores:
            return {}

        if weights is None:
            weights = {p: 1.0 / len(provider_scores) for p in provider_scores}

        normalized_by_provider: Dict[str, Dict[str, float]] = {}

        for provider, scores in provider_scores.items():
            normalized = self.normalize_min_max(scores, provider)
            normalized_by_provider[provider] = {
                ns.item_id: ns.normalized_score for ns in normalized
            }

        all_items = set()
        for norm_scores in normalized_by_provider.values():
            all_items.update(norm_scores.keys())

        fused: Dict[str, float] = {}

        for item_id in all_items:
            weighted_sum = 0.0
            for provider, norm_scores in normalized_by_provider.items():
                score = norm_scores.get(item_id, 0.0)
                weight = weights.get(provider, 0.0)
                weighted_sum += score * weight
            fused[item_id] = weighted_sum

        return fused
