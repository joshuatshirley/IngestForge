"""
Semantic Example Matcher.

Semantic Example Matcher
Uses embeddings to find relevant few-shot examples for a given task.

JPL Compliance:
- Rule #2: Bounded candidate sets.
- Rule #4: All functions < 60 lines.
- Rule #9: 100% type hints.
"""

from typing import List, Optional
import numpy as np
from ingestforge.learning.models import FewShotExample
from ingestforge.learning.registry import FewShotRegistry
from ingestforge.enrichment.embeddings import EmbeddingGenerator
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Safety limits
MAX_CANDIDATES = 100
MIN_SIMILARITY_SCORE = 0.3


class SemanticExampleMatcher:
    """Matches input text to relevant examples in the registry using semantic similarity."""

    def __init__(self, registry: Optional[FewShotRegistry] = None):
        """Initialize with embedding generator and registry."""
        self.registry = registry or FewShotRegistry()
        self.embedder = EmbeddingGenerator()

    def find_matches(
        self, input_text: str, domain: Optional[str] = None, limit: int = 3
    ) -> List[FewShotExample]:
        """Find the most semantically similar examples for the input.

        JPL Rule #2: Bound candidate set and results.
        """
        # JPL Rule #2: Enforce strict safety limits on retrieval
        safe_limit = min(limit, 10)

        # 1. Fetch candidates from registry (filtered by domain)
        candidates = self.registry.list_examples(domain=domain, limit=MAX_CANDIDATES)
        if not candidates:
            return []

        # 2. Generate embeddings for input and candidates
        try:
            input_vector = self.embedder.generate([input_text])[0]
            candidate_texts = [c.input_text for c in candidates]
            candidate_vectors = self.embedder.generate(candidate_texts)

            # 3. Calculate cosine similarity
            scores = self._cosine_similarity(input_vector, candidate_vectors)

            # 4. Sort and return top matches
            return self._rank_candidates(candidates, scores, safe_limit)

        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return candidates[:safe_limit]  # Fallback to first N examples

    def _cosine_similarity(self, target: np.ndarray, others: np.ndarray) -> np.ndarray:
        """Calculate similarity scores.

        Rule #4: Concise math helper.
        """
        dot_product = np.dot(others, target)
        norms = np.linalg.norm(others, axis=1) * np.linalg.norm(target)
        # Avoid division by zero
        return dot_product / (norms + 1e-9)

    def _rank_candidates(
        self, candidates: List[FewShotExample], scores: np.ndarray, limit: int
    ) -> List[FewShotExample]:
        """Ranks and filters candidates based on similarity scores."""
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # Filter by minimum similarity and return top N
        return [item[0] for item in ranked[:limit] if item[1] >= MIN_SIMILARITY_SCORE]
