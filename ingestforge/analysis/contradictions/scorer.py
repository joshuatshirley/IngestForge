"""Contradiction Scorer logic.

Uses Natural Language Inference (NLI) to detect factual conflicts.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from enum import Enum
from typing import List, Tuple, Optional
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class RelationType(Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"
    ERROR = "error"


class ContradictionScorer:
    """Logic for evaluating logical relationships between claims."""

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    @property
    def model(self):
        """Lazy-load the NLI cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded NLI model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. NLI check disabled.")
                raise ImportError("Run: pip install sentence-transformers")
        return self._model

    def evaluate_pair(self, text_a: str, text_b: str) -> Tuple[RelationType, float]:
        """Classify the relationship between two strings.

        Returns:
            Tuple of (RelationType, confidence_score)
        """
        if not text_a or not text_b:
            return RelationType.ERROR, 0.0

        try:
            # NLI models return scores for [Entailment, Neutral, Contradiction]
            scores = self.model.predict([(text_a, text_b)])[0]

            # Find index of max score
            import numpy as np

            label_idx = np.argmax(scores)
            confidence = float(scores[label_idx])

            # Map index to enum
            mapping = {
                0: RelationType.ENTAILMENT,
                1: RelationType.NEUTRAL,
                2: RelationType.CONTRADICTION,
            }

            return mapping.get(label_idx, RelationType.ERROR), confidence

        except Exception as e:
            logger.error(f"NLI evaluation failed: {e}")
            return RelationType.ERROR, 0.0

    def find_conflicts(self, pairs: List[Tuple[str, str]]) -> List[dict]:
        """Evaluate multiple pairs and return only contradictions."""
        if not pairs:
            return []

        results = []
        for a, b in pairs:
            relation, conf = self.evaluate_pair(a, b)
            if relation == RelationType.CONTRADICTION:
                results.append({"text_a": a, "text_b": b, "confidence": conf})

        return results
