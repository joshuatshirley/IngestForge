"""Cross-Encoder Entailment Scorer.

Uses local models to score the relationship between a claim and evidence.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from typing import Optional, List, Tuple
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class EntailmentScorer:
    """Scorer for semantic entailment using Cross-Encoders."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded entailment model: {self.model_name}")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. Entailment check disabled."
                )
                raise ImportError("Run: pip install sentence-transformers")
        return self._model

    def score(self, claim: str, evidence: str) -> float:
        """Score the entailment between a claim and evidence.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not claim or not evidence:
            return 0.0

        try:
            # Cross-encoder returns a single score for the pair
            score = self.model.predict([(claim, evidence)])[0]

            # Normalize score (ms-marco returns logits, usually mapped to 0-1 via sigmoid if needed,
            # but for our purposes we treat higher as better and clamp).
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Entailment scoring failed: {e}")
            return 0.0

    def batch_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score multiple pairs efficiently."""
        if not pairs:
            return []

        try:
            scores = self.model.predict(pairs)
            return [float(max(0.0, min(1.0, s))) for s in scores]
        except Exception as e:
            logger.error(f"Batch entailment scoring failed: {e}")
            return [0.0] * len(pairs)
