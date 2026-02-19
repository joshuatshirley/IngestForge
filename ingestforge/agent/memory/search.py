"""Semantic Memory Search.

Provides vector-based retrieval for persistent agent facts.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from sqlalchemy.orm import Session

from ingestforge.agent.memory.models import AgentFact
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class MemorySearcher:
    """Logic for recalling relevant facts from historical missions."""

    def __init__(self, embedding_model: Optional[Any] = None):
        self._model = embedding_model
        self._lazy_model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model:
            return self._model
        if not self._lazy_model:
            from sentence_transformers import SentenceTransformer

            self._lazy_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._lazy_model

    def recall_relevant(
        self, db: Session, query: str, limit: int = 3, threshold: float = 0.75
    ) -> List[AgentFact]:
        """Find facts in SQL that are semantically similar to the query.

        Rule #1: Simple logic with early returns.
        Rule #2: Fixed upper bound (limit).
        """
        if not query or len(query.strip()) < 5:
            return []
        if limit > 10:
            limit = 10

        # Fetch all facts (for small local corpora, in-memory similarity is fast)
        # For large memory banks, we would use a proper vector index
        facts = db.query(AgentFact).all()
        if not facts:
            return []

        # Perform similarity search
        query_emb = self.model.encode(query)
        fact_texts = [f.fact_text for f in facts]
        fact_embs = self.model.encode(fact_texts)

        # Calculate cosine similarity
        similarities = np.dot(fact_embs, query_emb) / (
            np.linalg.norm(fact_embs, axis=1) * np.linalg.norm(query_emb)
        )

        # Filter and sort
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                results.append((facts[idx], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]
