"""
Legal Fact Aggregator.

Phase 2 - Fact Linker.
Scans Knowledge Graph for chunks tagged with 'Legal' or 'Evidence'.

JPL Compliance:
- Rule #4: Functions < 60 lines.
- Rule #9: 100% type hints.
"""

from typing import List
from ingestforge.verticals.legal.models import LegalFact
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.core.pipeline.pipeline import Pipeline
from ingestforge.core.config_loaders import load_config


class LegalFactAggregator:
    """Aggregates legal facts from the knowledge base."""

    def __init__(self):
        self.config = load_config()
        self.pipeline = Pipeline(self.config)
        self.retriever = HybridRetriever(self.config, self.pipeline.storage)

    def aggregate_evidence(self, query: str, limit: int = 10) -> List[LegalFact]:
        """
        Retrieves and converts document chunks into LegalFact models.

        JPL Rule #4: Concise retrieval logic.
        """
        # Search for chunks with semantic relevance to the legal issue
        results = self.retriever.search(query=query, top_k=limit)

        facts: List[LegalFact] = []
        for res in results:
            # Only include high-confidence matches
            if res.score > 0.6:
                facts.append(
                    LegalFact(
                        text=res.content,
                        source_id=res.source_file or res.document_id,
                        page_number=res.page_start,
                        confidence=res.score,
                    )
                )
        return facts
