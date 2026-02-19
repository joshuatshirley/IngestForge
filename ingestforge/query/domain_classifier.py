"""
Query Domain Classifier.

Analyzes natural language queries to determine the target domain
using the weighted heuristic signals from DomainRouter.
"""

import logging
from typing import List

from ingestforge.enrichment.router import DomainRouter
from ingestforge.query.routing import DomainStrategy, get_merged_strategy

logger = logging.getLogger(__name__)


class QueryDomainClassifier:
    """
    Classifies queries into domains to optimize retrieval.
    """

    def __init__(self, min_score: int = 2):
        """
        Args:
            min_score: Lower threshold than ingestion because queries are short.
        """
        self.router = DomainRouter()
        self.min_score = min_score

    def classify_query(self, query: str) -> List[str]:
        """
        Returns a list of detected domains for the query.
        """
        ranked = self.router.classify_chunk(query)
        if not ranked:
            return []

        # For queries, we allow lower scores but prefer the top ones
        # If the top score is very strong, we take just that.
        # If there are multiple close scores, we take them all for a merged strategy.

        top_score = ranked[0][1]
        if top_score < self.min_score:
            return []

        detected = [ranked[0][0]]
        for domain, score in ranked[1:]:
            # If secondary domain is at least 80% as strong as primary
            if score >= (top_score * 0.8) and score >= self.min_score:
                detected.append(domain)

        return detected

    def get_query_strategy(self, query: str) -> DomainStrategy:
        """
        Classifies query and returns the appropriate retrieval strategy.
        """
        domains = self.classify_query(query)
        if not domains:
            return DomainStrategy(name="default")

        return get_merged_strategy(domains)
