"""
Metadata-Aware Search Rescorer.

Adjusts SearchResult scores based on field-specific boosts defined in DomainStrategies.
"""

import logging
from typing import List, Dict, Any
from ingestforge.storage.base import SearchResult
from ingestforge.query.routing import DomainStrategy

logger = logging.getLogger(__name__)


class MetadataRescorer:
    """
    Rescores search results by inspecting metadata for specific field matches.
    """

    def rescore(
        self, results: List[SearchResult], query: str, strategy: DomainStrategy
    ) -> List[SearchResult]:
        """
        Apply field boosts based on strategy. Rule #1: Reduced nesting.
        """
        if not strategy.boost_fields or not results:
            return results

        query_lower = query.lower()
        for res in results:
            self._boost_result(res, query_lower, strategy.boost_fields)

        # Re-sort after boosting
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _boost_result(
        self, res: SearchResult, query_lower: str, boost_fields: Dict[str, float]
    ) -> None:
        """Calculate and apply boosts for a single result. Rule #1: Reduced nesting."""
        metadata = res.metadata or {}
        new_score = res.score
        boosted = False

        for field, weight in boost_fields.items():
            if field not in metadata:
                continue

            if self._check_field_match(metadata[field], query_lower):
                new_score *= weight
                boosted = True

        if boosted:
            res.score = new_score

    def _check_field_match(self, metadata_val: Any, query_lower: str) -> bool:
        """Check if metadata value matches query. Rule #1: Reduced nesting."""
        vals = metadata_val if isinstance(metadata_val, list) else [metadata_val]

        for v in vals:
            v_str = str(v).lower()
            if not v_str:
                continue
            # Match: value in query (e.g. "CVE-2023-1234" in user query)
            if v_str in query_lower:
                return True
        return False
