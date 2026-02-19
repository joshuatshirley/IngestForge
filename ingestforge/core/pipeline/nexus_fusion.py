"""
Reciprocal Rank Fusion (RRF) for Workspace Nexus.

Task 129: Unbiased merging of local and remote search results.
NASA JPL Power of Ten: Rule #2 (Bounds), Rule #4 (Small functions).
"""

import logging
from typing import List, Dict
from ingestforge.core.models.search import SearchResult, MAX_TOP_K

logger = logging.getLogger(__name__)

# Standard RRF hyperparameter (prevents low-rank results from dominating)
RRF_K = 60
# JPL Rule #2: Fixed upper bound for the total merge pool
MAX_MERGE_POOL = 1000


class NexusResultFusion:
    """
    Merges and ranks search results from multiple Nexus instances using RRF.
    """

    def merge(
        self, results_by_source: Dict[str, List[SearchResult]]
    ) -> List[SearchResult]:
        """
        Merge multiple lists into one ranked list.
        Rule #4: Logic under 60 lines.
        """
        if not results_by_source:
            return []

        score_map: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}

        # JPL Rule #2: Bounded loops over sources and results
        for source_id, results in results_by_source.items():
            for rank, res in enumerate(results[:MAX_TOP_K]):
                # Enforce total merge pool limit to satisfy Rule #2
                if len(score_map) >= MAX_MERGE_POOL:
                    break

                art_id = f"{res.nexus_id}:{res.artifact_id}"

                # RRF Formula: 1 / (K + rank + 1)
                rrf_increment = 1.0 / (RRF_K + rank + 1)

                score_map[art_id] = score_map.get(art_id, 0.0) + rrf_increment
                result_map[art_id] = res

        # Sort by cumulative RRF score descending
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)

        # JPL Rule #2: Final slice to ensure memory safety
        return self._build_final_list(sorted_ids, result_map, score_map)

    def _build_final_list(
        self,
        sorted_ids: List[str],
        res_map: Dict[str, SearchResult],
        score_map: Dict[str, float],
    ) -> List[SearchResult]:
        """Convert maps back to ordered SearchResult list."""
        final_list: List[SearchResult] = []
        for art_id in sorted_ids[:MAX_TOP_K]:
            res = res_map[art_id]
            # Update score to the unified RRF score
            res.score = score_map[art_id]
            final_list.append(res)
        return final_list
