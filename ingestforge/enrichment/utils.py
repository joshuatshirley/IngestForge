"""
Enrichment Utilities.

Provides MetadataMerger for resolving conflicts when multiple 
domain refiners contribute to the same chunk.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MetadataMerger:
    """
    Handles merging metadata from multiple sources with collision resolution.
    """

    @staticmethod
    def merge(
        base: Dict[str, Any], new_data: Dict[str, Any], priority: int = 1
    ) -> Dict[str, Any]:
        """
        Merge new_data into base. Rule #1: Reduced nesting.
        """
        if not new_data:
            return base

        for key, value in new_data.items():
            if key not in base:
                base[key] = value
                continue

            # Key collision logic - extract to helper
            base[key] = MetadataMerger._resolve_collision(base[key], value)

        return base

    @staticmethod
    def _resolve_collision(existing: Any, new_val: Any) -> Any:
        """Helper to resolve metadata collisions. Rule #1: Reduced nesting."""
        # Case 1: Both are lists
        if isinstance(existing, list) and isinstance(new_val, list):
            return list(set(existing + new_val))

        # Case 2: Both are dicts
        if isinstance(existing, dict) and isinstance(new_val, dict):
            existing.update(new_val)
            return existing

        # Case 3: Different values, prefer non-None or listify
        if existing == new_val:
            return existing

        if not isinstance(existing, list):
            return [existing, new_val]

        if new_val not in existing:
            existing.append(new_val)
        return existing
