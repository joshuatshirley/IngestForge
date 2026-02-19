"""
Document Authority Hierarchy for Retrieval Boosting.

Provides a 5-level authority system that boosts retrieval scores for
more authoritative documents. Higher authority sources (like primary
documents) are prioritized over lower authority (like guides or summaries).

Authority Levels
----------------
1. Primary Sources (1.25x boost)
   - Original research, primary documents, official standards
   - Example: Original regulations, research papers, specifications

2. Core References (1.15x boost)
   - Well-established secondary sources with high reliability
   - Example: Official manuals, textbooks, peer-reviewed content

3. Supporting Materials (1.05x boost)
   - Good quality supplementary content
   - Example: Technical notes, application guides, tutorials

4. Standard Documents (1.00x - no boost)
   - Default level for unclassified content
   - Example: General documentation, articles

5. Guides and Summaries (0.90x penalty)
   - Derivative content, summaries, third-party guides
   - Example: Quick reference cards, cheat sheets, blog posts

Architecture Context
--------------------
Authority boosting integrates with HybridRetriever after score fusion:

    HybridRetriever._fuse_weighted() or _fuse_rrf()
            |
    _apply_authority_boost()
            |
    get_authority_boost(document_id, metadata) -> boost factor
            |
    final_score = fused_score * boost

Setting Authority Levels
------------------------
Authority level can be set per-document during ingestion via metadata:

    chunk.metadata["authority_level"] = 1  # Primary source

Or by document type patterns in configuration:

    retrieval:
      authority:
        patterns:
          - match: "regulations/*.pdf"
            level: 1
          - match: "guides/*.md"
            level: 5

Reference Implementation
------------------------
Based on Army Doctrine RAG reranker.py which uses authority boosting
to prioritize regulations over guides and manuals.
"""

from typing import Any, Dict, Optional, List
import re

from ingestforge.core.logging import get_logger


logger = get_logger(__name__)


# Authority level boost multipliers
# Higher authority = higher boost = ranks higher in results
AUTHORITY_LEVELS: Dict[int, float] = {
    1: 1.25,  # Primary sources - maximum boost
    2: 1.15,  # Core references
    3: 1.05,  # Supporting materials
    4: 1.00,  # Standard documents (no boost)
    5: 0.90,  # Guides/summaries (slight penalty)
}


def get_authority_boost(
    document_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    authority_patterns: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Determine authority boost factor for a document.

    Authority is determined in order of precedence:
    1. Explicit authority_level in metadata
    2. Pattern matching against document_id/source_file
    3. Default level (4 = 1.0x, no boost)

    Args:
        document_id: Document identifier
        metadata: Chunk/document metadata dict
        authority_patterns: Optional list of pattern rules from config:
            [{"match": "regulations/*", "level": 1}, ...]

    Returns:
        Boost multiplier (e.g., 1.25 for primary sources)
    """
    metadata = metadata or {}

    # 1. Check explicit authority_level in metadata
    if "authority_level" in metadata:
        level = metadata["authority_level"]
        try:
            level = int(level)
            return AUTHORITY_LEVELS.get(level, 1.0)
        except (ValueError, TypeError):
            logger.warning(f"Invalid authority_level '{level}' for {document_id}")

    # 2. Check pattern matching
    if authority_patterns:
        source_file = metadata.get("source_file", document_id)
        for pattern_rule in authority_patterns:
            pattern = pattern_rule.get("match", "")
            level = pattern_rule.get("level", 4)

            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            if re.search(regex_pattern, source_file, re.IGNORECASE):
                return AUTHORITY_LEVELS.get(level, 1.0)

    # 3. Default: level 4 = 1.0x (no boost)
    return AUTHORITY_LEVELS.get(4, 1.0)


def get_authority_level_name(level: int) -> str:
    """Get human-readable name for an authority level."""
    names = {
        1: "Primary Source",
        2: "Core Reference",
        3: "Supporting Material",
        4: "Standard Document",
        5: "Guide/Summary",
    }
    return names.get(level, "Unknown")


def apply_authority_boost_to_results(
    results: List[Any],
    authority_patterns: Optional[List[Dict[str, Any]]] = None,
) -> List[Any]:
    """
    Apply authority boost to a list of search results.

    Modifies result scores in-place based on document authority levels.

    Args:
        results: List of SearchResult objects with .score, .document_id, .metadata
        authority_patterns: Optional pattern rules from config

    Returns:
        Same list with scores adjusted by authority boost
    """
    for result in results:
        boost = get_authority_boost(
            result.document_id,
            result.metadata,
            authority_patterns,
        )
        result.score = result.score * boost

        # Store boost info in metadata for transparency
        if result.metadata is None:
            result.metadata = {}
        result.metadata["authority_boost"] = boost

    # Re-sort by boosted score
    results.sort(key=lambda r: r.score, reverse=True)
    return results
