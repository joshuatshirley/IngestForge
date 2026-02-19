"""
Retrieval configuration.

Provides configuration for retrieval strategies: BM25, semantic, hybrid search.
Includes parent document retrieval, authority hierarchy, and reranking settings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class HybridConfig:
    """Hybrid retrieval configuration."""

    bm25_weight: float = 0.4
    semantic_weight: float = 0.6


@dataclass
class ParentDocConfig:
    """Parent document retrieval configuration."""

    enabled: bool = True
    context_window_tokens: int = 2048  # Max tokens for parent chunk context
    context_window_chars: int = (
        8000  # Fallback: max characters if tokenizer unavailable
    )
    deduplicate_parents: bool = True  # Remove duplicate parents from results
    expand_on_search: bool = True  # Auto-expand to parents during search


@dataclass
class AuthorityConfig:
    """Document authority hierarchy configuration.

    Authority levels boost retrieval scores for more authoritative documents:
    - Level 1: Primary sources (1.25x boost)
    - Level 2: Core references (1.15x boost)
    - Level 3: Supporting materials (1.05x boost)
    - Level 4: Standard documents (1.00x - default)
    - Level 5: Guides/summaries (0.90x penalty)
    """

    enabled: bool = False  # Disabled by default
    default_level: int = 4  # Standard documents
    patterns: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    strategy: str = "hybrid"  # bm25, semantic, hybrid
    top_k: int = 10
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    parent_doc: ParentDocConfig = field(default_factory=ParentDocConfig)
    authority: AuthorityConfig = field(default_factory=AuthorityConfig)
