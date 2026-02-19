"""
Retrieval Engines for Search.

This module provides search capabilities over the indexed knowledge base,
combining keyword matching (BM25) with semantic similarity for optimal results.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

Query Flow

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   User Query    │────→│   Retrieval     │────→│  SearchResults  │
    │   "What is X?"  │     │ (Hybrid/BM25/   │     │  (ranked list)  │
    └─────────────────┘     │  Semantic)      │     └─────────────────┘
                            └─────────────────┘

Retrieval Strategies
--------------------
**BM25Retriever**
    Classic keyword-based search using BM25 scoring.
    - Exact term matching
    - TF-IDF weighting
    - Good for specific terminology
    - Fast, no embeddings needed

**SemanticRetriever**
    Embedding-based semantic similarity search.
    - Finds conceptually similar content
    - Handles synonyms and paraphrasing
    - Requires pre-computed embeddings
    - Uses cosine similarity

**HybridRetriever** (recommended)
    Combines BM25 and semantic search:
    - Best of both worlds
    - Configurable weights (default: 0.3 BM25, 0.7 semantic)
    - Reciprocal Rank Fusion for score combination
    - Handles both exact matches and semantic similarity

**Reranker**
    Second-stage ranking using cross-encoder models:
    - More accurate than bi-encoder similarity
    - Slower (evaluates query-document pairs)
    - Applied to top-N results from initial retrieval

**ParentRetriever**
    Expands results to include surrounding context:
    - Retrieves child chunks
    - Fetches parent chunk for context
    - Good for question-answering

Configuration
-------------
Retrieval behavior is controlled by RetrievalConfig:

    retrieval:
      strategy: hybrid           # bm25, semantic, hybrid
      top_k: 5                   # Results to return
      hybrid_weights:
        bm25: 0.3
        semantic: 0.7
      rerank: true               # Apply cross-encoder reranking
      rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2

Usage Example
-------------
    from ingestforge.retrieval import HybridRetriever

    # Create retriever
    retriever = HybridRetriever(config, storage)

    # Search
    results = retriever.search("quantum entanglement", top_k=10)

    for result in results:
        print(f"[{result.score:.2f}] {result.content[:100]}...")
        print(f"  Source: {result.metadata.get('source_file')}")

    # With reranking
    from ingestforge.retrieval import Reranker
    reranker = Reranker()
    reranked = reranker.rerank("quantum entanglement", results, top_k=5)

    # With parent context
    from ingestforge.retrieval import create_parent_retriever
    parent_retriever = create_parent_retriever(config, storage)
    expanded = parent_retriever.search("quantum entanglement", top_k=5)

Hybrid Score Fusion
-------------------
HybridRetriever combines scores using Reciprocal Rank Fusion (RRF):

    RRF_score = Σ 1 / (k + rank_i)

Where k is a constant (default 60) and rank_i is the rank in each system.
This handles different score scales between BM25 and semantic search.
"""

from ingestforge.retrieval.bm25 import BM25Retriever
from ingestforge.retrieval.semantic import SemanticRetriever
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.retrieval.reranker import Reranker
from ingestforge.retrieval.parent_retriever import (
    ParentRetriever,
    ParentExpandedResult,
    create_parent_retriever,
)
from ingestforge.retrieval.weight_profiles import (
    HybridWeightProfile,
    HYBRID_PROFILES,
    get_profile,
    get_available_intents,
)
from ingestforge.retrieval.authority import (
    AUTHORITY_LEVELS,
    get_authority_boost,
    get_authority_level_name,
    apply_authority_boost_to_results,
)
from ingestforge.retrieval.cross_corpus import (
    CrossCorpusLinker,
    CodeReference,
    StoryReference,
    CoverageEntry,
    GapReport,
)
from ingestforge.retrieval.proximity import (
    ProximityRanker,
    ProximityScore,
    ProximityReport,
    create_proximity_ranker,
    rank_by_proximity,
    MAX_CANDIDATES,
    MAX_QUERY_TERMS,
    DEFAULT_PROXIMITY_WINDOW,
)

__all__ = [
    "BM25Retriever",
    "SemanticRetriever",
    "HybridRetriever",
    "Reranker",
    "ParentRetriever",
    "ParentExpandedResult",
    "create_parent_retriever",
    # Weight profiles
    "HybridWeightProfile",
    "HYBRID_PROFILES",
    "get_profile",
    "get_available_intents",
    # Authority
    "AUTHORITY_LEVELS",
    "get_authority_boost",
    "get_authority_level_name",
    "apply_authority_boost_to_results",
    # Cross-corpus linking
    "CrossCorpusLinker",
    "CodeReference",
    "StoryReference",
    "CoverageEntry",
    "GapReport",
    # Proximity ranking
    "ProximityRanker",
    "ProximityScore",
    "ProximityReport",
    "create_proximity_ranker",
    "rank_by_proximity",
    "MAX_CANDIDATES",
    "MAX_QUERY_TERMS",
    "DEFAULT_PROXIMITY_WINDOW",
]
