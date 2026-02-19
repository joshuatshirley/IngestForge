"""
Intent-Aware Hybrid Weight Profiles.

Provides dynamic weight profiles for BM25 vs semantic search based on
query intent classification. Different query types benefit from different
retrieval strategies:

- Factual queries: High BM25 weight catches exact terminology matches
- Conceptual queries: Higher semantic weight finds related concepts
- Procedural queries: Balanced approach for step-by-step content
- Comparative queries: Higher semantic for cross-document comparisons
- Exploratory queries: Maximum semantic for broad discovery
- Literary queries: High semantic for thematic/symbolic content

Architecture Context
--------------------
Weight profiles integrate with the HybridRetriever to dynamically adjust
fusion weights based on the classified query intent:

    QueryPipeline.process(query)
            |
    QueryClassifier.classify() -> query_type
            |
    HybridRetriever.search(query_intent=query_type)
            |
    get_profile(query_type) -> HybridWeightProfile
            |
    Use weights for BM25/semantic fusion

Reference Implementation
------------------------
Based on Army Doctrine RAG hybrid_config.py which achieves 100% retrieval
accuracy with local Ollama LLMs using intent-aware weight profiles.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class HybridWeightProfile:
    """Weight configuration for hybrid retrieval fusion.

    Attributes:
        bm25_weight: Weight for BM25 (keyword) retrieval [0.0-1.0]
        semantic_weight: Weight for semantic (vector) retrieval [0.0-1.0]

    Note: Weights are automatically normalized in HybridRetriever if they
    don't sum to 1.0.
    """

    bm25_weight: float
    semantic_weight: float

    def __post_init__(self) -> None:
        """Validate weights are in valid range."""
        if not (0.0 <= self.bm25_weight <= 1.0):
            raise ValueError(f"bm25_weight must be 0.0-1.0, got {self.bm25_weight}")
        if not (0.0 <= self.semantic_weight <= 1.0):
            raise ValueError(
                f"semantic_weight must be 0.0-1.0, got {self.semantic_weight}"
            )


# Intent-specific weight profiles
# Higher BM25 for exact term matching, higher semantic for conceptual similarity
HYBRID_PROFILES: Dict[str, HybridWeightProfile] = {
    # Factual: "What is X?", "Who is Y?", "When did Z happen?"
    # High BM25 to catch exact terminology and named entities
    "factual": HybridWeightProfile(0.65, 0.35),
    # Procedural: "How do I...", "What are the steps to..."
    # Balanced - need both exact procedure names and related content
    "procedural": HybridWeightProfile(0.55, 0.45),
    # Conceptual: "Explain the concept of...", "What does X mean?"
    # Balanced - terminology matters but conceptual similarity helps
    "conceptual": HybridWeightProfile(0.50, 0.50),
    # Comparative: "Compare X and Y", "What's the difference between..."
    # Higher semantic to find related content across documents
    "comparative": HybridWeightProfile(0.45, 0.55),
    # Exploratory: "Tell me about...", "What do we know about..."
    # Higher semantic for broad discovery and related content
    "exploratory": HybridWeightProfile(0.40, 0.60),
    # Literary: Theme analysis, symbolism, character interpretation
    # Maximum semantic for thematic and symbolic connections
    "literary": HybridWeightProfile(0.35, 0.65),
    # Default fallback - balanced toward semantic
    "default": HybridWeightProfile(0.40, 0.60),
}


def get_profile(intent: str) -> HybridWeightProfile:
    """
    Get the appropriate weight profile for a query intent.

    Args:
        intent: Query intent classification from QueryClassifier.
                Common values: factual, procedural, conceptual,
                comparative, exploratory, literary

    Returns:
        HybridWeightProfile with appropriate BM25/semantic weights
    """
    # Normalize intent to lowercase
    intent_key = intent.lower() if intent else "default"

    return HYBRID_PROFILES.get(intent_key, HYBRID_PROFILES["default"])


def get_available_intents() -> list[str]:
    """Get list of available intent types with weight profiles."""
    return list(HYBRID_PROFILES.keys())
