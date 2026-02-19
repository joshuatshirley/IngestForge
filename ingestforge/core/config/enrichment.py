"""
Enrichment configuration.

Provides configuration for chunk enrichment: embeddings, entity extraction,
question generation, summarization, and quality scoring.
"""

from dataclasses import dataclass


@dataclass
class EnrichmentConfig:
    """Enrichment configuration."""

    generate_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32  # Batch size for embedding generation
    extract_entities: bool = True
    generate_questions: bool = False
    generate_summaries: bool = False  # Chunk-level summaries via LLM
    compute_quality: bool = True
    use_instructor_citation: bool = (
        False  # Use instructor library for high-fidelity citations
    )

    # Confidence-Aware-Extraction
    min_confidence: float = 0.5  # Minimum confidence threshold for entities

    # Sub-batching for large documents (prevents OOM on 5000+ chunk docs)
    enrichment_max_batch_size: int = 500  # Max chunks per enrichment batch
    strict_memory_check: bool = False  # Fail if insufficient memory detected

    def __post_init__(self) -> None:
        """Validate enrichment configuration."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
