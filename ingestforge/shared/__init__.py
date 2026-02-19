"""
Shared Utilities and Patterns for IngestForge.

This module forms the second layer of IngestForge's hexagonal architecture,
providing reusable utilities and interface patterns that feature modules depend on.

Architecture Position
---------------------
    CLI (outermost)
      └── Feature Modules (ingest, chunking, enrichment, storage, retrieval, query, llm)
            └── **Shared** (you are here - patterns, interfaces, utilities)
                  └── Core (innermost - config, logging, retry, security)

The Shared layer sits between Core (stable foundation) and Feature modules
(domain-specific logic). It provides:

1. **Interface patterns** - Abstract base classes that define contracts
2. **Utility functions** - Common operations used across features
3. **Serialization helpers** - JSON handling for complex types

Submodules
----------
**text_utils**
    Text processing utilities: cleaning, normalization, sentence splitting.
    Used by: ingest processors, chunking strategies, enrichers.

**lazy_imports**
    Lazy loading for optional dependencies. Prevents import errors when
    optional packages aren't installed.
    Used by: LLM clients, enrichers, processors with optional deps.

**metadata_serialization**
    JSON serialization for complex types (SourceLocation, Path, datetime).
    Used by: storage backends, state persistence.

**patterns/**
    Interface definitions (abstract base classes):
    - IEnricher: Contract for chunk enrichment
    - IChunkingStrategy: Contract for text chunking
    - IProcessor: Contract for document processing

Design Principles
-----------------
1. **No feature dependencies**: Shared depends only on Core, never on features.
2. **Interface segregation**: Small, focused interfaces.
3. **Reusable utilities**: Common operations extracted to avoid duplication.
4. **Optional dependency handling**: Graceful degradation when packages missing.

Usage Example
-------------
    # Use text utilities
    from ingestforge.shared import clean_text, normalize_whitespace
    cleaned = clean_text(raw_text)

    # Use lazy loading
    from ingestforge.shared import lazy_property

    class MyClient:
        @lazy_property
    def model(self) -> Any:
            import expensive_ml_library
            return expensive_ml_library.load_model()

    # Use patterns
    from ingestforge.shared.patterns import IEnricher, EnrichmentPipeline

    class MyEnricher(IEnricher):
        def enrich_chunk(self, chunk: Any):
            chunk.custom_field = compute_value(chunk)
            return chunk
"""


from ingestforge.shared.text_utils import (
    clean_text,
    normalize_whitespace,
    read_text_with_fallback,
)
from ingestforge.shared.lazy_imports import lazy_property
from ingestforge.shared.metadata_serialization import (
    serialize_source_location,
    deserialize_source_location,
)

__all__ = [
    "clean_text",
    "normalize_whitespace",
    "read_text_with_fallback",
    "lazy_property",
    "serialize_source_location",
    "deserialize_source_location",
]
