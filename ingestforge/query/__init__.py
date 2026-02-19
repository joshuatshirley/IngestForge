"""
Query Processing Pipeline.

This module orchestrates the full query-to-answer flow: understanding the query,
retrieving relevant chunks, and generating a response with citations.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

Query Pipeline Stages

    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │  User Query   │────→│   Classify    │────→│    Expand     │
    │  "What is X?" │     │ (type/intent) │     │  (synonyms)   │
    └───────────────┘     └───────────────┘     └───────────────┘
                                                        │
    ┌───────────────┐     ┌───────────────┐     ┌───────┴───────┐
    │   Response    │←────│   Generate    │←────│   Retrieve    │
    │ + Citations   │     │  (LLM answer) │     │  (top chunks) │
    └───────────────┘     └───────────────┘     └───────────────┘

Key Components
--------------
**QueryPipeline**
    Main orchestrator that coordinates the full query flow:
    - Query preprocessing and classification
    - Retrieval execution (hybrid search)
    - Response generation with LLM
    - Citation building
    - Caching for repeated queries

**QueryClassifier**
    Analyzes query intent and type:
    - Factual: "What is quantum entanglement?"
    - Procedural: "How do I configure X?"
    - Comparative: "What's the difference between A and B?"
    - Exploratory: "Tell me about machine learning"

    Classification helps select retrieval strategy and response format.

**QueryExpander**
    Enhances queries for better retrieval:
    - Synonym expansion
    - Acronym resolution
    - Related term addition

    Example: "ML" → "ML machine learning artificial intelligence"

Supporting Components
---------------------
**cache.py**
    Query result caching:
    - Avoids re-computation for repeated queries
    - Configurable TTL and cache size
    - Memory or disk-based storage

Usage Example
-------------
    from ingestforge.query import QueryPipeline

    # Create pipeline
    pipeline = QueryPipeline(config, storage, llm_client)

    # Run a query
    response = pipeline.query("What are the main themes in chapter 3?")

    # Access response parts
    print(response.answer)
    for citation in response.citations:
        print(f"  [{citation.short_cite}]")

    # With caching
    response2 = pipeline.query("What are the main themes in chapter 3?")
    # Returns cached result

Query Response Structure
------------------------
QueryPipeline.query() returns a QueryResponse with:
- answer: The generated text answer
- citations: List of source citations
- sources: The retrieved chunks used
- metadata: Query timing, cache hit, etc.

Configuration
-------------
Query behavior is controlled by config:

    query:
      max_tokens: 1000           # Response length limit
      temperature: 0.7           # LLM creativity
      include_citations: true    # Add source references
      cache_enabled: true        # Cache repeated queries
      cache_ttl: 3600            # Cache lifetime (seconds)
"""

from ingestforge.query.pipeline import QueryPipeline
from ingestforge.query.classifier import QueryClassifier
from ingestforge.query.expander import QueryExpander
from ingestforge.query.validation import (
    AnswerValidator,
    ValidationReport,
    validate_answer,
)
from ingestforge.query.conflicts import (
    ConflictDetector,
    ConflictReport,
    Conflict,
    detect_conflicts,
)

__all__ = [
    "QueryPipeline",
    "QueryClassifier",
    "QueryExpander",
    # Validation
    "AnswerValidator",
    "ValidationReport",
    "validate_answer",
    # Conflicts
    "ConflictDetector",
    "ConflictReport",
    "Conflict",
    "detect_conflicts",
]
