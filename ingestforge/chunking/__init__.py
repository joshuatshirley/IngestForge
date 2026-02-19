"""
Chunking Module for Semantic Text Splitting.

This module handles Stage 3 of the pipeline: splitting extracted text into
semantically coherent chunks suitable for retrieval and embedding.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

Pipeline Stage: 3 (Chunk)

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Extracted Text │────→│    Chunk        │────→│  ChunkRecords   │
    │  (raw string)   │     │  (semantic)     │     │  (300 words)    │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

Why Chunking Matters
--------------------
1. **Context window limits**: LLMs can only process limited text at once
2. **Retrieval precision**: Smaller chunks = more precise search matches
3. **Citation granularity**: Chunks map to specific source locations
4. **Embedding quality**: Embeddings work better on coherent text units

Chunking Strategies
-------------------
**SemanticChunker** (default)
    Splits on semantic boundaries (paragraphs, sections, sentences).
    Preserves meaning by not breaking mid-thought.
    Target: ~300 words with 50-word overlap.

**LegalChunker**
    Specialized for legal documents with numbered sections.
    Preserves legal citation structure (§1.2.3, Article IV, etc.).

**CodeChunker**
    Splits code by function/class boundaries.
    Preserves syntactic units for better code search.

Supporting Components
---------------------
**ChunkRecord**
    Dataclass representing a single chunk:
    - chunk_id: Unique identifier
    - content: The text content
    - document_id: Parent document reference
    - source_location: Citation information
    - embeddings: Vector representation (added later)

**SizeOptimizer**
    Adjusts chunk sizes based on content type and quality.
    Merges small chunks, splits oversized ones.

**Deduplicator**
    Removes duplicate or near-duplicate chunks.
    Uses content hashing and similarity thresholds.

**quality_scorer**
    Scores chunk quality for filtering low-value content.

Configuration
-------------
Chunking behavior is controlled by ChunkingConfig:

    chunking:
      strategy: semantic     # semantic, fixed, paragraph
      target_size: 300       # words
      min_size: 50           # minimum chunk size
      max_size: 1000         # maximum chunk size
      overlap: 50            # word overlap between chunks

Usage Example
-------------
    from ingestforge.chunking import SemanticChunker, ChunkRecord

    # Create chunker with config
    chunker = SemanticChunker(config)

    # Chunk extracted text
    chunks = chunker.chunk(
        text="Chapter 1\\n\\nThis is the content...",
        document_id="doc_123",
        source_file="textbook.pdf",
    )

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {len(chunk.content)} chars")

    # For legal documents
    from ingestforge.chunking import LegalChunker
    legal_chunks = LegalChunker().chunk(legal_text, "contract_001")

    # For code
    from ingestforge.chunking import CodeChunker
    code_chunks = CodeChunker().chunk(source_code, "module_py")
"""

from ingestforge.chunking.semantic_chunker import SemanticChunker, ChunkRecord
from ingestforge.chunking.size_optimizer import SizeOptimizer

# i: Compatibility alias for transition period
# Use IFChunkArtifact for new code
ChunkRecordCompat = ChunkRecord
from ingestforge.chunking.deduplicator import Deduplicator
from ingestforge.chunking.legal_chunker import LegalChunker
from ingestforge.chunking.code_chunker import CodeChunker
from ingestforge.chunking.header_chunker import HeaderChunker
from ingestforge.chunking.layout_chunker import (
    LayoutChunker,
    LayoutSection,
    chunk_by_layout,
    chunk_by_title,
)

__all__ = [
    "SemanticChunker",
    "ChunkRecord",
    "ChunkRecordCompat",  # i: Alias for transition
    "SizeOptimizer",
    "Deduplicator",
    "LegalChunker",
    "CodeChunker",
    "HeaderChunker",
    # Layout-aware chunking (Unstructured-style)
    "LayoutChunker",
    "LayoutSection",
    "chunk_by_layout",
    "chunk_by_title",
]
