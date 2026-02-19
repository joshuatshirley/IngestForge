"""
Storage Backends for Chunk Persistence.

This module handles Stage 5 of the pipeline: persisting enriched chunks to
storage backends that support both text and vector search.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

Pipeline Stage: 5 (Index)

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ EnrichedChunks  │────→│    Storage      │────→│   Queryable     │
    │ + embeddings    │     │  (JSONL/Chroma) │     │   Knowledge     │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

Storage Backends
----------------
**JSONLRepository**
    Simple file-based storage using JSON Lines format:
    - Zero dependencies
    - Human-readable data files
    - Good for development and small datasets
    - Linear search (no vector index)

**ChromaDBRepository**
    Vector database with embedding search:
    - Built-in vector similarity search
    - Persistent SQLite storage
    - Metadata filtering
    - Production-ready for medium datasets

Backend Selection
-----------------
The storage backend is selected via configuration:

    storage:
      backend: chromadb    # jsonl or chromadb
      path: .data/chroma   # Storage location

Or programmatically:

    from ingestforge.storage import get_storage_backend
    storage = get_storage_backend(config)

Key Components
--------------
**ChunkRepository (Base Class)**
    Abstract interface all backends implement:
    - add_chunks(): Store chunks with embeddings
    - get_chunk(): Retrieve by ID
    - search(): Text-based search
    - search_semantic(): Vector similarity search
    - delete_chunk(): Remove a chunk
    - clear(): Remove all data

**SearchResult**
    Dataclass returned by search operations:
    - chunk_id: Unique identifier
    - content: Chunk text
    - score: Relevance score (0-1)
    - metadata: Source location, entities, etc.

**ParentChunkMapping**
    Maps child chunks to parent chunks for context expansion.
    Used by ParentRetriever to fetch surrounding context.

**ParentMappingStore**
    Persistent storage for parent-child chunk relationships.

Factory Function
----------------
    from ingestforge.storage import get_storage_backend

    # Automatically selects based on config
    storage = get_storage_backend(config)

    # Add chunks
    storage.add_chunks(enriched_chunks)

    # Search
    results = storage.search("quantum computing", top_k=5)
    results = storage.search_semantic(query_embedding, top_k=5)

Usage Example
-------------
    from ingestforge.storage import ChromaDBRepository

    # Create repository
    repo = ChromaDBRepository(config)

    # Store enriched chunks
    count = repo.add_chunks(chunks)
    print(f"Indexed {count} chunks")

    # Retrieve a chunk
    chunk = repo.get_chunk("chunk_123")

    # Search by text
    results = repo.search("machine learning", top_k=10)

    # Search by embedding
    results = repo.search_semantic(query_vector, top_k=10)

    # Clean up
    repo.clear()

Migration Between Backends
--------------------------
To migrate from JSONL to ChromaDB:

    from ingestforge.storage import JSONLRepository, ChromaDBRepository

    # Load from JSONL
    jsonl = JSONLRepository(config)
    all_chunks = jsonl.get_all_chunks()

    # Save to ChromaDB
    chroma = ChromaDBRepository(config)
    chroma.add_chunks(all_chunks)
"""

from ingestforge.storage.base import ChunkRepository, SearchResult, ParentChunkMapping
from ingestforge.storage.jsonl import JSONLRepository
from ingestforge.storage.factory import get_storage_backend
from ingestforge.storage.parent_mapping import (
    ParentMappingStore,
    create_parent_mapping_store,
)
from ingestforge.storage.analysis import (
    AnalysisRecord,
    AnalysisStorage,
    get_analysis_storage,
    ANALYSIS_TYPES,
)
from ingestforge.storage.bookmarks import (
    Bookmark,
    BookmarkManager,
    get_bookmark_manager,
)
from ingestforge.storage.annotations import (
    Annotation,
    AnnotationManager,
    get_annotation_manager,
    compute_content_hash,
)

# ChromaDB is optional (requires native C++ extensions)
try:
    from ingestforge.storage.chromadb import ChromaDBRepository
except ImportError:
    ChromaDBRepository = None  # type: ignore[assignment,misc]

__all__ = [
    "ChunkRepository",
    "SearchResult",
    "ParentChunkMapping",
    "JSONLRepository",
    "ChromaDBRepository",
    "get_storage_backend",
    "ParentMappingStore",
    "create_parent_mapping_store",
    # Analysis storage
    "AnalysisRecord",
    "AnalysisStorage",
    "get_analysis_storage",
    "ANALYSIS_TYPES",
    # Bookmark storage (ORG-003)
    "Bookmark",
    "BookmarkManager",
    "get_bookmark_manager",
    # Annotation storage (ORG-004)
    "Annotation",
    "AnnotationManager",
    "get_annotation_manager",
    "compute_content_hash",
]
