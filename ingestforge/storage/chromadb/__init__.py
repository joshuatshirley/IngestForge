"""
ChromaDB Vector Storage Backend for IngestForge.

This module provides semantic search using ChromaDB vector database.

Public API
----------
All repository classes are re-exported here for backward compatibility:

    from ingestforge.storage.chromadb import ChromaDBRepository

Architecture
------------
ChromaDB storage is organized into focused modules:

    chromadb/
    ├── repository.py    # Main ChromaDBRepository class
    ├── crud.py          # CRUD operations mixin
    ├── search.py        # Search operations mixin
    ├── management.py    # Statistics and library management mixin
    └── lifecycle.py     # Resource cleanup and context manager mixin

Usage Example
-------------
    from ingestforge.storage.chromadb import ChromaDBRepository
    from pathlib import Path

    # Initialize
    repo = ChromaDBRepository(
        persist_directory=Path(".data/chromadb"),
        enable_multi_vector=True
    )

    # Add chunks
    repo.add_chunks(chunks)

    # Search
    results = repo.search("query text", top_k=10)
"""

__all__ = [
    "ChromaDBRepository",
]


def __getattr__(name):
    """Lazy imports to prevent circular dependencies at module load time."""
    if name == "ChromaDBRepository":
        from ingestforge.storage.chromadb.repository import ChromaDBRepository

        return ChromaDBRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
