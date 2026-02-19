# Storage Module

## Purpose

Persist enriched chunks to storage backends (ChromaDB, JSONL, PostgreSQL). Provides abstract `ChunkRepository` interface for swappable storage with semantic search, metadata filtering, and parent document retrieval.

## Architecture Context

```
┌─────────────────────────────────────────┐
│   enrichment/ - Add embeddings          │
│            ↓                             │
│   storage/ - Persist chunks             │  ← You are here
│   (ChromaDB, JSONL, PostgreSQL)         │
│            ↓                             │
│   retrieval/ - Search chunks            │
└─────────────────────────────────────────┘
```

## Key Components

| Component | Purpose | Backend |
|-----------|---------|---------|
| `base.py` | ChunkRepository interface + SearchResult | Abstract |
| `chromadb.py` | ChromaDBRepository - vector storage | ChromaDB |
| `jsonl.py` | JSONLRepository - file-based storage | JSONL |
| `factory.py` | get_storage_backend - auto-select backend | Factory |
| `parent_mapping.py` | Parent chunk mapping for context expansion | SQLite |

## ChunkRepository Interface

**Purpose:** Abstract interface that all storage backends implement.

**Contract:**

```python
class ChunkRepository(ABC):
    # Write operations
    @abstractmethod
    def add_chunk(self, chunk: ChunkRecord) -> bool

    @abstractmethod
    def add_chunks(self, chunks: List[ChunkRecord]) -> int

    # Read operations
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]

    @abstractmethod
    def get_chunks_by_document(self, document_id: str) -> List[ChunkRecord]

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[SearchResult]

    # Delete operations
    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool

    @abstractmethod
    def delete_document(self, document_id: str) -> int

    # Utilities
    @abstractmethod
    def count_chunks(self) -> int

    @abstractmethod
    def list_documents(self) -> List[str]
```

## SearchResult Data Structure

**Purpose:** Standardized search result format across all backends.

```python
@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float                          # Relevance score 0-1
    document_id: str
    section_title: str
    chunk_type: str
    source_file: str
    word_count: int

    # Optional fields
    metadata: Dict[str, Any]
    source_location: Optional[Dict]       # For citations
    page_start: Optional[int]
    page_end: Optional[int]
```

## ChromaDBRepository

**Purpose:** Vector database storage with semantic search using ChromaDB.

**Features:**
- Vector similarity search (cosine distance)
- Metadata filtering
- Persistent storage
- Automatic embedding generation (optional)
- Parent document expansion

**Usage:**

```python
from ingestforge.storage import ChromaDBRepository
from pathlib import Path

# Initialize
repo = ChromaDBRepository(
    persist_directory=Path(".data/chromadb"),
    collection_name="my_documents"
)

# Add chunks
repo.add_chunks(enriched_chunks)

# Semantic search
results = repo.search(
    query="What is quantum computing?",
    top_k=5
)

for result in results:
    print(f"Score: {result.score:.2f}")
    print(f"Content: {result.content[:100]}...")
    print(f"Source: {result.source_file}, p.{result.page_start}")
    print

# Get all chunks from document
doc_chunks = repo.get_chunks_by_document("doc-001")
print(f"Document has {len(doc_chunks)} chunks")

# Delete document
removed = repo.delete_document("doc-001")
print(f"Removed {removed} chunks")
```

**Configuration:**

```yaml
# config.yaml
storage:
  backend: chromadb
  chromadb:
    persist_directory: .data/chromadb
```

**Embedding Handling:**

ChromaDB can use:
1. Pre-computed embeddings (from EmbeddingGenerator)
2. Automatic embedding generation (SentenceTransformer)

```python
# Option 1: Use pre-computed embeddings
chunks = embedder.enrich_batch(chunks)  # Adds chunk.embedding
repo.add_chunks(chunks)

# Option 2: Let ChromaDB generate embeddings
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

repo = ChromaDBRepository(
    persist_directory=Path(".data/chromadb"),
    embedding_function=ef
)
```

## JSONLRepository

**Purpose:** Simple file-based storage for development and small datasets.

**Features:**
- No external dependencies
- Human-readable format
- In-memory BM25 index
- Fast for small datasets (<100K chunks)
- Automatic backup on save

**Usage:**

```python
from ingestforge.storage import JSONLRepository
from pathlib import Path

# Initialize
repo = JSONLRepository(data_path=Path(".data"))

# Add chunks (saved to .data/chunks/chunks.jsonl)
repo.add_chunks(chunks)

# Keyword search (BM25)
results = repo.search("quantum computing", top_k=5)

# File structure:
# .data/
# ├── chunks/
# │   └── chunks.jsonl      ← Chunks (one per line)
# └── index/
#     └── bm25_index.json   ← Term frequency index
```

**Limitations:**
- No vector search (BM25 only)
- Limited to ~100K chunks (memory constraints)
- Slower than ChromaDB for large datasets
- No concurrent write support

**When to Use:**
- Development and testing
- Small datasets (<10K chunks)
- No GPU/vector DB setup
- Quick prototyping

## Factory Pattern

**Purpose:** Automatically select storage backend based on configuration.

**Usage:**

```python
from ingestforge.storage import get_storage_backend
from ingestforge.core import load_config

config = load_config

# Auto-select based on config.storage.backend
repo = get_storage_backend(config)

# Use repo (works with any backend)
repo.add_chunks(chunks)
results = repo.search("query", top_k=5)
```

**Backend Selection:**

```yaml
# config.yaml
storage:
  backend: chromadb  # chromadb, jsonl, postgres
```

## Parent Document Mapping

**Purpose:** Track parent-child chunk relationships for context expansion.

**Use Case:** When searching returns a small chunk, expand to include parent chunk for more context.

**Usage:**

```python
from ingestforge.storage import ParentMappingStore, create_parent_mapping_store

# Create mapping store
mapping_store = create_parent_mapping_store(Path(".data/parent_map.db"))

# Store parent-child relationship
mapping_store.add_mapping(
    child_id="doc-001_chunk_5",
    parent_id="doc-001_chunk_0",  # Section header
    document_id="doc-001"
)

# Retrieve parent when needed
parent_id = mapping_store.get_parent("doc-001_chunk_5")
parent_chunk = repo.get_chunk(parent_id)

# Combine for full context
context = parent_chunk.content + "\n\n" + child_chunk.content
```

**Parent Chunk Strategy:**

```python
# Strategy 1: Section headers as parents
for i, chunk in enumerate(chunks):
    if chunk.chunk_type == "heading":
        parent_id = chunk.chunk_id
    else:
        mapping_store.add_mapping(chunk.chunk_id, parent_id, doc_id)

# Strategy 2: Sliding window parents
for i, chunk in enumerate(chunks):
    if i > 0:
        parent_id = chunks[i - 1].chunk_id
        mapping_store.add_mapping(chunk.chunk_id, parent_id, doc_id)
```

## Usage Examples

### Example 1: Full Storage Pipeline

```python
from ingestforge.core import load_config
from ingestforge.storage import get_storage_backend
from ingestforge.enrichment import EmbeddingGenerator

config = load_config

# Enrich chunks
embedder = EmbeddingGenerator(config)
enriched = embedder.enrich_batch(chunks)

# Store chunks
repo = get_storage_backend(config)
count = repo.add_chunks(enriched)

print(f"Stored {count} chunks")
print(f"Total in storage: {repo.count_chunks}")
```

### Example 2: Search with Metadata Filtering

```python
# ChromaDB supports metadata filtering
results = repo.search(
    query="quantum computing",
    top_k=10,
    where={"document_id": "doc-001"}  # Filter by document
)

# Or filter by chunk type
results = repo.search(
    query="definition of qubit",
    top_k=5,
    where={"chunk_type": "definition"}
)
```

### Example 3: Batch Operations with Progress

```python
from tqdm import tqdm

# Process in batches
batch_size = 1000
total_added = 0

for i in tqdm(range(0, len(all_chunks), batch_size), desc="Storing"):
    batch = all_chunks[i:i + batch_size]
    added = repo.add_chunks(batch)
    total_added += added

print(f"Stored {total_added} chunks")
```

### Example 4: Backend Migration

```python
# Migrate from JSONL to ChromaDB
from ingestforge.storage import JSONLRepository, ChromaDBRepository

# Load from JSONL
jsonl_repo = JSONLRepository(Path(".data"))
all_chunks = []
for doc_id in jsonl_repo.list_documents:
    chunks = jsonl_repo.get_chunks_by_document(doc_id)
    all_chunks.extend(chunks)

# Save to ChromaDB
chroma_repo = ChromaDBRepository(Path(".data/chromadb"))
chroma_repo.add_chunks(all_chunks)

print(f"Migrated {len(all_chunks)} chunks")
```

## Dependencies

### Required
- Python standard library (json, pathlib)

### Optional
- `chromadb>=0.4.0` - Vector database (for ChromaDBRepository)
- `psycopg2-binary>=2.9.0` - PostgreSQL adapter (for future PostgresRepository)

### Installation

```bash
# Minimal (JSONL only)
# No dependencies needed

# ChromaDB support
pip install chromadb

# PostgreSQL support
pip install psycopg2-binary
```

## Testing

```bash
# Run all storage tests
pytest tests/test_storage_*.py -v

# Test specific backend
pytest tests/test_storage_chromadb.py -v
pytest tests/test_storage_jsonl.py -v

# Test with coverage
pytest tests/test_storage_*.py --cov=ingestforge.storage
```

## Common Patterns

### Pattern 1: Idempotent Ingestion

```python
# Check if document already exists before re-ingesting
existing_docs = repo.list_documents

if document_id not in existing_docs:
    repo.add_chunks(chunks)
else:
    print(f"Document {document_id} already ingested")
    # Or update: repo.delete_document(document_id) then add
```

### Pattern 2: Chunked Search Results

```python
# Search and group by document
results = repo.search(query, top_k=20)

# Group by document
from collections import defaultdict
by_doc = defaultdict(list)

for result in results:
    by_doc[result.document_id].append(result)

# Show top document
for doc_id, doc_results in sorted(by_doc.items, key=lambda x: -len(x[1]))[:3]:
    print(f"\nDocument: {doc_id}")
    for result in doc_results[:2]:
        print(f"  - {result.content[:100]}... (score: {result.score:.2f})")
```

### Pattern 3: Incremental Indexing

```python
# Add documents incrementally without reloading
for file_path in pending_files:
    chunks = process_document(file_path)
    enriched = embedder.enrich_batch(chunks)
    repo.add_chunks(enriched)

    # Track progress
    total = repo.count_chunks
    print(f"Total chunks: {total}")
```

## Troubleshooting

### Issue 1: ChromaDB Import Error

**Symptom:** `ImportError: chromadb is required`

**Fix:**

```bash
pip install chromadb
```

### Issue 2: ChromaDB Persistence Not Working

**Symptom:** Chunks disappear after restart

**Cause:** Using ephemeral client instead of persistent

**Fix:**

```python
# ❌ Wrong - ephemeral
import chromadb
client = chromadb.Client

# ✅ Correct - persistent
client = chromadb.PersistentClient(path=".data/chromadb")
```

### Issue 3: JSONL Search Returns No Results

**Symptom:** `search` returns empty list

**Cause:** BM25 index not built or query mismatch

**Fix:**

```python
# Check index status
print(f"Indexed terms: {len(repo._term_index)}")
print(f"Chunks in memory: {len(repo._chunks)}")

# Rebuild index
repo._build_bm25_index
```

### Issue 4: Out of Memory with Large Datasets

**Symptom:** MemoryError when loading JSONL

**Cause:** JSONLRepository loads all chunks into memory

**Fix:**

```python
# Switch to ChromaDB for large datasets
# Or implement streaming for JSONL
```

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
- [ingestforge/retrieval/README.md](../retrieval/README.md) - Next: retrieval
- [ChromaDB docs](https://docs.trychroma.com/) - ChromaDB documentation
