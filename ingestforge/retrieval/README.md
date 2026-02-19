# Retrieval Module

## Purpose

Search chunks using BM25 (keyword), semantic (vector), or hybrid strategies. Includes reranking and parent document expansion for improved retrieval quality.

## Architecture Context

```
┌─────────────────────────────────────────┐
│   storage/ - Persisted chunks           │
│            ↓                             │
│   retrieval/ - Search strategies        │  ← You are here
│   (BM25, semantic, hybrid, rerank)      │
│            ↓                             │
│   query/ - Generate answers             │
└─────────────────────────────────────────┘
```

## Key Components

| Component | Purpose | Search Type |
|-----------|---------|-------------|
| `bm25.py` | BM25Retriever - keyword search | Lexical |
| `semantic.py` | SemanticRetriever - vector search | Semantic |
| `hybrid.py` | HybridRetriever - BM25 + semantic fusion | Hybrid |
| `reranker.py` | Reranker - rerank results with cross-encoder | Post-processing |
| `parent_retriever.py` | ParentRetriever - expand to parent chunks | Context expansion |

## Retrieval Strategies

### BM25Retriever

**Purpose:** Keyword-based search using BM25 algorithm (classic IR).

**Best For:**
- Exact keyword matches
- Technical terms, acronyms, names
- When semantic similarity might miss exact matches

**Usage:**

```python
from ingestforge.retrieval import BM25Retriever
from ingestforge.storage import get_storage_backend

repo = get_storage_backend(config)
retriever = BM25Retriever(repo)

# Search
results = retriever.search("quantum entanglement", top_k=5)

for result in results:
    print(f"BM25 Score: {result.score:.2f}")
    print(f"Content: {result.content[:100]}...")
```

### SemanticRetriever

**Purpose:** Vector similarity search using embeddings.

**Best For:**
- Conceptual similarity
- Paraphrased queries
- Cross-lingual search (with multilingual models)

**Usage:**

```python
from ingestforge.retrieval import SemanticRetriever

retriever = SemanticRetriever(repo, config)

# Semantic search
results = retriever.search("explain quantum phenomena", top_k=5)

for result in results:
    print(f"Similarity: {result.score:.2f}")
    print(f"Content: {result.content[:100]}...")
```

### HybridRetriever (Recommended)

**Purpose:** Combine BM25 and semantic search with weighted fusion.

**Best For:**
- General-purpose search
- Balance between keyword precision and semantic recall
- Production systems

**Usage:**

```python
from ingestforge.retrieval import HybridRetriever

retriever = HybridRetriever(repo, config)

# Hybrid search (configurable weights)
results = retriever.search("quantum computing", top_k=10)

# Each result has combined score from both methods
for result in results:
    print(f"Hybrid Score: {result.score:.2f}")
    print(f"Content: {result.content[:100]}...")
```

**Configuration:**

```yaml
# config.yaml
retrieval:
  strategy: hybrid       # bm25, semantic, hybrid
  top_k: 10
  rerank: true
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  hybrid:
    bm25_weight: 0.4     # 40% keyword
    semantic_weight: 0.6  # 60% semantic
```

**Fusion Algorithm:**

```python
# Reciprocal Rank Fusion (RRF)
def fuse_scores(bm25_results, semantic_results, k=60):
    scores = {}
    for rank, result in enumerate(bm25_results):
        scores[result.chunk_id] = 1.0 / (k + rank)

    for rank, result in enumerate(semantic_results):
        scores[result.chunk_id] += 1.0 / (k + rank)

    # Apply weights
    final_scores = {
        chunk_id: score * (bm25_weight + semantic_weight)
        for chunk_id, score in scores.items
    }
    return sorted(final_scores.items, key=lambda x: -x[1])
```

## Reranker

**Purpose:** Rerank top results using cross-encoder model for precision.

**Features:**
- More accurate than bi-encoder (embeddings)
- Slower - use only on top results (top 20-50)
- Considers query-document interaction

**Usage:**

```python
from ingestforge.retrieval import HybridRetriever, Reranker

# Initial retrieval
retriever = HybridRetriever(repo, config)
candidates = retriever.search("quantum computing", top_k=50)

# Rerank top candidates
reranker = Reranker(config)
reranked = reranker.rerank("quantum computing", candidates, top_k=5)

# Reranked results are more precise
for result in reranked:
    print(f"Rerank Score: {result.score:.3f}")  # 0-1, higher is better
    print(f"Content: {result.content[:100]}...")
```

**When to Use:**
- High-precision requirements
- Top-k is small (≤10)
- Latency acceptable (adds ~100-500ms)

**Model Options:**

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `ms-marco-MiniLM-L-6-v2` | Fast | Good | General (default) |
| `ms-marco-MiniLM-L-12-v2` | Medium | Better | Higher precision |
| `ms-marco-electra-base` | Slow | Best | Maximum accuracy |

## Parent Document Retrieval

**Purpose:** Expand small chunks to include parent context.

**Use Case:** Search returns focused chunk, but answer needs broader context.

**Usage:**

```python
from ingestforge.retrieval import create_parent_retriever

# Wrap existing retriever
base_retriever = HybridRetriever(repo, config)
parent_retriever = create_parent_retriever(base_retriever, config)

# Search with parent expansion
results = parent_retriever.search("definition of qubit", top_k=5)

for result in results:
    print(f"Child chunk: {result.child_content[:100]}...")
    print(f"Parent context: {result.parent_content[:100]}...")
    print(f"Combined: {result.combined_content[:200]}...")
```

**Parent Strategies:**

```python
# Strategy 1: Section-level parents
# Small chunks → section header chunk

# Strategy 2: Sliding window
# chunk[i] → chunk[i-1] as parent

# Strategy 3: Fixed size
# Expand to ±N chunks around match
```

## Usage Examples

### Example 1: Complete Retrieval Pipeline

```python
from ingestforge.core import load_config
from ingestforge.storage import get_storage_backend
from ingestforge.retrieval import HybridRetriever, Reranker

config = load_config
repo = get_storage_backend(config)

# 1. Hybrid retrieval (broad recall)
retriever = HybridRetriever(repo, config)
candidates = retriever.search("quantum entanglement", top_k=50)

# 2. Rerank for precision
if config.retrieval.rerank:
    reranker = Reranker(config)
    results = reranker.rerank("quantum entanglement", candidates, top_k=10)
else:
    results = candidates[:10]

# 3. Display results
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result.score:.3f}")
    print(f"   {result.content[:150]}...")
    print(f"   Source: {result.source_file}, p.{result.page_start}\n")
```

### Example 2: Multi-Strategy Comparison

```python
from ingestforge.retrieval import BM25Retriever, SemanticRetriever, HybridRetriever

query = "machine learning algorithms"

# Try all strategies
bm25 = BM25Retriever(repo).search(query, top_k=5)
semantic = SemanticRetriever(repo, config).search(query, top_k=5)
hybrid = HybridRetriever(repo, config).search(query, top_k=5)

print("BM25 Results:")
for r in bm25[:3]:
    print(f"  {r.chunk_id}: {r.score:.2f}")

print("\nSemantic Results:")
for r in semantic[:3]:
    print(f"  {r.chunk_id}: {r.score:.2f}")

print("\nHybrid Results:")
for r in hybrid[:3]:
    print(f"  {r.chunk_id}: {r.score:.2f}")
```

### Example 3: Query Expansion

```python
# Expand query with synonyms/related terms
query = "ML"
expanded_terms = ["machine learning", "ML", "artificial intelligence"]

# Search with expanded query
all_results = []
for term in expanded_terms:
    results = retriever.search(term, top_k=10)
    all_results.extend(results)

# Deduplicate by chunk_id
seen = set
unique_results = []
for result in sorted(all_results, key=lambda x: -x.score):
    if result.chunk_id not in seen:
        unique_results.append(result)
        seen.add(result.chunk_id)

# Top 5 after deduplication
final_results = unique_results[:5]
```

## Dependencies

### Required
- `sentence-transformers>=2.2.0` - Semantic search (embeddings)

### Optional
- `rank-bm25>=0.2.2` - BM25 implementation (lightweight fallback)

### Installation

```bash
# Minimal (semantic only)
pip install sentence-transformers

# With reranking
pip install sentence-transformers

# BM25 fallback
pip install rank-bm25
```

## Testing

```bash
# Run all retrieval tests
pytest tests/test_retrieval_*.py -v

# Test specific retriever
pytest tests/test_retrieval_hybrid.py -v
pytest tests/test_retrieval_reranker.py -v
```

## Common Patterns

### Pattern 1: Adaptive Top-K

```python
# Retrieve more candidates, rerank to fewer results
candidates = retriever.search(query, top_k=100)  # Broad recall
reranked = reranker.rerank(query, candidates, top_k=5)  # Precision
```

### Pattern 2: Fallback Strategy

```python
# Try semantic first, fallback to BM25 if no results
semantic_results = semantic_retriever.search(query, top_k=10)

if not semantic_results or semantic_results[0].score < 0.5:
    # Fallback to keyword search
    results = bm25_retriever.search(query, top_k=10)
else:
    results = semantic_results
```

## Troubleshooting

### Issue 1: Semantic Search Returns Empty

**Symptom:** SemanticRetriever returns no results

**Cause:** Chunks missing embeddings

**Fix:**

```python
# Check if chunks have embeddings
chunks = repo.get_chunks_by_document("doc-001")
has_embeddings = sum(1 for c in chunks if c.embedding)
print(f"Chunks with embeddings: {has_embeddings}/{len(chunks)}")

# Re-enrich if needed
from ingestforge.enrichment import EmbeddingGenerator
embedder = EmbeddingGenerator(config)
enriched = embedder.enrich_batch(chunks)
repo.add_chunks(enriched)
```

### Issue 2: Reranker Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory` during reranking

**Fix:**

```python
# Reduce candidate size
candidates = retriever.search(query, top_k=20)  # Instead of 100
reranked = reranker.rerank(query, candidates, top_k=5)

# Or use CPU
config.retrieval.rerank_model = "ms-marco-MiniLM-L-6-v2"  # Smaller model
```

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
- [ADR-002: Hybrid Retrieval](../../docs/architecture/ADR-002-hybrid-retrieval.md)
- [ingestforge/query/README.md](../query/README.md) - Next: query pipeline
