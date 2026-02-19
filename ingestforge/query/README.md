# Query Module

## Purpose

Query pipeline - orchestrate retrieval, generate LLM answers with citations, cache results. Combines search and generation into end-to-end Q&A system.

## Architecture Context

```
┌─────────────────────────────────────────┐
│   retrieval/ - Search chunks            │
│            ↓                             │
│   query/ - Generate answers             │  ← You are here
│   (pipeline, citations, cache)          │
│            ↓                             │
│   User receives answer + citations      │
└─────────────────────────────────────────┘
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `pipeline.py` | QueryPipeline - orchestrate query→retrieval→generation→citations |
| `classifier.py` | QueryClassifier - classify query type (factual, procedural, etc.) |
| `expander.py` | QueryExpander - expand query with synonyms |
| `cache.py` | QueryCache - cache query results |

## QueryPipeline

**Purpose:** End-to-end pipeline from query to cited answer.

**Workflow:**
1. Classify query type
2. Expand query (optional)
3. Retrieve relevant chunks
4. Generate answer with LLM
5. Build citations
6. Cache result

**Usage:**

```python
from ingestforge.query import QueryPipeline
from ingestforge.core import load_config
from ingestforge.storage import get_storage_backend
from ingestforge.retrieval import HybridRetriever

config = load_config
storage = get_storage_backend(config)
retriever = HybridRetriever(storage, config)

# Create pipeline
pipeline = QueryPipeline(
    config=config,
    retriever=retriever,
)

# Run query
result = pipeline.query("What is quantum entanglement?")

# Display answer
print(result.answer)
print("\nSources:")
for citation in result.citations:
    print(f"  [{citation.index}] {citation.text}")
```

**QueryResult Structure:**

```python
@dataclass
class QueryResult:
    query: str
    answer: str
    citations: List[Citation]
    chunks: List[SearchResult]  # Retrieved chunks
    processing_time: float
    cache_hit: bool
```

## Citations

**Purpose:** Generate academic-style citations for answers.

**Citation Formats:**

```python
# Short inline citation
citation = "[Smith 2023, p.47]"

# Full bibliography
citation = "Smith, J. (2023). Quantum Computing. MIT Press."

# APA, MLA, Chicago supported
```

**Usage:**

```python
# Citations included in query result
result = pipeline.query("...")

# Format citations
for i, citation in enumerate(result.citations, 1):
    print(f"[{i}] {citation.format('apa')}")
```

## Query Classification

**Purpose:** Classify queries to optimize retrieval strategy.

**Query Types:**
- `factual` - "What is X?" → precise retrieval
- `procedural` - "How to X?" → step-by-step chunks
- `comparative` - "X vs Y?" → multiple documents
- `exploratory` - "Tell me about X" → broad retrieval

**Usage:**

```python
from ingestforge.query import QueryClassifier

classifier = QueryClassifier

query_type = classifier.classify("What is quantum computing?")
# Output: "factual"

query_type = classifier.classify("How do I set up Docker?")
# Output: "procedural"
```

## Query Expansion

**Purpose:** Expand query with synonyms and related terms.

**Usage:**

```python
from ingestforge.query import QueryExpander

expander = QueryExpander

expanded = expander.expand("ML algorithms")
# Output: ["ML algorithms", "machine learning algorithms", "ML models"]

# Use in search
all_results = []
for term in expanded:
    results = retriever.search(term, top_k=5)
    all_results.extend(results)
```

## Query Cache

**Purpose:** Cache query results to avoid redundant processing.

**Features:**
- LRU eviction
- TTL expiration
- Configurable size
- Persistent storage

**Usage:**

```python
from ingestforge.query.cache import QueryCache

cache = QueryCache(max_size=1000, ttl_seconds=3600)

# Check cache
cached = cache.get("What is quantum computing?")
if cached:
    return cached

# Cache miss - process query
result = pipeline.query("What is quantum computing?")
cache.set("What is quantum computing?", result)
```

## Usage Examples

### Example 1: Complete Q&A System

```python
from ingestforge.core import load_config
from ingestforge.storage import get_storage_backend
from ingestforge.retrieval import HybridRetriever
from ingestforge.query import QueryPipeline

config = load_config
storage = get_storage_backend(config)
retriever = HybridRetriever(storage, config)
pipeline = QueryPipeline(config, retriever)

# Interactive Q&A
while True:
    query = input("\nQuestion: ")
    if query.lower in ['exit', 'quit']:
        break

    result = pipeline.query(query)

    print(f"\nAnswer: {result.answer}")
    print(f"\nSources:")
    for i, citation in enumerate(result.citations, 1):
        print(f"  [{i}] {citation.source_file}, p.{citation.page_start}")
    print(f"\n(Answered in {result.processing_time:.2f}s)")
```

### Example 2: Batch Query Processing

```python
questions = [
    "What is quantum computing?",
    "How do qubits work?",
    "What are quantum gates?"
]

results = []
for question in questions:
    result = pipeline.query(question)
    results.append(result)

# Export to markdown
with open("answers.md", "w") as f:
    for i, result in enumerate(results, 1):
        f.write(f"## Q{i}: {result.query}\n\n")
        f.write(f"{result.answer}\n\n")
        f.write("### Sources\n\n")
        for citation in result.citations:
            f.write(f"- {citation.format('apa')}\n")
        f.write("\n---\n\n")
```

## Dependencies

### Required
- `ingestforge.retrieval` - Search
- `ingestforge.llm` - Answer generation

### Optional
- `nltk` - Query expansion (synonyms)

## Testing

```bash
pytest tests/test_query_*.py -v
```

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
- [ADR-004: Citations](../../docs/architecture/ADR-004-citation-provenance.md)
- [ingestforge/llm/README.md](../llm/README.md) - LLM integration
