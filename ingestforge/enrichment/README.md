# Enrichment Module

## Purpose

Add metadata to chunks - embeddings for semantic search, named entities, hypothetical questions, and quality scores. All enrichers implement `IEnricher` interface for composable pipelines.

## Architecture Context

```
┌─────────────────────────────────────────┐
│   chunking/ - Create chunks             │
│            ↓                             │
│   enrichment/ - Add metadata            │  ← You are here
│   (embeddings, entities, questions)     │
│            ↓                             │
│   storage/ - Save to vector DB          │
└─────────────────────────────────────────┘
```

## Key Components

| Component | Purpose | Implements IEnricher |
|-----------|---------|---------------------|
| `embeddings.py` | EmbeddingGenerator - vector embeddings | ✅ Yes |
| `entities.py` | EntityExtractor - named entity recognition | ✅ Yes |
| `questions.py` | QuestionGenerator - hypothetical questions | ✅ Yes |
| `metadata.py` | MetadataEnricher - domain metadata | ✅ Yes |
| `benchmark.py` | Performance benchmarking | No (utility) |

## EmbeddingGenerator

**Purpose:** Generate vector embeddings for semantic search using sentence-transformers.

**Features:**
- GPU acceleration (CUDA)
- Automatic batch size optimization based on VRAM
- CPU fallback
- Progress tracking
- Retry logic for transient failures

**Usage:**

```python
from ingestforge.enrichment import EmbeddingGenerator
from ingestforge.core import load_config

config = load_config
embedder = EmbeddingGenerator(config)

# Check availability
if embedder.is_available:
    # Enrich single chunk
    chunk = embedder.enrich_chunk(chunk)
    print(f"Embedding: {len(chunk.embedding)} dimensions")

    # Batch processing (recommended)
    enriched = embedder.enrich_batch(chunks, batch_size=32)
```

**Auto Batch Sizing:**

```python
from ingestforge.enrichment import get_batch_size_recommendation

batch_size, reason = get_batch_size_recommendation(config)
print(f"Recommended batch size: {batch_size} ({reason})")

# VRAM-based recommendations:
# 2 GB  → 8
# 4 GB  → 16
# 8 GB  → 64
# 16 GB → 128
# CPU   → 32
```

**Configuration:**

```yaml
# config.yaml
enrichment:
  generate_embeddings: true
  embedding_model: "all-MiniLM-L6-v2"  # 384 dimensions, fast
  # Alternative models:
  # "all-mpnet-base-v2"   # 768 dims, more accurate
  # "multi-qa-MiniLM-L6"  # Optimized for Q&A
```

## EntityExtractor

**Purpose:** Extract named entities (people, organizations, dates, etc.) using pattern matching or optional spaCy.

**Features:**
- Pattern-based extraction (no dependencies)
- Optional spaCy integration (better accuracy)
- Multiple entity types
- Deduplication

**Usage:**

```python
from ingestforge.enrichment.entities import EntityExtractor

# Pattern-based (lightweight)
extractor = EntityExtractor(use_spacy=False)

# spaCy-based (more accurate)
extractor = EntityExtractor(use_spacy=True)

# Extract from chunk
chunk = extractor.enrich_chunk(chunk)
print(f"Entities: {chunk.entities}")

# Batch processing
enriched = extractor.enrich_batch(chunks)
```

**Entity Types Detected:**

| Type | Pattern-Based | spaCy-Based |
|------|--------------|-------------|
| Person | Names (John Smith) | PERSON |
| Organization | Companies (Apple Inc.) | ORG |
| Date | Jan 15, 2024 | DATE |
| Money | $1,234.56 | MONEY |
| Percentage | 45.2% | PERCENT |
| Email | user@example.com | ✓ |
| URL | https://... | ✓ |
| Phone | (555) 123-4567 | ✓ |
| Location | - | GPE, LOC |

**Entity Format:**

```python
# Pattern-based
chunk.entities = [
    "John Smith",
    "Microsoft Corporation",
    "Jan 15, 2024",
    "$500,000"
]

# spaCy-based (includes types)
chunk.entities = [
    "PERSON:John Smith",
    "ORG:Microsoft Corporation",
    "DATE:Jan 15, 2024",
    "MONEY:$500,000"
]
```

## QuestionGenerator

**Purpose:** Generate hypothetical questions that chunk content answers (improves retrieval).

**Features:**
- LLM-based generation (Gemini, Claude, OpenAI)
- Template fallback if LLM unavailable
- Configurable number of questions
- Content-type aware templates

**Usage:**

```python
from ingestforge.enrichment.questions import QuestionGenerator

generator = QuestionGenerator(config)

# Generate questions for chunk
chunk = generator.enrich_chunk(chunk, num_questions=3)
print(f"Questions: {chunk.hypothetical_questions}")

# Output:
# [
#   "What is quantum computing?",
#   "How do qubits differ from classical bits?",
#   "What are the main applications of quantum computing?"
# ]
```

**Configuration:**

```yaml
# config.yaml
enrichment:
  generate_questions: true

llm:
  default_provider: gemini  # or claude, openai
```

**Template Fallback:**

If LLM unavailable, uses templates based on chunk type:

```python
# Definition chunks
"What is {section_title}?"
"How is {section_title} defined?"

# Procedure chunks
"How do you {action}?"
"What are the steps to {action}?"
```

## EnrichmentPipeline Composition

Combine multiple enrichers in sequence:

```python
from ingestforge.shared.patterns import EnrichmentPipeline
from ingestforge.enrichment import (
    EntityExtractor,
    QuestionGenerator,
    EmbeddingGenerator
)

# Create pipeline
pipeline = EnrichmentPipeline([
    EntityExtractor,
    QuestionGenerator(config),
    EmbeddingGenerator(config),
], skip_unavailable=True)

# Apply all enrichers
enriched_chunks = pipeline.enrich(chunks, batch_size=32)

# Get summary
summary = pipeline.get_summary
print(f"Applied: {summary['active_enrichers']}/{summary['total_enrichers']}")
```

## Implementing IEnricher

Template for custom enrichers:

```python
from ingestforge.shared.patterns import IEnricher
from ingestforge.chunking import ChunkRecord

class SentimentEnricher(IEnricher):
    """Add sentiment analysis to chunks."""

    def enrich_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich single chunk."""
        # Compute sentiment
        sentiment = analyze_sentiment(chunk.content)

        # Add to chunk
        chunk.metadata = chunk.metadata or {}
        chunk.metadata['sentiment'] = sentiment

        return chunk

    def is_available(self) -> bool:
        """Check if enricher can run."""
        try:
            import sentiment_analyzer
            return True
        except ImportError:
            return False

# Use it
enricher = SentimentEnricher
if enricher.is_available:
    enriched = enricher.enrich_batch(chunks)
```

## Performance Benchmarking

Compare embedding models and batch sizes:

```python
from ingestforge.enrichment import run_comparison_benchmark, print_benchmark_report

# Benchmark different configurations
results = run_comparison_benchmark(
    chunks=sample_chunks[:100],
    models=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    batch_sizes=[8, 16, 32, 64],
)

# Print report
print_benchmark_report(results)

# Output:
# Model: all-MiniLM-L6-v2, Batch: 32
#   Time: 2.34s
#   Throughput: 42.7 chunks/sec
#   Memory: 1.2 GB
```

## Usage Examples

### Example 1: Full Enrichment Pipeline

```python
from ingestforge.core import load_config
from ingestforge.shared.patterns import EnrichmentPipeline
from ingestforge.enrichment import (
    EntityExtractor,
    QuestionGenerator,
    EmbeddingGenerator
)

config = load_config

# Build pipeline
enrichers = []

if config.enrichment.extract_entities:
    enrichers.append(EntityExtractor)

if config.enrichment.generate_questions:
    enrichers.append(QuestionGenerator(config))

if config.enrichment.generate_embeddings:
    enrichers.append(EmbeddingGenerator(config))

# Create pipeline
pipeline = EnrichmentPipeline(enrichers, skip_unavailable=True)

# Apply enrichments
enriched_chunks = pipeline.enrich(chunks)
```

### Example 2: Batch Processing with Progress

```python
from ingestforge.enrichment import EmbeddingGenerator
from tqdm import tqdm

embedder = EmbeddingGenerator(config)

# Process in batches with progress bar
batch_size = 32
all_enriched = []

for i in tqdm(range(0, len(chunks), batch_size), desc="Enriching"):
    batch = chunks[i:i + batch_size]
    enriched = embedder.enrich_batch(batch, batch_size=batch_size)
    all_enriched.extend(enriched)
```

### Example 3: Conditional Enrichment

```python
# Enrich only high-quality chunks
from ingestforge.enrichment import EmbeddingGenerator

embedder = EmbeddingGenerator(config)

for chunk in chunks:
    # Skip low-quality chunks
    if chunk.quality_score < 0.7:
        continue

    # Enrich high-quality chunks
    chunk = embedder.enrich_chunk(chunk)
```

## Dependencies

### Required
- `sentence-transformers>=2.2.0` - Embedding generation
- `torch>=2.0.0` - PyTorch (backend for sentence-transformers)

### Optional
- `spacy>=3.5.0` - Named entity recognition
- `en_core_web_sm` - spaCy English model (`python -m spacy download en_core_web_sm`)

### Installation

```bash
# Minimal (embeddings only, CPU)
pip install sentence-transformers

# With GPU support
pip install sentence-transformers torch

# With entity extraction
pip install spacy
python -m spacy download en_core_web_sm
```

## Testing

```bash
# Run all enrichment tests
pytest tests/test_enrichment_*.py -v

# Test specific enricher
pytest tests/test_enrichment_embeddings.py -v
pytest tests/test_enrichment_entities.py -v

# Test with coverage
pytest tests/test_enrichment_*.py --cov=ingestforge.enrichment
```

## Common Patterns

### Pattern 1: Enrichment with Error Handling

```python
from ingestforge.shared.patterns import EnrichmentPipeline

pipeline = EnrichmentPipeline(enrichers, skip_unavailable=True)

# Will continue even if enrichers fail
enriched = pipeline.enrich(chunks)

# Check what succeeded
summary = pipeline.get_summary
for enricher_info in summary['enrichers']:
    print(f"{enricher_info['name']}: {enricher_info['available']}")
```

### Pattern 2: Selective Enrichment

```python
# Enrich based on chunk type
for chunk in chunks:
    if chunk.chunk_type == "definition":
        chunk = question_gen.enrich_chunk(chunk, num_questions=5)
    else:
        chunk = question_gen.enrich_chunk(chunk, num_questions=2)

    # Always add embeddings
    chunk = embedder.enrich_chunk(chunk)
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Fix:**

```python
# Reduce batch size
batch_size, _ = get_batch_size_recommendation(config)
batch_size = batch_size // 2  # Halve recommended size

# Or force CPU
import torch
torch.cuda.is_available = lambda: False
```

### Issue 2: spaCy Model Not Found

**Symptom:** `OSError: Can't find model 'en_core_web_sm'`

**Fix:**

```bash
python -m spacy download en_core_web_sm
```

### Issue 3: Slow LLM Question Generation

**Symptom:** QuestionGenerator takes 5+ seconds per chunk

**Fix:**

```python
# Use template fallback
generator = QuestionGenerator(config)
generator.llm_client = None  # Force templates

# Or reduce number of questions
chunk = generator.enrich_chunk(chunk, num_questions=1)
```

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
- [ADR-005: IEnricher Interface](../../docs/architecture/ADR-005-ienricher-interface.md)
- [ingestforge/shared/README.md](../shared/README.md) - IEnricher interface
- [ingestforge/storage/README.md](../storage/README.md) - Next: storage
