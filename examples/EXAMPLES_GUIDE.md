# IngestForge Examples - Quick Reference Guide

## Overview

This guide provides quick navigation to all IngestForge examples with usage patterns, requirements, and expected outcomes.

## Examples by Category

### Quick Start (5-10 minutes)

Start here if you're new to IngestForge. These examples demonstrate core functionality with minimal setup.

#### 1. Basic Ingestion

**File**: `quickstart/01_basic_ingestion.py`

**What it does**: Loads a document (PDF, TXT, etc.), extracts text, and chunks it into manageable pieces.

**Time**: 5 minutes

**Usage**:
```bash
python examples/quickstart/01_basic_ingestion.py document.pdf
python examples/quickstart/01_basic_ingestion.py document.pdf --chunk-size 512 --overlap 50
```

**Output**:
- Chunk text samples
- Metadata (page numbers, word counts)
- Summary statistics

**Key imports**:
```python
from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker
```

---

#### 2. Simple Search

**File**: `quickstart/02_simple_search.py`

**What it does**: Builds a searchable knowledge base from documents and performs semantic queries.

**Time**: 10 minutes (plus indexing time)

**Usage**:
```bash
# Build knowledge base from documents
python examples/quickstart/02_simple_search.py --docs-dir examples/data --rebuild

# Search existing knowledge base
python examples/quickstart/02_simple_search.py --query "What is the main topic?"

# Interactive search
python examples/quickstart/02_simple_search.py --docs-dir examples/data
```

**Output**:
- Ranked search results
- Similarity scores
- Source metadata

**Key imports**:
```python
from ingestforge.storage.jsonl import JSONLStorage
from ingestforge.retrieval.semantic import SemanticRetriever
```

---

#### 3. Generate Flashcards

**File**: `quickstart/03_generate_flashcards.py`

**What it does**: Auto-generates question-answer flashcard pairs from any document.

**Time**: 5-15 minutes (depends on document size and LLM)

**Requirements**:
- OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable

**Usage**:
```bash
python examples/quickstart/03_generate_flashcards.py textbook.pdf
python examples/quickstart/03_generate_flashcards.py textbook.pdf --output my_flashcards.csv
python examples/quickstart/03_generate_flashcards.py --llm anthropic --questions-per-chunk 3
```

**Output**:
- CSV file with flashcards
- Front/back question pairs
- Difficulty levels
- Tagging information

**Key imports**:
```python
from ingestforge.enrichment.questions import QuestionGenerator
from ingestforge.llm.factory import LLMFactory
```

---

#### 4. Code Documentation

**File**: `quickstart/04_code_documentation.py`

**What it does**: Analyzes source code and generates comprehensive API documentation.

**Time**: 5-10 minutes

**Usage**:
```bash
python examples/quickstart/04_code_documentation.py ingestforge/
python examples/quickstart/04_code_documentation.py src/ --output docs/api.md
python examples/quickstart/04_code_documentation.py src/ --pattern "**/*.js"
```

**Output**:
- Markdown documentation
- Module structure
- Class hierarchies
- Function signatures

**Key imports**:
```python
import ast  # Python AST parsing
from ingestforge.enrichment.entities import EntityExtractor
```

---

### Academic Research (20-60 minutes)

Use these examples for research paper analysis and literature reviews.

#### 1. ArXiv Research Assistant

**File**: `academic/arxiv_research_assistant.py`

**What it does**: Search ArXiv, download papers, ingest them, and build a searchable knowledge base.

**Time**: 20-60 minutes (depends on paper count)

**Requirements**:
- `pip install arxiv` (no API key needed)

**Usage**:
```bash
# Search only
python examples/academic/arxiv_research_assistant.py --query "transformers" --max-results 10

# Search and download
python examples/academic/arxiv_research_assistant.py --query "attention mechanisms" --download --max-results 5

# Search, download, and index
python examples/academic/arxiv_research_assistant.py \
    --query "neural networks" \
    --download \
    --index \
    --bibliography papers.bib

# Resume from checkpoint
python examples/academic/arxiv_research_assistant.py \
    --query "machine learning" \
    --download \
    --index \
    --papers-dir papers_v2
```

**Output**:
- Downloaded PDF files
- Knowledge base with indexed papers
- Bibliography in BibTeX format
- Search results with citations

**Key imports**:
```python
import arxiv
```

---

#### 2. Literature Review Generator

**File**: `academic/literature_review.py`

**What it does**: Analyzes multiple papers to identify themes, concepts, and relationships.

**Time**: 15-30 minutes

**Usage**:
```bash
python examples/academic/literature_review.py --papers-dir papers/ --output review.md

python examples/academic/literature_review.py \
    --papers-dir papers/ \
    --output review.md \
    --graph citation_graph.json
```

**Output**:
- Markdown literature review document
- Common themes and concepts
- Citation relationship graph
- Research synthesis

**Features**:
- Thematic clustering
- Entity extraction from papers
- Cross-paper linking
- Theme frequency analysis

---

#### 3. Paper Summarizer (Stub)

**File**: `academic/paper_summarizer.py`

**What it does**: Extracts and summarizes key sections of research papers.

**Status**: Template provided, awaiting implementation

**Planned output**:
- Abstract and introduction
- Key findings summary
- Methodology overview
- Related work references

---

#### 4. Concept Extraction (Stub)

**File**: `academic/concept_extraction.py`

**What it does**: Builds concept maps from research papers.

**Status**: Template provided, awaiting implementation

**Planned output**:
- Concept hierarchy
- Relationship graphs
- Concept descriptions

---

### Learning & Study (15-45 minutes)

Create study materials from textbooks and educational documents.

#### 1. Textbook Processor

**File**: `learning/textbook_processor.py`

**What it does**: Converts textbooks into organized study materials with summaries, key terms, and concepts.

**Time**: 10-20 minutes

**Usage**:
```bash
python examples/learning/textbook_processor.py calculus.pdf
python examples/learning/textbook_processor.py book.pdf --output study_materials/
```

**Output**:
- Chapter-organized study materials
- Key terms glossary per chapter
- Key concepts lists
- Study guide with tips
- Metadata file

**Directory structure**:
```
study_materials/
├── chapter_01/
│   ├── text.txt
│   ├── key_terms.txt
│   ├── key_concepts.txt
│   └── metadata.json
├── chapter_02/
└── README.md
```

---

#### 2. Flashcard Generator (Advanced)

**File**: `learning/flashcard_generator.py`

**Status**: Template provided, awaiting implementation

**Planned features**:
- Multiple question types
- Spaced repetition scheduling
- Image support

---

#### 3. Quiz Builder (Stub)

**File**: `learning/quiz_builder.py`

**Status**: Template provided, awaiting implementation

**Planned features**:
- Difficulty-level control
- Multiple question types
- Answer keys and rubrics

---

#### 4. Study Package Generator (Stub)

**File**: `learning/study_package.py`

**Status**: Template provided, awaiting implementation

**Planned output**:
- Packaged study materials
- Integrated resources
- Study schedule

---

### Code Analysis (10-30 minutes)

Analyze and document source code.

#### 1. Code Search

**File**: `code/code_search.py`

**What it does**: Semantic search across a codebase using natural language queries.

**Time**: 5-10 minutes (first run includes indexing)

**Usage**:
```bash
# Index a codebase
python examples/code/code_search.py --codebase src/ --query "authentication logic"

# Interactive search
python examples/code/code_search.py --codebase src/

# Search pre-built index
python examples/code/code_search.py --index code_index.jsonl --query "error handling"

# Search specific language
python examples/code/code_search.py --codebase src/ --language javascript
```

**Supported languages**: python, javascript, java, csharp, cpp, go

**Output**:
- Relevant code snippets
- File locations
- Similarity scores
- Function/class context

---

#### 2. Codebase Documenter (Stub)

**File**: `code/codebase_documenter.py`

**Status**: Template provided, awaiting implementation

**Planned output**:
- Module documentation
- API references
- Architecture diagrams

---

#### 3. API Reference (Stub)

**File**: `code/api_reference.py`

**Status**: Template provided, awaiting implementation

**Planned output**:
- OpenAPI specification
- Endpoint documentation
- Usage examples

---

#### 4. Dependency Analyzer (Stub)

**File**: `code/dependency_analyzer.py`

**Status**: Template provided, awaiting implementation

**Planned output**:
- Dependency graphs
- Circular dependencies report
- Module coupling analysis

---

### Knowledge Management (15-30 minutes)

Build and search personal knowledge bases.

#### 1. Personal Wiki

**File**: `knowledge/personal_wiki.py`

**What it does**: Creates a searchable wiki from personal notes with full-text search, tagging, and linking.

**Time**: 5-15 minutes

**Usage**:
```bash
# Create wiki from notes
python examples/knowledge/personal_wiki.py --documents notes/ --database wiki.db

# Search wiki
python examples/knowledge/personal_wiki.py --search "machine learning" --database wiki.db

# Interactive search
python examples/knowledge/personal_wiki.py --documents notes/
```

**Output**:
- SQLite knowledge base
- Full-text search index
- Bidirectional links
- Tag index

**Features**:
- Wiki-style `[[Link]]` support
- `#hashtag` tagging
- Full-text search
- Backlink tracking

---

#### 2. Meeting Notes Processor (Stub)

**File**: `knowledge/meeting_notes.py`

**Status**: Template provided, awaiting implementation

**Planned features**:
- Action item extraction
- Decision logging
- Meeting linking

---

#### 3. Research Database (Stub)

**File**: `knowledge/research_database.py`

**Status**: Template provided, awaiting implementation

**Planned features**:
- Citation tracking
- Theme clustering
- Bibliography generation

---

#### 4. Timeline Builder (Stub)

**File**: `knowledge/timeline_builder.py`

**Status**: Template provided, awaiting implementation

**Planned output**:
- Chronological timelines
- Event relationships
- Interactive visualization

---

### Advanced Workflows (30-120 minutes)

Complex multi-step pipelines and integrations.

#### 1. Batch Processing

**File**: `advanced/batch_processing.py`

**What it does**: Parallel processing of large document collections with checkpointing and progress tracking.

**Time**: Depends on document count and system

**Usage**:
```bash
# Basic batch processing
python examples/advanced/batch_processing.py --input documents/ --output results.jsonl

# With parallel workers
python examples/advanced/batch_processing.py \
    --input documents/ \
    --batch-size 50 \
    --workers 8 \
    --output results.jsonl

# Resume from checkpoint
python examples/advanced/batch_processing.py \
    --input documents/ \
    --resume-from results.jsonl.checkpoint \
    --output results.jsonl
```

**Output**:
- JSONL file with all chunks
- Processing statistics
- Error report
- Checkpoint file

**Features**:
- Parallel processing with multiple workers
- Progress tracking
- Checkpoint and resume
- Error recovery
- Speed metrics

---

#### 2. Custom Enrichment Pipeline

**File**: `advanced/custom_enrichment.py`

**What it does**: Build custom enrichment pipelines combining sentiment analysis, NER, keywords, and more.

**Time**: 10-30 minutes

**Requirements**:
- Optional: `textblob` for sentiment, `spacy` for NER

**Usage**:
```bash
# With multiple enrichers
python examples/advanced/custom_enrichment.py \
    --input documents/ \
    --enrichers sentiment,ner,keywords,readability \
    --output enriched.jsonl

# With specific enrichers
python examples/advanced/custom_enrichment.py \
    --input documents/ \
    --enrichers sentiment,keywords \
    --output enriched.jsonl
```

**Available enrichers**:
- `sentiment`: Sentiment polarity and subjectivity
- `ner`: Named entity recognition
- `keywords`: Important keyword extraction
- `readability`: Readability metrics and grade level

**Output**:
- Chunks with enrichment metadata
- Quality scores
- Entity identification
- Sentiment labels

---

#### 3. Multi-Source Integration (Stub)

**File**: `advanced/multi_source_integration.py`

**Status**: Template provided, awaiting implementation

**Planned features**:
- Multiple file format support
- Cross-source linking
- Deduplication

---

#### 4. Vector Migration (Stub)

**File**: `advanced/vector_migration.py`

**Status**: Template provided, awaiting implementation

**Planned features**:
- Backend-to-backend migration
- Data validation
- Backup and restore

---

## Common Patterns

### Pattern 1: Document Processing Pipeline

```python
from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker
from ingestforge.storage.jsonl import JSONLStorage

# Process document
processor = DocumentProcessor
text = processor.process("document.pdf")

# Chunk it
chunker = SemanticChunker(target_size=512)
chunks = chunker.chunk(text)

# Store it
storage = JSONLStorage("output.jsonl")
storage.save(chunks)
```

### Pattern 2: Search and Retrieve

```python
from ingestforge.storage.jsonl import JSONLStorage
from ingestforge.retrieval.semantic import SemanticRetriever

# Load chunks
storage = JSONLStorage("output.jsonl")
chunks = storage.load

# Search
retriever = SemanticRetriever(chunks)
results = retriever.retrieve("your query", k=5)

# Print results
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
```

### Pattern 3: Enrichment

```python
from ingestforge.enrichment.entities import EntityExtractor
from ingestforge.enrichment.summary import SummaryGenerator

# Load chunks
chunks = storage.load

# Enrich
entity_extractor = EntityExtractor
summary_gen = SummaryGenerator

for chunk in chunks:
    chunk['entities'] = entity_extractor.extract(chunk['text'])
    chunk['summary'] = summary_gen.generate(chunk['text'])

# Save enriched chunks
storage.save(chunks)
```

---

## Environment Variables

Create a `.env` file in the `examples/` directory:

```env
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_GENERATIVEAI_API_KEY=...

# Configuration
INGESTFORGE_LOG_LEVEL=INFO
```

---

## Setup and Installation

### 1. Install IngestForge

```bash
# From the IngestForge repo
pip install -e .

# Or from PyPI (when published)
pip install ingestforge
```

### 2. Install Example Dependencies

```bash
pip install -r examples/requirements.txt
```

### 3. Set Up Environment

```bash
# Copy template
cp examples/.env.example examples/.env

# Edit with your API keys
nano examples/.env
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'ingestforge'`

**Solution**: Install IngestForge first:
```bash
cd /path/to/IngestForge
pip install -e .
```

### Issue: `API key not found`

**Solution**: Set environment variables:
```bash
export OPENAI_API_KEY=your_key
# Or add to examples/.env
```

### Issue: Memory errors with large documents

**Solution**: Use batch processing mode or increase chunk overlap.

### Issue: Slow embedding generation

**Solution**:
- Install GPU support: `pip install torch torchvision`
- Use `--workers` for parallel processing
- Reduce batch size

---

## Next Steps

1. **Start simple**: Run `quickstart/01_basic_ingestion.py`
2. **Try search**: Run `quickstart/02_simple_search.py`
3. **Explore your use case**: Find relevant examples above
4. **Adapt for your data**: Modify examples for your documents
5. **Combine approaches**: Build custom workflows from patterns

---

## Contributing Examples

Have a great example to share? Follow these steps:

1. Create the example script in appropriate `examples/<category>/` directory
2. Add comprehensive docstring with description, usage, and output
3. Include multiple usage examples showing different options
4. Test with sample data
5. Update `README.md` and `EXAMPLES_GUIDE.md`
6. Submit a pull request

---

## Resources

- [IngestForge Documentation](../README.md)
- [API Reference](../docs/api.md)
- [Configuration Guide](../docs/configuration.md)
- [FAQ](../docs/faq.md)

---

## License

All examples are part of IngestForge and follow the same license terms.
