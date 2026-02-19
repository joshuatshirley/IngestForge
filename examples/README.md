# IngestForge Examples

Real-world usage examples demonstrating IngestForge's capabilities for document processing, knowledge management, and RAG applications.

## Quick Links

- [Quick Start Examples](#quick-start-examples) - Get up and running in minutes
- [Academic Research](#academic-research) - Research paper analysis and management
- [Learning & Study](#learning--study) - Educational materials and study tools
- [Code Analysis](#code-analysis) - Codebase documentation and analysis
- [Knowledge Management](#knowledge-management) - Building searchable knowledge bases
- [Advanced Workflows](#advanced-workflows) - Complex multi-step pipelines

## Directory Structure

```
examples/
├── quickstart/              # Simple starter examples
│   ├── 01_basic_ingestion.py        # Ingest PDFs and print chunks
│   ├── 02_simple_search.py          # Store and search documents
│   ├── 03_generate_flashcards.py    # Create study materials
│   └── 04_code_documentation.py     # Document a codebase
├── academic/                # Research paper workflows
│   ├── arxiv_research_assistant.py  # Search and analyze papers
│   ├── literature_review.py         # Multi-paper analysis with citations
│   ├── paper_summarizer.py          # Generate paper summaries
│   └── concept_extraction.py        # Build concept maps
├── learning/                # Study and educational tools
│   ├── textbook_processor.py        # Convert textbooks to study materials
│   ├── flashcard_generator.py       # Auto-generate Anki flashcards
│   ├── quiz_builder.py              # Create practice quizzes
│   └── study_package.py             # Complete study folder generation
├── code/                    # Code analysis and documentation
│   ├── codebase_documenter.py       # Generate API documentation
│   ├── api_reference.py             # Extract API endpoints
│   ├── code_search.py               # Semantic code search
│   └── dependency_analyzer.py       # Analyze code dependencies
├── knowledge/               # Knowledge base workflows
│   ├── personal_wiki.py             # Build searchable knowledge base
│   ├── meeting_notes.py             # Process and link meeting notes
│   ├── research_database.py         # Cross-reference research materials
│   └── timeline_builder.py          # Create timelines from documents
├── advanced/                # Advanced workflows
│   ├── multi_source_integration.py  # Combine multiple data sources
│   ├── custom_enrichment.py         # Add custom enrichment pipeline
│   ├── vector_migration.py          # Migrate between storage backends
│   └── batch_processing.py          # Process large collections
├── data/                    # Sample input files
│   ├── sample.pdf
│   ├── sample.txt
│   ├── sample_code.py
│   └── research_paper.pdf
├── outputs/                 # Example output files
│   ├── flashcards.csv
│   ├── quiz.json
│   └── documentation.md
├── requirements.txt         # Dependencies for examples
└── README.md               # This file
```

## Quick Start Examples

Get started with basic IngestForge functionality in just a few lines of code.

### 1. Basic Ingestion

```bash
python examples/quickstart/01_basic_ingestion.py path/to/document.pdf
```

**What it does**: Loads a PDF, chunks it into manageable pieces, and prints the chunks with metadata.

**Useful for**: Understanding how IngestForge processes documents at a basic level.

### 2. Simple Search

```bash
python examples/quickstart/02_simple_search.py
```

**What it does**: Ingests sample documents and performs semantic search queries.

**Useful for**: Testing search functionality and understanding retrieval concepts.

### 3. Generate Flashcards

```bash
python examples/quickstart/03_generate_flashcards.py path/to/textbook.pdf
```

**What it does**: Extracts key concepts and generates flashcard-style Q&A pairs.

**Useful for**: Creating study materials from any document.

### 4. Code Documentation

```bash
python examples/quickstart/04_code_documentation.py src/ --output docs/
```

**What it does**: Analyzes source code and generates comprehensive documentation.

**Useful for**: Auto-documenting codebases.

## Academic Research

Tools for searching, analyzing, and managing research papers.

### ArXiv Research Assistant

```bash
python examples/academic/arxiv_research_assistant.py \
    --query "transformer architecture" \
    --max-results 10
```

**Features**:
- Search arXiv for papers
- Download and ingest papers
- Extract citations and references
- Generate summaries

### Literature Review

```bash
python examples/academic/literature_review.py \
    --papers-dir papers/ \
    --output review.md
```

**Features**:
- Multi-paper analysis
- Citation graph generation
- Thematic clustering
- Common themes extraction

### Paper Summarizer

```bash
python examples/academic/paper_summarizer.py path/to/paper.pdf
```

**Features**:
- Extract abstract and key sections
- Generate executive summary
- Identify key contributions
- List related work

### Concept Extraction

```bash
python examples/academic/concept_extraction.py papers/
```

**Features**:
- Extract domain concepts
- Build concept hierarchies
- Create concept maps
- Export to various formats

## Learning & Study

Educational tools for processing textbooks, creating study materials, and building quizzes.

### Textbook Processor

```bash
python examples/learning/textbook_processor.py \
    --textbook calculus.pdf \
    --output calculus_study/
```

**Produces**:
- Chapter summaries
- Key definitions
- Important theorems
- Practice problems

### Flashcard Generator

```bash
python examples/learning/flashcard_generator.py \
    --source material.pdf \
    --output flashcards.csv \
    --format anki
```

**Features**:
- Auto-generates Q&A pairs
- Exports to Anki format
- Creates spaced repetition decks
- Includes image support

### Quiz Builder

```bash
python examples/learning/quiz_builder.py \
    --content chapter.md \
    --num-questions 20 \
    --difficulty-levels beginner,intermediate,advanced
```

**Features**:
- Multiple choice question generation
- Difficulty level control
- Answer key generation
- Score calculation

### Study Package Generator

```bash
python examples/learning/study_package.py \
    --documents coursework/ \
    --output study_package.zip
```

**Produces**:
- Organized chapter structure
- Flashcards per chapter
- Chapter quizzes
- Glossary of terms
- Study schedule

## Code Analysis

Tools for analyzing, documenting, and searching codebases.

### Codebase Documenter

```bash
python examples/code/codebase_documenter.py \
    --source src/ \
    --output docs/ \
    --format markdown
```

**Generates**:
- Module documentation
- Class and function references
- Usage examples
- Architecture diagrams

### API Reference Extractor

```bash
python examples/code/api_reference.py \
    --module myapi.py \
    --output api_reference.md
```

**Extracts**:
- Public endpoints
- Request/response schemas
- Authentication methods
- Usage examples

### Code Search

```bash
python examples/code/code_search.py \
    --codebase src/ \
    --query "authentication logic" \
    --language python
```

**Features**:
- Semantic code search
- Function signature matching
- Cross-file references
- Usage pattern discovery

### Dependency Analyzer

```bash
python examples/code/dependency_analyzer.py src/ --output dependencies.json
```

**Analyzes**:
- Import dependencies
- Circular dependencies
- Dependency graphs
- Module statistics

## Knowledge Management

Tools for building and managing searchable knowledge bases.

### Personal Wiki

```bash
python examples/knowledge/personal_wiki.py \
    --documents notes/ \
    --output wiki.db
```

**Features**:
- Full-text search
- Bidirectional links
- Tagging system
- Export to HTML

### Meeting Notes Processor

```bash
python examples/knowledge/meeting_notes.py \
    --directory meetings/ \
    --output processed_notes.db
```

**Features**:
- Extract action items
- Link related meetings
- Build knowledge graph
- Generate meeting summaries

### Research Database

```bash
python examples/knowledge/research_database.py \
    --sources research_papers/ \
    --output research.db \
    --enable-citations
```

**Features**:
- Cross-reference materials
- Citation graph
- Theme clustering
- Export bibliography

### Timeline Builder

```bash
python examples/knowledge/timeline_builder.py \
    --documents historical_texts/ \
    --output timeline.json
```

**Generates**:
- Chronological timelines
- Event relationships
- Visual timeline data
- Interactive HTML

## Advanced Workflows

Complex multi-step pipelines and custom integrations.

### Multi-Source Integration

```bash
python examples/advanced/multi_source_integration.py \
    --sources pdfs/ emails/ videos/ \
    --output integrated_kb.db
```

**Combines**:
- Multiple file formats
- Different content types
- Various metadata sources
- Cross-source linking

### Custom Enrichment Pipeline

```bash
python examples/advanced/custom_enrichment.py \
    --input documents/ \
    --enrichers sentiment,ner,custom_classifier \
    --output enriched.db
```

**Features**:
- Pluggable enrichment modules
- Custom processors
- Conditional pipelines
- Quality scoring

### Vector Migration

```bash
python examples/advanced/vector_migration.py \
    --source-backend chromadb \
    --target-backend postgres \
    --source-path ./data/chroma \
    --target-url postgresql://...
```

**Supports**:
- ChromaDB to PostgreSQL
- PostgreSQL to JSONL
- Backup and restore
- Verification

### Batch Processing

```bash
python examples/advanced/batch_processing.py \
    --input-directory documents/ \
    --batch-size 100 \
    --workers 4 \
    --output results.db
```

**Features**:
- Parallel processing
- Progress tracking
- Error recovery
- Result aggregation

## Setup and Installation

### Requirements

- Python 3.10+
- IngestForge (installed and configured)
- Optional: API keys for external services

### Install Dependencies

```bash
# Install IngestForge
pip install ingestforge

# Install example dependencies
pip install -r examples/requirements.txt
```

### Environment Variables

Create a `.env` file in the `examples/` directory:

```env
# LLM Configuration
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional API Keys
ARXIV_API_KEY=optional
GOOGLE_SCHOLAR_TOKEN=optional

# Configuration
EXAMPLE_OUTPUT_DIR=./outputs
EXAMPLE_DATA_DIR=./data
```

## Running Examples

### Basic Usage

```bash
# Run with default settings
python examples/quickstart/01_basic_ingestion.py

# Run with custom arguments
python examples/quickstart/01_basic_ingestion.py document.pdf --output ./custom_output

# Run with verbose logging
INGESTFORGE_LOG_LEVEL=DEBUG python examples/quickstart/01_basic_ingestion.py
```

### With Docker

```bash
# Build example image
docker build -f examples/Dockerfile -t ingestforge-examples .

# Run example
docker run -e OPENAI_API_KEY=your_key \
    -v $(pwd)/examples/data:/data \
    -v $(pwd)/examples/outputs:/outputs \
    ingestforge-examples \
    python examples/quickstart/01_basic_ingestion.py /data/sample.pdf
```

## Example Outputs

Each example generates output demonstrating IngestForge's capabilities:

- **Chunks**: JSON with extracted text, metadata, and embeddings
- **Flashcards**: CSV format compatible with Anki
- **Quiz**: JSON format for web-based quizzes
- **Documentation**: Markdown or HTML documentation
- **Knowledge Base**: Searchable database
- **Timeline**: JSON for visualization

See `examples/outputs/` for sample output files.

## Common Patterns

### 1. Basic Workflow

```python
from ingestforge.ingest import DocumentProcessor
from ingestforge.chunking import SemanticChunker
from ingestforge.storage import JSONLStorage

# Process document
processor = DocumentProcessor
text = processor.process("document.pdf")

# Chunk the text
chunker = SemanticChunker
chunks = chunker.chunk(text)

# Store chunks
storage = JSONLStorage("output.jsonl")
storage.save(chunks)
```

### 2. Search Workflow

```python
from ingestforge.storage import JSONLStorage
from ingestforge.retrieval import SemanticRetriever

# Load chunks
storage = JSONLStorage("output.jsonl")
chunks = storage.load

# Search
retriever = SemanticRetriever(chunks)
results = retriever.retrieve("your query", k=5)

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.text}")
```

### 3. Enrichment Workflow

```python
from ingestforge.enrichment import EntityExtractor, SummaryGenerator
from ingestforge.storage import JSONLStorage

# Load chunks
storage = JSONLStorage("output.jsonl")
chunks = storage.load

# Enrich
entity_extractor = EntityExtractor
summary_gen = SummaryGenerator

for chunk in chunks:
    chunk.entities = entity_extractor.extract(chunk.text)
    chunk.summary = summary_gen.generate(chunk.text)

# Save enriched chunks
storage.save(chunks)
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'ingestforge'`
- **Solution**: Run `pip install -e .` in the IngestForge repo directory

**Issue**: `API key not found`
- **Solution**: Set environment variables or create `.env` file

**Issue**: `CUDA/GPU not available`
- **Solution**: Examples fall back to CPU automatically

**Issue**: Memory errors with large documents
- **Solution**: Use batch processing mode or increase chunk overlap

## Contributing Examples

Have a great example to share? Contributions welcome!

1. Create example script in appropriate directory
2. Add docstring explaining what it does
3. Include usage examples
4. Test with sample data
5. Submit pull request

## Resources

- [IngestForge Documentation](../README.md)
- [API Reference](../docs/api.md)
- [Configuration Guide](../docs/configuration.md)
- [Troubleshooting Guide](../docs/troubleshooting.md)

## License

These examples are part of IngestForge and follow the same license terms.
