# Developer Onboarding Guide

## Welcome to IngestForge! 🎯

This guide will take you from **zero to contributing in 15 minutes**.

By the end of this guide, you'll:
- ✅ Have IngestForge running locally
- ✅ Understand the architecture
- ✅ Know where key code lives
- ✅ Be ready for your first contribution

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.10+** (`python --version`)
- [ ] **Git** (`git --version`)
- [ ] **Your favorite IDE** (VSCode recommended)
- [ ] **10-15 minutes** of focused time

Optional but recommended:
- [ ] **GPU with CUDA** (for faster embeddings)
- [ ] **10GB+ free disk space** (for models and data)

---

## The 15-Minute Onboarding Path

### Minutes 0-5: Setup and Verification

#### 1. Clone and Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/ingestforge.git
cd ingestforge

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

**Expected output:**
```
Successfully installed ingestforge-0.1.0
✓ Development dependencies installed
```

#### 2. Verify Installation (1 minute)

```bash
# Check CLI is installed
ingestforge --version

# Run quick test
pytest tests/test_config_validation.py -v

# Check imports work
python -c "from ingestforge.core import load_config; print('✓ Core imports work')"
```

**Expected output:**
```
ingestforge version 0.1.0
✓ Core imports work
test_config_validation.py::test_config_load PASSED
```

#### 3. Quick Start Demo (2 minutes)

```bash
# Initialize test project
ingestforge init --name onboarding-test --with-sample

# Process sample document
ingestforge ingest .ingest/pending/sample_document.md

# Try a query
ingestforge query "What is IngestForge?"

# Check status
ingestforge status
```

**Expected output:**
```
✓ Project initialized: onboarding-test
✓ Sample document created

✓ Processing sample_document.md
  ├─ Created 15 chunks
  ├─ Generated embeddings
  └─ Stored in ChromaDB

Question: What is IngestForge?
Answer: IngestForge is a document processing and RAG framework...

Sources:
  [1] sample_document.md, p.1

Documents: 1, Chunks: 15
```

**🎉 Success!** If you got this far, IngestForge is working. You're ready to explore the codebase.

---

### Minutes 5-10: Architecture Understanding

Now that it works, let's understand **how** it works.

#### Read These (in order)

**1. [README.md](../../README.md) - What IngestForge does (3 min)**
- User-facing features
- Quick start guide
- When to use IngestForge

**2. [ARCHITECTURE.md](../../ARCHITECTURE.md) - How it works (5 min)**
- Hexagonal architecture (core → shared → features)
- Data flow: Document → Chunks → Enrichment → Storage → Retrieval
- Key design decisions

**3. [REFACTORING.md](../../REFACTORING.md) - Recent improvements (2 min)**
- Phase 1-3 refactoring completed
- Interface standardization (IEnricher, IProcessor, IChunkingStrategy)
- Code duplication eliminated

#### Architecture Quick Reference

```
User drops document.pdf
         ↓
[ingest/] PDFProcessor extracts text
         ↓
[chunking/] SemanticChunker splits into chunks
         ↓
[enrichment/] EnrichmentPipeline adds vectors, entities, questions
         ↓
[storage/] ChromaDBRepository stores chunks
         ↓
User queries "What is X?"
         ↓
[retrieval/] HybridRetriever finds relevant chunks
         ↓
[query/] QueryPipeline generates answer with citations
         ↓
User gets answer + sources
```

**Key Concepts:**

- **Hexagonal Architecture**: Core has no dependencies on features; features depend on core/shared
- **Interface-Based Design**: IEnricher, IProcessor, IChunkingStrategy enable swappable implementations
- **Provenance Tracking**: Every chunk knows its source for academic citations

---

### Minutes 10-15: Code Navigation Tour

Follow this **code tour** to see how everything connects.

#### Tour Stop 1: Entry Point (CLI)

**File:** `ingestforge/cli/main.py:45-80`

```python
@app.command
def ingest(file_path: Path):
    """Ingest a document."""
    pipeline = get_pipeline
    result = pipeline.process_document(file_path)
    # ↑ This kicks off the entire processing pipeline
```

**What happens:** User command → Pipeline orchestration

#### Tour Stop 2: Pipeline Orchestration

**File:** `ingestforge/core/pipeline.py:50-120`

```python
def process_document(self, file_path: Path):
    # 1. Process document (extract text)
    doc = self.processor.process(file_path)

    # 2. Chunk text
    chunks = self.chunker.chunk(doc.texts[0])

    # 3. Enrich chunks (embeddings, entities, questions per config flags)
    enriched = self.enricher.enrich_batch(chunks)

    # 4. Store chunks
    self.storage.add_chunks(enriched)

    return result
```

**What happens:** Orchestrates the 4 main stages

#### Tour Stop 3: Interface Implementation Example

**File:** `ingestforge/shared/patterns/enricher.py:14-56`

```python
class IEnricher(ABC):
    """Interface for chunk enrichers."""

    @abstractmethod
    def enrich_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich a single chunk."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if enricher is ready."""
        pass
```

**File:** `ingestforge/enrichment/entities.py:15-75`

```python
class EntityExtractor(IEnricher):
    """Extract named entities."""

    def enrich_chunk(self, chunk):
        entities = self.extract(chunk.content)
        chunk.entities = entities
        return chunk

    def is_available(self):
        return True  # Pattern-based, always available
```

**What happens:** Interface defines contract, implementations provide behavior

#### Tour Stop 4: Data Flow (Chunk Lifecycle)

**File:** `ingestforge/chunking/semantic_chunker.py:18-60`

```python
@dataclass
class ChunkRecord:
    """A chunk with full provenance."""
    chunk_id: str
    content: str
    # ... metadata fields ...
    source_location: Optional[SourceLocation]  # For citations
    embedding: Optional[List[float]]  # Added by enrichment
    entities: Optional[List[str]]  # Added by enrichment
```

**What happens:** Chunks carry all metadata through the pipeline

#### Tour Stop 5: Integration Test

**File:** `tests/test_enrichment_integration.py:15-50`

```python
def test_enrichment_pipeline:
    """See how everything works together."""
    # Create chunks
    chunks = SemanticChunker(config).chunk(text, "doc-001")

    # Build enrichment pipeline
    pipeline = EnrichmentPipeline([
        EntityExtractor,
        EmbeddingGenerator(config),
    ])

    # Enrich chunks
    enriched = pipeline.enrich(chunks)

    # Verify enrichments were applied
    assert enriched[0].entities is not None
    assert enriched[0].embedding is not None
```

**What happens:** Full pipeline in action

---

## After 15 Minutes: Next Steps

**You should now know:**
- ✅ What IngestForge does (RAG framework)
- ✅ How it works (document → chunks → enrichment → storage → retrieval)
- ✅ Where key code lives (module READMEs)
- ✅ How to run tests and use the CLI
- ✅ Interface-based architecture pattern

**Ready for your first contribution!** 🚀

---

## Your First Contribution

Choose one option based on your interests:

### Option 1: Fix a Bug (Good First Issue)

Browse [good first issues](https://github.com/yourusername/ingestforge/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22):

```bash
# Example: Fix a typo in documentation
git checkout -b fix/typo-in-readme
# Make your changes
git add .
git commit -m "docs: fix typo in README.md"
git push origin fix/typo-in-readme
# Create PR
```

### Option 2: Add a Test

Pick a module with <80% coverage and add tests:

```bash
# Check coverage
pytest tests/ --cov=ingestforge --cov-report=html
open htmlcov/index.html  # View coverage report

# Find untested function
# Write test in tests/test_[module]_[feature].py
pytest tests/test_your_new_test.py -v
```

**Example test:**

```python
# tests/test_chunking_semantic.py
def test_semantic_chunker_preserves_metadata:
    """Verify metadata is preserved during chunking."""
    config = load_config
    chunker = SemanticChunker(config)

    metadata = {"author": "Smith", "year": 2023}
    chunks = chunker.chunk(
        text="Test content",
        document_id="test-001",
        metadata=metadata
    )

    assert chunks[0].metadata == metadata
```

### Option 3: Improve Documentation

Pick a module README and:
- Add a missing example
- Clarify a confusing section
- Add a troubleshooting entry
- Fix broken links

```bash
# Example
git checkout -b docs/improve-enrichment-readme
# Edit ingestforge/enrichment/README.md
git add ingestforge/enrichment/README.md
git commit -m "docs: add example for custom enricher"
git push origin docs/improve-enrichment-readme
```

### Option 4: Add a Small Feature

Implement one of these starter features:

**Easy:**
- Add a new text cleaning pattern to `shared/text_utils.py`
- Add a new entity pattern to `enrichment/entities.py`
- Add a new CLI flag to existing command

**Medium:**
- Implement a new chunking strategy (e.g., SlidingWindowChunker)
- Add a new document processor (e.g., RTF, ODT)
- Implement a new enricher (e.g., SentimentEnricher)

See [Feature Template](FEATURE_TEMPLATE.md) for step-by-step guide.

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-feature
# or fix/bug-description
# or docs/documentation-update
```

**Branch naming:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test additions

### 2. Make Changes

Edit code following existing patterns:

```python
# Good: Follow interface pattern
class MyEnricher(IEnricher):
    def enrich_chunk(self, chunk):
        # Implementation
        return chunk

    def is_available(self):
        return True

# Good: Use existing utilities
from ingestforge.shared import clean_text, lazy_property
from ingestforge.core.logging import get_logger

# Good: Add type hints
def process_chunks(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
    ...
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_enrichment_*.py -v

# Run with coverage
pytest tests/ --cov=ingestforge --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"
```

### 4. Check Code Quality

```bash
# Format code (if you have black installed)
black ingestforge/ tests/

# Check types (if you have mypy installed)
mypy ingestforge/

# Check style (if you have flake8 installed)
flake8 ingestforge/
```

### 5. Update Documentation

If you:
- **Added a feature** → Update module README + CHANGELOG.md
- **Changed an interface** → Update ADR + affected READMEs
- **Fixed a bug** → Add to troubleshooting section

### 6. Commit Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>: <description>

# Examples:
git commit -m "feat: add sentiment analysis enricher"
git commit -m "fix: handle empty chunks in embedder"
git commit -m "docs: add example for custom processor"
git commit -m "test: add coverage for reranker"
git commit -m "refactor: extract common retry logic"
```

**Commit types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Tests only
- `refactor:` - Code restructuring
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

### 7. Push and Create PR

```bash
git push origin feature/my-feature

# Create PR on GitHub
# Use PR template to describe changes
```

**PR Title Format:**
```
feat: add sentiment analysis enricher

Implements SentimentEnricher using VADER for sentiment scoring.
Follows IEnricher interface and includes tests.

Closes #123
```

---

## IDE Setup

### VSCode (Recommended)

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.analysis.typeCheckingMode": "basic",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        "htmlcov": true,
        ".coverage": true
    }
}
```

**Recommended Extensions:**
- Python (Microsoft)
- Pylance (Microsoft)
- GitLens
- Better Comments
- Error Lens

**Keyboard Shortcuts:**
- `F5` - Run/Debug
- `Ctrl+Shift+P` → "Python: Run All Tests"
- `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv`

### PyCharm

1. **Open Project:** File → Open → Select `ingestforge/` folder
2. **Set Interpreter:** Settings → Project → Python Interpreter → Add → Existing → `.venv/bin/python`
3. **Enable pytest:** Settings → Tools → Python Integrated Tools → Testing → pytest
4. **Configure Black:** Settings → Tools → Black → Enable "Run on Save"

**Run Configuration for CLI:**
```
Script path: ingestforge/cli/main.py
Parameters: query "test query"
Working directory: /path/to/test/project
```

### Vim/Neovim

Install plugins:
- `python-mode` or `coc-python`
- `ale` for linting
- `vim-test` for running tests

Add to `.vimrc`:
```vim
" Use project virtualenv
let g:python3_host_prog = '.venv/bin/python'

" Run tests
nmap <silent> <leader>t :TestNearest<CR>
nmap <silent> <leader>T :TestFile<CR>
```

---

## Common Development Tasks

### Adding a New Enricher

**Step-by-step:**

1. **Create file:** `ingestforge/enrichment/sentiment.py`

```python
from ingestforge.shared.patterns import IEnricher
from ingestforge.chunking import ChunkRecord

class SentimentEnricher(IEnricher):
    """Add sentiment analysis to chunks."""

    def enrich_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        # Compute sentiment
        sentiment = self._analyze(chunk.content)
        chunk.metadata = chunk.metadata or {}
        chunk.metadata['sentiment'] = sentiment
        return chunk

    def is_available(self) -> bool:
        return True

    def _analyze(self, text: str) -> str:
        # Your implementation
        return "neutral"
```

2. **Add tests:** `tests/test_enrichment_sentiment.py`

```python
from ingestforge.enrichment.sentiment import SentimentEnricher
from ingestforge.chunking import ChunkRecord

def test_sentiment_enricher:
    enricher = SentimentEnricher
    chunk = ChunkRecord(
        chunk_id="test",
        document_id="doc",
        content="This is great!"
    )

    enriched = enricher.enrich_chunk(chunk)
    assert enriched.metadata['sentiment'] in ['positive', 'negative', 'neutral']
```

3. **Update README:** `ingestforge/enrichment/README.md`

Add example showing how to use `SentimentEnricher`.

4. **Run tests:**

```bash
pytest tests/test_enrichment_sentiment.py -v
```

### Adding a New Document Processor

1. **Implement IProcessor:** `ingestforge/ingest/rtf_processor.py`

```python
from ingestforge.shared.patterns import IProcessor, ExtractedContent

class RTFProcessor(IProcessor):
    def can_process(self, file_path):
        return file_path.suffix.lower == ".rtf"

    def process(self, file_path):
        # Extract text from RTF
        text = self._extract_rtf(file_path)
        return ExtractedContent(
            text=text,
            metadata={"format": "rtf"}
        )

    def get_supported_extensions(self):
        return [".rtf"]
```

2. **Register in processor factory**

3. **Add tests**

4. **Update documentation**

### Adding a New CLI Command

1. **Add to:** `ingestforge/cli/main.py`

```python
@app.command
def summarize(
    document: str = typer.Argument(..., help="Document to summarize"),
    max_length: int = typer.Option(200, help="Max summary length"),
):
    """Generate document summary."""
    from ingestforge.cli.console import console

    console.print(f"Summarizing {document}...")

    # Your implementation
    summary = "Document summary..."

    console.print(f"\n[bold]Summary:[/bold]\n{summary}")
```

2. **Test it:**

```bash
ingestforge summarize document.pdf --max-length 100
```

---

## Troubleshooting

### Issue 1: Tests Fail After Fresh Clone

**Symptom:** `ImportError: No module named 'ingestforge'`

**Fix:**

```bash
# Ensure you're in virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall in development mode
pip install -e ".[dev]"
```

### Issue 2: ChromaDB Database Locked

**Symptom:** `sqlite3.OperationalError: database is locked`

**Fix:**

```bash
# Close all IngestForge processes
# Delete lock files
rm .data/chromadb/*.lock

# Or use a fresh test directory
ingestforge init --name test-project-2
```

### Issue 3: Import Errors in IDE

**Symptom:** IDE shows import errors but code runs fine

**Fix:**

- **VSCode:** `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv`
- **PyCharm:** Settings → Project → Python Interpreter → Set to `.venv`
- **Restart IDE** after setting interpreter

### Issue 4: Tests Pass Locally But Fail in CI

**Symptom:** Tests pass on your machine but fail in GitHub Actions

**Common causes:**
- Path separators (Windows `\` vs Linux `/`)
- Missing test fixtures
- Environment-specific dependencies

**Fix:**

```python
# Use pathlib for cross-platform paths
from pathlib import Path
file_path = Path("data") / "file.txt"  # Works everywhere

# Don't use:
file_path = "data/file.txt"  # Breaks on Windows
```

---

## Getting Help

### Documentation Resources

- **Module READMEs:** Start with the relevant module README
  - `ingestforge/[module]/README.md`
- **Architecture Docs:** [ARCHITECTURE.md](../../ARCHITECTURE.md)
- **ADRs:** `docs/architecture/ADR-*.md` for design decisions

### Community Support

- **Questions?** Open a [Discussion](https://github.com/yourusername/ingestforge/discussions)
- **Bug?** File an [Issue](https://github.com/yourusername/ingestforge/issues)
- **Feature idea?** Start a [Discussion](https://github.com/yourusername/ingestforge/discussions/categories/ideas)

### Debug Mode

Enable verbose logging:

```python
from ingestforge.core.logging import configure_logging

configure_logging(level="DEBUG")
```

Or via environment variable:

```bash
export INGESTFORGE_LOG_LEVEL=DEBUG
ingestforge query "test"
```

---

## What's Next?

Now that you're onboarded:

- [ ] **Explore the codebase** - Read module READMEs that interest you
- [ ] **Pick an issue** - Start with "good first issue" label
- [ ] **Join discussions** - Share ideas and ask questions
- [ ] **Review PRs** - Learn from others' contributions
- [ ] **Write docs** - Help improve documentation

**Welcome to the IngestForge community!** 🚀

We're excited to have you contributing. Don't hesitate to ask questions—every expert was once a beginner.

---

## Quick Reference

### Common Commands

```bash
# Development
pip install -e ".[dev]"          # Install in dev mode
pytest tests/ -v                 # Run tests
pytest tests/ --cov=ingestforge  # Run with coverage

# CLI Usage
ingestforge init                 # Initialize project
ingestforge ingest file.pdf      # Ingest document
ingestforge query "question"     # Query knowledge base
ingestforge status               # Show status
ingestforge watch                # Watch for new docs

# Git Workflow
git checkout -b feature/my-feature  # Create branch
git add .                           # Stage changes
git commit -m "feat: description"   # Commit
git push origin feature/my-feature  # Push
```

### Project Structure

```
ingestforge/
├── core/          # Config, logging, retry, security
├── shared/        # Interfaces, utilities
├── ingest/        # Document processing
├── chunking/      # Text chunking
├── enrichment/    # Embeddings, entities
├── storage/       # ChromaDB, JSONL
├── retrieval/     # BM25, semantic, hybrid
├── query/         # Query pipeline
├── llm/           # LLM clients
└── cli/           # CLI commands

tests/             # All tests mirror src structure
docs/              # Documentation
  ├── guides/      # This guide
  └── architecture/ # ADRs
```

### Useful Links

- [README.md](../../README.md) - User documentation
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System design
- [REFACTORING.md](../../REFACTORING.md) - Recent changes
- [CHANGELOG.md](../../CHANGELOG.md) - Version history
- [FEATURE_TEMPLATE.md](FEATURE_TEMPLATE.md) - Feature development guidelines
