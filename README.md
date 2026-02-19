# 🛠️ IngestForge

[![Continuous Integration](https://github.com/joshuatshirley/IngestForge/actions/workflows/ci.yml/badge.svg)](https://github.com/joshuatshirley/IngestForge/actions/workflows/ci.yml)
[![Linting & Standards](https://github.com/joshuatshirley/IngestForge/actions/workflows/lint.yml/badge.svg)](https://github.com/joshuatshirley/IngestForge/actions/workflows/lint.yml)
[![Security Scan](https://github.com/joshuatshirley/IngestForge/actions/workflows/security.yml/badge.svg)](https://github.com/joshuatshirley/IngestForge/actions/workflows/security.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality: JPL](https://img.shields.io/badge/Code%20Quality-NASA%20JPL-green.svg)](https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code)

Modular document processing and RAG (Retrieval-Augmented Generation) framework for mission-critical research.

## 📂 Repository Organization

See [docs/architecture/REPOSITORY_GUIDE.md](docs/architecture/REPOSITORY_GUIDE.md) for a complete mapping of the project structure.

- `ingestforge/`: Core Python RAG engine and CLI.
- `frontend/`: Next.js Visual Workbench.
- `tests/`: GWT-based unit and integration suites.
- `docs/`: [Technical specifications](docs/architecture/ARCHITECTURE.md) and [user guides](docs/guides/COMMAND_REFERENCE.md).

## Overview

IngestForge is a modular Python framework for building RAG (Retrieval-Augmented Generation) applications. It follows NASA JPL's Power of Ten rules for mission-critical software quality.

## Features

### 🧠 Autonomous Intelligence
- **Autonomous Domain Router:** Zero-config auto-detection of content domain (Legal, Cyber, Medical, etc.) during ingestion.
- **Query Domain Classifier:** Real-time analysis of natural language queries to apply domain-specific retrieval strategies.
- **Cross-Domain Metadata Merger:** Intelligent resolution of metadata conflicts for mixed-domain documents.
- **Domain-Specific Field Boosting:** Automated post-retrieval rescoring based on high-value extracted fields (CVEs, Part Numbers, Zoning Codes).

### 🏛️ Specialized Verticals (24+)
IngestForge provides deep intelligence layers for specialized domains:
- **Professional:** Legal (Bluebook citations), Medical (ICD-10 patterns), Financial (EBITDA/Balance Sheets), Technical (Code metrics).
- **Security & Risk:** Cybersecurity (CVE/CVSS extraction, log flattening), Disaster Response (Sitrep parsing, coordinate extraction).
- **Science & Industry:** Bio/Lab (Experiment IDs, reagents), Manufacturing (Maintenance cycles, part numbers), Automotive (VIN, chassis, torque specs).
- **Public Sector:** Urban Planning (Zoning codes, FAR ratios, density targets), Political (Opposition research, campaign finance).
- **Culture & Lifestyle:** Museum (Provenance, era detection), Spiritual (Scripture citations), Wellness (Macro/Calorie extraction).
- **Hobby & Personal:** Gaming (Patch notes, stat changes), Tabletop RPGs (Stat blocks), Genealogy (GEDCOM/Family tree parsing), PKM (Obsidian/Wikilink support).

### 📄 Document Processing
- **Deep Format Support:** PDF (text + OCR), EPUB, DOCX, HTML, PPTX, Markdown, LaTeX, Jupyter, Source Code, Audio (MP3, WAV).
- **Multi-Stage Extraction:** Automated extraction of text, tables, images, metadata, and cross-references.
- **Smart Refiners:** LaTeX math-to-unicode normalization, OCR cleanup, PII redaction, and structure recovery.

### ✂️ Intelligent Chunking
- **Semantic Splitting:** Embedding-based boundary detection for coherent context preservation.
- **Vertical-Aware Chunking:** Specialized logic for Legal sections, Code blocks, and Academic chapters.
- **Optimization:** Automatic deduplication, size normalization, and parent-child hierarchy tracking.

### 🔎 Advanced Retrieval & RAG
- **Hybrid Search:** Fusion of BM25 keyword matching and vector semantic search (Cosine/Dot-product).
- **Query Pipeline:** HyDE (Hypothetical Document Embeddings), Query expansion, and multi-hop routing.
- **Reranking:** Cross-encoder integration for precision re-ordering of top candidates.
- **Diversity & MMR:** Maximal Marginal Relevance to reduce redundancy in LLM context.

### 📊 Analysis & Knowledge Graphs
- **Graph Construction:** Automated relationship extraction (SVO triples) and Knowledge Graph building.
- **Fact-Checking:** Contradiction detection and evidence linking across large corpora.
- **Literary Analysis:** Character arc tracking, theme detection, and narrative structure visualization.

### 🎓 Study & Spaced Repetition
- **Automated Learning:** Spaced repetition scheduling, quiz generation, and interactive flashcards.
- **Concept Mapping:** Visualizing connections between ingested ideas.

## Installation

### Basic Installation
```bash
pip install ingestforge
```

### With Optional Features
```bash
# Vector embeddings and ChromaDB
pip install ingestforge[embeddings]

# LLM providers (Claude, OpenAI, Gemini)
pip install ingestforge[llm]

# PostgreSQL vector storage
pip install ingestforge[postgres]

# Audio transcription (Whisper)
pip install ingestforge[audio]

# YouTube transcript extraction
pip install ingestforge[youtube]

# All features
pip install ingestforge[all]

# Development tools
pip install ingestforge[dev]
```

### From Source
```bash
git clone https://github.com/joshuatshirley/IngestForge.git
cd IngestForge
pip install -e ".[all,dev]"
```

## Quick Start

```bash
# Initialize project
ingestforge init my_project

# Ingest documents
ingestforge ingest documents/ --recursive

# Query knowledge base
ingestforge query "What is this about?"

# Check status
ingestforge status
```

## CLI Commands

IngestForge provides 16 command groups with 47+ commands:

| Group | Description | Example |
|-------|-------------|---------|
| **Core** | Project init, ingest, query | `ingestforge init`, `ingestforge ingest` |
| **Analyze** | Topic modeling, similarity | `ingestforge analyze topics` |
| **Citation** | Extract and format citations | `ingestforge citation extract` |
| **Code** | Code analysis and mapping | `ingestforge code analyze` |
| **Export** | Export to Markdown, JSON | `ingestforge export markdown` |
| **Literary** | Character/theme analysis | `ingestforge literary themes` |
| **Study** | Flashcards, quizzes | `ingestforge study quiz` |
| **Interactive** | REPL mode | `ingestforge interactive shell` |
| **Transform** | Text transformation | `ingestforge transform clean` |
| **Workflow** | Batch processing | `ingestforge workflow batch` |
| **Config** | Configuration management | `ingestforge config show` |
| **Maintenance** | Backup, cleanup, optimize | `ingestforge maintenance backup` |
| **Monitor** | Health checks, diagnostics | `ingestforge monitor health` |
| **Index** | Index management | `ingestforge index list` |
| **Discovery** | Academic search | `ingestforge discovery arxiv` |
| **Research** | Quality auditing | `ingestforge research audit` |

## Module Structure

```
ingestforge/
├── cli/              # Command-line interface (16 groups, 47+ commands)
├── core/             # Core infrastructure (config, logging, security)
├── ingest/           # Document processing (PDF, EPUB, HTML, LaTeX, Jupyter)
├── chunking/         # Text splitting strategies (semantic, code, legal)
├── enrichment/       # Metadata and embedding enrichment (NER, topics)
├── storage/          # Storage backends (JSONL, ChromaDB, PostgreSQL)
├── llm/              # LLM provider integrations
├── retrieval/        # Search and retrieval strategies
├── query/            # Query processing and sessions
├── analysis/         # Content analysis tools
├── curation/         # Quality scoring and deduplication
├── discovery/        # Academic paper discovery (arXiv, Semantic Scholar)
└── shared/           # Shared utilities and patterns
```

**Stats:** 361 Python modules, ~100,000 lines of code, 2,500+ tests

## Configuration

Configuration is stored in `.ingestforge/config.json`:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4"
  },
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small"
  },
  "storage": {
    "backend": "chromadb",
    "path": ".ingestforge/storage"
  }
}
```

Manage with:
```bash
ingestforge config show
ingestforge config set llm.model gpt-4
ingestforge config validate
```

## Examples

### Academic Research
```bash
# Setup
ingestforge init research_project
ingestforge config set llm.model gpt-4

# Ingest papers
ingestforge workflow batch ingest papers/ --pattern "*.pdf"

# Analysis
ingestforge analyze topics --topics 50
ingestforge citation extract
ingestforge research audit
```

### Code Documentation
```bash
# Analyze codebase
ingestforge workflow batch ingest src/ --pattern "*.py"
ingestforge code analyze src/
ingestforge code map project/ --format mermaid
```

### Content Curation
```bash
# Process content
ingestforge workflow pipeline full content/

# Analysis and cleanup
ingestforge analyze similarity --threshold 0.95
ingestforge transform clean content/

# Export
ingestforge export markdown curated.md
```

### Multi-Domain Intelligence
```bash
# Ingest mixed data (Legal, Cyber, and Urban docs)
ingestforge ingest ./mixed_research/ --recursive

# The system automatically detects domains and extracts:
# - Court citations for legal PDFs
# - CVE/CVSS scores for cyber security reports
# - Zoning codes & FAR ratios for city ordinances
# - PII Redaction across all domains (Air-Gap ready)
```

## Development

### Requirements
- Python 3.10+
- See `requirements.txt` for dependencies

### Testing
```bash
# Run tests
pytest tests/

# Type checking
mypy ingestforge/

# Validate NASA JPL compliance
python validate_commandments.py ingestforge/
```

### Code Quality

All code follows NASA JPL's Power of Ten rules:
- Maximum function size: 60 lines
- Maximum nesting depth: 3 levels
- Complete type hint coverage
- Fixed loop bounds
- Comprehensive parameter validation
- No silent error swallowing

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please ensure your code:
1. Passes all tests (`pytest tests/`)
2. Has type hints (`mypy ingestforge/`)
3. Follows NASA JPL coding rules (`python validate_commandments.py`)
4. Includes tests for new functionality
