# Configuration Reference

> Complete guide to IngestForge configuration options

---

## Overview

IngestForge uses a YAML configuration file (`config.yaml`) in your project directory. The file is created automatically by `ingestforge init` with sensible defaults.

**Location:** `./config.yaml` (or set `INGESTFORGE_CONFIG` environment variable)

---

## Quick Start

Minimal configuration:

```yaml
project:
  name: "my-research"
```

Everything else uses defaults. See below for all available options.

---

## Full Configuration

```yaml
# ============================================================
# IngestForge Configuration
# ============================================================

# ------------------------------------------------------------
# Project Settings
# ------------------------------------------------------------
project:
  name: "my-knowledge-base"      # Project identifier
  data_dir: ".data"              # Processed data storage
  ingest_dir: ".ingest"          # Document drop zone

# ------------------------------------------------------------
# Document Ingestion
# ------------------------------------------------------------
ingest:
  watch_enabled: true            # Enable directory watching
  watch_interval_sec: 5          # Polling interval for watcher
  supported_formats:             # File types to process
    - ".pdf"
    - ".epub"
    - ".docx"
    - ".txt"
    - ".md"
  move_completed: true           # Move processed files to completed/

# ------------------------------------------------------------
# Document Splitting
# ------------------------------------------------------------
split:
  use_toc: true                  # Use PDF table of contents
  deep_split: false              # Split at subsection level
  min_chapter_size_kb: 5         # Minimum chapter size
  fallback_single_file: true     # Keep as single file if no TOC

# ------------------------------------------------------------
# Text Chunking
# ------------------------------------------------------------
chunking:
  strategy: "semantic"           # semantic, fixed, paragraph, legal, code
  target_size: 300               # Target words per chunk
  min_size: 50                   # Minimum chunk size
  max_size: 1000                 # Maximum chunk size
  overlap: 50                    # Word overlap between chunks
  use_llm: true                  # Use LLM for smart chunking

# ------------------------------------------------------------
# Chunk Enrichment
# ------------------------------------------------------------
enrichment:
  generate_embeddings: true      # Generate vector embeddings
  embedding_model: "all-MiniLM-L6-v2"  # sentence-transformers model
  extract_entities: true         # NER extraction (requires spaCy)
  generate_questions: false      # Hypothetical Q&A (requires LLM)
  compute_quality: true          # Quality scoring

# ------------------------------------------------------------
# Storage Backend
# ------------------------------------------------------------
storage:
  backend: "chromadb"            # chromadb, jsonl, postgres

  # ChromaDB settings
  chromadb:
    persist_directory: ".data/chromadb"
    collection_name: null        # Auto-generated from project name

  # JSONL settings (fallback)
  jsonl:
    chunks_file: ".data/chunks/chunks.jsonl"

  # PostgreSQL settings
  postgres:
    connection_string: "${DATABASE_URL}"
    pool_size: 5

# ------------------------------------------------------------
# OCR Settings
# ------------------------------------------------------------
ocr:
  preferred_engine: "auto"       # auto, tesseract, easyocr
  language: "eng"                # Tesseract language code
  languages: ["en"]              # EasyOCR language codes
  scanned_threshold: 100         # chars/page threshold for scanned detection
  confidence_threshold: 0.3      # min confidence to keep OCR result
  use_gpu: false                 # GPU acceleration for EasyOCR
  page_timeout: 120              # seconds per page OCR timeout
  max_workers: 1                 # concurrent OCR workers

# ------------------------------------------------------------
# Retrieval Settings
# ------------------------------------------------------------
retrieval:
  strategy: "hybrid"             # bm25, semantic, hybrid
  top_k: 10                      # Default results to return
  rerank: true                   # Enable reranking
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_top_k: 5                # Results after reranking

  # Hybrid fusion settings
  hybrid:
    bm25_weight: 0.4             # Weight for keyword search
    semantic_weight: 0.6         # Weight for semantic search
    fusion_method: "rrf"         # rrf (Reciprocal Rank Fusion) or weighted

  # BM25 settings
  bm25:
    k1: 1.5                      # Term frequency saturation
    b: 0.75                      # Length normalization

# ------------------------------------------------------------
# Query Processing
# ------------------------------------------------------------
query:
  expand_queries: false          # Add synonyms/related terms
  classify_intent: true          # Classify query intent

  # Cache settings
  cache:
    enabled: true                # Enable query caching
    max_size: 1000               # Maximum cache entries
    ttl_seconds: 3600            # Cache TTL (1 hour)
    persist: true                # Save cache to disk

# ------------------------------------------------------------
# LLM Providers
# ------------------------------------------------------------
llm:
  default_provider: "gemini"     # gemini, claude, openai, ollama, llamacpp

  # Google Gemini
  gemini:
    model: "gemini-1.5-flash"
    api_key: "${GEMINI_API_KEY}"
    temperature: 0.7
    max_tokens: 2048

  # Anthropic Claude
  claude:
    model: "claude-3-haiku-20240307"
    api_key: "${ANTHROPIC_API_KEY}"
    temperature: 0.7
    max_tokens: 2048

  # OpenAI
  openai:
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
    max_tokens: 2048

  # Local Ollama
  ollama:
    model: "llama3:latest"
    url: "http://localhost:11434"
    temperature: 0.7

  # Local llama.cpp
  llamacpp:
    model_path: ".data/models/phi-2.gguf"
    n_ctx: 4096                  # Context window
    n_threads: 4                 # CPU threads
    n_gpu_layers: 0              # GPU layers (0 = CPU only)

# ------------------------------------------------------------
# API Server
# ------------------------------------------------------------
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "*"
  rate_limit: 100                # Requests per minute

# ------------------------------------------------------------
# Performance Mode (root-level setting)
# ------------------------------------------------------------
# Performance presets adjust chunking, enrichment, retrieval,
# and storage settings. Valid modes: quality, balanced, speed, mobile
performance_mode: "balanced"

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging:
  level: "INFO"                  # DEBUG, INFO, WARNING, ERROR
  format: "structured"           # structured, simple
  file: null                     # Log file path (null = stdout only)
```

---

## Section Details

### project

Basic project identification.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `"my-knowledge-base"` | Project name, used in ChromaDB collection |
| `data_dir` | string | `".data"` | Directory for processed data |
| `ingest_dir` | string | `".ingest"` | Directory for document ingestion |

### ingest

Document ingestion settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `watch_enabled` | bool | `true` | Enable the directory watcher |
| `watch_interval_sec` | int | `5` | How often to check for new files |
| `supported_formats` | list | See above | File extensions to process |
| `move_completed` | bool | `true` | Move files to `completed/` after processing |

> **Note:** OCR settings are in the separate `ocr` section below, not under `ingest`.

### ocr

OCR processing settings. HTML, image, and PPTX formats are supported regardless of `ingest.supported_formats`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `preferred_engine` | string | `"auto"` | OCR engine: `auto`, `tesseract`, `easyocr` |
| `language` | string | `"eng"` | Tesseract language code |
| `languages` | list | `["en"]` | EasyOCR language codes |
| `scanned_threshold` | int | `100` | Chars/page below which a page is considered scanned |
| `confidence_threshold` | float | `0.3` | Minimum OCR confidence to keep result |
| `use_gpu` | bool | `false` | GPU acceleration for EasyOCR |
| `page_timeout` | int | `120` | Seconds per page OCR timeout (0 = no limit) |
| `max_workers` | int | `1` | Concurrent OCR workers for multi-page PDFs |

**OCR Languages (Tesseract):**

Common codes: `eng` (English), `deu` (German), `fra` (French), `spa` (Spanish), `chi_sim` (Chinese Simplified)

Full list: [Tesseract Language Codes](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)

### chunking

How documents are split into chunks.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | string | `"semantic"` | Chunking strategy |
| `target_size` | int | `300` | Target words per chunk |
| `min_size` | int | `50` | Minimum chunk size (won't split smaller) |
| `max_size` | int | `1000` | Maximum chunk size (will split larger) |
| `overlap` | int | `50` | Word overlap between chunks |
| `use_llm` | bool | `true` | Use LLM for intelligent boundary detection |

**Strategies:**

- `semantic`: Split at semantic boundaries (paragraphs, sections)
- `fixed`: Fixed-size chunks with overlap
- `paragraph`: One chunk per paragraph
- `legal`: Section-aware chunking for legal/regulatory documents
- `code`: AST-aware chunking for source code files

### enrichment

Chunk enrichment options.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `generate_embeddings` | bool | `true` | Generate vector embeddings |
| `embedding_model` | string | `"all-MiniLM-L6-v2"` | sentence-transformers model |
| `extract_entities` | bool | `true` | Extract named entities (pattern-based + spaCy if installed) |
| `generate_questions` | bool | `false` | Generate hypothetical questions (requires LLM) |
| `compute_quality` | bool | `true` | Compute chunk quality scores |

**Embedding Models:**

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good |
| `all-mpnet-base-v2` | 768 | Medium | Better |
| `all-MiniLM-L12-v2` | 384 | Medium | Better |

### storage

Storage backend configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | string | `"chromadb"` | Storage backend to use |

**Backends:**

- `chromadb`: Vector database with semantic search (recommended)
- `jsonl`: Simple file-based storage (no semantic search)
- `postgres`: PostgreSQL + pgvector

#### chromadb

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `persist_directory` | string | `".data/chromadb"` | Database directory |
| `collection_name` | string | `null` | Collection name (auto-generated if null) |

### retrieval

Search and retrieval settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | string | `"hybrid"` | Retrieval strategy |
| `top_k` | int | `10` | Default number of results |
| `rerank` | bool | `true` | Enable cross-encoder reranking |
| `rerank_model` | string | See above | Reranking model |
| `rerank_top_k` | int | `5` | Results to return after reranking |

**Strategies:**

- `hybrid`: Combines BM25 + semantic (best quality)
- `semantic`: Vector similarity only
- `bm25`: Keyword matching only (fastest)

#### hybrid

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `bm25_weight` | float | `0.4` | Weight for BM25 results |
| `semantic_weight` | float | `0.6` | Weight for semantic results |
| `fusion_method` | string | `"rrf"` | How to combine results |

**Fusion Methods:**

- `rrf`: Reciprocal Rank Fusion (position-based)
- `weighted`: Score-based weighted average

### llm

LLM provider configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_provider` | string | `"gemini"` | Which provider to use |

Each provider has its own section. All support:

| Key | Type | Description |
|-----|------|-------------|
| `model` | string | Model name/ID |
| `api_key` | string | API key (use `${ENV_VAR}` syntax) |
| `temperature` | float | Generation temperature (0.0-1.0) |
| `max_tokens` | int | Maximum response tokens |

### performance

Performance modes are applied via `apply_performance_preset` which adjusts other config sections. The `performance_mode` field on the root Config object selects the preset.

> **Note:** `embedding_batch_size`, `max_memory_mb`, and `checkpoint_interval` are shown in the YAML example above for illustration but are **not currently wired** as config fields. The performance preset works by adjusting `chunking`, `enrichment`, `retrieval`, and `storage` settings directly.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `performance_mode` | string | `"balanced"` | Performance profile (root-level field) |

**Modes:**

- `fast`: Minimal processing, fastest results
- `balanced`: Good quality with reasonable speed
- `thorough`: Maximum quality, slower processing

---

## Environment Variables

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
llm:
  gemini:
    api_key: "${GEMINI_API_KEY}"
```

**Required variables by provider:**

| Provider | Variable |
|----------|----------|
| Gemini | `GEMINI_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Ollama | None (local) |
| llama.cpp | None (local) |

---

## Configuration Profiles

### Minimal (Fastest)

```yaml
project:
  name: "quick-search"

chunking:
  strategy: "fixed"
  target_size: 500

enrichment:
  generate_embeddings: true
  extract_entities: false
  generate_questions: false

retrieval:
  strategy: "bm25"
  rerank: false
```

### Balanced (Recommended)

```yaml
project:
  name: "research-project"

chunking:
  strategy: "semantic"
  target_size: 300

enrichment:
  generate_embeddings: true
  compute_quality: true

retrieval:
  strategy: "hybrid"
  rerank: true
```

### Thorough (Best Quality)

```yaml
project:
  name: "thesis-research"

chunking:
  strategy: "semantic"
  target_size: 200
  overlap: 100

enrichment:
  generate_embeddings: true
  embedding_model: "all-mpnet-base-v2"
  extract_entities: true
  compute_quality: true

retrieval:
  strategy: "hybrid"
  rerank: true
  top_k: 20
  rerank_top_k: 10

  hybrid:
    bm25_weight: 0.3
    semantic_weight: 0.7
```

### Low-End Device

```yaml
project:
  name: "laptop-research"

performance_mode: "speed"        # or "mobile" for Android/Termux

chunking:
  target_size: 400

enrichment:
  generate_embeddings: true
  embedding_model: "all-MiniLM-L6-v2"
  extract_entities: false

retrieval:
  strategy: "hybrid"
  rerank: false
  top_k: 5
```

---

## Validation

Configuration is validated on load. Common errors:

| Error | Solution |
|-------|----------|
| `Unknown backend: xyz` | Use `chromadb`, `jsonl`, or `postgres` |
| `Invalid strategy: xyz` | Use `semantic`, `fixed`, `paragraph`, `legal`, or `code` |
| `API key not set` | Set the environment variable |
| `Model not found` | Check model name spelling |

---

## Programmatic Access

```python
from ingestforge.core.config import Config, load_config
from pathlib import Path

# Load from file
config = load_config(base_path=Path.cwd)

# Access values
print(config.project.name)
print(config.chunking.target_size)
print(config.llm.default_provider)

# Create programmatically
config = Config
config.project.name = "my-project"
config.chunking.strategy = "semantic"
```

---

*Last updated: 2026-02-04*
