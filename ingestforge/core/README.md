# Core Module

## Purpose

Framework-level infrastructure and utilities used across all IngestForge features. Provides configuration management, logging, retry logic, security utilities, job queue, and provenance tracking.

## Architecture Context

The `core/` module sits at the foundation of IngestForge's hexagonal architecture. It provides cross-cutting concerns that all other modules depend on, but has no dependencies on feature modules.

```
┌─────────────────────────────────────────┐
│   CLI, Ingest, Enrichment, Query        │  Feature Modules
│   (depend on core utilities)            │
├─────────────────────────────────────────┤
│   core/ - Framework Infrastructure      │  ← You are here
│   (config, logging, retry, security)    │
└─────────────────────────────────────────┘
```

## Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `config.py` | YAML configuration with environment variable expansion | ✅ Complete |
| `logging.py` | Structured logging with context tracking | ✅ Complete |
| `retry.py` | Exponential backoff retry decorators for external APIs | ✅ Complete |
| `security.py` | Path sanitization and URL validation (SSRF prevention) | ✅ Complete |
| `provenance.py` | Source location tracking and citation generation | ✅ Complete |
| `jobs.py` | SQLite-based job queue for background processing | ✅ Complete |
| `pipeline.py` | Document processing pipeline orchestration | ✅ Complete |
| `state.py` | Processing state tracking for documents | ✅ Complete |

## Configuration Management

### Config Data Structure

IngestForge uses a hierarchical configuration system with 9 main sections:

```python
@dataclass
class Config:
    project: ProjectConfig        # Project name, data paths
    ingest: IngestConfig         # Document ingestion settings
    split: SplitConfig           # PDF/document splitting
    chunking: ChunkingConfig     # Chunking strategy and sizes
    enrichment: EnrichmentConfig # Embeddings, entities, questions
    storage: StorageConfig       # ChromaDB, PostgreSQL, JSONL
    retrieval: RetrievalConfig   # BM25, semantic, hybrid
    llm: LLMConfig              # Gemini, Claude, OpenAI, Ollama
    api: APIConfig              # API server host, port, CORS
```

### Loading Configuration

Configuration sources in order of precedence:
1. **Environment variables** (highest priority)
2. **YAML config file** (`config.yaml`)
3. **Built-in defaults** (lowest priority)

```python
from ingestforge.core import load_config

# Load from default location (./config.yaml)
config = load_config

# Load from specific path
config = load_config(config_path=Path("custom/config.yaml"))

# Access configuration values
print(config.chunking.target_size)  # 300
print(config.llm.default_provider)   # "gemini"
print(config.data_path)              # Path object to data directory
```

### Environment Variable Overrides

Override any config value with environment variables:

```bash
# LLM Configuration - set the API key for your chosen provider
export ANTHROPIC_API_KEY="your-claude-key"  # For Claude
# export GEMINI_API_KEY="your-gemini-key"   # Or for Gemini
# export OPENAI_API_KEY="your-openai-key"   # Or for OpenAI

export INGESTFORGE_LLM_PROVIDER="claude"
export INGESTFORGE_LLM_MODEL="claude-3-haiku-20240307"

# Storage Configuration
export INGESTFORGE_STORAGE_BACKEND="chromadb"
export INGESTFORGE_DATA_PATH=".data"

# Chunking Configuration
export INGESTFORGE_CHUNK_SIZE="500"
export INGESTFORGE_CHUNK_OVERLAP="75"

# API Server
export INGESTFORGE_API_HOST="0.0.0.0"
export INGESTFORGE_API_PORT="8000"
```

### Path Utilities

Config provides convenient path properties:

```python
config = load_config

# Automatic path resolution
config.data_path        # /absolute/path/to/.data
config.pending_path     # /absolute/path/to/.ingest/pending
config.completed_path   # /absolute/path/to/.ingest/completed
config.chunks_path      # /absolute/path/to/.data/chunks
config.chromadb_path    # /absolute/path/to/.data/chromadb

# Create all required directories
config.ensure_directories
```

## Structured Logging

### Basic Usage

```python
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Simple logging
logger.info("Processing document", document_id="doc_123")
logger.warning("Retrying API call", attempt=2, error="timeout")
logger.error("Failed to process", document_id="doc_456", reason="unsupported format")
```

### Context Tracking

Add persistent context to all log messages:

```python
# Create logger with context
doc_logger = logger.with_context(document_id="doc_789", user="alice")

# All messages include context automatically
doc_logger.info("Starting processing")  # Includes document_id and user
doc_logger.debug("Chunking complete")    # Includes document_id and user
```

### Specialized Loggers

#### PipelineLogger

Tracks document processing stages with timing:

```python
from ingestforge.core.logging import PipelineLogger

logger = PipelineLogger(document_id="doc_123")

logger.start_stage("extraction")
# ... do extraction work ...
logger.start_stage("chunking")  # Auto-logs completion of extraction
# ... do chunking work ...
logger.finish(success=True, chunks=42)
```

**Output:**
```
2026-02-02 10:15:30 | INFO | ingestforge.pipeline | Starting stage | document_id=doc_123 | stage=extraction
2026-02-02 10:15:35 | INFO | ingestforge.pipeline | Completed stage | document_id=doc_123 | stage=extraction | duration_sec=5.23
2026-02-02 10:15:35 | INFO | ingestforge.pipeline | Starting stage | document_id=doc_123 | stage=chunking
2026-02-02 10:15:40 | INFO | ingestforge.pipeline | Completed stage | document_id=doc_123 | stage=chunking | duration_sec=4.87
2026-02-02 10:15:40 | INFO | ingestforge.pipeline | Pipeline completed successfully | document_id=doc_123 | chunks_created=42
```

### Log Configuration

```python
from ingestforge.core.logging import configure_logging
from pathlib import Path

# Configure global logging
configure_logging(
    level="DEBUG",                      # DEBUG, INFO, WARNING, ERROR
    log_file=Path("logs/app.log"),     # Optional file output
    console=True                        # Console output
)
```

## Retry Logic

### Pre-configured Decorators

```python
from ingestforge.core.retry import llm_retry, embedding_retry, network_retry

# LLM API calls (3 attempts, 2s base delay, 30s max)
@llm_retry
def generate_answer(prompt: str) -> str:
    return llm_client.generate(prompt)

# Embedding API calls (3 attempts, 1s base delay, 15s max)
@embedding_retry
def get_embedding(text: str) -> list[float]:
    return embedding_model.encode(text)

# Network operations (5 attempts, 0.5s base delay, 30s max)
@network_retry
def fetch_url(url: str) -> str:
    return requests.get(url).text
```

### Custom Retry Decorator

```python
from ingestforge.core.retry import retry

@retry(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=(ConnectionError, TimeoutError),
)
def unreliable_operation:
    # Retries only on ConnectionError and TimeoutError
    # Delays: 2s, 4s, 8s, 16s (with jitter)
    ...
```

### Retry Behavior

- **Exponential backoff**: `delay = base_delay * (exponential_base ^ attempt)`
- **Jitter**: Adds 0-25% random variation to prevent thundering herd
- **Max delay cap**: Delays never exceed `max_delay`
- **Automatic logging**: Logs warnings on retry, error on final failure

Example retry sequence for `@llm_retry`:
1. **Attempt 1** fails → wait ~2s
2. **Attempt 2** fails → wait ~4s
3. **Attempt 3** fails → raise `RetryError`

### Retry Context Manager

For fine-grained control:

```python
from ingestforge.core.retry import RetryableOperation

with RetryableOperation(max_attempts=3) as op:
    while op.should_retry:
        try:
            result = call_api
            op.success
            break
        except Exception as e:
            op.failed(e)  # Handles retry logic
```

### Async Retry

```python
from ingestforge.core.retry import retry_async

@retry_async(max_attempts=3, base_delay=1.0)
async def fetch_async(url: str):
    async with aiohttp.ClientSession as session:
        async with session.get(url) as response:
            return await response.text
```

## Security Utilities

### Path Sanitization

Prevent directory traversal attacks:

```python
from ingestforge.core.security import PathSanitizer, PathTraversalError

sanitizer = PathSanitizer(base_dir=Path("/app/data"))

# Sanitize filenames
safe_name = sanitizer.sanitize_filename("../../../etc/passwd")
# Returns: "passwd" (traversal removed)

safe_name = sanitizer.sanitize_filename("file<name>.txt")
# Returns: "filename.txt" (dangerous chars removed)

# Validate paths stay within base directory
try:
    safe_path = sanitizer.sanitize("uploads/user_file.pdf")
    # Returns: Path("/app/data/uploads/user_file.pdf")

    safe_path = sanitizer.sanitize("../../../etc/passwd")
    # Raises: PathTraversalError
except PathTraversalError as e:
    print(f"Attack detected: {e}")
```

### URL Validation (SSRF Prevention)

Prevent Server-Side Request Forgery attacks:

```python
from ingestforge.core.security import URLValidator, SSRFError

validator = URLValidator

# Validate URLs before fetching
try:
    validator.validate("https://example.com/doc.pdf")
    # ✅ Valid external URL

    validator.validate("http://localhost:8080/admin")
    # ❌ Raises SSRFError - blocked localhost

    validator.validate("http://169.254.169.254/metadata")
    # ❌ Raises SSRFError - blocked AWS metadata endpoint

    validator.validate("http://192.168.1.1/config")
    # ❌ Raises SSRFError - blocked private network
except SSRFError as e:
    print(f"SSRF attempt blocked: {e}")
```

### Safe File Operations

Wrapper for secure file operations:

```python
from ingestforge.core.security import create_safe_file_ops

file_ops = create_safe_file_ops(base_dir=Path("/app/data"))

# All operations stay within base_dir
file_ops.write("user_upload.txt", content="safe content")
# ✅ Writes to /app/data/user_upload.txt

file_ops.write("../../../etc/passwd", content="malicious")
# ❌ Raises PathTraversalError

content = file_ops.read("documents/paper.pdf")
# ✅ Reads from /app/data/documents/paper.pdf

file_ops.delete("uploads/temp.txt")
# ✅ Deletes /app/data/uploads/temp.txt
```

## Citation and Provenance

Track source location down to chapter, section, page, and paragraph level.

### Creating Source Locations

```python
from ingestforge.core.provenance import (
    SourceLocation,
    Author,
    SourceType,
    create_web_source,
    create_pdf_source,
)

# Web source
web_loc = create_web_source(
    url="https://example.com/quantum",
    title="Introduction to Quantum Computing",
    author="Jane Smith",
    publication_date="2023",
    section="Section 2: Quantum Gates"
)

# PDF source with precise location
pdf_loc = create_pdf_source(
    file_path="docs/quantum_book.pdf",
    title="Quantum Computing: An Applied Approach",
    authors=["Jack Hidary"],
    publication_date="2019",
    page_start=47,
    page_end=48,
    chapter="Chapter 3: Quantum Gates",
    section="3.2 Single-Qubit Gates",
)

# Manual construction for full control
manual_loc = SourceLocation(
    source_type=SourceType.PDF,
    title="Advanced Quantum Algorithms",
    authors=[Author("Alice Johnson"), Author("Bob Chen")],
    publication_date="2024-01-15",
    chapter_number=5,
    section_number="5.3",
    page_start=142,
    page_end=145,
    paragraph_number=4,
)
```

### Generating Citations

```python
from ingestforge.core.provenance import CitationStyle

# Short inline citation
cite = pdf_loc.to_short_cite
# Output: "[Hidary 2019, Ch.3 §3.2, p.47-48]"

# Full APA citation
apa = pdf_loc.to_citation(CitationStyle.APA)
# Output: "Hidary, J. (2019). *Quantum Computing: An Applied Approach*."

# MLA citation
mla = pdf_loc.to_citation(CitationStyle.MLA)
# Output: "Hidary, Jack. *Quantum Computing: An Applied Approach*. 2019."

# BibTeX entry
bibtex = pdf_loc.to_citation(CitationStyle.BIBTEX)
# Output:
# @article{hidary2019quantum,
#   author = {Jack Hidary},
#   title = {Quantum Computing: An Applied Approach},
#   year = {2019},
# }
```

### Serialization

```python
# Convert to dictionary for JSON storage
data = pdf_loc.to_dict

# Restore from dictionary
restored = SourceLocation.from_dict(data)

# Use in chunk metadata
chunk.metadata["source"] = pdf_loc.to_dict
```

### Citation Style Support

| Style | Enum Value | Status |
|-------|-----------|--------|
| APA (7th ed) | `CitationStyle.APA` | ✅ Complete |
| MLA (9th ed) | `CitationStyle.MLA` | ✅ Complete |
| Chicago | `CitationStyle.CHICAGO` | ✅ Complete |
| BibTeX | `CitationStyle.BIBTEX` | ✅ Complete |
| Harvard | `CitationStyle.HARVARD` | ⚠️ Falls back to APA |
| IEEE | `CitationStyle.IEEE` | ⚠️ Falls back to APA |

## Job Queue (Background Processing)

### Creating Jobs

```python
from ingestforge.core.jobs import (
    create_job,
    create_job_queue,
    JobType,
    JobStatus,
)

# Create job queue
queue = create_job_queue(db_path=Path(".data/jobs.db"))

# Create a document processing job
job = create_job(
    job_type=JobType.INGEST_DOCUMENT,
    payload={"file_path": "documents/paper.pdf"},
    priority=1  # Higher number = higher priority
)

# Enqueue job
job_id = queue.enqueue(job)
```

### Worker Pool

```python
from ingestforge.core.jobs import WorkerPool

def process_document(job):
    """Worker function for document processing."""
    file_path = job.payload["file_path"]
    # ... process document ...
    return {"chunks": 42, "success": True}

# Create worker pool
pool = WorkerPool(
    queue=queue,
    num_workers=4,
    handlers={
        JobType.INGEST_DOCUMENT: process_document,
    }
)

# Start processing (blocks until shutdown)
pool.start

# In another thread/process
pool.shutdown  # Graceful shutdown
```

### Job Status Tracking

```python
# Check job status
job = queue.get_job(job_id)
print(job.status)  # JobStatus.PENDING, RUNNING, COMPLETED, FAILED

# Get results (blocks until complete)
result = queue.wait_for_result(job_id, timeout=60)

# List all jobs
jobs = queue.list_jobs(status=JobStatus.PENDING, limit=10)
```

## Pipeline Orchestration

```python
from ingestforge.core import Pipeline, ProcessingState

# Create pipeline
pipeline = Pipeline(config=config)

# Process document
result = pipeline.process_document(
    file_path=Path("documents/paper.pdf"),
    document_id="doc_123"
)

# Check state
if result.state == ProcessingState.COMPLETED:
    print(f"Created {result.chunks_count} chunks")
else:
    print(f"Failed: {result.error}")
```

## Extension Points

### Custom Retry Strategies

Create domain-specific retry decorators:

```python
from ingestforge.core.retry import retry

# Custom retry for database operations
db_retry = retry(
    max_attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    retryable_exceptions=(DatabaseError, ConnectionError),
)

@db_retry
def insert_chunks(chunks):
    db.insert_many(chunks)
```

### Custom Loggers

Extend structured logging for specific domains:

```python
from ingestforge.core.logging import get_logger
from datetime import datetime

class EnrichmentLogger:
    """Specialized logger for enrichment operations."""

    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id
        self.logger = get_logger("ingestforge.enrichment")
        self.start_time = datetime.now

    def log_enricher(self, name: str, duration: float):
        self.logger.info(
            f"Enricher completed",
            chunk_id=self.chunk_id,
            enricher=name,
            duration_sec=f"{duration:.3f}",
        )

    def finish(self, enrichers_applied: int):
        total_duration = (datetime.now - self.start_time).total_seconds
        self.logger.info(
            "Enrichment complete",
            chunk_id=self.chunk_id,
            enrichers=enrichers_applied,
            total_duration_sec=f"{total_duration:.3f}",
        )
```

### Custom Configuration Sections

Add new config sections for extensions:

```python
from dataclasses import dataclass, field
from ingestforge.core import Config

@dataclass
class CustomFeatureConfig:
    enabled: bool = True
    option1: str = "default"
    option2: int = 100

# Extend main Config (requires modifying config.py)
@dataclass
class ExtendedConfig(Config):
    custom_feature: CustomFeatureConfig = field(default_factory=CustomFeatureConfig)
```

## Dependencies

### Required
- `pyyaml>=6.0` - YAML configuration parsing
- Python standard library only for most utilities

### Optional
None - core is dependency-free beyond YAML for config

## Testing

### Running Tests

```bash
# Run all core tests
pytest tests/test_core_*.py -v

# Test specific component
pytest tests/test_core_config.py -v
pytest tests/test_core_retry.py -v
pytest tests/test_core_security.py -v

# Test with coverage
pytest tests/test_core_*.py --cov=ingestforge.core --cov-report=html
```

### Key Test Files

- `tests/test_core_config.py` - Configuration loading and environment variables
- `tests/test_core_retry.py` - Retry logic and exponential backoff
- `tests/test_core_security.py` - Path sanitization and SSRF prevention
- `tests/test_core_logging.py` - Structured logging and context tracking
- `tests/test_core_provenance.py` - Citation generation and serialization

## Common Patterns

### Pattern 1: Configuration + Logging Setup

```python
from ingestforge.core import load_config
from ingestforge.core.logging import configure_logging, get_logger

# Initialize application
config = load_config
configure_logging(level="INFO", log_file=config.data_path / "app.log")
logger = get_logger(__name__)

logger.info("Application started", config_source="config.yaml")
```

### Pattern 2: Retry with Logging

```python
from ingestforge.core.retry import retry
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

def on_retry_callback(exception, attempt):
    logger.warning(f"Retry attempt {attempt}", error=str(exception))

@retry(max_attempts=3, on_retry=on_retry_callback)
def call_api:
    ...
```

### Pattern 3: Secure File Processing

```python
from ingestforge.core.security import create_safe_file_ops
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
file_ops = create_safe_file_ops(base_dir=config.data_path)

try:
    content = file_ops.read(user_provided_path)
    logger.info("File read successfully", path=user_provided_path)
except PathTraversalError as e:
    logger.error("Path traversal detected", path=user_provided_path, error=str(e))
```

### Pattern 4: Pipeline with Provenance

```python
from ingestforge.core import Pipeline
from ingestforge.core.provenance import create_pdf_source

# Create source location
source = create_pdf_source(
    file_path="paper.pdf",
    title="Quantum Computing",
    authors=["Alice"],
    page_start=1
)

# Process with provenance
pipeline = Pipeline(config)
result = pipeline.process_document(
    file_path=Path("paper.pdf"),
    source_location=source  # Attached to all chunks
)

# All chunks now have source.to_short_cite available
```

## Troubleshooting

### Issue 1: Configuration Not Loading

**Symptom:** `load_config` returns default values instead of file values

**Cause:** `config.yaml` not found or malformed YAML

**Fix:**
```python
from pathlib import Path

# Verify file exists
config_path = Path("config.yaml")
if not config_path.exists:
    print("Config file not found!")

# Check for YAML syntax errors
import yaml
with open(config_path) as f:
    try:
        yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"YAML syntax error: {e}")
```

### Issue 2: Retry Not Working for Specific Exception

**Symptom:** Function raises exception immediately without retry

**Cause:** Exception not in `retryable_exceptions` tuple

**Fix:**
```python
from ingestforge.core.retry import retry

# Specify exception types explicitly
@retry(retryable_exceptions=(MyCustomError, ConnectionError))
def my_function:
    ...
```

### Issue 3: Path Traversal False Positives

**Symptom:** Valid paths rejected by PathSanitizer

**Cause:** Base directory not resolved correctly

**Fix:**
```python
from ingestforge.core.security import PathSanitizer
from pathlib import Path

# Use resolved absolute path
sanitizer = PathSanitizer(base_dir=Path("/app/data").resolve)

# Or use current working directory
sanitizer = PathSanitizer(base_dir=Path.cwd)
```

### Issue 4: Citations Missing Author Information

**Symptom:** Citations show "Unknown" instead of author name

**Cause:** Author parsing failed or author data incomplete

**Fix:**
```python
from ingestforge.core.provenance import Author

# Provide first_name and last_name explicitly
author = Author(
    name="Jane Smith",
    first_name="Jane",
    last_name="Smith"
)

# Or let Author parse it
author = Author(name="Jane Smith")  # Auto-parses to first/last
```

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System-wide architecture overview
- [config.yaml.example](../../config.yaml.example) - Full configuration example
- [ADR-004: Citation and Provenance Design](../../docs/architecture/ADR-004-citation-provenance.md)
- [ADR-006: LLM Retry Consolidation](../../docs/architecture/ADR-006-llm-retry-consolidation.md)
