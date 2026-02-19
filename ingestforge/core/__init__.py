"""
Core Infrastructure for IngestForge.

This module forms the innermost layer of IngestForge's hexagonal architecture,
providing foundational services that all other modules depend on.

Architecture Position
---------------------
    CLI (outermost)
      └── Feature Modules (ingest, chunking, enrichment, storage, retrieval, query, llm)
            └── Shared (patterns, interfaces, utilities)
                  └── **Core** (innermost - you are here)

The Core layer has NO dependencies on other IngestForge modules, making it the
stable foundation upon which everything else is built. Changes here propagate
outward, so this code is designed for stability and minimal churn.

Components
----------
**Configuration (config.py)**
    Manages application settings via nested dataclasses with YAML persistence.
    Supports environment variable expansion (${VAR_NAME}) and validation.

**Logging (logging.py)**
    Structured logging with context binding. Provides specialized loggers for
    pipeline processing (PipelineLogger).

**Retry (retry.py)**
    Resilience decorators for handling transient failures. Pre-configured for
    LLM API calls, embedding generation, and network operations with exponential
    backoff and jitter.

**Security (security.py)**
    Defensive utilities against common attacks:
    - PathSanitizer: Blocks directory traversal (../../../etc/passwd)
    - URLValidator: Prevents SSRF (requests to internal networks)
    - SafeFileOperations: File I/O with automatic path validation

**Provenance (provenance.py)**
    Academic citation and source tracking. Enables precise references down to
    chapter, section, page, and paragraph for scholarly use cases.

**State (state.py)**
    Processing state persistence. Tracks document processing through stages
    (PENDING → COMPLETED) with automatic JSON serialization.

**Jobs (jobs.py)**
    Background job queue with SQLite storage. Supports async processing,
    progress tracking, and automatic retry on failure.

**Pipeline (pipeline.py)**
    Main orchestrator that coordinates: Split → Extract → Chunk → Enrich → Index.
    Lazy-loads components for fast startup and integrates all core services.

Design Principles
-----------------
1. **Zero internal dependencies**: Core depends only on standard library and
   third-party packages, never on other IngestForge modules.

2. **Defensive by default**: Security validations happen automatically rather
   than requiring explicit opt-in.

3. **Lazy initialization**: Heavy components (ML models, DB connections) are
   loaded only when first accessed.

4. **Context-aware logging**: All loggers support key-value binding for
   structured log aggregation.

Usage Example
-------------
    from ingestforge.core import (
        Config, load_config,
        Pipeline,
        PathSanitizer,
    )

    # Load configuration
    config = load_config()

    # Run the full pipeline
    pipeline = Pipeline(config)
    result = pipeline.process_file(Path("document.pdf"))

    # Path operations are automatically validated
    sanitizer = PathSanitizer()
    safe_path = sanitizer.sanitize_path("user/input/../../../etc/passwd")
    # Raises PathTraversalError
"""

# Only import state at module level to avoid circular dependencies
# Config, Pipeline, Jobs, Security, Exceptions, and Constants are lazy-loaded
from typing import Any
from ingestforge.core.state import ProcessingState, DocumentState

__all__ = [
    # Note: Config, load_config, and Pipeline are NOT re-exported here to avoid
    # circular dependencies. Import them directly:
    #   from ingestforge.core.config import Config
    #   from ingestforge.core.config_loaders import load_config
    #   from ingestforge.core.pipeline import Pipeline
    # State
    "ProcessingState",
    "DocumentState",
    # Jobs
    "Job",
    "JobStatus",
    "JobType",
    "SQLiteJobQueue",
    "WorkerPool",
    "create_job",
    "create_job_queue",
    # Security
    "PathSanitizer",
    "sanitize_filename",
    "sanitize_path",
    "URLValidator",
    "validate_url",
    "SafeFileOperations",
    # Exceptions
    "IngestForgeError",
    "SecurityError",
    "PathTraversalError",
    "SSRFError",
    "ProcessingError",
    "ExtractionError",
    "ChunkingError",
    "EnrichmentError",
    "EmbeddingError",
    "LLMError",
    "RateLimitError",
    "ConfigurationError",
    "ContextLengthError",
    "StorageError",
    "RetryError",
    "ValidationError",
    "ConfigValidationError",
    # Constants
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "MIN_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "SUPPORTED_EXTENSIONS",
    "OCR_EXTENSIONS",
    "CODE_EXTENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "VRAM_BATCH_SIZES",
    "CPU_BATCH_SIZE",
    "MODEL_CONTEXT_LENGTHS",
    "LLM_PROVIDER_PRIORITY",
    "DEFAULT_STORAGE_BACKEND",
    "DEFAULT_TOP_K",
    "DEFAULT_BM25_WEIGHT",
    "DEFAULT_SEMANTIC_WEIGHT",
    "SYMBOLS",
    "SYMBOLS_ASCII",
    "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_RETRY_DELAY",
]


# Lazy imports to avoid circular dependencies and slow startup
def _try_load_jobs_item(name: str) -> Any:
    """Try to load a jobs module item.

    Rule #4: No large functions - Extracted from __getattr__
    """
    jobs_items = {
        "Job": "ingestforge.core.jobs",
        "JobStatus": "ingestforge.core.jobs",
        "JobType": "ingestforge.core.jobs",
        "SQLiteJobQueue": "ingestforge.core.jobs",
        "WorkerPool": "ingestforge.core.jobs",
        "create_job": "ingestforge.core.jobs",
        "create_job_queue": "ingestforge.core.jobs",
    }
    if name in jobs_items:
        module = __import__(jobs_items[name], fromlist=[name])
        return getattr(module, name)
    return None


def _try_load_security_item(name: str) -> Any:
    """Try to load a security module item.

    Rule #4: No large functions - Extracted from __getattr__
    """
    security_items = {
        "PathSanitizer": "ingestforge.core.security",
        "sanitize_filename": "ingestforge.core.security",
        "URLValidator": "ingestforge.core.security",
        "SafeFileOperations": "ingestforge.core.security",
    }
    if name in security_items:
        module = __import__(security_items[name], fromlist=[name])
        return getattr(module, name)
    return None


def _try_load_exception_item(name: str) -> Any:
    """Try to load an exception class.

    Rule #4: No large functions - Extracted from __getattr__
    """
    exception_items = [
        "IngestForgeError",
        "SecurityError",
        "PathTraversalError",
        "SSRFError",
        "ProcessingError",
        "ExtractionError",
        "ChunkingError",
        "EnrichmentError",
        "EmbeddingError",
        "LLMError",
        "RateLimitError",
        "ConfigurationError",
        "ContextLengthError",
        "StorageError",
        "RetryError",
        "ValidationError",
        "ConfigValidationError",
    ]
    if name in exception_items:
        return locals()[name]
    return None


def _try_load_constant_item(name: str) -> Any:
    """Try to load a constant.

    Rule #4: No large functions - Extracted from __getattr__
    """
    constant_items = [
        "DEFAULT_CHUNK_SIZE",
        "DEFAULT_CHUNK_OVERLAP",
        "MIN_CHUNK_SIZE",
        "MAX_CHUNK_SIZE",
        "SUPPORTED_EXTENSIONS",
        "OCR_EXTENSIONS",
        "CODE_EXTENSIONS",
        "DEFAULT_EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "VRAM_BATCH_SIZES",
        "CPU_BATCH_SIZE",
        "MODEL_CONTEXT_LENGTHS",
        "LLM_PROVIDER_PRIORITY",
        "DEFAULT_STORAGE_BACKEND",
        "DEFAULT_TOP_K",
        "DEFAULT_BM25_WEIGHT",
        "DEFAULT_SEMANTIC_WEIGHT",
        "SYMBOLS",
        "SYMBOLS_ASCII",
        "DEFAULT_RETRY_ATTEMPTS",
        "DEFAULT_RETRY_DELAY",
    ]
    if name in constant_items:
        return locals()[name]
    return None


def __getattr__(name: str) -> Any:
    """Lazy-load modules to avoid circular dependencies.

    Rule #4: No large functions - Refactored to <60 lines

    Note: Config, load_config, and Pipeline should be imported directly to avoid
    circular dependency issues:
        from ingestforge.core.config import Config
        from ingestforge.core.config_loaders import load_config
        from ingestforge.core.pipeline import Pipeline
    """
    # Try each category of lazy imports
    result = _try_load_jobs_item(name)
    if result is not None:
        return result

    result = _try_load_security_item(name)
    if result is not None:
        return result

    result = _try_load_exception_item(name)
    if result is not None:
        return result

    result = _try_load_constant_item(name)
    if result is not None:
        return result

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
