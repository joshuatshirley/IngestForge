"""
Configuration Management for IngestForge.

This module provides the application's configuration system using a hierarchy
of dataclasses that map to YAML configuration files. It supports environment
variable expansion for secrets and deployment-specific values.

Public API
----------
All configuration classes and loading functions are re-exported here for
backward compatibility. Existing imports continue to work:

    from ingestforge.core.config import Config, load_config
    from ingestforge.core.config import ChunkingConfig, LLMConfig

Architecture
------------
Configuration is organized into domain-specific modules:

    config/
    ├── base.py          # ProjectConfig, IngestConfig, SplitConfig
    ├── chunking.py      # ChunkingConfig, RefinementConfig
    ├── enrichment.py    # EnrichmentConfig
    ├── storage.py       # StorageConfig, ChromaDBConfig, PostgresConfig
    ├── retrieval.py     # RetrievalConfig, HybridConfig, ParentDocConfig
    ├── llm.py           # LLMConfig, LLMProviderConfig, LlamaCppConfig
    ├── features.py      # OCRConfig, APIConfig, ResearchConfig, etc.
    └── config.py        # Main Config class

Usage Example
-------------
    # Load from default location (./config.yaml)
    config = load_config()

    # Access nested settings
    model = config.enrichment.embedding_model
    top_k = config.retrieval.top_k
"""

# Main Config class
from ingestforge.core.config.config import Config

# Base configs
from ingestforge.core.config.base import IngestConfig, ProjectConfig, SplitConfig

# Chunking configs
from ingestforge.core.config.chunking import ChunkingConfig, RefinementConfig

# Enrichment config
from ingestforge.core.config.enrichment import EnrichmentConfig

# Storage configs
from ingestforge.core.config.storage import (
    ChromaDBConfig,
    PostgresConfig,
    StorageConfig,
)

# Retrieval configs
from ingestforge.core.config.retrieval import (
    AuthorityConfig,
    HybridConfig,
    ParentDocConfig,
    RetrievalConfig,
)

# LLM configs
from ingestforge.core.config.llm import LlamaCppConfig, LLMConfig, LLMProviderConfig

# Feature configs
from ingestforge.core.config.features import (
    APIConfig,
    DoctrineAPIConfig,
    FeatureAnalysisConfig,
    LiteraryAnalysisConfig,
    LiteraryConfig,
    LiteraryScrapingConfig,
    OCRConfig,
    RedactionConfig,
    ResearchConfig,
    WebSearchConfig,
)

__all__ = [
    # Main config
    "Config",
    # Base configs
    "ProjectConfig",
    "IngestConfig",
    "SplitConfig",
    # Chunking
    "ChunkingConfig",
    "RefinementConfig",
    # Enrichment
    "EnrichmentConfig",
    # Storage
    "StorageConfig",
    "ChromaDBConfig",
    "PostgresConfig",
    # Retrieval
    "RetrievalConfig",
    "HybridConfig",
    "ParentDocConfig",
    "AuthorityConfig",
    # LLM
    "LLMConfig",
    "LLMProviderConfig",
    "LlamaCppConfig",
    # Features
    "APIConfig",
    "OCRConfig",
    "WebSearchConfig",
    "ResearchConfig",
    "LiteraryConfig",
    "LiteraryScrapingConfig",
    "LiteraryAnalysisConfig",
    "DoctrineAPIConfig",
    "FeatureAnalysisConfig",
    "RedactionConfig",
]

# NOTE: Utility functions (load_config, save_config, expand_env_vars, apply_performance_preset)
# must be imported directly from config_loaders to avoid circular imports:
#
#   from ingestforge.core.config_loaders import load_config
#
# Old imports like "from ingestforge.core.config import load_config" need to be updated.
