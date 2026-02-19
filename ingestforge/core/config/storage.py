"""
Storage configuration.

Provides configuration for storage backends: JSONL, ChromaDB, PostgreSQL.
Includes backend-specific settings and compression options.
"""

from dataclasses import dataclass, field


@dataclass
class ChromaDBConfig:
    """ChromaDB storage configuration."""

    persist_directory: str = ".data/chromadb"


@dataclass
class PostgresConfig:
    """PostgreSQL storage configuration."""

    connection_string: str = ""
    table_name: str = "chunks"
    embedding_dim: int = 384
    min_pool_size: int = 1
    max_pool_size: int = 10


@dataclass
class StorageConfig:
    """Storage backend configuration."""

    backend: str = "chromadb"  # jsonl, chromadb, postgres
    compression: bool = False  # Enable gzip compression for JSONL storage
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
