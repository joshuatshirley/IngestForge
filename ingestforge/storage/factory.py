"""
Storage Backend Factory.

Provides factory functions for creating and managing storage backends.
IngestForge supports multiple storage backends (JSONL, ChromaDB, PostgreSQL),
and this module centralizes backend selection and instantiation.

Architecture Context
--------------------
Storage sits at the bottom of the pipeline: chunks flow through
Split → Extract → Chunk → Enrich → **Index (Storage)**.

    get_storage_backend(config)
        ├── "jsonl"    → JSONLRepository  (file-based, no dependencies)
        ├── "chromadb" → ChromaDBRepository  (vector search, requires chromadb)
        └── "postgres" → PostgresRepository  (pgvector, requires psycopg2)

The factory uses lazy imports so that optional dependencies (chromadb)
are only required when the corresponding backend is selected.

Backend Registration (Plugin Pattern)
-------------------------------------
Custom backends can be registered using register_backend():

    from ingestforge.storage.factory import register_backend

    def create_my_backend(config):
        return MyCustomRepository(config)

    register_backend("mybackend", create_my_backend)

    # Now usable in config:
    # storage:
    #   backend: mybackend

Usage
-----
    from ingestforge.storage.factory import get_storage_backend, check_health

    repo = get_storage_backend(config)
    repo.add_chunks(chunks)
    results = repo.search("query", top_k=10)

    # Verify backend is reachable
    health = check_health(config)
    if not health["healthy"]:
        print(f"Backend issue: {health['error']}")
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ingestforge.core.config import Config
from ingestforge.storage.base import ChunkRepository

# Type alias for backend creator functions
BackendCreator = Callable[[Config], ChunkRepository]

# Registry for backend creators (plugin pattern)
_backend_registry: Dict[str, BackendCreator] = {}


def register_backend(name: str, creator: BackendCreator) -> None:
    """Register a custom storage backend.

    Allows plugins to add new storage backends without modifying this module.

    Args:
        name: Backend identifier (e.g., "mybackend")
        creator: Function that takes Config and returns ChunkRepository

    Example:
        def create_redis_backend(config):
            return RedisRepository(config.storage.redis.url)

        register_backend("redis", create_redis_backend)
    """
    _backend_registry[name.lower()] = creator
    _Logger.get().info(f"Registered storage backend: {name}")


def unregister_backend(name: str) -> bool:
    """Unregister a custom storage backend.

    Args:
        name: Backend identifier to remove

    Returns:
        True if backend was removed, False if not found
    """
    return _backend_registry.pop(name.lower(), None) is not None


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns:
        List of backend names (builtin + registered)
    """
    builtins = ["jsonl", "chromadb", "postgres"]
    custom = list(_backend_registry.keys())
    return builtins + [b for b in custom if b not in builtins]


class _Logger:
    """Lazy logger holder.

    Rule #6: Encapsulates logger state in smallest scope.
    Avoids slow startup from rich library import.
    """

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


def get_storage_backend(
    config: Config,
    backend: Optional[str] = None,
) -> ChunkRepository:
    """
    Get storage backend based on configuration.

    Rule #1: Max 3 nesting levels via early returns and helper extraction.

    Checks registered backends first (plugin pattern), then falls back to
    builtin backends.

    Args:
        config: IngestForge configuration
        backend: Override backend type (jsonl, chromadb, or registered name)

    Returns:
        ChunkRepository instance

    Raises:
        ValueError: If backend type is unknown
        ImportError: If required dependencies are missing
    """
    backend_type = (backend or config.storage.backend).lower()

    # Check registered backends first (plugin pattern)
    if backend_type in _backend_registry:
        _Logger.get().info(f"Using registered storage backend: {backend_type}")
        return _backend_registry[backend_type](config)

    # Builtin backends
    if backend_type == "jsonl":
        from ingestforge.storage.jsonl import JSONLRepository

        _Logger.get().info("Using JSONL storage backend")
        return JSONLRepository(config.data_path)

    if backend_type == "chromadb":
        return _create_chromadb_backend(config)

    if backend_type == "postgres":
        return _create_postgres_backend(config)

    raise ValueError(
        f"Unknown storage backend: {backend_type}. "
        f"Available backends: {', '.join(list_backends())}"
    )


def _create_chromadb_backend(config: Config) -> ChunkRepository:
    """Create ChromaDB backend.

    Rule #1: Extracted to reduce nesting in get_storage_backend.

    Args:
        config: IngestForge configuration

    Returns:
        ChromaDBRepository instance
    """
    try:
        from ingestforge.storage.chromadb import ChromaDBRepository
    except ImportError:
        raise ImportError(
            "chromadb is required for the ChromaDB backend. "
            "Install with: pip install 'ingestforge[vectordb]'\n"
            "Or switch to JSONL storage: set storage.backend = 'jsonl' in config.yaml"
        ) from None
    _Logger.get().info("Using ChromaDB storage backend")
    return ChromaDBRepository(
        persist_directory=config.chromadb_path,
        collection_name=f"{config.project.name}_chunks",
    )


def _create_postgres_backend(config: Config) -> ChunkRepository:
    """Create PostgreSQL backend.

    Rule #1: Extracted to reduce nesting in get_storage_backend.

    Args:
        config: IngestForge configuration

    Returns:
        PostgresRepository instance
    """
    try:
        from ingestforge.storage.postgres import PostgresRepository
    except ImportError:
        raise ImportError(
            "psycopg2 is required for the PostgreSQL backend. "
            "Install with: pip install psycopg2-binary\n"
            "Or switch to JSONL/ChromaDB storage."
        ) from None

    connection_string = config.storage.postgres.connection_string
    if not connection_string:
        raise ValueError(
            "PostgreSQL connection string not configured. "
            "Set storage.postgres.connection_string in config.yaml"
        )

    _Logger.get().info("Using PostgreSQL storage backend with pgvector")
    return PostgresRepository(
        connection_string=connection_string,
        table_name=f"{config.project.name}_chunks",
    )


def check_health(
    config: Config,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check if the selected storage backend is available and reachable.

    Rule #1: Max 3 nesting levels via early returns and helper extraction.

    Performs backend-specific health checks:
    - jsonl: Verifies data directory exists and is writable
    - chromadb: Tests connection to ChromaDB
    - postgres: Tests database connection

    Args:
        config: IngestForge configuration
        backend: Override backend type (defaults to config.storage.backend)

    Returns:
        Dict with keys:
        - healthy (bool): True if backend is operational
        - backend (str): Backend type checked
        - error (str|None): Error message if unhealthy
        - details (dict): Backend-specific details
    """
    backend_type = (backend or config.storage.backend).lower()

    result: Dict[str, Any] = {
        "healthy": False,
        "backend": backend_type,
        "error": None,
        "details": {},
    }

    try:
        result = _perform_health_check(config, backend_type, result)
    except Exception as e:
        result["error"] = str(e)
        _Logger.get().warning(f"Health check failed for {backend_type}: {e}")

    return result


def _perform_health_check(
    config: Config,
    backend_type: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Perform backend-specific health check.

    Args:
        config: IngestForge configuration
        backend_type: Backend type to check
        result: Result dictionary to update

    Returns:
        Updated result dictionary
    """
    if backend_type == "jsonl":
        return _check_jsonl_health(config, result)

    if backend_type == "chromadb":
        return _check_chromadb_health(config, result)

    if backend_type == "postgres":
        return _check_postgres_health(config, result)

    if backend_type in _backend_registry:
        return _check_registered_backend(config, backend_type, result)

    result["error"] = f"Unknown backend: {backend_type}"
    return result


def _check_registered_backend(
    config: Config,
    backend_type: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Check health of registered backend.

    Args:
        config: IngestForge configuration
        backend_type: Backend type
        result: Result dictionary to update

    Returns:
        Updated result dictionary
    """
    repo = _backend_registry[backend_type](config)
    result["healthy"] = True
    result["details"]["count"] = repo.count()
    return result


def _check_jsonl_health(config: Config, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check JSONL backend health."""
    import os

    data_path = config.data_path
    result["details"]["path"] = str(data_path)

    # Check directory exists or can be created
    if not data_path.exists():
        try:
            data_path.mkdir(parents=True, exist_ok=True)
            result["details"]["created"] = True
        except OSError as e:
            result["error"] = f"Cannot create data directory: {e}"
            return result

    # Check writable
    if os.access(data_path, os.W_OK):
        result["healthy"] = True
        result["details"]["writable"] = True
    else:
        result["error"] = f"Data directory not writable: {data_path}"

    return result


def _check_chromadb_health(config: Config, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check ChromaDB backend health."""
    try:
        import chromadb
    except ImportError:
        result["error"] = "chromadb not installed"
        return result

    persist_dir = config.chromadb_path
    result["details"]["persist_directory"] = str(persist_dir)

    try:
        # Try to create/connect to ChromaDB
        client = chromadb.PersistentClient(path=str(persist_dir))
        result["details"]["collections"] = len(client.list_collections())
        result["healthy"] = True
    except Exception as e:
        result["error"] = f"ChromaDB connection failed: {e}"

    return result


def _check_postgres_health(config: Config, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check PostgreSQL backend health."""
    conn_string = config.storage.postgres.connection_string

    if not conn_string:
        result["error"] = "PostgreSQL connection string not configured"
        return result

    # Mask password in details
    result["details"]["connection"] = (
        conn_string.split("@")[-1] if "@" in conn_string else "configured"
    )

    try:
        import psycopg2
    except ImportError:
        result["error"] = "psycopg2 not installed"
        return result

    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        result["healthy"] = True
    except Exception as e:
        result["error"] = f"PostgreSQL connection failed: {e}"

    return result


def detect_best_backend(data_path: Path) -> str:
    """
    Detect the best available backend.

    Args:
        data_path: Path to data directory

    Returns:
        Backend name (chromadb, jsonl)
    """
    # Check for ChromaDB
    try:
        import chromadb

        return "chromadb"
    except ImportError as e:
        _Logger.get().debug(f"chromadb not available, defaulting to jsonl: {e}")

    # Fall back to JSONL
    return "jsonl"


def migrate_backend(
    source: ChunkRepository,
    target: ChunkRepository,
    batch_size: int = 100,
    verify: bool = True,
) -> int:
    """Migrate data from one backend to another.

    Args:
        source: Source repository to read chunks from.
        target: Target repository to write chunks to.
        batch_size: Number of chunks to transfer per batch.
        verify: Whether to verify migration success.

    Returns:
        Number of chunks migrated.
    """
    from ingestforge.storage.migration import migrate_storage

    result = migrate_storage(source, target, batch_size, verify)
    return result.chunks_migrated


class StorageFactory:
    """Factory class for storage backends.

    Provides a class-based interface to the storage factory functions.
    For simpler usage, prefer the module-level functions directly.
    """

    @staticmethod
    def create(config: Config, backend: Optional[str] = None) -> ChunkRepository:
        """Create a storage backend.

        Args:
            config: IngestForge configuration
            backend: Override backend type (jsonl, chromadb)

        Returns:
            ChunkRepository instance
        """
        return get_storage_backend(config, backend)

    @staticmethod
    def detect_best(data_path: Path) -> str:
        """Detect the best available backend.

        Args:
            data_path: Path to data directory

        Returns:
            Backend name (chromadb, jsonl)
        """
        return detect_best_backend(data_path)

    @staticmethod
    def check_health(config: Config, backend: Optional[str] = None) -> Dict[str, Any]:
        """Check if a storage backend is healthy and reachable.

        Args:
            config: IngestForge configuration
            backend: Override backend type

        Returns:
            Dict with health status (see check_health function)
        """
        return check_health(config, backend)

    @staticmethod
    def register(name: str, creator: BackendCreator) -> None:
        """Register a custom storage backend.

        Args:
            name: Backend identifier
            creator: Factory function that creates the repository
        """
        register_backend(name, creator)

    @staticmethod
    def list_available() -> list[str]:
        """List all available backend names.

        Returns:
            List of backend names (builtin + registered)
        """
        return list_backends()
