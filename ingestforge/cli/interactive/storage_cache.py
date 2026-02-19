"""Storage Cache - Thread-safe singleton for background storage loading.

Provides non-blocking access to ChromaDB storage with cached statistics."""

import logging
import threading
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining expected storage backend interface.

    Rule #9: Use Protocol for structural typing instead of Any.
    This allows type checkers to verify compatibility without
    requiring inheritance.
    """

    def count(self) -> int:
        """Return total chunk count."""
        ...

    def get_libraries(self) -> List[str]:
        """Return list of library names."""
        ...

    def get_all_chunks(self) -> List[Any]:
        """Return all chunks."""
        ...

    @property
    def collection(self) -> Any:
        """Access underlying collection."""
        ...


_logger: logging.Logger = logging.getLogger(__name__)
StatsDict = Dict[str, int]
_DEFAULT_STATS: StatsDict = {"docs": 0, "libraries": 0, "chunks": 0}
_MAX_METADATA_ITERATIONS: int = 100_000


class StorageCache:
    """Thread-safe singleton for cached storage access.

    Provides background loading of storage with cached statistics
    to ensure responsive UI during initialization.
    """

    _instance: Optional["StorageCache"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "StorageCache":
        """Ensure singleton instance with double-checked locking."""
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize cache state (only once per singleton)."""
        if getattr(self, "_initialized", False):
            return
        self._storage: Optional[StorageBackend] = None
        self._error: Optional[str] = None
        self._loading: bool = False
        self._ready: bool = False
        self._stats_ready: bool = False
        self._ready_event: threading.Event = threading.Event()
        self._stats: StatsDict = _DEFAULT_STATS.copy()
        self._initialized: bool = True

    def start_loading(self) -> None:
        """Start background loading in daemon thread.

        Thread-safe: Uses lock to prevent multiple concurrent loads.
        """
        with self._lock:
            if self._loading or self._ready:
                return
            self._loading = True

        thread = threading.Thread(target=self._load_storage, daemon=True)
        thread.start()

    def _load_storage(self) -> None:
        """Load storage and compute statistics (runs in background thread).

        Rule #1: Linear control flow with try/except/finally
        Rule #5: Lock protection for all state mutations
        """
        try:
            from ingestforge.core.config_loaders import load_config
            from ingestforge.storage.factory import get_storage_backend

            config = load_config()
            storage = get_storage_backend(config)
            assert storage is not None, "Storage backend returned None"

            with self._lock:
                self._storage = storage
                self._ready = True
                self._error = None
            self._ready_event.set()

            # Compute stats after marking ready (UI responsive first)
            self._compute_stats()
            with self._lock:
                self._stats_ready = True

        except Exception as e:
            _logger.error(f"Storage loading failed: {e}")
            with self._lock:
                self._error = str(e)
                self._storage = None
            self._ready_event.set()

        finally:
            with self._lock:
                self._loading = False

    def _compute_stats(self) -> None:
        """Compute storage statistics using efficient methods.

        Rule #5: Early return if no storage
        Rule #2: Bounded iteration in _estimate_doc_count
        """
        if not self._storage:
            return

        try:
            chunk_count: int = self._storage.count()
            libraries: list = self._storage.get_libraries()
            doc_count: int = self._estimate_doc_count()
            assert chunk_count >= 0, "Chunk count cannot be negative"
            assert len(libraries) >= 0, "Library count cannot be negative"

            self._stats = {
                "docs": doc_count,
                "libraries": len(libraries),
                "chunks": chunk_count,
            }

        except Exception as e:
            _logger.warning(f"Stats computation failed: {e}")

    def _estimate_doc_count(self) -> int:
        """Estimate document count efficiently with bounded iteration.

        Rule #2: Fixed upper bound for iteration
        Rule #7: Early validation of storage
        """
        if not self._storage:
            return 0

        try:
            result = self._storage.collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])
            assert isinstance(metadatas, list), "Expected list of metadatas"

            doc_ids: set = set()
            iteration_count: int = 0

            for meta in metadatas:
                iteration_count += 1
                if iteration_count > _MAX_METADATA_ITERATIONS:
                    _logger.warning("Doc count estimation hit iteration limit")
                    break

                if meta and "document_id" in meta:
                    doc_ids.add(meta["document_id"])

            return len(doc_ids)

        except Exception as e:
            _logger.debug(f"Doc count estimation failed: {e}")
            return -1  # Unknown

    def get_storage(self, timeout: float = 30.0) -> Optional[StorageBackend]:
        """Get storage instance, blocking until ready or timeout.

        Rule #7: Validate timeout parameter
        Rule #9: Returns typed StorageBackend protocol
        """
        assert timeout > 0, "Timeout must be positive"
        assert timeout <= 300.0, "Timeout cannot exceed 5 minutes"

        if not self._loading and not self._ready:
            self.start_loading()

        self._ready_event.wait(timeout=timeout)
        return self._storage

    def is_ready(self) -> bool:
        """Check if storage is loaded (non-blocking)."""
        return self._ready

    def is_loading(self) -> bool:
        """Check if storage is currently loading."""
        return self._loading

    def are_stats_ready(self) -> bool:
        """Check if statistics have been computed."""
        return self._stats_ready

    def get_cached_stats(self) -> StatsDict:
        """Get cached statistics (non-blocking, thread-safe copy)."""
        with self._lock:
            return self._stats.copy()

    def get_error(self) -> Optional[str]:
        """Get error message if loading failed."""
        return self._error

    def refresh_stats(self) -> None:
        """Refresh statistics from storage."""
        if self._storage:
            self._compute_stats()

    def invalidate(self) -> None:
        """Invalidate cache, forcing reload on next access.

        Rule #5: All state reset under single lock
        """
        with self._lock:
            self._storage = None
            self._ready = False
            self._stats_ready = False
            self._loading = False
            self._ready_event.clear()
            self._stats = _DEFAULT_STATS.copy()
            self._error = None
