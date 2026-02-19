"""
Query result caching for IngestForge.

Provides in-memory and optional persistent caching of query results
to improve response times for repeated queries.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.storage.base import SearchResult

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: List[Dict[str, Any]]
    created_at: float
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > ttl_seconds

    def touch(self) -> None:
        """Update access time and hit count."""
        self.last_accessed = time.time()
        self.hits += 1


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    max_size: int = 1000  # Maximum entries
    ttl_seconds: float = 3600  # 1 hour default
    persist_path: Optional[Path] = None  # Optional file persistence


class QueryCache:
    """
    LRU cache for query results.

    Features:
    - Thread-safe operations
    - TTL-based expiration
    - LRU eviction when full
    - Optional disk persistence
    - Cache statistics
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

        # Load from disk if persistence enabled
        if self.config.persist_path and self.config.persist_path.exists():
            self._load_from_disk()

    def _make_key(self, query: str, options: Optional[Dict] = None) -> str:
        """Generate cache key from query and options."""
        key_parts = [query.lower().strip()]
        if options:
            key_parts.append(json.dumps(options, sort_keys=True))
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(
        self,
        query: str,
        options: Optional[Dict] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results for a query.

        Args:
            query: The search query.
            options: Optional query options (top_k, etc.).

        Returns:
            Cached results or None if not found/expired.
        """
        if not self.config.enabled:
            return None

        key = self._make_key(query, options)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired(self.config.ttl_seconds):
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        query: str,
        results: List[SearchResult],
        options: Optional[Dict] = None,
    ) -> None:
        """
        Cache query results.

        Args:
            query: The search query.
            results: Search results to cache.
            options: Optional query options.
        """
        if not self.config.enabled:
            return

        key = self._make_key(query, options)

        # Convert SearchResult objects to dicts for serialization
        value = [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.config.max_size:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
            )

        # Persist if enabled
        if self.config.persist_path:
            self._save_to_disk()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]
        self._stats["evictions"] += 1

    def invalidate(self, query: Optional[str] = None) -> None:
        """
        Invalidate cache entries.

        Args:
            query: Specific query to invalidate, or None for all.
        """
        with self._lock:
            if query is None:
                self._cache.clear()
            else:
                # Try to find and remove matching entries
                key = self._make_key(query)
                if key in self._cache:
                    del self._cache[key]

    def invalidate_document(self, document_id: str) -> None:
        """
        Invalidate all cache entries that might contain results
        from a specific document.

        Args:
            document_id: The document ID to invalidate.
        """
        # For simplicity, clear entire cache when document changes
        # A more sophisticated approach would track document->cache mappings
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0

            return {
                "entries": len(self._cache),
                "max_size": self.config.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": f"{hit_rate:.2%}",
            }

    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.config.persist_path:
            return

        try:
            data = {
                key: {
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "hits": entry.hits,
                }
                for key, entry in self._cache.items()
            }

            self.config.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.persist_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Failed to persist cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.config.persist_path or not self.config.persist_path.exists():
            return

        try:
            with open(self.config.persist_path) as f:
                data = json.load(f)

            for key, entry_data in data.items():
                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    created_at=entry_data["created_at"],
                    hits=entry_data.get("hits", 0),
                )

                # Only load non-expired entries
                if not entry.is_expired(self.config.ttl_seconds):
                    self._cache[key] = entry
        except Exception as e:
            logger.debug(f"Failed to load cache from disk: {e}")
