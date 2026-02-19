"""
Lifecycle Management Mixin for ChromaDB.

Handles resource cleanup and context manager protocol.
"""

from typing import Any, Optional, Type
from types import TracebackType


class _Logger:
    """Lazy logger holder.

    Rule #6: Encapsulates logger state in smallest scope.
    Avoids slow startup from rich library import.
    """

    _instance = None

    @classmethod
    def get(cls) -> Any:
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


class ChromaDBLifecycleMixin:
    """
    Mixin providing lifecycle management for ChromaDB repository.

    Extracted from chromadb.py to reduce file size (Phase 4, Rule #4).
    """

    def _stop_server_manager(self) -> bool:
        """
        Stop ChromaDB server manager if available.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation (via assertion)
        Rule #9: Full type hints

        Returns:
            True if manager was stopped, False if not available
        """
        assert self._client is not None, "Client must not be None"
        if not hasattr(self._client, "_server"):
            return False
        if not hasattr(self._client._server, "_manager"):
            return False
        self._client._server._manager.stop()
        return True

    def _close_client_directly(self) -> bool:
        """
        Close ChromaDB client if it has a close method.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation (via assertion)
        Rule #9: Full type hints

        Returns:
            True if client was closed, False if close method not available
        """
        assert self._client is not None, "Client must not be None"
        if not hasattr(self._client, "close"):
            return False
        self._client.close()
        return True

    def _cleanup_client_resources(self) -> None:
        """
        Attempt to release ChromaDB client resources.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation (via assertion)
        Rule #9: Full type hints

        Raises:
            Exception: Any error during cleanup (caller should handle)
        """
        assert self._client is not None, "Client must not be None"
        if self._stop_server_manager():
            _Logger.get().debug("ChromaDB server manager stopped")
            return
        if self._close_client_directly():
            _Logger.get().debug("ChromaDB client closed directly")
            return

        # Neither method available
        _Logger.get().debug("No cleanup method available for ChromaDB client")

    def close(self) -> None:
        """
        Release ChromaDB client resources.

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #4: Function <60 lines
        Rule #5: No silent failures - logs all errors
        Rule #8: Resource management in finally block
        Rule #9: Full type hints

        Closes the underlying SQLite connection held by PersistentClient.
        Essential on Windows where open file handles prevent temp directory cleanup.
        """
        if self._client is None:
            _Logger.get().debug("ChromaDB client already closed")
            return

        try:
            self._cleanup_client_resources()

        except Exception as e:
            _Logger.get().debug(f"Error closing ChromaDB client: {e}")

        finally:
            self._client = None
            self._collection = None
            _Logger.get().debug("ChromaDB client references cleared")

    def __enter__(self) -> "ChromaDBLifecycleMixin":
        """Support context manager usage."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Close on context manager exit."""
        self.close()
        return False
