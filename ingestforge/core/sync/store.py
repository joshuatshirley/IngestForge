"""
Persistent storage for sync state.

Provides SyncStateStore for tracking file states across sync operations.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set

from ingestforge.core.logging import get_logger
from ingestforge.core.sync.models import FileState

logger = get_logger(__name__)


class SyncStateStore:
    """Persistent storage for sync state."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, FileState] = {}
        self._load()

    def _load(self) -> None:
        """Load state from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = {
                    k: FileState.from_dict(v) for k, v in data.get("files", {}).items()
                }
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
                self._state = {}

    def _save(self) -> None:
        """Save state to disk."""
        data = {
            "files": {k: v.to_dict() for k, v in self._state.items()},
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get(self, path: str) -> Optional[FileState]:
        """Get file state by path."""
        return self._state.get(path)

    def set(self, state: FileState) -> None:
        """Set file state."""
        self._state[state.path] = state
        self._save()

    def remove(self, path: str) -> bool:
        """Remove file state."""
        if path in self._state:
            del self._state[path]
            self._save()
            return True
        return False

    def get_all_paths(self) -> Set[str]:
        """Get all tracked paths."""
        return set(self._state.keys())

    def get_by_document_id(self, document_id: str) -> Optional[FileState]:
        """Find file state by document ID."""
        for state in self._state.values():
            if state.document_id == document_id:
                return state
        return None

    def clear(self) -> None:
        """Clear all state."""
        self._state = {}
        self._save()
