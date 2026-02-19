"""Query history tracking for IngestForge."""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class QueryHistoryEntry:
    """A single query history entry."""

    id: str
    timestamp: str
    query: str
    result_count: int
    library_filter: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryHistoryEntry":
        """Create from dict."""
        return cls(**data)


class QueryHistory:
    """Manages query history storage."""

    def __init__(self, history_path: Path) -> None:
        """Initialize with path to history file."""
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def add_query(
        self, query: str, result_count: int, library_filter: Optional[str] = None
    ) -> str:
        """Add a query to history. Returns query ID."""
        timestamp = datetime.now().isoformat()
        entry_id = timestamp.replace(":", "").replace(".", "")[
            :14
        ]  # Simple ID from timestamp

        entry = QueryHistoryEntry(
            id=entry_id,
            timestamp=timestamp,
            query=query,
            result_count=result_count,
            library_filter=library_filter,
        )

        # Append to JSONL file
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

        return entry_id

    def get_history(
        self, limit: Optional[int] = None, search: Optional[str] = None
    ) -> List[QueryHistoryEntry]:
        """Get query history, optionally filtered."""
        if not self.history_path.exists():
            return []

        entries = []
        with open(self.history_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = QueryHistoryEntry.from_dict(json.loads(line.strip()))

                # Apply search filter if provided
                if search and search.lower() not in entry.query.lower():
                    continue

                entries.append(entry)

        # Return most recent first
        entries.reverse()

        if limit:
            entries = entries[:limit]

        return entries

    def get_by_id(self, entry_id: str) -> Optional[QueryHistoryEntry]:
        """Get a specific query by ID."""
        if not self.history_path.exists():
            return None

        with open(self.history_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = QueryHistoryEntry.from_dict(json.loads(line.strip()))
                if entry.id == entry_id:
                    return entry

        return None
