"""
Data models for the web curation system.

Defines the state machine, data structures for curation items and sessions,
and the session persistence manager.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


class CurationState(str, Enum):
    """State machine for curation workflow."""

    IDLE = "idle"  # Ready to start new search
    SEARCHING = "searching"  # Executing web search
    PREVIEWING = "previewing"  # Fetching URL and displaying preview
    WAITING_FOR_DECISION = "waiting"  # Paused for user input
    INGESTING = "ingesting"  # Processing URL through Pipeline
    SKIPPING = "skipping"  # User chose skip; advancing queue
    SKIPPING_ERROR = "skipping_error"  # Fetch/ingest failed; allow skip
    COMPLETE = "complete"  # All items processed or user quit


class CurationDecision(str, Enum):
    """User decision for a curation item."""

    PENDING = "pending"  # Not yet reviewed
    INGEST = "ingest"  # User chose to ingest
    SKIP = "skip"  # User chose to skip
    SKIP_ERROR = "skip_error"  # Skipped due to error


@dataclass
class CurationItem:
    """A single item in the curation queue."""

    id: str  # URL hash
    url: str
    title: str
    snippet: str
    domain: str = ""
    relevance_score: float = 0.0
    is_educational: bool = False
    preview_text: Optional[str] = None  # First ~500 chars of content
    word_count: Optional[int] = None
    fetch_error: Optional[str] = None
    decision: CurationDecision = CurationDecision.PENDING
    chunks_created: int = 0
    document_id: Optional[str] = None
    processed_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc.lower()
        if not self.id:
            self.id = hashlib.md5(self.url.encode()).hexdigest()[:12]
        # Handle string enums from JSON deserialization
        if isinstance(self.decision, str):
            self.decision = CurationDecision(self.decision)

    @classmethod
    def from_search_result(cls, result: Any) -> "CurationItem":
        """Create CurationItem from a SearchResult."""
        return cls(
            id=hashlib.md5(result.url.encode()).hexdigest()[:12],
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            domain=result.domain,
            relevance_score=result.relevance_score,
            is_educational=result.is_educational,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data["decision"] = self.decision.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurationItem":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class CurationSession:
    """A curation session with query, state, and item queue."""

    id: str
    query: str
    state: CurationState = CurationState.IDLE
    current_index: int = 0  # Position in queue
    items: List[CurationItem] = field(default_factory=list)
    total_ingested: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    created_at: str = ""
    updated_at: str = ""
    academic_mode: bool = False

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.updated_at:
            self.updated_at = self.created_at
        # Handle string enums from JSON deserialization
        if isinstance(self.state, str):
            self.state = CurationState(self.state)

    @property
    def current_item(self) -> Optional[CurationItem]:
        """Get the current item in the queue."""
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return None

    @property
    def remaining_count(self) -> int:
        """Number of items not yet processed."""
        return sum(
            1 for item in self.items if item.decision == CurationDecision.PENDING
        )

    @property
    def is_complete(self) -> bool:
        """Check if all items have been processed."""
        return self.current_index >= len(self.items)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "state": self.state.value,
            "current_index": self.current_index,
            "items": [item.to_dict() for item in self.items],
            "total_ingested": self.total_ingested,
            "total_skipped": self.total_skipped,
            "total_errors": self.total_errors,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "academic_mode": self.academic_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurationSession":
        """Deserialize from dictionary."""
        items = [CurationItem.from_dict(item) for item in data.get("items", [])]
        return cls(
            id=data["id"],
            query=data["query"],
            state=data.get("state", CurationState.IDLE.value),
            current_index=data.get("current_index", 0),
            items=items,
            total_ingested=data.get("total_ingested", 0),
            total_skipped=data.get("total_skipped", 0),
            total_errors=data.get("total_errors", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            academic_mode=data.get("academic_mode", False),
        )


class CurationSessionManager:
    """
    Manage persistence and retrieval of curation sessions.

    Sessions are saved as JSON files in .ingest/data/curation_sessions/
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """
        Initialize the session manager.

        Args:
            base_path: Project base path. Defaults to current directory.
        """
        self.base_path = base_path or Path.cwd()
        self.sessions_dir = self.base_path / ".ingest" / "data" / "curation_sessions"

    def _ensure_dir(self) -> None:
        """Ensure sessions directory exists."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.sessions_dir / f"curate_{session_id}.json"

    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid

        return uuid.uuid4().hex[:8]

    def save(self, session: CurationSession) -> None:
        """Save a session to disk."""
        self._ensure_dir()
        session.touch()
        path = self._session_path(session.id)
        path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

    def load(self, session_id: str) -> Optional[CurationSession]:
        """Load a session from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CurationSession.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def delete(self, session_id: str) -> bool:
        """Delete a session from disk."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> List[CurationSession]:
        """List all saved sessions, sorted by most recent first."""
        self._ensure_dir()
        sessions = []
        for path in self.sessions_dir.glob("curate_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(CurationSession.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def get_latest_incomplete(self) -> Optional[CurationSession]:
        """Get the most recent incomplete session, if any."""
        for session in self.list_sessions():
            if session.state not in (CurationState.COMPLETE, CurationState.IDLE):
                return session
        return None

    def cleanup_old_sessions(self, keep_days: int = 30) -> None:
        """Remove sessions older than keep_days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

        for session in self.list_sessions():
            if session.updated_at < cutoff_str:
                self.delete(session.id)
