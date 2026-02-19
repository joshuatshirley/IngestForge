"""
Session persistence for conversations.

Stores conversation sessions as JSON files in .data/conversations/
for resuming previous sessions across CLI invocations.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.query.session import ConversationSession


class SessionStore:
    """JSON-file session persistence in .data/conversations/."""

    def __init__(self, data_path: Path) -> None:
        """
        Initialize session store.

        Args:
            data_path: Base data directory (e.g., .data/)
        """
        self.conversations_dir = data_path / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.conversations_dir / f"{session_id}.json"

    def save(self, session: ConversationSession) -> None:
        """
        Save a session to disk.

        Args:
            session: ConversationSession to persist
        """
        path = self._session_path(session.session_id)
        data = session.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, session_id: str) -> Optional[ConversationSession]:
        """
        Load a session from disk.

        Args:
            session_id: Session identifier

        Returns:
            ConversationSession or None if not found
        """
        path = self._session_path(session_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ConversationSession.from_dict(data)

    def list_sessions(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries sorted by most recent
        """
        sessions = []

        for path in self.conversations_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                turns = data.get("turns", [])
                # Determine last update time
                if turns:
                    updated_at = turns[-1].get("timestamp", data.get("created_at", 0))
                else:
                    updated_at = data.get("created_at", 0)

                # Derive topic from first question
                topic = "New conversation"
                if turns:
                    first_q = turns[0].get("question", "")
                    topic = first_q[:60] + "..." if len(first_q) > 60 else first_q

                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "topic": topic,
                        "turn_count": len(turns),
                        "updated_at": updated_at,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by most recent
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions[offset : offset + limit]

    def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        path = self._session_path(session_id)
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
