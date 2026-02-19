"""Saved search templates for IngestForge."""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Dict


@dataclass
class SavedSearch:
    """A saved search template."""

    name: str
    query: str
    description: Optional[str] = None
    filters: Optional[Dict[str, str]] = None
    top_k: int = 10
    library: Optional[str] = None
    created_at: Optional[str] = None
    last_used: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedSearch":
        """Create from dictionary."""
        return cls(**data)


class SavedSearchManager:
    """Manage saved search templates."""

    def __init__(self, storage_path: Path) -> None:
        """
        Initialize manager.

        Args:
            storage_path: Path to saved_searches.json
        """
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, search: SavedSearch) -> None:
        """
        Save a search template.

        Args:
            search: SavedSearch to save
        """
        searches = self.list_all()

        # Update existing or add new
        existing_idx = None
        for i, s in enumerate(searches):
            if s.name == search.name:
                existing_idx = i
                break

        if existing_idx is not None:
            searches[existing_idx] = search
        else:
            if not search.created_at:
                search.created_at = datetime.now().isoformat()
            searches.append(search)

        # Write to file
        self._write_searches(searches)

    def get(self, name: str) -> Optional[SavedSearch]:
        """
        Get a saved search by name.

        Args:
            name: Search name

        Returns:
            SavedSearch if found, None otherwise
        """
        searches = self.list_all()
        for search in searches:
            if search.name == name:
                # Update last_used timestamp
                search.last_used = datetime.now().isoformat()
                self.save(search)
                return search
        return None

    def delete(self, name: str) -> bool:
        """
        Delete a saved search.

        Args:
            name: Search name

        Returns:
            True if deleted, False if not found
        """
        searches = self.list_all()
        initial_count = len(searches)
        searches = [s for s in searches if s.name != name]

        if len(searches) < initial_count:
            self._write_searches(searches)
            return True
        return False

    def list_all(self) -> List[SavedSearch]:
        """
        List all saved searches.

        Returns:
            List of SavedSearch objects
        """
        if not self.storage_path.exists():
            return []

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [SavedSearch.from_dict(s) for s in data.get("searches", [])]
        except (json.JSONDecodeError, KeyError):
            return []

    def apply_variables(self, query: str, variables: Dict[str, str]) -> str:
        """
        Apply template variables to query.

        Args:
            query: Query template with {variables}
            variables: Dictionary of variable values

        Returns:
            Query with variables substituted

        Example:
            >>> apply_variables("research on {topic} in {year}", {"topic": "AI", "year": "2023"})
            "research on AI in 2023"
        """
        result = query
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, value)
        return result

    def _write_searches(self, searches: List[SavedSearch]) -> None:
        """Write searches to storage file."""
        data = {
            "searches": [asdict(s) for s in searches],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
