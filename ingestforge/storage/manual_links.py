"""Manual Link Storage - User-created relationships between entities.

Manual Graph Linker
Provides storage for user-defined relationships that persist across sessions.

NASA JPL Power of Ten Rules:
- Rule #2: Fixed upper bounds (MAX_LINKS)
- Rule #4: Functions <60 lines
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_LINKS = 10000
MAX_RELATION_LENGTH = 100
MAX_ENTITY_LENGTH = 500


@dataclass
class ManualLink:
    """A user-defined relationship between two entities.

    Manual Graph Linker
    Rule #9: Complete type hints.

    Attributes:
        link_id: Unique identifier for this link.
        source_entity: Source entity name or ID.
        target_entity: Target entity name or ID.
        relation: Relationship type (e.g., "related_to", "causes").
        confidence: Confidence score (1.0 for manual links).
        notes: Optional user notes about the relationship.
        created_at: ISO timestamp when created.
    """

    link_id: str
    source_entity: str
    target_entity: str
    relation: str
    confidence: float = 1.0
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "link_id": self.link_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation": self.relation,
            "confidence": self.confidence,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManualLink":
        """Create from dictionary."""
        return cls(
            link_id=data.get("link_id", ""),
            source_entity=data.get("source_entity", ""),
            target_entity=data.get("target_entity", ""),
            relation=data.get("relation", "related_to"),
            confidence=data.get("confidence", 1.0),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a unique link ID."""
        return f"link_{uuid.uuid4().hex[:12]}"


def validate_entity(entity: str, field_name: str) -> str:
    """Validate and sanitize entity string.

    Rule #7: Parameter validation.

    Args:
        entity: Entity string to validate.
        field_name: Field name for error messages.

    Returns:
        Sanitized entity string.

    Raises:
        ValueError: If entity is invalid.
    """
    if not entity or not entity.strip():
        raise ValueError(f"{field_name} cannot be empty")

    cleaned = entity.strip()
    if len(cleaned) > MAX_ENTITY_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length of {MAX_ENTITY_LENGTH}")

    return cleaned


def validate_relation(relation: str) -> str:
    """Validate and sanitize relation string.

    Rule #7: Parameter validation.

    Args:
        relation: Relation string to validate.

    Returns:
        Sanitized relation string.

    Raises:
        ValueError: If relation is invalid.
    """
    if not relation or not relation.strip():
        raise ValueError("Relation cannot be empty")

    cleaned = relation.strip().lower().replace(" ", "_")
    if len(cleaned) > MAX_RELATION_LENGTH:
        raise ValueError(f"Relation exceeds maximum length of {MAX_RELATION_LENGTH}")

    return cleaned


class ManualLinkManager:
    """Manage persistent manual links stored in JSON.

    Manual Graph Linker
    Stores links in .data/manual_links.json with atomic writes.

    Rule #4: Functions <60 lines.
    """

    DEFAULT_FILE = "manual_links.json"

    def __init__(self, data_dir: Path) -> None:
        """Initialize link manager.

        Args:
            data_dir: Directory for data storage (e.g., .data/).
        """
        self.data_dir = data_dir
        self.file_path = data_dir / self.DEFAULT_FILE

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing links
        self._links: Dict[str, ManualLink] = {}
        self._load()

    def _load(self) -> None:
        """Load links from disk."""
        if not self.file_path.exists():
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for link_data in data.get("links", []):
                link = ManualLink.from_dict(link_data)
                self._links[link.link_id] = link

            logger.debug(f"Loaded {len(self._links)} manual links")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse links file: {e}")
        except Exception as e:
            logger.error(f"Failed to load links: {e}")

    def _save(self) -> None:
        """Save links to disk atomically.

        Uses write-to-temp-then-rename pattern for atomic writes.
        """
        temp_path = self.file_path.with_suffix(".json.tmp")

        try:
            data = {
                "version": 1,
                "links": [link.to_dict() for link in self._links.values()],
            }

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_path.replace(self.file_path)
            logger.debug(f"Saved {len(self._links)} manual links")

        except Exception as e:
            logger.error(f"Failed to save links: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def add(
        self,
        source_entity: str,
        target_entity: str,
        relation: str,
        notes: str = "",
    ) -> ManualLink:
        """Add a new manual link.

        Create manual edges between entities.
        Rule #2: Enforce maximum links.
        Rule #7: Parameter validation.

        Args:
            source_entity: Source entity name.
            target_entity: Target entity name.
            relation: Relationship type.
            notes: Optional notes about the link.

        Returns:
            Created ManualLink.

        Raises:
            ValueError: If parameters are invalid or limit reached.
        """
        if len(self._links) >= MAX_LINKS:
            raise ValueError(f"Maximum link count ({MAX_LINKS}) reached")

        # Validate inputs
        source = validate_entity(source_entity, "source_entity")
        target = validate_entity(target_entity, "target_entity")
        rel = validate_relation(relation)

        # Check for duplicate
        for link in self._links.values():
            if (
                link.source_entity == source
                and link.target_entity == target
                and link.relation == rel
            ):
                raise ValueError("Link already exists")

        # Create link
        link = ManualLink(
            link_id=ManualLink.generate_id(),
            source_entity=source,
            target_entity=target,
            relation=rel,
            notes=notes.strip() if notes else "",
        )

        self._links[link.link_id] = link
        self._save()

        logger.info(f"Created link: {source} --[{rel}]--> {target}")
        return link

    def get(self, link_id: str) -> Optional[ManualLink]:
        """Get link by ID.

        Args:
            link_id: Link identifier.

        Returns:
            ManualLink or None if not found.
        """
        return self._links.get(link_id)

    def get_for_entity(self, entity: str) -> List[ManualLink]:
        """Get all links involving an entity.

        Args:
            entity: Entity name to search for.

        Returns:
            List of links (sorted by created_at).
        """
        entity_lower = entity.lower()
        links = [
            link
            for link in self._links.values()
            if link.source_entity.lower() == entity_lower
            or link.target_entity.lower() == entity_lower
        ]
        return sorted(links, key=lambda l: l.created_at)

    def delete(self, link_id: str) -> bool:
        """Delete a link.

        Args:
            link_id: Link to delete.

        Returns:
            True if deleted, False if not found.
        """
        if link_id not in self._links:
            return False

        del self._links[link_id]
        self._save()

        logger.info(f"Deleted link {link_id}")
        return True

    def list_all(self, limit: int = 100) -> List[ManualLink]:
        """List all links.

        Args:
            limit: Maximum number to return.

        Returns:
            List of links (sorted by created_at descending).
        """
        links = list(self._links.values())
        links.sort(key=lambda l: l.created_at, reverse=True)
        return links[:limit]

    def count(self) -> int:
        """Get total number of links."""
        return len(self._links)

    def to_graph_edges(self) -> List[Dict[str, Any]]:
        """Convert links to graph edge format.

        Integration with knowledge graph export.

        Returns:
            List of edge dictionaries for graph rendering.
        """
        return [
            {
                "source": link.source_entity,
                "target": link.target_entity,
                "relation": link.relation,
                "confidence": link.confidence,
                "manual": True,
                "notes": link.notes,
            }
            for link in self._links.values()
        ]


def get_link_manager(data_dir: Optional[Path] = None) -> ManualLinkManager:
    """Get link manager instance.

    Args:
        data_dir: Storage path (defaults to .data/).

    Returns:
        ManualLinkManager instance.
    """
    if data_dir is None:
        data_dir = Path.cwd() / ".data"

    return ManualLinkManager(data_dir=data_dir)
