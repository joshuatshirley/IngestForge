"""
Parent-child chunk mapping storage.

Stores mappings between child chunks (small, indexed for precision) and
parent chunks (larger, returned for context). This enables the parent
document retriever pattern.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.storage.base import ParentChunkMapping


@dataclass
class ParentMappingStore:
    """
    Stores parent-child chunk mappings in a JSONL file.

    Enables efficient lookup of parent chunks from child chunk IDs.
    """

    storage_path: Path
    _child_to_parent: Dict[str, ParentChunkMapping] = field(default_factory=dict)
    _parent_to_children: Dict[str, List[str]] = field(default_factory=dict)
    _loaded: bool = False

    def __post_init__(self) -> None:
        """Ensure path is Path object."""
        self.storage_path = Path(self.storage_path)

    def _store_mapping(self, mapping: ParentChunkMapping) -> None:
        """
        Store mapping in internal dictionaries.

        Rule #1: Simple dictionary operations
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            mapping: Mapping to store
        """
        self._child_to_parent[mapping.child_chunk_id] = mapping

        # Add to parent-children map
        if mapping.parent_chunk_id not in self._parent_to_children:
            self._parent_to_children[mapping.parent_chunk_id] = []
        self._parent_to_children[mapping.parent_chunk_id].append(mapping.child_chunk_id)

    def _process_mapping_line(self, line: str) -> None:
        """
        Process single line from mapping file.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            line: JSON line to process
        """
        if not line.strip():
            return

        # Parse and store mapping
        data = json.loads(line)
        mapping = ParentChunkMapping.from_dict(data)
        self._store_mapping(mapping)

    def _load_from_file(self) -> None:
        """
        Load mappings from file.

        Rule #1: Simple loop with helper
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                self._process_mapping_line(line)

    def _ensure_loaded(self) -> None:
        """
        Load mappings from disk if not already loaded.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints
        """
        if self._loaded:
            return

        # Initialize dictionaries
        self._child_to_parent = {}
        self._parent_to_children = {}

        # Load from file if exists
        if self.storage_path.exists():
            self._load_from_file()

        self._loaded = True

    def _save(self) -> None:
        """Persist mappings to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, "w", encoding="utf-8") as f:
            for mapping in self._child_to_parent.values():
                f.write(json.dumps(mapping.to_dict()) + "\n")

    def add_mapping(
        self,
        child_chunk_id: str,
        parent_chunk_id: str,
        document_id: str,
        child_position: int = 0,
        total_children: int = 1,
    ) -> bool:
        """
        Add a parent-child mapping.

        Args:
            child_chunk_id: ID of the child chunk
            parent_chunk_id: ID of the parent chunk
            document_id: Document both chunks belong to
            child_position: Position of child within parent
            total_children: Total number of children for this parent

        Returns:
            True if mapping was added
        """
        self._ensure_loaded()

        mapping = ParentChunkMapping(
            child_chunk_id=child_chunk_id,
            parent_chunk_id=parent_chunk_id,
            document_id=document_id,
            child_position=child_position,
            total_children=total_children,
        )

        self._child_to_parent[child_chunk_id] = mapping

        if parent_chunk_id not in self._parent_to_children:
            self._parent_to_children[parent_chunk_id] = []
        if child_chunk_id not in self._parent_to_children[parent_chunk_id]:
            self._parent_to_children[parent_chunk_id].append(child_chunk_id)

        self._save()
        return True

    def add_mappings_batch(
        self,
        mappings: List[ParentChunkMapping],
    ) -> int:
        """
        Add multiple mappings at once.

        Args:
            mappings: List of ParentChunkMapping objects

        Returns:
            Number of mappings added
        """
        self._ensure_loaded()

        for mapping in mappings:
            self._child_to_parent[mapping.child_chunk_id] = mapping

            if mapping.parent_chunk_id not in self._parent_to_children:
                self._parent_to_children[mapping.parent_chunk_id] = []
            if (
                mapping.child_chunk_id
                not in self._parent_to_children[mapping.parent_chunk_id]
            ):
                self._parent_to_children[mapping.parent_chunk_id].append(
                    mapping.child_chunk_id
                )

        self._save()
        return len(mappings)

    def get_mapping(self, child_chunk_id: str) -> Optional[ParentChunkMapping]:
        """
        Get the full mapping for a child chunk.

        Args:
            child_chunk_id: ID of the child chunk

        Returns:
            ParentChunkMapping or None
        """
        self._ensure_loaded()
        return self._child_to_parent.get(child_chunk_id)

    def delete_mapping(self, child_chunk_id: str) -> bool:
        """
        Delete a mapping by child chunk ID.

        Args:
            child_chunk_id: ID of the child chunk

        Returns:
            True if deleted
        """
        self._ensure_loaded()

        mapping = self._child_to_parent.pop(child_chunk_id, None)
        if mapping:
            children = self._parent_to_children.get(mapping.parent_chunk_id, [])
            if child_chunk_id in children:
                children.remove(child_chunk_id)
            self._save()
            return True
        return False

    def delete_document_mappings(self, document_id: str) -> int:
        """
        Delete all mappings for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of mappings deleted
        """
        self._ensure_loaded()

        to_delete = [
            cid
            for cid, mapping in self._child_to_parent.items()
            if mapping.document_id == document_id
        ]

        for child_id in to_delete:
            mapping = self._child_to_parent.pop(child_id)
            children = self._parent_to_children.get(mapping.parent_chunk_id, [])
            if child_id in children:
                children.remove(child_id)

        if to_delete:
            self._save()

        return len(to_delete)

    def count(self) -> int:
        """Get total number of mappings."""
        self._ensure_loaded()
        return len(self._child_to_parent)

    def clear(self) -> None:
        """Clear all mappings."""
        self._child_to_parent = {}
        self._parent_to_children = {}
        self._loaded = True

        if self.storage_path.exists():
            self.storage_path.unlink()

    def get_statistics(self) -> Dict[str, Any]:
        """Get mapping statistics."""
        self._ensure_loaded()

        avg_children = 0
        if self._parent_to_children:
            avg_children = len(self._child_to_parent) / len(self._parent_to_children)

        return {
            "total_mappings": len(self._child_to_parent),
            "unique_parents": len(self._parent_to_children),
            "avg_children_per_parent": round(avg_children, 2),
        }


def create_parent_mapping_store(data_path: Path) -> ParentMappingStore:
    """
    Create a ParentMappingStore instance.

    Args:
        data_path: Base data directory

    Returns:
        Configured ParentMappingStore
    """
    storage_path = data_path / "parent_mappings.jsonl"
    return ParentMappingStore(storage_path=storage_path)
