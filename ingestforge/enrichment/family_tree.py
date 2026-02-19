"""
Family tree enrichment for Genealogy vertical.

Converts GEDCOM extracted entities into semantic relationships in the
Knowledge Graph.
"""

import logging
from typing import List, Dict, Optional

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class FamilyTreeEnricher:
    """
    Enriches chunks with genealogical relationships.
    """

    def enrich(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """Enrich a list of chunks with family tree relationships."""
        # First, build a map of individuals by ID
        person_map = {c.metadata.get("id"): c for c in chunks if c.chunk_type == "INDI"}

        for chunk in chunks:
            if chunk.chunk_type == "FAM":
                self._process_family_chunk(chunk, person_map)

        return chunks

    def _process_family_chunk(
        self, fam_chunk: ChunkRecord, person_map: Dict[str, ChunkRecord]
    ) -> None:
        """Extract relationships from a family chunk."""
        metadata = fam_chunk.metadata or {}

        # GEDCOM FAM record typically has HUSB, WIFE, CHIL
        husb_id = metadata.get("HUSB")
        wife_id = metadata.get("WIFE")
        children_ids = metadata.get("CHIL", [])
        if isinstance(children_ids, str):
            children_ids = [children_ids]

        # Spouse relationship
        if husb_id and wife_id:
            self._add_relationship(
                person_map.get(husb_id), person_map.get(wife_id), "spouse_of"
            )

        # Parent-Child relationships
        parents = [husb_id, wife_id]
        for parent_id in parents:
            if not parent_id:
                continue
            parent_chunk = person_map.get(parent_id)
            for child_id in children_ids:
                child_chunk = person_map.get(child_id)
                self._add_relationship(parent_chunk, child_chunk, "parent_of")

    def _add_relationship(
        self, subj: Optional[ChunkRecord], obj: Optional[ChunkRecord], predicate: str
    ) -> None:
        """Add relationship to subject chunk metadata."""
        if not subj or not obj:
            return

        if "relationships" not in subj.metadata:
            subj.metadata["relationships"] = []

        rel = {
            "subject": subj.section_title or subj.chunk_id,
            "predicate": predicate,
            "object": obj.section_title or obj.chunk_id,
            "object_id": obj.chunk_id,
        }

        if rel not in subj.metadata["relationships"]:
            subj.metadata["relationships"].append(rel)
