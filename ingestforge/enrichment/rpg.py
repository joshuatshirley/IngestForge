"""
RPG campaign intelligence enrichment for Tabletop RPGs.

Extracts NPCs, locations, items, and stat blocks from campaign notes
and rulebooks.
"""

import re
import logging
from typing import Dict, Any, Optional

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class RPGMetadataRefiner:
    """
    Enriches chunks with RPG-specific metadata.
    """

    # Patterns for RPG entities
    NPC_PATTERNS = [
        re.compile(r"\b(?:NPC|character|person)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:he|she|they)\s+(?:is|was)\s+a\s+(?:level|lvl)\s+\d+", re.IGNORECASE
        ),
    ]

    LOCATION_PATTERNS = [
        re.compile(
            r"\b(?:Location|Place|City|Town|Village|Dungeon|Room|Area)\b", re.IGNORECASE
        ),
    ]

    # Stat block patterns (D&D 5e style)
    HP_PATTERN = re.compile(r"\b(?:Hit\s+Points|HP)[:\s]+(\d+)\b", re.IGNORECASE)
    AC_PATTERN = re.compile(r"\b(?:Armor\s+Class|AC)[:\s]+(\d+)\b", re.IGNORECASE)
    CR_PATTERN = re.compile(
        r"\b(?:Challenge\s+Rating|CR)[:\s]+(\d+(?:/\d+)?)\b", re.IGNORECASE
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with RPG metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Detect Entity Type
        rpg_type = self._detect_rpg_type(content)
        if rpg_type:
            metadata["rpg_type"] = rpg_type

        # Extract Stats
        stats = self._extract_stats(content)
        if stats:
            metadata["rpg_stats"] = stats

        chunk.metadata = metadata
        return chunk

    def _detect_rpg_type(self, text: str) -> Optional[str]:
        """Detect if the text describes an NPC, Location, or Item."""
        for pattern in self.NPC_PATTERNS:
            if pattern.search(text):
                return "NPC"
        for pattern in self.LOCATION_PATTERNS:
            if pattern.search(text):
                return "Location"

        # Look for magic items
        if "Magic Item" in text or "Rarity:" in text:
            return "Item"

        return None

    def _extract_stats(self, text: str) -> Dict[str, Any]:
        """Extract mechanical stats from the text."""
        stats = {}

        hp_match = self.HP_PATTERN.search(text)
        if hp_match:
            stats["hp"] = int(hp_match.group(1))

        ac_match = self.AC_PATTERN.search(text)
        if ac_match:
            stats["ac"] = int(ac_match.group(1))

        cr_match = self.CR_PATTERN.search(text)
        if cr_match:
            stats["cr"] = cr_match.group(1)

        return stats
