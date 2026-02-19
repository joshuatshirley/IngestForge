"""
Gaming and Esports enrichment.

Extracts patch versions, character names, and stat changes.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class GamingMetadataRefiner:
    """
    Enriches chunks with gaming-specific metadata.
    """

    # Gaming specific patterns
    VERSION_PATTERN = re.compile(
        r"\b(?:Patch|v|Version)[:\s]*(\d+\.\d+(?:\.\d+)*[a-z]?)\b", re.IGNORECASE
    )
    # Character name: usually capitalized, sometimes multi-word
    CHARACTER_PATTERN = re.compile(
        r"\b(?:Hero|Champion|Character|Agent)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?=[.,\n]|\Z)"
    )
    # Stat change: "Damage: 50 -> 55" or "+10 Armor"
    STAT_PATTERN = re.compile(
        r"\b([\w\s]{3,20}?)(?:[:\s]*)([\+\-]\d+%?|(?:\d+%?\s*->\s*\d+%?))(?=[.,\s]|$)",
        re.IGNORECASE,
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with gaming metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Version
        ver_match = self.VERSION_PATTERN.search(content)
        if ver_match:
            metadata["gaming_patch_version"] = ver_match.group(1).strip()

        # Extract Characters
        chars = self.CHARACTER_PATTERN.findall(content)
        if chars:
            metadata["gaming_characters"] = list(
                set([c.strip() for c in chars if c.strip()])
            )

        # Extract Stat Changes
        stats = []
        for stat_match in self.STAT_PATTERN.finditer(content):
            name = stat_match.group(1).strip()
            # Filter name noise (should be short and like a stat)
            if 2 < len(name) < 25 and any(
                kw in name.lower()
                for kw in [
                    "damage",
                    "armor",
                    "health",
                    "speed",
                    "haste",
                    "mana",
                    "power",
                    "strength",
                    "intelligence",
                    "agility",
                ]
            ):
                stats.append(f"{name}: {stat_match.group(2)}")
        if stats:
            metadata["gaming_stat_changes"] = list(set(stats))

        chunk.metadata = metadata
        return chunk
