"""
Museum and Gallery enrichment.

Extracts accession numbers, artist names, and historical eras.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class MuseumMetadataRefiner:
    """
    Enriches chunks with museum-specific metadata.
    """

    # Museum specific patterns
    ACCESSION_PATTERN = re.compile(
        r"\b(?:Accession|ID|Ref)(?:\s+Number)?[:\s\-#]*(\d{4}\.\d{1,5}\.\d{1,5})\b",
        re.IGNORECASE,
    )
    ARTIST_PATTERN = re.compile(
        r"\b(?:Artist|Creator|Maker|Contributor)[:\s]+([A-Z][a-z]+(?:\s+[a-z]+)?(?:\s+[A-Z][a-z]+)*?)(?=[.,\n]|\Z)"
    )
    ERA_PATTERN = re.compile(
        r"\b(?:Era|Period|Date)[:\s]+([\w\s\-]{3,20})(?=\n|\.|\Z)", re.IGNORECASE
    )
    MEDIUM_PATTERN = re.compile(
        r"\b(?:Medium|Material)[:\s]+([\w\s,]{3,30})(?=\n|\.|\Z)", re.IGNORECASE
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with museum metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Accession ID
        acc_match = self.ACCESSION_PATTERN.search(content)
        if acc_match:
            metadata["museum_accession_id"] = acc_match.group(1).strip()

        # Extract Artist
        artist_match = self.ARTIST_PATTERN.search(content)
        if artist_match:
            metadata["museum_artist"] = artist_match.group(1).strip()

        # Extract Era
        era_match = self.ERA_PATTERN.search(content)
        if era_match:
            metadata["museum_era"] = era_match.group(1).strip().capitalize()

        # Extract Medium
        medium_match = self.MEDIUM_PATTERN.search(content)
        if medium_match:
            metadata["museum_medium"] = medium_match.group(1).strip()

        chunk.metadata = metadata
        return chunk
