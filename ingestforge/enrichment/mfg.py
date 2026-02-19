"""
Manufacturing and Industrial enrichment.

Extracts part numbers, maintenance cycles, and error codes from manuals and logs.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class MfgMetadataRefiner:
    """
    Enriches chunks with manufacturing-specific metadata.
    """

    # Mfg specific patterns
    PART_PATTERN = re.compile(
        r"(?:Part|P/N|Ref|Number)[:\s\-#]+([A-Z0-9\-]{7,25})\b", re.IGNORECASE
    )
    MAINTENANCE_PATTERN = re.compile(
        r"\b(?:Maintenance|Service|Cycle)[:\s]+(every\s+\d+\s+(?:hours|days|months|cycles))\b",
        re.IGNORECASE,
    )
    ERROR_PATTERN = re.compile(
        r"\b(?:Error|Fault|Code)[:\s\-#]*([A-Z0-9]{2,5}-\d{2,5})\b", re.IGNORECASE
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with manufacturing metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Part Number
        part_match = self.PART_PATTERN.search(content)
        if part_match:
            metadata["mfg_part_number"] = part_match.group(1).strip().upper()

        # Extract Maintenance Cycle
        maint_match = self.MAINTENANCE_PATTERN.search(content)
        if maint_match:
            metadata["mfg_maintenance_cycle"] = maint_match.group(1).strip()

        # Extract Error Codes
        errors = self.ERROR_PATTERN.findall(content)
        if errors:
            metadata["mfg_error_codes"] = list(
                set(
                    [
                        e.upper()
                        for s in errors
                        for e in (s if isinstance(s, tuple) else [s])
                    ]
                )
            )

        chunk.metadata = metadata
        return chunk
