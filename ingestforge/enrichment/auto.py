"""
Automotive and Restoration enrichment.

Extracts VINs, part numbers, and restoration status.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class AutoMetadataRefiner:
    """
    Enriches chunks with automotive-specific metadata.
    """

    # Automotive specific patterns
    # Standard 17-char VIN (skipping I, O, Q)
    VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
    PART_PATTERN = re.compile(
        r"\b(?:Part|P/N|Ref)(?:\s+(?:Ref|ID|Number))?[:\s\-#]+([A-Z0-9]+(?:\-[A-Z0-9]+)*)\b",
        re.IGNORECASE,
    )
    STATUS_KEYWORDS = [
        "Original",
        "Replaced",
        "Restored",
        "Pending",
        "Damaged",
        "Missing",
    ]

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with automotive metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract VIN
        vin_match = self.VIN_PATTERN.search(content)
        if vin_match:
            metadata["auto_vin"] = vin_match.group(1).upper()

        # Extract Part Numbers
        parts = self.PART_PATTERN.findall(content)
        if parts:
            metadata["auto_part_numbers"] = list(
                set([p.strip().upper() for p in parts if p.strip()])
            )

        # Extract Restoration Status (Keyword based)
        for status in self.STATUS_KEYWORDS:
            if re.search(rf"\b{status}\b", content, re.IGNORECASE):
                metadata["auto_restoration_status"] = status
                break

        chunk.metadata = metadata
        return chunk
