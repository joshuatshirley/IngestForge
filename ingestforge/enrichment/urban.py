"""
Urban Planning and Zoning enrichment.

Extracts zoning codes, FAR metrics, and density targets.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class UrbanMetadataRefiner:
    """
    Enriches chunks with urban-planning-specific metadata.
    """

    # Urban specific patterns
    # Zoning: R-1, C-2, M-1, etc.
    ZONING_PATTERN = re.compile(r"\b([RCM][\-\s]?\d[A-Z]?)\b")
    # FAR: "FAR of 2.5" or "Floor Area Ratio: 3.0"
    FAR_PATTERN = re.compile(
        r"\b(?:FAR|Floor Area Ratio)(?:\s+of)?[:\s]*(\d+(?:\.\d+)?)\b", re.IGNORECASE
    )
    # Density: "50 units/acre" or "High Density"
    DENSITY_PATTERN = re.compile(
        r"\b(?:Density|Target)(?:\s+is)?[:\s]+([\w\s\/]{3,20})(?=[.,\n]|\Z)",
        re.IGNORECASE,
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with urban metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Zoning Code
        zoning_match = self.ZONING_PATTERN.search(content)
        if zoning_match:
            metadata["urban_zoning_code"] = zoning_match.group(1).strip().upper()

        # Extract FAR
        far_match = self.FAR_PATTERN.search(content)
        if far_match:
            metadata["urban_far_ratio"] = float(far_match.group(1))

        # Extract Density Target
        dens_match = self.DENSITY_PATTERN.search(content)
        if dens_match:
            metadata["urban_density_target"] = dens_match.group(1).strip()

        chunk.metadata = metadata
        return chunk
