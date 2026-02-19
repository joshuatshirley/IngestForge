"""
Disaster Response and Emergency enrichment.

Extracts incident types, GPS coordinates, and urgency levels.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class DisasterMetadataRefiner:
    """
    Enriches chunks with disaster-response metadata.
    """

    # Disaster specific patterns
    INCIDENT_PATTERN = re.compile(
        r"\b(?:Incident|Type|Nature)[:\s]+([\w\s]{3,25})(?=\n|\.)", re.IGNORECASE
    )
    GPS_PATTERN = re.compile(
        r"(\-?\d{1,3}\.\d{4,8}),\s*(\-?\d{1,3}\.\d{4,8})"
    )  # Simple decimal GPS
    URGENCY_PATTERN = re.compile(
        r"\b(?:Urgency|Priority|Severity)[:\s]*(Immediate|High|Medium|Low|Critical)\b",
        re.IGNORECASE,
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with disaster metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Incident Type
        incident_match = self.INCIDENT_PATTERN.search(content)
        if incident_match:
            metadata["disaster_incident_type"] = (
                incident_match.group(1).strip().capitalize()
            )

        # Extract GPS Coordinates
        gps_match = self.GPS_PATTERN.search(content)
        if gps_match:
            metadata[
                "disaster_coordinates"
            ] = f"{gps_match.group(1)}, {gps_match.group(2)}"

        # Extract Urgency
        urgency_match = self.URGENCY_PATTERN.search(content)
        if urgency_match:
            metadata["disaster_urgency"] = urgency_match.group(1).strip().capitalize()

        chunk.metadata = metadata
        return chunk
