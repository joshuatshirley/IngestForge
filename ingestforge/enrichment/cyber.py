"""
Cybersecurity and Threat Intelligence enrichment.

Extracts CVE IDs, CVSS scores, and affected software from security reports.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class CyberMetadataRefiner:
    """
    Enriches chunks with cybersecurity-specific metadata.
    """

    # Cyber specific patterns
    CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
    CVSS_PATTERN = re.compile(r"(?:CVSS|Score)[:\s]*(\d\.\d)\b", re.IGNORECASE)
    AFFECTED_PATTERN = re.compile(
        r"(?:Affected|Vulnerable|Software|Version):\s+([^:\n]+?)(?=\n|\Z)",
        re.IGNORECASE,
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with cyber metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract CVE IDs
        cves = self.CVE_PATTERN.findall(content)
        if cves:
            metadata["cyber_cve_id"] = cves[0].upper()  # Primary CVE
            metadata["cyber_all_cves"] = [c.upper() for c in list(set(cves))]

        # Extract CVSS Score
        cvss_match = self.CVSS_PATTERN.search(content)
        if cvss_match:
            metadata["cyber_cvss_score"] = float(cvss_match.group(1))

        # Extract Affected Software
        affected_match = self.AFFECTED_PATTERN.search(content)
        if affected_match:
            sw_list = [sw.strip() for sw in affected_match.group(1).split(",")]
            metadata["cyber_affected_sw"] = [sw for sw in sw_list if sw]

        chunk.metadata = metadata
        return chunk
