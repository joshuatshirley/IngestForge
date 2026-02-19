"""
Grant and Proposal enrichment.

Extracts Funding Opportunity Numbers, award amounts, and deadlines.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class GrantMetadataRefiner:
    """
    Enriches chunks with grant-solicitation metadata.
    """

    # Grant specific patterns
    FON_PATTERN = re.compile(
        r"(?:FON|Opportunity|Solicitation|ID|Number)[:\s\-#]+([A-Z0-9\-]{7,25})\b",
        re.IGNORECASE,
    )
    AMOUNT_PATTERN = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
    DEADLINE_PATTERN = re.compile(
        r"\b(?:Deadline|Due|Submission)[:\s]+([\w\s\d,]{5,30})(?=\n|\Z)", re.IGNORECASE
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with grant metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Solicitation ID (FON)
        fon_match = self.FON_PATTERN.search(content)
        if fon_match:
            metadata["grant_id"] = fon_match.group(1).strip()

        # Extract Award Amount
        amount_match = self.AMOUNT_PATTERN.search(content)
        if amount_match:
            metadata["grant_amount"] = float(amount_match.group(1).replace(",", ""))

        # Extract Deadline
        deadline_match = self.DEADLINE_PATTERN.search(content)
        if deadline_match:
            metadata["grant_deadline"] = deadline_match.group(1).strip()

        chunk.metadata = metadata
        return chunk
