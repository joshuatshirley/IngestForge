"""
Political and Campaign enrichment.

Extracts candidate names, vote results, and donor entities.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class PoliticalMetadataRefiner:
    """
    Enriches chunks with political-specific metadata.
    """

    # Political specific patterns
    CANDIDATE_PATTERN = re.compile(
        r"\b(?:Candidate|Senator|Rep|Representative|Mayor)[:\s]+([\w\s\.]{3,25}?)(?=[.,\n]|\Z)",
        re.IGNORECASE,
    )
    VOTE_PATTERN = re.compile(
        r"\b(?:Vote|Result|Action)[:\s]*(Yea|Nay|Aye|No|Present|Abstain)\b",
        re.IGNORECASE,
    )
    DONOR_PATTERN = re.compile(
        r"\b(?:Donor|Contributor|PAC|Entity)[:\s]+([\w\s\&]{3,40})(?=\n|\.|\,)",
        re.IGNORECASE,
    )
    AMOUNT_PATTERN = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with political metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Candidate
        candidate_match = self.CANDIDATE_PATTERN.search(content)
        if candidate_match:
            metadata["political_candidate"] = candidate_match.group(1).strip()

        # Extract Vote
        vote_match = self.VOTE_PATTERN.search(content)
        if vote_match:
            metadata["political_vote"] = vote_match.group(1).strip().capitalize()

        # Extract Donors (can be multiple)
        donors = self.DONOR_PATTERN.findall(content)
        if donors:
            metadata["political_donors"] = list(
                set([d.strip() for d in donors if d.strip()])
            )

        # Extract Amount (Total in chunk context)
        amounts = self.AMOUNT_PATTERN.findall(content)
        if amounts:
            # We take the largest amount found as the likely contribution total
            vals = [float(a.replace(",", "")) for a in amounts]
            metadata["political_contribution"] = max(vals)

        chunk.metadata = metadata
        return chunk
