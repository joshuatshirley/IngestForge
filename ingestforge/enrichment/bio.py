"""
Biology and Lab enrichment.

Extracts experiment IDs, chemical formulas, and protocols.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class BioMetadataRefiner:
    """
    Enriches chunks with bio-specific metadata.
    """

    # Bio specific patterns
    EXP_ID_PATTERN = re.compile(
        r"\b(?:EXP|Project|Lab|Experiment)(?:\s+(?:ID|Ref))?[:\s\-#]+([A-Z0-9]{2,15}(?:\-[A-Z0-9]+)*)\b",
        re.IGNORECASE,
    )
    # Simple chemical formula: Capitals followed by optional numbers (H2O, NaCl, C6H12O6)
    FORMULA_PATTERN = re.compile(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b")
    PROTOCOL_PATTERN = re.compile(
        r"\b(?:Protocol|Method|SOP)[:\s]+([\w\s]{3,40})(?=\n|\.|\Z)", re.IGNORECASE
    )

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with bio metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Experiment ID
        exp_match = self.EXP_ID_PATTERN.search(content)
        if exp_match:
            metadata["bio_exp_id"] = exp_match.group(1).strip().upper()

        # Extract Formulas (find all)
        formulas = self.FORMULA_PATTERN.findall(content)
        if formulas:
            # Filter formulas to ensure they look scientific (at least one letter and one number or multiple letters)
            valid_formulas = [
                f
                for f in formulas
                if any(c.isdigit() for f in formulas for c in f) or len(f) > 1
            ]
            if valid_formulas:
                metadata["bio_formulas"] = list(set(valid_formulas))

        # Extract Protocol
        prot_match = self.PROTOCOL_PATTERN.search(content)
        if prot_match:
            metadata["bio_protocol"] = prot_match.group(1).strip()

        chunk.metadata = metadata
        return chunk
