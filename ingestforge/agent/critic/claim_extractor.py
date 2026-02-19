"""Atomic Claim Extractor.

Decomposes long-form text into atomic, verifiable propositions.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class AtomicClaim:
    """A single, verifiable fact extracted from text."""

    text: str
    context_para: int  # Original paragraph index
    citation_id: Optional[str] = None


class ClaimExtractor:
    """Logic for breaking synthesis reports into verifiable atoms."""

    MAX_CLAIMS_PER_DOC = 50
    MIN_CLAIM_LENGTH = 10

    def __init__(self):
        # Using regex for high-performance segmentation without heavy dependencies
        self._sent_regex = re.compile(r"[^.!?]+[.!?]")
        self._cite_regex = re.compile(r"\[([a-f0-9]{16})\]")

    def extract(self, text: str) -> List[AtomicClaim]:
        """Extract atomic claims from document text.

        Rule #1: Linear control flow with early returns.
        Rule #7: Validate input.
        """
        if not text or len(text.strip()) < self.MIN_CLAIM_LENGTH:
            return []

        claims: List[AtomicClaim] = []
        paragraphs = text.split("\n\n")

        for p_idx, para in enumerate(paragraphs):
            if len(claims) >= self.MAX_CLAIMS_PER_DOC:
                break

            # Decompose paragraph into sentences
            sentences = self._sent_regex.findall(para)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < self.MIN_CLAIM_LENGTH:
                    continue

                # Identify if this claim has an attached citation
                cite_match = self._cite_regex.search(sent)
                chunk_id = cite_match.group(1) if cite_match else None

                claims.append(
                    AtomicClaim(text=sent, context_para=p_idx, citation_id=chunk_id)
                )

        return claims
