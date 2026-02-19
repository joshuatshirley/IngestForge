"""Fact-Check Engine.

Orchestrates the adversarial verification of claims against source text.
Follows NASA JPL Rule #4 (Modular) and Rule #5 (Fail Fast).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from ingestforge.agent.critic.claim_extractor import AtomicClaim
from ingestforge.agent.personas import get_persona, PersonaType
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient

logger = get_logger(__name__)


class VerificationStatus(Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """The outcome of an adversarial audit on a single claim."""

    claim: AtomicClaim
    status: VerificationStatus
    reasoning: str
    confidence: float


class FactChecker:
    """Adversarial critic engine."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._persona = get_persona(PersonaType.CRITIC)

    def verify_claim(self, claim: AtomicClaim, source_text: str) -> VerificationResult:
        """Audit a single claim against its source context.

        Rule #1: Flat control flow.
        Rule #5: Assert source text availability.
        """
        if not claim.citation_id or not source_text:
            return VerificationResult(
                claim, VerificationStatus.SKIPPED, "No source context", 0.0
            )

        # Construct the Skeptical Audit Prompt
        prompt = self._build_audit_prompt(claim.text, source_text)

        try:
            # Generate critique using the skeptical persona
            response = self._llm.generate(
                prompt,
                config=None,  # Uses default config from persona handled by adapter if needed, or simple gen here
            )
            return self._parse_critique(response, claim)
        except Exception as e:
            logger.error(f"Verification failed for claim '{claim.text[:30]}...': {e}")
            return VerificationResult(claim, VerificationStatus.SKIPPED, str(e), 0.0)

    def _build_audit_prompt(self, claim_text: str, context: str) -> str:
        """Construct the prompt for the LLM."""
        return (
            f"SYSTEM: {self._persona.system_prompt}\n\n"
            f"SOURCE TEXT:\n{context}\n\n"
            f"CLAIM TO AUDIT: {claim_text}\n\n"
            "TASK: Verify if the source text explicitly supports the claim.\n"
            "Respond in this format:\n"
            "STATUS: [VERIFIED | CONTRADICTED | UNSUPPORTED]\n"
            "REASONING: <Critical analysis of why>"
        )

    def _parse_critique(self, response: str, claim: AtomicClaim) -> VerificationResult:
        """Parse LLM output into structured result."""
        # Simple parsing logic (Rule #1)
        lines = response.strip().split("\n")
        status = VerificationStatus.UNSUPPORTED
        reasoning = "Could not parse critic response"

        for line in lines:
            if line.startswith("STATUS:"):
                val = line.replace("STATUS:", "").strip().upper()
                if "VERIFIED" in val:
                    status = VerificationStatus.VERIFIED
                elif "CONTRADICTED" in val:
                    status = VerificationStatus.CONTRADICTED
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return VerificationResult(claim, status, reasoning, 1.0)
