"""Multi-Agent Adversarial Fact-Checker.

Implements proponent/critic debate strategy for
fact verification through adversarial dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_DEBATE_ROUNDS = 10
MAX_CLAIMS = 20
MAX_EVIDENCE_ITEMS = 50
MAX_CLAIM_LENGTH = 1000


class VerificationStatus(Enum):
    """Status of claim verification."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    CONTESTED = "contested"


class DebateRole(Enum):
    """Role in the debate."""

    PROPONENT = "proponent"
    CRITIC = "critic"
    JUDGE = "judge"


@dataclass
class Evidence:
    """Evidence item supporting or refuting a claim."""

    content: str
    source: str = ""
    supports_claim: bool = True
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content[:MAX_CLAIM_LENGTH],
            "source": self.source,
            "supports": self.supports_claim,
            "confidence": self.confidence,
        }


@dataclass
class DebateArgument:
    """Single argument in a debate round."""

    role: DebateRole
    position: str
    evidence: list[Evidence] = field(default_factory=list)
    round_number: int = 0

    def __post_init__(self) -> None:
        """Validate argument on creation."""
        self.position = self.position[:MAX_CLAIM_LENGTH]
        self.evidence = self.evidence[:MAX_EVIDENCE_ITEMS]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "position": self.position,
            "evidence": [e.to_dict() for e in self.evidence],
            "round": self.round_number,
        }


@dataclass
class Claim:
    """A claim to be verified."""

    content: str
    source: str = ""
    context: str = ""

    def __post_init__(self) -> None:
        """Validate claim on creation."""
        self.content = self.content[:MAX_CLAIM_LENGTH]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "context": self.context[:500],
        }


@dataclass
class VerificationResult:
    """Result of claim verification."""

    claim: Claim
    status: VerificationStatus
    confidence: float
    proponent_score: float
    critic_score: float
    arguments: list[DebateArgument] = field(default_factory=list)
    summary: str = ""

    @property
    def is_verified(self) -> bool:
        """Check if claim is verified."""
        return self.status == VerificationStatus.VERIFIED

    @property
    def rounds_count(self) -> int:
        """Number of debate rounds."""
        return len([a for a in self.arguments if a.role == DebateRole.PROPONENT])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "proponent_score": self.proponent_score,
            "critic_score": self.critic_score,
            "arguments": [a.to_dict() for a in self.arguments],
            "summary": self.summary,
        }


class DebateStrategy:
    """Prompt strategy for debate participants."""

    def proponent_prompt(
        self,
        claim: Claim,
        history: list[DebateArgument],
    ) -> str:
        """Generate proponent prompt.

        Args:
            claim: Claim being defended
            history: Previous arguments

        Returns:
            Prompt string
        """
        base = f"Defend this claim: {claim.content}\n"

        if history:
            last_critic = [a for a in history if a.role == DebateRole.CRITIC]
            if last_critic:
                base += f"\nAddress this critique: {last_critic[-1].position}"

        return base + "\nProvide supporting evidence."

    def critic_prompt(
        self,
        claim: Claim,
        history: list[DebateArgument],
    ) -> str:
        """Generate critic prompt.

        Args:
            claim: Claim being critiqued
            history: Previous arguments

        Returns:
            Prompt string
        """
        base = f"Critique this claim: {claim.content}\n"

        if history:
            last_prop = [a for a in history if a.role == DebateRole.PROPONENT]
            if last_prop:
                base += f"\nChallenge this argument: {last_prop[-1].position}"

        return base + "\nIdentify weaknesses or counter-evidence."

    def judge_prompt(
        self,
        claim: Claim,
        arguments: list[DebateArgument],
    ) -> str:
        """Generate judge prompt.

        Args:
            claim: Claim being judged
            arguments: All debate arguments

        Returns:
            Prompt string
        """
        return f"""Judge this claim: {claim.content}

Review the debate arguments and determine:
1. Is the claim verified, refuted, or uncertain?
2. Which side presented stronger evidence?
3. What is your confidence level (0-1)?
"""


# Type for debate participant function
DebateFunction = Callable[[str, list[DebateArgument]], DebateArgument]


class DebateOrchestrator:
    """Orchestrates adversarial fact-checking debate.

    Manages proponent/critic exchanges to verify claims
    through structured argumentation.
    """

    def __init__(
        self,
        proponent_fn: DebateFunction,
        critic_fn: DebateFunction,
        max_rounds: int = MAX_DEBATE_ROUNDS,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            proponent_fn: Function for proponent role
            critic_fn: Function for critic role
            max_rounds: Maximum debate rounds
        """
        if max_rounds < 1:
            max_rounds = 1
        if max_rounds > MAX_DEBATE_ROUNDS:
            max_rounds = MAX_DEBATE_ROUNDS

        self._proponent_fn = proponent_fn
        self._critic_fn = critic_fn
        self._max_rounds = max_rounds
        self._strategy = DebateStrategy()

    def verify(self, claim: Claim) -> VerificationResult:
        """Verify a claim through debate.

        Args:
            claim: Claim to verify

        Returns:
            Verification result
        """
        if not claim.content.strip():
            return self._make_uncertain(claim, "Empty claim")

        arguments: list[DebateArgument] = []

        # Run debate rounds
        for round_num in range(self._max_rounds):
            # Proponent argues
            prop_prompt = self._strategy.proponent_prompt(claim, arguments)
            prop_arg = self._proponent_fn(prop_prompt, arguments)
            prop_arg.round_number = round_num
            arguments.append(prop_arg)

            # Critic responds
            critic_prompt = self._strategy.critic_prompt(claim, arguments)
            critic_arg = self._critic_fn(critic_prompt, arguments)
            critic_arg.round_number = round_num
            arguments.append(critic_arg)

            # Check for early termination
            if self._should_terminate(arguments):
                break

        # Judge the debate
        return self._judge_debate(claim, arguments)

    def _should_terminate(self, arguments: list[DebateArgument]) -> bool:
        """Check if debate should terminate early.

        Args:
            arguments: Current arguments

        Returns:
            True if should terminate
        """
        if len(arguments) < 4:
            return False

        # Check for strong consensus
        recent = arguments[-4:]
        prop_evidence = sum(
            len(a.evidence) for a in recent if a.role == DebateRole.PROPONENT
        )
        critic_evidence = sum(
            len(a.evidence) for a in recent if a.role == DebateRole.CRITIC
        )

        # Terminate if one side dominates
        return abs(prop_evidence - critic_evidence) > 3

    def _judge_debate(
        self,
        claim: Claim,
        arguments: list[DebateArgument],
    ) -> VerificationResult:
        """Judge the debate outcome.

        Args:
            claim: Original claim
            arguments: All arguments

        Returns:
            Verification result
        """
        # Calculate scores
        prop_score = self._calculate_score(arguments, DebateRole.PROPONENT)
        critic_score = self._calculate_score(arguments, DebateRole.CRITIC)

        # Determine status
        status, confidence = self._determine_status(prop_score, critic_score)

        # Generate summary
        summary = self._generate_summary(claim, status, prop_score, critic_score)

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            proponent_score=prop_score,
            critic_score=critic_score,
            arguments=arguments,
            summary=summary,
        )

    def _calculate_score(
        self,
        arguments: list[DebateArgument],
        role: DebateRole,
    ) -> float:
        """Calculate score for a debate role.

        Args:
            arguments: All arguments
            role: Role to score

        Returns:
            Score (0-1)
        """
        role_args = [a for a in arguments if a.role == role]

        if not role_args:
            return 0.0

        # Score based on evidence quantity and confidence
        total_evidence = sum(len(a.evidence) for a in role_args)
        avg_confidence = sum(e.confidence for a in role_args for e in a.evidence) / max(
            total_evidence, 1
        )

        # Normalize to 0-1
        evidence_score = min(total_evidence / 10, 1.0)
        return (evidence_score + avg_confidence) / 2

    def _determine_status(
        self,
        prop_score: float,
        critic_score: float,
    ) -> tuple[VerificationStatus, float]:
        """Determine verification status.

        Args:
            prop_score: Proponent score
            critic_score: Critic score

        Returns:
            Tuple of (status, confidence)
        """
        diff = prop_score - critic_score

        if diff > 0.3:
            return VerificationStatus.VERIFIED, min(diff + 0.5, 1.0)

        if diff < -0.3:
            return VerificationStatus.REFUTED, min(abs(diff) + 0.5, 1.0)

        if abs(diff) < 0.1:
            return VerificationStatus.CONTESTED, 0.5

        return VerificationStatus.UNCERTAIN, 0.6

    def _generate_summary(
        self,
        claim: Claim,
        status: VerificationStatus,
        prop_score: float,
        critic_score: float,
    ) -> str:
        """Generate result summary.

        Args:
            claim: Original claim
            status: Verification status
            prop_score: Proponent score
            critic_score: Critic score

        Returns:
            Summary text
        """
        status_text = {
            VerificationStatus.VERIFIED: "supported by evidence",
            VerificationStatus.REFUTED: "contradicted by evidence",
            VerificationStatus.UNCERTAIN: "cannot be definitively verified",
            VerificationStatus.CONTESTED: "remains contested",
        }

        return f"""The claim "{claim.content[:100]}..." is {status_text[status]}.

Proponent score: {prop_score:.2f}
Critic score: {critic_score:.2f}
"""

    def _make_uncertain(
        self,
        claim: Claim,
        reason: str,
    ) -> VerificationResult:
        """Create uncertain result.

        Args:
            claim: Original claim
            reason: Reason for uncertainty

        Returns:
            Uncertain result
        """
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.0,
            proponent_score=0.0,
            critic_score=0.0,
            summary=reason,
        )


def create_orchestrator(
    proponent_fn: DebateFunction,
    critic_fn: DebateFunction,
    max_rounds: int = MAX_DEBATE_ROUNDS,
) -> DebateOrchestrator:
    """Factory function to create orchestrator.

    Args:
        proponent_fn: Proponent function
        critic_fn: Critic function
        max_rounds: Maximum rounds

    Returns:
        Configured orchestrator
    """
    return DebateOrchestrator(
        proponent_fn=proponent_fn,
        critic_fn=critic_fn,
        max_rounds=max_rounds,
    )


def simple_proponent(
    prompt: str,
    history: list[DebateArgument],
) -> DebateArgument:
    """Simple proponent implementation for testing.

    Args:
        prompt: Debate prompt
        history: Previous arguments

    Returns:
        Proponent argument
    """
    evidence = [
        Evidence(
            content="Supporting evidence",
            source="test",
            supports_claim=True,
            confidence=0.8,
        )
    ]

    return DebateArgument(
        role=DebateRole.PROPONENT,
        position="The claim is well-supported by available evidence.",
        evidence=evidence,
    )


def simple_critic(
    prompt: str,
    history: list[DebateArgument],
) -> DebateArgument:
    """Simple critic implementation for testing.

    Args:
        prompt: Debate prompt
        history: Previous arguments

    Returns:
        Critic argument
    """
    evidence = [
        Evidence(
            content="Counter-evidence",
            source="test",
            supports_claim=False,
            confidence=0.6,
        )
    ]

    return DebateArgument(
        role=DebateRole.CRITIC,
        position="The claim lacks sufficient evidence.",
        evidence=evidence,
    )
