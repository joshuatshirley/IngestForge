"""Debate Adapter for Multi-Agent Fact-Checker.

Connects LLMClient implementations to DebateOrchestrator,
enabling LLM-powered adversarial fact verification.
Debate Response Format
----------------------
The adapter expects LLM responses in this format:

    Position: [Your stance on the claim]
    Evidence: [Supporting/refuting evidence]
    Confidence: [0.0-1.0]

Example Proponent:
    Position: The claim is well-supported by available research.
    Evidence: Multiple peer-reviewed studies show consistent results.
    Confidence: 0.85

Example Critic:
    Position: The claim lacks sufficient empirical support.
    Evidence: Key studies have methodological flaws.
    Confidence: 0.70
"""

from __future__ import annotations

import re

from ingestforge.agent.fact_checker import (
    DebateArgument,
    DebateFunction,
    DebateRole,
    Evidence,
)
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import GenerationConfig, LLMClient

logger = get_logger(__name__)
MAX_POSITION_LENGTH = 500
MAX_EVIDENCE_TEXT = 300
MAX_HISTORY_CONTEXT = 2


def create_proponent_function(llm_client: LLMClient) -> DebateFunction:
    """Create proponent debate function from LLM client.

    Args:
        llm_client: LLM client instance

    Returns:
        Debate function that defends claims

    Example:
        proponent_fn = create_proponent_function(llm_client)
        arg = proponent_fn("Defend: claim text", [])
    """
    if llm_client is None:
        raise ValueError("llm_client cannot be None")

    def proponent_debate_fn(
        prompt: str,
        history: list[DebateArgument],
    ) -> DebateArgument:
        """Generate proponent argument.

        Args:
            prompt: Debate prompt
            history: Previous arguments

        Returns:
            Proponent's debate argument
        """
        # Build enhanced prompt
        full_prompt = _build_proponent_prompt(prompt, history)

        # Generate response
        config = GenerationConfig(
            max_tokens=800,
            temperature=0.7,
        )

        try:
            response = llm_client.generate(full_prompt, config)
        except Exception as e:
            logger.error(f"LLM generation failed for proponent: {e}")
            return _make_fallback_argument(
                DebateRole.PROPONENT,
                "Unable to generate argument due to LLM error",
            )

        # Parse response
        return _parse_debate_response(response, DebateRole.PROPONENT)

    return proponent_debate_fn


def create_critic_function(llm_client: LLMClient) -> DebateFunction:
    """Create critic debate function from LLM client.

    Args:
        llm_client: LLM client instance

    Returns:
        Debate function that critiques claims

    Example:
        critic_fn = create_critic_function(llm_client)
        arg = critic_fn("Critique: claim text", [])
    """
    if llm_client is None:
        raise ValueError("llm_client cannot be None")

    def critic_debate_fn(
        prompt: str,
        history: list[DebateArgument],
    ) -> DebateArgument:
        """Generate critic argument.

        Args:
            prompt: Debate prompt
            history: Previous arguments

        Returns:
            Critic's debate argument
        """
        # Build enhanced prompt
        full_prompt = _build_critic_prompt(prompt, history)

        # Generate response
        config = GenerationConfig(
            max_tokens=800,
            temperature=0.7,
        )

        try:
            response = llm_client.generate(full_prompt, config)
        except Exception as e:
            logger.error(f"LLM generation failed for critic: {e}")
            return _make_fallback_argument(
                DebateRole.CRITIC,
                "Unable to generate argument due to LLM error",
            )

        # Parse response
        return _parse_debate_response(response, DebateRole.CRITIC)

    return critic_debate_fn


def _build_proponent_prompt(
    base_prompt: str,
    history: list[DebateArgument],
) -> str:
    """Build enhanced proponent prompt.

    Args:
        base_prompt: Base prompt from orchestrator
        history: Previous arguments

    Returns:
        Enhanced prompt string
    """
    system_instructions = """You are a proponent defending a claim in a debate.

Provide your argument in this exact format:

Position: [Your stance defending the claim]
Evidence: [Supporting evidence and reasoning]
Confidence: [0.0-1.0]

Be concise, evidence-based, and persuasive."""

    history_context = _format_recent_history(history)

    return f"""{system_instructions}

{base_prompt}

{history_context}

Provide your Position, Evidence, and Confidence:"""


def _build_critic_prompt(
    base_prompt: str,
    history: list[DebateArgument],
) -> str:
    """Build enhanced critic prompt.

    Args:
        base_prompt: Base prompt from orchestrator
        history: Previous arguments

    Returns:
        Enhanced prompt string
    """
    system_instructions = """You are a critic challenging a claim in a debate.

Provide your argument in this exact format:

Position: [Your stance challenging the claim]
Evidence: [Counter-evidence and weaknesses]
Confidence: [0.0-1.0]

Be rigorous, skeptical, and evidence-based."""

    history_context = _format_recent_history(history)

    return f"""{system_instructions}

{base_prompt}

{history_context}

Provide your Position, Evidence, and Confidence:"""


def _format_recent_history(history: list[DebateArgument]) -> str:
    """Format recent debate history for context.

    Args:
        history: Previous arguments

    Returns:
        Formatted history string
    """
    if not history:
        return ""
    recent = history[-MAX_HISTORY_CONTEXT:]

    parts = ["Previous arguments:"]
    for arg in recent:
        role_label = "Proponent" if arg.role == DebateRole.PROPONENT else "Critic"
        parts.append(f"{role_label}: {arg.position[:200]}")

    return "\n".join(parts)


def _parse_debate_response(
    response: str,
    role: DebateRole,
) -> DebateArgument:
    """Parse LLM response into DebateArgument.

    Args:
        response: LLM response text
        role: Debate role

    Returns:
        Parsed debate argument
    """
    if not response.strip():
        logger.warning("Empty debate response")
        return _make_fallback_argument(role, "No response provided")

    # Extract components
    position = _extract_position(response)
    evidence_text = _extract_evidence(response)
    confidence = _extract_confidence(response)

    # Create evidence object
    evidence_list = []
    if evidence_text:
        supports_claim = role == DebateRole.PROPONENT
        evidence_list.append(
            Evidence(
                content=evidence_text[:MAX_EVIDENCE_TEXT],
                source="llm_analysis",
                supports_claim=supports_claim,
                confidence=confidence,
            )
        )

    return DebateArgument(
        role=role,
        position=position[:MAX_POSITION_LENGTH],
        evidence=evidence_list,
    )


def _extract_position(text: str) -> str:
    """Extract position from response.

    Args:
        text: Response text

    Returns:
        Position string or default
    """
    match = re.search(
        r"Position:\s*(.+?)(?=\n(?:Evidence|Confidence|$))",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    logger.warning("No position found in response")
    return "Unable to parse position from response"


def _extract_evidence(text: str) -> str:
    """Extract evidence from response.

    Args:
        text: Response text

    Returns:
        Evidence string or empty
    """
    match = re.search(
        r"Evidence:\s*(.+?)(?=\n(?:Confidence|$))",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    logger.debug("No evidence found in response")
    return ""


def _extract_confidence(text: str) -> float:
    """Extract confidence from response.

    Args:
        text: Response text

    Returns:
        Confidence value (0.0-1.0)
    """
    match = re.search(
        r"Confidence:\s*(0?\.\d+|1\.0|[01])",
        text,
        re.IGNORECASE,
    )

    if match:
        try:
            confidence = float(match.group(1))
            # Clamp to valid range
            return max(0.0, min(1.0, confidence))
        except ValueError:
            logger.warning(f"Invalid confidence value: {match.group(1)}")

    # Default moderate confidence
    return 0.6


def _make_fallback_argument(role: DebateRole, message: str) -> DebateArgument:
    """Create fallback argument for error cases.

    Args:
        role: Debate role
        message: Error message

    Returns:
        Fallback debate argument
    """
    return DebateArgument(
        role=role,
        position=message,
        evidence=[
            Evidence(
                content=message,
                source="error_fallback",
                supports_claim=False,
                confidence=0.0,
            )
        ],
    )
