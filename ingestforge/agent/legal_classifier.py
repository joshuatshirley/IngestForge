"""Legal Facts vs. Opinions Classifier.

Multi-agent legal text classifier that distinguishes between
statements of fact, holdings, dicta, and other legal roles.
Classification Roles
--------------------
- FACTS: Statement of facts, procedural history
- HOLDING: The court's actual ruling/decision
- DICTA: Non-binding commentary
- DISSENT: Dissenting opinions
- CONCURRENCE: Concurring opinions
- ANALYSIS: Legal analysis/reasoning
- PROCEDURAL: Procedural matters

Usage Example
-------------
    from ingestforge.agent.legal_classifier import LegalClassifier
    from tests.fixtures.agents import MockLLM

    llm = MockLLM()
    classifier = LegalClassifier(llm_client=llm)

    result = classifier.classify("The court held that...")
    print(result.role)  # "HOLDING"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.llm.base import GenerationConfig, LLMClient

logger = get_logger(__name__)
MAX_TEXT_LENGTH = 5000
MAX_CHUNKS_PER_BATCH = 50
MAX_REASONING_LENGTH = 500
MAX_SECONDARY_ROLES = 3


class LegalRole(Enum):
    """Legal document role classifications."""

    FACTS = "FACTS"
    HOLDING = "HOLDING"
    DICTA = "DICTA"
    DISSENT = "DISSENT"
    CONCURRENCE = "CONCURRENCE"
    ANALYSIS = "ANALYSIS"
    PROCEDURAL = "PROCEDURAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class LegalClassification:
    """Result of classifying a legal text passage.

    Attributes:
        text: Original text that was classified
        role: Primary legal role (e.g., FACTS, HOLDING)
        confidence: Classification confidence (0.0-1.0)
        secondary_roles: Alternative role classifications
        reasoning: Explanation for the classification
    """

    text: str
    role: str
    confidence: float = 0.0
    secondary_roles: List[str] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self) -> None:
        """Validate and truncate fields."""
        self.text = self.text[:MAX_TEXT_LENGTH]
        self.reasoning = self.reasoning[:MAX_REASONING_LENGTH]
        self.secondary_roles = self.secondary_roles[:MAX_SECONDARY_ROLES]
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "role": self.role,
            "confidence": self.confidence,
            "secondary_roles": self.secondary_roles,
            "reasoning": self.reasoning,
        }

    @property
    def is_binding(self) -> bool:
        """Check if classification represents binding legal content."""
        return self.role in ("HOLDING", "FACTS")

    @property
    def is_opinion(self) -> bool:
        """Check if classification represents an opinion."""
        return self.role in ("DICTA", "DISSENT", "CONCURRENCE", "ANALYSIS")


class ClassificationPrompts:
    """Prompt templates for legal classification."""

    @staticmethod
    def primary_classifier_prompt(text: str, context: Optional[str] = None) -> str:
        """Generate primary classifier prompt.

        Args:
            text: Legal text to classify
            context: Optional surrounding context

        Returns:
            Formatted prompt string
        """
        context_section = ""
        if context:
            context_section = f"\n\nSurrounding Context:\n{context[:1000]}"

        return f"""You are a legal document analyst. Classify the following legal text passage.

Legal Text:
{text[:MAX_TEXT_LENGTH]}
{context_section}

Classify this text into ONE of these roles:
- FACTS: Statement of facts, procedural history, case background
- HOLDING: The court's actual ruling, binding decision
- DICTA: Non-binding commentary, hypothetical discussions
- DISSENT: Dissenting opinion content
- CONCURRENCE: Concurring opinion content
- ANALYSIS: Legal reasoning, argument analysis
- PROCEDURAL: Procedural matters, jurisdiction, standing

Respond in this exact format:
Role: [PRIMARY_ROLE]
Confidence: [0.0-1.0]
Secondary: [ROLE1, ROLE2] (or "None" if not applicable)
Reasoning: [Brief explanation for classification]"""

    @staticmethod
    def context_aware_prompt(
        text: str,
        preceding: Optional[str] = None,
        following: Optional[str] = None,
    ) -> str:
        """Generate context-aware classification prompt.

        Args:
            text: Target text to classify
            preceding: Text that comes before
            following: Text that comes after

        Returns:
            Formatted prompt string
        """
        context_parts = []
        if preceding:
            context_parts.append(f"[PRECEDING]\n{preceding[:500]}")
        context_parts.append(f"[TARGET TEXT]\n{text[:MAX_TEXT_LENGTH]}")
        if following:
            context_parts.append(f"[FOLLOWING]\n{following[:500]}")

        combined = "\n\n".join(context_parts)

        return f"""Classify the [TARGET TEXT] based on its position in the legal document.

{combined}

Consider how the surrounding context affects the classification.
The target text should be classified as one of:
FACTS, HOLDING, DICTA, DISSENT, CONCURRENCE, ANALYSIS, PROCEDURAL

Respond in this exact format:
Role: [PRIMARY_ROLE]
Confidence: [0.0-1.0]
Secondary: [ROLE1, ROLE2] (or "None")
Reasoning: [Brief explanation]"""


class LegalClassifier:
    """Multi-agent legal text classifier.

    Uses LLM prompts to classify legal text passages into
    categories like facts, holdings, dicta, etc.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            llm_client: LLM client for classification
            config: Optional generation configuration

        Raises:
            ValueError: If llm_client is None
        """
        if llm_client is None:
            raise ValueError("llm_client cannot be None")

        self._llm = llm_client
        self._config = config or GenerationConfig(
            max_tokens=500,
            temperature=0.2,  # Low for consistent classification
        )

    def classify(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> LegalClassification:
        """Classify a single legal text passage.

        Args:
            text: Legal text to classify
            context: Optional surrounding context

        Returns:
            LegalClassification with role and confidence
        """
        if not text or not text.strip():
            return self._make_unknown(text, "Empty text provided")

        # Build prompt
        prompt = ClassificationPrompts.primary_classifier_prompt(text, context)

        # Generate classification
        try:
            response = self._llm.generate(prompt, self._config)
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._make_unknown(text, f"Classification error: {e}")

        # Parse response
        return self._parse_classification_response(text, response)

    def classify_with_context(
        self,
        text: str,
        preceding: Optional[str] = None,
        following: Optional[str] = None,
    ) -> LegalClassification:
        """Classify text considering surrounding context.

        Args:
            text: Target text to classify
            preceding: Text that comes before
            following: Text that comes after

        Returns:
            LegalClassification with context-informed role
        """
        if not text or not text.strip():
            return self._make_unknown(text, "Empty text provided")

        # Build context-aware prompt
        prompt = ClassificationPrompts.context_aware_prompt(text, preceding, following)

        # Generate classification
        try:
            response = self._llm.generate(prompt, self._config)
        except Exception as e:
            logger.error(f"Context-aware classification failed: {e}")
            return self._make_unknown(text, f"Classification error: {e}")

        return self._parse_classification_response(text, response)

    def classify_document(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[LegalClassification]:
        """Classify all chunks in a document.

        Args:
            chunks: List of chunk dictionaries with 'text' key

        Returns:
            List of LegalClassification results
        """
        if not chunks:
            return []
        chunks = chunks[:MAX_CHUNKS_PER_BATCH]

        results = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            preceding = chunks[i - 1].get("text") if i > 0 else None
            following = chunks[i + 1].get("text") if i < len(chunks) - 1 else None

            result = self.classify_with_context(text, preceding, following)
            results.append(result)

        return results

    def enrich_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add legal role metadata to chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Chunks with 'legal_role' and related metadata added
        """
        if not chunks:
            return []

        # Get classifications
        classifications = self.classify_document(chunks)

        # Enrich each chunk
        enriched = []
        for chunk, classification in zip(chunks, classifications):
            enriched_chunk = chunk.copy()
            enriched_chunk["legal_role"] = classification.role
            enriched_chunk["legal_confidence"] = classification.confidence
            enriched_chunk["legal_secondary_roles"] = classification.secondary_roles
            enriched_chunk["legal_reasoning"] = classification.reasoning
            enriched_chunk["legal_is_binding"] = classification.is_binding
            enriched.append(enriched_chunk)

        return enriched

    def _parse_classification_response(
        self,
        text: str,
        response: str,
    ) -> LegalClassification:
        """Parse LLM response into classification.

        Args:
            text: Original text
            response: LLM response to parse

        Returns:
            Parsed LegalClassification
        """
        if not response or not response.strip():
            return self._make_unknown(text, "Empty LLM response")

        role = self._extract_role(response)
        confidence = self._extract_confidence(response)
        secondary = self._extract_secondary_roles(response)
        reasoning = self._extract_reasoning(response)

        return LegalClassification(
            text=text,
            role=role,
            confidence=confidence,
            secondary_roles=secondary,
            reasoning=reasoning,
        )

    def _extract_role(self, response: str) -> str:
        """Extract primary role from response.

        Args:
            response: LLM response text

        Returns:
            Role string or UNKNOWN
        """
        match = re.search(
            r"Role:\s*(\w+)",
            response,
            re.IGNORECASE,
        )
        if match:
            role = match.group(1).upper()
            if role in [r.value for r in LegalRole]:
                return role

        # Try to find role keyword anywhere
        for legal_role in LegalRole:
            if legal_role.value in response.upper():
                return legal_role.value

        return LegalRole.UNKNOWN.value

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response.

        Args:
            response: LLM response text

        Returns:
            Confidence value (0.0-1.0)
        """
        match = re.search(
            r"Confidence:\s*([\d.]+)",
            response,
            re.IGNORECASE,
        )
        if match:
            try:
                return float(match.group(1))
            except ValueError as e:
                logger.debug(f"Failed to parse confidence: {e}")

        return 0.5  # Default confidence

    def _extract_secondary_roles(self, response: str) -> List[str]:
        """Extract secondary roles from response.

        Args:
            response: LLM response text

        Returns:
            List of secondary role strings
        """
        match = re.search(
            r"Secondary:\s*\[?([^\]\n]+)\]?",
            response,
            re.IGNORECASE,
        )
        if not match:
            return []

        content = match.group(1).strip()
        if content.lower() == "none":
            return []

        # Parse comma-separated roles
        roles = []
        for part in content.split(","):
            role = part.strip().upper()
            if role in [r.value for r in LegalRole]:
                roles.append(role)

        return roles[:MAX_SECONDARY_ROLES]

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response.

        Args:
            response: LLM response text

        Returns:
            Reasoning string
        """
        match = re.search(
            r"Reasoning:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()[:MAX_REASONING_LENGTH]

        return ""

    def _make_unknown(
        self,
        text: str,
        reason: str,
    ) -> LegalClassification:
        """Create unknown classification.

        Args:
            text: Original text
            reason: Reason for unknown classification

        Returns:
            LegalClassification with UNKNOWN role
        """
        return LegalClassification(
            text=text,
            role=LegalRole.UNKNOWN.value,
            confidence=0.0,
            reasoning=reason,
        )


def create_legal_classifier(
    llm_client: LLMClient,
    config: Optional[GenerationConfig] = None,
) -> LegalClassifier:
    """Factory function to create legal classifier.

    Args:
        llm_client: LLM client for classification
        config: Optional generation configuration

    Returns:
        Configured LegalClassifier
    """
    return LegalClassifier(llm_client=llm_client, config=config)


def classify_legal_text(
    text: str,
    llm_client: LLMClient,
    context: Optional[str] = None,
) -> LegalClassification:
    """Convenience function for single text classification.

    Args:
        text: Legal text to classify
        llm_client: LLM client for classification
        context: Optional surrounding context

    Returns:
        LegalClassification result
    """
    classifier = create_legal_classifier(llm_client)
    return classifier.classify(text, context)
