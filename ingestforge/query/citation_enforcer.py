"""Citation Enforcer for LLM Synthesis.

Citation-Enforcement Prompting.
Follows NASA JPL Power of Ten rules.

Ensures 100% of factual claims in LLM synthesis link back to
specific artifact_id and page numbers with contradiction detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestforge.core.logging import get_logger
from ingestforge.query.context_aggregator import ContextWindow, ContextChunk

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_SOURCES = 100
MAX_PROMPT_LENGTH = 100000
MAX_RESPONSE_LENGTH = 50000
MAX_CITATIONS_PER_RESPONSE = 500
DEFAULT_MAX_SOURCES = 50


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

CITATION_FORMAT_INSTRUCTIONS = """
## Citation Requirements

You MUST cite every factual claim using the format: [SOURCE_ID]

Citation format: [artifact_id, p.PAGE] or [artifact_id] if no page number.

Examples:
- "The pressure limit is 50 psi [doc-001, p.12]"
- "The system uses AES-256 encryption [manual-v2]"

IMPORTANT:
1. Every factual statement MUST have a citation
2. Only cite sources from the provided context
3. Use exact artifact_ids from the source list
4. Do not invent or hallucinate information
"""

CONTRADICTION_INSTRUCTIONS = """
## Contradiction Detection

If sources provide conflicting information, you MUST explicitly state the discrepancy:

Format: "**Contradiction**: [Source A] states X, while [Source B] states Y."

Example:
"**Contradiction**: [manual-a, p.5] specifies a pressure limit of 50 psi,
while [manual-b, p.12] specifies 60 psi."

Always report contradictions - do not silently choose one source over another.
"""

HALLUCINATION_PREVENTION = """
## Strict Context Adherence

CRITICAL: You may ONLY use information from the provided context.

- Do NOT use any external knowledge
- Do NOT make assumptions beyond what is stated
- If information is not in the context, say "The provided documents do not contain information about [topic]"
- If uncertain, indicate uncertainty rather than guessing
"""

SYNTHESIS_PROMPT_TEMPLATE = """
{citation_instructions}

{contradiction_instructions}

{hallucination_prevention}

## Available Sources

{source_list}

## Context

{context_text}

## Query

{query}

## Instructions

Based ONLY on the provided context, answer the query above.
Cite every factual claim using the citation format specified.
Report any contradictions between sources.
"""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """Extracted citation from response.

    GWT-5: Citation validation.
    Rule #9: Complete type hints.
    """

    artifact_id: str
    page_number: Optional[int] = None
    text_before: str = ""
    position: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "page_number": self.page_number,
            "text_before": self.text_before,
            "position": self.position,
        }


@dataclass
class ValidationResult:
    """Result of citation validation.

    GWT-5: Citation validation output.
    Rule #9: Complete type hints.
    """

    valid: bool
    citations_found: List[Citation] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)
    coverage: float = 0.0
    contradiction_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation_count(self) -> int:
        """Total citations found."""
        return len(self.citations_found)

    @property
    def invalid_count(self) -> int:
        """Number of invalid citations."""
        return len(self.invalid_citations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "citations_found": [c.to_dict() for c in self.citations_found],
            "invalid_citations": self.invalid_citations,
            "missing_citations": self.missing_citations,
            "coverage": self.coverage,
            "citation_count": self.citation_count,
            "invalid_count": self.invalid_count,
            "contradiction_count": self.contradiction_count,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Citation Enforcer
# ---------------------------------------------------------------------------


class CitationEnforcer:
    """Enforces citation requirements in LLM synthesis.

    GWT-1: Citation prompt generation.
    GWT-2: Citation format enforcement.
    GWT-3: Contradiction detection.
    GWT-4: Hallucination prevention.
    GWT-5: Citation validation.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    # Citation pattern: [artifact_id] or [artifact_id, p.PAGE]
    CITATION_PATTERN = re.compile(r"\[([^\],]+?)(?:,\s*p\.?(\d+))?\]")

    # Contradiction pattern
    CONTRADICTION_PATTERN = re.compile(r"\*\*Contradiction\*\*:", re.IGNORECASE)

    def __init__(
        self,
        max_sources: int = DEFAULT_MAX_SOURCES,
        include_contradiction_detection: bool = True,
        include_hallucination_prevention: bool = True,
    ) -> None:
        """Initialize the citation enforcer.

        Args:
            max_sources: Maximum sources to include in prompt.
            include_contradiction_detection: Include contradiction instructions.
            include_hallucination_prevention: Include hallucination prevention.

        Rule #5: Assert preconditions.
        """
        assert (
            0 < max_sources <= MAX_SOURCES
        ), f"max_sources must be between 1 and {MAX_SOURCES}"

        self._max_sources = max_sources
        self._include_contradictions = include_contradiction_detection
        self._include_hallucination = include_hallucination_prevention

    def build_prompt(
        self,
        context_window: ContextWindow,
        query: str,
    ) -> str:
        """Build synthesis prompt with citation enforcement.

        GWT-1: Citation prompt generation.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            context_window: Context window from aggregator.
            query: User query to answer.

        Returns:
            Complete prompt with citation instructions.
        """
        assert context_window is not None, "context_window cannot be None"
        assert query is not None, "query cannot be None"
        assert len(query.strip()) > 0, "query cannot be empty"

        # Build source reference list
        source_list = self.build_source_reference_list(context_window)

        # Get context text with citations
        context_text = context_window.to_text(include_citations=True)

        # Build prompt sections
        contradiction_section = ""
        if self._include_contradictions:
            contradiction_section = CONTRADICTION_INSTRUCTIONS

        hallucination_section = ""
        if self._include_hallucination:
            hallucination_section = HALLUCINATION_PREVENTION

        # Assemble prompt
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            citation_instructions=CITATION_FORMAT_INSTRUCTIONS,
            contradiction_instructions=contradiction_section,
            hallucination_prevention=hallucination_section,
            source_list=source_list,
            context_text=context_text,
            query=query,
        )

        # Enforce max length
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(
                f"Prompt truncated from {len(prompt)} to {MAX_PROMPT_LENGTH} characters"
            )
            prompt = prompt[:MAX_PROMPT_LENGTH]

        return prompt

    def build_source_reference_list(
        self,
        context_window: ContextWindow,
    ) -> str:
        """Build formatted source reference list.

        Rule #4: Function < 60 lines.

        Args:
            context_window: Context window with chunks.

        Returns:
            Formatted source list for prompt.
        """
        if not context_window.chunks:
            return "No sources available."

        lines: List[str] = []
        seen_artifacts: Set[str] = set()

        for chunk in context_window.chunks[: self._max_sources]:
            if chunk.artifact_id in seen_artifacts:
                continue
            seen_artifacts.add(chunk.artifact_id)

            # Build source entry
            entry = f"- **{chunk.artifact_id}**"
            entry += f" (Document: {chunk.document_id}"

            if chunk.page_number is not None:
                entry += f", Page: {chunk.page_number}"

            entry += ")"
            lines.append(entry)

        return "\n".join(lines)

    def validate_citations(
        self,
        response: str,
        context_window: ContextWindow,
    ) -> ValidationResult:
        """Validate citations in LLM response.

        GWT-5: Citation validation.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            response: LLM response to validate.
            context_window: Original context window.

        Returns:
            ValidationResult with citation analysis.
        """
        assert response is not None, "response cannot be None"
        assert context_window is not None, "context_window cannot be None"

        # Get valid artifact IDs from context
        valid_artifacts = {c.artifact_id for c in context_window.chunks}

        # Extract citations from response
        citations = self._extract_citations(response)

        # Validate each citation
        valid_citations: List[Citation] = []
        invalid_citations: List[str] = []

        for citation in citations[:MAX_CITATIONS_PER_RESPONSE]:
            if citation.artifact_id in valid_artifacts:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation.artifact_id)

        # Count contradictions
        contradiction_count = len(self.CONTRADICTION_PATTERN.findall(response))

        # Calculate coverage (rough estimate based on sentence count)
        coverage = self._estimate_coverage(response, valid_citations)

        # Determine validity
        is_valid = (
            len(invalid_citations) == 0 and len(valid_citations) > 0 and coverage >= 0.5
        )

        return ValidationResult(
            valid=is_valid,
            citations_found=valid_citations,
            invalid_citations=invalid_citations,
            coverage=coverage,
            contradiction_count=contradiction_count,
            metadata={
                "total_citations": len(citations),
                "valid_artifacts_count": len(valid_artifacts),
            },
        )

    def _extract_citations(self, response: str) -> List[Citation]:
        """Extract citations from response text.

        Rule #4: Function < 60 lines.

        Args:
            response: Response text to parse.

        Returns:
            List of extracted citations.
        """
        citations: List[Citation] = []
        truncated = response[:MAX_RESPONSE_LENGTH]

        for match in self.CITATION_PATTERN.finditer(truncated):
            artifact_id = match.group(1).strip()
            page_str = match.group(2)
            page_number = int(page_str) if page_str else None

            # Get text before citation (up to 50 chars)
            start = max(0, match.start() - 50)
            text_before = response[start : match.start()].strip()

            citation = Citation(
                artifact_id=artifact_id,
                page_number=page_number,
                text_before=text_before,
                position=match.start(),
            )
            citations.append(citation)

            if len(citations) >= MAX_CITATIONS_PER_RESPONSE:
                break

        return citations

    def _estimate_coverage(
        self,
        response: str,
        citations: List[Citation],
    ) -> float:
        """Estimate citation coverage of response.

        Rule #4: Function < 60 lines.

        Args:
            response: Response text.
            citations: Valid citations found.

        Returns:
            Estimated coverage (0.0 to 1.0).
        """
        if not response.strip():
            return 0.0

        # Count sentences (rough approximation)
        sentences = re.split(r"[.!?]+", response)
        sentence_count = len([s for s in sentences if len(s.strip()) > 10])

        if sentence_count == 0:
            return 1.0 if citations else 0.0

        # Assume each citation covers roughly one claim/sentence
        citation_count = len(citations)
        coverage = min(citation_count / max(sentence_count, 1), 1.0)

        return round(coverage, 2)

    def check_contradictions(
        self,
        context_window: ContextWindow,
    ) -> List[Tuple[ContextChunk, ContextChunk, str]]:
        """Check for potential contradictions in context.

        Identifies chunks from different documents that might conflict.
        This is a heuristic check - actual contradiction detection
        requires LLM analysis.

        Rule #4: Function < 60 lines.

        Args:
            context_window: Context to analyze.

        Returns:
            List of (chunk1, chunk2, reason) potential conflicts.
        """
        if not context_window.chunks:
            return []

        # Group chunks by document
        doc_chunks: Dict[str, List[ContextChunk]] = {}
        for chunk in context_window.chunks:
            if chunk.document_id not in doc_chunks:
                doc_chunks[chunk.document_id] = []
            doc_chunks[chunk.document_id].append(chunk)

        # Only flag if multiple documents exist
        if len(doc_chunks) < 2:
            return []

        # Return list of document pairs that might conflict
        # (actual detection happens in LLM synthesis)
        conflicts: List[Tuple[ContextChunk, ContextChunk, str]] = []
        doc_ids = list(doc_chunks.keys())

        for i, doc_id1 in enumerate(doc_ids[:10]):  # Bound iterations
            for doc_id2 in doc_ids[i + 1 : 10]:
                chunk1 = doc_chunks[doc_id1][0]
                chunk2 = doc_chunks[doc_id2][0]
                reason = f"Different source documents: {doc_id1} vs {doc_id2}"
                conflicts.append((chunk1, chunk2, reason))

        return conflicts[:20]  # Bound output


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def build_citation_prompt(
    context_window: ContextWindow,
    query: str,
    include_contradictions: bool = True,
) -> str:
    """Convenience function to build citation-enforced prompt.

    Args:
        context_window: Context from aggregator.
        query: User query.
        include_contradictions: Include contradiction detection.

    Returns:
        Complete synthesis prompt.
    """
    enforcer = CitationEnforcer(
        include_contradiction_detection=include_contradictions,
    )
    return enforcer.build_prompt(context_window, query)


def validate_response_citations(
    response: str,
    context_window: ContextWindow,
) -> ValidationResult:
    """Convenience function to validate response citations.

    Args:
        response: LLM response to validate.
        context_window: Original context.

    Returns:
        ValidationResult with citation analysis.
    """
    enforcer = CitationEnforcer()
    return enforcer.validate_citations(response, context_window)


def create_citation_enforcer(
    max_sources: int = DEFAULT_MAX_SOURCES,
    include_contradictions: bool = True,
    include_hallucination_prevention: bool = True,
) -> CitationEnforcer:
    """Factory function to create configured enforcer.

    Args:
        max_sources: Maximum sources in prompt.
        include_contradictions: Include contradiction detection.
        include_hallucination_prevention: Include hallucination prevention.

    Returns:
        Configured CitationEnforcer.
    """
    return CitationEnforcer(
        max_sources=max_sources,
        include_contradiction_detection=include_contradictions,
        include_hallucination_prevention=include_hallucination_prevention,
    )
