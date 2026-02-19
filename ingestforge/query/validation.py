"""
Answer Validation for RAG Quality Assurance.

Validates generated answers against source context to ensure:
1. Key claims are supported by source content
2. Numbers, dates, and names are accurate
3. Citations reference actual sources
4. Coverage score indicates answer completeness

Architecture Context
--------------------
Answer validation integrates into the QueryPipeline after answer generation:

    QueryPipeline._generate_answer(query, sources)
            |
    AnswerValidator.validate(answer, context, query)
            |
    ValidationReport
            |
    If invalid + low coverage -> Regenerate with strict mode

The validator uses simple heuristics (not ML) to check answer quality:
- Extract factual claims from the answer
- Check if claims appear in source context
- Verify citations reference real sources
- Calculate coverage score

Reference Implementation
------------------------
Based on Army Doctrine RAG validation patterns that trigger
regeneration when answers have unsupported claims.
"""

import re
from dataclasses import dataclass, field
from typing import List, Any

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Report from answer validation.

    Attributes:
        is_valid: Whether the answer passes validation
        coverage_score: 0-1 score of how much answer is supported by sources
        unsupported_claims: List of claims not found in context
        warnings: Non-critical issues found
        citation_count: Number of citations found in answer
        missing_citations: Claims that should have citations but don't
    """

    is_valid: bool
    coverage_score: float
    unsupported_claims: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    citation_count: int = 0
    missing_citations: List[str] = field(default_factory=list)


class AnswerValidator:
    """
    Validates RAG-generated answers against source context.

    Performs lightweight validation without requiring LLM calls:
    - Checks citation references are valid
    - Extracts key factual claims and verifies against context
    - Calculates coverage score
    - Identifies potential unsupported statements
    """

    def __init__(
        self,
        min_coverage_threshold: float = 0.6,
        require_citations: bool = True,
    ):
        """
        Initialize validator.

        Args:
            min_coverage_threshold: Minimum coverage score to pass validation
            require_citations: Whether to require citation markers in answer
        """
        self.min_coverage_threshold = min_coverage_threshold
        self.require_citations = require_citations

    def _validate_citations(
        self, answer: str, num_sources: int
    ) -> tuple[List[int], List[str]]:
        """
        Validate citations in answer.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            answer: Answer text
            num_sources: Number of sources provided

        Returns:
            Tuple of (citations, warnings)
        """
        citations = self._extract_citations(answer)
        warnings = []

        invalid_citations = [c for c in citations if c > num_sources]
        if invalid_citations:
            warnings.append(
                f"Invalid citation references: {invalid_citations} "
                f"(only {num_sources} sources provided)"
            )

        return (citations, warnings)

    def _validate_claims(
        self, answer: str, context_lower: str, citations: List[int]
    ) -> tuple[int, int, List[str], List[str]]:
        """
        Validate factual claims in answer.

        Rule #1: Early continue pattern
        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            answer: Answer text
            context_lower: Lowercase context for comparison
            citations: List of citation numbers

        Returns:
            Tuple of (supported_claims, total_claims, unsupported_claims, missing_citations)
        """
        claims = self._extract_claims(answer)
        supported_claims = 0
        unsupported_claims = []
        missing_citations = []
        for claim in claims:
            claim_lower = claim.lower()
            if self._claim_supported(claim_lower, context_lower):
                supported_claims += 1
                continue

            unsupported_claims.append(claim)
            if not self._claim_has_citation(claim, citations, answer):
                missing_citations.append(claim)

        return (supported_claims, len(claims), unsupported_claims, missing_citations)

    def _validate_value_claims(self, answer: str, context: str) -> List[str]:
        """
        Validate specific value claims (numbers, dates, names).

        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            answer: Answer text
            context: Source context

        Returns:
            List of warnings
        """
        warnings = []
        value_claims = self._extract_value_claims(answer)
        for value, claim_text in value_claims:
            if value not in context:
                warnings.append(
                    f"Value '{value}' not found in sources: '{claim_text[:50]}...'"
                )

        return warnings

    def _calculate_coverage(
        self, supported_claims: int, total_claims: int, answer: str
    ) -> float:
        """
        Calculate coverage score.

        Rule #1: Early return pattern
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            supported_claims: Number of supported claims
            total_claims: Total number of claims
            answer: Answer text

        Returns:
            Coverage score (0.0 to 1.0)
        """
        if total_claims > 0:
            return supported_claims / total_claims

        # No extractable claims - assume valid if answer is reasonable
        return 1.0 if len(answer) > 50 else 0.5

    def _check_not_found_response(self, answer_lower: str) -> bool:
        """
        Check if answer is a "not found" response.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            answer_lower: Lowercase answer text

        Returns:
            True if answer indicates information not found
        """
        not_found_phrases = [
            "not in the provided sources",
            "not in sources",
            "cannot find",
            "no information",
            "not mentioned",
        ]
        return any(phrase in answer_lower for phrase in not_found_phrases)

    def validate(
        self,
        answer: str,
        context: str,
        query: str,
        num_sources: int = 5,
    ) -> ValidationReport:
        """
        Validate answer against source context.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Checks:
        1. Key claims appear in context
        2. Numbers/dates/names are accurate
        3. Citations reference actual sources
        4. Overall coverage score

        Args:
            answer: Generated answer to validate
            context: Concatenated source content used for generation
            query: Original user query
            num_sources: Number of sources provided (for citation validation)

        Returns:
            ValidationReport with validation results
        """
        # Normalize texts for comparison
        answer_lower = answer.lower()
        context_lower = context.lower()

        # Validate components using helpers
        citations, warnings = self._validate_citations(answer, num_sources)
        supported, total, unsupported, missing = self._validate_claims(
            answer, context_lower, citations
        )
        warnings.extend(self._validate_value_claims(answer, context))

        # Calculate metrics
        coverage_score = self._calculate_coverage(supported, total, answer)
        is_not_found = self._check_not_found_response(answer_lower)
        is_valid = coverage_score >= self.min_coverage_threshold or is_not_found

        # Add citation warning if needed
        if self.require_citations and len(citations) == 0 and not is_not_found:
            warnings.append("No citations found in answer")

        return ValidationReport(
            is_valid=is_valid,
            coverage_score=coverage_score,
            unsupported_claims=unsupported[:5],
            warnings=warnings,
            citation_count=len(citations),
            missing_citations=missing[:3],
        )

    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text like [1], [2], etc."""
        pattern = r"\[(\d+)\]"
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims (sentences) from the answer.

        Filters out:
        - Questions
        - Hedged statements ("might be", "could be")
        - Meta-statements about sources
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", answer)

        claims = []
        hedge_patterns = [
            r"\bmight\b",
            r"\bcould\b",
            r"\bperhaps\b",
            r"\bpossibly\b",
            r"\bit seems\b",
            r"\bappears to\b",
            r"\bmay\b",
        ]
        meta_patterns = [
            r"\baccording to\b",
            r"\bthe sources?\b",
            r"\bthe documents?\b",
            r"\bprovided\b",
            r"\breferences?\b",
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue

            # Skip questions
            if "?" in sentence:
                continue

            # Skip hedged statements
            if any(re.search(p, sentence, re.I) for p in hedge_patterns):
                continue

            # Skip pure meta-statements (but keep cited claims)
            if any(re.search(p, sentence, re.I) for p in meta_patterns):
                if not re.search(r"\[\d+\]", sentence):
                    continue

            claims.append(sentence)

        return claims

    def _claim_supported(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by context content.

        Uses keyword overlap to determine support. A claim is considered
        supported if significant content words appear in context.
        """
        # Extract content words (skip common words)
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "and",
            "but",
            "or",
            "if",
            "because",
            "until",
            "while",
            "although",
            "though",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "their",
            "them",
            "which",
            "who",
            "whom",
            "whose",
        }

        # Tokenize claim
        words = re.findall(r"\b[a-z]+\b", claim)
        content_words = [w for w in words if w not in stopwords and len(w) > 2]

        if not content_words:
            return True  # No content words to verify

        # Check how many content words appear in context
        found_words = sum(1 for w in content_words if w in context)
        coverage = found_words / len(content_words)

        # Require at least 50% of content words to be found
        return coverage >= 0.5

    def _claim_has_citation(
        self,
        claim: str,
        citations: List[int],
        full_answer: str,
    ) -> bool:
        """Check if a claim has an associated citation."""
        # Find position of claim in answer
        claim_pos = full_answer.find(claim)
        if claim_pos == -1:
            return False

        # Check if there's a citation within or shortly after the claim
        claim_end = claim_pos + len(claim)
        search_window = full_answer[claim_pos : min(claim_end + 20, len(full_answer))]

        return bool(re.search(r"\[\d+\]", search_window))

    def _extract_value_claims(self, answer: str) -> List[tuple[Any, ...]]:
        """Extract specific value claims (numbers, dates, percentages).

        Returns list of (value, surrounding_text) tuples.
        """
        value_claims = []

        # Numbers with context
        number_pattern = r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:percent|%|dollars|\$|people|items|years?|months?|days?|hours?)?"
        for match in re.finditer(number_pattern, answer, re.I):
            value = match.group(1)
            start = max(0, match.start() - 30)
            end = min(len(answer), match.end() + 30)
            context = answer[start:end]
            value_claims.append((value, context))

        # Dates
        date_pattern = r"\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b"
        for match in re.finditer(date_pattern, answer, re.I):
            value = match.group(1)
            start = max(0, match.start() - 20)
            end = min(len(answer), match.end() + 20)
            context = answer[start:end]
            value_claims.append((value, context))

        return value_claims


def validate_answer(
    answer: str,
    context: str,
    query: str,
    num_sources: int = 5,
    min_coverage: float = 0.6,
) -> ValidationReport:
    """
    Convenience function to validate an answer.

    Args:
        answer: Generated answer to validate
        context: Source content used for generation
        query: Original user query
        num_sources: Number of sources provided
        min_coverage: Minimum coverage threshold

    Returns:
        ValidationReport with validation results
    """
    validator = AnswerValidator(min_coverage_threshold=min_coverage)
    return validator.validate(answer, context, query, num_sources)
