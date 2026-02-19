"""
Query Clarification Agent.

Ambiguity Guard that evaluates user queries before execution
to prevent low-quality RAG results.

Follows NASA JPL Power of Ten:
- Rule #1: No recursion
- Rule #2: Fixed bounds on all data structures
- Rule #4: Functions under 60 lines
- Rule #5: Assertions at entry points
- Rule #7: Check model response; fail-fast if invalid
- Rule #9: Complete type hints
"""

from __future__ import annotations

from enum import Enum
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import pattern definitions
from ingestforge.query.ambiguity_patterns import (
    CONTEXTUAL_PRONOUNS,
    DEMONSTRATIVE_PRONOUNS,
    MULTI_MEANING_TERMS,
    VAGUE_TEMPORAL_QUALIFIERS,
    VAGUE_PATTERNS,
    SPECIFIC_PATTERNS,
    BROAD_SCOPE_INDICATORS,
    SPECIFIC_SCOPE_INDICATORS,
)

# JPL Rule #2: Fixed upper bounds
MAX_QUERY_LENGTH = 2000
MAX_SUGGESTIONS = 5
MIN_QUERY_LENGTH = 2
CLARITY_THRESHOLD = 0.7
MAX_LLM_RETRIES = 2
MAX_QUESTIONS = 5
MAX_CONTEXT_QUERIES = 10
MAX_AMBIGUOUS_TERMS = 20

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class AmbiguityType(str, Enum):
    """Types of ambiguity in queries."""

    PRONOUN = "pronoun_reference"
    MULTI_MEANING = "multi_meaning"
    TEMPORAL = "temporal_ambiguity"
    SCOPE = "scope_ambiguity"
    VAGUE = "vague_query"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ClarityScore:
    """
    Score indicating query clarity.

    Rule #9: Complete type hints.
    """

    score: float  # 0.0 to 1.0, higher = clearer
    is_clear: bool
    factors: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert 0.0 <= self.score <= 1.0, f"score must be 0-1, got {self.score}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "is_clear": self.is_clear,
            "factors": self.factors,
        }


@dataclass
class ClarificationQuestion:
    """
    A single clarification question with multiple choice options.

    Structured question format for UI rendering.
    Rule #9: Complete type hints.
    """

    type: AmbiguityType
    question: str
    options: List[str]
    confidence: float
    term: Optional[str] = None  # The ambiguous term being clarified

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.question, "question must be non-empty"
        assert len(self.options) >= 2, "must have at least 2 options"
        assert (
            0.0 <= self.confidence <= 1.0
        ), f"confidence must be 0-1, got {self.confidence}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "question": self.question,
            "options": self.options,
            "confidence": self.confidence,
            "term": self.term,
        }


@dataclass
class AmbiguityReport:
    """
    Detailed report of query ambiguities.

    Enhanced ambiguity detection report.
    Rule #9: Complete type hints.
    """

    is_ambiguous: bool
    ambiguity_score: float
    ambiguous_terms: List[Tuple[AmbiguityType, str]]
    questions: List[ClarificationQuestion]
    suggested_query: Optional[str] = None

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert (
            0.0 <= self.ambiguity_score <= 1.0
        ), f"score must be 0-1, got {self.ambiguity_score}"
        assert (
            len(self.questions) <= MAX_QUESTIONS
        ), f"too many questions: {len(self.questions)}"
        assert (
            len(self.ambiguous_terms) <= MAX_AMBIGUOUS_TERMS
        ), f"too many terms: {len(self.ambiguous_terms)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_score": self.ambiguity_score,
            "ambiguous_terms": [(t.value, s) for t, s in self.ambiguous_terms],
            "questions": [q.to_dict() for q in self.questions],
            "suggested_query": self.suggested_query,
        }


@dataclass
class ClarificationArtifact:
    """
    Artifact containing clarification suggestions.

    Rule #9: Complete type hints.
    """

    original_query: str
    clarity_score: ClarityScore
    suggestions: List[str]
    reason: str
    needs_clarification: bool
    ambiguity_report: Optional[AmbiguityReport] = None  # Enhanced report

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.original_query, "original_query must be non-empty"
        assert len(self.suggestions) <= MAX_SUGGESTIONS, "too many suggestions"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "original_query": self.original_query,
            "clarity_score": self.clarity_score.to_dict(),
            "suggestions": self.suggestions,
            "reason": self.reason,
            "needs_clarification": self.needs_clarification,
        }
        if self.ambiguity_report:
            result["ambiguity_report"] = self.ambiguity_report.to_dict()
        return result


@dataclass
class ClarifierConfig:
    """
    Configuration for query clarifier.

    Rule #9: Complete type hints.
    """

    threshold: float = CLARITY_THRESHOLD
    use_llm: bool = False
    max_suggestions: int = 3

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert 0.0 <= self.threshold <= 1.0, "threshold must be 0-1"
        assert 1 <= self.max_suggestions <= MAX_SUGGESTIONS, "invalid max_suggestions"


# =============================================================================
# Query Clarifier
# =============================================================================


class IFQueryClarifier:
    """
    Evaluates query clarity and suggests refinements.

    Ambiguity Guard for RAG quality.

    Rule #2: Fixed bounds on query length.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        config: Optional[ClarifierConfig] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Initialize query clarifier.

        Args:
            config: Clarifier configuration.
            llm_fn: Optional LLM function for enhanced clarification.

        Rule #5: Assert preconditions.
        """
        self._config = config or ClarifierConfig()
        self._llm_fn = llm_fn

    def detect_pronouns(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Detect unclear pronoun references.

        Pronoun ambiguity detection.
        Rule #4: Under 60 lines.

        Args:
            query: The query to analyze.
            context: Optional conversation context for resolution.

        Returns:
            List of ambiguous pronouns found.
        """
        query_lower = query.lower()
        words = set(query_lower.split())
        ambiguous: List[str] = []

        # JPL Rule #2: Bounded iteration over PRONOUN_PATTERNS
        for pronoun in CONTEXTUAL_PRONOUNS:
            if pronoun in words:
                # Check if resolvable from context
                if not self._can_resolve_from_context(pronoun, context):
                    ambiguous.append(pronoun)

        # Check demonstrative pronouns
        for pronoun in DEMONSTRATIVE_PRONOUNS:
            if pronoun in words:
                if not self._can_resolve_from_context(pronoun, context):
                    ambiguous.append(pronoun)

        # JPL Rule #2: Limit results
        return ambiguous[:MAX_AMBIGUOUS_TERMS]

    def detect_multi_meaning(self, query: str) -> List[Tuple[str, List[str]]]:
        """
        Detect terms with multiple meanings.

        Multi-meaning term detection.
        Rule #4: Under 60 lines.

        Args:
            query: The query to analyze.

        Returns:
            List of (term, possible_meanings) tuples.
        """
        query_lower = query.lower()
        multi_meaning: List[Tuple[str, List[str]]] = []

        # JPL Rule #2: Bounded iteration over MULTI_MEANING_TERMS
        for term, meanings in MULTI_MEANING_TERMS.items():
            if term in query_lower:
                multi_meaning.append((term, meanings))

        # JPL Rule #2: Limit results
        return multi_meaning[:MAX_AMBIGUOUS_TERMS]

    def detect_temporal_ambiguity(self, query: str) -> List[str]:
        """
        Detect vague temporal references.

        Temporal ambiguity detection.
        Rule #4: Under 60 lines.

        Args:
            query: The query to analyze.

        Returns:
            List of ambiguous temporal terms.
        """
        query_lower = query.lower()
        words = set(query_lower.split())
        temporal_terms: List[str] = []

        # JPL Rule #2: Bounded iteration over TEMPORAL_AMBIGUITY_PATTERNS
        for term in VAGUE_TEMPORAL_QUALIFIERS:
            if term in words:
                temporal_terms.append(term)

        # JPL Rule #2: Limit results
        return temporal_terms[:MAX_AMBIGUOUS_TERMS]

    def detect_scope_ambiguity(self, query: str) -> bool:
        """
        Detect overly broad scope.

        Scope ambiguity detection.
        Rule #4: Under 60 lines.

        Args:
            query: The query to analyze.

        Returns:
            True if query appears too broad.
        """
        query_lower = query.lower()
        words = set(query_lower.split())

        # Check for broad scope indicators
        has_broad = bool(words & BROAD_SCOPE_INDICATORS)
        has_specific = bool(words & SPECIFIC_SCOPE_INDICATORS)

        # Broad scope without specificity = ambiguous
        return has_broad and not has_specific

    def generate_questions(
        self, ambiguous_terms: List[Tuple[AmbiguityType, str]], query: str
    ) -> List[ClarificationQuestion]:
        """
        Generate clarification questions from ambiguous terms.

        Question generation for UI.
        Rule #4: Under 60 lines.

        Args:
            ambiguous_terms: List of (type, term) tuples.
            query: Original query for context.

        Returns:
            List of clarification questions.
        """
        questions: List[ClarificationQuestion] = []

        # JPL Rule #2: Bounded iteration
        for amb_type, term in ambiguous_terms[:MAX_QUESTIONS]:
            if amb_type == AmbiguityType.PRONOUN:
                questions.append(
                    ClarificationQuestion(
                        type=AmbiguityType.PRONOUN,
                        question=f"Who or what does '{term}' refer to?",
                        options=["Specify referent", "Context unclear"],
                        confidence=0.8,
                        term=term,
                    )
                )
            elif amb_type == AmbiguityType.MULTI_MEANING:
                meanings = MULTI_MEANING_TERMS.get(term, [])
                if meanings:
                    questions.append(
                        ClarificationQuestion(
                            type=AmbiguityType.MULTI_MEANING,
                            question=f"Which '{term}' do you mean?",
                            options=meanings[:4],  # Max 4 options
                            confidence=0.9,
                            term=term,
                        )
                    )
            elif amb_type == AmbiguityType.TEMPORAL:
                questions.append(
                    ClarificationQuestion(
                        type=AmbiguityType.TEMPORAL,
                        question=f"What time range does '{term}' refer to?",
                        options=[
                            "Past week",
                            "Past month",
                            "Past year",
                            "Custom range",
                        ],
                        confidence=0.7,
                        term=term,
                    )
                )

        return questions[:MAX_QUESTIONS]

    def refine_query(
        self, original_query: str, clarifications: Dict[str, str]
    ) -> Tuple[str, float]:
        """
        Refine query by applying clarifications.

        Query refinement after clarification.
        Rule #4: Under 60 lines.

        Args:
            original_query: The original ambiguous query.
            clarifications: Map of term -> clarification.

        Returns:
            Tuple of (refined_query, confidence).
        """
        # JPL Rule #5: Validate inputs
        assert original_query, "original_query cannot be empty"
        assert len(clarifications) <= MAX_QUESTIONS, "too many clarifications"

        refined = original_query
        confidence = 0.5  # Base confidence

        # Apply each clarification
        # JPL Rule #2: Bounded iteration
        for term, clarification in list(clarifications.items())[:MAX_QUESTIONS]:
            if term in refined.lower():
                # Replace term with clarification
                refined = re.sub(
                    rf"\b{re.escape(term)}\b",
                    clarification,
                    refined,
                    flags=re.IGNORECASE,
                )
                confidence += 0.15  # Boost confidence per clarification

        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)

        return refined, confidence

    def _can_resolve_from_context(
        self, pronoun: str, context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Check if pronoun can be resolved from context.

        Rule #4: Under 60 lines.

        Args:
            pronoun: The pronoun to resolve.
            context: Conversation context.

        Returns:
            True if pronoun can be resolved.
        """
        if not context:
            return False

        # Check previous queries for referents
        previous_queries = context.get("previous_queries", [])
        if not previous_queries:
            return False

        # JPL Rule #2: Bounded check
        recent = previous_queries[:MAX_CONTEXT_QUERIES]

        # Simple heuristic: if recent queries mention specific entities,
        # pronouns might be resolvable
        for prev_query in recent:
            # Check for proper nouns (capitalized words)
            if re.search(r"\b[A-Z][a-z]+", str(prev_query)):
                return True

        return False

    def evaluate(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ClarificationArtifact:
        """
        Evaluate query clarity and return artifact.

        Enhanced evaluation with ambiguity detection.
        Rule #4: Under 60 lines.
        Rule #5: Validate inputs.

        Args:
            query: The user query to evaluate.
            context: Optional conversation context for resolution.

        Returns:
            ClarificationArtifact with score and suggestions.
        """
        assert query is not None, "query cannot be None"

        # Normalize query
        query = query.strip()[:MAX_QUERY_LENGTH]

        # Handle empty or too short queries
        if len(query) < MIN_QUERY_LENGTH:
            return self._create_low_clarity_artifact(
                query,
                0.0,
                "Query is too short",
                ["Please provide a more detailed question"],
            )

        # Run all ambiguity detectors
        pronouns = self.detect_pronouns(query, context)
        multi_meaning = self.detect_multi_meaning(query)
        temporal = self.detect_temporal_ambiguity(query)
        scope_ambiguous = self.detect_scope_ambiguity(query)

        # Build ambiguous terms list
        ambiguous_terms: List[Tuple[AmbiguityType, str]] = []
        ambiguous_terms.extend((AmbiguityType.PRONOUN, p) for p in pronouns)
        ambiguous_terms.extend(
            (AmbiguityType.MULTI_MEANING, t) for t, _ in multi_meaning
        )
        ambiguous_terms.extend((AmbiguityType.TEMPORAL, t) for t in temporal)
        if scope_ambiguous:
            ambiguous_terms.append((AmbiguityType.SCOPE, "query scope"))

        # Calculate clarity score
        clarity = self._calculate_clarity(query)

        # Adjust score based on detected ambiguities
        ambiguity_penalty = len(ambiguous_terms) * 0.1
        adjusted_score = max(0.0, clarity.score - ambiguity_penalty)

        # Generate questions for ambiguous terms
        questions = self.generate_questions(ambiguous_terms, query)

        # Create ambiguity report
        is_ambiguous = adjusted_score < self._config.threshold or len(questions) > 0
        report = AmbiguityReport(
            is_ambiguous=is_ambiguous,
            ambiguity_score=1.0 - adjusted_score,  # Invert for ambiguity score
            ambiguous_terms=ambiguous_terms[:MAX_AMBIGUOUS_TERMS],
            questions=questions,
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(query, clarity)

        return ClarificationArtifact(
            original_query=query,
            clarity_score=ClarityScore(
                score=adjusted_score,
                is_clear=not is_ambiguous,
                factors=clarity.factors,
            ),
            suggestions=suggestions[: self._config.max_suggestions],
            reason=self._get_clarity_reason(clarity)
            if not is_ambiguous
            else "Query has ambiguities that need clarification",
            needs_clarification=is_ambiguous,
            ambiguity_report=report,
        )

    def _calculate_clarity(self, query: str) -> ClarityScore:
        """
        Calculate clarity score for query.

        Rule #4: Under 60 lines.

        Args:
            query: The query to score.

        Returns:
            ClarityScore with factor breakdown.
        """
        query_lower = query.lower().strip()
        factors: Dict[str, float] = {}

        # Factor 1: Length score (longer = potentially more specific)
        length_score = min(len(query) / 50.0, 1.0)
        factors["length"] = length_score

        # Factor 2: Check for vague patterns
        vague_score = 1.0
        for pattern, penalty in VAGUE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                vague_score = min(vague_score, penalty)
        factors["vagueness"] = vague_score

        # Factor 3: Check for specific patterns
        specific_score = 0.5  # Default baseline
        for pattern, bonus in SPECIFIC_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                specific_score = max(specific_score, bonus)
        factors["specificity"] = specific_score

        # Factor 4: Word count (more words often = more context)
        word_count = len(query.split())
        word_score = min(word_count / 5.0, 1.0)
        factors["word_count"] = word_score

        # Factor 5: Contains question words with context
        question_score = 0.5
        if re.search(r"\b(what|who|when|where|why|how)\b.*\b\w{4,}\b", query_lower):
            question_score = 0.8
        factors["question_structure"] = question_score

        # Weighted combination
        final_score = (
            factors["length"] * 0.15
            + factors["vagueness"] * 0.30
            + factors["specificity"] * 0.30
            + factors["word_count"] * 0.10
            + factors["question_structure"] * 0.15
        )

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        return ClarityScore(
            score=final_score,
            is_clear=final_score >= self._config.threshold,
            factors=factors,
        )

    def _generate_suggestions(self, query: str, clarity: ClarityScore) -> List[str]:
        """
        Generate clarification suggestions.

        Rule #4: Under 60 lines.

        Args:
            query: The original query.
            clarity: The clarity score.

        Returns:
            List of suggested refinements.
        """
        suggestions: List[str] = []

        # Use LLM if available and configured
        if self._config.use_llm and self._llm_fn:
            llm_suggestions = self._get_llm_suggestions(query)
            if llm_suggestions:
                return llm_suggestions

        # Rule-based suggestions
        query_lower = query.lower().strip()

        # Too short
        if len(query) < 10:
            suggestions.append(
                "Can you provide more context about what you're looking for?"
            )
            suggestions.append("What specific aspect are you interested in?")
            suggestions.append("Could you rephrase with more details?")
            return suggestions

        # Generic "tell me more" type
        if re.search(r"^(tell me|more|explain|help)", query_lower):
            suggestions.append("What specific topic would you like to know more about?")
            suggestions.append(
                "Are you looking for a definition, process, or comparison?"
            )
            suggestions.append("Can you specify a particular aspect or context?")
            return suggestions

        # Missing specificity
        if clarity.factors.get("specificity", 0) < 0.6:
            suggestions.append(
                f"Could you specify which {self._extract_subject(query)} you mean?"
            )
            suggestions.append(
                "Are you asking about a specific time period or context?"
            )
            suggestions.append(
                "Would you like information from a particular source or domain?"
            )
            return suggestions

        # Default suggestions
        suggestions.append("Could you add more specific details to your question?")
        suggestions.append("What outcome are you hoping to achieve?")
        suggestions.append("Is there a specific context for this query?")

        return suggestions

    def _get_llm_suggestions(self, query: str) -> List[str]:
        """
        Get suggestions from LLM.

        Rule #2: All loops bounded by MAX constants.
        Rule #4: Under 60 lines.
        Rule #7: Check model response; fail-fast if invalid.

        Args:
            query: The query to clarify.

        Returns:
            List of LLM-generated suggestions, or empty if failed.
        """
        if not self._llm_fn:
            return []

        prompt = f"""Analyze this search query and suggest 3 specific clarifying questions:

Query: "{query}"

Return ONLY a JSON array of 3 strings, each a clarifying question.
Example: ["What time period?", "Which company?", "What aspect?"]"""

        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = self._llm_fn(prompt)

                # JPL Rule #7: Check model response
                if not response or not response.strip():
                    logger.warning(f"Empty LLM response on attempt {attempt + 1}")
                    continue

                # Parse JSON response
                suggestions = json.loads(response.strip())

                # Validate response structure
                if not isinstance(suggestions, list):
                    logger.warning(f"LLM response is not a list: {type(suggestions)}")
                    continue

                # JPL Rule #2: Bound suggestions list BEFORE validation loop
                bounded_suggestions = suggestions[:MAX_SUGGESTIONS]

                # Now safe to iterate over bounded list
                if not all(isinstance(s, str) for s in bounded_suggestions):
                    logger.warning("LLM response contains non-string elements")
                    continue

                return bounded_suggestions

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from LLM: {e}")
                continue
            except Exception as e:
                logger.error(f"LLM error: {e}")
                continue

        return []

    def _extract_subject(self, query: str) -> str:
        """
        Extract likely subject from query.

        Rule #2: Bounded loop over words.

        Args:
            query: The query to extract subject from.

        Returns:
            Extracted subject or "topic" if none found.
        """
        words = query.split()

        # JPL Rule #2: Bound words list BEFORE iteration (max 50 words)
        bounded_words = words[:50]

        # Find first noun-like word (simplified)
        stop_words = {"what", "when", "where", "which", "that", "this"}
        for word in bounded_words:
            if len(word) > 3 and word.lower() not in stop_words:
                return word.lower()

        return "topic"

    def _get_clarity_reason(self, clarity: ClarityScore) -> str:
        """Get human-readable reason for low clarity."""
        factors = clarity.factors

        if factors.get("vagueness", 1.0) < 0.3:
            return "Query appears too vague or general"
        if factors.get("length", 1.0) < 0.3:
            return "Query is too short to determine intent"
        if factors.get("specificity", 1.0) < 0.5:
            return "Query lacks specific details"
        if factors.get("word_count", 1.0) < 0.3:
            return "Query needs more context"

        return "Query could benefit from more specificity"

    def _create_low_clarity_artifact(
        self, query: str, score: float, reason: str, suggestions: List[str]
    ) -> ClarificationArtifact:
        """Create artifact for very low clarity queries."""
        return ClarificationArtifact(
            original_query=query or "",
            clarity_score=ClarityScore(score=score, is_clear=False, factors={}),
            suggestions=suggestions,
            reason=reason,
            needs_clarification=True,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_clarifier(
    threshold: float = CLARITY_THRESHOLD,
    use_llm: bool = False,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> IFQueryClarifier:
    """
    Create a query clarifier.

    Args:
        threshold: Clarity threshold (default 0.7).
        use_llm: Whether to use LLM for suggestions.
        llm_fn: LLM function if use_llm is True.

    Returns:
        Configured IFQueryClarifier.
    """
    config = ClarifierConfig(threshold=threshold, use_llm=use_llm)
    return IFQueryClarifier(config, llm_fn)


def evaluate_query_clarity(
    query: str, threshold: float = CLARITY_THRESHOLD
) -> ClarificationArtifact:
    """
    Convenience function to evaluate query clarity.

    Args:
        query: The query to evaluate.
        threshold: Clarity threshold.

    Returns:
        ClarificationArtifact with results.
    """
    clarifier = create_clarifier(threshold=threshold)
    return clarifier.evaluate(query)


def needs_clarification(query: str, threshold: float = CLARITY_THRESHOLD) -> bool:
    """
    Quick check if query needs clarification.

    Args:
        query: The query to check.
        threshold: Clarity threshold.

    Returns:
        True if clarification is recommended.
    """
    result = evaluate_query_clarity(query, threshold)
    return result.needs_clarification
