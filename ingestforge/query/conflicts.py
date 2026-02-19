"""
Cross-Source Conflict Detection for RAG Quality Assurance.

Detects contradictory or inconsistent information across multiple sources
to help users identify potential conflicts in their knowledge base.

Architecture Context
--------------------
Conflict detection runs after retrieval to identify potential issues:

    QueryPipeline.process(query)
            |
    HybridRetriever.search() -> sources
            |
    ConflictDetector.detect(sources) -> ConflictReport
            |
    Include warnings in QueryResult.metadata

Types of Conflicts Detected
---------------------------
1. Numeric Conflicts: Different values for same quantity
   - Example: "The budget is $10,000" vs "The budget is $15,000"

2. Date Conflicts: Different dates for same event
   - Example: "Issued in 2020" vs "Effective from 2021"

3. Categorical Conflicts: Different classifications
   - Example: "Required" vs "Optional"

4. Direct Contradictions: Opposite statements
   - Example: "X is allowed" vs "X is prohibited"

Severity Levels
---------------
- HIGH: Direct contradictions that likely indicate data quality issues
- MEDIUM: Numeric/date discrepancies that may be version differences
- LOW: Minor wording differences that are likely contextual

Reference Implementation
------------------------
Based on Army Doctrine RAG conflict detection patterns that flag
contradictory regulations across different source documents.
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ingestforge.core.logging import get_logger
from ingestforge.storage.base import SearchResult

logger = get_logger(__name__)


@dataclass
class Conflict:
    """A detected conflict between sources.

    Attributes:
        conflict_type: Type of conflict (numeric, date, categorical, contradiction)
        severity: HIGH, MEDIUM, or LOW
        claim_a: First conflicting claim
        claim_b: Second conflicting claim
        source_a: Source of first claim (source index)
        source_b: Source of second claim (source index)
        value_a: Extracted value from first claim (if applicable)
        value_b: Extracted value from second claim (if applicable)
        context: Additional context about the conflict
    """

    conflict_type: str
    severity: str
    claim_a: str
    claim_b: str
    source_a: int
    source_b: int
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    context: Optional[str] = None


@dataclass
class ConflictReport:
    """Report of detected conflicts across sources.

    Attributes:
        has_conflicts: Whether any conflicts were detected
        conflicts: List of detected Conflict objects
        high_severity_count: Number of HIGH severity conflicts
        sources_analyzed: Number of sources analyzed
        summary: Human-readable summary of findings
    """

    has_conflicts: bool
    conflicts: List[Conflict] = field(default_factory=list)
    high_severity_count: int = 0
    sources_analyzed: int = 0
    summary: str = ""

    def __post_init__(self) -> None:
        """Calculate summary statistics."""
        self.high_severity_count = sum(
            1 for c in self.conflicts if c.severity == "HIGH"
        )


class ConflictDetector:
    """
    Detects contradictory information across multiple sources.

    Uses pattern matching and heuristics to identify:
    - Numeric discrepancies
    - Date conflicts
    - Categorical contradictions
    - Direct negations
    """

    def __init__(
        self,
        numeric_tolerance: float = 0.1,
        check_numeric: bool = True,
        check_dates: bool = True,
        check_categorical: bool = True,
        check_negations: bool = True,
    ):
        """
        Initialize conflict detector.

        Args:
            numeric_tolerance: Relative tolerance for numeric comparison (0.1 = 10%)
            check_numeric: Enable numeric conflict detection
            check_dates: Enable date conflict detection
            check_categorical: Enable categorical conflict detection
            check_negations: Enable negation/contradiction detection
        """
        self.numeric_tolerance = numeric_tolerance
        self.check_numeric = check_numeric
        self.check_dates = check_dates
        self.check_categorical = check_categorical
        self.check_negations = check_negations

        # Categorical opposition pairs
        self.opposition_pairs = [
            ("required", "optional"),
            ("mandatory", "voluntary"),
            ("allowed", "prohibited"),
            ("permitted", "forbidden"),
            ("approved", "denied"),
            ("enabled", "disabled"),
            ("active", "inactive"),
            ("valid", "invalid"),
            ("yes", "no"),
            ("true", "false"),
            ("include", "exclude"),
            ("must", "must not"),
            ("shall", "shall not"),
            ("will", "will not"),
        ]

    def detect(
        self,
        sources: List[SearchResult],
        query: Optional[str] = None,
    ) -> ConflictReport:
        """
        Detect conflicts across multiple sources.

        Args:
            sources: List of SearchResult objects to analyze
            query: Optional query context to focus analysis

        Returns:
            ConflictReport with detected conflicts
        """
        if len(sources) < 2:
            return ConflictReport(
                has_conflicts=False,
                sources_analyzed=len(sources),
                summary="Insufficient sources for conflict detection (need >= 2)",
            )

        conflicts: List[Conflict] = []

        # Compare each pair of sources
        for i, source_a in enumerate(sources):
            for j, source_b in enumerate(sources[i + 1 :], i + 1):
                pair_conflicts = self._compare_sources(
                    source_a,
                    source_b,
                    i + 1,
                    j + 1,  # 1-indexed for citation references
                )
                conflicts.extend(pair_conflicts)

        # Build summary
        if conflicts:
            high_count = sum(1 for c in conflicts if c.severity == "HIGH")
            medium_count = sum(1 for c in conflicts if c.severity == "MEDIUM")
            low_count = sum(1 for c in conflicts if c.severity == "LOW")

            summary_parts = [f"Found {len(conflicts)} potential conflict(s)"]
            if high_count:
                summary_parts.append(f"{high_count} HIGH severity")
            if medium_count:
                summary_parts.append(f"{medium_count} MEDIUM severity")
            if low_count:
                summary_parts.append(f"{low_count} LOW severity")
            summary = ": ".join(summary_parts)
        else:
            summary = "No conflicts detected across sources"

        return ConflictReport(
            has_conflicts=len(conflicts) > 0,
            conflicts=conflicts,
            sources_analyzed=len(sources),
            summary=summary,
        )

    def _compare_sources(
        self,
        source_a: SearchResult,
        source_b: SearchResult,
        idx_a: int,
        idx_b: int,
    ) -> List[Conflict]:
        """Compare two sources for conflicts."""
        conflicts = []

        content_a = source_a.content
        content_b = source_b.content

        if self.check_numeric:
            conflicts.extend(
                self._detect_numeric_conflicts(content_a, content_b, idx_a, idx_b)
            )

        if self.check_dates:
            conflicts.extend(
                self._detect_date_conflicts(content_a, content_b, idx_a, idx_b)
            )

        if self.check_categorical:
            conflicts.extend(
                self._detect_categorical_conflicts(content_a, content_b, idx_a, idx_b)
            )

        if self.check_negations:
            conflicts.extend(
                self._detect_negation_conflicts(content_a, content_b, idx_a, idx_b)
            )

        return conflicts

    def _try_parse_float(self, value: str) -> Optional[float]:
        """
        Parse numeric value as float.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            value: String value to parse (may contain commas)

        Returns:
            Parsed float or None if parsing fails
        """
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None

    def _calculate_difference_ratio(self, v_a: float, v_b: float) -> Optional[float]:
        """
        Calculate relative difference between two values.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            v_a: First numeric value
            v_b: Second numeric value

        Returns:
            Relative difference ratio or None if division by zero
        """
        max_val = max(abs(v_a), abs(v_b))
        if max_val == 0:
            return None

        return abs(v_a - v_b) / max_val

    def _create_numeric_conflict(
        self,
        value_a: str,
        value_b: str,
        context_a: str,
        context_b: str,
        idx_a: int,
        idx_b: int,
        diff_ratio: float,
    ) -> Conflict:
        """
        Create numeric conflict object.

        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            value_a: First numeric value string
            value_b: Second numeric value string
            context_a: Context of first value
            context_b: Context of second value
            idx_a: Source index for first value
            idx_b: Source index for second value
            diff_ratio: Calculated difference ratio

        Returns:
            Conflict object
        """
        return Conflict(
            conflict_type="numeric",
            severity="MEDIUM" if diff_ratio < 0.5 else "HIGH",
            claim_a=context_a,
            claim_b=context_b,
            source_a=idx_a,
            source_b=idx_b,
            value_a=value_a,
            value_b=value_b,
            context=f"Values differ by {diff_ratio*100:.1f}%",
        )

    def _check_numeric_pair_conflict(
        self,
        value_a: str,
        value_b: str,
        context_a: str,
        context_b: str,
        idx_a: int,
        idx_b: int,
    ) -> Optional[Conflict]:
        """
        Check if two numeric values create a conflict.

        Rule #1: Early returns eliminate nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            value_a: First numeric value string
            value_b: Second numeric value string
            context_a: Context of first value
            context_b: Context of second value
            idx_a: Source index for first value
            idx_b: Source index for second value

        Returns:
            Conflict if found, None otherwise
        """
        if value_a == value_b:
            return None
        context_similarity = self._context_similarity(context_a, context_b)
        if context_similarity <= 0.5:
            return None
        v_a = self._try_parse_float(value_a)
        v_b = self._try_parse_float(value_b)
        if v_a is None or v_b is None:
            return None
        diff_ratio = self._calculate_difference_ratio(v_a, v_b)
        if diff_ratio is None:
            return None
        if diff_ratio <= self.numeric_tolerance:
            return None

        return self._create_numeric_conflict(
            value_a, value_b, context_a, context_b, idx_a, idx_b, diff_ratio
        )

    def _detect_numeric_conflicts(
        self,
        content_a: str,
        content_b: str,
        idx_a: int,
        idx_b: int,
    ) -> List[Conflict]:
        """
        Detect conflicts in numeric values.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            content_a: Content from first source
            content_b: Content from second source
            idx_a: Source index for first source
            idx_b: Source index for second source

        Returns:
            List of detected numeric conflicts
        """
        conflicts = []

        nums_a = self._extract_numeric_claims(content_a)
        nums_b = self._extract_numeric_claims(content_b)
        for value_a, context_a in nums_a:
            for value_b, context_b in nums_b:
                conflict = self._check_numeric_pair_conflict(
                    value_a, value_b, context_a, context_b, idx_a, idx_b
                )
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def _extract_numeric_claims(self, content: str) -> List[Tuple[str, str]]:
        """Extract numeric values with surrounding context."""
        claims = []
        pattern = r"(\d+(?:,\d{3})*(?:\.\d+)?)"

        for match in re.finditer(pattern, content):
            value = match.group(1)
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].strip()
            claims.append((value, context))

        return claims

    def _detect_date_conflicts(
        self,
        content_a: str,
        content_b: str,
        idx_a: int,
        idx_b: int,
    ) -> List[Conflict]:
        """Detect conflicts in dates."""
        conflicts = []

        # Extract dates
        dates_a = self._extract_dates(content_a)
        dates_b = self._extract_dates(content_b)

        # Look for same event keywords with different dates
        event_keywords = [
            "effective",
            "issued",
            "published",
            "updated",
            "revised",
            "deadline",
            "due",
            "expires",
            "valid",
            "started",
            "ended",
        ]

        for date_a, context_a in dates_a:
            for date_b, context_b in dates_b:
                if date_a == date_b:
                    continue

                # Check if both contexts mention similar events
                ctx_lower_a = context_a.lower()
                ctx_lower_b = context_b.lower()

                shared_events = [
                    kw
                    for kw in event_keywords
                    if kw in ctx_lower_a and kw in ctx_lower_b
                ]

                if shared_events:
                    conflicts.append(
                        Conflict(
                            conflict_type="date",
                            severity="MEDIUM",
                            claim_a=context_a,
                            claim_b=context_b,
                            source_a=idx_a,
                            source_b=idx_b,
                            value_a=date_a,
                            value_b=date_b,
                            context=f"Different dates for: {', '.join(shared_events)}",
                        )
                    )

        return conflicts

    def _extract_dates(self, content: str) -> List[Tuple[str, str]]:
        """Extract dates with surrounding context."""
        dates = []

        # Various date patterns
        patterns = [
            r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",  # 01/15/2024
            r"\b(\d{4}-\d{2}-\d{2})\b",  # 2024-01-15
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
            r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                date_str = match.group(1)
                start = max(0, match.start() - 40)
                end = min(len(content), match.end() + 40)
                context = content[start:end].strip()
                dates.append((date_str, context))

        return dates

    def _detect_categorical_conflicts(
        self,
        content_a: str,
        content_b: str,
        idx_a: int,
        idx_b: int,
    ) -> List[Conflict]:
        """Detect conflicts in categorical statements."""
        conflicts = []
        content_lower_a = content_a.lower()
        content_lower_b = content_b.lower()

        for term_a, term_b in self.opposition_pairs:
            # Check if one source uses term_a and other uses term_b
            a_has_term_a = term_a in content_lower_a
            a_has_term_b = term_b in content_lower_a
            b_has_term_a = term_a in content_lower_b
            b_has_term_b = term_b in content_lower_b

            # Conflict: source A says term_a, source B says term_b (or vice versa)
            if (
                a_has_term_a and not a_has_term_b and b_has_term_b and not b_has_term_a
            ) or (
                a_has_term_b and not a_has_term_a and b_has_term_a and not b_has_term_b
            ):
                # Extract context around the terms
                context_a = self._extract_term_context(
                    content_a, term_a if a_has_term_a else term_b
                )
                context_b = self._extract_term_context(
                    content_b, term_b if b_has_term_b else term_a
                )

                conflicts.append(
                    Conflict(
                        conflict_type="categorical",
                        severity="HIGH",
                        claim_a=context_a,
                        claim_b=context_b,
                        source_a=idx_a,
                        source_b=idx_b,
                        value_a=term_a if a_has_term_a else term_b,
                        value_b=term_b if b_has_term_b else term_a,
                        context=f"Opposing terms: '{term_a}' vs '{term_b}'",
                    )
                )

        return conflicts

    def _check_negation_match_pair(
        self,
        pos_match: Any,
        neg_match: Any,
        content_a: str,
        content_b: str,
        idx_a: int,
        idx_b: int,
    ) -> Optional[Conflict]:
        """
        Check if positive and negative matches form a conflict.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            pos_match: Positive pattern match
            neg_match: Negative pattern match
            content_a: First content string
            content_b: Second content string
            idx_a: First source index
            idx_b: Second source index

        Returns:
            Conflict if matched, None otherwise
        """
        if len(pos_match.groups()) < 2 or len(neg_match.groups()) < 2:
            return None

        # Extract words to compare
        pos_word = pos_match.group(2)
        neg_word = neg_match.group(2) if neg_match.lastindex >= 2 else ""
        if pos_word != neg_word:
            return None

        # Create conflict
        return Conflict(
            conflict_type="contradiction",
            severity="HIGH",
            claim_a=self._extract_sentence(content_a, pos_match.start()),
            claim_b=self._extract_sentence(content_b, neg_match.start()),
            source_a=idx_a,
            source_b=idx_b,
            context="Direct negation detected",
        )

    def _check_negation_pattern(
        self,
        pos_pattern: str,
        neg_pattern: str,
        content_a: str,
        content_b: str,
        content_lower_a: str,
        content_lower_b: str,
        idx_a: int,
        idx_b: int,
    ) -> List[Conflict]:
        """
        Check single negation pattern pair.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            pos_pattern: Positive regex pattern
            neg_pattern: Negative regex pattern
            content_a: First content string
            content_b: Second content string
            content_lower_a: Lowercase first content
            content_lower_b: Lowercase second content
            idx_a: First source index
            idx_b: Second source index

        Returns:
            List of conflicts found
        """
        conflicts = []

        pos_matches_a = list(re.finditer(pos_pattern, content_lower_a))
        neg_matches_b = list(re.finditer(neg_pattern, content_lower_b))

        for pos_match in pos_matches_a:
            for neg_match in neg_matches_b:
                conflict = self._check_negation_match_pair(
                    pos_match, neg_match, content_a, content_b, idx_a, idx_b
                )
                if conflict is None:
                    continue

                conflicts.append(conflict)

        return conflicts

    def _detect_negation_conflicts(
        self,
        content_a: str,
        content_b: str,
        idx_a: int,
        idx_b: int,
    ) -> List[Conflict]:
        """
        Detect direct negation/contradiction patterns.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            content_a: First content string
            content_b: Second content string
            idx_a: First source index
            idx_b: Second source index

        Returns:
            List of conflicts detected
        """
        conflicts = []

        # Patterns for negation
        negation_patterns = [
            (r"\b(is|are|was|were)\s+(\w+)", r"\b(is|are|was|were)\s+not\s+\2"),
            (
                r"\b(can|could|may|might|will|would|should)\s+(\w+)",
                r"\b(can|could|may|might|will|would|should)\s+not\s+\2",
            ),
        ]

        content_lower_a = content_a.lower()
        content_lower_b = content_b.lower()

        # Check each negation pattern
        for pos_pattern, neg_pattern in negation_patterns:
            pattern_conflicts = self._check_negation_pattern(
                pos_pattern,
                neg_pattern,
                content_a,
                content_b,
                content_lower_a,
                content_lower_b,
                idx_a,
                idx_b,
            )
            conflicts.extend(pattern_conflicts)

        return conflicts

    def _context_similarity(self, context_a: str, context_b: str) -> float:
        """Calculate similarity between two contexts based on word overlap."""
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
            "and",
            "or",
            "but",
            "if",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "their",
        }

        words_a = set(re.findall(r"\b\w+\b", context_a.lower())) - stopwords
        words_b = set(re.findall(r"\b\w+\b", context_b.lower())) - stopwords

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    def _extract_term_context(self, content: str, term: str) -> str:
        """Extract context around a term."""
        idx = content.lower().find(term.lower())
        if idx == -1:
            return ""
        start = max(0, idx - 50)
        end = min(len(content), idx + len(term) + 50)
        return content[start:end].strip()

    def _extract_sentence(self, content: str, position: int) -> str:
        """Extract the sentence containing a position."""
        # Find sentence boundaries
        start = content.rfind(".", 0, position) + 1
        end = content.find(".", position)
        if end == -1:
            end = len(content)
        return content[start : end + 1].strip()


def detect_conflicts(
    sources: List[SearchResult],
    query: Optional[str] = None,
) -> ConflictReport:
    """
    Convenience function to detect conflicts across sources.

    Args:
        sources: List of SearchResult objects to analyze
        query: Optional query context

    Returns:
        ConflictReport with detected conflicts
    """
    detector = ConflictDetector()
    return detector.detect(sources, query)
