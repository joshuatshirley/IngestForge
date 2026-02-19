"""Glossary parsing utilities.

Provides robust text parsing, term deduplication, validation,
and category inference for glossary generation.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedTerm:
    """Represents a parsed glossary term.

    Attributes:
        term: The term name
        definition: The term definition
        category: Category classification (defaults to "general")
        related_terms: List of related term names
        source_line: Original line number for debugging
        confidence: Parsing confidence score (0-1)
    """

    term: str
    definition: str
    category: str = "general"
    related_terms: List[str] = field(default_factory=list)
    source_line: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON output."""
        return {
            "term": self.term,
            "definition": self.definition,
            "category": self.category,
            "related_terms": self.related_terms,
        }


class GlossaryTextParser:
    """Parse glossary terms from various text formats.

    Supports formats:
    - term: definition (basic colon separator)
    - **term**: definition (markdown bold)
    - 1. term: definition (numbered list)
    - * term - definition (bullet list)
    - term (category): definition (with category hint)
    """

    # Regex patterns for different formats (order matters - more specific first)
    PATTERNS = [
        # **term**: definition (markdown bold)
        (
            r"^\*\*([^*]+)\*\*\s*[:\-]\s*(.+)$",
            "markdown_bold",
            0.95,
        ),
        # 1. term: definition (numbered list)
        (
            r"^\d+\.\s+([^:\-]+?)\s*[:\-]\s*(.+)$",
            "numbered_list",
            0.9,
        ),
        # * term - definition or - term: definition (bullet list)
        (
            r"^[\*\-\u2022]\s+([^:\-]+?)\s*[:\-]\s*(.+)$",
            "bullet_list",
            0.9,
        ),
        # term (category): definition
        (
            r"^([^(:\-]+?)\s*\([^)]+\)\s*[:\-]\s*(.+)$",
            "with_category",
            0.85,
        ),
        # term: definition (basic - must be last)
        (
            r"^([^:\-]+?)\s*[:\-]\s*(.+)$",
            "basic",
            0.8,
        ),
    ]

    # Category extraction pattern
    CATEGORY_PATTERN = re.compile(r"\(([^)]+)\)")

    def __init__(
        self,
        min_term_length: int = 2,
        min_definition_length: int = 10,
    ):
        """Initialize parser.

        Args:
            min_term_length: Minimum characters for a valid term
            min_definition_length: Minimum characters for a valid definition
        """
        # Commandment #5: Assertion density - validate preconditions
        assert (
            min_term_length > 0
        ), f"min_term_length must be positive, got {min_term_length}"
        assert (
            min_definition_length > 0
        ), f"min_definition_length must be positive, got {min_definition_length}"

        self.min_term_length = min_term_length
        self.min_definition_length = min_definition_length
        self._compiled_patterns = [
            (re.compile(pattern), name, conf) for pattern, name, conf in self.PATTERNS
        ]

    def parse(self, text: str) -> List[ParsedTerm]:
        """Parse text into glossary terms.

        Args:
            text: Raw text containing glossary entries

        Returns:
            List of parsed terms
        """
        if not text or not text.strip():
            return []

        terms: List[ParsedTerm] = []
        lines = text.split("\n")

        for line_num, line in enumerate(lines, 1):
            term = self._parse_line(line, line_num)
            if term:
                terms.append(term)

        logger.debug(
            f"Parsed {len(terms)} terms from {len(lines)} lines",
            terms_found=len(terms),
            lines_processed=len(lines),
        )

        return terms

    def _parse_line(self, line: str, line_num: int) -> Optional[ParsedTerm]:
        """Parse a single line for term and definition.

        Args:
            line: Line to parse
            line_num: Line number for debugging

        Returns:
            ParsedTerm or None if not parseable
        """
        line = line.strip()
        if not line:
            return None

        # Try each pattern
        for pattern, pattern_name, confidence in self._compiled_patterns:
            match = pattern.match(line)
            if match:
                term = self._clean_term(match.group(1))
                definition = self._clean_definition(match.group(2))

                # Validate lengths
                if not self._validate_lengths(term, definition):
                    continue

                # Extract category if present
                category = self._extract_category(line)

                logger.debug(
                    f"Parsed term using {pattern_name} pattern",
                    term=term,
                    line=line_num,
                )

                return ParsedTerm(
                    term=term,
                    definition=definition,
                    category=category,
                    source_line=line_num,
                    confidence=confidence,
                )

        return None

    def _clean_term(self, term: str) -> str:
        """Clean and normalize a term.

        Args:
            term: Raw term string

        Returns:
            Cleaned term
        """
        # Remove markdown formatting
        term = re.sub(r"[\*_`]", "", term)
        # Remove leading/trailing punctuation and whitespace
        term = term.strip(" \t\n#*-")
        # Normalize whitespace
        term = " ".join(term.split())
        return term

    def _clean_definition(self, definition: str) -> str:
        """Clean and normalize a definition.

        Args:
            definition: Raw definition string

        Returns:
            Cleaned definition
        """
        # Strip whitespace
        definition = definition.strip()
        # Remove trailing punctuation duplicates
        definition = re.sub(r"\.{2,}$", ".", definition)
        # Ensure ends with period
        if definition and definition[-1] not in ".!?":
            definition += "."
        return definition

    def _validate_lengths(self, term: str, definition: str) -> bool:
        """Validate term and definition lengths.

        Args:
            term: Term string
            definition: Definition string

        Returns:
            True if valid
        """
        if len(term) < self.min_term_length:
            return False
        if len(definition) < self.min_definition_length:
            return False
        # Term must contain at least one letter
        if not any(c.isalpha() for c in term):
            return False
        return True

    def _extract_category(self, line: str) -> str:
        """Extract category hint from line.

        Args:
            line: Full line text

        Returns:
            Category string or "general"
        """
        match = self.CATEGORY_PATTERN.search(line)
        if match:
            category = match.group(1).strip().lower()
            if len(category) > 2 and len(category) < 50:
                return category
        return "general"


class TermDeduplicator:
    """Remove duplicate or similar terms using Jaccard similarity.

    Uses trigram-based similarity for fuzzy matching.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        """Initialize deduplicator.

        Args:
            similarity_threshold: Minimum similarity to consider duplicate (0-1)
        """
        # Commandment #5: Assertion density - validate preconditions
        assert (
            0.0 <= similarity_threshold <= 1.0
        ), f"similarity_threshold must be 0-1, got {similarity_threshold}"

        self.similarity_threshold = similarity_threshold

    def deduplicate(self, terms: List[ParsedTerm]) -> List[ParsedTerm]:
        """Remove duplicate/similar terms.

        Args:
            terms: List of parsed terms

        Returns:
            Deduplicated list
        """
        # Commandment #7: Check parameters - validate inputs
        if not terms:
            return []

        if len(terms) <= 1:
            return terms

        # Build trigram sets for each term
        term_trigrams: Dict[int, Set[str]] = {}
        for i, t in enumerate(terms):
            term_trigrams[i] = self._compute_trigrams(t.term)

        # Find similar groups
        groups = self._find_similar_groups(terms, term_trigrams)

        # Select best from each group (Commandment #1: simplified control flow)
        result: List[ParsedTerm] = []
        processed: Set[int] = set()

        for group in groups:
            best_term = self._process_group(terms, group)
            result.append(best_term)
            processed.update(group)

        # Add any ungrouped terms
        for i, t in enumerate(terms):
            if i not in processed:
                result.append(t)

        self._log_deduplication_result(len(terms), len(result))

        return result

    def _process_group(self, terms: List[ParsedTerm], group: List[int]) -> ParsedTerm:
        """Select best term from a duplicate group.

        Args:
            terms: Full term list
            group: Indices of similar terms

        Returns:
            Best term from the group
        """
        if len(group) == 1:
            return terms[group[0]]

        best_idx = self._select_best(terms, group)
        self._log_merge(terms, group, best_idx)
        return terms[best_idx]

    def _log_merge(
        self, terms: List[ParsedTerm], group: List[int], best_idx: int
    ) -> None:
        """Log merge operation for debugging.

        Args:
            terms: Full term list
            group: Indices of merged terms
            best_idx: Index of kept term
        """
        if len(group) > 1:
            logger.debug(
                f"Merged {len(group)} similar terms",
                kept=terms[best_idx].term,
                merged=[terms[i].term for i in group if i != best_idx],
            )

    def _log_deduplication_result(self, original: int, final: int) -> None:
        """Log deduplication results.

        Args:
            original: Original term count
            final: Final term count
        """
        removed = original - final
        if removed > 0:
            logger.info(
                f"Deduplication removed {removed} terms",
                original=original,
                final=final,
            )

    def _compute_trigrams(self, text: str) -> Set[str]:
        """Compute character trigrams for similarity.

        Args:
            text: Input text

        Returns:
            Set of trigrams
        """
        # Normalize
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = text.strip()

        if len(text) < 3:
            return {text}

        trigrams = set()
        for i in range(len(text) - 2):
            trigrams.add(text[i : i + 3])

        return trigrams

    def _compute_jaccard(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Similarity score (0-1)
        """
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _find_similar_groups(
        self,
        terms: List[ParsedTerm],
        trigrams: Dict[int, Set[str]],
    ) -> List[List[int]]:
        """Find groups of similar terms.

        Args:
            terms: List of terms
            trigrams: Precomputed trigram sets

        Returns:
            List of groups (each group is list of indices)
        """
        n = len(terms)
        groups: List[List[int]] = []
        processed: Set[int] = set()

        for i in range(n):
            if i in processed:
                continue

            # Commandment #1: Extract inner loop to reduce nesting
            group = self._find_similar_to(i, n, trigrams, processed)
            groups.append(group)
            processed.update(group)

        return groups

    def _find_similar_to(
        self,
        idx: int,
        n: int,
        trigrams: Dict[int, Set[str]],
        processed: Set[int],
    ) -> List[int]:
        """Find all terms similar to term at idx.

        Args:
            idx: Index of reference term
            n: Total number of terms
            trigrams: Precomputed trigram sets
            processed: Already processed indices

        Returns:
            List of similar term indices (including idx)
        """
        group = [idx]

        for j in range(idx + 1, n):
            if j in processed:
                continue

            similarity = self._compute_jaccard(trigrams[idx], trigrams[j])
            if similarity >= self.similarity_threshold:
                group.append(j)

        return group

    def _select_best(
        self,
        terms: List[ParsedTerm],
        indices: List[int],
    ) -> int:
        """Select best term from a group of similar terms.

        Prefers:
        1. Higher confidence score
        2. Longer definition
        3. Proper case (not all lowercase/uppercase)

        Args:
            terms: Full term list
            indices: Indices of similar terms

        Returns:
            Index of best term
        """

        def score(idx: int) -> Tuple[float, int, int]:
            t = terms[idx]
            # Proper case bonus
            case_score = 1 if (not t.term.isupper() and not t.term.islower()) else 0
            return (t.confidence, len(t.definition), case_score)

        return max(indices, key=score)


class GlossaryValidator:
    """Validate and clean glossary terms."""

    def __init__(
        self,
        min_term_chars: int = 2,
        max_term_chars: int = 100,
        min_definition_words: int = 3,
        max_definition_words: int = 200,
        max_related_terms: int = 5,
    ):
        """Initialize validator.

        Args:
            min_term_chars: Minimum characters in term
            max_term_chars: Maximum characters in term
            min_definition_words: Minimum words in definition
            max_definition_words: Maximum words in definition
            max_related_terms: Maximum related terms to keep
        """
        # Commandment #5: Assertion density - validate preconditions
        assert 0 < min_term_chars <= max_term_chars, (
            f"min_term_chars must be positive and <= max_term_chars, "
            f"got min={min_term_chars}, max={max_term_chars}"
        )
        assert 0 < min_definition_words <= max_definition_words, (
            f"min_definition_words must be positive and <= max_definition_words, "
            f"got min={min_definition_words}, max={max_definition_words}"
        )
        assert (
            max_related_terms > 0
        ), f"max_related_terms must be positive, got {max_related_terms}"

        self.min_term_chars = min_term_chars
        self.max_term_chars = max_term_chars
        self.min_definition_words = min_definition_words
        self.max_definition_words = max_definition_words
        self.max_related_terms = max_related_terms

    def validate_and_clean(
        self,
        terms: List[ParsedTerm],
    ) -> Tuple[List[ParsedTerm], int]:
        """Validate and clean terms.

        Args:
            terms: List of parsed terms

        Returns:
            Tuple of (valid terms, count of invalid removed)
        """
        valid: List[ParsedTerm] = []
        invalid_count = 0

        for term in terms:
            cleaned = self._validate_term(term)
            if cleaned:
                valid.append(cleaned)
            else:
                invalid_count += 1
                logger.debug(
                    "Term validation failed",
                    term=term.term,
                    reason="failed validation",
                )

        if invalid_count > 0:
            logger.info(
                f"Validation removed {invalid_count} invalid terms",
                valid=len(valid),
                invalid=invalid_count,
            )

        return valid, invalid_count

    def _validate_term(self, term: ParsedTerm) -> Optional[ParsedTerm]:
        """Validate and potentially clean a single term.

        Args:
            term: Term to validate

        Returns:
            Cleaned term or None if invalid
        """
        # Check term length
        if len(term.term) < self.min_term_chars:
            return None
        if len(term.term) > self.max_term_chars:
            return None

        # Term must contain letters
        if not any(c.isalpha() for c in term.term):
            return None

        # Clean and validate definition
        definition = self._clean_definition(term.definition)
        if not definition:
            return None

        # Truncate related terms if needed
        related = term.related_terms[: self.max_related_terms]

        # Default category if empty
        category = term.category.strip() if term.category else "general"
        if not category:
            category = "general"

        return ParsedTerm(
            term=term.term,
            definition=definition,
            category=category,
            related_terms=related,
            source_line=term.source_line,
            confidence=term.confidence,
        )

    def _clean_definition(self, definition: str) -> Optional[str]:
        """Clean and validate definition.

        Args:
            definition: Raw definition

        Returns:
            Cleaned definition or None if invalid
        """
        if not definition:
            return None

        words = definition.split()
        word_count = len(words)

        # Too short
        if word_count < self.min_definition_words:
            return None

        # Truncate if too long
        if word_count > self.max_definition_words:
            words = words[: self.max_definition_words]
            definition = " ".join(words)
            # Ensure proper ending
            if not definition.endswith("."):
                definition = definition.rstrip(".,;:!?") + "..."

        return definition


class CategoryInferrer:
    """Infer categories for terms based on keywords."""

    # Category keyword mappings
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "computer science": [
            "algorithm",
            "data",
            "function",
            "variable",
            "class",
            "object",
            "method",
            "array",
            "loop",
            "database",
            "server",
            "client",
            "api",
            "code",
            "program",
            "software",
            "binary",
            "cache",
            "compile",
            "debug",
            "execute",
            "hash",
            "iterate",
            "memory",
            "network",
            "protocol",
            "query",
            "recursive",
            "stack",
            "thread",
        ],
        "mathematics": [
            "equation",
            "theorem",
            "calculate",
            "proof",
            "formula",
            "derivative",
            "integral",
            "matrix",
            "vector",
            "polynomial",
            "coefficient",
            "factor",
            "function",
            "graph",
            "limit",
            "logarithm",
            "ratio",
            "sequence",
            "series",
            "sum",
            "variable",
            "axiom",
            "conjecture",
            "lemma",
            "postulate",
        ],
        "science": [
            "experiment",
            "hypothesis",
            "molecule",
            "atom",
            "cell",
            "organism",
            "evolution",
            "gene",
            "energy",
            "force",
            "mass",
            "velocity",
            "chemical",
            "reaction",
            "biology",
            "physics",
            "chemistry",
            "element",
            "compound",
            "particle",
            "wave",
            "electron",
            "proton",
            "neutron",
            "nucleus",
        ],
        "business": [
            "revenue",
            "profit",
            "loss",
            "market",
            "strategy",
            "customer",
            "sales",
            "budget",
            "investment",
            "asset",
            "liability",
            "equity",
            "capital",
            "dividend",
            "share",
            "stock",
            "bond",
            "merger",
            "acquisition",
            "valuation",
            "roi",
            "kpi",
        ],
        "language": [
            "noun",
            "verb",
            "adjective",
            "adverb",
            "pronoun",
            "syntax",
            "grammar",
            "sentence",
            "clause",
            "phrase",
            "tense",
            "plural",
            "singular",
            "conjugate",
            "declension",
            "etymology",
            "morpheme",
            "phoneme",
            "semantics",
            "pragmatics",
            "metaphor",
            "simile",
        ],
        "medicine": [
            "diagnosis",
            "symptom",
            "treatment",
            "patient",
            "disease",
            "medication",
            "surgery",
            "therapy",
            "chronic",
            "acute",
            "infection",
            "virus",
            "bacteria",
            "immune",
            "vaccine",
            "prescription",
            "dosage",
            "anatomy",
            "physiology",
        ],
        "law": [
            "statute",
            "regulation",
            "contract",
            "liability",
            "plaintiff",
            "defendant",
            "judge",
            "court",
            "verdict",
            "appeal",
            "tort",
            "precedent",
            "jurisdiction",
            "litigation",
            "arbitration",
        ],
    }

    def __init__(self):
        """Initialize category inferrer."""
        # Pre-process keywords to lowercase sets
        self._keyword_sets: Dict[str, Set[str]] = {
            cat: set(kw.lower() for kw in keywords)
            for cat, keywords in self.CATEGORY_KEYWORDS.items()
        }

    def infer_category(self, term: ParsedTerm) -> str:
        """Infer category for a term.

        Args:
            term: Term to categorize

        Returns:
            Inferred category or "general"
        """
        # If already has non-general category, keep it
        if term.category and term.category != "general":
            return term.category

        # Combine term and definition for keyword matching
        text = f"{term.term} {term.definition}".lower()
        words = set(re.findall(r"\b\w+\b", text))

        # Score each category
        scores: Dict[str, int] = {}
        for category, keywords in self._keyword_sets.items():
            score = len(words & keywords)
            if score > 0:
                scores[category] = score

        # Return best match or general
        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            if best[1] >= 2:  # Require at least 2 keyword matches
                logger.debug(
                    "Inferred category for term",
                    term=term.term,
                    category=best[0],
                    score=best[1],
                )
                return best[0]

        return "general"

    def infer_categories(self, terms: List[ParsedTerm]) -> List[ParsedTerm]:
        """Infer categories for all terms.

        Args:
            terms: List of terms

        Returns:
            Terms with inferred categories
        """
        result = []
        inferred_count = 0

        for term in terms:
            new_category = self.infer_category(term)
            if new_category != term.category:
                inferred_count += 1
                term = ParsedTerm(
                    term=term.term,
                    definition=term.definition,
                    category=new_category,
                    related_terms=term.related_terms,
                    source_line=term.source_line,
                    confidence=term.confidence,
                )
            result.append(term)

        if inferred_count > 0:
            logger.info(
                f"Inferred categories for {inferred_count} terms",
                total=len(terms),
                inferred=inferred_count,
            )

        return result
