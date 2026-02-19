"""NER-targeted cloze deletion generator (NLP-001.1).

This module provides intelligent cloze deletion (fill-in-the-blank)
generation by targeting named entities and key concepts using spaCy NER.

Cloze deletions are a powerful study technique where key terms are
replaced with blanks, forcing active recall rather than passive reading.

Supports Anki cloze format: {{c1::answer}}, {{c2::answer}}, etc.

NASA JPL Commandments compliance:
- Rule #1: No deep nesting, early returns
- Rule #2: Fixed iteration bounds
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.enrichment.cloze_ner import ClozeGenerator

    # Generate individual cloze cards
    generator = ClozeGenerator()
    result = generator.generate_clozes(
        "Napoleon was born in 1769 in Corsica.",
        max_clozes=3,
    )
    for cloze in result.clozes:
        print(cloze.cloze_text)
    # Output:
    # {{c1::Napoleon}} was born in 1769 in Corsica.
    # Napoleon was born in {{c2::1769}} in Corsica.
    # Napoleon was born in 1769 in {{c3::Corsica}}.

    # Or generate single multi-cloze card
    anki_text = generator.generate_anki_cloze(
        "Napoleon was born in 1769 in Corsica."
    )
    print(anki_text)
    # Output: {{c1::Napoleon}} was born in {{c2::1769}} in {{c3::Corsica}}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Maximum clozes per text (Rule #2: Fixed bounds)
DEFAULT_MAX_CLOZES: int = 5
MAX_ALLOWED_CLOZES: int = 20


class ClozeType(Enum):
    """Types of cloze deletions for learning prioritization."""

    DATE = "date"  # Temporal information (1066, July 4th)
    PERSON = "person"  # People names (Napoleon, Einstein)
    LOCATION = "location"  # Geographic locations (Paris, Nile)
    ORGANIZATION = "org"  # Organizations (NASA, United Nations)
    EVENT = "event"  # Historical events (World War II)
    CONCEPT = "concept"  # Key concepts (photosynthesis)
    QUANTITY = "quantity"  # Numbers, percentages (50%, 1000)
    WORK = "work"  # Works of art, books (Hamlet)


# Entity type mapping from spaCy to ClozeType
ENTITY_TYPE_MAP: dict[str, ClozeType] = {
    "DATE": ClozeType.DATE,
    "TIME": ClozeType.DATE,
    "PERSON": ClozeType.PERSON,
    "GPE": ClozeType.LOCATION,  # Geopolitical entity
    "LOC": ClozeType.LOCATION,
    "FAC": ClozeType.LOCATION,  # Facility
    "ORG": ClozeType.ORGANIZATION,
    "NORP": ClozeType.ORGANIZATION,  # Nationalities, religions
    "EVENT": ClozeType.EVENT,
    "WORK_OF_ART": ClozeType.WORK,
    "LAW": ClozeType.WORK,
    "PRODUCT": ClozeType.CONCEPT,
    "MONEY": ClozeType.QUANTITY,
    "QUANTITY": ClozeType.QUANTITY,
    "PERCENT": ClozeType.QUANTITY,
    "CARDINAL": ClozeType.QUANTITY,
    "ORDINAL": ClozeType.QUANTITY,
}

# Priority weights for different cloze types (higher = more important)
CLOZE_PRIORITIES: dict[ClozeType, float] = {
    ClozeType.DATE: 1.0,  # Dates are highly testable
    ClozeType.PERSON: 0.95,  # Names are important
    ClozeType.EVENT: 0.9,  # Events contextualize learning
    ClozeType.LOCATION: 0.85,  # Geography matters
    ClozeType.ORGANIZATION: 0.8,
    ClozeType.CONCEPT: 0.75,
    ClozeType.WORK: 0.7,
    ClozeType.QUANTITY: 0.65,  # Numbers are testable but less memorable
}


@dataclass
class ClozeCandidate:
    """A candidate term for cloze deletion.

    Attributes:
        text: The text to be replaced
        cloze_type: Type of cloze (date, person, etc.)
        start: Start position in original text
        end: End position in original text
        priority: Selection priority (0-1)
        context: Surrounding context for hint
    """

    text: str
    cloze_type: ClozeType
    start: int
    end: int
    priority: float = 0.5
    context: str = ""

    @property
    def length(self) -> int:
        """Length of the cloze term."""
        return len(self.text)

    def __hash__(self) -> int:
        """Enable hashing for deduplication."""
        return hash((self.text.lower(), self.start, self.end))


@dataclass
class Cloze:
    """A cloze deletion (fill-in-the-blank item).

    Attributes:
        original_text: The original text with cloze term
        cloze_text: Text with blank replacing the term
        answer: The correct answer (deleted term)
        cloze_type: Type of cloze for categorization
        hint: Optional hint for the learner
    """

    original_text: str
    cloze_text: str
    answer: str
    cloze_type: ClozeType
    hint: Optional[str] = None

    def __str__(self) -> str:
        """String representation showing the cloze."""
        return f"{self.cloze_text} (Answer: {self.answer})"


@dataclass
class ClozeResult:
    """Result of cloze generation.

    Attributes:
        clozes: List of generated cloze items
        source_text: Original source text
        candidate_count: Number of candidates found
        selected_count: Number actually selected
    """

    clozes: List[Cloze] = field(default_factory=list)
    source_text: str = ""
    candidate_count: int = 0
    selected_count: int = 0

    @property
    def count(self) -> int:
        """Number of clozes generated."""
        return len(self.clozes)

    @property
    def has_clozes(self) -> bool:
        """Whether any clozes were generated."""
        return len(self.clozes) > 0


class ClozeGenerator:
    """Generates intelligent cloze deletions using NER.

    This class uses spaCy NER to identify important entities and
    concepts in text, then generates fill-in-the-blank study items.

    Args:
        spacy_model: Optional pre-loaded spaCy model
        max_clozes: Default maximum clozes per text
        min_term_length: Minimum length for cloze terms
        cloze_marker: Marker for blank (default: "___")

    Example:
        generator = ClozeGenerator()
        result = generator.generate_clozes(
            "Napoleon Bonaparte was born in 1769.",
        )
        for cloze in result.clozes:
            print(cloze)
    """

    def __init__(
        self,
        spacy_model: Optional[object] = None,
        max_clozes: int = DEFAULT_MAX_CLOZES,
        min_term_length: int = 2,
        use_anki_format: bool = True,
        cloze_marker: str = "___",
    ) -> None:
        """Initialize the cloze generator.

        Args:
            spacy_model: Optional pre-loaded spaCy model
            max_clozes: Default maximum clozes per text
            min_term_length: Minimum length for cloze terms
            use_anki_format: Use Anki {{c1::...}} format (default: True)
            cloze_marker: Marker for blank when not using Anki format
        """
        if max_clozes > MAX_ALLOWED_CLOZES:
            raise ValueError(f"max_clozes cannot exceed {MAX_ALLOWED_CLOZES}")
        if max_clozes <= 0:
            raise ValueError("max_clozes must be positive")

        self._spacy_model = spacy_model
        self._spacy_available: Optional[bool] = None
        self.max_clozes = max_clozes
        self.min_term_length = min_term_length
        self.use_anki_format = use_anki_format
        self.cloze_marker = cloze_marker

    @property
    def spacy_model(self) -> Optional[object]:
        """Lazy-load spaCy model."""
        if self._spacy_model is not None:
            return self._spacy_model

        try:
            from ingestforge.enrichment.ner import _load_spacy_model

            self._spacy_model = _load_spacy_model()
            return self._spacy_model
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if NER is available."""
        if self._spacy_available is not None:
            return self._spacy_available

        self._spacy_available = self.spacy_model is not None
        return self._spacy_available

    def generate_clozes(
        self,
        text: str,
        max_clozes: Optional[int] = None,
        target_types: Optional[Set[ClozeType]] = None,
    ) -> ClozeResult:
        """Generate cloze deletions from text.

        Args:
            text: Source text to process
            max_clozes: Override default max clozes
            target_types: Only generate these cloze types

        Returns:
            ClozeResult with generated clozes
        """
        if not text or not text.strip():
            return ClozeResult(source_text="")

        # Apply limits (Rule #2)
        limit = min(
            max_clozes or self.max_clozes,
            MAX_ALLOWED_CLOZES,
        )

        # Find candidates
        candidates = self._find_candidates(text, target_types)

        # Select best candidates
        selected = self._select_best_candidates(candidates, limit)

        # Generate clozes
        clozes = self._create_clozes(text, selected)

        return ClozeResult(
            clozes=clozes,
            source_text=text,
            candidate_count=len(candidates),
            selected_count=len(selected),
        )

    def _find_candidates(
        self,
        text: str,
        target_types: Optional[Set[ClozeType]],
    ) -> List[ClozeCandidate]:
        """Find cloze candidates in text.

        Args:
            text: Source text
            target_types: Filter by these types

        Returns:
            List of candidates
        """
        if not self.is_available:
            logger.warning("spaCy unavailable, using fallback extraction")
            return self._fallback_candidates(text, target_types)

        return self._spacy_candidates(text, target_types)

    def _spacy_candidates(
        self,
        text: str,
        target_types: Optional[Set[ClozeType]],
    ) -> List[ClozeCandidate]:
        """Extract candidates using spaCy NER.

        Args:
            text: Source text
            target_types: Filter by these types

        Returns:
            List of candidates
        """
        nlp = self.spacy_model
        if nlp is None:
            return []

        doc = nlp(text)
        candidates: List[ClozeCandidate] = []

        for ent in doc.ents:
            # Map entity type to cloze type
            cloze_type = ENTITY_TYPE_MAP.get(ent.label_)
            if cloze_type is None:
                continue

            # Filter by target types
            if target_types and cloze_type not in target_types:
                continue

            # Skip short terms
            if len(ent.text) < self.min_term_length:
                continue

            # Calculate priority
            priority = CLOZE_PRIORITIES.get(cloze_type, 0.5)

            # Get context
            context = self._get_context(text, ent.start_char, ent.end_char)

            candidate = ClozeCandidate(
                text=ent.text,
                cloze_type=cloze_type,
                start=ent.start_char,
                end=ent.end_char,
                priority=priority,
                context=context,
            )
            candidates.append(candidate)

        return candidates

    def _fallback_candidates(
        self,
        text: str,
        target_types: Optional[Set[ClozeType]],
    ) -> List[ClozeCandidate]:
        """Extract candidates using regex fallback.

        Args:
            text: Source text
            target_types: Filter by these types

        Returns:
            List of candidates
        """
        import re

        candidates: List[ClozeCandidate] = []

        # Simple patterns for common entities
        patterns = [
            # Dates (1066, January 1, 2024)
            (r"\b\d{4}\b", ClozeType.DATE),
            (
                r"\b(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b",
                ClozeType.DATE,
            ),
            # Capitalized words (potential names/places)
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", ClozeType.CONCEPT),
        ]

        for pattern, cloze_type in patterns:
            # Filter by target types
            if target_types and cloze_type not in target_types:
                continue

            for match in re.finditer(pattern, text):
                term = match.group()
                if len(term) < self.min_term_length:
                    continue

                priority = CLOZE_PRIORITIES.get(cloze_type, 0.5)
                context = self._get_context(text, match.start(), match.end())

                candidate = ClozeCandidate(
                    text=term,
                    cloze_type=cloze_type,
                    start=match.start(),
                    end=match.end(),
                    priority=priority * 0.8,  # Reduce priority for fallback
                    context=context,
                )
                candidates.append(candidate)

        return candidates

    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 30,
    ) -> str:
        """Get surrounding context for a term.

        Args:
            text: Full text
            start: Term start position
            end: Term end position
            window: Context window size

        Returns:
            Context string
        """
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end].strip()

    def _select_best_candidates(
        self,
        candidates: List[ClozeCandidate],
        limit: int,
    ) -> List[ClozeCandidate]:
        """Select best candidates for cloze generation.

        Uses priority scores and avoids overlapping terms.

        Args:
            candidates: All found candidates
            limit: Maximum to select

        Returns:
            Selected candidates
        """
        if not candidates:
            return []

        # Sort by priority (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.priority,
            reverse=True,
        )

        # Select non-overlapping candidates
        selected: List[ClozeCandidate] = []
        used_ranges: List[Tuple[int, int]] = []

        for candidate in sorted_candidates:
            if len(selected) >= limit:
                break

            # Check for overlap with existing selections
            if self._overlaps(candidate, used_ranges):
                continue

            selected.append(candidate)
            used_ranges.append((candidate.start, candidate.end))

        return selected

    def _overlaps(
        self,
        candidate: ClozeCandidate,
        ranges: List[Tuple[int, int]],
    ) -> bool:
        """Check if candidate overlaps with existing ranges.

        Args:
            candidate: Candidate to check
            ranges: Existing selected ranges

        Returns:
            True if overlaps
        """
        for start, end in ranges:
            if candidate.start < end and candidate.end > start:
                return True
        return False

    def _create_clozes(
        self,
        text: str,
        candidates: List[ClozeCandidate],
    ) -> List[Cloze]:
        """Create cloze items from selected candidates.

        Clozes are numbered by text position (c1, c2, c3...) not priority.

        Args:
            text: Original text
            candidates: Selected candidates

        Returns:
            List of Cloze items sorted by text position
        """
        # Sort by text position for consistent numbering
        sorted_candidates = sorted(candidates, key=lambda c: c.start)
        clozes: List[Cloze] = []

        for index, candidate in enumerate(sorted_candidates, start=1):
            # Create cloze text based on format
            if self.use_anki_format:
                # Anki format: {{c1::answer}}
                cloze_marker = f"{{{{c{index}::{candidate.text}}}}}"
            else:
                # Simple marker format
                cloze_marker = self.cloze_marker

            cloze_text = text[: candidate.start] + cloze_marker + text[candidate.end :]

            # Create hint based on type
            hint = self._create_hint(candidate)

            cloze = Cloze(
                original_text=text,
                cloze_text=cloze_text,
                answer=candidate.text,
                cloze_type=candidate.cloze_type,
                hint=hint,
            )
            clozes.append(cloze)

        return clozes

    def _create_hint(self, candidate: ClozeCandidate) -> str:
        """Create a hint for a cloze.

        Args:
            candidate: The cloze candidate

        Returns:
            Hint string
        """
        type_hints = {
            ClozeType.DATE: "a date",
            ClozeType.PERSON: "a person's name",
            ClozeType.LOCATION: "a location",
            ClozeType.ORGANIZATION: "an organization",
            ClozeType.EVENT: "an event",
            ClozeType.CONCEPT: "a key concept",
            ClozeType.QUANTITY: "a number or quantity",
            ClozeType.WORK: "a work or title",
        }
        return f"({type_hints.get(candidate.cloze_type, 'a term')})"

    def generate_anki_cloze(
        self,
        text: str,
        max_clozes: Optional[int] = None,
        target_types: Optional[Set[ClozeType]] = None,
    ) -> str:
        """Generate single Anki cloze card with multiple deletions.

        Creates a single text with all clozes marked as {{c1::...}}, {{c2::...}}.
        This is ideal for Anki import where one card tests multiple facts.

        Args:
            text: Source text to process
            max_clozes: Override default max clozes
            target_types: Only generate these cloze types

        Returns:
            Text with Anki cloze markers inserted

        Example:
            >>> generator = ClozeGenerator()
            >>> result = generator.generate_anki_cloze(
            ...     "Napoleon was born in 1769 in Corsica."
            ... )
            >>> print(result)
            {{c1::Napoleon}} was born in {{c2::1769}} in {{c3::Corsica}}.
        """
        # Apply limits (Rule #2)
        limit = min(
            max_clozes or self.max_clozes,
            MAX_ALLOWED_CLOZES,
        )

        # Find and select candidates
        candidates = self._find_candidates(text, target_types)
        selected = self._select_best_candidates(candidates, limit)
        if not selected:
            return text

        # Sort by text position for numbering (c1, c2, c3...)
        sorted_by_position = sorted(selected, key=lambda c: c.start)

        # Create mapping of position to cloze number
        position_to_number = {
            candidate.start: index
            for index, candidate in enumerate(sorted_by_position, start=1)
        }

        # Replace from end to start to avoid offset issues
        sorted_for_replacement = sorted(selected, key=lambda c: c.start, reverse=True)

        result = text
        for candidate in sorted_for_replacement:
            # Get the cloze number based on text position
            cloze_number = position_to_number[candidate.start]
            cloze_marker = f"{{{{c{cloze_number}::{candidate.text}}}}}"
            result = result[: candidate.start] + cloze_marker + result[candidate.end :]

        return result


def generate_clozes(
    text: str,
    max_clozes: int = DEFAULT_MAX_CLOZES,
    target_types: Optional[Set[ClozeType]] = None,
    anki_format: bool = True,
) -> List[str]:
    """Convenience function to generate cloze deletions as strings.

    Args:
        text: Source text
        max_clozes: Maximum clozes
        target_types: Filter by types
        anki_format: Return Anki-formatted strings (default: True)

    Returns:
        List of cloze text strings in Anki format

    Example:
        >>> clozes = generate_clozes("Napoleon was born in 1769.")
        >>> print(clozes[0])
        {{c1::Napoleon}} was born in 1769.
        >>> print(clozes[1])
        Napoleon was born in {{c2::1769}}.
    """
    generator = ClozeGenerator(max_clozes=max_clozes, use_anki_format=anki_format)
    result = generator.generate_clozes(text, target_types=target_types)
    return [cloze.cloze_text for cloze in result.clozes]
