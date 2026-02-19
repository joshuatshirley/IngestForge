"""PII redaction refiner engine (SEC-001.1).

Implements regex and NER-based masking for personally identifiable
information (PII) including email, phone, SSN, and person names.

NASA JPL Commandments compliance:
- Rule #1: Linear filter chain, no backtracking regex
- Rule #2: Fixed upper bounds on patterns
- Rule #4: Functions <60 lines
- Rule #7: Validate patterns to prevent over-masking
- Rule #9: Full type hints

Usage:
    from ingestforge.ingest.refiners.redaction import (
        PIIRedactor,
        RedactionConfig,
        redact_pii,
    )

    redactor = PIIRedactor()
    result = redactor.redact(text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Optional, Tuple, Pattern

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_PATTERNS_PER_TYPE = 10
MAX_WHITELIST_ENTRIES = 1000
MAX_REDACTIONS_PER_TEXT = 10000


class PIIType(Enum):
    """Types of PII that can be redacted."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"


@dataclass
class RedactionMatch:
    """A match found during redaction.

    Attributes:
        pii_type: Type of PII detected
        original: Original text that was matched
        start: Start position in text
        end: End position in text
        replacement: The replacement string used
        confidence: Confidence score (0-1)
    """

    pii_type: PIIType
    original: str
    start: int
    end: int
    replacement: str
    confidence: float = 1.0

    @property
    def length(self) -> int:
        """Length of the original match."""
        return self.end - self.start


@dataclass
class RedactionResult:
    """Result of a redaction operation.

    Attributes:
        original_text: Original input text
        redacted_text: Text with PII redacted
        matches: List of matches found
        skipped: Items skipped due to whitelist
        stats: Redaction statistics by type
    """

    original_text: str
    redacted_text: str
    matches: List[RedactionMatch] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def total_redactions(self) -> int:
        """Total number of redactions made."""
        return len(self.matches)

    @property
    def has_redactions(self) -> bool:
        """Whether any redactions were made."""
        return len(self.matches) > 0


@dataclass
class RedactionConfig:
    """Configuration for PII redaction.

    Attributes:
        enabled_types: Which PII types to detect
        whitelist: Terms to skip (technical names, etc.)
        mask_char: Character to use for masking
        preserve_length: Whether to preserve original length
        show_type: Whether to show PII type in replacement
        custom_patterns: Additional regex patterns
    """

    enabled_types: Set[PIIType] = field(
        default_factory=lambda: {
            PIIType.EMAIL,
            PIIType.PHONE,
            PIIType.SSN,
            PIIType.PERSON_NAME,
        }
    )
    whitelist: Set[str] = field(default_factory=set)
    mask_char: str = "*"
    preserve_length: bool = False
    show_type: bool = True
    custom_patterns: Dict[str, str] = field(default_factory=dict)

    def add_to_whitelist(self, term: str) -> bool:
        """Add a term to whitelist.

        Args:
            term: Term to whitelist

        Returns:
            True if added, False if at limit
        """
        if len(self.whitelist) >= MAX_WHITELIST_ENTRIES:
            return False
        self.whitelist.add(term.lower())
        return True

    def is_whitelisted(self, term: str) -> bool:
        """Check if a term is whitelisted."""
        return term.lower() in self.whitelist


# Pre-compiled regex patterns (Rule #1: No backtracking)
# These patterns are designed to be linear-time
PATTERNS: Dict[PIIType, List[Tuple[Pattern[str], float]]] = {
    PIIType.EMAIL: [
        # Standard email pattern
        (
            re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
            ),
            1.0,
        ),
    ],
    PIIType.PHONE: [
        # US phone formats
        (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), 0.9),
        # International format
        (re.compile(r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"), 0.9),
        # Parentheses format
        (re.compile(r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}"), 0.95),
    ],
    PIIType.SSN: [
        # SSN format: XXX-XX-XXXX
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 1.0),
        # SSN without dashes
        (re.compile(r"\b\d{9}\b"), 0.5),  # Lower confidence - could be other numbers
    ],
    PIIType.CREDIT_CARD: [
        # Credit card with spaces/dashes
        (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), 0.9),
        # Amex format
        (re.compile(r"\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b"), 0.9),
    ],
    PIIType.IP_ADDRESS: [
        # IPv4
        (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), 0.8),
    ],
    PIIType.DATE_OF_BIRTH: [
        # MM/DD/YYYY or MM-DD-YYYY
        (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"), 0.7),
        # YYYY-MM-DD
        (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), 0.6),
    ],
}


class PIIRedactor:
    """Redacts PII from text using regex and optional NER.

    Args:
        config: Redaction configuration
    """

    def __init__(self, config: Optional[RedactionConfig] = None) -> None:
        """Initialize the redactor."""
        self.config = config or RedactionConfig()
        self._ner_model: Optional[object] = None
        self._ner_available: Optional[bool] = None

    @property
    def ner_available(self) -> bool:
        """Check if NER is available for name detection."""
        if self._ner_available is not None:
            return self._ner_available

        try:
            import spacy

            self._ner_model = spacy.load("en_core_web_sm")
            self._ner_available = True
        except (ImportError, OSError):
            self._ner_available = False
            logger.info("spaCy not available, using regex-only name detection")

        return self._ner_available

    def redact(self, text: str) -> RedactionResult:
        """Redact PII from text.

        Args:
            text: Input text to redact

        Returns:
            RedactionResult with redacted text and statistics
        """
        if not text or not text.strip():
            return RedactionResult(
                original_text=text,
                redacted_text=text,
            )

        matches: List[RedactionMatch] = []
        skipped: List[str] = []

        # Find all matches for enabled types
        for pii_type in self.config.enabled_types:
            type_matches = self._find_matches(text, pii_type)

            for match in type_matches:
                # Check whitelist
                if self.config.is_whitelisted(match.original):
                    skipped.append(match.original)
                    continue

                matches.append(match)
                if len(matches) >= MAX_REDACTIONS_PER_TEXT:
                    logger.warning(
                        f"Hit max redactions limit ({MAX_REDACTIONS_PER_TEXT})"
                    )
                    break

        # Sort matches by position (reverse for replacement)
        matches.sort(key=lambda m: m.start, reverse=True)

        # Apply redactions
        redacted_text = text
        for match in matches:
            redacted_text = (
                redacted_text[: match.start]
                + match.replacement
                + redacted_text[match.end :]
            )

        # Calculate stats
        stats: Dict[str, int] = {}
        for match in matches:
            type_name = match.pii_type.value
            stats[type_name] = stats.get(type_name, 0) + 1

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            matches=matches,
            skipped=skipped,
            stats=stats,
        )

    def _find_matches(
        self,
        text: str,
        pii_type: PIIType,
    ) -> List[RedactionMatch]:
        """Find all matches for a PII type.

        Args:
            text: Text to search
            pii_type: Type of PII to find

        Returns:
            List of matches
        """
        matches: List[RedactionMatch] = []

        # Use regex patterns
        if pii_type in PATTERNS:
            for pattern, confidence in PATTERNS[pii_type]:
                for match in pattern.finditer(text):
                    replacement = self._get_replacement(
                        match.group(),
                        pii_type,
                    )
                    matches.append(
                        RedactionMatch(
                            pii_type=pii_type,
                            original=match.group(),
                            start=match.start(),
                            end=match.end(),
                            replacement=replacement,
                            confidence=confidence,
                        )
                    )

        # Use NER for person names
        if pii_type == PIIType.PERSON_NAME:
            matches.extend(self._find_person_names(text))

        # Use custom patterns
        if pii_type == PIIType.CUSTOM:
            matches.extend(self._find_custom(text))

        return matches

    def _find_person_names(self, text: str) -> List[RedactionMatch]:
        """Find person names using NER or fallback regex.

        Args:
            text: Text to search

        Returns:
            List of name matches
        """
        matches: List[RedactionMatch] = []

        if self.ner_available and self._ner_model:
            # Use spaCy NER
            doc = self._ner_model(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    replacement = self._get_replacement(ent.text, PIIType.PERSON_NAME)
                    matches.append(
                        RedactionMatch(
                            pii_type=PIIType.PERSON_NAME,
                            original=ent.text,
                            start=ent.start_char,
                            end=ent.end_char,
                            replacement=replacement,
                            confidence=0.85,
                        )
                    )
        else:
            # Fallback: Simple title + name pattern
            # This is less accurate but better than nothing
            name_pattern = re.compile(
                r"\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
            )
            for match in name_pattern.finditer(text):
                replacement = self._get_replacement(match.group(), PIIType.PERSON_NAME)
                matches.append(
                    RedactionMatch(
                        pii_type=PIIType.PERSON_NAME,
                        original=match.group(),
                        start=match.start(),
                        end=match.end(),
                        replacement=replacement,
                        confidence=0.7,
                    )
                )

        return matches

    def _find_custom(self, text: str) -> List[RedactionMatch]:
        """Find matches using custom patterns.

        Args:
            text: Text to search

        Returns:
            List of custom matches
        """
        matches: List[RedactionMatch] = []

        for name, pattern_str in self.config.custom_patterns.items():
            try:
                pattern = re.compile(pattern_str)
                for match in pattern.finditer(text):
                    replacement = self._get_replacement(match.group(), PIIType.CUSTOM)
                    matches.append(
                        RedactionMatch(
                            pii_type=PIIType.CUSTOM,
                            original=match.group(),
                            start=match.start(),
                            end=match.end(),
                            replacement=replacement,
                            confidence=0.9,
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid custom pattern '{name}': {e}")

        return matches

    def _get_replacement(self, original: str, pii_type: PIIType) -> str:
        """Generate replacement string for a match.

        Args:
            original: Original matched text
            pii_type: Type of PII

        Returns:
            Replacement string
        """
        if self.config.preserve_length:
            return self.config.mask_char * len(original)

        if self.config.show_type:
            return f"[{pii_type.value.upper()}]"

        return self.config.mask_char * 8

    def add_custom_pattern(self, name: str, pattern: str) -> bool:
        """Add a custom regex pattern.

        Args:
            name: Name for the pattern
            pattern: Regex pattern string

        Returns:
            True if valid and added
        """
        try:
            re.compile(pattern)
        except re.error:
            return False

        if len(self.config.custom_patterns) >= MAX_PATTERNS_PER_TYPE:
            return False

        self.config.custom_patterns[name] = pattern
        return True


def redact_pii(
    text: str,
    config: Optional[RedactionConfig] = None,
) -> RedactionResult:
    """Convenience function to redact PII from text.

    Args:
        text: Text to redact
        config: Optional configuration

    Returns:
        RedactionResult
    """
    redactor = PIIRedactor(config)
    return redactor.redact(text)


def redact_batch(
    texts: List[str],
    config: Optional[RedactionConfig] = None,
) -> List[RedactionResult]:
    """Redact PII from multiple texts.

    Args:
        texts: List of texts to redact
        config: Optional configuration

    Returns:
        List of RedactionResult
    """
    redactor = PIIRedactor(config)
    return [redactor.redact(text) for text in texts]
