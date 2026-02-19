"""Domain pattern registry for technical term prioritization (NLP-001.2).

This module provides domain-specific patterns for identifying key terms
in different subject areas (History, Science, Psychology, etc.).

NASA JPL Commandments compliance:
- Rule #1: No deep nesting, dictionary dispatch
- Rule #2: Fixed iteration bounds
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.enrichment.patterns import (
        PatternRegistry,
        Domain,
        find_domain_terms,
    )

    registry = PatternRegistry()
    terms = registry.find_terms(text, domain=Domain.HISTORY)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Pattern, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Maximum terms per search (Rule #2)
DEFAULT_MAX_TERMS: int = 20
MAX_ALLOWED_TERMS: int = 100


class Domain(Enum):
    """Subject domains for pattern matching."""

    HISTORY = "history"
    SCIENCE = "science"
    PSYCHOLOGY = "psychology"
    MATHEMATICS = "mathematics"
    LITERATURE = "literature"
    MEDICINE = "medicine"
    LAW = "law"
    GENERAL = "general"


@dataclass
class TermMatch:
    """A matched domain term.

    Attributes:
        text: The matched text
        domain: Domain this term belongs to
        category: Sub-category within domain
        start: Start position in text
        end: End position in text
        confidence: Match confidence (0-1)
    """

    text: str
    domain: Domain
    category: str
    start: int
    end: int
    confidence: float = 0.9

    @property
    def length(self) -> int:
        """Length of the term."""
        return len(self.text)

    def __hash__(self) -> int:
        """Enable hashing."""
        return hash((self.text.lower(), self.start, self.end))


@dataclass
class DomainPattern:
    """A pattern for matching domain-specific terms.

    Attributes:
        pattern: Regex pattern
        domain: Target domain
        category: Term category
        priority: Matching priority (higher = first)
        description: What this pattern matches
    """

    pattern: str
    domain: Domain
    category: str
    priority: float = 1.0
    description: str = ""

    _compiled: Optional[Pattern] = field(default=None, repr=False)

    @property
    def compiled(self) -> Pattern:
        """Get compiled regex pattern."""
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        return self._compiled


# History domain patterns
HISTORY_PATTERNS: List[DomainPattern] = [
    # Wars and conflicts
    DomainPattern(
        pattern=r"\b(?:World War|Civil War|Revolutionary War|Cold War)"
        r"(?:\s+[IV]+|\s+\d+)?\b",
        domain=Domain.HISTORY,
        category="war",
        priority=1.0,
        description="Major wars and conflicts",
    ),
    # Historical periods
    DomainPattern(
        pattern=r"\b(?:Renaissance|Enlightenment|Industrial Revolution|"
        r"Dark Ages|Middle Ages|Bronze Age|Iron Age|Stone Age|"
        r"Victorian Era|Gilded Age)\b",
        domain=Domain.HISTORY,
        category="period",
        priority=0.95,
        description="Historical periods and eras",
    ),
    # Treaties and agreements
    DomainPattern(
        pattern=r"\b(?:Treaty of|Pact of|Agreement of|Convention of|"
        r"Accord of)\s+[A-Z][a-z]+\b",
        domain=Domain.HISTORY,
        category="treaty",
        priority=0.9,
        description="Treaties and international agreements",
    ),
    # Historical events
    DomainPattern(
        pattern=r"\b(?:Battle of|Siege of|Massacre of|Revolution of)\s+"
        r"[A-Z][a-z]+\b",
        domain=Domain.HISTORY,
        category="event",
        priority=0.9,
        description="Historical battles and events",
    ),
    # Century references
    DomainPattern(
        pattern=r"\b\d{1,2}(?:st|nd|rd|th)\s+century\b",
        domain=Domain.HISTORY,
        category="time",
        priority=0.85,
        description="Century references",
    ),
]

# Science domain patterns
SCIENCE_PATTERNS: List[DomainPattern] = [
    # Scientific laws and principles
    DomainPattern(
        pattern=r"\b(?:Newton's|Einstein's|Boyle's|Charles'|Ohm's|"
        r"Kepler's|Mendel's|Darwin's)\s+(?:Law|Laws|Theory|Principle)\b",
        domain=Domain.SCIENCE,
        category="law",
        priority=1.0,
        description="Scientific laws and theories",
    ),
    # Processes
    DomainPattern(
        pattern=r"\b(?:photosynthesis|mitosis|meiosis|osmosis|diffusion|"
        r"respiration|fermentation|oxidation|reduction|"
        r"evaporation|condensation|sublimation)\b",
        domain=Domain.SCIENCE,
        category="process",
        priority=0.95,
        description="Scientific processes",
    ),
    # Particles and units
    DomainPattern(
        pattern=r"\b(?:electron|proton|neutron|photon|quark|neutrino|"
        r"atom|molecule|ion|isotope|nucleus)\b",
        domain=Domain.SCIENCE,
        category="particle",
        priority=0.9,
        description="Particles and atomic components",
    ),
    # Chemical elements
    DomainPattern(
        pattern=r"\b(?:hydrogen|helium|lithium|carbon|nitrogen|oxygen|"
        r"sodium|potassium|calcium|iron|copper|zinc|gold|silver|"
        r"uranium|plutonium)\b",
        domain=Domain.SCIENCE,
        category="element",
        priority=0.85,
        description="Chemical elements",
    ),
    # Scientific units
    DomainPattern(
        pattern=r"\b\d+(?:\.\d+)?\s*(?:meters?|kilometres?|kilometers?|"
        r"grams?|kilograms?|liters?|litres?|seconds?|hertz|"
        r"joules?|watts?|volts?|amperes?|ohms?)\b",
        domain=Domain.SCIENCE,
        category="measurement",
        priority=0.8,
        description="Scientific measurements",
    ),
]

# Psychology domain patterns
PSYCHOLOGY_PATTERNS: List[DomainPattern] = [
    # Psychological theories
    DomainPattern(
        pattern=r"\b(?:Freud's|Jung's|Piaget's|Maslow's|Skinner's|"
        r"Pavlov's|Erikson's|Bandura's)\s+(?:Theory|Hierarchy|"
        r"Model|Stages|Experiment)\b",
        domain=Domain.PSYCHOLOGY,
        category="theory",
        priority=1.0,
        description="Psychological theories",
    ),
    # Psychological concepts
    DomainPattern(
        pattern=r"\b(?:cognitive dissonance|classical conditioning|"
        r"operant conditioning|confirmation bias|hindsight bias|"
        r"placebo effect|halo effect|Stockholm syndrome|"
        r"Dunning-Kruger effect|impostor syndrome)\b",
        domain=Domain.PSYCHOLOGY,
        category="concept",
        priority=0.95,
        description="Psychological concepts and effects",
    ),
    # Mental disorders (clinical)
    DomainPattern(
        pattern=r"\b(?:depression|anxiety|PTSD|bipolar disorder|"
        r"schizophrenia|OCD|ADHD|autism spectrum|"
        r"personality disorder)\b",
        domain=Domain.PSYCHOLOGY,
        category="disorder",
        priority=0.9,
        description="Mental health conditions",
    ),
    # Brain regions
    DomainPattern(
        pattern=r"\b(?:hippocampus|amygdala|prefrontal cortex|"
        r"cerebellum|hypothalamus|thalamus|brain stem|"
        r"temporal lobe|frontal lobe|parietal lobe|occipital lobe)\b",
        domain=Domain.PSYCHOLOGY,
        category="anatomy",
        priority=0.85,
        description="Brain regions and anatomy",
    ),
]

# General patterns (domain-agnostic)
GENERAL_PATTERNS: List[DomainPattern] = [
    # Definitions (X is Y, X refers to)
    DomainPattern(
        pattern=r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|refers to|means|"
        r"denotes|signifies)\s+",
        domain=Domain.GENERAL,
        category="definition",
        priority=0.8,
        description="Defined terms",
    ),
    # Proper nouns (capitalized sequences)
    DomainPattern(
        pattern=r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
        domain=Domain.GENERAL,
        category="proper_noun",
        priority=0.6,
        description="Proper nouns",
    ),
]

# Domain to patterns mapping
DOMAIN_PATTERNS: Dict[Domain, List[DomainPattern]] = {
    Domain.HISTORY: HISTORY_PATTERNS,
    Domain.SCIENCE: SCIENCE_PATTERNS,
    Domain.PSYCHOLOGY: PSYCHOLOGY_PATTERNS,
    Domain.GENERAL: GENERAL_PATTERNS,
}


class PatternRegistry:
    """Registry for domain-specific term patterns.

    Provides methods to find domain-specific terms in text
    using configurable pattern sets.

    Args:
        custom_patterns: Additional patterns to register
        default_domain: Domain to use when none specified

    Example:
        registry = PatternRegistry()
        terms = registry.find_terms(
            "The Renaissance began in Italy.",
            domain=Domain.HISTORY,
        )
    """

    def __init__(
        self,
        custom_patterns: Optional[List[DomainPattern]] = None,
        default_domain: Domain = Domain.GENERAL,
    ) -> None:
        """Initialize the pattern registry."""
        self._patterns: Dict[Domain, List[DomainPattern]] = {}
        self.default_domain = default_domain

        # Load built-in patterns
        for domain, patterns in DOMAIN_PATTERNS.items():
            self._patterns[domain] = list(patterns)

        # Add custom patterns
        if custom_patterns:
            for pattern in custom_patterns:
                self.register_pattern(pattern)

    def register_pattern(self, pattern: DomainPattern) -> None:
        """Register a custom pattern.

        Args:
            pattern: Pattern to register
        """
        domain = pattern.domain
        if domain not in self._patterns:
            self._patterns[domain] = []
        self._patterns[domain].append(pattern)

    def get_patterns(self, domain: Domain) -> List[DomainPattern]:
        """Get patterns for a domain.

        Args:
            domain: Target domain

        Returns:
            List of patterns
        """
        patterns = list(self._patterns.get(domain, []))

        # Always include general patterns
        if domain != Domain.GENERAL:
            patterns.extend(self._patterns.get(Domain.GENERAL, []))

        return patterns

    def find_terms(
        self,
        text: str,
        domain: Optional[Domain] = None,
        max_terms: int = DEFAULT_MAX_TERMS,
    ) -> List[TermMatch]:
        """Find domain-specific terms in text.

        Args:
            text: Text to search
            domain: Domain to use (or default)
            max_terms: Maximum terms to return

        Returns:
            List of matched terms
        """
        if not text or not text.strip():
            return []
        limit = min(max_terms, MAX_ALLOWED_TERMS)

        domain = domain or self.default_domain
        patterns = self.get_patterns(domain)

        # Find all matches
        matches: List[TermMatch] = []

        for pattern in patterns:
            regex = pattern.compiled

            for match in regex.finditer(text):
                term_match = TermMatch(
                    text=match.group(),
                    domain=pattern.domain,
                    category=pattern.category,
                    start=match.start(),
                    end=match.end(),
                    confidence=pattern.priority,
                )
                matches.append(term_match)

        # Deduplicate by position
        unique_matches = self._deduplicate(matches)

        # Sort by priority and return limited
        sorted_matches = sorted(
            unique_matches,
            key=lambda m: m.confidence,
            reverse=True,
        )

        return sorted_matches[:limit]

    def _deduplicate(
        self,
        matches: List[TermMatch],
    ) -> List[TermMatch]:
        """Remove duplicate matches, keeping highest priority.

        Args:
            matches: All matches

        Returns:
            Deduplicated matches
        """
        seen_positions: Dict[Tuple[int, int], TermMatch] = {}

        for match in matches:
            key = (match.start, match.end)
            if key not in seen_positions:
                seen_positions[key] = match
            elif match.confidence > seen_positions[key].confidence:
                seen_positions[key] = match

        return list(seen_positions.values())

    def detect_domain(self, text: str) -> Domain:
        """Auto-detect the domain of text.

        Uses pattern matching to guess the most likely domain.

        Args:
            text: Text to analyze

        Returns:
            Detected domain
        """
        if not text or not text.strip():
            return Domain.GENERAL

        domain_scores: Dict[Domain, int] = {}

        # Count matches per domain
        for domain in [Domain.HISTORY, Domain.SCIENCE, Domain.PSYCHOLOGY]:
            patterns = self._patterns.get(domain, [])

            score = 0
            for pattern in patterns:
                matches = list(pattern.compiled.finditer(text))
                score += len(matches)

            if score > 0:
                domain_scores[domain] = score
        if not domain_scores:
            return Domain.GENERAL

        # Return highest scoring domain
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain

    def list_domains(self) -> List[Domain]:
        """List all registered domains.

        Returns:
            List of domains with patterns
        """
        return [d for d in Domain if d in self._patterns and self._patterns[d]]


def find_domain_terms(
    text: str,
    domain: Optional[Domain] = None,
    max_terms: int = DEFAULT_MAX_TERMS,
) -> List[TermMatch]:
    """Convenience function to find domain terms.

    Args:
        text: Text to search
        domain: Optional domain
        max_terms: Maximum results

    Returns:
        List of matched terms
    """
    registry = PatternRegistry()

    # Auto-detect domain if not specified
    if domain is None:
        domain = registry.detect_domain(text)

    return registry.find_terms(text, domain=domain, max_terms=max_terms)
