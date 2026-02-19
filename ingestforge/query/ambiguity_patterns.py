"""
Ambiguity Pattern Definitions for Query Clarification.

Query Clarification - Pattern library for detecting ambiguous queries.

Follows NASA JPL Power of Ten:
- Rule #2: Fixed bounds on all data structures
- Rule #9: Complete type hints
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# JPL Rule #2: Fixed upper bounds
MAX_PATTERNS = 100
MAX_MULTI_MEANING_TERMS = 50
MAX_PRONOUN_PATTERNS = 20

# =============================================================================
# Pronoun Patterns
# =============================================================================

PRONOUN_PATTERNS: Set[str] = {
    "he",
    "she",
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "them",
    "him",
    "her",
    "his",
    "hers",
    "its",
    "their",
    "theirs",
}

# Contextual pronouns that need resolution
CONTEXTUAL_PRONOUNS: Set[str] = {
    "he",
    "she",
    "it",
    "they",
    "them",
    "him",
    "her",
}

# Demonstrative pronouns
DEMONSTRATIVE_PRONOUNS: Set[str] = {
    "this",
    "that",
    "these",
    "those",
}


# =============================================================================
# Multi-Meaning Terms
# =============================================================================

MULTI_MEANING_TERMS: Dict[str, List[str]] = {
    "python": [
        "Python programming language",
        "Python snake species",
        "Monty Python comedy",
    ],
    "java": ["Java programming language", "Java island in Indonesia", "Java coffee"],
    "apple": ["Apple Inc. (technology company)", "Apple fruit"],
    "mercury": ["Mercury planet", "Mercury chemical element", "Mercury Roman god"],
    "mars": ["Mars planet", "Mars candy company", "Mars Roman god"],
    "oracle": ["Oracle Corporation", "Oracle ancient prophecy"],
    "amazon": ["Amazon company", "Amazon rainforest", "Amazon river"],
    "bolt": [
        "Usain Bolt athlete",
        "Lightning bolt",
        "Bolt fastener",
        "Bolt run quickly",
    ],
    "bass": ["Bass fish", "Bass musical instrument", "Bass low frequency"],
    "bat": ["Baseball bat", "Bat animal"],
    "bank": ["Financial bank", "River bank"],
    "bark": ["Tree bark", "Dog bark"],
    "bow": ["Bow and arrow", "Bow (tie)", "Bow (ship)", "Bow (bend)"],
    "club": ["Social club", "Golf club", "Nightclub"],
    "crane": ["Crane bird", "Construction crane"],
    "date": ["Calendar date", "Romantic date", "Date fruit"],
    "fan": ["Cooling fan", "Sports fan"],
    "pitch": ["Baseball pitch", "Musical pitch", "Sales pitch", "Pitch (tar)"],
    "row": ["Row of items", "Row a boat", "Argument/row"],
    "scale": ["Musical scale", "Weight scale", "Fish scale", "Map scale"],
    "spring": ["Spring season", "Coiled spring", "Water spring"],
    "tie": ["Necktie", "Tie score", "Tie knot"],
    "wave": ["Ocean wave", "Wave gesture", "Sound wave"],
    "ruby": ["Ruby programming language", "Ruby gemstone"],
    "swift": ["Swift programming language", "Swift bird", "Swift (fast)"],
    "go": ["Go programming language", "Go (verb)", "Go board game"],
    "rust": ["Rust programming language", "Rust (corrosion)"],
    "c": ["C programming language", "C note (music)", "C (letter)"],
    "r": ["R programming language", "R (letter)"],
    "julia": ["Julia programming language", "Julia (name)"],
}


# =============================================================================
# Temporal Ambiguity Patterns
# =============================================================================

TEMPORAL_AMBIGUITY_PATTERNS: Set[str] = {
    "recent",
    "recently",
    "lately",
    "soon",
    "before",
    "after",
    "earlier",
    "later",
    "previously",
    "formerly",
    "now",
    "current",
    "modern",
    "contemporary",
    "historic",
    "past",
    "future",
    "upcoming",
    "today",
    "yesterday",
    "tomorrow",
    "last",
    "next",
    "old",
    "new",
}

# Temporal qualifiers that need specific time range
VAGUE_TEMPORAL_QUALIFIERS: Set[str] = {
    "recent",
    "lately",
    "soon",
    "old",
    "new",
    "modern",
    "historic",
    "current",
    "past",
    "future",
}


# =============================================================================
# Vague/Ambiguous Query Patterns
# =============================================================================

# Patterns indicating vague/ambiguous queries (pattern, clarity_penalty)
VAGUE_PATTERNS: List[Tuple[str, float]] = [
    (r"^tell me more$", 0.1),
    (r"^more info$", 0.1),
    (r"^explain$", 0.2),
    (r"^help$", 0.2),
    (r"^what\??$", 0.1),
    (r"^how\??$", 0.1),
    (r"^why\??$", 0.2),
    (r"^it$", 0.1),
    (r"^this$", 0.1),
    (r"^that$", 0.1),
    (r"^something$", 0.2),
    (r"^anything$", 0.2),
    (r"^stuff$", 0.2),
    (r"^things$", 0.2),
    (r"^tell me about", 0.3),
    (r"^more about", 0.3),
    (r"^what about", 0.4),
]


# =============================================================================
# Specific/Clear Query Patterns
# =============================================================================

# Patterns indicating specific/clear queries (pattern, clarity_bonus)
SPECIFIC_PATTERNS: List[Tuple[str, float]] = [
    (r"\b(?:CEO|CTO|CFO|president|director)\s+of\s+\w+", 0.9),  # "CEO of Apple"
    (r"\b\d{4}\b", 0.8),  # Contains year (2024)
    (
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
        0.8,
    ),  # Month name
    (r"\"[^\"]+\"", 0.85),  # Quoted phrase
    (r"\b(?:in|at|on|by|from)\s+\w+", 0.7),  # Prepositional specificity
    (r"\b(?:how\s+(?:to|do|does|can))\s+\w+", 0.75),  # How-to questions
    (r"\b(?:what\s+is\s+(?:the|a|an))\s+\w+", 0.75),  # What-is questions
    (r"\b(?:who\s+(?:is|was|are|were))\s+\w+", 0.8),  # Who questions with subject
    (r"\b(?:version|v)\s*\d+\.\d+", 0.85),  # Version numbers (Python 3.10)
    (r"\b(?:chapter|section|page)\s+\d+", 0.85),  # Document references
    (r"\b\w+@\w+\.\w+", 0.9),  # Email addresses
    (r"\bhttps?://", 0.9),  # URLs
]


# =============================================================================
# Scope Ambiguity Indicators
# =============================================================================

# Keywords indicating query might be too broad
BROAD_SCOPE_INDICATORS: Set[str] = {
    "everything",
    "all",
    "any",
    "overview",
    "general",
    "introduction",
    "basics",
    "fundamentals",
    "summary",
}

# Keywords indicating specific scope
SPECIFIC_SCOPE_INDICATORS: Set[str] = {
    "specifically",
    "exactly",
    "precisely",
    "particular",
    "certain",
    "specific",
    "detailed",
    "in-depth",
}


# =============================================================================
# Validation
# =============================================================================

# JPL Rule #2: Assert bounds at module load
assert (
    len(VAGUE_PATTERNS) <= MAX_PATTERNS
), f"VAGUE_PATTERNS exceeds MAX_PATTERNS: {len(VAGUE_PATTERNS)}"
assert (
    len(SPECIFIC_PATTERNS) <= MAX_PATTERNS
), f"SPECIFIC_PATTERNS exceeds MAX_PATTERNS: {len(SPECIFIC_PATTERNS)}"
assert (
    len(MULTI_MEANING_TERMS) <= MAX_MULTI_MEANING_TERMS
), f"MULTI_MEANING_TERMS exceeds limit: {len(MULTI_MEANING_TERMS)}"
assert (
    len(PRONOUN_PATTERNS) <= MAX_PRONOUN_PATTERNS
), f"PRONOUN_PATTERNS exceeds limit: {len(PRONOUN_PATTERNS)}"
