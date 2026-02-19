"""
Data models for literary structure detection.

Provides data structures for sections, dialogue, and literary structure.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DialogueLine:
    """A single line of dialogue with speaker attribution."""

    speaker: str
    text: str
    line_number: int


@dataclass
class LiterarySection:
    """A structural division in a literary work."""

    section_type: str  # chapter, act, scene, part, book, canto
    label: str  # "Chapter 1", "Act I", etc.
    title: str  # Optional title after label
    start_line: int
    end_line: int
    level: int  # Nesting depth (0 = top level)
    children: List["LiterarySection"] = field(default_factory=list)


@dataclass
class LiteraryStructure:
    """Complete structural analysis of a literary text."""

    sections: List[LiterarySection]
    dialogue: List[DialogueLine]
    is_verse: bool
    is_dramatic: bool
    estimated_form: str  # novel, play, poem, short_story, epic_poem, novella


# Section type hierarchy levels (lower = higher in hierarchy)
SECTION_LEVELS = {
    "part": 0,
    "book": 1,
    "act": 1,
    "canto": 2,
    "chapter": 2,
    "scene": 3,
}
_compiled_section_patterns: List[Tuple[re.Pattern[str], str, int]] = []


def _get_section_patterns() -> List[Tuple[re.Pattern[str], str, int]]:
    """
    Get compiled section patterns (lazy-loaded and cached).

    Rule #6: Encapsulates pattern compilation in smallest scope.
    """
    # Cache patterns after first compilation
    if not _compiled_section_patterns:
        raw = [
            (
                r"^Part\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten"
                r"|[IVXLCDMivxlcdm]+|\d+)(?:\s*[:\-–—]\s*(.+))?$",
                "part",
                0,
                re.IGNORECASE,
            ),
            (
                r"^Book\s+(?:[IVXLCDMivxlcdm]+|\d+)(?:\s*[:\-–—]\s*(.+))?$",
                "book",
                1,
                re.IGNORECASE,
            ),
            (r"^Act\s+(?:[IVXLCDMivxlcdm]+|\d+)$", "act", 1, re.IGNORECASE),
            (r"^Canto\s+(?:[IVXLCDMivxlcdm]+|\d+)$", "canto", 2, re.IGNORECASE),
            (
                r"^Chapter\s+(?:[IVXLCDMivxlcdm]+|\d+)(?:\s*[:\-–—]\s*(.+))?$",
                "chapter",
                2,
                re.IGNORECASE,
            ),
            (r"^Scene\s+(?:[IVXLCDMivxlcdm]+|\d+)$", "scene", 3, re.IGNORECASE),
            # "1. Title"
            (r"^\d+\.\s+(.+)$", "chapter", 2, 0),
            # "3 - Title"
            (r"^\d+\s*[-–—]\s+(.+)$", "chapter", 2, 0),
            # Standalone Roman numeral (uppercase only to avoid false positives)
            (r"^([IVXLCDM]+)$", "chapter", 2, 0),
        ]
        _compiled_section_patterns.extend(
            [(re.compile(pat, flags), stype, level) for pat, stype, level, flags in raw]
        )
    return _compiled_section_patterns


SECTION_PATTERNS = _get_section_patterns()

# Prose dialogue: "text," said/asked/replied/etc Speaker
PROSE_DIALOGUE_RE = re.compile(
    r'["\u201c\u201d]([^"\u201c\u201d\n]+?)["\u201c\u201d]\s*'
    r"(?:,\s*)?"
    r"(?:said|asked|replied|whispered|exclaimed|cried|returned|murmured|shouted|answered|spoke|muttered|remarked)\s+"
    r"([A-Z][A-Za-z.]+(?:\s+[A-Za-z.]+){0,4}?)"
    r"(?:\s+(?:to|anxiously|quietly|impatiently|softly|sadly|angrily)\b[^\n]*?)?"
    r"[,.\n]"
)

# Reverse form: Speaker said/replied, "text"
PROSE_DIALOGUE_REV_RE = re.compile(
    r"([A-Z][A-Za-z.]+(?:\s+[A-Za-z.]+){0,4}?)\s+"
    r"(?:said|asked|replied|whispered|exclaimed|cried|returned|murmured|shouted|answered|spoke|muttered|remarked)"
    r"(?:\s+\w+)?"
    r',?\s*["\u201c\u201d]([^"\u201c\u201d\n]+?)["\u201c\u201d]'
)

# Dramatic dialogue: SPEAKER. text  or SPEAKER NAME. text
DRAMATIC_DIALOGUE_RE = re.compile(r"^([A-Z][A-Z\s]+?)\.\s+(.+?)$", re.MULTILINE)

# Stage directions in brackets or parentheses
STAGE_DIRECTION_RE = re.compile(r"\[.*?\]|\(.*?\)")
