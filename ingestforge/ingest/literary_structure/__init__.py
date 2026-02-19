"""
Literary structure detection for prose, poetry, and dramatic works.

Detects chapters, acts/scenes, parts/books/cantos, dialogue, verse,
and estimates literary form (novel, play, poem, short story, etc.).
"""

# Models
from ingestforge.ingest.literary_structure.models import (
    DRAMATIC_DIALOGUE_RE,
    PROSE_DIALOGUE_RE,
    PROSE_DIALOGUE_REV_RE,
    SECTION_LEVELS,
    SECTION_PATTERNS,
    STAGE_DIRECTION_RE,
    DialogueLine,
    LiterarySection,
    LiteraryStructure,
    _get_section_patterns,
)

# Detector
from ingestforge.ingest.literary_structure.detector import LiteraryStructureDetector

__all__ = [
    # Models
    "DialogueLine",
    "LiterarySection",
    "LiteraryStructure",
    "SECTION_LEVELS",
    "SECTION_PATTERNS",
    "_get_section_patterns",
    "PROSE_DIALOGUE_RE",
    "PROSE_DIALOGUE_REV_RE",
    "DRAMATIC_DIALOGUE_RE",
    "STAGE_DIRECTION_RE",
    # Detector
    "LiteraryStructureDetector",
]
