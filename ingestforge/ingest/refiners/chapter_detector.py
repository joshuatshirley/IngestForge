"""
Chapter and Section Boundary Detector.

Detects chapter and section boundaries in text for better chunking:
- Chapter patterns: "Chapter X", "CHAPTER X", roman numerals
- Section patterns: "Section X.Y", numbered headers
- ALL CAPS headings as potential breaks
"""

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

from ingestforge.ingest.refinement import ChapterMarker, IRefiner, RefinedText


class ChapterDetector(IRefiner):
    """Detect chapter and section boundaries in text.

    This refiner identifies structural markers that indicate chapter or
    section boundaries. The detected markers can be used by downstream
    chunking to make better decisions about where to split text.

    Detection Patterns
    ------------------
    1. **Chapter Patterns**: Matches variations of chapter headings
       - "Chapter 1", "CHAPTER I", "Chapter One"
       - "Ch. 1", "Ch 1"

    2. **Section Patterns**: Matches numbered section headings
       - "Section 1.2", "1.2 Introduction"
       - "Part I", "Part 1"

    3. **ALL CAPS Headers**: Short lines in all capitals
       - "INTRODUCTION", "METHODOLOGY", "CONCLUSIONS"

    4. **Markdown Headers**: Lines starting with # symbols
       - "# Title", "## Section", "### Subsection"

    The refiner does NOT modify the text - it only detects markers
    that can be used by the chunking stage.

    Examples:
        >>> detector = ChapterDetector()
        >>> result = detector.refine("\\n\\nCHAPTER 1\\n\\nIntroduction\\n\\n...")
        >>> for marker in result.chapter_markers:
        ...     print(f"{marker.title} at position {marker.position}")
        CHAPTER 1 at position 2
    """

    # Roman numeral pattern
    ROMAN_PATTERN = r"(?:I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XL|L|LX{0,3}|XC|C{1,3})"

    # Chapter detection patterns with hierarchy levels
    CHAPTER_PATTERNS: List[Tuple[Pattern[str], int, str]] = []

    def __init__(
        self,
        detect_chapters: bool = True,
        detect_sections: bool = True,
        detect_caps_headers: bool = True,
        detect_markdown: bool = True,
        min_header_length: int = 3,
        max_header_length: int = 100,
    ):
        """Initialize chapter detector.

        Args:
            detect_chapters: Look for chapter-level markers (level 1)
            detect_sections: Look for section-level markers (levels 2-3)
            detect_caps_headers: Treat ALL CAPS lines as headers
            detect_markdown: Detect markdown-style headers (#, ##, etc.)
            min_header_length: Minimum characters for a header
            max_header_length: Maximum characters for a header
        """
        self.detect_chapters = detect_chapters
        self.detect_sections = detect_sections
        self.detect_caps_headers = detect_caps_headers
        self.detect_markdown = detect_markdown
        self.min_header_length = min_header_length
        self.max_header_length = max_header_length

        # Build pattern list
        self._patterns: List[Tuple[Pattern[str], int, str]] = self._build_patterns()

    def _add_chapter_patterns(
        self, patterns: list[tuple[Pattern[str], int, str]]
    ) -> None:
        """Add chapter and part patterns.

        Rule #4: No large functions - Extracted from _build_patterns
        """
        # "Chapter X" patterns (level 1)
        patterns.append(
            (
                re.compile(
                    rf"^(?:Chapter|CHAPTER|Ch\.?)\s+(?:\d+|{self.ROMAN_PATTERN}|(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten))\b[^\n]*",
                    re.IGNORECASE | re.MULTILINE,
                ),
                1,
                "chapter",
            )
        )

        # "Part X" patterns (level 1)
        patterns.append(
            (
                re.compile(
                    rf"^(?:Part|PART)\s+(?:\d+|{self.ROMAN_PATTERN}|(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten))\b[^\n]*",
                    re.IGNORECASE | re.MULTILINE,
                ),
                1,
                "part",
            )
        )

    def _add_section_patterns(
        self, patterns: list[tuple[Pattern[str], int, str]]
    ) -> None:
        """Add section patterns.

        Rule #4: No large functions - Extracted from _build_patterns
        """
        # "Section X.Y" patterns (level 2)
        patterns.append(
            (
                re.compile(
                    r"^(?:Section|SECTION|Sec\.?)\s+[\d.]+\b[^\n]*",
                    re.IGNORECASE | re.MULTILINE,
                ),
                2,
                "section",
            )
        )

        # Numbered section patterns like "1.2 Title" (level 2)
        patterns.append(
            (
                re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+[A-Z][^\n]{3,60}$", re.MULTILINE),
                2,
                "numbered_section",
            )
        )

        # Numbered patterns like "1. Title" at start of line (level 2)
        patterns.append(
            (
                re.compile(r"^(\d+)\.\s+[A-Z][^\n]{3,60}$", re.MULTILINE),
                2,
                "numbered_heading",
            )
        )

    def _add_markdown_patterns(
        self, patterns: list[tuple[Pattern[str], int, str]]
    ) -> None:
        """Add markdown header patterns.

        Rule #4: No large functions - Extracted from _build_patterns
        """
        # Markdown headers - level depends on # count
        # # = level 1, ## = level 2, ### = level 3, etc.
        patterns.append(
            (
                re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),
                0,  # Special: level determined by # count
                "markdown",
            )
        )

    def _build_patterns(self) -> List[Tuple[Pattern[str], int, str]]:
        """Build list of detection patterns.

        Rule #4: Function <60 lines (refactored to 18 lines)
        """
        patterns: list[tuple[Pattern[str], int, str]] = []

        if self.detect_chapters:
            self._add_chapter_patterns(patterns)

        if self.detect_sections:
            self._add_section_patterns(patterns)

        if self.detect_markdown:
            self._add_markdown_patterns(patterns)

        return patterns

    def is_available(self) -> bool:
        """Always available - uses only standard library."""
        return True

    def _apply_patterns_to_text(self, text: str) -> List[ChapterMarker]:
        """Apply detection patterns to text and collect markers.

        Rule #4: No large functions - Extracted from refine
        """
        markers: List[ChapterMarker] = []

        # Apply each pattern
        for pattern, default_level, pattern_type in self._patterns:
            for match in pattern.finditer(text):
                title = match.group(0).strip()

                # Determine level
                if pattern_type == "markdown":
                    # Count # symbols for level
                    hashes = match.group(1)
                    level = len(hashes)
                    title = match.group(2).strip()
                else:
                    level = default_level

                # Skip if too short or too long
                if len(title) < self.min_header_length:
                    continue
                if len(title) > self.max_header_length:
                    continue

                markers.append(
                    ChapterMarker(
                        position=match.start(),
                        title=title,
                        level=level,
                    )
                )

        return markers

    def _build_changes_summary(self, markers: List[ChapterMarker]) -> List[str]:
        """Build summary of detected changes.

        Rule #4: No large functions - Extracted from refine
        """
        changes = []
        if markers:
            chapter_count = sum(1 for m in markers if m.level == 1)
            section_count = sum(1 for m in markers if m.level > 1)
            if chapter_count:
                changes.append(f"Detected {chapter_count} chapter markers")
            if section_count:
                changes.append(f"Detected {section_count} section markers")
        return changes

    def refine(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RefinedText:
        """Detect chapter and section boundaries.

        Rule #4: Function <60 lines (refactored to 27 lines)

        Args:
            text: Text to analyze
            metadata: Optional metadata (not used by this refiner)

        Returns:
            RefinedText with chapter_markers populated (text unchanged)
        """
        if not text:
            return RefinedText(original=text, refined=text)

        # Apply patterns to collect markers
        markers = self._apply_patterns_to_text(text)

        # Detect ALL CAPS headers (if enabled)
        if self.detect_caps_headers:
            caps_markers = self._detect_caps_headers(text)
            markers.extend(caps_markers)

        # Sort by position and deduplicate overlapping markers
        markers = self._deduplicate_markers(markers)

        # Build changes summary
        changes = self._build_changes_summary(markers)

        return RefinedText(
            original=text,
            refined=text,  # Text is not modified
            changes=changes,
            chapter_markers=markers,
        )

    def _detect_caps_headers(self, text: str) -> List[ChapterMarker]:
        """Detect ALL CAPS lines as potential headers."""
        markers = []

        # Split into lines with positions
        lines = text.split("\n")
        pos = 0

        for line in lines:
            stripped = line.strip()

            # Check if it looks like a header
            if self._is_caps_header(stripped):
                markers.append(
                    ChapterMarker(
                        position=pos + (len(line) - len(line.lstrip())),
                        title=stripped,
                        level=2,  # Default to section level
                    )
                )

            pos += len(line) + 1  # +1 for newline

        return markers

    def _is_caps_header(self, line: str) -> bool:
        """Check if a line looks like an ALL CAPS header."""
        if not line:
            return False

        # Length constraints
        if len(line) < self.min_header_length:
            return False
        if len(line) > self.max_header_length:
            return False

        # Must be mostly uppercase letters
        alpha_chars = [c for c in line if c.isalpha()]
        if not alpha_chars:
            return False

        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio < 0.9:
            return False

        # Should not be a single word that's an acronym (e.g., "NASA")
        words = line.split()
        if len(words) == 1 and len(words[0]) <= 5:
            return False

        # Should not contain sentence-ending punctuation
        if line.endswith((".?!")):
            return False

        # Should not look like a list item
        if line.startswith(("-", "*", "•", "·")):
            return False

        return True

    def _deduplicate_markers(self, markers: List[ChapterMarker]) -> List[ChapterMarker]:
        """Remove duplicate and overlapping markers."""
        if not markers:
            return markers

        # Sort by position
        markers = sorted(markers, key=lambda m: m.position)

        # Remove duplicates at same position (keep highest level)
        seen_positions: Dict[int, ChapterMarker] = {}
        for marker in markers:
            if marker.position not in seen_positions:
                seen_positions[marker.position] = marker
            else:
                # Keep the one with lower level number (higher hierarchy)
                existing = seen_positions[marker.position]
                if marker.level < existing.level:
                    seen_positions[marker.position] = marker

        return sorted(seen_positions.values(), key=lambda m: m.position)
