"""
Text Cleaning Functions for Unstructured-style Processing.

Provides advanced text cleaning capabilities:
- group_broken_paragraphs: Join visually-split paragraphs from OCR
- clean_bullets: Normalize bullet formats to standard markers
- clean_prefix_postfix: Remove page numbers, headers, footers"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.ingest.refinement import IRefiner, RefinedText


class TextCleanerRefiner(IRefiner):
    """Advanced text cleaning for OCR artifacts and formatting issues.

    This refiner provides Unstructured-style text cleaning functions:

    1. **group_broken_paragraphs**: Joins lines that were visually split
       by OCR or PDF extraction but belong to the same paragraph.

    2. **clean_bullets**: Normalizes various bullet characters to standard
       markdown-compatible bullets.

    3. **clean_prefix_postfix**: Removes common artifacts like page numbers,
       running headers, and running footers.

    Each function can be independently toggled via constructor parameters.

    Examples:
        >>> cleaner = TextCleanerRefiner()
        >>> result = cleaner.refine("This is a broken\\nline of text.")
        >>> print(result.refined)
        'This is a broken line of text.'
    """

    # Bullet character mappings to normalize
    BULLET_MAPPINGS: Dict[str, str] = {
        "\u2022": "-",  # Bullet
        "\u2023": "-",  # Triangular bullet
        "\u2043": "-",  # Hyphen bullet
        "\u25aa": "-",  # Black small square
        "\u25ab": "-",  # White small square
        "\u25cf": "-",  # Black circle
        "\u25cb": "-",  # White circle
        "\u25e6": "-",  # White bullet
        "\u2219": "-",  # Bullet operator
        "\u00b7": "-",  # Middle dot
        "\u2013": "-",  # En dash (when used as bullet)
        "\u2014": "-",  # Em dash (when used as bullet)
        "\u27a2": "-",  # Three-D top-lighted rightwards arrowhead
        "\u2192": "-",  # Rightwards arrow
        "\u25ba": "-",  # Black right-pointing pointer
        "\u25b6": "-",  # Black right-pointing triangle
    }

    # Patterns for page numbers and running headers/footers
    PAGE_NUMBER_PATTERNS = [
        re.compile(
            r"^\s*-?\s*\d{1,4}\s*-?\s*$", re.MULTILINE
        ),  # Standalone page numbers
        re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE),  # 1/10 style
    ]

    def __init__(
        self,
        group_paragraphs: bool = True,
        clean_bullets: bool = True,
        clean_prefix_postfix: bool = True,
        min_line_length: int = 50,
        max_lines_to_merge: int = 5,
    ) -> None:
        """Initialize text cleaner.

        Args:
            group_paragraphs: Join broken paragraphs (default True)
            clean_bullets: Normalize bullet characters (default True)
            clean_prefix_postfix: Remove page numbers/headers (default True)
            min_line_length: Minimum line length to consider broken (default 50)
            max_lines_to_merge: Maximum consecutive lines to merge (default 5)
        """
        self.group_paragraphs = group_paragraphs
        self.clean_bullets = clean_bullets
        self.clean_prefix_postfix = clean_prefix_postfix
        self.min_line_length = min_line_length
        self.max_lines_to_merge = max_lines_to_merge

    def is_available(self) -> bool:
        """Always available - uses only standard library."""
        return True

    def refine(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RefinedText:
        """Apply text cleaning operations.

        Args:
            text: Text to clean
            metadata: Optional metadata (not used)

        Returns:
            RefinedText with cleaned text and change log
        """
        if not text:
            return RefinedText(original=text, refined=text)

        changes: List[str] = []
        result = text

        # 1. Clean prefix/postfix (page numbers, headers)
        if self.clean_prefix_postfix:
            result, prefix_changes = self._clean_prefix_postfix(result)
            if prefix_changes > 0:
                changes.append(f"Removed {prefix_changes} page numbers/headers")

        # 2. Clean bullets
        if self.clean_bullets:
            result, bullet_changes = self._clean_bullets(result)
            if bullet_changes > 0:
                changes.append(f"Normalized {bullet_changes} bullet characters")

        # 3. Group broken paragraphs
        if self.group_paragraphs:
            result, para_changes = self._group_broken_paragraphs(result)
            if para_changes > 0:
                changes.append(f"Merged {para_changes} broken paragraph lines")

        return RefinedText(
            original=text,
            refined=result,
            changes=changes,
        )

    def _clean_prefix_postfix(self, text: str) -> Tuple[str, int]:
        """Remove page numbers and running headers/footers.

        Args:
            text: Text to clean

        Returns:
            Tuple of (cleaned text, number of removals)
        """
        count = 0
        result = text

        for pattern in self.PAGE_NUMBER_PATTERNS:
            matches = pattern.findall(result)
            count += len(matches)
            result = pattern.sub("", result)

        # Clean up resulting multiple blank lines
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result, count

    def _clean_bullets(self, text: str) -> Tuple[str, int]:
        """Normalize bullet characters to standard markdown format.

        Args:
            text: Text to clean

        Returns:
            Tuple of (cleaned text, number of replacements)
        """
        count = 0

        for special, standard in self.BULLET_MAPPINGS.items():
            occurrences = text.count(special)
            if occurrences > 0:
                text = text.replace(special, standard)
                count += occurrences

        return text, count

    def _group_broken_paragraphs(self, text: str) -> Tuple[str, int]:
        """Join lines that appear to be broken paragraphs.

        Detects lines that end without sentence-ending punctuation
        and joins them with the following line.

        Args:
            text: Text to process

        Returns:
            Tuple of (processed text, number of lines merged)
        """
        lines = text.split("\n")
        merged_count = 0

        result_lines: List[str] = []
        i = 0

        while i < len(lines):
            current_line = lines[i]

            # Check if this line should be merged with the next
            if self._should_merge_with_next(current_line, lines, i):
                merged_line, lines_consumed = self._merge_lines(lines, i)
                result_lines.append(merged_line)
                merged_count += lines_consumed - 1
                i += lines_consumed
            else:
                result_lines.append(current_line)
                i += 1

        return "\n".join(result_lines), merged_count

    def _should_merge_with_next(self, line: str, lines: List[str], index: int) -> bool:
        """Check if a line should be merged with the next line.

        Args:
            line: Current line
            lines: All lines
            index: Current index

        Returns:
            True if lines should be merged
        """
        stripped = line.strip()

        # Empty line - don't merge
        if not stripped:
            return False

        # No next line
        if index + 1 >= len(lines):
            return False

        next_stripped = lines[index + 1].strip()

        # Next line is empty - don't merge
        if not next_stripped:
            return False

        # Current line ends with sentence-ending punctuation - don't merge
        if stripped.endswith((".", "!", "?", ":", ";", '"', "'", ")")):
            return False

        # Next line starts with capital - probably new sentence
        if next_stripped and next_stripped[0].isupper():
            # But check if current line is clearly incomplete
            if len(stripped) < self.min_line_length:
                return True
            return False

        # Next line starts with lowercase - likely continuation
        if next_stripped and next_stripped[0].islower():
            return True

        return False

    def _merge_lines(self, lines: List[str], start_index: int) -> Tuple[str, int]:
        """Merge consecutive broken lines.

        Args:
            lines: All lines
            start_index: Starting index

        Returns:
            Tuple of (merged line, number of lines consumed)
        """
        merged_parts: List[str] = [lines[start_index].strip()]
        consumed = 1

        i = start_index + 1
        while i < len(lines) and consumed < self.max_lines_to_merge:
            current = lines[i].strip()

            # Stop at empty lines
            if not current:
                break

            # Check if this continues the paragraph
            if self._is_continuation(current):
                merged_parts.append(current)
                consumed += 1
                i += 1
            else:
                break

        return " ".join(merged_parts), consumed

    def _is_continuation(self, line: str) -> bool:
        """Check if a line is a continuation of the previous line.

        Args:
            line: Line to check

        Returns:
            True if this appears to be a continuation
        """
        if not line:
            return False

        # Starts with lowercase letter - likely continuation
        if line[0].islower():
            return True

        # Starts with common continuing words
        continuing_words = ("and", "or", "but", "so", "yet", "nor", "for")
        first_word = line.split()[0].lower() if line.split() else ""
        if first_word in continuing_words:
            return True

        return False


def group_broken_paragraphs(text: str) -> str:
    """Convenience function to join broken paragraphs.

    Args:
        text: Text to process

    Returns:
        Text with broken paragraphs joined
    """
    cleaner = TextCleanerRefiner(
        group_paragraphs=True,
        clean_bullets=False,
        clean_prefix_postfix=False,
    )
    return cleaner.refine(text).refined


def clean_bullets(text: str) -> str:
    """Convenience function to normalize bullet characters.

    Args:
        text: Text to process

    Returns:
        Text with normalized bullets
    """
    cleaner = TextCleanerRefiner(
        group_paragraphs=False,
        clean_bullets=True,
        clean_prefix_postfix=False,
    )
    return cleaner.refine(text).refined


def clean_prefix_postfix(text: str) -> str:
    """Convenience function to remove page numbers and headers.

    Args:
        text: Text to process

    Returns:
        Text with page artifacts removed
    """
    cleaner = TextCleanerRefiner(
        group_paragraphs=False,
        clean_bullets=False,
        clean_prefix_postfix=True,
    )
    return cleaner.refine(text).refined
