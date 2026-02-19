"""
OCR Artifact Cleanup Refiner.

Fixes common OCR artifacts in extracted text:
- Ligature issues (fi, fl, ff → fi, fl, ff)
- Hyphenation artifacts (broken- words → broken words)
- Common OCR character errors (rn→m, 0→O, l→1)
- Repeated characters (........ → ...)
- Line-break issues mid-word
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.ingest.refinement import IRefiner, RefinedText


class OCRCleanupRefiner(IRefiner):
    """Clean up common OCR artifacts in text.

    This refiner addresses issues commonly seen in text extracted via OCR:

    1. **Ligature Issues**: Unicode ligatures (ﬁ, ﬂ, ﬀ, ﬃ, ﬄ) that should be
       expanded to their component characters.

    2. **Hyphenation Artifacts**: Words broken across lines with hyphens that
       should be rejoined (e.g., "broken-\\nword" → "brokenword").

    3. **Common OCR Errors**: Character substitutions like:
       - "rn" misread as "m"
       - "0" (zero) confused with "O" (letter)
       - "l" (lowercase L) confused with "1" (one) or "I"

    4. **Repeated Characters**: Excessive punctuation or whitespace that
       should be normalized (e.g., "......" → "...").

    5. **Mid-word Line Breaks**: Words split across lines without proper
       hyphenation indicators.

    Examples:
        >>> refiner = OCRCleanupRefiner()
        >>> result = refiner.refine("The ﬁrst ﬂoor has broken-\\nwindows")
        >>> print(result.refined)
        'The first floor has brokenwindows'
    """

    # Unicode ligatures to expand
    LIGATURES = {
        "\ufb01": "fi",  # ﬁ
        "\ufb02": "fl",  # ﬂ
        "\ufb00": "ff",  # ﬀ
        "\ufb03": "ffi",  # ﬃ
        "\ufb04": "ffl",  # ﬄ
        "\u0132": "IJ",  # Ĳ
        "\u0133": "ij",  # ĳ
        "\u0152": "OE",  # Œ
        "\u0153": "oe",  # œ
        "\u00c6": "AE",  # Æ
        "\u00e6": "ae",  # æ
    }

    # Common OCR word corrections (context-sensitive)
    # Format: (pattern, replacement, requires_word_boundary)
    OCR_WORD_FIXES: List[Tuple[str, str, bool]] = [
        # These are conservative - only fix clear errors
        (r"\brnay\b", "may", True),
        (r"\brnade\b", "made", True),
        (r"\bsarne\b", "same", True),
        (r"\bnarne\b", "name", True),
        (r"\btirne\b", "time", True),
        (r"\bcorne\b", "come", True),
        (r"\bhorne\b", "home", True),
        (r"\bsorne\b", "some", True),
    ]

    def __init__(
        self,
        fix_ligatures: bool = True,
        fix_hyphenation: bool = True,
        fix_repeated_chars: bool = True,
        fix_common_errors: bool = False,  # Conservative: disabled by default
    ):
        """Initialize OCR cleanup refiner.

        Args:
            fix_ligatures: Expand Unicode ligatures to component chars
            fix_hyphenation: Fix word breaks at line endings
            fix_repeated_chars: Normalize repeated punctuation/whitespace
            fix_common_errors: Apply common OCR word corrections (aggressive)
        """
        self.fix_ligatures = fix_ligatures
        self.fix_hyphenation = fix_hyphenation
        self.fix_repeated_chars = fix_repeated_chars
        self.fix_common_errors = fix_common_errors

    def is_available(self) -> bool:
        """Always available - uses only standard library."""
        return True

    def refine(self, text: str, metadata: Optional[Dict] = None) -> RefinedText:
        """Clean OCR artifacts from text.

        Args:
            text: Text to clean
            metadata: Optional metadata (not used by this refiner)

        Returns:
            RefinedText with cleaned text and change log
        """
        if not text:
            return RefinedText(original=text, refined=text)

        changes: List[str] = []
        result = text

        # 1. Fix ligatures
        if self.fix_ligatures:
            result, ligature_count = self._fix_ligatures(result)
            if ligature_count > 0:
                changes.append(f"Fixed {ligature_count} ligatures")

        # 2. Fix hyphenation artifacts
        if self.fix_hyphenation:
            result, hyphen_count = self._fix_hyphenation(result)
            if hyphen_count > 0:
                changes.append(f"Fixed {hyphen_count} hyphenation breaks")

        # 3. Fix repeated characters
        if self.fix_repeated_chars:
            result, repeat_count = self._fix_repeated_chars(result)
            if repeat_count > 0:
                changes.append(
                    f"Normalized {repeat_count} repeated character sequences"
                )

        # 4. Fix common OCR errors (if enabled)
        if self.fix_common_errors:
            result, error_count = self._fix_common_errors(result)
            if error_count > 0:
                changes.append(f"Fixed {error_count} common OCR errors")

        return RefinedText(
            original=text,
            refined=result,
            changes=changes,
        )

    def _fix_ligatures(self, text: str) -> Tuple[str, int]:
        """Expand Unicode ligatures to component characters."""
        count = 0
        for ligature, expansion in self.LIGATURES.items():
            occurrences = text.count(ligature)
            if occurrences > 0:
                text = text.replace(ligature, expansion)
                count += occurrences
        return text, count

    def _fix_hyphenation(self, text: str) -> Tuple[str, int]:
        """Fix words broken across lines with hyphens.

        Handles patterns like:
        - "word-\\n" followed by continuation
        - "word-\\r\\n" followed by continuation
        - Preserves legitimate hyphens in compound words
        """
        # Pattern: hyphen at end of line followed by lowercase continuation
        # This avoids breaking legitimate hyphenated words like "well-known"
        pattern = r"(\w+)-\s*\n\s*([a-z])"

        def rejoin(match: Any) -> Any:
            return match.group(1) + match.group(2)

        result, count = re.subn(pattern, rejoin, text)
        return result, count

    def _fix_repeated_chars(self, text: str) -> Tuple[str, int]:
        """Normalize repeated punctuation and whitespace."""
        count = 0

        # Normalize multiple periods to ellipsis (max 3)
        pattern = r"\.{4,}"
        result, n = re.subn(pattern, "...", text)
        count += n
        text = result

        # Normalize multiple spaces to single space (preserve paragraph breaks)
        pattern = r"[ \t]{2,}"
        result, n = re.subn(pattern, " ", text)
        count += n
        text = result

        # Normalize multiple dashes
        pattern = r"-{3,}"
        result, n = re.subn(pattern, "--", text)
        count += n
        text = result

        # Normalize multiple exclamation/question marks
        pattern = r"[!]{2,}"
        result, n = re.subn(pattern, "!", text)
        count += n
        text = result

        pattern = r"[?]{2,}"
        result, n = re.subn(pattern, "?", text)
        count += n
        text = result

        return text, count

    def _fix_common_errors(self, text: str) -> Tuple[str, int]:
        """Apply common OCR word corrections."""
        count = 0
        for pattern, replacement, _ in self.OCR_WORD_FIXES:
            result, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
            if n > 0:
                text = result
                count += n
        return text, count
