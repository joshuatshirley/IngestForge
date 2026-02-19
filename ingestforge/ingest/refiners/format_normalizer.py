"""
Text Format Normalizer Refiner.

Normalizes text formatting for consistency:
- Unicode normalization (NFKC)
- Whitespace standardization
- Quote and dash normalization
- Control character removal
- Paragraph boundary preservation
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from ingestforge.ingest.refinement import IRefiner, RefinedText


class FormatNormalizer(IRefiner):
    """Normalize text formatting for consistency.

    This refiner ensures consistent text formatting:

    1. **Unicode Normalization**: Apply NFKC normalization to convert
       compatibility characters to their canonical forms.

    2. **Whitespace Standardization**: Convert tabs to spaces, normalize
       multiple spaces to single space, standardize line endings.

    3. **Quote Normalization**: Convert curly quotes to straight quotes,
       normalize various quote characters.

    4. **Dash Normalization**: Convert en-dash, em-dash, and other dash
       variants to standard hyphens or double-hyphens.

    5. **Control Character Removal**: Remove invisible control characters
       that can cause issues in downstream processing.

    6. **Paragraph Preservation**: Maintain paragraph boundaries (double
       newlines) while normalizing internal whitespace.

    Examples:
        >>> normalizer = FormatNormalizer()
        >>> result = normalizer.refine("Hello\\u2014World")  # em-dash
        >>> print(result.refined)
        'Hello--World'
    """

    # Quote normalization mappings
    QUOTE_MAPPINGS = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201a": "'",  # Single low-9 quotation mark
        "\u201b": "'",  # Single high-reversed-9 quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u201e": '"',  # Double low-9 quotation mark
        "\u201f": '"',  # Double high-reversed-9 quotation mark
        "\u00ab": '"',  # Left-pointing double angle quotation mark
        "\u00bb": '"',  # Right-pointing double angle quotation mark
        "\u2039": "'",  # Single left-pointing angle quotation mark
        "\u203a": "'",  # Single right-pointing angle quotation mark
        "\u0060": "'",  # Grave accent
        "\u00b4": "'",  # Acute accent
    }

    # Dash normalization mappings
    DASH_MAPPINGS = {
        "\u2013": "-",  # En dash → hyphen
        "\u2014": "--",  # Em dash → double hyphen
        "\u2015": "--",  # Horizontal bar → double hyphen
        "\u2012": "-",  # Figure dash → hyphen
        "\u2010": "-",  # Hyphen (Unicode) → ASCII hyphen
        "\u2011": "-",  # Non-breaking hyphen → hyphen
        "\u2043": "-",  # Hyphen bullet → hyphen
        "\u2212": "-",  # Minus sign → hyphen
    }

    # Control characters to remove (excluding standard whitespace)
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    def __init__(
        self,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        normalize_quotes: bool = True,
        normalize_dashes: bool = True,
        remove_control_chars: bool = True,
        preserve_paragraphs: bool = True,
    ):
        """Initialize format normalizer.

        Args:
            normalize_unicode: Apply NFKC unicode normalization
            normalize_whitespace: Standardize whitespace characters
            normalize_quotes: Convert curly quotes to straight
            normalize_dashes: Convert dash variants to standard forms
            remove_control_chars: Remove invisible control characters
            preserve_paragraphs: Maintain paragraph boundaries (double newlines)
        """
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.normalize_quotes = normalize_quotes
        self.normalize_dashes = normalize_dashes
        self.remove_control_chars = remove_control_chars
        self.preserve_paragraphs = preserve_paragraphs

    def is_available(self) -> bool:
        """Always available - uses only standard library."""
        return True

    def refine(self, text: str, metadata: Optional[Dict] = None) -> RefinedText:
        """Normalize text formatting.

        Args:
            text: Text to normalize
            metadata: Optional metadata (not used by this refiner)

        Returns:
            RefinedText with normalized text and change log
        """
        if not text:
            return RefinedText(original=text, refined=text)

        changes: List[str] = []
        result = text

        # 1. Remove control characters first
        if self.remove_control_chars:
            result, ctrl_count = self._remove_control_chars(result)
            if ctrl_count > 0:
                changes.append(f"Removed {ctrl_count} control characters")

        # 2. Unicode normalization
        if self.normalize_unicode:
            original_len = len(result)
            result = unicodedata.normalize("NFKC", result)
            if len(result) != original_len:
                changes.append("Applied NFKC unicode normalization")

        # 3. Normalize quotes
        if self.normalize_quotes:
            result, quote_count = self._normalize_quotes(result)
            if quote_count > 0:
                changes.append(f"Normalized {quote_count} quote characters")

        # 4. Normalize dashes
        if self.normalize_dashes:
            result, dash_count = self._normalize_dashes(result)
            if dash_count > 0:
                changes.append(f"Normalized {dash_count} dash characters")

        # 5. Normalize whitespace (preserve paragraphs)
        if self.normalize_whitespace:
            result, ws_changes = self._normalize_whitespace(result)
            if ws_changes:
                changes.append("Normalized whitespace")

        return RefinedText(
            original=text,
            refined=result,
            changes=changes,
        )

    def _remove_control_chars(self, text: str) -> Tuple[str, int]:
        """Remove invisible control characters."""
        result, count = self.CONTROL_CHAR_PATTERN.subn("", text)
        return result, count

    def _normalize_quotes(self, text: str) -> Tuple[str, int]:
        """Convert curly quotes to straight quotes."""
        count = 0
        for curly, straight in self.QUOTE_MAPPINGS.items():
            occurrences = text.count(curly)
            if occurrences > 0:
                text = text.replace(curly, straight)
                count += occurrences
        return text, count

    def _normalize_dashes(self, text: str) -> Tuple[str, int]:
        """Convert dash variants to standard forms."""
        count = 0
        for special, standard in self.DASH_MAPPINGS.items():
            occurrences = text.count(special)
            if occurrences > 0:
                text = text.replace(special, standard)
                count += occurrences
        return text, count

    def _normalize_whitespace(self, text: str) -> Tuple[str, bool]:
        """Normalize whitespace while preserving paragraph structure."""
        original = text

        # Convert Windows line endings to Unix
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Convert tabs to spaces
        text = text.replace("\t", " ")

        # Normalize non-breaking spaces to regular spaces
        text = text.replace("\u00a0", " ")
        text = text.replace("\u2007", " ")  # Figure space
        text = text.replace("\u2008", " ")  # Punctuation space
        text = text.replace("\u2009", " ")  # Thin space
        text = text.replace("\u200a", " ")  # Hair space
        text = text.replace("\u200b", "")  # Zero-width space (remove)
        text = text.replace("\u202f", " ")  # Narrow no-break space
        text = text.replace("\u205f", " ")  # Medium mathematical space
        text = text.replace("\u3000", " ")  # Ideographic space

        if self.preserve_paragraphs:
            # Preserve paragraph breaks (2+ newlines) by marking them
            text = re.sub(r"\n{2,}", "\n\n", text)

            # Split into paragraphs, normalize each, rejoin
            paragraphs = text.split("\n\n")
            normalized_paragraphs = []
            for para in paragraphs:
                # Collapse multiple spaces to single space within paragraph
                para = re.sub(r"[ ]+", " ", para)
                # Replace single newlines with space (soft wrap)
                para = re.sub(r"\n", " ", para)
                # Strip leading/trailing whitespace from paragraph
                para = para.strip()
                if para:
                    normalized_paragraphs.append(para)
            text = "\n\n".join(normalized_paragraphs)
        else:
            # Simple normalization: collapse all whitespace
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        return text, text != original
