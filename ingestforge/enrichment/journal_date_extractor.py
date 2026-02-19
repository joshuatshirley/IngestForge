"""
Journal date extraction.

Extracts dates from journal entries via:
- Filenames (2023-01-01.md, 20230101.md, Jan-01-2023.md)
- YAML frontmatter (date: 2023-01-01)
- Content headers (# January 1, 2023)
"""

import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Any, List, Tuple


@dataclass
class ExtractedDate:
    """A date extracted from a journal entry."""

    date: date
    source: str  # 'filename', 'frontmatter', 'content'
    confidence: float  # 0.0 to 1.0
    raw_value: str  # Original string that was parsed
    field_name: Optional[str] = None  # For frontmatter: 'date', 'created', etc.


class JournalDateExtractor:
    """
    Extract dates from journal entries.

    Supports multiple date formats and extraction sources:
    - Filenames: 2023-01-01.md, 20230101.md, Jan-01-2023.md
    - YAML frontmatter: date, created, created_at, published
    - Content: Headers with dates, first line dates
    """

    # Filename date patterns (most specific first)
    FILENAME_PATTERNS = [
        # ISO format: 2023-01-15.md
        (r"^(\d{4})-(\d{2})-(\d{2})(?:\s|_|\.)", "iso", 1.0),
        # Compact: 20230115.md
        (r"^(\d{4})(\d{2})(\d{2})(?:\s|_|\.)", "compact", 0.95),
        # US format: 01-15-2023.md
        (r"^(\d{2})-(\d{2})-(\d{4})(?:\s|_|\.)", "us", 0.9),
        # Month name: Jan-15-2023.md, January-15-2023.md
        (r"^([A-Za-z]+)-(\d{1,2})-(\d{4})(?:\s|_|\.)", "month_name", 0.9),
        # Month name reversed: 2023-Jan-15.md
        (r"^(\d{4})-([A-Za-z]+)-(\d{1,2})(?:\s|_|\.)", "month_name_rev", 0.9),
        # Year-month: 2023-01.md (assume first of month)
        (r"^(\d{4})-(\d{2})(?:\s|_|\.)", "year_month", 0.7),
        # Week number: 2023-W01.md
        (r"^(\d{4})-W(\d{2})(?:\s|_|\.)", "week", 0.6),
    ]

    # YAML frontmatter date field names (in priority order)
    FRONTMATTER_FIELDS = [
        "date",
        "created",
        "created_at",
        "createdAt",
        "published",
        "published_at",
        "publishedAt",
        "modified",
        "modified_at",
        "modifiedAt",
        "updated",
        "updated_at",
        "updatedAt",
        "timestamp",
    ]

    # Content date patterns (headers, first lines)
    CONTENT_PATTERNS = [
        # Header with full date: # January 15, 2023
        (r"^#\s*([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", "header_full", 0.85),
        # Header with ISO: # 2023-01-15
        (r"^#\s*(\d{4})-(\d{2})-(\d{2})", "header_iso", 0.9),
        # Date line: Date: January 15, 2023
        (r"^[Dd]ate:\s*([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", "date_line", 0.85),
        # Date line ISO: Date: 2023-01-15
        (r"^[Dd]ate:\s*(\d{4})-(\d{2})-(\d{2})", "date_line_iso", 0.9),
    ]

    # Month name mapping
    MONTHS = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }

    def __init__(self) -> None:
        # Compile patterns
        self._filename_patterns = [
            (re.compile(p), fmt, conf) for p, fmt, conf in self.FILENAME_PATTERNS
        ]
        self._content_patterns = [
            (re.compile(p, re.MULTILINE), fmt, conf)
            for p, fmt, conf in self.CONTENT_PATTERNS
        ]

    def extract(
        self,
        content: str,
        filename: Optional[str] = None,
        prefer_source: Optional[str] = None,
    ) -> Optional[ExtractedDate]:
        """
        Extract date from journal entry.

        Args:
            content: Journal entry content (may include YAML frontmatter)
            filename: Source filename
            prefer_source: Preferred source ('filename', 'frontmatter', 'content')

        Returns:
            ExtractedDate if found, None otherwise
        """
        candidates = self.extract_all(content, filename)

        if not candidates:
            return None

        # Sort by preference and confidence
        def sort_key(d: ExtractedDate) -> Tuple[int, float]:
            source_priority = {
                "frontmatter": 0,
                "filename": 1,
                "content": 2,
            }
            if prefer_source and d.source == prefer_source:
                return (-1, -d.confidence)
            return (source_priority.get(d.source, 3), -d.confidence)

        candidates.sort(key=sort_key)
        return candidates[0]

    def extract_all(
        self,
        content: str,
        filename: Optional[str] = None,
    ) -> List[ExtractedDate]:
        """
        Extract all dates from journal entry.

        Args:
            content: Journal entry content
            filename: Source filename

        Returns:
            List of all found dates
        """
        results = []

        # Extract from filename
        if filename:
            date_result = self._extract_from_filename(filename)
            if date_result:
                results.append(date_result)

        # Extract from frontmatter
        frontmatter_dates = self._extract_from_frontmatter(content)
        results.extend(frontmatter_dates)

        # Extract from content
        content_dates = self._extract_from_content(content)
        results.extend(content_dates)

        return results

    def _extract_from_filename(self, filename: str) -> Optional[ExtractedDate]:
        """
        Extract date from filename.

        Rule #1: Reduced nesting with helper method
        """
        # Get just the filename without path
        name = Path(filename).stem

        for pattern, fmt, confidence in self._filename_patterns:
            result = self._try_parse_filename_pattern(name, pattern, fmt, confidence)
            if result:
                return result

        return None

    def _try_parse_filename_pattern(
        self, name: str, pattern: Any, fmt: str, confidence: float
    ) -> Optional[ExtractedDate]:
        """
        Try to parse filename using a specific pattern.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        match = pattern.match(name)
        if not match:
            return None

        try:
            parsed_date = self._parse_filename_match(match, fmt)
            if parsed_date:
                return ExtractedDate(
                    date=parsed_date,
                    source="filename",
                    confidence=confidence,
                    raw_value=name,
                )
        except (ValueError, KeyError):
            return None

        return None

    def _parse_filename_match(self, match: re.Match[str], fmt: str) -> Optional[date]:
        """Parse a filename regex match into a date."""
        groups = match.groups()

        parsers = {
            "iso": lambda g: date(int(g[0]), int(g[1]), int(g[2])),
            "compact": lambda g: date(int(g[0]), int(g[1]), int(g[2])),
            "us": lambda g: date(int(g[2]), int(g[0]), int(g[1])),
            "month_name": lambda g: self._parse_month_name(g, 0, 2, 1),
            "month_name_rev": lambda g: self._parse_month_name(g, 1, 0, 2),
            "year_month": lambda g: date(int(g[0]), int(g[1]), 1),
            "week": lambda g: date.fromisocalendar(int(g[0]), int(g[1]), 1),
        }

        parser = parsers.get(fmt)
        return parser(groups) if parser else None

    def _parse_month_name(
        self, groups: tuple[str, ...], month_idx: int, year_idx: int, day_idx: int
    ) -> Optional[date]:
        """Parse date from month name format."""
        month = self.MONTHS.get(groups[month_idx].lower())
        return (
            date(int(groups[year_idx]), month, int(groups[day_idx])) if month else None
        )

    def _extract_from_frontmatter(self, content: str) -> List[ExtractedDate]:
        """
        Extract dates from YAML frontmatter.

        Rule #1: Reduced nesting with guard clauses and helper
        """
        if not content.startswith("---"):
            return []

        # Find frontmatter end
        end_match = re.search(r"\n---\s*\n", content[3:])
        if not end_match:
            return []

        frontmatter = content[3 : end_match.start() + 3]
        results = []
        for field in self.FRONTMATTER_FIELDS:
            date_result = self._parse_frontmatter_field(field, frontmatter)
            if date_result:
                results.append(date_result)

        return results

    def _parse_frontmatter_field(
        self, field: str, frontmatter: str
    ) -> Optional[ExtractedDate]:
        """
        Parse a specific date field from frontmatter.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        # Match field: value (handling various YAML formats)
        patterns = [
            rf'^{field}:\s*["\']?(\d{{4}}-\d{{2}}-\d{{2}})["\']?',  # ISO
            rf'^{field}:\s*["\']?(\d{{4}}-\d{{2}}-\d{{2}}T[^\s"\']+)["\']?',  # ISO with time
            rf'^{field}:\s*["\']?([A-Za-z]+\s+\d{{1,2}},?\s+\d{{4}})["\']?',  # Month Day, Year
            rf'^{field}:\s*["\']?(\d{{1,2}}/\d{{1,2}}/\d{{4}})["\']?',  # US slash format
        ]

        for pattern in patterns:
            match = re.search(pattern, frontmatter, re.MULTILINE | re.IGNORECASE)
            if not match:
                continue

            raw_value = match.group(1)
            parsed_date = self._parse_date_string(raw_value)
            if not parsed_date:
                continue

            # Higher confidence for 'date' field
            confidence = 1.0 if field == "date" else 0.9
            return ExtractedDate(
                date=parsed_date,
                source="frontmatter",
                confidence=confidence,
                raw_value=raw_value,
                field_name=field,
            )

        return None

    def _extract_from_content(self, content: str) -> List[ExtractedDate]:
        """
        Extract dates from content headers and first lines.

        Rule #1: Reduced nesting with helper method
        """
        # Skip frontmatter if present
        text = self._skip_frontmatter(content)
        results = []
        for pattern, fmt, confidence in self._content_patterns:
            date_result = self._try_parse_content_pattern(
                text, pattern, fmt, confidence
            )
            if date_result:
                results.append(date_result)

        return results

    def _skip_frontmatter(self, content: str) -> str:
        """
        Skip YAML frontmatter if present.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        if not content.startswith("---"):
            return content

        end_match = re.search(r"\n---\s*\n", content[3:])
        if end_match:
            return content[end_match.end() + 3 :]

        return content

    def _try_parse_content_pattern(
        self, text: str, pattern: Any, fmt: str, confidence: float
    ) -> Optional[ExtractedDate]:
        """
        Try to parse content using a specific pattern.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        match = pattern.search(text[:500])  # Only check first 500 chars
        if not match:
            return None

        try:
            parsed_date = self._parse_content_match(match, fmt)
            if parsed_date:
                return ExtractedDate(
                    date=parsed_date,
                    source="content",
                    confidence=confidence,
                    raw_value=match.group(0),
                )
        except (ValueError, KeyError):
            return None

        return None

    def _parse_content_match(self, match: re.Match[str], fmt: str) -> Optional[date]:
        """Parse a content regex match into a date."""
        groups = match.groups()

        if fmt in ("header_iso", "date_line_iso"):
            return date(int(groups[0]), int(groups[1]), int(groups[2]))

        elif fmt in ("header_full", "date_line"):
            month = self.MONTHS.get(groups[0].lower())
            if month:
                return date(int(groups[2]), month, int(groups[1]))

        return None

    def _parse_date_string(self, value: str) -> Optional[date]:
        """Parse a date string in various formats."""
        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%B %d, %Y",
            "%B %d %Y",
            "%b %d, %Y",
            "%b %d %Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(value.strip(), fmt)
                return dt.date()
            except ValueError:
                continue

        return None
