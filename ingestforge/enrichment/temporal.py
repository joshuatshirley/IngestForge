"""Temporal event extraction.

Extracts time-anchored events from text for timeline building."""

from __future__ import annotations

from typing import List, Dict, Any
import re
from datetime import datetime


class TemporalExtractor:
    """Extract temporal events from text."""

    def extract(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal events from chunk.

        Args:
            chunk: Chunk dictionary with 'text' field

        Returns:
            Enriched chunk with 'temporal_events' field
        """
        text = chunk.get("text", "")

        if not text:
            chunk["temporal_events"] = []
            return chunk

        # Extract events
        events = self._extract_events(text)

        # Sort by date
        events.sort(key=lambda x: x.get("normalized_date", ""))

        # Add to chunk
        chunk["temporal_events"] = events
        chunk["event_count"] = len(events)

        return chunk

    def _extract_events(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal events from text.

        Args:
            text: Input text

        Returns:
            List of event dictionaries
        """
        events = []

        # Pattern: "In YEAR, EVENT"
        year_events = self._extract_year_events(text)
        events.extend(year_events)

        # Pattern: "On DATE, EVENT"
        date_events = self._extract_date_events(text)
        events.extend(date_events)

        return events

    def _extract_year_events(self, text: str) -> List[Dict[str, Any]]:
        """Extract events with year markers.

        Args:
            text: Input text

        Returns:
            List of events
        """
        events = []

        # Pattern: "In YYYY" or "During YYYY"
        pattern = r"(?:In|During|By)\s+(\d{4}),?\s+([^.!?]+[.!?])"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            year = match.group(1)
            event_text = match.group(2).strip()

            events.append(
                {
                    "date": year,
                    "normalized_date": f"{year}-01-01",
                    "event": event_text,
                    "granularity": "year",
                    "position": match.start(),
                }
            )

        return events

    def _extract_date_events(self, text: str) -> List[Dict[str, Any]]:
        """Extract events with specific dates.

        Args:
            text: Input text

        Returns:
            List of events
        """
        events = []

        # Pattern: "On MONTH DAY, YEAR" or "MONTH DAY, YEAR"
        months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        pattern = (
            rf"(?:On\s+)?({months})\s+(\d{{1,2}}),?\s+(\d{{4}}),?\s+([^.!?]+[.!?])"
        )

        for match in re.finditer(pattern, text, re.IGNORECASE):
            month = match.group(1)
            day = match.group(2)
            year = match.group(3)
            event_text = match.group(4).strip()

            # Normalize date
            try:
                date_obj = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
                normalized = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                try:
                    date_obj = datetime.strptime(f"{month} {day} {year}", "%b %d %Y")
                    normalized = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    normalized = f"{year}-01-01"

            events.append(
                {
                    "date": f"{month} {day}, {year}",
                    "normalized_date": normalized,
                    "event": event_text,
                    "granularity": "day",
                    "position": match.start(),
                }
            )

        return events


def extract_temporal_events(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Extract temporal events from chunk.

    Args:
        chunk: Chunk dictionary

    Returns:
        Enriched chunk with temporal events
    """
    extractor = TemporalExtractor()
    return extractor.extract(chunk)


def build_timeline(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build timeline from chunks.

    Args:
        chunks: List of chunks with temporal events

    Returns:
        Sorted timeline of events
    """
    extractor = TemporalExtractor()

    # Collect all events
    all_events = []
    for chunk in chunks:
        enriched = extractor.extract(chunk)
        all_events.extend(enriched.get("temporal_events", []))

    # Sort by normalized date
    all_events.sort(key=lambda x: x.get("normalized_date", ""))

    return all_events
