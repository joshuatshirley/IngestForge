"""
Google Slides integration for parsing and importing presentations.

Supports:
- Fetching presentations via Google Slides API
- Extracting slides, speaker notes, and images
- Converting to various output formats
- OAuth2 authentication flow
"""

from typing import Any, Dict, Optional

# Models
from ingestforge.ingest.google_slides_parser.models import (
    ParsedSlide,
    SlideElement,
    SlideElementType,
    TextRun,
    TextStyle,
)

# Presentation
from ingestforge.ingest.google_slides_parser.presentation import ParsedPresentation

# Parser
from ingestforge.ingest.google_slides_parser.parser import GoogleSlidesParser

# Extractors
from ingestforge.ingest.google_slides_parser.extractors import (
    extract_position_dimensions,
    extract_text_from_shape,
    parse_element,
    parse_shape_element,
    parse_table_element,
    parse_text_content,
)

__all__ = [
    # Models
    "SlideElementType",
    "TextStyle",
    "TextRun",
    "SlideElement",
    "ParsedSlide",
    # Presentation
    "ParsedPresentation",
    # Parser
    "GoogleSlidesParser",
    # Extractors
    "extract_position_dimensions",
    "extract_text_from_shape",
    "parse_element",
    "parse_shape_element",
    "parse_table_element",
    "parse_text_content",
    # Convenience functions
    "parse_slides_json",
    "extract_presentation_id",
    "slides_to_text",
    "slides_to_markdown",
]


# Convenience functions
def parse_slides_json(api_response: Dict[str, Any]) -> ParsedPresentation:
    """Parse a Google Slides API response."""
    parser = GoogleSlidesParser()
    return parser.parse_from_json(api_response)


def extract_presentation_id(url_or_id: str) -> Optional[str]:
    """Extract presentation ID from URL or validate ID."""
    return GoogleSlidesParser.extract_presentation_id(url_or_id)


def slides_to_text(
    api_response: Dict[str, Any],
    include_notes: bool = True,
) -> str:
    """Convert API response to plain text."""
    presentation = parse_slides_json(api_response)
    return presentation.to_text(include_notes)


def slides_to_markdown(
    api_response: Dict[str, Any],
    include_notes: bool = True,
) -> str:
    """Convert API response to Markdown."""
    presentation = parse_slides_json(api_response)
    return presentation.to_markdown(include_notes)
