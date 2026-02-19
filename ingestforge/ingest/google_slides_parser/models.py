"""
Core data models for Google Slides parsing.

Provides data structures for slides, elements, and text styling.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SlideElementType(Enum):
    """Types of elements in a slide."""

    TEXT = "text"
    IMAGE = "image"
    SHAPE = "shape"
    TABLE = "table"
    VIDEO = "video"
    CHART = "chart"
    LINE = "line"
    GROUP = "group"
    SPEAKER_NOTES = "speaker_notes"


@dataclass
class TextStyle:
    """Text styling information."""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    foreground_color: Optional[str] = None
    background_color: Optional[str] = None
    link_url: Optional[str] = None


@dataclass
class TextRun:
    """A run of text with consistent styling."""

    text: str
    style: TextStyle = field(default_factory=TextStyle)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "style": asdict(self.style),
        }


@dataclass
class SlideElement:
    """An element within a slide."""

    element_type: SlideElementType
    object_id: str
    # Position and size
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    # Content based on type
    text_runs: List[TextRun] = field(default_factory=list)
    plain_text: str = ""
    image_url: Optional[str] = None
    image_content_type: Optional[str] = None
    # Table data
    table_rows: int = 0
    table_cols: int = 0
    table_data: List[List[str]] = field(default_factory=list)
    # Metadata
    alt_text: Optional[str] = None
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type.value,
            "object_id": self.object_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "text_runs": [tr.to_dict() for tr in self.text_runs],
            "plain_text": self.plain_text,
            "image_url": self.image_url,
            "table_rows": self.table_rows,
            "table_cols": self.table_cols,
            "table_data": self.table_data,
            "alt_text": self.alt_text,
            "title": self.title,
        }


@dataclass
class ParsedSlide:
    """A parsed slide from a Google Slides presentation."""

    slide_id: str
    index: int
    # Layout info
    layout_id: Optional[str] = None
    layout_name: Optional[str] = None
    # Content
    elements: List[SlideElement] = field(default_factory=list)
    speaker_notes: str = ""
    # Extracted text (convenience)
    title: Optional[str] = None
    body_text: str = ""
    # Background
    background_color: Optional[str] = None
    background_image_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "index": self.index,
            "layout_id": self.layout_id,
            "layout_name": self.layout_name,
            "elements": [e.to_dict() for e in self.elements],
            "speaker_notes": self.speaker_notes,
            "title": self.title,
            "body_text": self.body_text,
            "background_color": self.background_color,
            "background_image_url": self.background_image_url,
        }
