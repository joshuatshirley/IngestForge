"""
Parsed presentation model with conversion methods.

Provides ParsedPresentation for representing fully parsed Google Slides.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ingestforge.ingest.google_slides_parser.models import (
    ParsedSlide,
    SlideElementType,
    TextRun,
)


@dataclass
class ParsedPresentation:
    """A fully parsed Google Slides presentation."""

    presentation_id: str
    title: str
    locale: str = "en"
    # Size
    page_width: float = 0.0
    page_height: float = 0.0
    # Content
    slides: List[ParsedSlide] = field(default_factory=list)
    # Metadata
    revision_id: Optional[str] = None
    notes_master_id: Optional[str] = None
    # Extracted data
    all_text: str = ""
    all_speaker_notes: str = ""
    image_urls: List[str] = field(default_factory=list)
    # Timestamps
    fetched_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "presentation_id": self.presentation_id,
            "title": self.title,
            "locale": self.locale,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "slides": [s.to_dict() for s in self.slides],
            "revision_id": self.revision_id,
            "all_text": self.all_text,
            "all_speaker_notes": self.all_speaker_notes,
            "image_urls": self.image_urls,
            "fetched_at": self.fetched_at,
        }

    def to_text(self, include_notes: bool = True) -> str:
        """Convert to plain text."""
        lines = [f"# {self.title}", ""]

        for slide in self.slides:
            lines.append(f"## Slide {slide.index + 1}")
            if slide.title:
                lines.append(f"### {slide.title}")
            if slide.body_text:
                lines.append(slide.body_text)
            if include_notes and slide.speaker_notes:
                lines.append(f"\n[Speaker Notes: {slide.speaker_notes}]")
            lines.append("")

        return "\n".join(lines)

    def to_markdown(self, include_notes: bool = True) -> str:
        """Convert to Markdown format."""
        lines = [f"# {self.title}", "", "---", ""]

        for slide in self.slides:
            lines.append(f"## Slide {slide.index + 1}")
            if slide.title:
                lines.append(f"### {slide.title}")
                lines.append("")

            # Process all elements in the slide
            element_lines = self._process_slide_elements(slide)
            lines.extend(element_lines)

            if include_notes and slide.speaker_notes:
                lines.append(f"> **Speaker Notes:** {slide.speaker_notes}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _process_slide_elements(self, slide: ParsedSlide) -> List[str]:
        """
        Process all elements in a slide to markdown lines.

        Rule #1: Dictionary dispatch eliminates nesting
        """
        lines = []

        for element in slide.elements:
            element_lines = self._process_single_element(element)
            lines.extend(element_lines)

        return lines

    def _process_single_element(self, element: Any) -> List[str]:
        """
        Process a single slide element to markdown.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        handlers = {
            SlideElementType.TEXT: self._process_text_element,
            SlideElementType.IMAGE: self._process_image_element,
            SlideElementType.TABLE: self._process_table_element,
        }

        handler = handlers.get(element.element_type)
        if handler:
            return handler(element)

        return []

    def _process_text_element(self, element: Any) -> List[str]:
        """Process text element."""
        if not element.plain_text:
            return []
        md_text = self._text_runs_to_markdown(element.text_runs)
        return [md_text, ""]

    def _process_image_element(self, element: Any) -> List[str]:
        """Process image element."""
        if not element.image_url:
            return []
        alt = element.alt_text or "Image"
        return [f"![{alt}]({element.image_url})", ""]

    def _process_table_element(self, element: Any) -> List[str]:
        """Process table element."""
        return [self._table_to_markdown(element.table_data), ""]

    def _text_runs_to_markdown(self, runs: List[TextRun]) -> str:
        """Convert text runs to markdown."""
        result: list[Any] = []
        for run in runs:
            text = run.text
            style = run.style

            if style.bold:
                text = f"**{text}**"
            if style.italic:
                text = f"*{text}*"
            if style.strikethrough:
                text = f"~~{text}~~"
            if style.link_url:
                text = f"[{text}]({style.link_url})"

            result.append(text)

        return "".join(result)

    def _table_to_markdown(self, data: List[List[str]]) -> str:
        """Convert table data to markdown."""
        if not data:
            return ""

        lines = []
        # Header
        lines.append("| " + " | ".join(data[0]) + " |")
        lines.append("| " + " | ".join(["---"] * len(data[0])) + " |")
        # Rows
        for row in data[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
