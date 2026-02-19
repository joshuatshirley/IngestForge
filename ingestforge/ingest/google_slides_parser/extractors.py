"""
Element extraction helpers for Google Slides parsing.

Provides functions for extracting and parsing slide elements.
"""

from typing import Any, Dict, List, Optional, Tuple

from ingestforge.ingest.google_slides_parser.models import (
    SlideElement,
    SlideElementType,
    TextRun,
    TextStyle,
)


def extract_position_dimensions(
    elem_data: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    """
    Extract position and dimensions from element data.

    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        elem_data: Element data dictionary

    Returns:
        Tuple of (x, y, width, height)
    """
    # Get transform for position/size
    transform = elem_data.get("transform", {})
    x = _emu_to_points(transform.get("translateX", 0))
    y = _emu_to_points(transform.get("translateY", 0))
    width = _emu_to_points(transform.get("scaleX", 1) * 100)
    height = _emu_to_points(transform.get("scaleY", 1) * 100)

    # Override with explicit size if available
    size = elem_data.get("size", {})
    if size:
        width = _emu_to_points(size.get("width", {}).get("magnitude", 0))
        height = _emu_to_points(size.get("height", {}).get("magnitude", 0))

    return (x, y, width, height)


def parse_shape_element(
    shape: Dict[str, Any],
    object_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> SlideElement:
    """
    Parse shape element (text boxes, shapes).

    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        shape: Shape data dictionary
        object_id: Element object ID
        x, y, width, height: Position and dimensions

    Returns:
        SlideElement with shape/text data
    """
    shape_type = shape.get("shapeType", "")
    element = SlideElement(
        element_type=SlideElementType.TEXT
        if "TEXT" in shape_type
        else SlideElementType.SHAPE,
        object_id=object_id,
        x=x,
        y=y,
        width=width,
        height=height,
    )

    # Extract text content
    text_content = shape.get("text", {})
    text_runs, plain_text = parse_text_content(text_content)
    element.text_runs = text_runs
    element.plain_text = plain_text

    return element


def parse_table_element(
    table: Dict[str, Any],
    object_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> SlideElement:
    """
    Parse table element.

    Rule #1: Reduced nesting (max 2 levels)
    Rule #2: Fixed loop bound
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        table: Table data dictionary
        object_id: Element object ID
        x, y, width, height: Position and dimensions

    Returns:
        SlideElement with table data
    """
    rows = table.get("rows", 0)
    cols = table.get("columns", 0)

    # Extract table data
    table_data = []
    table_rows = table.get("tableRows", [])
    for row in table_rows:
        row_data = []
        cells = row.get("tableCells", [])
        for cell in cells:
            text_content = cell.get("text", {})
            _, plain_text = parse_text_content(text_content)
            row_data.append(plain_text)

        table_data.append(row_data)

    return SlideElement(
        element_type=SlideElementType.TABLE,
        object_id=object_id,
        x=x,
        y=y,
        width=width,
        height=height,
        table_rows=rows,
        table_cols=cols,
        table_data=table_data,
    )


def parse_element(elem_data: Dict[str, Any]) -> Optional[SlideElement]:
    """
    Parse a page element.

    Rule #1: Early return pattern
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        elem_data: Element data dictionary

    Returns:
        SlideElement or None if unrecognized type
    """
    object_id = elem_data.get("objectId", "")
    x, y, width, height = extract_position_dimensions(elem_data)
    if "shape" in elem_data:
        return parse_shape_element(elem_data["shape"], object_id, x, y, width, height)

    if "image" in elem_data:
        image = elem_data["image"]
        return SlideElement(
            element_type=SlideElementType.IMAGE,
            object_id=object_id,
            x=x,
            y=y,
            width=width,
            height=height,
            image_url=image.get("contentUrl"),
            image_content_type=image.get("sourceUrl"),
        )

    if "table" in elem_data:
        return parse_table_element(elem_data["table"], object_id, x, y, width, height)

    if "video" in elem_data:
        video = elem_data["video"]
        return SlideElement(
            element_type=SlideElementType.VIDEO,
            object_id=object_id,
            x=x,
            y=y,
            width=width,
            height=height,
            image_url=video.get("url"),
        )

    if "line" in elem_data:
        return SlideElement(
            element_type=SlideElementType.LINE,
            object_id=object_id,
            x=x,
            y=y,
            width=width,
            height=height,
        )

    return None


def parse_text_content(
    text_content: Dict[str, Any],
) -> Tuple[List[TextRun], str]:
    """Parse text content into runs and plain text."""
    text_runs = []
    plain_text_parts = []

    text_elements = text_content.get("textElements", [])

    for text_elem in text_elements:
        text_run = text_elem.get("textRun", {})
        content = text_run.get("content", "")

        if not content:
            continue

        # Parse style
        style_data = text_run.get("style", {})
        style = TextStyle(
            bold=style_data.get("bold", False),
            italic=style_data.get("italic", False),
            underline=style_data.get("underline", False),
            strikethrough=style_data.get("strikethrough", False),
            font_size=style_data.get("fontSize", {}).get("magnitude"),
            font_family=style_data.get("fontFamily"),
        )

        # Link
        link = style_data.get("link", {})
        if link:
            style.link_url = link.get("url")

        text_runs.append(TextRun(text=content, style=style))
        plain_text_parts.append(content)

    plain_text = "".join(plain_text_parts).strip()
    return text_runs, plain_text


def extract_text_from_shape(shape: Dict[str, Any]) -> str:
    """Extract plain text from a shape."""
    text_content = shape.get("text", {})
    _, plain_text = parse_text_content(text_content)
    return plain_text


def _emu_to_points(emu: float) -> float:
    """Convert EMU (English Metric Units) to points."""
    # 1 point = 12700 EMU
    return emu / 12700.0 if emu else 0.0
