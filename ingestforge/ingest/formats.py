"""
Format handlers for different document types.

Registry of supported formats and their handlers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FormatHandler:
    """Handler information for a document format."""

    extension: str
    name: str
    mime_types: List[str]
    requires_split: bool
    extractor: Optional[str] = None


# Registry of supported formats
FORMAT_HANDLERS: Dict[str, FormatHandler] = {
    ".pdf": FormatHandler(
        extension=".pdf",
        name="PDF Document",
        mime_types=["application/pdf"],
        requires_split=True,
        extractor="pdf",
    ),
    ".epub": FormatHandler(
        extension=".epub",
        name="EPUB E-Book",
        mime_types=["application/epub+zip"],
        requires_split=False,
        extractor="epub",
    ),
    ".docx": FormatHandler(
        extension=".docx",
        name="Word Document",
        mime_types=[
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ],
        requires_split=False,
        extractor="docx",
    ),
    ".txt": FormatHandler(
        extension=".txt",
        name="Plain Text",
        mime_types=["text/plain"],
        requires_split=False,
        extractor="text",
    ),
    ".md": FormatHandler(
        extension=".md",
        name="Markdown",
        mime_types=["text/markdown", "text/x-markdown"],
        requires_split=False,
        extractor="markdown",
    ),
    ".markdown": FormatHandler(
        extension=".markdown",
        name="Markdown",
        mime_types=["text/markdown", "text/x-markdown"],
        requires_split=False,
        extractor="markdown",
    ),
    ".mdown": FormatHandler(
        extension=".mdown",
        name="Markdown",
        mime_types=["text/markdown"],
        requires_split=False,
        extractor="markdown",
    ),
    # LaTeX formats
    ".tex": FormatHandler(
        extension=".tex",
        name="LaTeX Document",
        mime_types=["application/x-latex", "text/x-latex"],
        requires_split=False,
        extractor="latex",
    ),
    ".latex": FormatHandler(
        extension=".latex",
        name="LaTeX Document",
        mime_types=["application/x-latex"],
        requires_split=False,
        extractor="latex",
    ),
    ".ltx": FormatHandler(
        extension=".ltx",
        name="LaTeX Document",
        mime_types=["application/x-latex"],
        requires_split=False,
        extractor="latex",
    ),
    # Jupyter notebooks
    ".ipynb": FormatHandler(
        extension=".ipynb",
        name="Jupyter Notebook",
        mime_types=["application/x-ipynb+json"],
        requires_split=False,
        extractor="jupyter",
    ),
}


def get_format_handler(file_path: Path) -> Optional[FormatHandler]:
    """Get format handler for a file."""
    suffix = file_path.suffix.lower()
    return FORMAT_HANDLERS.get(suffix)


def is_supported(file_path: Path) -> bool:
    """Check if file format is supported."""
    return get_format_handler(file_path) is not None


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return list(FORMAT_HANDLERS.keys())


def get_mime_type(file_path: Path) -> Optional[str]:
    """Get primary MIME type for a file."""
    handler = get_format_handler(file_path)
    if handler and handler.mime_types:
        return handler.mime_types[0]
    return None


def requires_splitting(file_path: Path) -> bool:
    """Check if file type requires splitting."""
    handler = get_format_handler(file_path)
    return handler.requires_split if handler else False
