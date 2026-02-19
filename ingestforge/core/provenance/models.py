"""
Core data models for provenance tracking.

Provides enumerations for source types and citation styles.
"""

from enum import Enum


class SourceType(str, Enum):
    """Type of source document."""

    PDF = "pdf"
    WEBPAGE = "webpage"
    EPUB = "epub"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"  # Audio files (mp3, wav, m4a, etc.)
    SLIDES = "slides"
    IMAGE = "image"
    CODE = "code"  # Source code files (Apex, LWC, etc.)
    ADO_WORK_ITEM = "ado_work_item"  # Azure DevOps work items
    UNKNOWN = "unknown"


class CitationStyle(str, Enum):
    """Supported citation styles."""

    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    BIBTEX = "bibtex"
