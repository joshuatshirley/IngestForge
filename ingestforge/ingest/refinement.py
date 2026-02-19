"""
Pre-Chunking Text Refinement for IngestForge.

This module provides an interface and pipeline for refining extracted text
before it gets chunked. Refinement includes OCR cleanup, format normalization,
and chapter boundary detection.

Architecture Context
--------------------
Refinement sits between text extraction and chunking:

    PDF → PDFSplitter → TextExtractor → [TextRefiner] → SemanticChunker → Enrichers

The refiner pipeline cleans up text artifacts and detects structural markers
that help downstream chunking produce better semantic boundaries.

Available Refiners
------------------
- OCRCleanupRefiner: Fix ligatures, hyphenation, common OCR errors
- FormatNormalizer: Normalize unicode, whitespace, quotes, dashes
- ChapterDetector: Detect chapter/section boundaries for better chunking

Usage Example
-------------
    from ingestforge.ingest.refinement import TextRefinementPipeline
    from ingestforge.ingest.refiners import (
        OCRCleanupRefiner,
        FormatNormalizer,
        ChapterDetector,
    )

    pipeline = TextRefinementPipeline([
        OCRCleanupRefiner(),
        FormatNormalizer(),
        ChapterDetector(),
    ])

    result = pipeline.refine(extracted_text, metadata={"source": "pdf"})
    clean_text = result.refined
    chapters = result.chapter_markers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger


logger = get_logger(__name__)


class DocumentElementType(str, Enum):
    """Types of document elements for Unstructured-style classification.

    Used to classify text elements during processing for better retrieval
    weighting and content filtering.
    """

    TITLE = "Title"
    NARRATIVE_TEXT = "NarrativeText"
    LIST_ITEM = "ListItem"
    TABLE = "Table"
    HEADER = "Header"
    FOOTER = "Footer"
    CODE = "Code"
    IMAGE = "Image"
    UNCATEGORIZED = "Uncategorized"


@dataclass
class ChapterMarker:
    """Marks a chapter or section boundary in the text.

    Attributes:
        position: Character offset where the chapter/section starts
        title: The chapter or section title
        level: Hierarchy level (1=chapter, 2=section, 3=subsection, etc.)
    """

    position: int
    title: str
    level: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position,
            "title": self.title,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChapterMarker":
        """Create from dictionary."""
        return cls(
            position=data["position"],
            title=data["title"],
            level=data.get("level", 1),
        )


@dataclass
class RefinedText:
    """Result of text refinement.

    Attributes:
        original: The original input text
        refined: The cleaned/refined text
        changes: Log of changes made during refinement
        chapter_markers: Detected chapter/section boundaries
    """

    original: str
    refined: str
    changes: List[str] = field(default_factory=list)
    chapter_markers: List[ChapterMarker] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        """Check if any refinement was performed."""
        return self.original != self.refined

    @property
    def change_count(self) -> int:
        """Number of changes made."""
        return len(self.changes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_length": len(self.original),
            "refined_length": len(self.refined),
            "changes": self.changes,
            "chapter_markers": [m.to_dict() for m in self.chapter_markers],
            "was_modified": self.was_modified,
        }


class IRefiner(ABC):
    """Interface for text refiners.

    Refiners take extracted text and clean/enhance it before chunking.
    Each refiner should focus on a specific type of refinement.

    Implementing a Custom Refiner
    -----------------------------
        class SpellingCorrector(IRefiner):
            def refine(self, text: str, metadata: Dict) -> RefinedText:
                corrected = self._correct_spelling(text)
                changes = ["Corrected N spelling errors"]
                return RefinedText(
                    original=text,
                    refined=corrected,
                    changes=changes,
                )

            def is_available(self) -> bool:
                return True  # or check for required dependencies
    """

    @abstractmethod
    def refine(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RefinedText:
        """Refine the input text.

        Args:
            text: The text to refine
            metadata: Optional metadata about the text (source type, etc.)

        Returns:
            RefinedText with original, refined text, and change log
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the refiner is available and ready to use.

        Returns:
            True if the refiner can process text
        """
        pass

    @property
    def name(self) -> str:
        """Get the refiner name."""
        return self.__class__.__name__

    def get_metadata(self) -> Dict[str, Any]:
        """Get refiner metadata for logging."""
        return {
            "name": self.name,
            "available": self.is_available(),
        }

    def __repr__(self) -> str:
        available = "available" if self.is_available() else "unavailable"
        return f"{self.name}({available})"


class TextRefinementPipeline:
    """Pipeline for chaining multiple refiners together.

    Applies refiners in sequence, accumulating changes and chapter markers.
    Unavailable refiners are skipped by default.

    Examples:
        >>> pipeline = TextRefinementPipeline([
        ...     OCRCleanupRefiner(),
        ...     FormatNormalizer(),
        ...     ChapterDetector(),
        ... ])
        >>> result = pipeline.refine(messy_text)
        >>> print(f"Made {result.change_count} changes")
        >>> print(f"Found {len(result.chapter_markers)} chapters")
    """

    def __init__(
        self,
        refiners: List[IRefiner],
        skip_unavailable: bool = True,
    ):
        """Initialize refinement pipeline.

        Args:
            refiners: List of refiners to apply in order
            skip_unavailable: If True, skip refiners that aren't available
        """
        self.refiners = refiners
        self.skip_unavailable = skip_unavailable

        # Filter to available refiners
        if skip_unavailable:
            self.active_refiners = [r for r in refiners if r.is_available()]
        else:
            self.active_refiners = refiners

        if self.active_refiners:
            logger.debug(
                f"TextRefinementPipeline initialized with {len(self.active_refiners)} refiners: "
                f"{[r.name for r in self.active_refiners]}"
            )

    def refine(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RefinedText:
        """Apply all refiners to the text.

        Args:
            text: The text to refine
            metadata: Optional metadata about the text

        Returns:
            RefinedText with accumulated changes and chapter markers
        """
        if not text or not text.strip():
            return RefinedText(original=text, refined=text)

        if not self.active_refiners:
            return RefinedText(original=text, refined=text)

        metadata = metadata or {}
        original_text = text
        current_text = text
        all_changes: List[str] = []
        all_markers: List[ChapterMarker] = []

        for refiner in self.active_refiners:
            try:
                result = refiner.refine(current_text, metadata)
                current_text = result.refined
                all_changes.extend(result.changes)

                # Accumulate chapter markers (adjust positions if text changed)
                if result.chapter_markers:
                    all_markers.extend(result.chapter_markers)

            except Exception as e:
                logger.warning(f"Refiner {refiner.name} failed: {e}")
                if not self.skip_unavailable:
                    raise

        return RefinedText(
            original=original_text,
            refined=current_text,
            changes=all_changes,
            chapter_markers=all_markers,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline configuration."""
        return {
            "total_refiners": len(self.refiners),
            "active_refiners": len(self.active_refiners),
            "skip_unavailable": self.skip_unavailable,
            "refiners": [r.get_metadata() for r in self.active_refiners],
        }
