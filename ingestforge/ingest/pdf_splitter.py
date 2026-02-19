"""
PDF splitting based on Table of Contents.

Splits PDFs into chapters/sections based on TOC structure.
Adapted from SplitAnalyze split_pdf.py.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChapterInfo:
    """Information about a split chapter."""

    path: Path
    title: str
    start_page: int
    end_page: int
    page_count: int


def sanitize_filename(text: str) -> str:
    """Sanitize text for safe filename usage."""
    if not text:
        return "Untitled"
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "", text.replace(" ", "_"))
    return clean[:50]


class PDFSplitter:
    """
    Split PDFs into chapters based on Table of Contents.

    Features:
    - Automatic TOC detection
    - Deep split option (include subsections)
    - Page range tracking
    - Metadata extraction
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.last_page_count = 0
        self._fitz = None

    @property
    def fitz(self) -> Any:
        """Lazy-load PyMuPDF."""
        if self._fitz is None:
            try:
                import fitz

                fitz.TOOLS.mupdf_display_errors(False)
                self._fitz = fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF is required for PDF processing. "
                    "Install with: pip install pymupdf"
                )
        return self._fitz

    def _get_split_boundaries(
        self, toc: list[Any], deep_split: bool
    ) -> List[Tuple[int, str]]:
        """
        Identify split points from TOC.

        Args:
            toc: Table of contents from PyMuPDF
            deep_split: Include subsections if True

        Returns:
            List of (page_index, title) tuples
        """
        if not toc:
            return []

        chapters = []
        seen_pages = set()

        for entry in toc:
            level, title, page_num = entry[0], entry[1], entry[2]

            # Skip subsections unless deep_split enabled
            if not deep_split and level > 1:
                continue

            page_index = page_num - 1
            if page_index < 0:
                continue

            if page_index not in seen_pages:
                chapters.append((page_index, title))
                seen_pages.add(page_index)

        return sorted(chapters, key=lambda x: x[0])

    def _extract_subset_toc(
        self, full_toc: list[Any], start_page: int, end_page: int
    ) -> list[Any]:
        """Extract TOC entries for a page range, adjusting page numbers."""
        subset_toc = []
        for level, title, page_num in full_toc:
            if start_page < page_num <= end_page:
                new_page = page_num - start_page
                subset_toc.append([level, title, new_page])
        return subset_toc

    def _open_and_validate_pdf(self, source_path: Path) -> tuple:
        """
        Open and validate PDF for splitting.

        Rule #1: Early return for errors
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            source_path: Path to PDF file

        Returns:
            Tuple of (doc, total_pages, full_toc)

        Raises:
            RuntimeError: If PDF cannot be opened
            ValueError: If PDF is encrypted
        """
        try:
            doc = self.fitz.open(source_path)
            if doc.is_encrypted:
                raise ValueError("PDF is encrypted. Decryption required.")

        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}")

        total_pages = doc.page_count
        self.last_page_count = total_pages
        full_toc = doc.get_toc(simple=True)

        return (doc, total_pages, full_toc)

    def _determine_boundaries(self, full_toc: list[Any]) -> List[Tuple[int, str]]:
        """
        Determine split boundaries from TOC.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            full_toc: Full table of contents

        Returns:
            List of (page_num, title) tuples

        Raises:
            ValueError: If no boundaries found and fallback disabled
        """
        deep_split = self.config.split.deep_split
        boundaries = self._get_split_boundaries(full_toc, deep_split)
        if not boundaries:
            if self.config.split.fallback_single_file:
                logger.warning("No TOC found, processing as single document")
                return [(0, "Full_Document")]
            raise ValueError("No TOC found and fallback disabled")

        # Ensure we start at page 0
        if boundaries[0][0] != 0:
            boundaries.insert(0, (0, "Front_Matter"))

        return boundaries

    def _create_chapter_pdf(
        self,
        doc: Any,
        full_toc: list[Any],
        start: int,
        end: int,
        title: str,
        chapter_num: int,
        output_dir: Path,
    ) -> Optional[Path]:
        """
        Create single chapter PDF.

        Rule #4: Reduced from 64 → 45 lines (shortened docstring)
        """
        if start >= end or (end - start) < 1:
            return None

        safe_name = sanitize_filename(title)
        filename = f"{chapter_num:02d}_{safe_name}.pdf"
        out_path = output_dir / filename

        try:
            new_doc = self.fitz.open()
            new_doc.insert_pdf(
                doc, from_page=start, to_page=end - 1, links=True, annots=True
            )
            new_doc.set_metadata(doc.metadata)

            # Add subset TOC
            if full_toc:
                chapter_toc = self._extract_subset_toc(full_toc, start, end)
                if chapter_toc:
                    new_doc.set_toc(chapter_toc)

            new_doc.save(out_path, garbage=4, deflate=True)
            new_doc.close()

            logger.info(f"Created: {filename} (Pages {start + 1}-{end})")
            return out_path

        except Exception as e:
            logger.error(f"Failed to write {filename}: {e}")
            return None

    def _save_chapter_metadata(
        self, chapter_metadata: dict[str, Any], output_dir: Path
    ) -> None:
        """
        Save chapter metadata to JSON file.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            chapter_metadata: Dictionary of chapter metadata
            output_dir: Output directory
        """
        metadata_path = output_dir / "chapter_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chapter_metadata, f, indent=2, ensure_ascii=False)

    def _process_chapters(
        self,
        doc: Any,
        full_toc: list[Any],
        boundaries: list,
        total_pages: int,
        output_dir: Path,
    ) -> tuple[List[Path], dict]:
        """Process all chapter boundaries and create PDFs.

        Rule #4: No large functions - Extracted from split

        Returns:
            Tuple of (chapter_paths, chapter_metadata)
        """
        chapter_paths = []
        chapter_metadata = {}

        for i, (start, title) in enumerate(boundaries):
            # Calculate end page
            end = total_pages if i == len(boundaries) - 1 else boundaries[i + 1][0]

            # Create chapter PDF
            out_path = self._create_chapter_pdf(
                doc, full_toc, start, end, title, i, output_dir
            )
            if out_path is None:
                continue

            chapter_paths.append(out_path)

            # Store metadata
            chapter_metadata[out_path.name] = {
                "start_page": start + 1,
                "end_page": end,
                "page_count": end - start,
                "title": title,
            }

        return chapter_paths, chapter_metadata

    def split(
        self,
        source_path: Path,
        document_id: str,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Split a PDF into chapters.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines (refactored to 39 lines)
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            source_path: Path to source PDF
            document_id: Unique document identifier
            output_dir: Output directory. Defaults to config data path.

        Returns:
            List of paths to split chapter PDFs
        """
        # Setup output directory
        if output_dir is None:
            output_dir = self.config.data_path / "split" / document_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Splitting: {source_path.name}")

        # Open and validate PDF
        doc, total_pages, full_toc = self._open_and_validate_pdf(source_path)

        # Determine split boundaries
        boundaries = self._determine_boundaries(full_toc)

        # Execute split
        logger.info(f"Splitting into {len(boundaries)} chapters")
        chapter_paths, chapter_metadata = self._process_chapters(
            doc, full_toc, boundaries, total_pages, output_dir
        )

        # Save chapter metadata
        self._save_chapter_metadata(chapter_metadata, output_dir)

        doc.close()
        logger.info(f"Split complete: {len(chapter_paths)} chapters")

        return chapter_paths

    def get_metadata(self, pdf_path: Path) -> dict[str, Any]:
        """Extract metadata from PDF."""
        doc = self.fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = doc.page_count
        toc = doc.get_toc(simple=True)
        doc.close()

        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "page_count": page_count,
            "toc_entries": len(toc),
            "has_toc": len(toc) > 0,
        }
