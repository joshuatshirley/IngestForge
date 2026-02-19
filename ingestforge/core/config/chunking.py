"""
Chunking and refinement configuration.

Provides configuration for semantic chunking strategies and text refinement
that occurs between extraction and chunking (OCR cleanup, formatting normalization).
"""

from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """Chunking configuration."""

    strategy: str = "semantic"  # semantic, fixed, paragraph, legal, code, header
    target_size: int = 300  # words
    min_size: int = 50
    max_size: int = 1000
    overlap: int = 50
    use_llm: bool = True
    # Layout-aware chunking (Unstructured-style )
    respect_section_boundaries: bool = True
    chunk_by_title: bool = False
    combine_text_under_n_chars: int = 200


@dataclass
class RefinementConfig:
    """Text refinement configuration.

    Text refinement happens between extraction and chunking to clean up
    OCR artifacts, normalize formatting, and detect chapter boundaries.
    """

    enabled: bool = True
    cleanup_ocr: bool = True
    normalize_formatting: bool = True
    detect_chapters: bool = True
    use_llm: bool = False  # Future: LLM-based enhancement
    # Text cleaning (Unstructured-style )
    group_paragraphs: bool = True
    clean_bullets: bool = True
    clean_prefix_postfix: bool = True
    # Element classification (Unstructured-style )
    classify_elements: bool = True
    detect_titles: bool = True
    detect_lists: bool = True
    detect_code: bool = True
    detect_tables: bool = True
