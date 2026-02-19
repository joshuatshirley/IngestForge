"""
Built-in Text Refiners for IngestForge.

This package provides standard refiners for common text cleanup tasks:

- OCRCleanupRefiner: Fix OCR artifacts (ligatures, hyphenation, common errors)
- FormatNormalizer: Normalize unicode, whitespace, quotes, dashes
- ChapterDetector: Detect chapter/section boundaries
- PIIRedactor: Redact personally identifiable information (SEC-001.1)

Usage
-----
    from ingestforge.ingest.refiners import (
        OCRCleanupRefiner,
        FormatNormalizer,
        ChapterDetector,
        PIIRedactor,
        redact_pii,
    )
    from ingestforge.ingest.refinement import TextRefinementPipeline

    pipeline = TextRefinementPipeline([
        OCRCleanupRefiner(),
        FormatNormalizer(),
        ChapterDetector(),
    ])

    result = pipeline.refine(extracted_text)

    # PII Redaction
    redactor = PIIRedactor()
    redacted = redactor.redact(text)
"""

from ingestforge.ingest.refiners.ocr_cleanup import OCRCleanupRefiner
from ingestforge.ingest.refiners.format_normalizer import FormatNormalizer
from ingestforge.ingest.refiners.chapter_detector import ChapterDetector
from ingestforge.ingest.refiners.redaction import (
    PIIType,
    RedactionMatch,
    RedactionResult,
    RedactionConfig,
    PIIRedactor,
    redact_pii,
    redact_batch,
)
from ingestforge.ingest.refiners.text_cleaners import (
    TextCleanerRefiner,
    clean_bullets,
    clean_prefix_postfix,
    group_broken_paragraphs,
)
from ingestforge.ingest.refiners.element_classifier import (
    ElementClassifier,
    ClassifiedElement,
    classify_elements,
    get_element_type,
)

__all__ = [
    "OCRCleanupRefiner",
    "FormatNormalizer",
    "ChapterDetector",
    # PII Redaction (SEC-001.1)
    "PIIType",
    "RedactionMatch",
    "RedactionResult",
    "RedactionConfig",
    "PIIRedactor",
    "redact_pii",
    "redact_batch",
    # Text Cleaners (Unstructured-style)
    "TextCleanerRefiner",
    "clean_bullets",
    "clean_prefix_postfix",
    "group_broken_paragraphs",
    # Element Classifier (Unstructured-style)
    "ElementClassifier",
    "ClassifiedElement",
    "classify_elements",
    "get_element_type",
]
