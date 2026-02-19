"""OCR processing utilities.

This package provides OCR-related functionality:
- batch_aggregator: Merge multiple scans into one document (OCR-003.2)
- cleanup_core: Image deskew and binarization (OCR-002.1)
- spatial_parser: hOCR/ALTO XML parsing (OCR-001.1)
- sequencer: Multi-column reading order (OCR-001.2)
- table_builder: Table reconstruction (OCR-001.3)
- confidence_eval: OCR quality analysis (OCR-004.1)
- vlm_escalator: VLM escalation provider (OCR-004.2)
- handwriting_preprocessor: Handwriting detection and preprocessing (P3-SOURCE-006.2)

Usage:
    from ingestforge.ingest.ocr import (
        # Spatial parsing
        parse_ocr_file,
        BoundingBox,
        OCRDocument,
        # Reading order
        sequence_ocr_document,
        detect_column_layout,
        # Tables
        find_tables_on_page,
        table_to_markdown,
        # Handwriting
        HandwritingPreprocessor,
        preprocess_handwriting,
    )
"""

from __future__ import annotations

from ingestforge.ingest.ocr.batch_aggregator import (
    BatchAggregator,
    ImageBatch,
    AggregatedDocument,
    DEFAULT_MAX_BATCH_SIZE,
)

from ingestforge.ingest.ocr.cleanup_core import (
    BinarizationMethod,
    PreprocessingConfig,
    PreprocessingResult,
    ImagePreprocessor,
    preprocess_image,
    preprocess_batch,
)

from ingestforge.ingest.ocr.spatial_parser import (
    ALTOParser,
    BoundingBox,
    ElementType,
    HOCRParser,
    OCRDocument,
    OCRElement,
    OCRPage,
    parse_ocr_file,
)

from ingestforge.ingest.ocr.sequencer import (
    Column,
    LayoutType,
    MultiColumnSequencer,
    ReadingOrder,
    SequencerConfig,
    detect_column_layout,
    sequence_ocr_document,
)

from ingestforge.ingest.ocr.table_builder import (
    CellType,
    Table,
    TableBuilder,
    TableBuilderConfig,
    TableCell,
    TableRow,
    find_tables_on_page,
    table_to_markdown,
)

from ingestforge.ingest.ocr.confidence_eval import (
    BlockScore,
    ConfidenceEvaluator,
    DocumentEvaluation,
    EvaluatorConfig,
    PageEvaluation,
    QualityLevel,
    evaluate_ocr_quality,
    get_escalation_candidates,
)

from ingestforge.ingest.ocr.vlm_escalator import (
    ConsentStatus,
    CropRegion,
    EscalationRequest,
    EscalationResult,
    EscalatorConfig,
    VLMEscalator,
    VLMProvider,
    create_escalator,
    escalate_elements,
    escalate_low_confidence_ocr,
    identify_low_confidence_elements,
)

from ingestforge.ingest.ocr.handwriting_preprocessor import (
    HandwritingMethod,
    HandwritingPreprocessor,
    HandwritingRegion,
    PreprocessorConfig as HandwritingConfig,
    PreprocessorResult as HandwritingResult,
    ThresholdMethod,
    VLM_ESCALATION_THRESHOLD,
    detect_handwriting_regions,
    get_vlm_escalation_candidates,
    preprocess_handwriting,
)

from ingestforge.ingest.ocr.bbox_bridge import (
    BoundingBoxBridge,
    ChunkBoundingBox,
    bbox_to_metadata,
    extract_bbox_from_elements,
)

__all__ = [
    # Batch aggregator (OCR-003.2)
    "BatchAggregator",
    "ImageBatch",
    "AggregatedDocument",
    "DEFAULT_MAX_BATCH_SIZE",
    # Cleanup core (OCR-002.1)
    "BinarizationMethod",
    "PreprocessingConfig",
    "PreprocessingResult",
    "ImagePreprocessor",
    "preprocess_image",
    "preprocess_batch",
    # Spatial parser (OCR-001.1)
    "ALTOParser",
    "BoundingBox",
    "ElementType",
    "HOCRParser",
    "OCRDocument",
    "OCRElement",
    "OCRPage",
    "parse_ocr_file",
    # Sequencer (OCR-001.2)
    "Column",
    "LayoutType",
    "MultiColumnSequencer",
    "ReadingOrder",
    "SequencerConfig",
    "detect_column_layout",
    "sequence_ocr_document",
    # Table builder (OCR-001.3)
    "CellType",
    "Table",
    "TableBuilder",
    "TableBuilderConfig",
    "TableCell",
    "TableRow",
    "find_tables_on_page",
    "table_to_markdown",
    # Confidence evaluator (OCR-004.1)
    "BlockScore",
    "ConfidenceEvaluator",
    "DocumentEvaluation",
    "EvaluatorConfig",
    "PageEvaluation",
    "QualityLevel",
    "evaluate_ocr_quality",
    "get_escalation_candidates",
    # VLM escalator (OCR-004.2)
    "ConsentStatus",
    "CropRegion",
    "EscalationRequest",
    "EscalationResult",
    "EscalatorConfig",
    "VLMEscalator",
    "VLMProvider",
    "create_escalator",
    "escalate_elements",
    "escalate_low_confidence_ocr",
    "identify_low_confidence_elements",
    # Handwriting preprocessor (P3-SOURCE-006.2)
    "HandwritingMethod",
    "HandwritingPreprocessor",
    "HandwritingRegion",
    "HandwritingConfig",
    "HandwritingResult",
    "ThresholdMethod",
    "VLM_ESCALATION_THRESHOLD",
    "detect_handwriting_regions",
    "get_vlm_escalation_candidates",
    "preprocess_handwriting",
    # Bounding Box Bridge (Unstructured-style)
    "BoundingBoxBridge",
    "ChunkBoundingBox",
    "bbox_to_metadata",
    "extract_bbox_from_elements",
]
