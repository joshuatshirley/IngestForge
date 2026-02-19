"""OCR Confidence Evaluator for quality-based engine switching.

Analyzes OCR output confidence to detect poor quality regions
that may benefit from VLM escalation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from ingestforge.ingest.ocr.spatial_parser import (
    OCRDocument,
    OCRElement,
    OCRPage,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_BLOCKS_TO_EVALUATE = 500
MAX_ELEMENTS_PER_BLOCK = 100
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
MIN_CONFIDENCE_FOR_GOOD = 0.8
MIN_BLOCK_TEXT_LENGTH = 5


class QualityLevel(str, Enum):
    """Quality classification for OCR blocks."""

    EXCELLENT = "excellent"  # >= 0.9 confidence
    GOOD = "good"  # >= 0.8 confidence
    ACCEPTABLE = "acceptable"  # >= 0.6 confidence
    POOR = "poor"  # < 0.6 confidence - needs escalation
    UNKNOWN = "unknown"  # No confidence data


@dataclass
class BlockScore:
    """Confidence score for an OCR block."""

    element: OCRElement
    confidence: float
    quality: QualityLevel
    word_count: int = 0
    low_conf_word_count: int = 0
    needs_escalation: bool = False

    @property
    def escalation_reason(self) -> str:
        """Get reason for escalation if needed."""
        if not self.needs_escalation:
            return ""
        if self.confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            return f"Low confidence ({self.confidence:.2%})"
        if self.low_conf_word_count > self.word_count // 2:
            return f"Many low-confidence words ({self.low_conf_word_count}/{self.word_count})"
        return "Unknown quality issue"


@dataclass
class PageEvaluation:
    """Evaluation results for a page."""

    page_number: int
    block_scores: List[BlockScore] = field(default_factory=list)
    overall_confidence: float = 0.0
    overall_quality: QualityLevel = QualityLevel.UNKNOWN
    escalation_candidates: List[BlockScore] = field(default_factory=list)

    @property
    def needs_escalation(self) -> bool:
        """Check if any blocks need escalation."""
        return len(self.escalation_candidates) > 0

    @property
    def escalation_count(self) -> int:
        """Get number of blocks needing escalation."""
        return len(self.escalation_candidates)


@dataclass
class DocumentEvaluation:
    """Evaluation results for entire document."""

    page_evaluations: List[PageEvaluation] = field(default_factory=list)
    overall_confidence: float = 0.0
    overall_quality: QualityLevel = QualityLevel.UNKNOWN
    total_blocks: int = 0
    escalation_count: int = 0

    @property
    def needs_escalation(self) -> bool:
        """Check if document needs any escalation."""
        return self.escalation_count > 0

    @property
    def escalation_rate(self) -> float:
        """Get percentage of blocks needing escalation."""
        if self.total_blocks == 0:
            return 0.0
        return self.escalation_count / self.total_blocks


@dataclass
class EvaluatorConfig:
    """Configuration for confidence evaluator."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    min_text_length: int = MIN_BLOCK_TEXT_LENGTH
    include_empty_blocks: bool = False
    word_level_analysis: bool = True


class ConfidenceEvaluator:
    """Evaluates OCR confidence to identify poor quality regions.

    Analyzes block-level and word-level confidence scores to
    identify regions that should be escalated to VLM processing.
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        """Initialize evaluator.

        Args:
            config: Evaluator configuration
        """
        self.config = config or EvaluatorConfig()

    def evaluate_document(self, doc: OCRDocument) -> DocumentEvaluation:
        """Evaluate entire document for quality.

        Args:
            doc: OCR document to evaluate

        Returns:
            DocumentEvaluation with per-page results
        """
        evaluation = DocumentEvaluation()

        for page in doc.pages:
            page_eval = self.evaluate_page(page)
            evaluation.page_evaluations.append(page_eval)

        # Calculate document-level stats
        self._calculate_document_stats(evaluation)

        return evaluation

    def evaluate_page(self, page: OCRPage) -> PageEvaluation:
        """Evaluate a single page for quality.

        Args:
            page: OCR page to evaluate

        Returns:
            PageEvaluation with block scores
        """
        evaluation = PageEvaluation(page_number=page.page_number)

        blocks = self._get_evaluatable_blocks(page)

        for block in blocks[:MAX_BLOCKS_TO_EVALUATE]:
            score = self.evaluate_block(block)
            evaluation.block_scores.append(score)
            if score.needs_escalation:
                evaluation.escalation_candidates.append(score)

        # Calculate page-level stats
        self._calculate_page_stats(evaluation)

        return evaluation

    def evaluate_block(self, block: OCRElement) -> BlockScore:
        """Evaluate a single block for quality.

        Args:
            block: OCR element to evaluate

        Returns:
            BlockScore with confidence analysis
        """
        # Check text length
        if len(block.text.strip()) < self.config.min_text_length:
            if not self.config.include_empty_blocks:
                return BlockScore(
                    element=block,
                    confidence=0.0,
                    quality=QualityLevel.UNKNOWN,
                )

        # Get confidence from element
        confidence = block.confidence

        # Analyze word-level confidence if available
        word_count, low_conf_count = self._analyze_words(block)

        # Calculate effective confidence
        effective_conf = self._calculate_effective_confidence(
            confidence, word_count, low_conf_count
        )

        # Determine quality level
        quality = self._classify_quality(effective_conf)

        # Determine if escalation needed
        needs_escalation = self._check_escalation_needed(
            effective_conf, quality, low_conf_count, word_count
        )

        return BlockScore(
            element=block,
            confidence=effective_conf,
            quality=quality,
            word_count=word_count,
            low_conf_word_count=low_conf_count,
            needs_escalation=needs_escalation,
        )

    def _get_evaluatable_blocks(self, page: OCRPage) -> List[OCRElement]:
        """Get blocks suitable for evaluation.

        Args:
            page: OCR page

        Returns:
            List of blocks to evaluate
        """
        blocks = page.get_blocks()
        if not blocks:
            # Fall back to all elements
            blocks = page.elements

        return blocks

    def _analyze_words(self, block: OCRElement) -> Tuple[int, int]:
        """Analyze word-level confidence in a block.

        Args:
            block: Block to analyze

        Returns:
            Tuple of (total_words, low_confidence_words)
        """
        if not self.config.word_level_analysis:
            return (0, 0)

        words = self._get_words(block)
        total = len(words)
        low_conf = sum(
            1 for w in words if w.confidence < self.config.confidence_threshold
        )

        return (total, low_conf)

    def _get_words(self, element: OCRElement) -> List[OCRElement]:
        """Get word elements from block hierarchy.

        Args:
            element: Parent element

        Returns:
            List of word elements
        """
        words: List[OCRElement] = []

        for child in element.children[:MAX_ELEMENTS_PER_BLOCK]:
            if child.element_type.value == "word":
                words.append(child)
            # Check grandchildren (line -> word hierarchy)
            for grandchild in child.children[:MAX_ELEMENTS_PER_BLOCK]:
                if grandchild.element_type.value == "word":
                    words.append(grandchild)

        return words

    def _calculate_effective_confidence(
        self,
        block_conf: float,
        word_count: int,
        low_conf_count: int,
    ) -> float:
        """Calculate effective confidence considering word analysis.

        Args:
            block_conf: Block-level confidence
            word_count: Total word count
            low_conf_count: Low confidence word count

        Returns:
            Effective confidence score
        """
        if word_count == 0:
            return block_conf

        # Weight word confidence
        word_conf_ratio = 1.0 - (low_conf_count / word_count)

        # Average block and word confidence
        return (block_conf + word_conf_ratio) / 2.0

    def _classify_quality(self, confidence: float) -> QualityLevel:
        """Classify quality level from confidence.

        Args:
            confidence: Confidence score

        Returns:
            QualityLevel classification
        """
        if confidence >= 0.9:
            return QualityLevel.EXCELLENT
        if confidence >= MIN_CONFIDENCE_FOR_GOOD:
            return QualityLevel.GOOD
        if confidence >= self.config.confidence_threshold:
            return QualityLevel.ACCEPTABLE
        return QualityLevel.POOR

    def _check_escalation_needed(
        self,
        confidence: float,
        quality: QualityLevel,
        low_conf_count: int,
        word_count: int,
    ) -> bool:
        """Check if block should be escalated to VLM.

        Args:
            confidence: Effective confidence
            quality: Quality classification
            low_conf_count: Low confidence word count
            word_count: Total word count

        Returns:
            True if escalation recommended
        """
        # Clear case: below threshold
        if confidence < self.config.confidence_threshold:
            return True

        # Edge case: many low-confidence words
        if word_count > 0 and low_conf_count > word_count // 2:
            return True

        return quality == QualityLevel.POOR

    def _calculate_page_stats(self, evaluation: PageEvaluation) -> None:
        """Calculate aggregate page statistics.

        Args:
            evaluation: Page evaluation to update
        """
        valid_scores = [
            s for s in evaluation.block_scores if s.quality != QualityLevel.UNKNOWN
        ]

        if not valid_scores:
            return

        total_conf = sum(s.confidence for s in valid_scores)
        evaluation.overall_confidence = total_conf / len(valid_scores)
        evaluation.overall_quality = self._classify_quality(
            evaluation.overall_confidence
        )

    def _calculate_document_stats(self, evaluation: DocumentEvaluation) -> None:
        """Calculate aggregate document statistics.

        Args:
            evaluation: Document evaluation to update
        """
        all_scores: List[BlockScore] = []
        for page_eval in evaluation.page_evaluations:
            all_scores.extend(page_eval.block_scores)
            evaluation.escalation_count += page_eval.escalation_count

        valid_scores = [s for s in all_scores if s.quality != QualityLevel.UNKNOWN]

        evaluation.total_blocks = len(all_scores)

        if not valid_scores:
            return

        total_conf = sum(s.confidence for s in valid_scores)
        evaluation.overall_confidence = total_conf / len(valid_scores)
        evaluation.overall_quality = self._classify_quality(
            evaluation.overall_confidence
        )


def evaluate_ocr_quality(doc: OCRDocument) -> DocumentEvaluation:
    """Convenience function to evaluate document quality.

    Args:
        doc: OCR document

    Returns:
        DocumentEvaluation
    """
    evaluator = ConfidenceEvaluator()
    return evaluator.evaluate_document(doc)


def get_escalation_candidates(
    doc: OCRDocument, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> List[OCRElement]:
    """Get blocks that need VLM escalation.

    Args:
        doc: OCR document
        threshold: Confidence threshold

    Returns:
        List of elements needing escalation
    """
    config = EvaluatorConfig(confidence_threshold=threshold)
    evaluator = ConfidenceEvaluator(config=config)
    evaluation = evaluator.evaluate_document(doc)

    candidates: List[OCRElement] = []
    for page_eval in evaluation.page_evaluations:
        for score in page_eval.escalation_candidates:
            candidates.append(score.element)

    return candidates
