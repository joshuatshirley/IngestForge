"""Tests for OCR confidence evaluation.

Tests quality analysis and escalation candidate detection."""

from __future__ import annotations


from ingestforge.ingest.ocr.spatial_parser import (
    BoundingBox,
    ElementType,
    OCRDocument,
    OCRElement,
    OCRPage,
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

# QualityLevel tests


class TestQualityLevel:
    """Tests for QualityLevel enum."""

    def test_quality_levels_defined(self) -> None:
        """Test all quality levels are defined."""
        levels = [q.value for q in QualityLevel]

        assert "excellent" in levels
        assert "good" in levels
        assert "acceptable" in levels
        assert "poor" in levels
        assert "unknown" in levels


# BlockScore tests


class TestBlockScore:
    """Tests for BlockScore dataclass."""

    def test_score_creation(self) -> None:
        """Test creating a block score."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="Test text",
            confidence=0.95,
        )

        score = BlockScore(
            element=elem,
            confidence=0.95,
            quality=QualityLevel.EXCELLENT,
        )

        assert score.confidence == 0.95
        assert score.quality == QualityLevel.EXCELLENT
        assert score.needs_escalation is False

    def test_escalation_reason_low_confidence(self) -> None:
        """Test escalation reason for low confidence."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="Test",
            confidence=0.5,
        )

        score = BlockScore(
            element=elem,
            confidence=0.5,
            quality=QualityLevel.POOR,
            needs_escalation=True,
        )

        reason = score.escalation_reason
        assert "Low confidence" in reason
        assert "50" in reason

    def test_escalation_reason_many_low_words(self) -> None:
        """Test escalation reason for many low-confidence words."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="Test",
            confidence=0.7,
        )

        score = BlockScore(
            element=elem,
            confidence=0.7,
            quality=QualityLevel.ACCEPTABLE,
            word_count=10,
            low_conf_word_count=6,
            needs_escalation=True,
        )

        reason = score.escalation_reason
        assert "low-confidence words" in reason

    def test_no_escalation_reason_when_not_needed(self) -> None:
        """Test no reason when escalation not needed."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="Test",
            confidence=0.9,
        )

        score = BlockScore(
            element=elem,
            confidence=0.9,
            quality=QualityLevel.EXCELLENT,
            needs_escalation=False,
        )

        assert score.escalation_reason == ""


# PageEvaluation tests


class TestPageEvaluation:
    """Tests for PageEvaluation dataclass."""

    def test_evaluation_creation(self) -> None:
        """Test creating page evaluation."""
        evaluation = PageEvaluation(page_number=1)

        assert evaluation.page_number == 1
        assert len(evaluation.block_scores) == 0
        assert evaluation.needs_escalation is False

    def test_needs_escalation_with_candidates(self) -> None:
        """Test needs_escalation with candidates."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")

        score = BlockScore(
            element=elem,
            confidence=0.4,
            quality=QualityLevel.POOR,
            needs_escalation=True,
        )

        evaluation = PageEvaluation(
            page_number=1,
            escalation_candidates=[score],
        )

        assert evaluation.needs_escalation is True
        assert evaluation.escalation_count == 1


# DocumentEvaluation tests


class TestDocumentEvaluation:
    """Tests for DocumentEvaluation dataclass."""

    def test_evaluation_creation(self) -> None:
        """Test creating document evaluation."""
        evaluation = DocumentEvaluation()

        assert len(evaluation.page_evaluations) == 0
        assert evaluation.needs_escalation is False

    def test_escalation_rate(self) -> None:
        """Test escalation rate calculation."""
        evaluation = DocumentEvaluation(
            total_blocks=10,
            escalation_count=3,
        )

        assert evaluation.escalation_rate == 0.3

    def test_escalation_rate_zero_blocks(self) -> None:
        """Test escalation rate with zero blocks."""
        evaluation = DocumentEvaluation(total_blocks=0)

        assert evaluation.escalation_rate == 0.0


# ConfidenceEvaluator tests


class TestConfidenceEvaluator:
    """Tests for ConfidenceEvaluator."""

    def test_evaluator_creation(self) -> None:
        """Test creating evaluator."""
        evaluator = ConfidenceEvaluator()
        assert evaluator.config is not None

    def test_evaluator_with_config(self) -> None:
        """Test evaluator with custom config."""
        config = EvaluatorConfig(confidence_threshold=0.7)
        evaluator = ConfidenceEvaluator(config=config)

        assert evaluator.config.confidence_threshold == 0.7

    def test_evaluate_empty_page(self) -> None:
        """Test evaluating empty page."""
        page = OCRPage(page_number=1, width=612, height=792)
        evaluator = ConfidenceEvaluator()

        result = evaluator.evaluate_page(page)

        assert result.page_number == 1
        assert len(result.block_scores) == 0

    def test_evaluate_high_confidence_block(self) -> None:
        """Test evaluating high confidence block."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="High quality text",
            confidence=0.95,
        )

        evaluator = ConfidenceEvaluator()
        score = evaluator.evaluate_block(elem)

        assert score.quality in (QualityLevel.EXCELLENT, QualityLevel.GOOD)
        assert score.needs_escalation is False

    def test_evaluate_low_confidence_block(self) -> None:
        """Test evaluating low confidence block."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="Low quality text",
            confidence=0.4,
        )

        evaluator = ConfidenceEvaluator()
        score = evaluator.evaluate_block(elem)

        assert score.quality == QualityLevel.POOR
        assert score.needs_escalation is True

    def test_evaluate_short_text_block(self) -> None:
        """Test evaluating block with short text."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="Hi",
            confidence=0.5,
        )

        evaluator = ConfidenceEvaluator()
        score = evaluator.evaluate_block(elem)

        # Short text should be marked unknown (not enough text)
        assert score.quality == QualityLevel.UNKNOWN


class TestQualityClassification:
    """Tests for quality classification."""

    def test_classify_excellent(self) -> None:
        """Test excellent classification."""
        evaluator = ConfidenceEvaluator()

        quality = evaluator._classify_quality(0.95)
        assert quality == QualityLevel.EXCELLENT

    def test_classify_good(self) -> None:
        """Test good classification."""
        evaluator = ConfidenceEvaluator()

        quality = evaluator._classify_quality(0.85)
        assert quality == QualityLevel.GOOD

    def test_classify_acceptable(self) -> None:
        """Test acceptable classification."""
        evaluator = ConfidenceEvaluator()

        quality = evaluator._classify_quality(0.65)
        assert quality == QualityLevel.ACCEPTABLE

    def test_classify_poor(self) -> None:
        """Test poor classification."""
        evaluator = ConfidenceEvaluator()

        quality = evaluator._classify_quality(0.4)
        assert quality == QualityLevel.POOR


class TestEscalationDetection:
    """Tests for escalation detection."""

    def test_below_threshold_triggers_escalation(self) -> None:
        """Test that confidence below threshold triggers escalation."""
        evaluator = ConfidenceEvaluator()

        needs = evaluator._check_escalation_needed(
            confidence=0.5,
            quality=QualityLevel.POOR,
            low_conf_count=0,
            word_count=10,
        )

        assert needs is True

    def test_many_low_words_triggers_escalation(self) -> None:
        """Test that many low-confidence words trigger escalation."""
        evaluator = ConfidenceEvaluator()

        needs = evaluator._check_escalation_needed(
            confidence=0.7,
            quality=QualityLevel.ACCEPTABLE,
            low_conf_count=6,
            word_count=10,
        )

        assert needs is True

    def test_high_quality_no_escalation(self) -> None:
        """Test that high quality does not trigger escalation."""
        evaluator = ConfidenceEvaluator()

        needs = evaluator._check_escalation_needed(
            confidence=0.9,
            quality=QualityLevel.EXCELLENT,
            low_conf_count=1,
            word_count=10,
        )

        assert needs is False


# Convenience function tests


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_evaluate_ocr_quality(self) -> None:
        """Test evaluate_ocr_quality function."""
        doc = OCRDocument()
        page = OCRPage(page_number=1, width=612, height=792)

        bbox = BoundingBox(x1=50, y1=100, x2=200, y2=150)
        page.elements.append(
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Sample text content",
                confidence=0.85,
            )
        )
        doc.pages.append(page)

        evaluation = evaluate_ocr_quality(doc)

        assert len(evaluation.page_evaluations) == 1
        assert evaluation.total_blocks == 1

    def test_get_escalation_candidates_with_poor_blocks(self) -> None:
        """Test getting escalation candidates."""
        doc = OCRDocument()
        page = OCRPage(page_number=1, width=612, height=792)

        bbox = BoundingBox(x1=50, y1=100, x2=200, y2=150)

        # High confidence block
        page.elements.append(
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Good quality text",
                confidence=0.9,
            )
        )

        # Low confidence block
        page.elements.append(
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Poor quality text",
                confidence=0.3,
            )
        )

        doc.pages.append(page)

        candidates = get_escalation_candidates(doc)

        # Should have one candidate (the low confidence block)
        assert len(candidates) == 1
        assert candidates[0].confidence == 0.3

    def test_get_escalation_candidates_empty(self) -> None:
        """Test no candidates when all blocks are good."""
        doc = OCRDocument()
        page = OCRPage(page_number=1, width=612, height=792)

        bbox = BoundingBox(x1=50, y1=100, x2=200, y2=150)
        page.elements.append(
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Excellent quality",
                confidence=0.95,
            )
        )
        doc.pages.append(page)

        candidates = get_escalation_candidates(doc)

        assert len(candidates) == 0


class TestDocumentEvaluation:
    """Tests for full document evaluation."""

    def test_multi_page_evaluation(self) -> None:
        """Test evaluating multi-page document."""
        doc = OCRDocument()
        bbox = BoundingBox(x1=50, y1=100, x2=200, y2=150)

        for i in range(3):
            page = OCRPage(page_number=i + 1, width=612, height=792)
            page.elements.append(
                OCRElement(
                    element_type=ElementType.BLOCK,
                    bbox=bbox,
                    text=f"Page {i + 1} content here",
                    confidence=0.8,
                )
            )
            doc.pages.append(page)

        evaluator = ConfidenceEvaluator()
        evaluation = evaluator.evaluate_document(doc)

        assert len(evaluation.page_evaluations) == 3
        assert evaluation.total_blocks == 3

    def test_overall_quality_calculation(self) -> None:
        """Test overall quality is calculated correctly."""
        doc = OCRDocument()
        page = OCRPage(page_number=1, width=612, height=792)
        bbox = BoundingBox(x1=50, y1=100, x2=200, y2=150)

        # Add blocks with varying confidence
        for conf in [0.9, 0.85, 0.8]:
            page.elements.append(
                OCRElement(
                    element_type=ElementType.BLOCK,
                    bbox=bbox,
                    text="Sample text content",
                    confidence=conf,
                )
            )

        doc.pages.append(page)

        evaluator = ConfidenceEvaluator()
        evaluation = evaluator.evaluate_document(doc)

        # Overall should be average: (0.9 + 0.85 + 0.8) / 3 = 0.85
        assert 0.80 <= evaluation.overall_confidence <= 0.90
        assert evaluation.overall_quality == QualityLevel.GOOD
