"""Tests for VLM escalation provider.

Tests consent management, escalation requests, and VLM integration."""

from __future__ import annotations

from pathlib import Path


from ingestforge.ingest.ocr.spatial_parser import BoundingBox, ElementType, OCRElement
from ingestforge.ingest.ocr.vlm_escalator import (
    ConsentStatus,
    CropRegion,
    EscalationRequest,
    EscalationResult,
    EscalatorConfig,
    VLMClientStub,
    VLMEscalator,
    VLMProvider,
    create_escalator,
    escalate_elements,
)

# VLMProvider tests


class TestVLMProvider:
    """Tests for VLMProvider enum."""

    def test_providers_defined(self) -> None:
        """Test all providers are defined."""
        providers = [p.value for p in VLMProvider]

        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "local" in providers


# ConsentStatus tests


class TestConsentStatus:
    """Tests for ConsentStatus enum."""

    def test_statuses_defined(self) -> None:
        """Test all statuses are defined."""
        statuses = [s.value for s in ConsentStatus]

        assert "granted" in statuses
        assert "denied" in statuses
        assert "not_asked" in statuses


# CropRegion tests


class TestCropRegion:
    """Tests for CropRegion dataclass."""

    def test_region_creation(self) -> None:
        """Test creating crop region."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        region = CropRegion(bbox=bbox)

        assert region.bbox.x1 == 100
        assert region.padding == 20  # default

    def test_expanded_bbox(self) -> None:
        """Test expanded bbox with padding."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        region = CropRegion(bbox=bbox, padding=10)

        expanded = region.expanded_bbox

        assert expanded.x1 == 90
        assert expanded.y1 == 190
        assert expanded.x2 == 310
        assert expanded.y2 == 410

    def test_expanded_bbox_at_edge(self) -> None:
        """Test expanded bbox doesn't go negative."""
        bbox = BoundingBox(x1=5, y1=5, x2=100, y2=100)
        region = CropRegion(bbox=bbox, padding=20)

        expanded = region.expanded_bbox

        assert expanded.x1 == 0
        assert expanded.y1 == 0


# EscalationRequest tests


class TestEscalationRequest:
    """Tests for EscalationRequest dataclass."""

    def test_request_creation(self) -> None:
        """Test creating escalation request."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="Original text",
            confidence=0.4,
        )
        region = CropRegion(bbox=bbox)

        request = EscalationRequest(
            element=elem,
            crop_region=region,
            original_text="Original text",
            original_confidence=0.4,
            page_number=1,
        )

        assert request.original_text == "Original text"
        assert request.original_confidence == 0.4


# EscalationResult tests


class TestEscalationResult:
    """Tests for EscalationResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")
        region = CropRegion(bbox=bbox)

        request = EscalationRequest(
            element=elem,
            crop_region=region,
            original_text="Original",
            original_confidence=0.4,
        )

        result = EscalationResult(
            request=request,
            success=True,
            vlm_text="Corrected text",
            vlm_confidence=0.9,
            provider_used=VLMProvider.OPENAI,
        )

        assert result.success is True
        assert result.improved is True

    def test_failed_result(self) -> None:
        """Test failed result."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")
        region = CropRegion(bbox=bbox)

        request = EscalationRequest(
            element=elem,
            crop_region=region,
            original_text="Original",
            original_confidence=0.4,
        )

        result = EscalationResult(
            request=request,
            success=False,
            error="API error",
        )

        assert result.success is False
        assert result.improved is False

    def test_not_improved_result(self) -> None:
        """Test result that didn't improve."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")
        region = CropRegion(bbox=bbox)

        request = EscalationRequest(
            element=elem,
            crop_region=region,
            original_text="Original",
            original_confidence=0.8,
        )

        result = EscalationResult(
            request=request,
            success=True,
            vlm_text="Similar text",
            vlm_confidence=0.7,  # Lower than original
        )

        assert result.success is True
        assert result.improved is False  # Not better than original


# VLMEscalator tests


class TestVLMEscalator:
    """Tests for VLMEscalator."""

    def test_escalator_creation(self) -> None:
        """Test creating escalator."""
        escalator = VLMEscalator()
        assert escalator.config is not None

    def test_escalator_with_config(self) -> None:
        """Test escalator with custom config."""
        config = EscalatorConfig(provider=VLMProvider.ANTHROPIC)
        escalator = VLMEscalator(config=config)

        assert escalator.config.provider == VLMProvider.ANTHROPIC

    def test_consent_not_granted_by_default(self) -> None:
        """Test consent is not granted by default."""
        escalator = VLMEscalator()

        assert escalator.check_consent() is False

    def test_consent_granted_explicit(self) -> None:
        """Test explicitly granted consent."""
        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        escalator = VLMEscalator(config=config)

        assert escalator.check_consent() is True

    def test_consent_not_required(self) -> None:
        """Test when consent is not required."""
        config = EscalatorConfig(require_consent=False)
        escalator = VLMEscalator(config=config)

        assert escalator.check_consent() is True

    def test_consent_callback(self) -> None:
        """Test consent callback."""
        escalator = VLMEscalator()
        escalator.set_consent_callback(lambda: True)

        granted = escalator.request_consent()

        assert granted is True
        assert escalator.config.consent_status == ConsentStatus.GRANTED

    def test_consent_callback_denied(self) -> None:
        """Test consent callback denial."""
        escalator = VLMEscalator()
        escalator.set_consent_callback(lambda: False)

        granted = escalator.request_consent()

        assert granted is False
        assert escalator.config.consent_status == ConsentStatus.DENIED


class TestEscalation:
    """Tests for escalation operations."""

    def test_escalate_without_consent(self) -> None:
        """Test escalation fails without consent."""
        escalator = VLMEscalator()

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")
        region = CropRegion(bbox=bbox)

        request = EscalationRequest(
            element=elem,
            crop_region=region,
            original_text="Test",
            original_confidence=0.4,
        )

        result = escalator.escalate_single(request)

        assert result.success is False
        assert "consent" in result.error.lower()

    def test_escalate_with_stub_client(self) -> None:
        """Test escalation with stub client."""
        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        stub_client = VLMClientStub(
            response_text="Extracted text",
            response_confidence=0.95,
        )
        escalator = VLMEscalator(config=config, client=stub_client)

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")
        region = CropRegion(bbox=bbox, cropped_image_data=b"fake image data")

        request = EscalationRequest(
            element=elem,
            crop_region=region,
            original_text="Test",
            original_confidence=0.4,
        )

        result = escalator.escalate_single(request)

        assert result.success is True
        assert result.vlm_text == "Extracted text"
        assert result.vlm_confidence == 0.95

    def test_escalate_missing_image_data(self) -> None:
        """Test escalation fails with missing image data."""
        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        stub_client = VLMClientStub()
        escalator = VLMEscalator(config=config, client=stub_client)

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")
        region = CropRegion(bbox=bbox)  # No image data

        request = EscalationRequest(
            element=elem,
            crop_region=region,
            original_text="Test",
            original_confidence=0.4,
        )

        result = escalator.escalate_single(request)

        assert result.success is False

    def test_escalate_batch(self) -> None:
        """Test batch escalation."""
        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        stub_client = VLMClientStub(response_text="Text", response_confidence=0.9)
        escalator = VLMEscalator(config=config, client=stub_client)

        requests = []
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")

        for i in range(3):
            region = CropRegion(bbox=bbox, cropped_image_data=b"data")
            req = EscalationRequest(
                element=elem,
                crop_region=region,
                original_text=f"Text {i}",
                original_confidence=0.4,
                request_id=f"req_{i}",
            )
            requests.append(req)

        results = escalator.escalate_batch(requests)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_escalate_batch_without_consent(self) -> None:
        """Test batch escalation fails without consent."""
        escalator = VLMEscalator()

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="T")
        region = CropRegion(bbox=bbox)

        requests = [
            EscalationRequest(
                element=elem,
                crop_region=region,
                original_text="Test",
                original_confidence=0.4,
            )
        ]

        results = escalator.escalate_batch(requests)

        assert len(results) == 1
        assert results[0].success is False


# Factory function tests


class TestCreateEscalator:
    """Tests for create_escalator factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        escalator = create_escalator()

        assert escalator.config.provider == VLMProvider.OPENAI
        assert escalator.config.consent_status == ConsentStatus.NOT_ASKED

    def test_create_with_provider(self) -> None:
        """Test creating with specific provider."""
        escalator = create_escalator(provider="anthropic")

        assert escalator.config.provider == VLMProvider.ANTHROPIC

    def test_create_with_consent(self) -> None:
        """Test creating with consent granted."""
        escalator = create_escalator(consent_granted=True)

        assert escalator.config.consent_status == ConsentStatus.GRANTED

    def test_create_invalid_provider(self) -> None:
        """Test creating with invalid provider falls back to default."""
        escalator = create_escalator(provider="invalid_provider")

        assert escalator.config.provider == VLMProvider.OPENAI


# Convenience function tests


class TestEscalateElements:
    """Tests for escalate_elements function."""

    def test_escalate_elements_creates_requests(self, tmp_path: Path) -> None:
        """Test that escalate_elements creates proper requests."""
        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        stub_client = VLMClientStub(response_text="Text", response_confidence=0.9)
        escalator = VLMEscalator(config=config, client=stub_client)

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elements = [
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Test text",
                confidence=0.4,
            ),
        ]

        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image")

        # Note: escalate_elements doesn't auto-load images
        # so this will fail due to missing image data
        results = escalate_elements(elements, image_path, escalator)

        assert len(results) == 1

    def test_escalate_elements_skips_no_bbox(self, tmp_path: Path) -> None:
        """Test that elements without bbox are skipped."""
        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        escalator = VLMEscalator(config=config)

        elements = [
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=None,  # No bbox
                text="Test",
                confidence=0.4,
            ),
        ]

        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image")

        results = escalate_elements(elements, image_path, escalator)

        assert len(results) == 0


# Integration tests for new functionality


class TestIdentifyLowConfidence:
    """Tests for identify_low_confidence_elements."""

    def test_identify_low_confidence(self) -> None:
        """Test identifying low confidence elements."""
        from ingestforge.ingest.ocr.vlm_escalator import (
            identify_low_confidence_elements,
        )

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elements = [
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="High confidence",
                confidence=0.9,
            ),
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Low confidence",
                confidence=0.5,
            ),
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Medium confidence",
                confidence=0.7,
            ),
        ]

        low_conf = identify_low_confidence_elements(elements, confidence_threshold=0.7)

        assert len(low_conf) == 1
        assert low_conf[0].confidence == 0.5

    def test_identify_with_invalid_threshold(self) -> None:
        """Test invalid threshold gets clamped."""
        from ingestforge.ingest.ocr.vlm_escalator import (
            identify_low_confidence_elements,
        )

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elements = [
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Test",
                confidence=0.5,
            ),
        ]

        # Invalid threshold > 1.0
        low_conf = identify_low_confidence_elements(elements, confidence_threshold=1.5)

        # Should use default 0.7
        assert len(low_conf) == 1

    def test_identify_skips_no_bbox(self) -> None:
        """Test elements without bbox are skipped."""
        from ingestforge.ingest.ocr.vlm_escalator import (
            identify_low_confidence_elements,
        )

        elements = [
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=None,
                text="No bbox",
                confidence=0.3,
            ),
        ]

        low_conf = identify_low_confidence_elements(elements, confidence_threshold=0.7)

        assert len(low_conf) == 0


class TestEscalateLowConfidenceOCR:
    """Tests for escalate_low_confidence_ocr."""

    def test_escalate_full_flow(self, tmp_path: Path) -> None:
        """Test full escalation flow."""
        from ingestforge.ingest.ocr.vlm_escalator import escalate_low_confidence_ocr

        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        stub_client = VLMClientStub(response_text="Corrected", response_confidence=0.95)
        escalator = VLMEscalator(config=config, client=stub_client)

        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image")

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elements = [
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="Low conf",
                confidence=0.4,
            ),
        ]

        # Note: This will fail without actual image loading
        # but tests the flow
        results = escalate_low_confidence_ocr(
            image_path, elements, escalator, confidence_threshold=0.7
        )

        assert len(results) == 1

    def test_escalate_no_low_confidence(self, tmp_path: Path) -> None:
        """Test when there are no low confidence elements."""
        from ingestforge.ingest.ocr.vlm_escalator import escalate_low_confidence_ocr

        config = EscalatorConfig(consent_status=ConsentStatus.GRANTED)
        escalator = VLMEscalator(config=config)

        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake")

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elements = [
            OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text="High conf",
                confidence=0.95,
            ),
        ]

        results = escalate_low_confidence_ocr(
            image_path, elements, escalator, confidence_threshold=0.7
        )

        assert len(results) == 0


class TestLoadCropImageData:
    """Tests for load_crop_image_data helper."""

    def test_load_from_existing_data(self) -> None:
        """Test loading when data already exists."""
        from ingestforge.ingest.ocr.vlm_escalator import load_crop_image_data

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        region = CropRegion(bbox=bbox, cropped_image_data=b"existing data")

        data = load_crop_image_data(region)

        assert data == b"existing data"

    def test_load_no_source(self) -> None:
        """Test loading with no source returns None."""
        from ingestforge.ingest.ocr.vlm_escalator import load_crop_image_data

        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        region = CropRegion(bbox=bbox)

        data = load_crop_image_data(region)

        assert data is None
