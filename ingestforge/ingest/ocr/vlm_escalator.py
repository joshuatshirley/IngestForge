"""VLM Escalation Provider for OCR enhancement.

Escalates low-confidence OCR regions to Vision Language Models
for improved text extraction accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Protocol

from ingestforge.ingest.ocr.spatial_parser import BoundingBox, OCRElement
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


def load_crop_image_data(crop_region: "CropRegion") -> Optional[bytes]:
    """Load image data for a crop region.

    Args:
        crop_region: Crop region with source path

    Returns:
        Image bytes or None
    """
    # Return existing data if available
    if crop_region.cropped_image_data:
        return crop_region.cropped_image_data

    # Load from source if path provided
    if crop_region.source_image_path:
        from ingestforge.ingest.ocr.image_cropper import crop_image_region

        return crop_image_region(
            crop_region.source_image_path,
            crop_region.bbox,
            crop_region.padding,
        )

    return None


MAX_ESCALATIONS_PER_BATCH = 50
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_CROP_DIMENSION = 4096
DEFAULT_PADDING_PIXELS = 20
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30


class VLMProvider(str, Enum):
    """Supported VLM providers."""

    OPENAI = "openai"  # GPT-4 Vision
    ANTHROPIC = "anthropic"  # Claude Vision
    GOOGLE = "google"  # Gemini Vision
    LOCAL = "local"  # Local VLM (e.g., LLaVA)


class ConsentStatus(str, Enum):
    """User consent status for cloud API usage."""

    GRANTED = "granted"
    DENIED = "denied"
    NOT_ASKED = "not_asked"


@dataclass
class CropRegion:
    """Image crop region for escalation."""

    bbox: BoundingBox
    padding: int = DEFAULT_PADDING_PIXELS
    source_image_path: Optional[Path] = None
    cropped_image_data: Optional[bytes] = None

    @property
    def expanded_bbox(self) -> BoundingBox:
        """Get bbox with padding applied."""
        return BoundingBox(
            x1=max(0, self.bbox.x1 - self.padding),
            y1=max(0, self.bbox.y1 - self.padding),
            x2=min(MAX_CROP_DIMENSION, self.bbox.x2 + self.padding),
            y2=min(MAX_CROP_DIMENSION, self.bbox.y2 + self.padding),
        )


@dataclass
class EscalationRequest:
    """Request to escalate a region to VLM."""

    element: OCRElement
    crop_region: CropRegion
    original_text: str
    original_confidence: float
    page_number: int = 1
    request_id: str = ""


@dataclass
class EscalationResult:
    """Result from VLM escalation."""

    request: EscalationRequest
    success: bool
    vlm_text: str = ""
    vlm_confidence: float = 0.0
    error: str = ""
    provider_used: Optional[VLMProvider] = None
    tokens_used: int = 0
    latency_ms: float = 0.0

    @property
    def improved(self) -> bool:
        """Check if VLM result is better than original."""
        if not self.success:
            return False
        return self.vlm_confidence > self.request.original_confidence


@dataclass
class EscalatorConfig:
    """Configuration for VLM escalator."""

    provider: VLMProvider = VLMProvider.OPENAI
    api_key: str = ""
    model_name: str = ""
    max_tokens: int = 500
    temperature: float = 0.0
    crop_padding: int = DEFAULT_PADDING_PIXELS
    require_consent: bool = True
    consent_status: ConsentStatus = ConsentStatus.NOT_ASKED


class VLMClient(Protocol):
    """Protocol for VLM client implementations."""

    def extract_text(
        self,
        image_data: bytes,
        prompt: str,
    ) -> tuple[str, float]:
        """Extract text from image.

        Args:
            image_data: Image bytes
            prompt: Extraction prompt

        Returns:
            Tuple of (extracted_text, confidence)
        """
        ...


@dataclass
class VLMClientStub:
    """Stub VLM client for testing."""

    response_text: str = ""
    response_confidence: float = 0.9

    def extract_text(
        self,
        image_data: bytes,
        prompt: str,
    ) -> tuple[str, float]:
        """Return stub response."""
        return (self.response_text, self.response_confidence)


class VLMEscalator:
    """Escalates low-confidence OCR regions to VLM providers.

    Handles consent management, image cropping, and API calls
    to vision language models for text extraction.
    """

    def __init__(
        self,
        config: Optional[EscalatorConfig] = None,
        client: Optional[VLMClient] = None,
    ) -> None:
        """Initialize escalator.

        Args:
            config: Escalator configuration
            client: Optional VLM client (for testing)
        """
        self.config = config or EscalatorConfig()
        self._client = client
        self._consent_callback: Optional[Callable[[], bool]] = None

    def set_consent_callback(self, callback: Callable[[], bool]) -> None:
        """Set callback for requesting user consent.

        Args:
            callback: Function that returns True if user consents
        """
        self._consent_callback = callback

    def request_consent(self) -> bool:
        """Request user consent for cloud API usage.

        Returns:
            True if consent granted
        """
        if self.config.consent_status == ConsentStatus.GRANTED:
            return True
        if self.config.consent_status == ConsentStatus.DENIED:
            return False

        # Use callback if available
        if self._consent_callback:
            granted = self._consent_callback()
            self.config.consent_status = (
                ConsentStatus.GRANTED if granted else ConsentStatus.DENIED
            )
            return granted

        # No callback, require explicit configuration
        logger.warning("VLM escalation requires explicit user consent")
        return False

    def check_consent(self) -> bool:
        """Check if consent has been granted.

        Returns:
            True if consent is granted
        """
        if not self.config.require_consent:
            return True
        return self.config.consent_status == ConsentStatus.GRANTED

    def escalate_batch(
        self,
        requests: List[EscalationRequest],
    ) -> List[EscalationResult]:
        """Escalate a batch of regions to VLM.

        Args:
            requests: List of escalation requests

        Returns:
            List of escalation results
        """
        results: List[EscalationResult] = []

        # Check consent
        if not self.check_consent():
            for req in requests[:MAX_ESCALATIONS_PER_BATCH]:
                results.append(
                    EscalationResult(
                        request=req,
                        success=False,
                        error="User consent not granted for cloud API usage",
                    )
                )
            return results

        # Process requests
        for req in requests[:MAX_ESCALATIONS_PER_BATCH]:
            result = self.escalate_single(req)
            results.append(result)

        return results

    def escalate_single(self, request: EscalationRequest) -> EscalationResult:
        """Escalate a single region to VLM.

        Args:
            request: Escalation request

        Returns:
            EscalationResult
        """
        # Check consent
        if not self.check_consent():
            return EscalationResult(
                request=request,
                success=False,
                error="User consent not granted",
            )

        # Validate request
        if not self._validate_request(request):
            return EscalationResult(
                request=request,
                success=False,
                error="Invalid request: missing image data",
            )

        # Get client
        client = self._get_client()
        if not client:
            return EscalationResult(
                request=request,
                success=False,
                error=f"No VLM client available for {self.config.provider}",
            )

        # Build prompt
        prompt = self._build_extraction_prompt(request)

        # Get or load image data
        image_data = load_crop_image_data(request.crop_region)
        if not image_data:
            return EscalationResult(
                request=request,
                success=False,
                error="Failed to load image data",
            )

        # Call VLM
        try:
            text, confidence = client.extract_text(image_data, prompt)
            return EscalationResult(
                request=request,
                success=True,
                vlm_text=text,
                vlm_confidence=confidence,
                provider_used=self.config.provider,
            )
        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            return EscalationResult(
                request=request,
                success=False,
                error=str(e),
            )

    def _validate_request(self, request: EscalationRequest) -> bool:
        """Validate escalation request.

        Args:
            request: Request to validate

        Returns:
            True if valid
        """
        if not request.crop_region:
            return False
        if not request.crop_region.cropped_image_data:
            if not request.crop_region.source_image_path:
                return False
        return True

    def _get_client(self) -> Optional[VLMClient]:
        """Get VLM client for configured provider.

        Returns:
            VLM client or None
        """
        if self._client:
            return self._client
        if not self.config.api_key:
            logger.warning(f"No API key configured for {self.config.provider}")
            return None

        # Initialize client based on provider
        try:
            if self.config.provider == VLMProvider.OPENAI:
                from ingestforge.ingest.ocr.vlm_clients import OpenAIVisionClient

                return OpenAIVisionClient(
                    api_key=self.config.api_key,
                    model_name=self.config.model_name or "gpt-4o-mini",
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            elif self.config.provider == VLMProvider.ANTHROPIC:
                from ingestforge.ingest.ocr.vlm_clients import ClaudeVisionClient

                return ClaudeVisionClient(
                    api_key=self.config.api_key,
                    model_name=self.config.model_name or "claude-3-haiku-20240307",
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            else:
                logger.warning(f"Unsupported VLM provider: {self.config.provider}")
                return None
        except ImportError as e:
            logger.error(f"Failed to import VLM client: {e}")
            return None

    def _build_extraction_prompt(self, request: EscalationRequest) -> str:
        """Build prompt for VLM text extraction.

        Args:
            request: Escalation request

        Returns:
            Prompt string
        """
        prompt = (
            "Extract the text visible in this image region. "
            "Focus on accuracy and preserve the exact wording. "
            "Return only the extracted text without any additional commentary."
        )

        if request.original_text:
            prompt += (
                f"\n\nContext: OCR detected this text with low confidence: "
                f'"{request.original_text}". '
                "Please verify and correct if needed."
            )

        return prompt


def create_escalator(
    provider: str = "openai",
    api_key: str = "",
    consent_granted: bool = False,
) -> VLMEscalator:
    """Factory function to create VLM escalator.

    Args:
        provider: VLM provider name
        api_key: API key for provider
        consent_granted: Whether user has consented

    Returns:
        Configured VLMEscalator
    """
    try:
        vlm_provider = VLMProvider(provider.lower())
    except ValueError:
        vlm_provider = VLMProvider.OPENAI

    consent = ConsentStatus.GRANTED if consent_granted else ConsentStatus.NOT_ASKED

    config = EscalatorConfig(
        provider=vlm_provider,
        api_key=api_key,
        consent_status=consent,
    )

    return VLMEscalator(config=config)


def identify_low_confidence_elements(
    elements: List[OCRElement],
    confidence_threshold: float = 0.7,
) -> List[OCRElement]:
    """Identify elements with low OCR confidence.

    Args:
        elements: OCR elements to check
        confidence_threshold: Threshold below which to escalate

    Returns:
        List of low-confidence elements
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        logger.warning(f"Invalid threshold {confidence_threshold}, using 0.7")
        confidence_threshold = 0.7

    low_confidence = []
    for elem in elements[:MAX_ESCALATIONS_PER_BATCH]:
        if elem.confidence < confidence_threshold and elem.bbox:
            low_confidence.append(elem)

    return low_confidence


def escalate_elements(
    elements: List[OCRElement],
    image_path: Path,
    escalator: VLMEscalator,
) -> List[EscalationResult]:
    """Convenience function to escalate elements.

    Args:
        elements: Elements to escalate
        image_path: Source image path
        escalator: Configured escalator

    Returns:
        List of escalation results
    """
    requests: List[EscalationRequest] = []

    for idx, elem in enumerate(elements):
        if not elem.bbox:
            continue

        request = EscalationRequest(
            element=elem,
            crop_region=CropRegion(
                bbox=elem.bbox,
                source_image_path=image_path,
            ),
            original_text=elem.text,
            original_confidence=elem.confidence,
            request_id=f"esc_{idx}",
        )
        requests.append(request)

    return escalator.escalate_batch(requests)


def escalate_low_confidence_ocr(
    image_path: Path,
    ocr_elements: List[OCRElement],
    escalator: VLMEscalator,
    confidence_threshold: float = 0.7,
) -> List[EscalationResult]:
    """Identify and escalate low-confidence OCR elements.

    This is the main entry point for VLM escalation.

    Args:
        image_path: Source image file
        ocr_elements: All OCR elements from the page
        escalator: Configured VLM escalator
        confidence_threshold: Confidence below which to escalate

    Returns:
        List of escalation results
    """
    # Identify low-confidence regions
    low_conf = identify_low_confidence_elements(ocr_elements, confidence_threshold)

    if not low_conf:
        logger.info("No low-confidence regions found")
        return []

    logger.info(
        f"Found {len(low_conf)} low-confidence regions "
        f"(threshold: {confidence_threshold})"
    )

    # Escalate them
    return escalate_elements(low_conf, image_path, escalator)
