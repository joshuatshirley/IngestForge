"""
VLM Vision Processor for Visual Evidence Extraction.

VLM Vision Processor.
Extracts structured data from charts and diagrams using Vision-Language Models.

Features:
- IFVisionProcessor implementing IFProcessor interface
- CSV data extraction from chart images
- Bounding box coordinate tracking for data points
- Support for Ollama and llama.cpp local VLM backends
- Graceful error handling (JPL Rule #7)

NASA JPL Power of Ten Rules:
- Rule #2: Bounded image sizes and data structures
- Rule #4: Functions <60 lines
- Rule #7: Check all return values, fail gracefully
- Rule #9: Complete type hints
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFArtifact, IFProcessor

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_IMAGE_SIZE_MB = 10  # Maximum image size
MAX_DATA_POINTS = 10000  # Maximum data points in extracted CSV
MAX_BOUNDING_BOXES = 1000  # Maximum bounding boxes per image
SUPPORTED_IMAGE_TYPES = frozenset(["png", "jpg", "jpeg", "webp", "gif"])
DEFAULT_VLM_TIMEOUT = 60  # VLM call timeout in seconds


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BoundingBox:
    """
    Bounding box coordinates for a visual element.

    AC: Hard-linking of data points to bounding box coordinates.
    Rule #9: Complete type hints.
    """

    x: float
    y: float
    width: float
    height: float
    label: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata storage."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox:
        """Create from dictionary."""
        return cls(
            x=float(data.get("x", 0)),
            y=float(data.get("y", 0)),
            width=float(data.get("width", 0)),
            height=float(data.get("height", 0)),
            label=str(data.get("label", "")),
            confidence=float(data.get("confidence", 0)),
        )


@dataclass
class ChartDataResult:
    """
    Result of chart data extraction.

    AC: Logic to extract CSV data from chart images.
    Rule #9: Complete type hints.
    """

    chart_type: str = ""  # bar, line, pie, scatter, etc.
    csv_data: str = ""  # Extracted data in CSV format
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0
    success: bool = False
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for artifact metadata."""
        return {
            "chart_type": self.chart_type,
            "csv_data": self.csv_data,
            "data_points": self.data_points[:MAX_DATA_POINTS],
            "bounding_boxes": [
                bb.to_dict() for bb in self.bounding_boxes[:MAX_BOUNDING_BOXES]
            ],
            "description": self.description,
            "confidence": self.confidence,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class VisionProcessorConfig:
    """
    Configuration for the Vision Processor.

    Rule #9: Complete type hints.
    """

    backend: str = "ollama"  # ollama, llamacpp, openai
    model_name: str = "llava"  # Model name for the backend
    timeout: int = DEFAULT_VLM_TIMEOUT  # Timeout in seconds
    temperature: float = 0.1  # Low temperature for structured extraction
    max_tokens: int = 2000  # Max tokens in response
    ollama_host: str = "http://localhost:11434"
    api_key: Optional[str] = None  # For OpenAI backend


# =============================================================================
# IMAGE ARTIFACT
# =============================================================================


class IFImageArtifact(IFArtifact):
    """
    Artifact representing an image with extracted visual data.

    Store bounding boxes in metadata.
    Rule #9: Complete type hints.
    """

    image_path: Optional[str] = Field(None, description="Path to source image")
    image_data: Optional[bytes] = Field(
        None, description="Image bytes (base64 encoded for storage)"
    )
    mime_type: str = Field("image/png", description="Image MIME type")
    width: int = Field(0, description="Image width in pixels")
    height: int = Field(0, description="Image height in pixels")

    # Extracted data
    chart_result: Optional[Dict[str, Any]] = Field(
        None, description="Chart extraction result"
    )
    description: str = Field("", description="VLM-generated description")
    extraction_confidence: float = Field(0.0, description="Confidence of extraction")

    model_config = {
        "frozen": True,
        "extra": "forbid",
        "arbitrary_types_allowed": True,
    }

    def derive(self, processor_id: str, **kwargs: Any) -> "IFImageArtifact":
        """Create a derived image artifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )

    @property
    def has_bounding_boxes(self) -> bool:
        """Check if bounding boxes are present."""
        if self.chart_result is None:
            return False
        boxes = self.chart_result.get("bounding_boxes", [])
        return len(boxes) > 0

    @property
    def bounding_boxes(self) -> List[BoundingBox]:
        """Get bounding boxes from chart result."""
        if self.chart_result is None:
            return []
        boxes = self.chart_result.get("bounding_boxes", [])
        return [BoundingBox.from_dict(b) for b in boxes]


# =============================================================================
# VLM CLIENT INTERFACE
# =============================================================================


class VLMClient:
    """
    Base interface for VLM clients.

    Rule #7: All implementations must return result or raise exception.
    """

    def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
    ) -> Tuple[str, float]:
        """
        Analyze image with VLM.

        Args:
            image_data: Image bytes.
            prompt: Analysis prompt.

        Returns:
            Tuple of (response_text, confidence).
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if VLM backend is available."""
        return False


class OllamaVLMClient(VLMClient):
    """
    Ollama VLM client for local vision models.

    Use Ollama for local vision support.
    Rule #4: Methods <60 lines.
    Rule #7: Check return values.
    """

    def __init__(
        self,
        model_name: str = "llava",
        host: str = "http://localhost:11434",
        timeout: int = DEFAULT_VLM_TIMEOUT,
    ) -> None:
        """Initialize Ollama client."""
        self._model_name = model_name
        self._host = host.rstrip("/")
        self._timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests

            response = requests.get(f"{self._host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
    ) -> Tuple[str, float]:
        """
        Analyze image using Ollama vision model.

        Rule #7: Check response, return tuple.

        Args:
            image_data: Image bytes.
            prompt: Analysis prompt.

        Returns:
            Tuple of (response_text, confidence).
        """
        import requests

        # Encode image
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # Build request
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self._host}/api/generate",
                json=payload,
                timeout=self._timeout,
            )

            # Rule #7: Check return value
            if response.status_code != 200:
                logger.error(f"Ollama returned status {response.status_code}")
                return ("", 0.0)

            data = response.json()
            text = data.get("response", "")

            # Estimate confidence based on response length/quality
            confidence = 0.8 if len(text) > 50 else 0.5

            return (text, confidence)

        except Exception as e:
            logger.error(f"Ollama VLM call failed: {e}")
            return ("", 0.0)


class LlamaCppVLMClient(VLMClient):
    """
    llama.cpp VLM client for local vision models.

    Use llama.cpp for local vision support.
    Rule #4: Methods <60 lines.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
    ) -> None:
        """Initialize llama.cpp client."""
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._model = None

    def is_available(self) -> bool:
        """Check if llama.cpp vision is available."""
        if not self._model_path:
            return False
        return Path(self._model_path).exists()

    def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
    ) -> Tuple[str, float]:
        """
        Analyze image using llama.cpp vision.

        Rule #7: Check return value.
        """
        try:
            # Import llama-cpp-python with vision support
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            if self._model is None:
                if not self._model_path or not self._mmproj_path:
                    return ("Model paths not configured", 0.0)

                chat_handler = Llava15ChatHandler(clip_model_path=self._mmproj_path)
                self._model = Llama(
                    model_path=self._model_path,
                    chat_handler=chat_handler,
                    n_ctx=4096,
                )

            # Encode image as data URI
            base64_image = base64.b64encode(image_data).decode("utf-8")
            image_uri = f"data:image/jpeg;base64,{base64_image}"

            # Create chat completion
            response = self._model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                        ],
                    }
                ],
            )

            # Rule #7: Check response
            if not response or "choices" not in response:
                return ("", 0.0)

            text = response["choices"][0]["message"]["content"]
            return (text, 0.85)

        except ImportError:
            logger.warning("llama-cpp-python not installed")
            return ("", 0.0)
        except Exception as e:
            logger.error(f"llama.cpp VLM call failed: {e}")
            return ("", 0.0)


# =============================================================================
# CHART DATA EXTRACTION
# =============================================================================

CHART_EXTRACTION_PROMPT = """Analyze this chart/graph image and extract the data.

Your response MUST be in this exact format:

CHART_TYPE: [bar|line|pie|scatter|area|other]

CSV_DATA:
column1,column2,column3
value1,value2,value3
...

BOUNDING_BOXES:
label,x,y,width,height
"Data Point 1",100,200,50,50
...

DESCRIPTION:
[Brief description of what the chart shows]

If you cannot extract the data, respond with:
ERROR: [reason]

Be precise with numbers. Extract all visible data points."""


def parse_chart_extraction_response(response: str) -> ChartDataResult:
    """
    Parse VLM response into structured ChartDataResult.

    Rule #4: <60 lines.
    Rule #7: Handle malformed responses gracefully.

    Args:
        response: Raw VLM response text.

    Returns:
        ChartDataResult with extracted data.
    """
    result = ChartDataResult()

    if not response:
        result.error_message = "Empty VLM response"
        return result

    # Check for error
    if response.strip().startswith("ERROR:"):
        result.error_message = response.strip()
        return result

    lines = response.strip().split("\n")

    # Parse sections
    current_section = None
    csv_lines: List[str] = []
    bbox_lines: List[str] = []
    description_lines: List[str] = []

    for line in lines:
        line = line.strip()

        if line.startswith("CHART_TYPE:"):
            result.chart_type = line.replace("CHART_TYPE:", "").strip().lower()
            current_section = None
        elif line == "CSV_DATA:":
            current_section = "csv"
        elif line == "BOUNDING_BOXES:":
            current_section = "bbox"
        elif line == "DESCRIPTION:":
            current_section = "desc"
        elif current_section == "csv" and line:
            csv_lines.append(line)
        elif current_section == "bbox" and line:
            bbox_lines.append(line)
        elif current_section == "desc" and line:
            description_lines.append(line)

    # Build CSV data
    result.csv_data = "\n".join(csv_lines)

    # Parse bounding boxes (skip header) - Refactored to helper (Rule #1, #4)
    for line in bbox_lines[1:] if bbox_lines else []:
        bbox = _parse_bounding_box_line(line)
        if bbox is not None:
            result.bounding_boxes.append(bbox)

    result.description = " ".join(description_lines)
    result.success = bool(result.csv_data or result.chart_type)
    result.confidence = 0.8 if result.success else 0.0

    return result


def _parse_csv_line(line: str) -> List[str]:
    """
    Parse a CSV line handling quoted values.

    Rule #5: Assert precondition.
    Rule #7: Handle parsing errors.
    """
    assert isinstance(line, str), "line must be string"

    import csv

    try:
        reader = csv.reader([line])
        result = next(reader)
        return result if result else []
    except Exception:
        return line.split(",")


def _parse_bounding_box_line(line: str) -> Optional[BoundingBox]:
    """
    Parse a single bounding box line.

    Rule #4: Extracted from parse_chart_extraction_response.
    Rule #5: Assert precondition.
    Rule #7: Return None on error.

    Args:
        line: CSV line with label,x,y,width,height.

    Returns:
        BoundingBox if valid, None otherwise.
    """
    assert isinstance(line, str), "line must be string"

    parts = _parse_csv_line(line)

    if len(parts) < 5:
        return None

    try:
        return BoundingBox(
            label=parts[0].strip('"'),
            x=float(parts[1]),
            y=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )
    except ValueError:
        return None


# =============================================================================
# VISION PROCESSOR
# =============================================================================


class IFVisionProcessor(IFProcessor):
    """
    Vision processor for extracting structured data from chart/diagram images.

    VLM Vision Processor.

    Features:
    - Extract CSV data from chart images
    - Track bounding box coordinates for data points
    - Support Ollama and llama.cpp backends
    - Graceful error handling

    Example:
        processor = IFVisionProcessor()
        result = processor.process(image_artifact)
        print(result.chart_result["csv_data"])
    """

    def __init__(
        self,
        config: Optional[VisionProcessorConfig] = None,
    ) -> None:
        """
        Initialize vision processor.

        Args:
            config: Processor configuration.
        """
        self._config = config or VisionProcessorConfig()
        self._client: Optional[VLMClient] = None

    @property
    def processor_id(self) -> str:
        """Get processor identifier."""
        return "if-vision-processor"

    @property
    def version(self) -> str:
        """Get processor version."""
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        """Get processor capabilities."""
        return ["chart_extraction", "diagram_analysis", "visual_evidence"]

    def is_available(self) -> bool:
        """
        Check if VLM backend is available.

        Rule #7: Check dependencies before use.

        Returns:
            True if VLM is available.
        """
        client = self._get_client()
        return client.is_available()

    def _get_client(self) -> VLMClient:
        """Get or create VLM client based on config."""
        if self._client is not None:
            return self._client

        if self._config.backend == "ollama":
            self._client = OllamaVLMClient(
                model_name=self._config.model_name,
                host=self._config.ollama_host,
                timeout=self._config.timeout,
            )
        elif self._config.backend == "llamacpp":
            model_path = os.environ.get("LLAMA_MODEL_PATH")
            mmproj_path = os.environ.get("LLAMA_MMPROJ_PATH")
            self._client = LlamaCppVLMClient(
                model_path=model_path,
                mmproj_path=mmproj_path,
            )
        else:
            # Fallback to Ollama
            self._client = OllamaVLMClient()

        return self._client

    def process(self, artifact: IFArtifact) -> IFImageArtifact:
        """
        Process an image artifact to extract chart data.

        AC: Extract CSV data, store bounding boxes.
        Rule #4: <60 lines.
        Rule #5: Assert preconditions.
        Rule #7: Check return values, fail gracefully.

        Args:
            artifact: Input artifact (must have image data or path).

        Returns:
            IFImageArtifact with extracted chart data.
        """
        # Preconditions (Rule #5)
        assert artifact is not None, "Artifact cannot be None"

        # Get image data
        image_data = self._get_image_data(artifact)

        if image_data is None:
            # Fail gracefully (Rule #7)
            return self._create_error_artifact(artifact, "Could not read image data")

        # Check image size (Rule #2)
        if len(image_data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            return self._create_error_artifact(
                artifact, f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit"
            )

        # Check VLM availability
        if not self.is_available():
            return self._create_error_artifact(artifact, "VLM backend not available")

        # Extract chart data
        chart_result = self._extract_chart_data(image_data)

        # Create result artifact
        return self._create_result_artifact(artifact, image_data, chart_result)

    def _get_image_data(self, artifact: IFArtifact) -> Optional[bytes]:
        """
        Get image bytes from artifact.

        Rule #4: <60 lines.
        Rule #7: Return None on error.
        """
        # Check if artifact has image_data directly
        if hasattr(artifact, "image_data") and artifact.image_data:
            return artifact.image_data

        # Check if artifact has image_path
        if hasattr(artifact, "image_path") and artifact.image_path:
            path = Path(artifact.image_path)
            if path.exists():
                return path.read_bytes()

        # Check metadata for image path
        image_path = artifact.metadata.get("image_path")
        if image_path:
            path = Path(image_path)
            if path.exists():
                return path.read_bytes()

        # Check file artifact
        if hasattr(artifact, "file_path"):
            path = Path(artifact.file_path)
            if (
                path.exists()
                and path.suffix.lower().lstrip(".") in SUPPORTED_IMAGE_TYPES
            ):
                return path.read_bytes()

        return None

    def _extract_chart_data(self, image_data: bytes) -> ChartDataResult:
        """
        Extract structured data from chart image.

        AC: Logic to extract CSV data from chart images.
        Rule #4: <60 lines.
        Rule #7: Handle VLM errors gracefully.
        """
        client = self._get_client()

        # Call VLM (Rule #7: check return)
        response, confidence = client.analyze_image(
            image_data,
            CHART_EXTRACTION_PROMPT,
        )

        if not response:
            return ChartDataResult(
                success=False,
                error_message="VLM returned empty response",
            )

        # Parse response
        result = parse_chart_extraction_response(response)
        result.confidence = confidence

        logger.info(
            f"Chart extraction: type={result.chart_type}, "
            f"data_rows={result.csv_data.count(chr(10))}, "
            f"bboxes={len(result.bounding_boxes)}"
        )

        return result

    def _create_error_artifact(
        self,
        source: IFArtifact,
        error_message: str,
    ) -> IFImageArtifact:
        """Create error artifact with failure information."""
        import uuid

        return IFImageArtifact(
            artifact_id=str(uuid.uuid4()),
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self.processor_id],
            metadata={
                **source.metadata,
                "vision_error": error_message,
            },
            chart_result={
                "success": False,
                "error_message": error_message,
            },
            extraction_confidence=0.0,
        )

    def _create_result_artifact(
        self,
        source: IFArtifact,
        image_data: bytes,
        chart_result: ChartDataResult,
    ) -> IFImageArtifact:
        """Create result artifact with extracted data."""
        import uuid

        # Get image dimensions if possible
        width, height = self._get_image_dimensions(image_data)

        return IFImageArtifact(
            artifact_id=str(uuid.uuid4()),
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self.processor_id],
            metadata={
                **source.metadata,
                "chart_type": chart_result.chart_type,
                "data_point_count": len(chart_result.data_points),
                "bounding_box_count": len(chart_result.bounding_boxes),
            },
            image_data=image_data,
            width=width,
            height=height,
            chart_result=chart_result.to_dict(),
            description=chart_result.description,
            extraction_confidence=chart_result.confidence,
        )

    def _get_image_dimensions(self, image_data: bytes) -> Tuple[int, int]:
        """Get image dimensions. Returns (0, 0) if cannot determine."""
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_data))
            return img.size
        except Exception:
            return (0, 0)

    def teardown(self) -> bool:
        """Release resources."""
        self._client = None
        return True
