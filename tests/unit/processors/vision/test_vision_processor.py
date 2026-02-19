"""
Tests for VLM Vision Processor.

VLM Vision Processor for chart and diagram analysis.
Tests follow GWT (Given-When-Then) pattern and NASA JPL Power of Ten rules.
"""

from unittest.mock import MagicMock, patch
import pytest

from ingestforge.processors.vision.vision_processor import (
    IFVisionProcessor,
    IFImageArtifact,
    ChartDataResult,
    BoundingBox,
    VisionProcessorConfig,
    OllamaVLMClient,
    parse_chart_extraction_response,
    MAX_IMAGE_SIZE_MB,
    MAX_BOUNDING_BOXES,
)


# =============================================================================
# BOUNDING BOX TESTS
# =============================================================================


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_creation(self):
        """
        GWT:
        Given valid coordinates
        When BoundingBox is created
        Then all fields are set correctly.
        """
        bbox = BoundingBox(
            x=100.0,
            y=200.0,
            width=50.0,
            height=30.0,
            label="Data Point 1",
            confidence=0.95,
        )

        assert bbox.x == 100.0
        assert bbox.y == 200.0
        assert bbox.width == 50.0
        assert bbox.height == 30.0
        assert bbox.label == "Data Point 1"
        assert bbox.confidence == 0.95

    def test_bounding_box_to_dict(self):
        """
        GWT:
        Given BoundingBox
        When to_dict is called
        Then dictionary contains all fields.
        """
        bbox = BoundingBox(x=10, y=20, width=30, height=40, label="test")

        data = bbox.to_dict()

        assert data["x"] == 10
        assert data["y"] == 20
        assert data["label"] == "test"

    def test_bounding_box_from_dict(self):
        """
        GWT:
        Given dictionary with bbox data
        When from_dict is called
        Then BoundingBox is created correctly.
        """
        data = {"x": 5, "y": 10, "width": 15, "height": 20, "label": "test"}

        bbox = BoundingBox.from_dict(data)

        assert bbox.x == 5
        assert bbox.label == "test"


# =============================================================================
# CHART DATA RESULT TESTS
# =============================================================================


class TestChartDataResult:
    """Tests for ChartDataResult dataclass."""

    def test_chart_result_creation(self):
        """
        GWT:
        Given valid chart data
        When ChartDataResult is created
        Then all fields are set.
        """
        result = ChartDataResult(
            chart_type="bar",
            csv_data="x,y\n1,10\n2,20",
            success=True,
            confidence=0.9,
        )

        assert result.chart_type == "bar"
        assert "1,10" in result.csv_data
        assert result.success is True

    def test_chart_result_to_dict(self):
        """
        GWT:
        Given ChartDataResult with bounding boxes
        When to_dict is called
        Then all data is included.
        """
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        result = ChartDataResult(
            chart_type="line",
            bounding_boxes=[bbox],
        )

        data = result.to_dict()

        assert data["chart_type"] == "line"
        assert len(data["bounding_boxes"]) == 1

    def test_chart_result_bounded_data(self):
        """
        GWT:
        Given result with many bounding boxes (JPL Rule #2)
        When to_dict is called
        Then boxes are bounded.
        """
        boxes = [BoundingBox(x=i, y=i, width=1, height=1) for i in range(2000)]
        result = ChartDataResult(bounding_boxes=boxes)

        data = result.to_dict()

        assert len(data["bounding_boxes"]) <= MAX_BOUNDING_BOXES


# =============================================================================
# IMAGE ARTIFACT TESTS
# =============================================================================


class TestIFImageArtifact:
    """Tests for IFImageArtifact."""

    def test_image_artifact_creation(self):
        """
        GWT:
        Given valid image parameters
        When IFImageArtifact is created
        Then artifact is initialized.
        """
        artifact = IFImageArtifact(
            artifact_id="img-001",
            image_path="/path/to/image.png",
            width=800,
            height=600,
        )

        assert artifact.artifact_id == "img-001"
        assert artifact.width == 800

    def test_image_artifact_derive(self):
        """
        GWT:
        Given existing image artifact
        When derive is called
        Then new artifact has lineage.
        """
        original = IFImageArtifact(
            artifact_id="orig-001",
            image_path="/test.png",
        )

        derived = original.derive(
            processor_id="test-proc",
            description="Processed",
        )

        assert derived.parent_id == "orig-001"
        assert "test-proc" in derived.provenance
        assert derived.lineage_depth == 1

    def test_has_bounding_boxes_false(self):
        """
        GWT:
        Given artifact without chart result
        When has_bounding_boxes is checked
        Then False is returned.
        """
        artifact = IFImageArtifact(artifact_id="test")

        assert artifact.has_bounding_boxes is False

    def test_has_bounding_boxes_true(self):
        """
        GWT:
        Given artifact with bounding boxes
        When has_bounding_boxes is checked
        Then True is returned.
        """
        artifact = IFImageArtifact(
            artifact_id="test",
            chart_result={
                "bounding_boxes": [{"x": 0, "y": 0, "width": 10, "height": 10}]
            },
        )

        assert artifact.has_bounding_boxes is True


# =============================================================================
# RESPONSE PARSING TESTS
# =============================================================================


class TestParseChartExtractionResponse:
    """Tests for parse_chart_extraction_response function."""

    def test_parse_valid_response(self):
        """
        GWT:
        Given valid VLM response
        When parsed
        Then ChartDataResult has extracted data.
        """
        response = """CHART_TYPE: bar

CSV_DATA:
month,sales
Jan,100
Feb,150
Mar,200

BOUNDING_BOXES:
label,x,y,width,height
"Jan",100,300,50,100
"Feb",160,250,50,150

DESCRIPTION:
Monthly sales data showing growth trend."""

        result = parse_chart_extraction_response(response)

        assert result.chart_type == "bar"
        assert "Jan,100" in result.csv_data
        assert len(result.bounding_boxes) == 2
        assert result.bounding_boxes[0].label == "Jan"
        assert "Monthly sales" in result.description
        assert result.success is True

    def test_parse_empty_response(self):
        """
        GWT:
        Given empty response
        When parsed
        Then error result is returned (JPL Rule #7).
        """
        result = parse_chart_extraction_response("")

        assert result.success is False
        assert "Empty" in result.error_message

    def test_parse_error_response(self):
        """
        GWT:
        Given ERROR response
        When parsed
        Then error is captured.
        """
        result = parse_chart_extraction_response("ERROR: Cannot read chart")

        assert result.success is False
        assert "Cannot read" in result.error_message

    def test_parse_partial_response(self):
        """
        GWT:
        Given response with only chart type
        When parsed
        Then partial result is extracted.
        """
        response = "CHART_TYPE: pie"

        result = parse_chart_extraction_response(response)

        assert result.chart_type == "pie"
        assert result.success is True


# =============================================================================
# OLLAMA CLIENT TESTS
# =============================================================================


class TestOllamaVLMClient:
    """Tests for OllamaVLMClient."""

    def test_client_initialization(self):
        """
        GWT:
        Given model name and host
        When client is created
        Then configuration is stored.
        """
        client = OllamaVLMClient(
            model_name="llava:13b",
            host="http://localhost:11434",
        )

        assert client._model_name == "llava:13b"

    @patch("requests.get")
    def test_is_available_true(self, mock_get):
        """
        GWT:
        Given Ollama server running
        When is_available is called
        Then True is returned.
        """
        mock_get.return_value.status_code = 200

        client = OllamaVLMClient()
        result = client.is_available()

        assert result is True

    @patch("requests.get")
    def test_is_available_false(self, mock_get):
        """
        GWT:
        Given Ollama server not running
        When is_available is called
        Then False is returned (JPL Rule #7).
        """
        mock_get.side_effect = Exception("Connection refused")

        client = OllamaVLMClient()
        result = client.is_available()

        assert result is False

    @patch("requests.post")
    def test_analyze_image_success(self, mock_post):
        """
        GWT:
        Given valid image and running server
        When analyze_image is called
        Then response is returned.
        """
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "CHART_TYPE: bar\nCSV_DATA:\nx,y\n1,10"
        }

        client = OllamaVLMClient()
        text, confidence = client.analyze_image(b"fake_image", "analyze")

        assert "bar" in text
        assert confidence > 0

    @patch("requests.post")
    def test_analyze_image_failure(self, mock_post):
        """
        GWT:
        Given server error
        When analyze_image is called
        Then empty result is returned (JPL Rule #7).
        """
        mock_post.return_value.status_code = 500

        client = OllamaVLMClient()
        text, confidence = client.analyze_image(b"fake_image", "analyze")

        assert text == ""
        assert confidence == 0.0


# =============================================================================
# VISION PROCESSOR TESTS
# =============================================================================


class TestIFVisionProcessor:
    """Tests for IFVisionProcessor."""

    def test_processor_initialization(self):
        """
        GWT:
        Given default configuration
        When processor is created
        Then defaults are set.
        """
        processor = IFVisionProcessor()

        assert processor.processor_id == "if-vision-processor"
        assert processor.version == "1.0.0"
        assert "chart_extraction" in processor.capabilities

    def test_processor_custom_config(self):
        """
        GWT:
        Given custom configuration
        When processor is created
        Then config is applied.
        """
        config = VisionProcessorConfig(
            backend="llamacpp",
            model_name="llava-v1.5",
        )

        processor = IFVisionProcessor(config=config)

        assert processor._config.backend == "llamacpp"

    @patch.object(OllamaVLMClient, "is_available", return_value=False)
    def test_process_vlm_unavailable(self, mock_available):
        """
        GWT:
        Given VLM backend not available
        When process is called
        Then error artifact is returned (JPL Rule #7).
        """
        processor = IFVisionProcessor()
        artifact = IFImageArtifact(
            artifact_id="test",
            image_data=b"fake_image_data",
        )

        result = processor.process(artifact)

        assert result.chart_result["success"] is False
        assert "not available" in result.chart_result["error_message"]

    def test_process_no_image_data(self):
        """
        GWT:
        Given artifact without image data
        When process is called
        Then error artifact is returned.
        """
        processor = IFVisionProcessor()
        artifact = IFImageArtifact(artifact_id="empty")

        result = processor.process(artifact)

        assert result.chart_result["success"] is False
        assert "Could not read" in result.chart_result["error_message"]

    @patch.object(OllamaVLMClient, "is_available", return_value=True)
    @patch.object(OllamaVLMClient, "analyze_image")
    def test_process_success(self, mock_analyze, mock_available):
        """
        GWT:
        Given valid image and working VLM
        When process is called
        Then chart data is extracted.
        """
        mock_analyze.return_value = (
            "CHART_TYPE: line\nCSV_DATA:\nx,y\n1,10\n2,20",
            0.9,
        )

        processor = IFVisionProcessor()
        artifact = IFImageArtifact(
            artifact_id="test",
            image_data=b"fake_png_data",
        )

        result = processor.process(artifact)

        assert result.chart_result["success"] is True
        assert result.chart_result["chart_type"] == "line"

    def test_process_image_too_large(self):
        """
        GWT:
        Given image exceeding size limit (JPL Rule #2)
        When process is called
        Then error artifact is returned.
        """
        processor = IFVisionProcessor()
        large_data = b"x" * (MAX_IMAGE_SIZE_MB * 1024 * 1024 + 1)
        artifact = IFImageArtifact(
            artifact_id="large",
            image_data=large_data,
        )

        result = processor.process(artifact)

        assert result.chart_result["success"] is False
        assert "exceeds" in result.chart_result["error_message"]

    def test_process_none_artifact(self):
        """
        GWT:
        Given None artifact (JPL Rule #5)
        When process is called
        Then AssertionError is raised.
        """
        processor = IFVisionProcessor()

        with pytest.raises(AssertionError):
            processor.process(None)

    def test_teardown(self):
        """
        GWT:
        Given initialized processor
        When teardown is called
        Then resources are released.
        """
        processor = IFVisionProcessor()
        processor._client = MagicMock()

        result = processor.teardown()

        assert result is True
        assert processor._client is None


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLCompliance:
    """Tests verifying NASA JPL Power of Ten rule compliance."""

    def test_jpl_rule_2_image_size_bounded(self):
        """
        GWT:
        Given oversized image
        When processed
        Then rejected (JPL Rule #2).
        """
        processor = IFVisionProcessor()
        huge_data = b"x" * 20_000_000  # 20MB

        artifact = IFImageArtifact(artifact_id="huge", image_data=huge_data)
        result = processor.process(artifact)

        assert result.chart_result["success"] is False

    def test_jpl_rule_7_graceful_failure(self):
        """
        GWT:
        Given VLM failure
        When processed
        Then graceful error result (JPL Rule #7).
        """
        with patch.object(OllamaVLMClient, "is_available", return_value=True):
            with patch.object(OllamaVLMClient, "analyze_image", return_value=("", 0.0)):
                processor = IFVisionProcessor()
                artifact = IFImageArtifact(
                    artifact_id="test",
                    image_data=b"test",
                )

                result = processor.process(artifact)

                # Should not raise, should return error artifact
                assert isinstance(result, IFImageArtifact)
                assert result.chart_result["success"] is False

    def test_jpl_rule_5_precondition_check(self):
        """
        GWT:
        Given None artifact
        When process is called
        Then AssertionError raised (JPL Rule #5).
        """
        processor = IFVisionProcessor()

        with pytest.raises(AssertionError):
            processor.process(None)

    def test_jpl_rule_9_type_hints(self):
        """
        GWT:
        Given BoundingBox and ChartDataResult
        When inspecting fields
        Then all have type annotations (JPL Rule #9).
        """
        # BoundingBox has typed fields
        bbox = BoundingBox(x=0, y=0, width=1, height=1)
        assert hasattr(bbox, "__dataclass_fields__")

        # ChartDataResult has typed fields
        result = ChartDataResult()
        assert hasattr(result, "__dataclass_fields__")
