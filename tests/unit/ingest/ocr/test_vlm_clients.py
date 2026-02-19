"""Tests for VLM client adapters.

Tests OpenAI and Claude vision clients for OCR escalation."""

from __future__ import annotations


from ingestforge.ingest.ocr.vlm_clients import (
    OpenAIVisionClient,
    ClaudeVisionClient,
)


class TestOpenAIVisionClient:
    """Tests for OpenAI Vision client."""

    def test_client_creation(self) -> None:
        """Test creating OpenAI vision client."""
        client = OpenAIVisionClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model_name == "gpt-4o-mini"

    def test_client_with_custom_model(self) -> None:
        """Test client with custom model."""
        client = OpenAIVisionClient(api_key="test-key", model_name="gpt-4o")
        assert client.model_name == "gpt-4o"

    def test_extract_text_no_image(self) -> None:
        """Test extraction with no image data."""
        client = OpenAIVisionClient(api_key="test-key")
        text, confidence = client.extract_text(b"", "Extract text")

        assert text == ""
        assert confidence == 0.0

    def test_extract_text_oversized_image(self) -> None:
        """Test extraction with oversized image."""
        client = OpenAIVisionClient(api_key="test-key")
        # Create 11MB of fake data
        large_data = b"x" * (11 * 1024 * 1024)
        text, confidence = client.extract_text(large_data, "Extract")

        assert text == ""
        assert confidence == 0.0


class TestClaudeVisionClient:
    """Tests for Claude Vision client."""

    def test_client_creation(self) -> None:
        """Test creating Claude vision client."""
        client = ClaudeVisionClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model_name == "claude-3-haiku-20240307"

    def test_client_with_custom_model(self) -> None:
        """Test client with custom model."""
        client = ClaudeVisionClient(
            api_key="test-key", model_name="claude-3-sonnet-20240307"
        )
        assert client.model_name == "claude-3-sonnet-20240307"

    def test_extract_text_no_image(self) -> None:
        """Test extraction with no image data."""
        client = ClaudeVisionClient(api_key="test-key")
        text, confidence = client.extract_text(b"", "Extract text")

        assert text == ""
        assert confidence == 0.0

    def test_detect_media_type_png(self) -> None:
        """Test PNG detection."""
        client = ClaudeVisionClient(api_key="test-key")
        png_header = b"\x89PNG\r\n\x1a\n"
        media_type = client._detect_media_type(png_header)

        assert media_type == "image/png"

    def test_detect_media_type_jpeg(self) -> None:
        """Test JPEG detection."""
        client = ClaudeVisionClient(api_key="test-key")
        jpeg_header = b"\xff\xd8\xff\xe0"
        media_type = client._detect_media_type(jpeg_header)

        assert media_type == "image/jpeg"

    def test_detect_media_type_unknown(self) -> None:
        """Test unknown format defaults to PNG."""
        client = ClaudeVisionClient(api_key="test-key")
        unknown = b"UNKNOWN"
        media_type = client._detect_media_type(unknown)

        assert media_type == "image/png"
