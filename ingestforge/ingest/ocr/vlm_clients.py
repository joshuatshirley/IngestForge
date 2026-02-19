"""VLM Client Adapters for OCR Escalation.

Provides vision-enabled LLM clients for text extraction from images.
Integrates with OpenAI GPT-4o-mini and Anthropic Claude Vision."""

from __future__ import annotations

import base64
import os
from typing import Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
REQUEST_TIMEOUT_SECONDS = 30


class OpenAIVisionClient:
    """OpenAI GPT-4o Vision client for text extraction.

    Uses GPT-4o-mini or GPT-4o for image-based text extraction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> None:
        """Initialize OpenAI Vision client.

        Args:
            api_key: OpenAI API key
            model_name: Model to use (gpt-4o-mini or gpt-4o)
            max_tokens: Max tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client.

        Returns:
            OpenAI client instance
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def extract_text(
        self,
        image_data: bytes,
        prompt: str,
    ) -> Tuple[str, float]:
        """Extract text from image using GPT-4o Vision.

        Args:
            image_data: Image bytes (PNG, JPEG, etc.)
            prompt: Extraction prompt

        Returns:
            Tuple of (extracted_text, confidence)
        """
        if not image_data:
            logger.error("No image data provided")
            return ("", 0.0)

        if len(image_data) > MAX_IMAGE_SIZE_BYTES:
            logger.error(f"Image too large: {len(image_data)} bytes")
            return ("", 0.0)

        base64_image = base64.b64encode(image_data).decode("utf-8")

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if response.choices and response.choices[0].message.content:
                return (response.choices[0].message.content.strip(), 0.9)
            return ("", 0.0)
        except Exception as e:
            logger.error(f"OpenAI vision extraction failed: {e}")
            return ("", 0.0)


class ClaudeVisionClient:
    """Anthropic Claude Vision client for text extraction.

    Uses Claude 3 Haiku/Sonnet for image-based text extraction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-haiku-20240307",
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> None:
        """Initialize Claude Vision client.

        Args:
            api_key: Anthropic API key
            model_name: Model to use
            max_tokens: Max tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client.

        Returns:
            Anthropic client instance
        """
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def extract_text(
        self,
        image_data: bytes,
        prompt: str,
    ) -> Tuple[str, float]:
        """Extract text from image using Claude Vision.

        Args:
            image_data: Image bytes (PNG, JPEG, etc.)
            prompt: Extraction prompt

        Returns:
            Tuple of (extracted_text, confidence)
        """
        if not image_data:
            logger.error("No image data provided")
            return ("", 0.0)

        if len(image_data) > MAX_IMAGE_SIZE_BYTES:
            logger.error(f"Image too large: {len(image_data)} bytes")
            return ("", 0.0)

        base64_image = base64.b64encode(image_data).decode("utf-8")
        media_type = self._detect_media_type(image_data)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            if response.content:
                text = "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )
                if text:
                    return (text.strip(), 0.9)
            return ("", 0.0)
        except Exception as e:
            logger.error(f"Claude vision extraction failed: {e}")
            return ("", 0.0)

    def _detect_media_type(self, image_data: bytes) -> str:
        """Detect media type from image data.

        Args:
            image_data: Image bytes

        Returns:
            MIME type string
        """
        if image_data.startswith(b"\x89PNG"):
            return "image/png"
        if image_data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_data.startswith(b"GIF8"):
            return "image/gif"
        if image_data.startswith(b"RIFF") and b"WEBP" in image_data[:12]:
            return "image/webp"
        return "image/png"
