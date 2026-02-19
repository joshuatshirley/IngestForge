"""
Google Gemini LLM provider.

Supports the Gemini API via google-generativeai package.
"""

import os
import time
from typing import Any, Generator, Optional

from ingestforge.core.logging import get_logger
from ingestforge.core.retry import llm_retry, RetryError
from ingestforge.llm.base import (
    ConfigurationError,
    GenerationConfig,
    GenerationResult,
    LLMClient,
    LLMError,
    RateLimitError,
    get_model_context_length,
)

logger = get_logger(__name__)


class GeminiClient(LLMClient):
    """
    Google Gemini API client.

    Requires GEMINI_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        max_retries: int = 3,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: API key (defaults to GEMINI_API_KEY env var)
            model: Model name
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy-load Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                if not self.api_key:
                    raise ConfigurationError(
                        "GEMINI_API_KEY not set. Set it in environment or pass to constructor."
                    )

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self._model_name)

            except ImportError:
                raise ImportError(
                    "google-generativeai is required for Gemini. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        config = config or GenerationConfig()

        generation_config = {
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.stop_sequences:
            generation_config["stop_sequences"] = config.stop_sequences

        return self._generate_with_retry(prompt, generation_config)

    @llm_retry
    def _generate_with_retry(self, prompt: str, generation_config: dict) -> str:
        """Internal generation with retry logic."""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
            )

            if response.text:
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    self._record_usage(
                        prompt_tokens=getattr(
                            response.usage_metadata, "prompt_token_count", 0
                        )
                        or 0,
                        completion_tokens=getattr(
                            response.usage_metadata, "candidates_token_count", 0
                        )
                        or 0,
                    )
                return response.text.strip()
            else:
                raise LLMError("Empty response from Gemini")

        except RetryError:
            raise RateLimitError("Max retries exceeded for Gemini API")
        except Exception as e:
            error_msg = str(e).lower()
            # Let retry handle rate limits
            if any(
                term in error_msg
                for term in ["rate limit", "quota", "429", "resource exhausted"]
            ):
                raise  # Will be retried
            # Non-retryable errors
            raise LLMError(f"Gemini generation failed: {e}")

    @llm_retry
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """Generate with system prompt and context."""
        parts = [system_prompt]

        if context:
            parts.append(f"\n\nContext:\n{context}")

        parts.append(f"\n\n{user_prompt}")

        full_prompt = "\n".join(parts)
        return self.generate(full_prompt, config, **kwargs)

    @property
    def supports_streaming(self) -> bool:
        return True

    def stream_generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream text generation with Gemini."""
        config = config or GenerationConfig()

        parts = [system_prompt]
        if context:
            parts.append(f"\n\nContext:\n{context}")
        parts.append(f"\n\n{user_prompt}")
        full_prompt = "\n".join(parts)

        generation_config = {
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.stop_sequences:
            generation_config["stop_sequences"] = config.stop_sequences

        try:
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(f"Gemini streaming failed: {e}")

    def _extract_finish_reason(self, response: Any) -> str:
        """
        Extract finish reason from Gemini response.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            response: Gemini API response

        Returns:
            Finish reason string
        """
        if not hasattr(response, "candidates") or not response.candidates:
            return "stop"

        candidate = response.candidates[0]
        if not hasattr(candidate, "finish_reason"):
            return "stop"

        fr = str(candidate.finish_reason)

        # Determine reason from string
        if "MAX_TOKENS" in fr or "LENGTH" in fr:
            return "length"
        if "SAFETY" in fr:
            return "content_filter"

        return "stop"

    def _extract_usage_metadata(self, response: Any) -> dict[str, Any]:
        """
        Extract usage metadata from Gemini response.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            response: Gemini API response

        Returns:
            Usage dictionary with token counts
        """
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if not hasattr(response, "usage_metadata") or not response.usage_metadata:
            return usage

        usage["prompt_tokens"] = (
            getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        )
        usage["completion_tokens"] = (
            getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        )
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        self._record_usage(usage["prompt_tokens"], usage["completion_tokens"])

        return usage

    def generate_with_result(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text and return rich result with metadata."""
        config = config or GenerationConfig()
        start_time = time.time()

        parts = [system_prompt]
        if context:
            parts.append(f"\n\nContext:\n{context}")
        parts.append(f"\n\n{user_prompt}")
        full_prompt = "\n".join(parts)

        generation_config = {
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.stop_sequences:
            generation_config["stop_sequences"] = config.stop_sequences

        try:
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            latency_ms = (time.time() - start_time) * 1000

            text = response.text.strip() if response.text else ""

            # Determine finish reason from candidates
            finish_reason = self._extract_finish_reason(response)

            # Extract usage metadata
            usage = self._extract_usage_metadata(response)

            return GenerationResult(
                text=text,
                finish_reason=finish_reason,
                model=self._model_name,
                latency_ms=latency_ms,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            raise LLMError(f"Gemini generation failed: {e}")

    @property
    def context_length(self) -> int:
        """Get context length for Gemini models."""
        return get_model_context_length(self._model_name, default=1000000)

    @property
    def supports_vision(self) -> bool:
        """Gemini 1.5 models support vision."""
        return "1.5" in self._model_name or "vision" in self._model_name.lower()

    @property
    def supports_json_mode(self) -> bool:
        """Gemini supports JSON mode via response_mime_type."""
        return True

    @property
    def supports_tools(self) -> bool:
        """Gemini models support function calling."""
        return True
