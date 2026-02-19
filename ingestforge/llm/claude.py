"""
Anthropic Claude LLM provider.

Uses the Anthropic SDK for API access.
"""

import os
import time
from typing import Any, Generator, Optional

from ingestforge.core.logging import get_logger
from ingestforge.core.retry import llm_retry, RetryError
from ingestforge.shared.lazy_imports import lazy_property
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


class ClaudeClient(LLMClient):
    """
    Anthropic Claude API client.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        max_retries: int = 3,
    ):
        """
        Initialize Claude client.

        Args:
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name
            max_retries: Maximum retry attempts (deprecated, uses @llm_retry decorator)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model_name = model
        # max_retries is deprecated but kept for backward compatibility

    @lazy_property
    def client(self) -> Any:
        """Lazy-load Anthropic client."""
        try:
            from anthropic import Anthropic

            if not self.api_key:
                raise ConfigurationError(
                    "ANTHROPIC_API_KEY not set. Set it in environment or pass to constructor."
                )

            return Anthropic(api_key=self.api_key)

        except ImportError:
            raise ImportError(
                "anthropic is required for Claude. "
                "Install with: pip install anthropic"
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Check if Claude is available."""
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        return self.generate_with_context(
            system_prompt="You are a helpful assistant.",
            user_prompt=prompt,
            config=config,
            **kwargs,
        )

    @llm_retry
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Generate with system prompt and context.

        Rule #4: Reduced from 61 → 32 lines via helper extraction
        """
        config = config or GenerationConfig()

        # Build user message
        user_message = user_prompt
        if context:
            user_message = f"{context}\n\n{user_prompt}"

        try:
            params = self._build_claude_params(system_prompt, user_message, config)
            response = self.client.messages.create(**params)
            return self._extract_and_record_response(response)

        except RetryError as e:
            # Convert RetryError to RateLimitError for backward compatibility
            raise RateLimitError(f"Max retries exceeded: {e.last_exception}")
        except Exception as e:
            error_msg = str(e).lower()
            # Let retry decorator handle rate limits by re-raising
            if any(term in error_msg for term in ["rate limit", "overloaded", "429"]):
                raise  # Will be retried by @llm_retry
            # Non-retryable errors
            raise LLMError(f"Claude generation failed: {e}")

    def _build_claude_params(
        self, system_prompt: str, user_message: str, config: GenerationConfig
    ) -> dict[str, Any]:
        """
        Build Claude API request parameters.

        Rule #4: Extracted to reduce generate_with_context() size
        """
        params = {
            "model": self._model_name,
            "max_tokens": config.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        # Claude uses temperature but not frequency/presence penalty
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.stop_sequences:
            params["stop_sequences"] = config.stop_sequences

        return params

    def _extract_and_record_response(self, response: Any) -> str:
        """
        Extract text from response and record usage.

        Rule #4: Extracted to reduce generate_with_context() size
        """
        output = ""
        for block in response.content:
            if hasattr(block, "text"):
                output += block.text

        if output:
            if hasattr(response, "usage") and response.usage:
                self._record_usage(
                    prompt_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                    completion_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                )
            return output.strip()
        else:
            raise LLMError("Empty response from Claude")

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

        user_message = user_prompt if not context else f"{context}\n\n{user_prompt}"

        try:
            params = self._build_request_params(system_prompt, user_message, config)
            response = self.client.messages.create(**params)
            latency_ms = (time.time() - start_time) * 1000

            text = self._extract_response_text(response)
            finish_reason = self._normalize_finish_reason(response)
            usage = self._extract_usage_stats(response)

            return GenerationResult(
                text=text,
                finish_reason=finish_reason,
                model=response.model,
                latency_ms=latency_ms,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            raise LLMError(f"Claude generation failed: {e}")

    def _build_request_params(
        self,
        system_prompt: str,
        user_message: str,
        config: GenerationConfig,
    ) -> dict[str, Any]:
        """Build API request parameters from config."""
        params = {
            "model": self._model_name,
            "max_tokens": config.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.stop_sequences:
            params["stop_sequences"] = config.stop_sequences

        return params

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from Claude API response."""
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return text.strip()

    def _normalize_finish_reason(self, response: Any) -> str:
        """Normalize Claude's stop_reason to standard format."""
        finish_reason = getattr(response, "stop_reason", "stop")
        if finish_reason == "max_tokens":
            return "length"
        if finish_reason in ("end_turn", "stop_sequence"):
            return "stop"
        return finish_reason

    def _extract_usage_stats(self, response: Any) -> dict[str, Any]:
        """Extract token usage statistics from response."""
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if hasattr(response, "usage") and response.usage:
            usage["prompt_tokens"] = getattr(response.usage, "input_tokens", 0) or 0
            usage["completion_tokens"] = (
                getattr(response.usage, "output_tokens", 0) or 0
            )
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            self._record_usage(usage["prompt_tokens"], usage["completion_tokens"])

        return usage

    @property
    def context_length(self) -> int:
        """Get context length for Claude models (200k for Claude 3)."""
        return get_model_context_length(self._model_name, default=200000)

    @property
    def supports_vision(self) -> bool:
        """Claude 3 models support vision."""
        return "claude-3" in self._model_name.lower()

    @property
    def supports_tools(self) -> bool:
        """Claude 3 models support tool use."""
        return "claude-3" in self._model_name.lower()

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
        """Stream text generation with Claude."""
        config = config or GenerationConfig()

        user_message = user_prompt
        if context:
            user_message = f"{context}\n\n{user_prompt}"

        try:
            with self.client.messages.stream(
                model=self._model_name,
                max_tokens=config.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise LLMError(f"Claude streaming failed: {e}")
