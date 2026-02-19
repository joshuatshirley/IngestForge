"""
OpenAI GPT LLM provider.

Uses the OpenAI SDK for API access.
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


class OpenAIClient(LLMClient):
    """
    OpenAI API client.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model: Model name
            max_retries: Maximum retry attempts (deprecated, uses @llm_retry decorator)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model_name = model
        # max_retries is deprecated but kept for backward compatibility

    @lazy_property
    def client(self) -> Any:
        """Lazy-load OpenAI client."""
        try:
            from openai import OpenAI

            if not self.api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY not set. Set it in environment or pass to constructor."
                )

            return OpenAI(api_key=self.api_key)

        except ImportError:
            raise ImportError(
                "openai is required for OpenAI. " "Install with: pip install openai"
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
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

    def _build_request_params(self, config: GenerationConfig) -> dict[str, Any]:
        """Build request parameters from config, supporting extended options."""
        params = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        # Extended parameters
        if config.frequency_penalty is not None:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            params["presence_penalty"] = config.presence_penalty
        if config.seed is not None:
            params["seed"] = config.seed
        if config.stop_sequences:
            params["stop"] = config.stop_sequences

        # JSON mode
        if config.json_mode:
            params["response_format"] = {"type": "json_object"}

        return params

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
        config = config or GenerationConfig()

        messages = [{"role": "system", "content": system_prompt}]

        # Build user message
        user_message = user_prompt
        if context:
            user_message = f"{context}\n\n{user_prompt}"

        messages.append({"role": "user", "content": user_message})

        try:
            params = self._build_request_params(config)
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **params,
            )

            if response.choices and response.choices[0].message.content:
                if response.usage:
                    self._record_usage(
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                    )
                return response.choices[0].message.content.strip()
            else:
                raise LLMError("Empty response from OpenAI")

        except RetryError as e:
            # Convert RetryError to RateLimitError for backward compatibility
            raise RateLimitError(f"Max retries exceeded: {e.last_exception}")
        except Exception as e:
            error_msg = str(e).lower()
            # Let retry decorator handle rate limits by re-raising
            if any(term in error_msg for term in ["rate limit", "429", "quota"]):
                raise  # Will be retried by @llm_retry
            # Non-retryable errors
            raise LLMError(f"OpenAI generation failed: {e}")

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

        messages = [{"role": "system", "content": system_prompt}]
        user_message = user_prompt if not context else f"{context}\n\n{user_prompt}"
        messages.append({"role": "user", "content": user_message})

        try:
            params = self._build_request_params(config)
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **params,
            )

            latency_ms = (time.time() - start_time) * 1000

            text = ""
            finish_reason = None
            if response.choices:
                choice = response.choices[0]
                text = (choice.message.content or "").strip()
                finish_reason = (
                    choice.finish_reason
                )  # "stop", "length", "content_filter"

            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens or 0
                usage["completion_tokens"] = response.usage.completion_tokens or 0
                usage["total_tokens"] = response.usage.total_tokens or 0
                self._record_usage(usage["prompt_tokens"], usage["completion_tokens"])

            return GenerationResult(
                text=text,
                finish_reason=finish_reason,
                model=response.model,
                latency_ms=latency_ms,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            raise LLMError(f"OpenAI generation failed: {e}")

    @property
    def context_length(self) -> int:
        """Get context length for the current model."""
        return get_model_context_length(self._model_name, default=128000)

    @property
    def supports_json_mode(self) -> bool:
        """OpenAI supports JSON mode."""
        return True

    @property
    def supports_vision(self) -> bool:
        """GPT-4o models support vision."""
        return (
            "gpt-4o" in self._model_name.lower() or "vision" in self._model_name.lower()
        )

    @property
    def supports_tools(self) -> bool:
        """OpenAI models support function calling."""
        return True

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
        """Stream text generation with OpenAI."""
        config = config or GenerationConfig()

        messages = [{"role": "system", "content": system_prompt}]
        user_message = user_prompt
        if context:
            user_message = f"{context}\n\n{user_prompt}"
        messages.append({"role": "user", "content": user_message})

        try:
            stream = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"OpenAI streaming failed: {e}")
