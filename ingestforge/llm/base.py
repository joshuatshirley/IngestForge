"""
Base LLM Provider Interface.

This module defines the LLMClient interface that all LLM providers must implement.
This abstraction allows swapping providers (Claude, OpenAI, Ollama, etc.) without
changing application code.

Architecture Context
--------------------
LLM clients are used for text generation throughout the application:

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ QueryPipeline   │     │ QuestionGen     │     │  Summarizer     │
    │ (answer gen)    │     │ (hypotheticals) │     │ (chunk summary) │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┴───────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │     LLMClient       │
                          │   (abstract base)   │
                          └──────────┬──────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ↓                           ↓                           ↓
    ┌─────────┐               ┌─────────┐               ┌─────────┐
    │  Claude │               │  OpenAI │               │  Ollama │
    │  Client │               │  Client │               │  Client │
    └─────────┘               └─────────┘               └─────────┘

Key Data Structures
-------------------
**GenerationConfig**
    Controls generation behavior:
    - max_tokens: Maximum response length
    - temperature: Creativity (0=deterministic, 1=creative)
    - top_p: Nucleus sampling parameter
    - stop_sequences: Strings that stop generation

Exception Hierarchy
-------------------
    LLMError (base)
    ├── RateLimitError      # API rate limits exceeded
    └── ConfigurationError  # Invalid API key, model, etc.

All exceptions are caught by @llm_retry decorator for automatic retry.

Interface Contract
------------------
Implementations must provide:
- generate(): Basic text generation
- generate_with_context(): Generation with system prompt and context
- is_available(): Check if provider is ready

Retry Handling
--------------
All implementations should use the @llm_retry decorator from core.retry:

    from ingestforge.core.retry import llm_retry

    class MyClient(LLMClient):
        @llm_retry
        def generate(self, prompt, config: Any=None) -> None:
            # Implementation - retries on transient errors
            ...

This provides automatic exponential backoff for rate limits and timeouts.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""

    pass


class ConfigurationError(LLMError):
    """Raised when there's a configuration issue."""

    pass


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_tokens: Maximum tokens to generate
        temperature: Creativity (0=deterministic, 1=creative). Default 0.3 for
                     factual accuracy. Use get_generation_config() from factory
                     for command-specific temperatures.
        top_p: Nucleus sampling parameter
        stop_sequences: Strings that stop generation
        frequency_penalty: Reduce repetition of tokens (-2.0 to 2.0)
        presence_penalty: Encourage new topics (-2.0 to 2.0)
        seed: Random seed for reproducibility (if supported)
        json_mode: Force JSON output (if supported by provider)
        grammar: GBNF grammar string for constrained generation (llama-cpp only)
    """

    max_tokens: int = 4096
    temperature: float = 0.3  # Low default for factual accuracy
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None  # -2.0 to 2.0
    presence_penalty: Optional[float] = None  # -2.0 to 2.0
    seed: Optional[int] = None
    json_mode: bool = False
    grammar: Optional[str] = None  # GBNF grammar for constrained output


@dataclass
class GenerationResult:
    """
    Rich result from text generation with metadata.

    Attributes:
        text: The generated text content
        finish_reason: Why generation stopped ("stop", "length", "content_filter", etc.)
        model: The model that was actually used (may differ from requested)
        latency_ms: Time taken for generation in milliseconds
        usage: Token usage dict with prompt_tokens, completion_tokens, total_tokens
        raw_response: Original response object from the provider (for debugging)
    """

    text: str
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    raw_response: Optional[Any] = None

    def __str__(self) -> str:
        """Return just the text for easy string conversion."""
        return self.text

    @property
    def was_truncated(self) -> bool:
        """Check if the response was truncated due to max_tokens."""
        return self.finish_reason == "length"


# Known context lengths for common models
MODEL_CONTEXT_LENGTHS = {
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    # Anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-haiku": 200000,
    # Google
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-1.0-pro": 32768,
    # Local models (typical defaults)
    "qwen2.5": 32768,
    "llama3": 8192,
    "llama2": 4096,
    "mistral": 32768,
    "phi": 2048,
}


def get_model_context_length(model_name: str, default: int = 4096) -> int:
    """
    Get context length for a model name.

    Args:
        model_name: Model identifier (partial match supported)
        default: Default context length if model unknown

    Returns:
        Context length in tokens
    """
    model_lower = model_name.lower()

    # Try exact match first
    for key, length in MODEL_CONTEXT_LENGTHS.items():
        if key in model_lower:
            return length

    return default


class LLMClient(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers must implement this interface.
    """

    def _get_usage(self) -> Dict[str, int]:
        """Get or initialize the usage accumulator."""
        if not hasattr(self, "_usage"):
            self._usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        return self._usage

    def _record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record token usage from a generation call."""
        usage = self._get_usage()
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens
        usage["total_tokens"] += prompt_tokens + completion_tokens

    def get_usage(self) -> Dict[str, int]:
        """
        Get cumulative token usage.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens
        """
        return dict(self._get_usage())

    def reset_usage(self) -> None:
        """Reset token usage counters to zero."""
        usage = self._get_usage()
        usage["prompt_tokens"] = 0
        usage["completion_tokens"] = 0
        usage["total_tokens"] = 0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with system prompt and context.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            context: Additional context (e.g., retrieved documents)
            config: Generation configuration

        Returns:
            Generated text
        """
        pass

    def generate_with_result(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate text and return rich result with metadata.

        Default implementation wraps generate_with_context.
        Providers should override for full metadata support.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            context: Additional context
            config: Generation configuration

        Returns:
            GenerationResult with text and metadata
        """
        start_time = time.time()
        text = self.generate_with_context(
            system_prompt, user_prompt, context, config, **kwargs
        )
        latency_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            text=text,
            finish_reason="stop",  # Assumed if no error
            model=self.model_name,
            latency_ms=latency_ms,
            usage=self.get_usage(),
        )

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    def context_length(self) -> int:
        """
        Get the model's context window size in tokens.

        Override in subclasses for accurate values.
        Default uses lookup table based on model name.
        """
        return get_model_context_length(self.model_name)

    @property
    def supports_json_mode(self) -> bool:
        """Whether this provider supports JSON mode output."""
        return False

    @property
    def supports_vision(self) -> bool:
        """Whether this provider supports image inputs."""
        return False

    @property
    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        return False

    def stream_generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Stream text generation with system prompt and context.

        Yields text chunks as they become available. Default implementation
        falls back to generate_with_context and yields the full result.

        Providers should override this for true token-by-token streaming.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            context: Additional context
            config: Generation configuration

        Yields:
            Text chunks as strings
        """
        result = self.generate_with_context(
            system_prompt, user_prompt, context, config, **kwargs
        )
        yield result

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports true token-by-token streaming."""
        return False
