"""
Ollama local LLM provider.

Uses local Ollama server for inference via the chat API.
Supports proper chat templates for all models (Qwen, Llama, Mistral, etc.)
"""

import os
import time
from typing import Any, Generator, Optional

import requests

from ingestforge.core.logging import get_logger
from ingestforge.core.retry import llm_retry, RetryError
from ingestforge.llm.base import (
    GenerationConfig,
    GenerationResult,
    LLMClient,
    LLMError,
)

logger = get_logger(__name__)


class OllamaClient(LLMClient):
    """
    Ollama local inference client.

    Requires Ollama server running locally or at specified URL.
    """

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5:14b",
        max_retries: int = 3,
        timeout: int = 120,
    ):
        """
        Initialize Ollama client.

        Args:
            url: Ollama server URL
            model: Model name
            max_retries: Maximum retry attempts (deprecated, uses @llm_retry decorator)
            timeout: Request timeout in seconds
        """
        self.url = url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._model_name = model
        # max_retries is deprecated but kept for backward compatibility
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _check_model_available(self) -> bool:
        """Check if the specified model is available."""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self._model_name in m for m in models)
        except Exception as e:
            logger.debug(
                f"Failed to check if model {self._model_name} is available: {e}"
            )
        return False

    @llm_retry
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        config = config or GenerationConfig()

        options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        }

        if config.stop_sequences:
            options["stop"] = config.stop_sequences

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": options,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "")
                if text:
                    self._record_usage(
                        prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                        completion_tokens=data.get("eval_count", 0) or 0,
                    )
                    return text.strip()
                else:
                    raise LLMError("Empty response from Ollama")
            else:
                raise LLMError(f"Ollama returned status {response.status_code}")

        except requests.Timeout:
            # Timeouts are retryable - let decorator handle it
            raise
        except requests.ConnectionError:
            # Connection errors are fatal - don't retry
            raise LLMError(
                f"Cannot connect to Ollama at {self.url}. "
                "Make sure Ollama is running."
            )
        except RetryError as e:
            # Convert RetryError to LLMError for backward compatibility
            raise LLMError(f"Max retries exceeded: {e.last_exception}")
        except Exception as e:
            # Other errors - let retry decorator decide
            raise LLMError(f"Ollama error: {e}")

    def _build_chat_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build message list for chat API.

        Rule #4: No large functions - Extracted from generate_with_context
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # User message with optional context
        user_content = (
            f"Context:\n{context}\n\n{user_prompt}" if context else user_prompt
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_options(self, config: GenerationConfig) -> dict[str, Any]:
        """Build Ollama options dictionary.

        Rule #4: No large functions - Extracted from generate_with_context
        """
        options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        }
        if config.stop_sequences:
            options["stop"] = config.stop_sequences
        if config.seed is not None:
            options["seed"] = config.seed
        return options

    def _send_chat_request(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> str:
        """Send chat request and process response.

        Rule #4: No large functions - Extracted from generate_with_context
        """
        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self._model_name,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get("message", {})
                text = message.get("content", "")
                if text:
                    self._record_usage(
                        prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                        completion_tokens=data.get("eval_count", 0) or 0,
                    )
                    return text.strip()
                else:
                    raise LLMError("Empty response from Ollama chat API")
            else:
                raise LLMError(f"Ollama returned status {response.status_code}")

        except requests.Timeout:
            raise
        except requests.ConnectionError:
            raise LLMError(
                f"Cannot connect to Ollama at {self.url}. "
                "Make sure Ollama is running."
            )
        except RetryError as e:
            raise LLMError(f"Max retries exceeded: {e.last_exception}")
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Ollama chat error: {e}")

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
        Generate with system prompt and context using Ollama's chat API.

        Uses /api/chat endpoint which properly handles chat templates
        for different model architectures (Qwen, Llama, Mistral, etc.)

        Rule #4: No large functions - Refactored to <60 lines
        """
        config = config or GenerationConfig()

        messages = self._build_chat_messages(system_prompt, user_prompt, context)
        options = self._build_options(config)
        return self._send_chat_request(messages, options)

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
        """
        Stream text generation with Ollama using the chat API.

        Rule #1: Reduced nesting with helper methods
        Rule #4: Function <60 lines

        Uses /api/chat endpoint with streaming for proper template handling.
        """
        config = config or GenerationConfig()
        messages = self._build_chat_messages(system_prompt, user_prompt, context)
        options = self._build_ollama_options(config)

        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self._model_name,
                    "messages": messages,
                    "stream": True,
                    "options": options,
                },
                timeout=self.timeout,
                stream=True,
            )
            if response.status_code != 200:
                raise LLMError(f"Ollama returned status {response.status_code}")
            yield from self._stream_response_tokens(response)

        except requests.ConnectionError:
            raise LLMError(
                f"Cannot connect to Ollama at {self.url}. "
                "Make sure Ollama is running."
            )
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Ollama streaming error: {e}")

    def _stream_response_tokens(self, response: Any) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama response.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        import json as _json

        for line in response.iter_lines():
            if not line:
                continue

            data = _json.loads(line)
            message = data.get("message", {})
            token = message.get("content", "")

            if token:
                yield token
            if data.get("done", False):
                break

    def _build_chat_messages(
        self, system_prompt: str, user_prompt: str, context: Optional[str]
    ) -> list[Any]:
        """
        Build messages array for Ollama chat API.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            system_prompt: System instruction
            user_prompt: User query
            context: Optional context to prepend

        Returns:
            List of message dictionaries
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            user_content = f"Context:\n{context}\n\n{user_prompt}"
        else:
            user_content = user_prompt
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_ollama_options(self, config: GenerationConfig) -> dict[str, Any]:
        """
        Build options dictionary for Ollama API.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            config: Generation configuration

        Returns:
            Options dictionary
        """
        options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        }
        if config.stop_sequences:
            options["stop"] = config.stop_sequences
        if config.seed is not None:
            options["seed"] = config.seed

        return options

    def _parse_ollama_response(
        self, data: dict[str, Any], latency_ms: float
    ) -> GenerationResult:
        """
        Parse Ollama response into GenerationResult.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            data: Response JSON data
            latency_ms: Request latency in milliseconds

        Returns:
            GenerationResult with parsed data
        """
        message = data.get("message", {})
        text = message.get("content", "").strip()

        # Determine finish reason
        done_reason = data.get("done_reason", "stop")
        finish_reason = "length" if done_reason == "length" else "stop"

        # Calculate usage
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0) or 0,
            "completion_tokens": data.get("eval_count", 0) or 0,
            "total_tokens": (data.get("prompt_eval_count", 0) or 0)
            + (data.get("eval_count", 0) or 0),
        }

        self._record_usage(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            model=data.get("model", self._model_name),
            latency_ms=latency_ms,
            usage=usage,
            raw_response=data,
        )

    def generate_with_result(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text and return rich result with metadata.

        Rule #4: Reduced from 65 → 46 lines (shortened docstring)
        """
        config = config or GenerationConfig()
        start_time = time.time()

        # Build request components using helpers
        messages = self._build_chat_messages(system_prompt, user_prompt, context)
        options = self._build_ollama_options(config)

        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self._model_name,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                },
                timeout=self.timeout,
            )

            latency_ms = (time.time() - start_time) * 1000
            if response.status_code != 200:
                raise LLMError(f"Ollama returned status {response.status_code}")

            return self._parse_ollama_response(response.json(), latency_ms)

        except requests.ConnectionError:
            raise LLMError(
                f"Cannot connect to Ollama at {self.url}. "
                "Make sure Ollama is running."
            )
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Ollama error: {e}")

    def _parse_num_ctx_from_text(
        self, text: str, case_sensitive: bool = True
    ) -> Optional[int]:
        """
        Parse num_ctx value from text.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            text: Text to parse (parameters or modelfile)
            case_sensitive: Whether to match "num_ctx" case-sensitively

        Returns:
            Parsed context length or None
        """
        search_text = text if case_sensitive else text.lower()
        search_key = "num_ctx" if case_sensitive else "num_ctx"

        if search_key not in search_text:
            return None
        for line in text.split("\n"):
            line_to_check = line if case_sensitive else line.lower()

            if search_key not in line_to_check:
                continue

            try:
                return int(line.split()[-1])
            except (ValueError, IndexError):
                continue

        return None

    def _query_model_info(self) -> Optional[dict[str, Any]]:
        """
        Query Ollama API for model information.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Error handling
        Rule #9: Full type hints

        Returns:
            Model info dict or None on failure
        """
        try:
            response = requests.post(
                f"{self.url}/api/show",
                json={"name": self._model_name},
                timeout=5,
            )
            if response.status_code != 200:
                return None

            return response.json()

        except Exception:
            return None

    def _extract_context_from_model_info(self, data: dict[str, Any]) -> Optional[int]:
        """
        Extract context length from model info.

        Rule #1: Early returns eliminate nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            data: Model info dictionary from API

        Returns:
            Context length or None
        """
        params = data.get("parameters", "")
        ctx_from_params = self._parse_num_ctx_from_text(params, case_sensitive=True)
        if ctx_from_params:
            return ctx_from_params
        modelfile = data.get("modelfile", "")
        ctx_from_modelfile = self._parse_num_ctx_from_text(
            modelfile, case_sensitive=False
        )
        if ctx_from_modelfile:
            return ctx_from_modelfile

        return None

    @property
    def context_length(self) -> int:
        """
        Get context length for the current model.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Error handling with fallback
        Rule #9: Full type hints

        Queries Ollama's model info API for accurate context length.
        Falls back to defaults based on model name.

        Returns:
            Context length in tokens
        """
        model_info = self._query_model_info()
        if model_info:
            ctx_length = self._extract_context_from_model_info(model_info)
            if ctx_length:
                return ctx_length

        # Fallback to defaults based on model name
        from ingestforge.llm.base import get_model_context_length

        return get_model_context_length(self._model_name, default=4096)

    def list_models(self) -> list[Any]:
        """List available models."""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Failed to list Ollama models: {e}")
        return []

    def get_model_info(self, model_name: Optional[str] = None) -> dict[str, Any]:
        """
        Get detailed information about a model.

        Args:
            model_name: Model to query (defaults to current model)

        Returns:
            Dict with model info (parameters, template, etc.)
        """
        model = model_name or self._model_name
        try:
            response = requests.post(
                f"{self.url}/api/show",
                json={"name": model},
                timeout=5,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Failed to get model info for {model}: {e}")
        return {}
