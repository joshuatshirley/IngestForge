"""Mock LLM implementations for testing agent systems.

This module provides mock LLM clients for testing ReAct agents
and other components that require LLM responses.
Usage Example
-------------
    from tests.fixtures.agents import MockLLM

    def test_agent():
        llm = MockLLM()
        llm.set_responses([
            "Thought: I should search\\nAction: search\\nInput: query",
            "Thought: I have the answer\\nFinal Answer: result"
        ])

        response1 = llm.generate("What is the task?")
        response2 = llm.generate("Continue thinking")
        # Third call raises AssertionError (no more responses)
"""

from __future__ import annotations

from typing import Any, List, Optional

from ingestforge.llm.base import GenerationConfig, LLMClient


class MockLLM(LLMClient):
    """Mock LLM client for testing agents.

    Provides deterministic responses from a pre-configured sequence.
    Tracks call history for verification and raises AssertionError
    if more calls are made than responses provided (Rule #5).

    Attributes:
        call_count: Number of times generate() was called
        call_history: List of all prompts passed to generate()
        response_history: List of all responses returned
    """

    def __init__(self) -> None:
        """Initialize mock LLM with empty state."""
        self._responses: List[str] = []
        self._call_index: int = 0
        self._call_history: List[str] = []
        self._response_history: List[str] = []
        self._model_name: str = "mock-llm"
        self._is_available: bool = True

    def set_responses(self, sequence: List[str]) -> None:
        """Set sequence of responses to return.

        Args:
            sequence: List of response strings to return in order

        Raises:
            ValueError: If sequence is empty or contains non-strings

        Example:
            llm.set_responses(["response 1", "response 2"])
        """
        if not sequence:
            raise ValueError("Response sequence cannot be empty")

        if not isinstance(sequence, list):
            raise ValueError("Sequence must be a list")

        for i, response in enumerate(sequence):
            if not isinstance(response, str):
                raise ValueError(
                    f"Response at index {i} must be a string, got {type(response)}"
                )

        self._responses = sequence.copy()
        self._call_index = 0

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generate response from pre-configured sequence.

        Args:
            prompt: Input prompt (recorded in history)
            config: Ignored (for interface compatibility)
            **kwargs: Ignored (for interface compatibility)

        Returns:
            Next response in sequence

        Raises:
            AssertionError: If called more times than responses available
            ValueError: If prompt is not a string

        Example:
            response = llm.generate("What is 2+2?")
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be a string, got {type(prompt)}")

        # Record call
        self._call_history.append(prompt)
        assert self._call_index < len(self._responses), (
            f"MockLLM called {self._call_index + 1} times but only "
            f"{len(self._responses)} responses configured. "
            f"Use set_responses() to provide more responses."
        )

        # Get next response
        response = self._responses[self._call_index]
        self._call_index += 1

        # Record response
        self._response_history.append(response)

        return response

    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generate with context (delegates to generate).

        Args:
            system_prompt: System instructions
            user_prompt: User query
            context: Additional context (ignored)
            config: Generation config (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Next response in sequence

        Raises:
            AssertionError: If called more times than responses available
        """
        # Combine prompts for history tracking
        combined = f"System: {system_prompt}\nUser: {user_prompt}"
        if context:
            combined += f"\nContext: {context[:100]}..."

        return self.generate(combined, config, **kwargs)

    def is_available(self) -> bool:
        """Check if mock LLM is available.

        Returns:
            True if available (default), False if disabled
        """
        return self._is_available

    def set_availability(self, available: bool) -> None:
        """Set availability status for testing.

        Args:
            available: Whether LLM should report as available
        """
        self._is_available = available

    @property
    def model_name(self) -> str:
        """Get model name.

        Returns:
            Mock model name
        """
        return self._model_name

    def set_model_name(self, name: str) -> None:
        """Set model name for testing.

        Args:
            name: Model name to report
        """
        if not isinstance(name, str):
            raise ValueError(f"Model name must be a string, got {type(name)}")
        self._model_name = name

    @property
    def call_count(self) -> int:
        """Get number of generate() calls made.

        Returns:
            Total number of calls
        """
        return len(self._call_history)

    @property
    def call_history(self) -> List[str]:
        """Get history of all prompts.

        Returns:
            List of prompts (copy to prevent modification)
        """
        return self._call_history.copy()

    @property
    def response_history(self) -> List[str]:
        """Get history of all responses.

        Returns:
            List of responses (copy to prevent modification)
        """
        return self._response_history.copy()

    def get_last_prompt(self) -> str:
        """Get the most recent prompt.

        Returns:
            Last prompt string, or empty string if no calls made

        Example:
            llm.generate("test prompt")
            assert llm.get_last_prompt() == "test prompt"
        """
        if not self._call_history:
            return ""
        return self._call_history[-1]

    def reset(self) -> None:
        """Reset all state except configured responses.

        Clears call history and resets call index to start of sequence.
        Responses remain configured for reuse.
        """
        self._call_index = 0
        self._call_history.clear()
        self._response_history.clear()

    def clear_responses(self) -> None:
        """Clear all configured responses and reset state."""
        self._responses.clear()
        self.reset()

    def get_remaining_count(self) -> int:
        """Get number of responses remaining.

        Returns:
            Count of responses not yet returned
        """
        return len(self._responses) - self._call_index

    def has_responses_remaining(self) -> bool:
        """Check if there are responses remaining.

        Returns:
            True if responses available, False otherwise
        """
        return self._call_index < len(self._responses)
