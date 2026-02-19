"""
Few-Shot Prompt Injector for IngestForge.

Dynamically update LLM extraction prompts with relevant
few-shot examples to improve accuracy.

Follows NASA JPL Power of Ten rules.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_EXAMPLES = 5
MAX_EXAMPLE_TEXT_LENGTH = 2000
MAX_EXAMPLE_JSON_LENGTH = 4000
MAX_PROMPT_TOKENS = 8000  # Conservative default for most models
CHARS_PER_TOKEN = 4  # Rough estimate

# Model context limits (in tokens)
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "gemini-pro": 32000,
    "_default": 8000,
}

# PII patterns for sanitization
PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (
        re.compile(r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
        "[PHONE]",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CREDIT_CARD]"),
]


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.

    Rule #4: Function under 60 lines.
    Rule #5: Assert preconditions.
    """
    assert text is not None, "text cannot be None"
    return len(text) // CHARS_PER_TOKEN


def _get_context_limit(model_name: str) -> int:
    """
    Get context limit for a model.

    Rule #4: Function under 60 lines.
    """
    # Check exact match
    if model_name in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model_name]

    # Check partial match
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key != "_default" and key in model_name.lower():
            return limit

    return MODEL_CONTEXT_LIMITS["_default"]


def _sanitize_text(text: str) -> str:
    """
    Sanitize PII from text for privacy protection.

    AC: Examples are sanitized/redacted before injection
    if using cloud providers.

    Rule #4: Function under 60 lines.
    Rule #5: Assert preconditions.
    """
    assert text is not None, "text cannot be None"
    result = text

    for pattern, replacement in PII_PATTERNS:
        result = pattern.sub(replacement, result)

    return result


def _truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to max length with ellipsis.

    Rule #4: Function under 60 lines.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


@dataclass
class FewShotExample:
    """
    A single few-shot example for prompt injection.

    Structured example for LLM prompts.

    Attributes:
        chunk_text: Source text that was extracted from.
        entities_json: JSON representation of extracted entities.
        sanitized: Whether the example has been sanitized.
    """

    chunk_text: str
    entities_json: str
    sanitized: bool = False

    def token_estimate(self) -> int:
        """Estimate total tokens for this example."""
        return _estimate_tokens(self.chunk_text) + _estimate_tokens(self.entities_json)


class IFPromptTuner:
    """
    Middleware for injecting few-shot examples into LLM prompts.

    Dynamically update extraction prompts with relevant examples.

    JPL Rule #2: Fixed bounds on examples and prompt size.
    JPL Rule #5: Assert context limits not exceeded.
    JPL Rule #9: Complete type hints.

    Usage:
        tuner = IFPromptTuner(model="gpt-4o-mini", cloud_provider=True)
        examples = registry.find_similar(chunk_embedding, limit=3)
        prompt = tuner.build_prompt(base_prompt, examples)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cloud_provider: bool = True,
        max_examples: int = 3,
        max_prompt_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize the prompt tuner.

        Args:
            model: LLM model name (used for context limit).
            cloud_provider: If True, sanitize PII from examples.
            max_examples: Maximum number of examples to inject.
            max_prompt_tokens: Override for max prompt tokens (default: auto).
        """
        assert max_examples > 0, "max_examples must be positive"
        assert (
            max_examples <= MAX_EXAMPLES
        ), f"max_examples cannot exceed {MAX_EXAMPLES}"

        self.model = model
        self.cloud_provider = cloud_provider
        self.max_examples = max_examples
        self._context_limit = _get_context_limit(model)

        # Allow override but enforce minimum
        if max_prompt_tokens is not None:
            self._max_prompt_tokens = max(max_prompt_tokens, 1000)
        else:
            # Use 50% of context for prompt (leave room for response)
            self._max_prompt_tokens = self._context_limit // 2

    def _format_example(
        self, chunk_text: str, entities: Dict[str, Any]
    ) -> FewShotExample:
        """
        Format a single example for prompt injection.

        Rule #1: Extracted helper.
        Rule #4: Function under 60 lines.
        """
        # Truncate if too long
        text = _truncate_text(chunk_text, MAX_EXAMPLE_TEXT_LENGTH)
        json_str = json.dumps(entities, indent=2, ensure_ascii=False)
        json_str = _truncate_text(json_str, MAX_EXAMPLE_JSON_LENGTH)

        # Sanitize if cloud provider
        sanitized = False
        if self.cloud_provider:
            text = _sanitize_text(text)
            json_str = _sanitize_text(json_str)
            sanitized = True

        return FewShotExample(
            chunk_text=text,
            entities_json=json_str,
            sanitized=sanitized,
        )

    def _select_examples_within_budget(
        self,
        examples: List[FewShotExample],
        available_tokens: int,
    ) -> List[FewShotExample]:
        """
        Select examples that fit within token budget.

        Rule #1: Extracted helper.
        Rule #4: Function under 60 lines.
        """
        selected: List[FewShotExample] = []
        tokens_used = 0

        for example in examples:
            example_tokens = example.token_estimate()
            if tokens_used + example_tokens <= available_tokens:
                selected.append(example)
                tokens_used += example_tokens

            if len(selected) >= self.max_examples:
                break

        return selected

    def prepare_examples(
        self,
        raw_examples: List[Tuple[str, Dict[str, Any]]],
    ) -> List[FewShotExample]:
        """
        Prepare raw examples from semantic matcher for injection.

        Convert [Text, JSON] pairs to FewShotExample objects.

        Rule #4: Function under 60 lines.
        Rule #5: Assert preconditions.

        Args:
            raw_examples: List of (chunk_text, entities_dict) tuples from
                          IFExampleRegistry.find_similar().

        Returns:
            List of prepared FewShotExample objects.
        """
        assert raw_examples is not None, "raw_examples cannot be None"

        prepared: List[FewShotExample] = []
        for chunk_text, entities in raw_examples[:MAX_EXAMPLES]:
            example = self._format_example(chunk_text, entities)
            prepared.append(example)

        return prepared

    def build_few_shot_block(self, examples: List[FewShotExample]) -> str:
        """
        Build the few-shot examples block for prompt injection.

        Creates structured examples block.

        Rule #4: Function under 60 lines.
        Rule #5: Assert preconditions.

        Args:
            examples: Prepared few-shot examples.

        Returns:
            Formatted string for prompt injection.
        """
        assert examples is not None, "examples cannot be None"

        if not examples:
            return ""

        lines = ["## Few-Shot Examples\n"]
        lines.append("Use the following examples as reference for extraction:\n")

        for i, example in enumerate(examples, 1):
            lines.append(f"### Example {i}")
            lines.append("**Source Text:**")
            lines.append(f"```\n{example.chunk_text}\n```")
            lines.append("**Extracted Entities:**")
            lines.append(f"```json\n{example.entities_json}\n```\n")

        return "\n".join(lines)

    def inject_examples(
        self,
        base_prompt: str,
        raw_examples: List[Tuple[str, Dict[str, Any]]],
        reserved_tokens: int = 2000,
    ) -> str:
        """
        Inject few-shot examples into base prompt.

        Main injection method with context limit checking.

        JPL Rule #5: Assert total prompt does not exceed context limits.

        Args:
            base_prompt: Original system prompt.
            raw_examples: Examples from IFExampleRegistry.find_similar().
            reserved_tokens: Tokens reserved for user content (default: 2000).

        Returns:
            Enhanced prompt with few-shot examples.

        Raises:
            AssertionError: If prompt would exceed context limits.
        """
        assert base_prompt is not None, "base_prompt cannot be None"
        assert reserved_tokens > 0, "reserved_tokens must be positive"

        # Calculate available budget
        base_tokens = _estimate_tokens(base_prompt)
        available_for_examples = self._max_prompt_tokens - base_tokens - reserved_tokens

        if available_for_examples <= 0:
            logger.warning("No token budget available for few-shot examples")
            return base_prompt

        # Prepare and select examples within budget
        prepared = self.prepare_examples(raw_examples or [])
        selected = self._select_examples_within_budget(prepared, available_for_examples)

        if not selected:
            logger.debug("No examples fit within token budget")
            return base_prompt

        # Build and inject the few-shot block
        few_shot_block = self.build_few_shot_block(selected)
        enhanced_prompt = f"{base_prompt}\n\n{few_shot_block}"

        # JPL Rule #5: Assert final prompt within limits
        final_tokens = _estimate_tokens(enhanced_prompt) + reserved_tokens
        assert (
            final_tokens <= self._max_prompt_tokens
        ), f"Prompt exceeds max tokens: {final_tokens} > {self._max_prompt_tokens}"

        logger.info(
            f"Injected {len(selected)} few-shot examples "
            f"(~{_estimate_tokens(few_shot_block)} tokens)"
        )

        return enhanced_prompt

    def get_context_info(self) -> Dict[str, Any]:
        """
        Get context limit information for diagnostics.

        Returns:
            Dictionary with model, context_limit, max_prompt_tokens.
        """
        return {
            "model": self.model,
            "context_limit": self._context_limit,
            "max_prompt_tokens": self._max_prompt_tokens,
            "cloud_provider": self.cloud_provider,
            "sanitization_enabled": self.cloud_provider,
        }
