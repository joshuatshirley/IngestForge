"""LLM Adapter for ReActEngine.

Connects existing LLMClient implementations to the ReActEngine's
ThinkFunction Protocol, enabling LLM-powered autonomous reasoning.
ReAct Format
------------
The adapter expects LLM responses in this format:

    Thought: [reasoning about what to do next]
    Action: [tool_name or "FINISH"]
    Action Input: {"param1": "value1", "param2": "value2"}

Example:
    Thought: I need to find information about solar panels.
    Action: search_documents
    Action Input: {"query": "solar panel efficiency"}

When the task is complete:
    Thought: I now have enough information to answer.
    Action: FINISH
    Action Input: {}

Grammar-Constrained Generation
------------------------------
When using llama-cpp-python, the adapter can use GBNF grammar to
guarantee valid ReAct format output, eliminating parsing failures.
Enable with: LLMThinkAdapter(llm_client, use_grammar=True)
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ingestforge.agent.react_engine import ReActStep
from ingestforge.agent.react_grammar import get_react_grammar
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import GenerationConfig, LLMClient

logger = get_logger(__name__)
MAX_PROMPT_LENGTH = 16000
MAX_HISTORY_STEPS = 10
MAX_TOOL_DESCRIPTION = 200


class LLMThinkAdapter:
    """Adapter connecting LLMClient to ReActEngine ThinkFunction.

    Converts task and conversation history into ReAct-formatted prompts,
    sends them to an LLM, and parses the response to extract thought,
    action, and action_input.

    Grammar-Constrained Mode:
        When use_grammar=True and using llama-cpp-python, the adapter
        uses GBNF grammar to guarantee valid ReAct format output.
        This eliminates parsing failures from smaller/weaker models.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[GenerationConfig] = None,
        use_grammar: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            llm_client: LLM client instance
            config: Optional generation config (uses defaults if None)
            use_grammar: Use GBNF grammar for constrained output (llama-cpp only)
        """
        if llm_client is None:
            raise ValueError("llm_client cannot be None")

        self._llm = llm_client
        self._use_grammar = use_grammar
        self._config = config or GenerationConfig(
            max_tokens=1000,
            temperature=0.7,
            stop_sequences=["Observation:"],
        )

    def __call__(
        self,
        task: str,
        history: list[ReActStep],
        tools: list[str],
    ) -> tuple[str, Optional[str], dict[str, Any]]:
        """Generate thought and action for the current task state.

        Implements the ThinkFunction Protocol.

        Args:
            task: The original task description
            history: Previous ReAct steps
            tools: Available tool names

        Returns:
            Tuple of (thought, action_name, action_input).
            If action_name is None, the agent considers the task complete.
        """
        if not task.strip():
            logger.warning("Empty task provided")
            return ("No task provided", None, {})

        # Build ReAct prompt
        prompt = self._build_react_prompt(task, history, tools)

        # Build config with grammar if enabled
        config = self._get_config_with_grammar(tools)

        # Generate response from LLM
        try:
            response = self._llm.generate(prompt, config)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (f"Error: {e}", None, {})

        # Parse ReAct format response
        return self._parse_react_response(response)

    def _get_config_with_grammar(self, tools: list[str]) -> GenerationConfig:
        """Get generation config with grammar if enabled.

        Args:
            tools: Available tool names for grammar

        Returns:
            GenerationConfig with optional grammar
        """
        if not self._use_grammar:
            return self._config

        # Build grammar constrained to available tools
        grammar = get_react_grammar(tools=tools, strict=True)
        logger.debug(f"Using ReAct grammar with {len(tools)} tools")

        # Create new config with grammar
        return GenerationConfig(
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            stop_sequences=self._config.stop_sequences,
            grammar=grammar,
        )

    def _build_react_prompt(
        self,
        task: str,
        history: list[ReActStep],
        tools: list[str],
    ) -> str:
        """Build a ReAct-formatted prompt.

        Args:
            task: Original task
            history: Previous steps
            tools: Available tools

        Returns:
            Formatted prompt string
        """
        # Build system prompt with ReAct instructions
        system = self._build_system_prompt(tools)

        # Build conversation history
        history_text = self._format_history(history)

        # Combine into final prompt
        prompt = f"""{system}

Task: {task}

{history_text}

Now, provide your next Thought, Action, and Action Input:"""

        return prompt[:MAX_PROMPT_LENGTH]

    def _build_system_prompt(self, tools: list[str]) -> str:
        """Build the system prompt explaining ReAct format.

        Args:
            tools: Available tool names

        Returns:
            System prompt string
        """
        tools_list = "\n".join(f"- {tool}" for tool in tools)

        return f"""You are an autonomous research assistant using the ReAct framework.

IMPORTANT: Use EXACTLY this format for EVERY response:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param": "value"}}

Example response:
Thought: I need to search for information about the topic.
Action: search_web
Action Input: {{"query": "topic to search"}}

Available tools:
{tools_list}

When you have enough information to answer:
Thought: I now have the information needed to answer.
Action: FINISH
Action Input: {{}}

Rules:
1. Start EVERY response with "Thought:"
2. Include "Action:" with a tool name from the list above
3. Include "Action Input:" with JSON parameters
4. Use FINISH when done"""

    def _format_history(self, history: list[ReActStep]) -> str:
        """Format conversation history for context.

        Args:
            history: Previous ReAct steps

        Returns:
            Formatted history string
        """
        if not history:
            return ""
        recent_history = history[-MAX_HISTORY_STEPS:]

        history_parts = []
        for step in recent_history:
            history_parts.append(f"Thought: {step.thought}")
            if step.action:
                history_parts.append(f"Action: {step.action}")
                input_str = json.dumps(step.action_input)
                history_parts.append(f"Action Input: {input_str}")
            if step.observation:
                history_parts.append(f"Observation: {step.observation}")

        return "\n".join(history_parts)

    def _parse_react_response(
        self,
        response: str,
    ) -> tuple[str, Optional[str], dict[str, Any]]:
        """Parse LLM response in ReAct format.

        Uses regex to extract:
        - Thought: reasoning text
        - Action: tool name or "FINISH"
        - Action Input: JSON parameters

        Args:
            response: LLM response text

        Returns:
            Tuple of (thought, action, action_input)
        """
        if not response.strip():
            logger.warning("Empty LLM response")
            return ("No response from LLM", None, {})

        thought = self._extract_thought(response)
        action = self._extract_action(response)
        action_input = self._extract_action_input(response)

        # Handle FINISH action
        if action and action.upper() == "FINISH":
            return (thought, None, {})

        return (thought, action, action_input)

    def _extract_thought(self, text: str) -> str:
        """Extract thought from response text.

        Tries multiple patterns to handle LLM variations:
        1. Standard "Thought:" prefix
        2. Just text before "Action:" (for LLMs that skip "Thought:")
        3. Fallback to first meaningful non-action paragraph

        Args:
            text: Response text

        Returns:
            Thought string or default message
        """
        # Try standard "Thought:" format
        match = re.search(
            r"Thought:\s*(.+?)(?=\n(?:Action|$))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try extracting text before "Action:" as implicit thought
        match = re.search(
            r"^(.+?)(?=\n*Action:)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            thought = match.group(1).strip()
            # Must be meaningful content and not itself an action line
            if (
                thought
                and len(thought) > 10
                and not thought.lower().startswith("action")
            ):
                logger.debug("Using implicit thought (text before Action:)")
                return thought

        # Fallback: use first paragraph as thought (but not action lines)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines:
            # Skip lines that are clearly action/input lines
            if not line.lower().startswith(("action:", "action input:")):
                logger.warning(
                    "No thought found, using first non-action line as thought"
                )
                return line[:500]  # Limit length

        logger.warning("No thought found in response")
        return "Unable to parse thought"

    def _extract_action(self, text: str) -> Optional[str]:
        """Extract action from response text.

        Handles LLM variations:
        - "Action: tool_name"
        - "Action: tool_name\n"
        - Ignores "Action Input:" to avoid false matches

        Args:
            text: Response text

        Returns:
            Action name or None
        """
        # Look for "Action:" followed by tool name (not "Input")
        # Use word boundary to avoid matching "Action Input:"
        match = re.search(
            r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)",
            text,
            re.IGNORECASE,
        )
        if match:
            action = match.group(1).strip()
            # Skip if we accidentally matched "Input" from "Action Input:"
            if action.lower() == "input":
                logger.debug("Skipped 'Input' - likely from 'Action Input:'")
            else:
                return action

        # Try alternative: look for tool names after "Action"
        # Handle case where LLM says "I will use search_web" or similar
        known_tools = [
            "search_web",
            "search_knowledge_base",
            "ingest_document",
            "get_chunk_details",
            "describe_figure",
            "discover_cve",
            "discover_arxiv",
            "discover_law",
            "discover_medical",
            "echo",
            "format",
            "FINISH",
            "complete",
        ]
        text_lower = text.lower()
        for tool in known_tools:
            if tool.lower() in text_lower:
                logger.debug(f"Found tool name by scanning: {tool}")
                return tool if tool != "complete" else "FINISH"

        logger.warning("No action found in response")
        return None

    def _extract_action_input(self, text: str) -> dict[str, Any]:
        """Extract action input from response text.

        Args:
            text: Response text

        Returns:
            Action input dictionary (empty dict if parsing fails)
        """
        match = re.search(
            r"Action Input:\s*(\{.*?\})",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        if not match:
            logger.debug("No action input found")
            return {}

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse action input JSON: {e}")
            return {}


def create_llm_think_adapter(
    llm_client: LLMClient,
    config: Optional[GenerationConfig] = None,
    use_grammar: bool = False,
) -> LLMThinkAdapter:
    """Factory function to create LLM think adapter.

    Args:
        llm_client: LLM client instance
        config: Optional generation config
        use_grammar: Use GBNF grammar for constrained output (llama-cpp only).
                     When True, guarantees valid ReAct format, eliminating
                     parsing failures from smaller models.

    Returns:
        Configured adapter
    """
    return LLMThinkAdapter(
        llm_client=llm_client,
        config=config,
        use_grammar=use_grammar,
    )
