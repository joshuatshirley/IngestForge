"""ReAct Grammar for Constrained LLM Output.

Provides GBNF (Generalized Backus-Naur Form) grammar for llama-cpp-python
to constrain LLM output to valid ReAct format. This eliminates parsing
failures by guaranteeing the output structure.

GBNF Grammar Syntax:
- rule ::= expansion       Define a rule
- "literal"                Match exact string
- [characters]             Character class
- rule1 | rule2            Alternation
- rule?                    Optional (0 or 1)
- rule*                    Zero or more
- rule+                    One or more
- (group)                  Grouping"""

from __future__ import annotations

from typing import List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Maximum lengths for grammar-constrained content
MAX_THOUGHT_CHARS = 500
MAX_ACTION_INPUT_CHARS = 200


def build_react_grammar(tools: Optional[List[str]] = None) -> str:
    """Build GBNF grammar for ReAct format with available tools.

    The grammar ensures the LLM outputs exactly:
        Thought: [reasoning text]
        Action: [tool_name]
        Action Input: {"key": "value"}

    Args:
        tools: List of available tool names. If None, allows any identifier.

    Returns:
        GBNF grammar string for llama-cpp
    """
    # Build tool alternation if tools provided
    if tools:
        # Escape any special chars and build alternation
        escaped_tools = [_escape_tool_name(t) for t in tools]
        # Add FINISH as always available
        if "FINISH" not in escaped_tools:
            escaped_tools.append('"FINISH"')
        tool_rule = " | ".join(escaped_tools)
    else:
        # Allow any identifier if no tools specified
        tool_rule = "identifier"

    grammar = f"""
root ::= thought action action-input

# Thought section - reasoning about what to do
thought ::= "Thought: " thought-content "\\n"
thought-content ::= [^\\n]{{1,{MAX_THOUGHT_CHARS}}}

# Action section - which tool to use
action ::= "Action: " tool-name "\\n"
tool-name ::= {tool_rule}

# Action Input section - JSON parameters
action-input ::= "Action Input: " json-object

# JSON object (simplified for ReAct)
json-object ::= "{{" ws json-members? ws "}}"
json-members ::= json-pair ("," ws json-pair)*
json-pair ::= ws string ws ":" ws json-value ws
json-value ::= string | number | "true" | "false" | "null" | json-object | json-array
json-array ::= "[" ws (json-value ("," ws json-value)*)? ws "]"
string ::= "\\"" string-chars* "\\""
string-chars ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= [ \\t]*

# Identifier for unknown tools
identifier ::= [a-zA-Z_][a-zA-Z0-9_]*
"""
    return grammar.strip()


def _escape_tool_name(name: str) -> str:
    """Escape tool name for GBNF grammar.

    Args:
        name: Tool name to escape

    Returns:
        Quoted string safe for GBNF
    """
    # Wrap in quotes for literal match
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# Pre-built grammar for common case (no tool restriction)
REACT_GRAMMAR_GENERIC = build_react_grammar(None)

# Simplified grammar that's more lenient but still structured
REACT_GRAMMAR_LENIENT = """
root ::= thought action action-input

thought ::= "Thought:" ws thought-text newline
thought-text ::= [^\\n]+

action ::= "Action:" ws action-name newline
action-name ::= [a-zA-Z_][a-zA-Z0-9_]*

action-input ::= "Action Input:" ws json-value
json-value ::= json-object | empty-object
json-object ::= "{" [^}]* "}"
empty-object ::= "{}"

ws ::= [ \\t]*
newline ::= "\\n"
""".strip()


def get_react_grammar(
    tools: Optional[List[str]] = None,
    strict: bool = True,
) -> str:
    """Get ReAct grammar with optional tool constraints.

    Args:
        tools: List of available tool names
        strict: If True, use strict JSON parsing. If False, use lenient grammar.

    Returns:
        GBNF grammar string
    """
    if not strict:
        return REACT_GRAMMAR_LENIENT

    if tools:
        return build_react_grammar(tools)

    return REACT_GRAMMAR_GENERIC


class ReActGrammarHelper:
    """Helper for managing ReAct grammar with dynamic tool lists.

    Usage:
        helper = ReActGrammarHelper()
        helper.set_tools(["search_web", "search_knowledge_base", "FINISH"])
        grammar = helper.get_grammar()
    """

    def __init__(self, strict: bool = True) -> None:
        """Initialize grammar helper.

        Args:
            strict: Use strict JSON grammar (True) or lenient (False)
        """
        self._tools: List[str] = []
        self._strict = strict
        self._cached_grammar: Optional[str] = None

    def set_tools(self, tools: List[str]) -> None:
        """Set available tools and invalidate cache.

        Args:
            tools: List of tool names
        """
        self._tools = list(tools)
        self._cached_grammar = None
        logger.debug(f"ReAct grammar tools set: {len(tools)} tools")

    def get_grammar(self) -> str:
        """Get GBNF grammar string.

        Returns:
            GBNF grammar for current tool set
        """
        if self._cached_grammar is None:
            self._cached_grammar = get_react_grammar(
                tools=self._tools if self._tools else None,
                strict=self._strict,
            )
        return self._cached_grammar

    @property
    def tools(self) -> List[str]:
        """Get current tool list."""
        return list(self._tools)
