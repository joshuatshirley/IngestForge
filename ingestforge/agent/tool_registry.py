"""Tool Registry & Discovery for autonomous agents.

Manages tool registration, discovery, and metadata
for the ReAct engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ingestforge.agent.react_engine import Tool, ToolOutput, ToolResult
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_REGISTRY_TOOLS = 100
MAX_TOOL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 500
MAX_PARAM_COUNT = 20
MAX_DOMAINS_PER_TOOL = 10


class ToolCategory(Enum):
    """Categories for tool organization."""

    SEARCH = "search"
    RETRIEVE = "retrieve"
    ANALYZE = "analyze"
    GENERATE = "generate"
    TRANSFORM = "transform"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Metadata for a tool parameter."""

    name: str
    param_type: str
    description: str
    required: bool = True
    default: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.param_type,
            "description": self.description[:MAX_DESCRIPTION_LENGTH],
            "required": self.required,
            "default": self.default,
        }


@dataclass
class ToolMetadata:
    """Metadata describing a tool.

    STORY-30: Added domains field for vertical-aware tool filtering.
    """

    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    domains: list[str] = field(default_factory=list)  # STORY-30: Vertical domains

    def __post_init__(self) -> None:
        """Validate metadata on creation."""
        self.name = self.name[:MAX_TOOL_NAME_LENGTH]
        self.description = self.description[:MAX_DESCRIPTION_LENGTH]
        self.parameters = self.parameters[:MAX_PARAM_COUNT]
        self.domains = self.domains[:MAX_DOMAINS_PER_TOOL]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [p.to_dict() for p in self.parameters],
            "examples": self.examples[:5],
            "version": self.version,
            "domains": self.domains,
        }

    def to_prompt_string(self) -> str:
        """Format metadata for LLM prompt."""
        params = ", ".join(p.name for p in self.parameters)
        domain_tag = f" [{', '.join(self.domains)}]" if self.domains else ""
        return f"{self.name}({params}): {self.description}{domain_tag}"


@dataclass
class RegisteredTool:
    """A tool registered in the registry."""

    metadata: ToolMetadata
    _fn: Callable[..., Any]
    enabled: bool = True

    @property
    def name(self) -> str:
        """Tool name from metadata."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Tool description from metadata."""
        return self.metadata.description

    def execute(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool."""
        if not self.enabled:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=f"Tool '{self.name}' is disabled",
            )

        try:
            result = self._fn(**kwargs)

            # If function already returns ToolOutput, use it directly
            if isinstance(result, ToolOutput):
                return result

            # Otherwise wrap in success ToolOutput
            return ToolOutput(status=ToolResult.SUCCESS, data=result)
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}")
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=str(e),
            )


class ToolRegistry:
    """Registry for managing agent tools.

    Provides tool registration, discovery, and metadata
    management for the ReAct engine.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._tools: dict[str, RegisteredTool] = {}

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    @property
    def tool_names(self) -> list[str]:
        """Names of all tools."""
        return list(self._tools.keys())

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str,
        category: ToolCategory = ToolCategory.UTILITY,
        parameters: Optional[list[ToolParameter]] = None,
        domains: Optional[list[str]] = None,
    ) -> bool:
        """Register a new tool.

        STORY-30: Added domains parameter for vertical-aware filtering.

        Args:
            name: Tool name
            fn: Tool function
            description: Tool description
            category: Tool category
            parameters: Parameter metadata
            domains: List of domain verticals this tool serves

        Returns:
            True if registered
        """
        if not name or not name.strip():
            logger.warning("Empty tool name")
            return False

        if len(self._tools) >= MAX_REGISTRY_TOOLS:
            logger.warning(f"Registry full ({MAX_REGISTRY_TOOLS})")
            return False

        name = name[:MAX_TOOL_NAME_LENGTH].strip()

        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered")
            return False

        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            parameters=parameters or [],
            domains=domains or [],
        )

        self._tools[name] = RegisteredTool(metadata=metadata, _fn=fn)
        logger.debug(f"Registered tool: {name}")
        return True

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if unregistered
        """
        if name not in self._tools:
            return False

        del self._tools[name]
        return True

    def get(self, name: str) -> Optional[RegisteredTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None
        """
        return self._tools.get(name)

    def get_as_protocol(self, name: str) -> Optional[Tool]:
        """Get tool as Tool protocol.

        Args:
            name: Tool name

        Returns:
            Tool or None
        """
        return self._tools.get(name)

    def enable(self, name: str) -> bool:
        """Enable a tool.

        Args:
            name: Tool name

        Returns:
            True if enabled
        """
        tool = self._tools.get(name)
        if not tool:
            return False

        tool.enabled = True
        return True

    def disable(self, name: str) -> bool:
        """Disable a tool.

        Args:
            name: Tool name

        Returns:
            True if disabled
        """
        tool = self._tools.get(name)
        if not tool:
            return False

        tool.enabled = False
        return True

    def is_enabled(self, name: str) -> bool:
        """Check if tool is enabled.

        Args:
            name: Tool name

        Returns:
            True if enabled
        """
        tool = self._tools.get(name)
        return tool.enabled if tool else False

    def list_by_category(
        self,
        category: ToolCategory,
    ) -> list[RegisteredTool]:
        """List tools in a category.

        Args:
            category: Tool category

        Returns:
            List of tools
        """
        return [t for t in self._tools.values() if t.metadata.category == category]

    def list_enabled(self) -> list[RegisteredTool]:
        """List all enabled tools.

        Returns:
            List of enabled tools
        """
        return [t for t in self._tools.values() if t.enabled]

    def get_tools_for_domain(self, domain: str) -> list[RegisteredTool]:
        """Get tools that serve a specific domain.

        STORY-30: Domain-aware tool filtering.

        Args:
            domain: Domain name (e.g., "legal", "cyber")

        Returns:
            List of tools serving that domain
        """
        if not domain:
            return []

        domain = domain.lower()
        return [
            t
            for t in self._tools.values()
            if domain in [d.lower() for d in t.metadata.domains]
        ]

    def get_tools_for_domains(self, domains: list[str]) -> list[RegisteredTool]:
        """Get tools that serve any of the specified domains.

        STORY-30: Multi-domain tool filtering.

        Args:
            domains: List of domain names

        Returns:
            List of tools serving any domain (deduplicated)
        """
        if not domains:
            return []

        # Normalize domains to lowercase
        target_domains = {d.lower() for d in domains}

        # Collect tools matching any domain
        matching: list[RegisteredTool] = []
        seen_names: set[str] = set()

        for tool in self._tools.values():
            tool_domains = {d.lower() for d in tool.metadata.domains}

            # Tool matches if it has any overlap with target domains
            if tool_domains & target_domains:
                if tool.name not in seen_names:
                    matching.append(tool)
                    seen_names.add(tool.name)

        return matching

    def get_universal_tools(self) -> list[RegisteredTool]:
        """Get tools with no domain restriction (available to all).

        STORY-30: Tools without domains are universally available.

        Returns:
            List of tools with empty domain list
        """
        return [t for t in self._tools.values() if not t.metadata.domains]

    def activate_for_domains(self, domains: list[str]) -> int:
        """Enable only tools relevant to specified domains.

        STORY-30: Activate domain-specific + universal tools.

        Args:
            domains: List of detected domains

        Returns:
            Number of tools enabled
        """
        # First disable all tools
        for tool in self._tools.values():
            tool.enabled = False

        # Enable universal tools (no domain restriction)
        universal = self.get_universal_tools()
        for tool in universal:
            tool.enabled = True

        # Enable domain-specific tools
        domain_tools = self.get_tools_for_domains(domains)
        for tool in domain_tools:
            tool.enabled = True

        enabled_count = len([t for t in self._tools.values() if t.enabled])
        logger.debug(f"Activated {enabled_count} tools for domains: {domains}")
        return enabled_count

    def search(self, query: str) -> list[RegisteredTool]:
        """Search tools by name or description.

        Args:
            query: Search query

        Returns:
            Matching tools
        """
        if not query:
            return []

        query = query.lower()
        results: list[RegisteredTool] = []

        for tool in self._tools.values():
            name_match = query in tool.name.lower()
            desc_match = query in tool.description.lower()

            if name_match or desc_match:
                results.append(tool)

        return results

    def get_prompt_description(self) -> str:
        """Get formatted tool descriptions for LLM.

        Returns:
            Formatted string
        """
        lines: list[str] = []

        for tool in self.list_enabled():
            lines.append(tool.metadata.to_prompt_string())

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export registry as dictionary.

        Returns:
            Registry data
        """
        return {
            "tool_count": self.tool_count,
            "tools": {
                name: tool.metadata.to_dict() for name, tool in self._tools.items()
            },
        }

    def clear(self) -> None:
        """Remove all tools."""
        self._tools.clear()


def create_registry() -> ToolRegistry:
    """Factory function to create a registry.

    Returns:
        New registry
    """
    return ToolRegistry()


def register_builtin_tools(registry: ToolRegistry) -> int:
    """Register built-in utility tools.

    Args:
        registry: Target registry

    Returns:
        Number of tools registered
    """
    count = 0

    # Echo tool for testing
    if registry.register(
        name="echo",
        fn=lambda text="", message="", **kwargs: text or message,
        description="Echo input text back",
        category=ToolCategory.UTILITY,
        parameters=[
            ToolParameter(
                name="text",
                param_type="str",
                description="Text to echo",
                required=False,
            )
        ],
    ):
        count += 1

    # Format tool
    if registry.register(
        name="format",
        fn=lambda template="", text="", **kwargs: (template or text).format(**kwargs),
        description="Format a template string with values",
        category=ToolCategory.TRANSFORM,
        parameters=[
            ToolParameter(
                name="template",
                param_type="str",
                description="Template with {placeholders}",
                required=False,
            )
        ],
    ):
        count += 1

    return count
