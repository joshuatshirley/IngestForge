"""Tests for tool registry.

Tests tool registration and discovery."""

from __future__ import annotations


from ingestforge.agent.tool_registry import (
    ToolCategory,
    ToolParameter,
    ToolMetadata,
    RegisteredTool,
    ToolRegistry,
    create_registry,
    register_builtin_tools,
    MAX_REGISTRY_TOOLS,
    MAX_TOOL_NAME_LENGTH,
)

# ToolCategory tests


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_categories_defined(self) -> None:
        """Test all categories are defined."""
        categories = [c.value for c in ToolCategory]

        assert "search" in categories
        assert "analyze" in categories
        assert "utility" in categories

    def test_category_count(self) -> None:
        """Test correct number of categories."""
        assert len(ToolCategory) == 6


# ToolParameter tests


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_parameter_creation(self) -> None:
        """Test creating a parameter."""
        param = ToolParameter(
            name="query",
            param_type="str",
            description="Search query",
        )

        assert param.name == "query"
        assert param.required is True

    def test_optional_parameter(self) -> None:
        """Test optional parameter."""
        param = ToolParameter(
            name="limit",
            param_type="int",
            description="Max results",
            required=False,
            default=10,
        )

        assert param.required is False
        assert param.default == 10

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        param = ToolParameter(
            name="value",
            param_type="float",
            description="A number",
        )

        d = param.to_dict()

        assert d["name"] == "value"
        assert d["type"] == "float"


# ToolMetadata tests


class TestToolMetadata:
    """Tests for ToolMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test creating metadata."""
        meta = ToolMetadata(
            name="search",
            description="Search for information",
            category=ToolCategory.SEARCH,
        )

        assert meta.name == "search"
        assert meta.category == ToolCategory.SEARCH

    def test_metadata_with_parameters(self) -> None:
        """Test metadata with parameters."""
        params = [
            ToolParameter(name="query", param_type="str", description="Query"),
        ]
        meta = ToolMetadata(
            name="lookup",
            description="Look up data",
            category=ToolCategory.RETRIEVE,
            parameters=params,
        )

        assert len(meta.parameters) == 1

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        meta = ToolMetadata(
            name="analyze",
            description="Analyze data",
            category=ToolCategory.ANALYZE,
            version="2.0.0",
        )

        d = meta.to_dict()

        assert d["name"] == "analyze"
        assert d["version"] == "2.0.0"

    def test_to_prompt_string(self) -> None:
        """Test formatting for LLM."""
        params = [
            ToolParameter(name="text", param_type="str", description="Input"),
            ToolParameter(name="count", param_type="int", description="Count"),
        ]
        meta = ToolMetadata(
            name="process",
            description="Process text",
            category=ToolCategory.TRANSFORM,
            parameters=params,
        )

        prompt = meta.to_prompt_string()

        assert "process" in prompt
        assert "text" in prompt
        assert "count" in prompt

    def test_name_truncation(self) -> None:
        """Test name is truncated."""
        long_name = "x" * 100
        meta = ToolMetadata(
            name=long_name,
            description="Test",
            category=ToolCategory.UTILITY,
        )

        assert len(meta.name) == MAX_TOOL_NAME_LENGTH


# RegisteredTool tests


class TestRegisteredTool:
    """Tests for RegisteredTool dataclass."""

    def test_tool_execution(self) -> None:
        """Test executing a tool."""
        meta = ToolMetadata(
            name="add",
            description="Add numbers",
            category=ToolCategory.UTILITY,
        )
        tool = RegisteredTool(
            metadata=meta,
            _fn=lambda a, b: a + b,
        )

        output = tool.execute(a=2, b=3)

        assert output.is_success is True
        assert output.data == 5

    def test_disabled_tool(self) -> None:
        """Test disabled tool returns error."""
        meta = ToolMetadata(
            name="blocked",
            description="Blocked tool",
            category=ToolCategory.UTILITY,
        )
        tool = RegisteredTool(
            metadata=meta,
            _fn=lambda: "result",
            enabled=False,
        )

        output = tool.execute()

        assert output.is_success is False
        assert "disabled" in output.error_message

    def test_tool_error(self) -> None:
        """Test tool that raises exception."""
        meta = ToolMetadata(
            name="error",
            description="Error tool",
            category=ToolCategory.UTILITY,
        )
        tool = RegisteredTool(
            metadata=meta,
            _fn=lambda: 1 / 0,
        )

        output = tool.execute()

        assert output.is_success is False


# ToolRegistry tests


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_registry_creation(self) -> None:
        """Test creating a registry."""
        registry = ToolRegistry()

        assert registry.tool_count == 0
        assert registry.tool_names == []

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        registry = ToolRegistry()

        result = registry.register(
            name="test",
            fn=lambda: "result",
            description="Test tool",
        )

        assert result is True
        assert registry.tool_count == 1
        assert "test" in registry.tool_names

    def test_register_with_category(self) -> None:
        """Test registering with category."""
        registry = ToolRegistry()

        registry.register(
            name="search",
            fn=lambda q: q,
            description="Search",
            category=ToolCategory.SEARCH,
        )

        tools = registry.list_by_category(ToolCategory.SEARCH)
        assert len(tools) == 1

    def test_register_duplicate(self) -> None:
        """Test registering duplicate name."""
        registry = ToolRegistry()
        registry.register(name="dup", fn=lambda: 1, description="First")

        result = registry.register(
            name="dup",
            fn=lambda: 2,
            description="Second",
        )

        assert result is False
        assert registry.tool_count == 1

    def test_register_empty_name(self) -> None:
        """Test registering empty name."""
        registry = ToolRegistry()

        result = registry.register(
            name="",
            fn=lambda: None,
            description="Empty",
        )

        assert result is False

    def test_unregister(self) -> None:
        """Test unregistering a tool."""
        registry = ToolRegistry()
        registry.register(name="remove", fn=lambda: None, description="Remove")

        result = registry.unregister("remove")

        assert result is True
        assert registry.tool_count == 0

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent tool."""
        registry = ToolRegistry()

        result = registry.unregister("nonexistent")

        assert result is False


class TestToolLookup:
    """Tests for tool lookup methods."""

    def test_get_tool(self) -> None:
        """Test getting a tool."""
        registry = ToolRegistry()
        registry.register(name="fetch", fn=lambda: "data", description="Fetch")

        tool = registry.get("fetch")

        assert tool is not None
        assert tool.name == "fetch"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent tool."""
        registry = ToolRegistry()

        tool = registry.get("missing")

        assert tool is None

    def test_get_as_protocol(self) -> None:
        """Test getting as Tool protocol."""
        registry = ToolRegistry()
        registry.register(name="proto", fn=lambda: None, description="Proto")

        tool = registry.get_as_protocol("proto")

        assert tool is not None
        assert hasattr(tool, "execute")


class TestToolEnableDisable:
    """Tests for tool enable/disable."""

    def test_enable_tool(self) -> None:
        """Test enabling a tool."""
        registry = ToolRegistry()
        registry.register(name="toggle", fn=lambda: None, description="Toggle")
        registry.disable("toggle")

        result = registry.enable("toggle")

        assert result is True
        assert registry.is_enabled("toggle") is True

    def test_disable_tool(self) -> None:
        """Test disabling a tool."""
        registry = ToolRegistry()
        registry.register(name="stop", fn=lambda: None, description="Stop")

        result = registry.disable("stop")

        assert result is True
        assert registry.is_enabled("stop") is False

    def test_list_enabled(self) -> None:
        """Test listing enabled tools."""
        registry = ToolRegistry()
        registry.register(name="a", fn=lambda: None, description="A")
        registry.register(name="b", fn=lambda: None, description="B")
        registry.disable("b")

        enabled = registry.list_enabled()

        assert len(enabled) == 1
        assert enabled[0].name == "a"


class TestToolSearch:
    """Tests for tool search."""

    def test_search_by_name(self) -> None:
        """Test searching by name."""
        registry = ToolRegistry()
        registry.register(name="web_search", fn=lambda: None, description="Search")
        registry.register(name="file_read", fn=lambda: None, description="Read")

        results = registry.search("search")

        assert len(results) == 1
        assert results[0].name == "web_search"

    def test_search_by_description(self) -> None:
        """Test searching by description."""
        registry = ToolRegistry()
        registry.register(name="tool1", fn=lambda: None, description="Analyze data")
        registry.register(name="tool2", fn=lambda: None, description="Generate output")

        results = registry.search("analyze")

        assert len(results) == 1
        assert results[0].name == "tool1"

    def test_search_empty(self) -> None:
        """Test empty search."""
        registry = ToolRegistry()

        results = registry.search("")

        assert results == []


class TestRegistryExport:
    """Tests for registry export."""

    def test_get_prompt_description(self) -> None:
        """Test getting prompt description."""
        registry = ToolRegistry()
        registry.register(
            name="analyze",
            fn=lambda: None,
            description="Analyze content",
            parameters=[
                ToolParameter(name="text", param_type="str", description="Input"),
            ],
        )

        prompt = registry.get_prompt_description()

        assert "analyze" in prompt
        assert "text" in prompt

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        registry = ToolRegistry()
        registry.register(name="export", fn=lambda: None, description="Export")

        d = registry.to_dict()

        assert d["tool_count"] == 1
        assert "export" in d["tools"]

    def test_clear(self) -> None:
        """Test clearing registry."""
        registry = ToolRegistry()
        registry.register(name="temp", fn=lambda: None, description="Temp")

        registry.clear()

        assert registry.tool_count == 0


# Factory function tests


class TestCreateRegistry:
    """Tests for create_registry factory."""

    def test_create(self) -> None:
        """Test creating registry."""
        registry = create_registry()

        assert isinstance(registry, ToolRegistry)
        assert registry.tool_count == 0


class TestRegisterBuiltinTools:
    """Tests for register_builtin_tools."""

    def test_register_builtins(self) -> None:
        """Test registering built-in tools."""
        registry = create_registry()

        count = register_builtin_tools(registry)

        assert count >= 2
        assert "echo" in registry.tool_names
        assert "format" in registry.tool_names

    def test_echo_works(self) -> None:
        """Test echo tool works."""
        registry = create_registry()
        register_builtin_tools(registry)

        tool = registry.get("echo")
        output = tool.execute(text="hello")

        assert output.is_success is True
        assert output.data == "hello"


# Limit tests


class TestRegistryLimits:
    """Tests for registry limits."""

    def test_max_tools_enforced(self) -> None:
        """Test MAX_REGISTRY_TOOLS limit."""
        registry = ToolRegistry()

        # Fill registry
        for i in range(MAX_REGISTRY_TOOLS):
            registry.register(
                name=f"tool_{i}",
                fn=lambda: None,
                description=f"Tool {i}",
            )

        # Next should fail
        result = registry.register(
            name="overflow",
            fn=lambda: None,
            description="Overflow",
        )

        assert result is False
        assert registry.tool_count == MAX_REGISTRY_TOOLS
