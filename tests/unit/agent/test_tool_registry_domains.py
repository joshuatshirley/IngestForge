"""Tool Registry Domain Awareness Tests.

STORY-30: GWT-style tests verifying domain-aware tool filtering.
- ToolMetadata includes domains
- ToolRegistry filters by domain
- activate_for_domains enables correct tools

Follows NASA JPL Power of Ten rules.
"""


import pytest

from ingestforge.agent.tool_registry import (
    MAX_DOMAINS_PER_TOOL,
    ToolCategory,
    ToolMetadata,
    ToolRegistry,
    create_registry,
)


# --- Fixtures ---


@pytest.fixture
def registry() -> ToolRegistry:
    """Create a fresh tool registry."""
    return create_registry()


@pytest.fixture
def registry_with_domain_tools() -> ToolRegistry:
    """Create registry with domain-tagged tools."""
    reg = create_registry()

    # Universal tool (no domains)
    reg.register(
        name="echo",
        fn=lambda text="": text,
        description="Echo input back",
        category=ToolCategory.UTILITY,
        domains=[],
    )

    # Cyber domain tool
    reg.register(
        name="discover_cve",
        fn=lambda query: f"CVE: {query}",
        description="Search CVE database",
        category=ToolCategory.SEARCH,
        domains=["cyber", "technical"],
    )

    # Legal domain tool
    reg.register(
        name="discover_law",
        fn=lambda query: f"Law: {query}",
        description="Search legal cases",
        category=ToolCategory.SEARCH,
        domains=["legal"],
    )

    # Research domain tool
    reg.register(
        name="discover_arxiv",
        fn=lambda query: f"arXiv: {query}",
        description="Search academic papers",
        category=ToolCategory.SEARCH,
        domains=["research", "technical", "ai_safety"],
    )

    # Medical domain tool
    reg.register(
        name="discover_medical",
        fn=lambda query: f"Medical: {query}",
        description="Search medical literature",
        category=ToolCategory.SEARCH,
        domains=["medical", "wellness"],
    )

    return reg


# --- GWT Scenario 1: ToolMetadata Includes Domains ---


class TestToolMetadataDomains:
    """Tests for ToolMetadata domain support."""

    def test_metadata_has_domains_field(self) -> None:
        """Given ToolMetadata, When created,
        Then domains field exists."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.UTILITY,
        )

        assert hasattr(metadata, "domains")
        assert metadata.domains == []

    def test_metadata_stores_domains(self) -> None:
        """Given domains list, When ToolMetadata created,
        Then domains are stored."""
        metadata = ToolMetadata(
            name="cyber_tool",
            description="Cyber tool",
            category=ToolCategory.SEARCH,
            domains=["cyber", "technical"],
        )

        assert metadata.domains == ["cyber", "technical"]

    def test_metadata_truncates_excess_domains(self) -> None:
        """Given more than MAX_DOMAINS_PER_TOOL domains,
        When ToolMetadata created,
        Then domains are truncated."""
        many_domains = [f"domain_{i}" for i in range(20)]

        metadata = ToolMetadata(
            name="multi_domain",
            description="Multi-domain tool",
            category=ToolCategory.UTILITY,
            domains=many_domains,
        )

        assert len(metadata.domains) == MAX_DOMAINS_PER_TOOL

    def test_metadata_to_dict_includes_domains(self) -> None:
        """Given ToolMetadata with domains, When to_dict called,
        Then domains included in output."""
        metadata = ToolMetadata(
            name="domain_tool",
            description="Domain tool",
            category=ToolCategory.SEARCH,
            domains=["legal", "research"],
        )

        result = metadata.to_dict()

        assert "domains" in result
        assert result["domains"] == ["legal", "research"]

    def test_metadata_prompt_string_includes_domains(self) -> None:
        """Given ToolMetadata with domains, When to_prompt_string called,
        Then domains appear in output."""
        metadata = ToolMetadata(
            name="legal_tool",
            description="Legal tool",
            category=ToolCategory.SEARCH,
            domains=["legal"],
        )

        result = metadata.to_prompt_string()

        assert "[legal]" in result


# --- GWT Scenario 2: Registry Filters by Domain ---


class TestRegistryDomainFiltering:
    """Tests for ToolRegistry domain filtering methods."""

    def test_get_tools_for_domain_returns_matching(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given tools with domains, When get_tools_for_domain called,
        Then only matching tools returned."""
        cyber_tools = registry_with_domain_tools.get_tools_for_domain("cyber")

        assert len(cyber_tools) == 1
        assert cyber_tools[0].name == "discover_cve"

    def test_get_tools_for_domain_case_insensitive(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given domain in different case, When get_tools_for_domain called,
        Then matching still works."""
        legal_tools = registry_with_domain_tools.get_tools_for_domain("LEGAL")

        assert len(legal_tools) == 1
        assert legal_tools[0].name == "discover_law"

    def test_get_tools_for_domain_empty_query(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given empty domain, When get_tools_for_domain called,
        Then empty list returned."""
        result = registry_with_domain_tools.get_tools_for_domain("")

        assert result == []

    def test_get_tools_for_domain_no_matches(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given non-existent domain, When get_tools_for_domain called,
        Then empty list returned."""
        result = registry_with_domain_tools.get_tools_for_domain("unknown_domain")

        assert result == []

    def test_get_tools_for_domains_multiple(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given multiple domains, When get_tools_for_domains called,
        Then tools from all domains returned."""
        tools = registry_with_domain_tools.get_tools_for_domains(["cyber", "legal"])

        tool_names = [t.name for t in tools]
        assert "discover_cve" in tool_names
        assert "discover_law" in tool_names

    def test_get_tools_for_domains_deduplicates(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given overlapping domains, When get_tools_for_domains called,
        Then tools are deduplicated."""
        # discover_cve has both "cyber" and "technical"
        # discover_arxiv has "technical"
        tools = registry_with_domain_tools.get_tools_for_domains(["cyber", "technical"])

        tool_names = [t.name for t in tools]
        # Should not have duplicates
        assert len(tool_names) == len(set(tool_names))


# --- GWT Scenario 3: Universal Tools ---


class TestUniversalTools:
    """Tests for tools without domain restrictions."""

    def test_get_universal_tools_returns_domainless(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given registry with mixed tools, When get_universal_tools called,
        Then only domain-less tools returned."""
        universal = registry_with_domain_tools.get_universal_tools()

        assert len(universal) == 1
        assert universal[0].name == "echo"

    def test_universal_tools_have_empty_domains(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given universal tools, When domains checked,
        Then domains list is empty."""
        universal = registry_with_domain_tools.get_universal_tools()

        for tool in universal:
            assert tool.metadata.domains == []


# --- GWT Scenario 4: Domain Activation ---


class TestDomainActivation:
    """Tests for activate_for_domains method."""

    def test_activate_for_domains_enables_matching(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given domains list, When activate_for_domains called,
        Then matching tools are enabled."""
        registry_with_domain_tools.activate_for_domains(["cyber"])

        assert registry_with_domain_tools.is_enabled("discover_cve")
        assert registry_with_domain_tools.is_enabled("echo")  # Universal

    def test_activate_for_domains_disables_non_matching(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given domains list, When activate_for_domains called,
        Then non-matching tools are disabled."""
        registry_with_domain_tools.activate_for_domains(["cyber"])

        assert not registry_with_domain_tools.is_enabled("discover_law")
        assert not registry_with_domain_tools.is_enabled("discover_medical")

    def test_activate_for_domains_keeps_universal(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given any domains, When activate_for_domains called,
        Then universal tools remain enabled."""
        registry_with_domain_tools.activate_for_domains(["legal"])

        assert registry_with_domain_tools.is_enabled("echo")

    def test_activate_for_domains_returns_count(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given domains, When activate_for_domains called,
        Then returns count of enabled tools."""
        count = registry_with_domain_tools.activate_for_domains(["cyber"])

        # echo (universal) + discover_cve (cyber)
        assert count == 2

    def test_activate_for_multiple_domains(
        self,
        registry_with_domain_tools: ToolRegistry,
    ) -> None:
        """Given multiple domains, When activate_for_domains called,
        Then all matching tools enabled."""
        registry_with_domain_tools.activate_for_domains(["cyber", "medical"])

        assert registry_with_domain_tools.is_enabled("discover_cve")
        assert registry_with_domain_tools.is_enabled("discover_medical")
        assert registry_with_domain_tools.is_enabled("echo")


# --- GWT Scenario 5: Registration with Domains ---


class TestRegistrationWithDomains:
    """Tests for registering tools with domains."""

    def test_register_with_domains(self, registry: ToolRegistry) -> None:
        """Given tool with domains, When registered,
        Then domains are stored."""
        registry.register(
            name="legal_search",
            fn=lambda q: q,
            description="Search legal docs",
            category=ToolCategory.SEARCH,
            domains=["legal", "government"],
        )

        tool = registry.get("legal_search")
        assert tool is not None
        assert tool.metadata.domains == ["legal", "government"]

    def test_register_without_domains(self, registry: ToolRegistry) -> None:
        """Given tool without domains, When registered,
        Then domains list is empty."""
        registry.register(
            name="echo",
            fn=lambda t: t,
            description="Echo",
            category=ToolCategory.UTILITY,
        )

        tool = registry.get("echo")
        assert tool is not None
        assert tool.metadata.domains == []


# --- JPL Compliance Tests ---


class TestJPLComplianceDomainMethods:
    """JPL Power of Ten compliance tests for domain methods."""

    def test_get_tools_for_domain_under_60_lines(self) -> None:
        """Given get_tools_for_domain, When lines counted,
        Then count < 60."""
        import inspect

        source = inspect.getsource(ToolRegistry.get_tools_for_domain)
        lines = [
            line
            for line in source.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Method has {len(lines)} lines"

    def test_activate_for_domains_under_60_lines(self) -> None:
        """Given activate_for_domains, When lines counted,
        Then count < 60."""
        import inspect

        source = inspect.getsource(ToolRegistry.activate_for_domains)
        lines = [
            line
            for line in source.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        assert len(lines) < 60, f"Method has {len(lines)} lines"


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_metadata_domains_covered(self) -> None:
        """GWT Scenario 1 (Metadata Domains) is tested."""
        assert hasattr(TestToolMetadataDomains, "test_metadata_has_domains_field")

    def test_scenario_2_domain_filtering_covered(self) -> None:
        """GWT Scenario 2 (Domain Filtering) is tested."""
        assert hasattr(
            TestRegistryDomainFiltering, "test_get_tools_for_domain_returns_matching"
        )

    def test_scenario_3_universal_tools_covered(self) -> None:
        """GWT Scenario 3 (Universal Tools) is tested."""
        assert hasattr(
            TestUniversalTools, "test_get_universal_tools_returns_domainless"
        )

    def test_scenario_4_domain_activation_covered(self) -> None:
        """GWT Scenario 4 (Domain Activation) is tested."""
        assert hasattr(
            TestDomainActivation, "test_activate_for_domains_enables_matching"
        )

    def test_scenario_5_registration_covered(self) -> None:
        """GWT Scenario 5 (Registration with Domains) is tested."""
        assert hasattr(TestRegistrationWithDomains, "test_register_with_domains")
