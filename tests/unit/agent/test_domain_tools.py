"""Domain Discovery Tools Tests.

STORY-30: GWT-style tests for domain-specific discovery tools.
"""


from ingestforge.agent.domain_tools import (
    discover_cve,
    discover_arxiv,
    discover_law,
    discover_medical,
    register_domain_tools,
)
from ingestforge.agent.react_engine import ToolResult
from ingestforge.agent.tool_registry import create_registry


# --- GWT Scenario 1: CVE Discovery ---


class TestDiscoverCVE:
    """Tests for discover_cve tool."""

    def test_discover_cve_success(self) -> None:
        """Given valid query, When discover_cve called,
        Then success result returned."""
        result = discover_cve("buffer overflow")

        assert result.status == ToolResult.SUCCESS
        assert "CVE Discovery" in result.data

    def test_discover_cve_empty_query(self) -> None:
        """Given empty query, When discover_cve called,
        Then error returned."""
        result = discover_cve("")

        assert result.status == ToolResult.ERROR
        assert "empty" in result.error_message.lower()

    def test_discover_cve_with_severity(self) -> None:
        """Given query with severity, When discover_cve called,
        Then severity included in output."""
        result = discover_cve("sql injection", severity="critical")

        assert result.status == ToolResult.SUCCESS
        assert "critical" in result.data


# --- GWT Scenario 2: arXiv Discovery ---


class TestDiscoverArxiv:
    """Tests for discover_arxiv tool."""

    def test_discover_arxiv_success(self) -> None:
        """Given valid query, When discover_arxiv called,
        Then success result returned."""
        result = discover_arxiv("transformer architecture")

        assert result.status == ToolResult.SUCCESS
        assert "arXiv Search" in result.data

    def test_discover_arxiv_empty_query(self) -> None:
        """Given empty query, When discover_arxiv called,
        Then error returned."""
        result = discover_arxiv("")

        assert result.status == ToolResult.ERROR

    def test_discover_arxiv_with_max_results(self) -> None:
        """Given max_results, When discover_arxiv called,
        Then max_results shown in output."""
        result = discover_arxiv("neural networks", max_results=10)

        assert result.status == ToolResult.SUCCESS
        assert "10" in result.data


# --- GWT Scenario 3: Legal Discovery ---


class TestDiscoverLaw:
    """Tests for discover_law tool."""

    def test_discover_law_success(self) -> None:
        """Given valid query, When discover_law called,
        Then success result returned."""
        result = discover_law("first amendment")

        assert result.status == ToolResult.SUCCESS
        assert "Legal Discovery" in result.data

    def test_discover_law_with_jurisdiction(self) -> None:
        """Given jurisdiction, When discover_law called,
        Then jurisdiction shown in output."""
        result = discover_law("patent infringement", jurisdiction="federal")

        assert result.status == ToolResult.SUCCESS
        assert "federal" in result.data


# --- GWT Scenario 4: Medical Discovery ---


class TestDiscoverMedical:
    """Tests for discover_medical tool."""

    def test_discover_medical_success(self) -> None:
        """Given valid query, When discover_medical called,
        Then success result returned."""
        result = discover_medical("diabetes treatment")

        assert result.status == ToolResult.SUCCESS
        assert "Medical Discovery" in result.data

    def test_discover_medical_with_source(self) -> None:
        """Given source filter, When discover_medical called,
        Then source shown in output."""
        result = discover_medical("aspirin", source="drugs")

        assert result.status == ToolResult.SUCCESS
        assert "drugs" in result.data


# --- GWT Scenario 5: Registration ---


class TestDomainToolRegistration:
    """Tests for domain tool registration."""

    def test_register_domain_tools_count(self) -> None:
        """Given empty registry, When register_domain_tools called,
        Then 4 tools registered."""
        registry = create_registry()
        count = register_domain_tools(registry)

        assert count == 4

    def test_registered_tools_have_domains(self) -> None:
        """Given registered domain tools, When domains checked,
        Then each tool has appropriate domains."""
        registry = create_registry()
        register_domain_tools(registry)

        cve = registry.get("discover_cve")
        assert cve is not None
        assert "cyber" in cve.metadata.domains

        arxiv = registry.get("discover_arxiv")
        assert arxiv is not None
        assert "research" in arxiv.metadata.domains

        law = registry.get("discover_law")
        assert law is not None
        assert "legal" in law.metadata.domains

        medical = registry.get("discover_medical")
        assert medical is not None
        assert "medical" in medical.metadata.domains

    def test_domain_tools_are_searchable(self) -> None:
        """Given registered domain tools, When filtered by domain,
        Then correct tools returned."""
        registry = create_registry()
        register_domain_tools(registry)

        cyber_tools = registry.get_tools_for_domain("cyber")
        assert len(cyber_tools) == 1
        assert cyber_tools[0].name == "discover_cve"
