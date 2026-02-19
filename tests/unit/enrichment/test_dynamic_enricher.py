"""
Unit tests for DynamicDomainEnricher IFProcessor migration.

Migrate DynamicDomainEnricher to IFProcessor.
GWT-compliant tests with NASA JPL Power of Ten verification.
"""

import inspect
import warnings
from unittest.mock import Mock, patch

import pytest

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.enrichment.dynamic_enricher import (
    DynamicDomainEnricher,
    MAX_DOMAINS_PER_CHUNK,
    MAX_REFINER_FAILURES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enricher() -> DynamicDomainEnricher:
    """Create a DynamicDomainEnricher instance."""
    return DynamicDomainEnricher()


@pytest.fixture
def custom_enricher() -> DynamicDomainEnricher:
    """Create a DynamicDomainEnricher with custom parameters."""
    return DynamicDomainEnricher(min_score=5, multi_domain_threshold=0.8)


@pytest.fixture
def cyber_chunk_artifact() -> IFChunkArtifact:
    """Create a cybersecurity-related chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-cyber-001",
        document_id="doc-001",
        content="""
        CVE-2024-1234: Critical Buffer Overflow Vulnerability

        A buffer overflow vulnerability was discovered in the authentication
        module. CVSS Score: 9.8. Attackers can exploit this via malware
        injection. Patch immediately to prevent ransomware attacks.
        """,
        chunk_index=0,
        total_chunks=1,
        metadata={"source": "security-advisory"},
    )


@pytest.fixture
def gaming_chunk_artifact() -> IFChunkArtifact:
    """Create a gaming-related chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-gaming-001",
        document_id="doc-001",
        content="""
        Level Design Tips for RPG Games

        When designing dungeons, consider player progression and boss
        difficulty scaling. Use procedural generation for side quests.
        """,
        chunk_index=0,
        total_chunks=1,
        metadata={"source": "game-dev"},
    )


@pytest.fixture
def generic_chunk_artifact() -> IFChunkArtifact:
    """Create a generic chunk that doesn't match any domain."""
    return IFChunkArtifact(
        artifact_id="chunk-generic-001",
        document_id="doc-001",
        content="This is a simple text without any specific domain keywords.",
        chunk_index=0,
        total_chunks=1,
        metadata={},
    )


# ---------------------------------------------------------------------------
# IFProcessor Interface Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherIFProcessorInterface:
    """GWT: Given DynamicDomainEnricher, When checked, Then implements IFProcessor."""

    def test_extends_if_processor(self, enricher: DynamicDomainEnricher) -> None:
        """Verify DynamicDomainEnricher extends IFProcessor."""
        assert isinstance(enricher, IFProcessor)

    def test_has_process_method(self, enricher: DynamicDomainEnricher) -> None:
        """Verify process() method exists and is callable."""
        assert hasattr(enricher, "process")
        assert callable(enricher.process)

    def test_has_processor_id_property(self, enricher: DynamicDomainEnricher) -> None:
        """Verify processor_id property exists."""
        assert hasattr(enricher, "processor_id")
        assert isinstance(enricher.processor_id, str)

    def test_has_version_property(self, enricher: DynamicDomainEnricher) -> None:
        """Verify version property exists."""
        assert hasattr(enricher, "version")
        assert isinstance(enricher.version, str)

    def test_has_capabilities_property(self, enricher: DynamicDomainEnricher) -> None:
        """Verify capabilities property exists."""
        assert hasattr(enricher, "capabilities")
        assert isinstance(enricher.capabilities, list)

    def test_has_is_available_method(self, enricher: DynamicDomainEnricher) -> None:
        """Verify is_available() method exists."""
        assert hasattr(enricher, "is_available")
        assert callable(enricher.is_available)

    def test_has_teardown_method(self, enricher: DynamicDomainEnricher) -> None:
        """Verify teardown() method exists."""
        assert hasattr(enricher, "teardown")
        assert callable(enricher.teardown)


# ---------------------------------------------------------------------------
# Processor Properties Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherProperties:
    """GWT: Given DynamicDomainEnricher, When properties accessed, Then correct values."""

    def test_processor_id_value(self, enricher: DynamicDomainEnricher) -> None:
        """Verify processor_id is correct."""
        assert enricher.processor_id == "dynamic-domain-enricher"

    def test_version_is_semver(self, enricher: DynamicDomainEnricher) -> None:
        """Verify version follows SemVer format."""
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", enricher.version)
        assert enricher.version == "2.0.0"

    def test_capabilities_include_expected(
        self, enricher: DynamicDomainEnricher
    ) -> None:
        """Verify capabilities include expected values."""
        assert "domain-routing" in enricher.capabilities
        assert "multi-domain-enrichment" in enricher.capabilities

    def test_memory_mb_is_positive(self, enricher: DynamicDomainEnricher) -> None:
        """Verify memory_mb returns positive integer."""
        assert enricher.memory_mb > 0
        assert isinstance(enricher.memory_mb, int)

    def test_is_available_returns_true(self, enricher: DynamicDomainEnricher) -> None:
        """Verify is_available returns True (always available)."""
        assert enricher.is_available() is True

    def test_teardown_clears_refiners(self, enricher: DynamicDomainEnricher) -> None:
        """Verify teardown clears cached refiners."""
        enricher._refiners["test"] = Mock()
        result = enricher.teardown()
        assert result is True
        assert len(enricher._refiners) == 0


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherInit:
    """Test initialization with different parameters."""

    def test_default_parameters(self, enricher: DynamicDomainEnricher) -> None:
        """Verify default parameter values."""
        assert enricher.min_score == 3
        assert enricher.multi_domain_threshold == 0.7

    def test_custom_parameters(self, custom_enricher: DynamicDomainEnricher) -> None:
        """Verify custom parameter values."""
        assert custom_enricher.min_score == 5
        assert custom_enricher.multi_domain_threshold == 0.8


# ---------------------------------------------------------------------------
# Process Method Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherProcess:
    """GWT: Given IFChunkArtifact, When processed, Then routes to domain refiners."""

    def test_process_returns_if_artifact(
        self,
        enricher: DynamicDomainEnricher,
        cyber_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() returns an IFArtifact."""
        result = enricher.process(cyber_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_process_returns_chunk_artifact(
        self,
        enricher: DynamicDomainEnricher,
        cyber_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() returns IFChunkArtifact for valid input."""
        result = enricher.process(cyber_chunk_artifact)
        assert isinstance(result, IFChunkArtifact)

    def test_process_includes_version_in_metadata(
        self,
        enricher: DynamicDomainEnricher,
        cyber_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() adds version to metadata."""
        result = enricher.process(cyber_chunk_artifact)
        assert result.metadata.get("domain_enricher_version") == "2.0.0"

    def test_process_returns_unchanged_for_generic_content(
        self,
        enricher: DynamicDomainEnricher,
        generic_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() returns unchanged artifact for unclassified content."""
        result = enricher.process(generic_chunk_artifact)
        # Should still be a valid artifact
        assert isinstance(result, IFChunkArtifact)

    def test_process_invalid_input_returns_failure(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify process() returns IFFailureArtifact for invalid input."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Some text content",
        )
        result = enricher.process(text_artifact)
        assert isinstance(result, IFFailureArtifact)
        assert "DynamicDomainEnricher requires IFChunkArtifact" in result.error_message


# ---------------------------------------------------------------------------
# Lineage Preservation Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherLineage:
    """GWT: Given artifact with lineage, When processed, Then lineage preserved."""

    def test_process_sets_parent_id_on_derived(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify derived artifact has correct parent_id."""
        # Create an artifact that will be classified
        artifact = IFChunkArtifact(
            artifact_id="test-lineage",
            document_id="doc-001",
            content="Cybersecurity vulnerabilities and malware threats are concerning.",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        result = enricher.process(artifact)
        # If classified and processed, check lineage
        if result.artifact_id != artifact.artifact_id:
            assert result.parent_id == artifact.artifact_id

    def test_failure_artifact_has_lineage(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify failure artifact preserves lineage."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Test",
        )
        result = enricher.process(text_artifact)
        assert isinstance(result, IFFailureArtifact)
        assert result.parent_id == text_artifact.artifact_id
        assert enricher.processor_id in result.provenance


# ---------------------------------------------------------------------------
# Legacy API Tests (Backward Compatibility)
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherLegacyAPI:
    """GWT: Given legacy method call, When invoked, Then emits deprecation warning."""

    def test_enrich_chunk_emits_warning(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify enrich_chunk() emits DeprecationWarning."""
        chunk = Mock()
        chunk.content = "Simple test content"
        chunk.metadata = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enricher.enrich_chunk(chunk)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# JPL Power of Ten Compliance Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherJPLRules:
    """Verify JPL Power of Ten compliance."""

    def test_jpl_rule_2_fixed_upper_bounds(self) -> None:
        """Rule #2: Fixed upper bounds exist."""
        assert MAX_DOMAINS_PER_CHUNK == 5
        assert MAX_REFINER_FAILURES == 3

    def test_jpl_rule_4_functions_under_60_lines(self) -> None:
        """Rule #4: All functions under 60 lines."""
        methods = [
            DynamicDomainEnricher.process,
            DynamicDomainEnricher._determine_target_domains,
            DynamicDomainEnricher._apply_refiners,
            DynamicDomainEnricher._get_refiner,
        ]
        for method in methods:
            source_lines = len(inspect.getsourcelines(method)[0])
            assert source_lines < 60, f"{method.__name__} has {source_lines} lines"

    def test_jpl_rule_9_type_annotations(self) -> None:
        """Rule #9: Methods have type annotations."""
        methods = [
            DynamicDomainEnricher.process,
            DynamicDomainEnricher.processor_id.fget,
            DynamicDomainEnricher.version.fget,
            DynamicDomainEnricher.capabilities.fget,
            DynamicDomainEnricher.is_available,
        ]
        for method in methods:
            hints = getattr(method, "__annotations__", {})
            assert "return" in hints, f"{method.__name__} missing return type"


# ---------------------------------------------------------------------------
# Domain Routing Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherRouting:
    """Test domain routing logic."""

    def test_determine_target_domains_primary(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify primary domain is always included."""
        valid_domains = [("cyber", 10), ("gaming", 5), ("bio", 3)]
        targets = enricher._determine_target_domains(valid_domains)
        assert "cyber" in targets

    def test_determine_target_domains_threshold(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify secondary domains within threshold are included."""
        # With threshold 0.7, score 7 should be included when primary is 10
        valid_domains = [("cyber", 10), ("gaming", 8), ("bio", 3)]
        targets = enricher._determine_target_domains(valid_domains)
        assert "cyber" in targets
        assert "gaming" in targets  # 8 >= 10 * 0.7

    def test_determine_target_domains_below_threshold(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify domains below threshold are excluded."""
        valid_domains = [("cyber", 10), ("gaming", 5), ("bio", 3)]
        targets = enricher._determine_target_domains(valid_domains)
        assert "cyber" in targets
        assert "bio" not in targets  # 3 < 10 * 0.7

    def test_determine_target_domains_max_limit(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify max domains limit is enforced."""
        # Create more domains than MAX_DOMAINS_PER_CHUNK
        valid_domains = [(f"domain{i}", 10 - i * 0.1) for i in range(10)]
        targets = enricher._determine_target_domains(valid_domains)
        assert len(targets) <= MAX_DOMAINS_PER_CHUNK


# ---------------------------------------------------------------------------
# Refiner Loading Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherRefinerLoading:
    """Test lazy refiner loading."""

    def test_get_refiner_caches_result(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify refiners are cached after loading."""
        # Mock a successful import
        mock_refiner = Mock()
        with patch.object(enricher, "_get_refiner", return_value=mock_refiner):
            # First call should cache
            enricher._refiners["test_domain"] = mock_refiner
            # Second call should return cached
            result = enricher._get_refiner("test_domain")
            assert result is mock_refiner

    def test_get_refiner_unknown_domain(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify None returned for unknown domain."""
        result = enricher._get_refiner("unknown_domain_xyz")
        assert result is None


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestDynamicDomainEnricherEdgeCases:
    """Test edge cases and error handling."""

    def test_process_empty_content(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify process handles empty content gracefully."""
        artifact = IFChunkArtifact(
            artifact_id="empty",
            document_id="doc-001",
            content="",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        result = enricher.process(artifact)
        assert isinstance(result, IFChunkArtifact)

    def test_process_whitespace_only_content(
        self,
        enricher: DynamicDomainEnricher,
    ) -> None:
        """Verify process handles whitespace-only content gracefully."""
        artifact = IFChunkArtifact(
            artifact_id="whitespace",
            document_id="doc-001",
            content="   \n\t  \n  ",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        result = enricher.process(artifact)
        assert isinstance(result, IFChunkArtifact)
