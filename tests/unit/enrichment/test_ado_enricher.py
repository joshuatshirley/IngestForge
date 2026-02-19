"""
Unit tests for ADOEnricher IFProcessor migration.

Migrate ADOEnricher to IFProcessor.
GWT-compliant tests with NASA JPL Power of Ten verification.
"""

import inspect
import warnings

import pytest

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.enrichment.ado_enricher import (
    ADOEnricher,
    MAX_ENTITIES_PER_CHUNK,
    MAX_CONCEPTS_PER_CHUNK,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enricher() -> ADOEnricher:
    """Create an ADOEnricher instance."""
    return ADOEnricher()


@pytest.fixture
def ado_chunk_artifact() -> IFChunkArtifact:
    """Create a sample ADO work item chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-ado-001",
        document_id="doc-001",
        content="""
        Work Item #29232: Implement AccountsSelector for MIRS Integration

        This feature adds the AccountsSelector class to the aie-base-code package.
        The selector will query Account and Contact SObjects for the ARISS interface.

        Related work items: #12345, #67890
        LWC Component: accountListComponent
        """,
        chunk_index=0,
        total_chunks=1,
        metadata={"source": "Azure DevOps"},
    )


@pytest.fixture
def security_chunk_artifact() -> IFChunkArtifact:
    """Create a security-related ADO chunk."""
    return IFChunkArtifact(
        artifact_id="chunk-security-001",
        document_id="doc-001",
        content="""
        Security Review #54321: STIG Compliance for User Permissions

        Ensure all access controls meet ATO requirements. Check vulnerability
        in permission settings for the OpportunitiesService.
        """,
        chunk_index=0,
        total_chunks=1,
        metadata={"source": "Azure DevOps"},
    )


# ---------------------------------------------------------------------------
# IFProcessor Interface Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestADOEnricherIFProcessorInterface:
    """GWT: Given ADOEnricher, When checked, Then it implements IFProcessor."""

    def test_extends_if_processor(self, enricher: ADOEnricher) -> None:
        """Verify ADOEnricher extends IFProcessor."""
        assert isinstance(enricher, IFProcessor)

    def test_has_process_method(self, enricher: ADOEnricher) -> None:
        """Verify process() method exists and is callable."""
        assert hasattr(enricher, "process")
        assert callable(enricher.process)

    def test_has_processor_id_property(self, enricher: ADOEnricher) -> None:
        """Verify processor_id property exists."""
        assert hasattr(enricher, "processor_id")
        assert isinstance(enricher.processor_id, str)

    def test_has_version_property(self, enricher: ADOEnricher) -> None:
        """Verify version property exists."""
        assert hasattr(enricher, "version")
        assert isinstance(enricher.version, str)

    def test_has_capabilities_property(self, enricher: ADOEnricher) -> None:
        """Verify capabilities property exists."""
        assert hasattr(enricher, "capabilities")
        assert isinstance(enricher.capabilities, list)

    def test_has_is_available_method(self, enricher: ADOEnricher) -> None:
        """Verify is_available() method exists."""
        assert hasattr(enricher, "is_available")
        assert callable(enricher.is_available)

    def test_has_teardown_method(self, enricher: ADOEnricher) -> None:
        """Verify teardown() method exists."""
        assert hasattr(enricher, "teardown")
        assert callable(enricher.teardown)


# ---------------------------------------------------------------------------
# Processor Properties Tests
# ---------------------------------------------------------------------------


class TestADOEnricherProperties:
    """GWT: Given ADOEnricher, When properties accessed, Then correct values."""

    def test_processor_id_value(self, enricher: ADOEnricher) -> None:
        """Verify processor_id is 'ado-enricher'."""
        assert enricher.processor_id == "ado-enricher"

    def test_version_is_semver(self, enricher: ADOEnricher) -> None:
        """Verify version follows SemVer format."""
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", enricher.version)
        assert enricher.version == "2.0.0"

    def test_capabilities_include_expected(self, enricher: ADOEnricher) -> None:
        """Verify capabilities include expected values."""
        assert "ado-enrichment" in enricher.capabilities
        assert "entity-extraction" in enricher.capabilities

    def test_memory_mb_is_positive(self, enricher: ADOEnricher) -> None:
        """Verify memory_mb returns positive integer."""
        assert enricher.memory_mb > 0
        assert isinstance(enricher.memory_mb, int)

    def test_is_available_returns_true(self, enricher: ADOEnricher) -> None:
        """Verify is_available returns True (regex-based, always available)."""
        assert enricher.is_available() is True

    def test_teardown_returns_true(self, enricher: ADOEnricher) -> None:
        """Verify teardown returns True."""
        assert enricher.teardown() is True


# ---------------------------------------------------------------------------
# Process Method Tests
# ---------------------------------------------------------------------------


class TestADOEnricherProcess:
    """GWT: Given IFChunkArtifact, When processed, Then extracts ADO entities."""

    def test_process_returns_if_artifact(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() returns an IFArtifact."""
        result = enricher.process(ado_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_process_returns_chunk_artifact(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() returns IFChunkArtifact for valid input."""
        result = enricher.process(ado_chunk_artifact)
        assert isinstance(result, IFChunkArtifact)

    def test_process_extracts_ado_ids(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() extracts ADO work item IDs."""
        result = enricher.process(ado_chunk_artifact)
        entities = result.metadata.get("entities", [])
        ado_ids = [e for e in entities if e.startswith("ado_id:")]
        assert len(ado_ids) >= 1
        assert "ado_id:#29232" in ado_ids

    def test_process_extracts_apex_classes(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() extracts Apex class references."""
        result = enricher.process(ado_chunk_artifact)
        entities = result.metadata.get("entities", [])
        apex_classes = [e for e in entities if e.startswith("apex_class:")]
        assert "apex_class:AccountsSelector" in apex_classes

    def test_process_extracts_packages(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() extracts package references."""
        result = enricher.process(ado_chunk_artifact)
        entities = result.metadata.get("entities", [])
        packages = [e for e in entities if e.startswith("package:")]
        assert "package:aie-base-code" in packages

    def test_process_extracts_integrations(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() extracts integration references."""
        result = enricher.process(ado_chunk_artifact)
        entities = result.metadata.get("entities", [])
        integrations = [e for e in entities if e.startswith("integration:")]
        assert "integration:MIRS" in integrations or "integration:ARISS" in integrations

    def test_process_classifies_security_domain(
        self,
        enricher: ADOEnricher,
        security_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() classifies security-related content."""
        result = enricher.process(security_chunk_artifact)
        entities = result.metadata.get("entities", [])
        domain_entities = [e for e in entities if e.startswith("domain:")]
        assert any("security" in d for d in domain_entities)

    def test_process_includes_version_in_metadata(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() adds version to metadata."""
        result = enricher.process(ado_chunk_artifact)
        assert result.metadata.get("ado_enricher_version") == "2.0.0"

    def test_process_invalid_input_returns_failure(
        self,
        enricher: ADOEnricher,
    ) -> None:
        """Verify process() returns IFFailureArtifact for invalid input."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Some text content",
        )
        result = enricher.process(text_artifact)
        assert isinstance(result, IFFailureArtifact)
        assert "ADOEnricher requires IFChunkArtifact" in result.error_message


# ---------------------------------------------------------------------------
# Lineage Preservation Tests
# ---------------------------------------------------------------------------


class TestADOEnricherLineage:
    """GWT: Given artifact with lineage, When processed, Then lineage preserved."""

    def test_process_sets_parent_id(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify derived artifact has correct parent_id."""
        result = enricher.process(ado_chunk_artifact)
        assert result.parent_id == ado_chunk_artifact.artifact_id

    def test_process_increments_lineage_depth(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify lineage_depth is incremented."""
        result = enricher.process(ado_chunk_artifact)
        assert result.lineage_depth == ado_chunk_artifact.lineage_depth + 1

    def test_process_appends_to_provenance(
        self,
        enricher: ADOEnricher,
        ado_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify processor_id is appended to provenance."""
        result = enricher.process(ado_chunk_artifact)
        assert enricher.processor_id in result.provenance


# ---------------------------------------------------------------------------
# Legacy API Tests (Backward Compatibility)
# ---------------------------------------------------------------------------


class TestADOEnricherLegacyAPI:
    """GWT: Given legacy method call, When invoked, Then emits deprecation warning."""

    def test_enrich_chunk_emits_warning(self, enricher: ADOEnricher) -> None:
        """Verify enrich_chunk() emits DeprecationWarning."""
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        chunk = ChunkRecord(
            chunk_id="test-chunk",
            document_id="doc-001",
            content="Work item #12345 for aie-base-code package.",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enricher.enrich_chunk(chunk)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_enrich_batch_emits_warning(self, enricher: ADOEnricher) -> None:
        """Verify enrich_batch() emits DeprecationWarning."""
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        chunks = [
            ChunkRecord(chunk_id="c1", document_id="d1", content="Test #12345"),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enricher.enrich_batch(chunks)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


# ---------------------------------------------------------------------------
# JPL Power of Ten Compliance Tests
# ---------------------------------------------------------------------------


class TestADOEnricherJPLRules:
    """Verify JPL Power of Ten compliance."""

    def test_jpl_rule_2_fixed_upper_bounds(self) -> None:
        """Rule #2: Fixed upper bounds exist."""
        assert MAX_ENTITIES_PER_CHUNK == 50
        assert MAX_CONCEPTS_PER_CHUNK == 20

    def test_jpl_rule_4_functions_under_60_lines(self) -> None:
        """Rule #4: All functions under 60 lines."""
        methods = [
            ADOEnricher.process,
            ADOEnricher._extract_ado_entities,
            ADOEnricher._classify_domain,
        ]
        for method in methods:
            source_lines = len(inspect.getsourcelines(method)[0])
            assert source_lines < 60, f"{method.__name__} has {source_lines} lines"

    def test_jpl_rule_9_type_annotations(self) -> None:
        """Rule #9: Methods have type annotations."""
        methods = [
            ADOEnricher.process,
            ADOEnricher.processor_id.fget,
            ADOEnricher.version.fget,
            ADOEnricher.capabilities.fget,
            ADOEnricher.is_available,
        ]
        for method in methods:
            hints = getattr(method, "__annotations__", {})
            assert "return" in hints, f"{method.__name__} missing return type"


# ---------------------------------------------------------------------------
# Entity Extraction Logic Tests
# ---------------------------------------------------------------------------


class TestADOEnricherExtraction:
    """Test specific extraction patterns."""

    def test_extract_multiple_ado_ids(self, enricher: ADOEnricher) -> None:
        """Verify extraction of multiple ADO IDs."""
        artifact = IFChunkArtifact(
            artifact_id="multi-id",
            document_id="doc-001",
            content="Related items: #11111, #22222, #33333, #44444",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        result = enricher.process(artifact)
        entities = result.metadata.get("entities", [])
        ado_ids = [e for e in entities if e.startswith("ado_id:")]
        assert len(ado_ids) == 4

    def test_extract_sobjects(self, enricher: ADOEnricher) -> None:
        """Verify extraction of Salesforce SObject references."""
        artifact = IFChunkArtifact(
            artifact_id="sobject",
            document_id="doc-001",
            content="Update Account and Contact records for Opportunity tracking.",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        result = enricher.process(artifact)
        entities = result.metadata.get("entities", [])
        sobjects = [e for e in entities if e.startswith("sobject:")]
        assert len(sobjects) >= 2

    def test_extract_lwc_components(self, enricher: ADOEnricher) -> None:
        """Verify extraction of LWC component references."""
        artifact = IFChunkArtifact(
            artifact_id="lwc",
            document_id="doc-001",
            content="The accountListComponent and contactFormModal need updates.",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        result = enricher.process(artifact)
        entities = result.metadata.get("entities", [])
        lwc_refs = [e for e in entities if e.startswith("lwc:")]
        assert len(lwc_refs) >= 1

    def test_utility_extract_references(self, enricher: ADOEnricher) -> None:
        """Test utility method extract_references."""
        content = "#12345 for aie-core-package with AccountsSelector"
        refs = enricher.extract_references(content)
        assert "12345" in refs["ado_ids"]
        assert "aie-core-package" in refs["packages"]
        assert "AccountsSelector" in refs["apex_classes"]

    def test_utility_get_linked_work_items(self, enricher: ADOEnricher) -> None:
        """Test utility method get_linked_work_items."""
        content = "See #11111 and #22222 for details"
        work_items = enricher.get_linked_work_items(content)
        assert 11111 in work_items
        assert 22222 in work_items
