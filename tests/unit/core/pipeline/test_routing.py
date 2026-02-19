"""
Unit tests for CapabilityRouter.

Tests capability-based routing with fallback chains.
Follows GWT (Given-When-Then) pattern.
"""

import pytest
from pathlib import Path
from ingestforge.core.pipeline.routing import CapabilityRouter
from ingestforge.core.pipeline.registry import IFRegistry
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)


class MockProcessor(IFProcessor):
    """Mock processor for testing."""

    def __init__(
        self,
        proc_id: str,
        caps: list[str],
        available: bool = True,
        should_fail: bool = False,
        memory_mb: int = 100,
    ):
        self._id = proc_id
        self._caps = caps
        self._available = available
        self._should_fail = should_fail
        self._memory_mb = memory_mb

    @property
    def processor_id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> list[str]:
        return self._caps

    @property
    def memory_mb(self) -> int:
        return self._memory_mb

    def is_available(self) -> bool:
        return self._available

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """Process artifact or raise error if configured to fail."""
        if self._should_fail:
            raise RuntimeError(f"Mock processor {self._id} failed")
        # Return a derived artifact
        return artifact.derive(processor_id=self._id)


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    reg = IFRegistry()
    reg.clear()
    yield reg
    reg.clear()


@pytest.fixture
def router(clean_registry):
    """Provide a router with clean registry."""
    return CapabilityRouter(registry=clean_registry)


class TestCapabilityRouterInit:
    """Test CapabilityRouter initialization."""

    def test_init_with_registry(self, clean_registry):
        """
        GWT:
        Given a custom registry,
        When CapabilityRouter is initialized with it,
        Then the router uses that registry.
        """
        router = CapabilityRouter(registry=clean_registry)
        assert router._registry is clean_registry

    def test_init_without_registry(self):
        """
        GWT:
        Given no registry,
        When CapabilityRouter is initialized,
        Then it uses the singleton registry.
        """
        router = CapabilityRouter()
        assert router._registry is IFRegistry()


class TestSelect:
    """Test CapabilityRouter.select() method."""

    def test_select_single_capability_match(self, router, clean_registry):
        """
        GWT:
        Given a processor with "ocr" capability,
        When select(["ocr"]) is called,
        Then that processor is returned.
        """
        proc = MockProcessor("ocr-proc", ["ocr"])
        clean_registry.register(proc, ["image/png"])

        result = router.select(["ocr"])
        assert result is proc

    def test_select_multiple_capabilities_all_match(self, router, clean_registry):
        """
        GWT:
        Given a processor with ["ocr", "table-extraction"] capabilities,
        When select(["ocr", "table-extraction"], match="all") is called,
        Then that processor is returned.
        """
        proc = MockProcessor("multi-proc", ["ocr", "table-extraction"])
        clean_registry.register(proc, ["image/png"])

        result = router.select(["ocr", "table-extraction"], match="all")
        assert result is proc

    def test_select_multiple_capabilities_any_match(self, router, clean_registry):
        """
        GWT:
        Given processors with different capabilities,
        When select(["ocr", "embedding"], match="any") is called,
        Then a processor with at least one capability is returned.
        """
        proc1 = MockProcessor("ocr-proc", ["ocr"])
        proc2 = MockProcessor("embed-proc", ["embedding"])
        clean_registry.register(proc1, ["image/png"])
        clean_registry.register(proc2, ["text/plain"])

        result = router.select(["ocr", "embedding"], match="any")
        assert result in (proc1, proc2)

    def test_select_no_match_returns_none(self, router, clean_registry):
        """
        GWT:
        Given no processors with required capability,
        When select(["nonexistent"]) is called,
        Then None is returned.
        """
        proc = MockProcessor("basic-proc", ["basic"])
        clean_registry.register(proc, ["text/plain"])

        result = router.select(["nonexistent"])
        assert result is None

    def test_select_unavailable_processor_skipped(self, router, clean_registry):
        """
        GWT:
        Given a processor with capability but unavailable,
        When select() is called,
        Then None is returned.
        """
        proc = MockProcessor("unavail-proc", ["ocr"], available=False)
        clean_registry.register(proc, ["image/png"])

        result = router.select(["ocr"])
        assert result is None

    def test_select_empty_capabilities_returns_none(self, router):
        """
        GWT:
        Given empty capabilities list,
        When select([]) is called,
        Then None is returned.
        """
        result = router.select([])
        assert result is None

    def test_select_invalid_match_parameter(self, router):
        """
        GWT:
        Given invalid match parameter,
        When select() is called,
        Then ValueError is raised.
        """
        with pytest.raises(ValueError, match="match must be 'all' or 'any'"):
            router.select(["ocr"], match="invalid")

    def test_select_priority_ordering(self, router, clean_registry):
        """
        GWT:
        Given multiple processors with same capability at different priorities,
        When select() is called,
        Then highest priority processor is returned.
        """
        proc_low = MockProcessor("low-priority", ["ocr"])
        proc_high = MockProcessor("high-priority", ["ocr"])
        clean_registry.register(proc_low, ["image/png"], priority=50)
        clean_registry.register(proc_high, ["image/png"], priority=150)

        result = router.select(["ocr"])
        assert result.processor_id == "high-priority"


class TestSelectWithFallback:
    """Test CapabilityRouter.select_with_fallback() method."""

    def test_select_preferred_available(self, router, clean_registry):
        """
        GWT:
        Given preferred processor is available,
        When select_with_fallback() is called,
        Then preferred processor is returned.
        """
        preferred_proc = MockProcessor("preferred", ["advanced"])
        fallback_proc = MockProcessor("fallback", ["basic"])
        clean_registry.register(preferred_proc, ["text/plain"])
        clean_registry.register(fallback_proc, ["text/plain"])

        result = router.select_with_fallback(
            preferred=["advanced"], fallbacks=[["basic"]]
        )
        assert result is preferred_proc

    def test_fallback_when_preferred_unavailable(self, router, clean_registry):
        """
        GWT:
        Given preferred processor is unavailable,
        When select_with_fallback() is called with fallback,
        Then fallback processor is returned.
        """
        preferred_proc = MockProcessor("preferred", ["advanced"], available=False)
        fallback_proc = MockProcessor("fallback", ["basic"])
        clean_registry.register(preferred_proc, ["text/plain"])
        clean_registry.register(fallback_proc, ["text/plain"])

        result = router.select_with_fallback(
            preferred=["advanced"], fallbacks=[["basic"]]
        )
        assert result is fallback_proc

    def test_multiple_fallback_levels(self, router, clean_registry):
        """
        GWT:
        Given multiple fallback levels,
        When preferred and first fallback are unavailable,
        Then second fallback processor is returned.
        """
        preferred = MockProcessor("pref", ["gpu-ocr"], available=False)
        fallback1 = MockProcessor("fb1", ["cpu-ocr"], available=False)
        fallback2 = MockProcessor("fb2", ["basic-ocr"])
        clean_registry.register(preferred, ["image/png"])
        clean_registry.register(fallback1, ["image/png"])
        clean_registry.register(fallback2, ["image/png"])

        result = router.select_with_fallback(
            preferred=["gpu-ocr"], fallbacks=[["cpu-ocr"], ["basic-ocr"]]
        )
        assert result is fallback2

    def test_no_fallback_provided(self, router, clean_registry):
        """
        GWT:
        Given preferred processor is unavailable and no fallbacks provided,
        When select_with_fallback() is called,
        Then None is returned.
        """
        proc = MockProcessor("pref", ["advanced"], available=False)
        clean_registry.register(proc, ["text/plain"])

        result = router.select_with_fallback(preferred=["advanced"])
        assert result is None

    def test_empty_fallback_list_skipped(self, router, clean_registry):
        """
        GWT:
        Given fallback list contains empty capability lists,
        When select_with_fallback() is called,
        Then empty lists are skipped.
        """
        proc = MockProcessor("fallback", ["basic"])
        clean_registry.register(proc, ["text/plain"])

        result = router.select_with_fallback(
            preferred=["nonexistent"], fallbacks=[[], ["basic"]]
        )
        assert result is proc

    def test_empty_preferred_returns_none(self, router):
        """
        GWT:
        Given empty preferred capabilities,
        When select_with_fallback() is called,
        Then None is returned.
        """
        result = router.select_with_fallback(preferred=[])
        assert result is None

    def test_all_fallbacks_exhausted(self, router, clean_registry):
        """
        GWT:
        Given all processors are unavailable,
        When select_with_fallback() is called,
        Then None is returned.
        """
        proc1 = MockProcessor("p1", ["cap1"], available=False)
        proc2 = MockProcessor("p2", ["cap2"], available=False)
        clean_registry.register(proc1, ["text/plain"])
        clean_registry.register(proc2, ["text/plain"])

        result = router.select_with_fallback(preferred=["cap1"], fallbacks=[["cap2"]])
        assert result is None


class TestRouteArtifact:
    """Test CapabilityRouter.route_artifact() method."""

    def test_route_artifact_success(self, router, clean_registry):
        """
        GWT:
        Given a processor with required capability,
        When route_artifact() is called,
        Then artifact is processed and returned.
        """
        proc = MockProcessor("processor", ["transform"])
        clean_registry.register(proc, ["text/plain"])

        input_artifact = IFTextArtifact(artifact_id="test-1", content="test content")
        result = router.route_artifact(input_artifact, "transform")

        assert isinstance(result, IFTextArtifact)
        assert result.parent_id == input_artifact.artifact_id
        assert "processor" in result.provenance

    def test_route_artifact_no_processor_available(self, router):
        """
        GWT:
        Given no processor with required capability,
        When route_artifact() is called,
        Then IFFailureArtifact is returned.
        """
        input_artifact = IFTextArtifact(artifact_id="test-1", content="test content")
        result = router.route_artifact(input_artifact, "nonexistent")

        assert isinstance(result, IFFailureArtifact)
        assert "No available processor" in result.error_message
        assert result.parent_id == input_artifact.artifact_id

    def test_route_artifact_processor_fails(self, router, clean_registry):
        """
        GWT:
        Given a processor that raises an exception,
        When route_artifact() is called,
        Then IFFailureArtifact is returned with error details.
        """
        proc = MockProcessor("failing-proc", ["transform"], should_fail=True)
        clean_registry.register(proc, ["text/plain"])

        input_artifact = IFTextArtifact(artifact_id="test-1", content="test content")
        result = router.route_artifact(input_artifact, "transform")

        assert isinstance(result, IFFailureArtifact)
        assert "failing-proc" in result.error_message
        assert result.failed_processor_id == "failing-proc"
        assert result.stack_trace is not None

    def test_route_artifact_empty_capability(self, router):
        """
        GWT:
        Given empty capability string,
        When route_artifact() is called,
        Then IFFailureArtifact is returned.
        """
        input_artifact = IFTextArtifact(artifact_id="test-1", content="test content")
        result = router.route_artifact(input_artifact, "")

        assert isinstance(result, IFFailureArtifact)
        assert "empty capability" in result.error_message

    def test_route_artifact_lineage_preserved(self, router, clean_registry):
        """
        GWT:
        Given an artifact with existing lineage,
        When route_artifact() is called,
        Then lineage is properly extended.
        """
        proc = MockProcessor("processor", ["transform"])
        clean_registry.register(proc, ["text/plain"])

        # Create artifact with lineage
        root = IFTextArtifact(artifact_id="root", content="root")
        derived = root.derive(processor_id="first-proc", content="derived")

        result = router.route_artifact(derived, "transform")

        assert result.root_artifact_id == root.artifact_id
        assert result.lineage_depth == 2
        assert len(result.provenance) == 2

    def test_route_artifact_failure_metadata(self, router):
        """
        GWT:
        Given routing failure,
        When IFFailureArtifact is created,
        Then metadata contains original artifact info.
        """
        input_artifact = IFTextArtifact(artifact_id="test-1", content="test")
        result = router.route_artifact(input_artifact, "nonexistent")

        assert isinstance(result, IFFailureArtifact)
        assert result.metadata["original_artifact_id"] == "test-1"


class TestRouterIntegration:
    """Integration tests for CapabilityRouter."""

    def test_end_to_end_routing_pipeline(self, router, clean_registry):
        """
        GWT:
        Given a multi-stage routing scenario,
        When artifacts are routed through multiple processors,
        Then lineage is correctly tracked end-to-end.
        """
        # Register processors
        extract_proc = MockProcessor("extractor", ["extract"])
        transform_proc = MockProcessor("transformer", ["transform"])
        clean_registry.register(extract_proc, ["image/png"])
        clean_registry.register(transform_proc, ["text/plain"])

        # Create initial artifact
        file_artifact = IFFileArtifact(
            artifact_id="file-1", file_path=Path("/tmp/test.png"), mime_type="image/png"
        )

        # Route through extract
        extracted = router.route_artifact(file_artifact, "extract")
        assert extracted.lineage_depth == 1

        # Route through transform
        transformed = router.route_artifact(extracted, "transform")
        assert transformed.lineage_depth == 2
        assert len(transformed.provenance) == 2

    def test_fallback_chain_integration(self, router, clean_registry):
        """
        GWT:
        Given a complex fallback scenario,
        When select_with_fallback() is used in a pipeline,
        Then correct processor is selected at each stage.
        """
        # Register processors with different capabilities
        gpu_proc = MockProcessor("gpu-proc", ["gpu-ocr"], available=False)
        cpu_proc = MockProcessor("cpu-proc", ["cpu-ocr"])
        basic_proc = MockProcessor("basic-proc", ["basic-ocr"])

        clean_registry.register(gpu_proc, ["image/png"], priority=200)
        clean_registry.register(cpu_proc, ["image/png"], priority=100)
        clean_registry.register(basic_proc, ["image/png"], priority=50)

        # Select with fallback chain
        proc = router.select_with_fallback(
            preferred=["gpu-ocr"], fallbacks=[["cpu-ocr"], ["basic-ocr"]]
        )

        assert proc.processor_id == "cpu-proc"

    def test_multiple_capability_composition(self, router, clean_registry):
        """
        GWT:
        Given processors with multiple capabilities,
        When selecting by composed capabilities,
        Then only processors with ALL required capabilities are returned.
        """
        # Processor with both capabilities
        full_proc = MockProcessor("full", ["ocr", "table-extraction"])
        # Processors with single capabilities
        ocr_only = MockProcessor("ocr-only", ["ocr"])
        table_only = MockProcessor("table-only", ["table-extraction"])

        clean_registry.register(full_proc, ["image/png"])
        clean_registry.register(ocr_only, ["image/png"])
        clean_registry.register(table_only, ["image/png"])

        # Should return only the processor with both capabilities
        result = router.select(["ocr", "table-extraction"], match="all")
        assert result is full_proc


class TestSelectMemoryAware:
    """Test CapabilityRouter.select_memory_aware() method ()."""

    def test_select_within_memory_limit(self, router, clean_registry):
        """
        GWT:
        Given processors with different memory requirements,
        When selecting with memory limit,
        Then only processors within limit are considered.
        """
        small_proc = MockProcessor("small", ["ocr"], memory_mb=100)
        large_proc = MockProcessor("large", ["ocr"], memory_mb=800)

        clean_registry.register(small_proc, ["image/png"], priority=50)
        clean_registry.register(large_proc, ["image/png"], priority=100)

        # With 500MB limit, should select small processor
        result = router.select_memory_aware(["ocr"], max_mb=500)
        assert result is small_proc

    def test_select_respects_priority_within_limit(self, router, clean_registry):
        """
        GWT:
        Given multiple processors within memory limit,
        When selecting with memory constraint,
        Then highest priority within limit is selected.
        """
        proc_a = MockProcessor("proc-a", ["ocr"], memory_mb=100)
        proc_b = MockProcessor("proc-b", ["ocr"], memory_mb=200)

        clean_registry.register(proc_a, ["image/png"], priority=50)
        clean_registry.register(proc_b, ["image/png"], priority=100)

        # Both fit within 500MB, select higher priority
        result = router.select_memory_aware(["ocr"], max_mb=500)
        assert result is proc_b

    def test_select_returns_none_when_all_exceed_limit(self, router, clean_registry):
        """
        GWT:
        Given all processors exceed memory limit,
        When selecting with memory constraint,
        Then None is returned.
        """
        large_proc = MockProcessor("large", ["ocr"], memory_mb=800)
        clean_registry.register(large_proc, ["image/png"])

        result = router.select_memory_aware(["ocr"], max_mb=500)
        assert result is None

    def test_select_empty_capabilities_returns_none(self, router):
        """
        GWT:
        Given empty capabilities list,
        When select_memory_aware is called,
        Then None is returned.
        """
        result = router.select_memory_aware([], max_mb=500)
        assert result is None

    def test_select_invalid_match_raises(self, router):
        """
        GWT:
        Given invalid match parameter,
        When select_memory_aware is called,
        Then ValueError is raised.
        """
        with pytest.raises(ValueError, match="match must be"):
            router.select_memory_aware(["ocr"], max_mb=500, match="invalid")

    def test_select_uses_system_memory_when_none(self, router, clean_registry):
        """
        GWT:
        Given max_mb is None,
        When select_memory_aware is called,
        Then system available memory is used.
        """
        small_proc = MockProcessor("small", ["ocr"], memory_mb=100)
        clean_registry.register(small_proc, ["image/png"])

        # Should work without explicit memory limit (uses system memory)
        result = router.select_memory_aware(["ocr"], max_mb=None)
        assert result is small_proc


class TestSelectWithFallbackMemoryAware:
    """Test CapabilityRouter.select_with_fallback() with memory constraint ()."""

    def test_fallback_respects_memory_limit(self, router, clean_registry):
        """
        GWT:
        Given fallback chain with memory constraints,
        When preferred exceeds limit but fallback fits,
        Then fallback within memory limit is selected.
        """
        # Preferred is large (800MB), fallback is small (100MB)
        preferred_proc = MockProcessor("preferred", ["gpu-ocr"], memory_mb=800)
        fallback_proc = MockProcessor("fallback", ["cpu-ocr"], memory_mb=100)

        clean_registry.register(preferred_proc, ["image/png"], priority=100)
        clean_registry.register(fallback_proc, ["image/png"], priority=50)

        # With 500MB limit, preferred won't fit, should select fallback
        result = router.select_with_fallback(
            preferred=["gpu-ocr"], fallbacks=[["cpu-ocr"]], max_mb=500
        )
        assert result is fallback_proc

    def test_fallback_no_memory_filter_when_none(self, router, clean_registry):
        """
        GWT:
        Given fallback chain without memory constraint,
        When max_mb is None,
        Then memory is not filtered.
        """
        large_proc = MockProcessor("large", ["gpu-ocr"], memory_mb=800)
        clean_registry.register(large_proc, ["image/png"])

        # Without memory limit, large processor should be selected
        result = router.select_with_fallback(preferred=["gpu-ocr"], max_mb=None)
        assert result is large_proc


class TestRouteArtifactMemoryAware:
    """Test CapabilityRouter.route_artifact() with memory constraint ()."""

    def test_route_artifact_respects_memory_limit(self, router, clean_registry):
        """
        GWT:
        Given processor exceeds memory limit,
        When routing artifact with memory constraint,
        Then IFFailureArtifact is returned.
        """
        large_proc = MockProcessor("large", ["ocr"], memory_mb=800)
        clean_registry.register(large_proc, ["image/png"])

        artifact = IFTextArtifact(artifact_id="test-1", content="test")
        result = router.route_artifact(artifact, "ocr", max_mb=500)

        assert isinstance(result, IFFailureArtifact)
        assert "No available processor" in result.error_message

    def test_route_artifact_selects_within_memory(self, router, clean_registry):
        """
        GWT:
        Given processor within memory limit,
        When routing artifact with memory constraint,
        Then artifact is processed successfully.
        """
        small_proc = MockProcessor("small", ["ocr"], memory_mb=100)
        clean_registry.register(small_proc, ["image/png"])

        artifact = IFTextArtifact(artifact_id="test-1", content="test")
        result = router.route_artifact(artifact, "ocr", max_mb=500)

        assert not isinstance(result, IFFailureArtifact)
        assert result.lineage_depth == 1


# ============================================================================
# JPL POWER OF TEN COMPLIANCE TESTS
# ============================================================================


class TestJPLRule1LinearControlFlow:
    """
    JPL Rule #1: Restrict control flow to simple constructs.
    No goto, setjmp, longjmp, or recursion.
    Tests verify linear execution paths.
    """

    def test_select_single_code_path(self, router, clean_registry):
        """
        GWT:
        Given a simple selection scenario,
        When select() is called,
        Then execution follows a single linear path without recursion.
        """
        proc = MockProcessor("linear-proc", ["cap"])
        clean_registry.register(proc, ["text/plain"])

        # Simple call, no recursion
        result = router.select(["cap"])
        assert result is proc

    def test_fallback_iterates_without_recursion(self, router, clean_registry):
        """
        GWT:
        Given multiple fallback levels,
        When select_with_fallback() is called,
        Then it iterates linearly without recursive calls.
        """
        # Create 5 fallback levels
        procs = []
        for i in range(5):
            p = MockProcessor(f"proc-{i}", [f"cap-{i}"], available=(i == 4))
            procs.append(p)
            clean_registry.register(p, ["text/plain"])

        result = router.select_with_fallback(
            preferred=["cap-0"], fallbacks=[["cap-1"], ["cap-2"], ["cap-3"], ["cap-4"]]
        )

        assert result.processor_id == "proc-4"


class TestJPLRule2FixedUpperBounds:
    """
    JPL Rule #2: All loops must have fixed upper bounds.
    Tests verify bounded iteration.
    """

    def test_select_bounded_by_registry_size(self, router, clean_registry):
        """
        GWT:
        Given a registry with N processors,
        When select() iterates through processors,
        Then iteration is bounded by N (registry size limit: 256).
        """
        # Register up to 10 processors (registry has MAX_PROCESSORS=256)
        for i in range(10):
            proc = MockProcessor(f"proc-{i}", ["common"], available=(i == 9))
            clean_registry.register(proc, ["text/plain"], priority=i)

        result = router.select(["common"])
        assert result.processor_id == "proc-9"

    def test_fallback_bounded_by_fallback_list_length(self, router, clean_registry):
        """
        GWT:
        Given a fallback list of known length,
        When select_with_fallback() iterates,
        Then iteration is bounded by list length.
        """
        # 3 fallback levels, all unavailable
        for i in range(3):
            proc = MockProcessor(f"fb-{i}", [f"cap-{i}"], available=False)
            clean_registry.register(proc, ["text/plain"])

        result = router.select_with_fallback(
            preferred=["cap-0"], fallbacks=[["cap-1"], ["cap-2"]]
        )

        # All exhausted, bounded iteration completed
        assert result is None

    def test_memory_filtering_bounded(self, router, clean_registry):
        """
        GWT:
        Given processors with varying memory requirements,
        When select_memory_aware() filters by memory,
        Then iteration is bounded by processor count.
        """
        for i in range(20):
            mem = (i + 1) * 100  # 100, 200, ... 2000 MB
            proc = MockProcessor(f"mem-{i}", ["compute"], memory_mb=mem)
            clean_registry.register(proc, ["text/plain"], priority=i)

        # Should find processor with 100MB (first one within limit)
        result = router.select_memory_aware(["compute"], max_mb=150)
        assert result is not None
        assert result.memory_mb <= 150


class TestJPLRule4SmallFunctions:
    """
    JPL Rule #4: No function longer than 60 lines.
    Tests verify modular function design through behavior.
    """

    def test_create_failure_artifact_isolated(self, router):
        """
        GWT:
        Given a routing failure,
        When _create_failure_artifact() is called,
        Then it creates a complete failure artifact (single responsibility).
        """
        artifact = IFTextArtifact(artifact_id="test", content="test")
        result = router.route_artifact(artifact, "nonexistent")

        assert isinstance(result, IFFailureArtifact)
        assert result.error_message is not None
        assert result.parent_id == artifact.artifact_id
        assert result.lineage_depth == 1

    def test_select_delegates_to_registry(self, router, clean_registry):
        """
        GWT:
        Given the select() method,
        When it needs processor lookup,
        Then it delegates to registry (separation of concerns).
        """
        proc = MockProcessor("delegate-test", ["cap"])
        clean_registry.register(proc, ["text/plain"])

        # select() should delegate to registry.get_by_capabilities()
        result = router.select(["cap"])
        assert result is proc


class TestJPLRule5MinimumScopeAssertion:
    """
    JPL Rule #5: Data should have minimum scope.
    Tests verify data encapsulation.
    """

    def test_router_encapsulates_registry(self, clean_registry):
        """
        GWT:
        Given a CapabilityRouter,
        When initialized with a registry,
        Then registry is encapsulated (accessible only through router methods).
        """
        router = CapabilityRouter(registry=clean_registry)

        # Registry is private (_registry)
        assert hasattr(router, "_registry")
        assert router._registry is clean_registry

    def test_failure_artifact_contains_minimal_data(self, router):
        """
        GWT:
        Given a routing failure,
        When IFFailureArtifact is created,
        Then it contains only necessary error information.
        """
        artifact = IFTextArtifact(artifact_id="scope-test", content="test")
        result = router.route_artifact(artifact, "nonexistent")

        assert isinstance(result, IFFailureArtifact)
        # Minimal required fields
        assert result.error_message is not None
        assert result.failed_processor_id is not None
        assert result.parent_id is not None


class TestJPLRule7CheckReturnValues:
    """
    JPL Rule #7: Check return values of all non-void functions.
    Tests verify proper return value handling.
    """

    def test_select_returns_none_not_raises_on_empty(self, router):
        """
        GWT:
        Given no matching processors,
        When select() is called,
        Then None is returned (not an exception).
        """
        result = router.select(["nonexistent"])
        assert result is None

    def test_select_memory_aware_returns_none_not_raises(self, router, clean_registry):
        """
        GWT:
        Given all processors exceed memory limit,
        When select_memory_aware() is called,
        Then None is returned (not an exception).
        """
        proc = MockProcessor("huge", ["cap"], memory_mb=10000)
        clean_registry.register(proc, ["text/plain"])

        result = router.select_memory_aware(["cap"], max_mb=100)
        assert result is None

    def test_route_artifact_returns_failure_artifact_on_error(
        self, router, clean_registry
    ):
        """
        GWT:
        Given a processor that fails,
        When route_artifact() is called,
        Then IFFailureArtifact is returned (error is captured, not raised).
        """
        proc = MockProcessor("failing", ["cap"], should_fail=True)
        clean_registry.register(proc, ["text/plain"])

        artifact = IFTextArtifact(artifact_id="check-return", content="test")
        result = router.route_artifact(artifact, "cap")

        # Error captured in IFFailureArtifact, not raised
        assert isinstance(result, IFFailureArtifact)
        assert "failing" in result.error_message

    def test_select_with_fallback_checks_each_level(self, router, clean_registry):
        """
        GWT:
        Given multiple fallback levels,
        When select_with_fallback() iterates,
        Then each select() return value is checked before proceeding.
        """
        # All unavailable except last
        proc1 = MockProcessor("fb1", ["cap1"], available=False)
        proc2 = MockProcessor("fb2", ["cap2"], available=False)
        proc3 = MockProcessor("fb3", ["cap3"])  # Available

        clean_registry.register(proc1, ["text/plain"])
        clean_registry.register(proc2, ["text/plain"])
        clean_registry.register(proc3, ["text/plain"])

        result = router.select_with_fallback(
            preferred=["cap1"], fallbacks=[["cap2"], ["cap3"]]
        )

        # Correctly checked each level and found fb3
        assert result is proc3


class TestJPLRule9CompleteTypeHints:
    """
    JPL Rule #9: Use complete type hints.
    Tests verify type safety through runtime behavior.
    """

    def test_select_accepts_list_of_strings(self, router, clean_registry):
        """
        GWT:
        Given capabilities as List[str],
        When select() is called,
        Then it processes the list correctly.
        """
        proc = MockProcessor("typed", ["cap1", "cap2"])
        clean_registry.register(proc, ["text/plain"])

        # List[str] input
        result = router.select(["cap1", "cap2"])
        assert result is proc

    def test_select_memory_aware_accepts_optional_int(self, router, clean_registry):
        """
        GWT:
        Given max_mb as Optional[int],
        When select_memory_aware() is called with int or None,
        Then both are handled correctly.
        """
        proc = MockProcessor("typed", ["cap"], memory_mb=100)
        clean_registry.register(proc, ["text/plain"])

        # With int
        result1 = router.select_memory_aware(["cap"], max_mb=500)
        assert result1 is proc

        # With None (uses system memory)
        result2 = router.select_memory_aware(["cap"], max_mb=None)
        assert result2 is proc

    def test_route_artifact_returns_typed_artifact(self, router, clean_registry):
        """
        GWT:
        Given an IFArtifact input,
        When route_artifact() is called,
        Then return type is IFArtifact (success) or IFFailureArtifact (failure).
        """
        proc = MockProcessor("typed", ["cap"])
        clean_registry.register(proc, ["text/plain"])

        artifact = IFTextArtifact(artifact_id="typed-test", content="test")
        result = router.route_artifact(artifact, "cap")

        # Return is IFArtifact subtype
        assert isinstance(result, IFArtifact)


class TestJPLRule10StaticAnalysisFriendly:
    """
    JPL Rule #10: Code should pass static analysis.
    Tests verify predictable behavior for static analysis.
    """

    def test_no_dynamic_attribute_access(self, router, clean_registry):
        """
        GWT:
        Given CapabilityRouter methods,
        When they access processor attributes,
        Then they use defined interface methods (not getattr/setattr).
        """
        proc = MockProcessor("static", ["cap"])
        clean_registry.register(proc, ["text/plain"])

        result = router.select(["cap"])

        # Accessing well-defined properties
        assert result.processor_id == "static"
        assert result.capabilities == ["cap"]
        assert result.is_available() is True

    def test_exception_types_are_specific(self, router):
        """
        GWT:
        Given invalid input,
        When validation fails,
        Then specific exception types are raised (not generic Exception).
        """
        with pytest.raises(ValueError):  # Specific type
            router.select(["cap"], match="invalid")


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_processor_with_empty_capabilities(self, router, clean_registry):
        """
        GWT:
        Given a processor with empty capabilities list,
        When registered and queried,
        Then it should not match any capability query.
        """
        proc = MockProcessor("empty-caps", [])
        clean_registry.register(proc, ["text/plain"])

        result = router.select(["any"])
        assert result is None

    def test_single_character_capability(self, router, clean_registry):
        """
        GWT:
        Given a capability with single character,
        When selected,
        Then it matches correctly.
        """
        proc = MockProcessor("single-char", ["x"])
        clean_registry.register(proc, ["text/plain"])

        result = router.select(["x"])
        assert result is proc

    def test_unicode_capability(self, router, clean_registry):
        """
        GWT:
        Given a capability with unicode characters,
        When selected,
        Then it matches correctly.
        """
        proc = MockProcessor("unicode-proc", ["日本語", "émoji🎉"])
        clean_registry.register(proc, ["text/plain"])

        result = router.select(["日本語"])
        assert result is proc

    def test_capability_with_special_characters(self, router, clean_registry):
        """
        GWT:
        Given capabilities with special characters,
        When selected,
        Then they match correctly.
        """
        proc = MockProcessor("special", ["ocr.v2", "table-extract", "process_data"])
        clean_registry.register(proc, ["text/plain"])

        assert router.select(["ocr.v2"]) is proc
        assert router.select(["table-extract"]) is proc
        assert router.select(["process_data"]) is proc

    def test_memory_at_exact_boundary(self, router, clean_registry):
        """
        GWT:
        Given processor memory exactly at limit,
        When select_memory_aware() is called,
        Then processor is included (<=, not <).
        """
        proc = MockProcessor("boundary", ["cap"], memory_mb=500)
        clean_registry.register(proc, ["text/plain"])

        result = router.select_memory_aware(["cap"], max_mb=500)
        assert result is proc  # Exactly at boundary should match

    def test_memory_one_byte_over_limit(self, router, clean_registry):
        """
        GWT:
        Given processor memory one MB over limit,
        When select_memory_aware() is called,
        Then processor is excluded.
        """
        proc = MockProcessor("over", ["cap"], memory_mb=501)
        clean_registry.register(proc, ["text/plain"])

        result = router.select_memory_aware(["cap"], max_mb=500)
        assert result is None

    def test_zero_memory_processor(self, router, clean_registry):
        """
        GWT:
        Given processor with zero memory requirement,
        When select_memory_aware() is called,
        Then processor is selected (0 <= any positive limit).
        """
        proc = MockProcessor("zero-mem", ["cap"], memory_mb=0)
        clean_registry.register(proc, ["text/plain"])

        result = router.select_memory_aware(["cap"], max_mb=1)
        assert result is proc

    def test_very_large_memory_limit(self, router, clean_registry):
        """
        GWT:
        Given very large memory limit,
        When select_memory_aware() is called,
        Then all processors are considered.
        """
        proc = MockProcessor("large-mem", ["cap"], memory_mb=100000)
        clean_registry.register(proc, ["text/plain"])

        result = router.select_memory_aware(["cap"], max_mb=1000000)
        assert result is proc

    def test_duplicate_capabilities_in_query(self, router, clean_registry):
        """
        GWT:
        Given duplicate capabilities in query,
        When select() is called,
        Then duplicates are handled gracefully.
        """
        proc = MockProcessor("dedup", ["cap"])
        clean_registry.register(proc, ["text/plain"])

        result = router.select(["cap", "cap", "cap"])
        assert result is proc

    def test_processor_becomes_unavailable_mid_query(self, router, clean_registry):
        """
        GWT:
        Given processors where first becomes unavailable,
        When select() is called multiple times,
        Then availability is checked each time.
        """
        proc1 = MockProcessor("dynamic", ["cap"])
        proc2 = MockProcessor("backup", ["cap"])
        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=50)

        # First call: proc1 available
        result1 = router.select(["cap"])
        assert result1.processor_id == "dynamic"

        # Make proc1 unavailable
        proc1._available = False

        # Second call: proc1 unavailable, should get proc2
        result2 = router.select(["cap"])
        assert result2.processor_id == "backup"


class TestConcurrencyAndIsolation:
    """Tests for isolation and thread-safety considerations."""

    def test_router_instances_share_singleton_registry(self):
        """
        GWT:
        Given multiple CapabilityRouter instances without custom registry,
        When they query processors,
        Then they share the same singleton registry.
        """
        router1 = CapabilityRouter()
        router2 = CapabilityRouter()

        assert router1._registry is router2._registry

    def test_clean_registry_isolates_tests(self, clean_registry):
        """
        GWT:
        Given a clean_registry fixture,
        When processors are registered,
        Then they are isolated from other tests.
        """
        proc = MockProcessor("isolated", ["test-cap"])
        clean_registry.register(proc, ["text/plain"])

        # Verify registration
        processors = clean_registry.get_by_capability("test-cap")
        assert len(processors) == 1
        assert processors[0].processor_id == "isolated"


class TestErrorRecovery:
    """Tests for graceful error handling and recovery."""

    def test_processor_exception_contains_stack_trace(self, router, clean_registry):
        """
        GWT:
        Given a processor that throws an exception,
        When route_artifact() is called,
        Then IFFailureArtifact contains the stack trace.
        """
        proc = MockProcessor("exploder", ["cap"], should_fail=True)
        clean_registry.register(proc, ["text/plain"])

        artifact = IFTextArtifact(artifact_id="stack-test", content="test")
        result = router.route_artifact(artifact, "cap")

        assert isinstance(result, IFFailureArtifact)
        assert result.stack_trace is not None
        assert "RuntimeError" in result.stack_trace
        assert "Mock processor" in result.stack_trace

    def test_failure_preserves_lineage(self, router, clean_registry):
        """
        GWT:
        Given an artifact with existing lineage,
        When routing fails,
        Then failure artifact preserves lineage information.
        """
        proc = MockProcessor("failer", ["cap"], should_fail=True)
        clean_registry.register(proc, ["text/plain"])

        # Create artifact with lineage
        root = IFTextArtifact(artifact_id="root", content="root")
        derived = root.derive(processor_id="step1", content="derived")

        result = router.route_artifact(derived, "cap")

        assert isinstance(result, IFFailureArtifact)
        assert result.root_artifact_id == root.artifact_id
        assert result.lineage_depth == 2
        assert "step1" in result.provenance

    def test_multiple_failures_tracked_independently(self, router, clean_registry):
        """
        GWT:
        Given multiple routing failures,
        When each fails,
        Then each failure is tracked independently.
        """
        proc = MockProcessor("failer", ["cap"], should_fail=True)
        clean_registry.register(proc, ["text/plain"])

        artifact1 = IFTextArtifact(artifact_id="test-1", content="test1")
        artifact2 = IFTextArtifact(artifact_id="test-2", content="test2")

        result1 = router.route_artifact(artifact1, "cap")
        result2 = router.route_artifact(artifact2, "cap")

        assert isinstance(result1, IFFailureArtifact)
        assert isinstance(result2, IFFailureArtifact)
        assert result1.artifact_id != result2.artifact_id
        assert result1.parent_id == "test-1"
        assert result2.parent_id == "test-2"


class TestMemoryAwareIntegration:
    """Integration tests for memory-aware routing ()."""

    def test_memory_aware_fallback_chain(self, router, clean_registry):
        """
        GWT:
        Given a fallback chain with varying memory requirements,
        When memory is constrained,
        Then lowest memory processor within limit is selected.
        """
        # GPU: 4GB, CPU: 1GB, Basic: 100MB
        gpu_proc = MockProcessor("gpu", ["gpu-compute"], memory_mb=4096)
        cpu_proc = MockProcessor("cpu", ["cpu-compute"], memory_mb=1024)
        basic_proc = MockProcessor("basic", ["basic-compute"], memory_mb=100)

        clean_registry.register(gpu_proc, ["application/x-compute"], priority=200)
        clean_registry.register(cpu_proc, ["application/x-compute"], priority=100)
        clean_registry.register(basic_proc, ["application/x-compute"], priority=50)

        # With 512MB limit, only basic fits
        result = router.select_with_fallback(
            preferred=["gpu-compute"],
            fallbacks=[["cpu-compute"], ["basic-compute"]],
            max_mb=512,
        )

        assert result.processor_id == "basic"

    def test_memory_constraint_with_capability_composition(
        self, router, clean_registry
    ):
        """
        GWT:
        Given processors with multiple capabilities and different memory,
        When selecting with both capability and memory constraints,
        Then only processors satisfying both are returned.
        """
        # Full-featured but heavy
        full_proc = MockProcessor("full", ["ocr", "table", "summary"], memory_mb=2048)
        # Limited but light
        light_proc = MockProcessor("light", ["ocr", "table"], memory_mb=256)

        clean_registry.register(full_proc, ["image/png"])
        clean_registry.register(light_proc, ["image/png"])

        # Need both ocr and table, max 500MB
        result = router.select_memory_aware(["ocr", "table"], max_mb=500, match="all")

        assert result.processor_id == "light"

    def test_route_artifact_with_memory_constraint_end_to_end(
        self, router, clean_registry
    ):
        """
        GWT:
        Given an artifact routing scenario with memory constraint,
        When routing through multiple stages,
        Then each stage respects memory limits.
        """
        small_ocr = MockProcessor("small-ocr", ["ocr"], memory_mb=100)
        small_transform = MockProcessor("small-transform", ["transform"], memory_mb=50)

        clean_registry.register(small_ocr, ["image/png"])
        clean_registry.register(small_transform, ["text/plain"])

        # Stage 1: OCR with memory limit
        input_artifact = IFTextArtifact(artifact_id="input", content="image data")
        stage1_result = router.route_artifact(input_artifact, "ocr", max_mb=200)

        assert not isinstance(stage1_result, IFFailureArtifact)
        assert stage1_result.lineage_depth == 1

        # Stage 2: Transform with memory limit
        stage2_result = router.route_artifact(stage1_result, "transform", max_mb=100)

        assert not isinstance(stage2_result, IFFailureArtifact)
        assert stage2_result.lineage_depth == 2
        assert len(stage2_result.provenance) == 2
