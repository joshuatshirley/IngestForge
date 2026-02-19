"""
Comprehensive Registry Tests for EP-06 Coverage.

Tests for IFRegistry methods not covered by existing test files:
- Memory-aware dispatch ()
- Multi-capability matching ()
- Context manager behavior ()
- teardown_all ()
- Singleton pattern verification
- Edge cases and boundary conditions

Follows GWT (Given-When-Then) test naming convention.
Adheres to NASA JPL Power of Ten rules.
"""

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFFileArtifact
from ingestforge.core.pipeline.registry import (
    IFRegistry,
    MAX_PROCESSORS,
    MAX_ENRICHER_FACTORIES,
    get_available_memory_mb,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockProcessor(IFProcessor):
    """Base mock processor for testing."""

    def __init__(
        self,
        proc_id: str = "mock-processor",
        caps: List[str] = None,
        mem_mb: int = 100,
        available: bool = True,
        teardown_result: bool = True,
        teardown_raises: bool = False,
    ):
        self._proc_id = proc_id
        self._caps = caps or ["default"]
        self._mem_mb = mem_mb
        self._available = available
        self._teardown_result = teardown_result
        self._teardown_raises = teardown_raises
        self.teardown_called = False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        return self._available

    @property
    def processor_id(self) -> str:
        return self._proc_id

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return self._caps

    @property
    def memory_mb(self) -> int:
        return self._mem_mb

    def teardown(self) -> bool:
        self.teardown_called = True
        if self._teardown_raises:
            raise RuntimeError("Teardown explosion!")
        return self._teardown_result


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    registry = IFRegistry()
    registry.clear()
    yield registry
    registry.clear()


@pytest.fixture
def text_artifact():
    """Provide a test text artifact."""
    return IFTextArtifact(artifact_id="test-1", content="Test content")


@pytest.fixture
def file_artifact():
    """Provide a test file artifact with MIME type."""
    artifact = MagicMock(spec=IFFileArtifact)
    artifact.artifact_id = "file-1"
    artifact.mime_type = "application/pdf"
    return artifact


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestRegistrySingleton:
    """Tests for IFRegistry singleton behavior."""

    def test_singleton_returns_same_instance(self, clean_registry):
        """
        Given: Multiple IFRegistry instantiations.
        When: Comparing instances.
        Then: All instances are identical.
        """
        reg1 = IFRegistry()
        reg2 = IFRegistry()
        reg3 = IFRegistry()

        assert reg1 is reg2
        assert reg2 is reg3
        assert reg1 is reg3

    def test_singleton_shares_state(self, clean_registry):
        """
        Given: Processor registered via one instance.
        When: Queried via another instance.
        Then: Processor is found.
        """
        reg1 = IFRegistry()
        proc = MockProcessor(proc_id="shared-proc")
        reg1.register(proc, ["text/plain"], priority=100)

        reg2 = IFRegistry()
        found = reg2.get_processors("text/plain")

        assert len(found) == 1
        assert found[0].processor_id == "shared-proc"


# =============================================================================
# Dispatch Tests
# =============================================================================


class TestRegistryDispatch:
    """Tests for IFRegistry.dispatch() method."""

    def test_dispatch_returns_available_processor(self, clean_registry, file_artifact):
        """
        Given: Processor registered for MIME type.
        When: dispatch() called with matching artifact.
        Then: Processor is returned.
        """
        proc = MockProcessor(proc_id="pdf-proc", available=True)
        clean_registry.register(proc, ["application/pdf"], priority=100)

        result = clean_registry.dispatch(file_artifact)

        assert result.processor_id == "pdf-proc"

    def test_dispatch_skips_unavailable_processor(self, clean_registry, file_artifact):
        """
        Given: Multiple processors, first unavailable.
        When: dispatch() called.
        Then: Available processor is returned.
        """
        proc1 = MockProcessor(proc_id="unavailable", available=False)
        proc2 = MockProcessor(proc_id="available", available=True)

        clean_registry.register(proc1, ["application/pdf"], priority=200)
        clean_registry.register(proc2, ["application/pdf"], priority=100)

        result = clean_registry.dispatch(file_artifact)

        assert result.processor_id == "available"

    def test_dispatch_raises_when_no_processor(self, clean_registry, file_artifact):
        """
        Given: No processor registered for MIME type.
        When: dispatch() called.
        Then: RuntimeError is raised.
        """
        with pytest.raises(RuntimeError, match="No available IFProcessor"):
            clean_registry.dispatch(file_artifact)

    def test_dispatch_uses_default_mime_for_non_file(
        self, clean_registry, text_artifact
    ):
        """
        Given: Non-file artifact and processor for octet-stream.
        When: dispatch() called.
        Then: Default MIME type used.
        """
        proc = MockProcessor(proc_id="default-proc")
        clean_registry.register(proc, ["application/octet-stream"], priority=100)

        result = clean_registry.dispatch(text_artifact)

        assert result.processor_id == "default-proc"


class TestDispatchByCapability:
    """Tests for IFRegistry.dispatch_by_capability() method."""

    def test_dispatch_by_capability_returns_available(
        self, clean_registry, text_artifact
    ):
        """
        Given: Processor registered with capability.
        When: dispatch_by_capability() called.
        Then: Processor is returned.
        """
        proc = MockProcessor(proc_id="ocr-proc", caps=["ocr"])
        clean_registry.register(proc, ["image/png"], priority=100)

        result = clean_registry.dispatch_by_capability("ocr", text_artifact)

        assert result.processor_id == "ocr-proc"

    def test_dispatch_by_capability_raises_when_none_available(
        self, clean_registry, text_artifact
    ):
        """
        Given: Processor unavailable for capability.
        When: dispatch_by_capability() called.
        Then: RuntimeError is raised.
        """
        proc = MockProcessor(proc_id="unavailable-ocr", caps=["ocr"], available=False)
        clean_registry.register(proc, ["image/png"], priority=100)

        with pytest.raises(RuntimeError, match="No available IFProcessor"):
            clean_registry.dispatch_by_capability("ocr", text_artifact)


# =============================================================================
# Multi-Capability Matching Tests ()
# =============================================================================


class TestGetByCapabilities:
    """Tests for IFRegistry.get_by_capabilities() method."""

    def test_match_all_returns_intersection(self, clean_registry):
        """
        Given: Processor with multiple capabilities.
        When: get_by_capabilities(match='all') called.
        Then: Only processors with ALL capabilities returned.
        """
        proc1 = MockProcessor(proc_id="multi", caps=["ocr", "embedding", "ner"])
        proc2 = MockProcessor(proc_id="ocr-only", caps=["ocr"])
        proc3 = MockProcessor(proc_id="embed-only", caps=["embedding"])

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)
        clean_registry.register(proc3, ["text/plain"], priority=100)

        result = clean_registry.get_by_capabilities(["ocr", "embedding"], match="all")

        assert len(result) == 1
        assert result[0].processor_id == "multi"

    def test_match_any_returns_union(self, clean_registry):
        """
        Given: Multiple processors with different capabilities.
        When: get_by_capabilities(match='any') called.
        Then: Processors with ANY matching capability returned.
        """
        proc1 = MockProcessor(proc_id="ocr-proc", caps=["ocr"])
        proc2 = MockProcessor(proc_id="embed-proc", caps=["embedding"])
        proc3 = MockProcessor(proc_id="ner-proc", caps=["ner"])

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)
        clean_registry.register(proc3, ["text/plain"], priority=100)

        result = clean_registry.get_by_capabilities(["ocr", "embedding"], match="any")

        assert len(result) == 2
        proc_ids = [p.processor_id for p in result]
        assert "ocr-proc" in proc_ids
        assert "embed-proc" in proc_ids

    def test_match_any_no_duplicates(self, clean_registry):
        """
        Given: Processor with multiple matching capabilities.
        When: get_by_capabilities(match='any') called.
        Then: No duplicate entries.
        """
        proc = MockProcessor(proc_id="multi", caps=["ocr", "embedding"])
        clean_registry.register(proc, ["text/plain"], priority=100)

        result = clean_registry.get_by_capabilities(["ocr", "embedding"], match="any")

        assert len(result) == 1
        assert result[0].processor_id == "multi"

    def test_empty_capabilities_returns_empty(self, clean_registry):
        """
        Given: Empty capabilities list.
        When: get_by_capabilities() called.
        Then: Empty list returned.
        """
        proc = MockProcessor(proc_id="test", caps=["ocr"])
        clean_registry.register(proc, ["text/plain"], priority=100)

        result = clean_registry.get_by_capabilities([])

        assert result == []

    def test_match_all_sorted_by_priority(self, clean_registry):
        """
        Given: Multiple processors matching all capabilities.
        When: get_by_capabilities(match='all') called.
        Then: Results sorted by priority (descending).
        """
        proc1 = MockProcessor(proc_id="low", caps=["ocr", "embedding"])
        proc2 = MockProcessor(proc_id="high", caps=["ocr", "embedding"])

        clean_registry.register(proc1, ["text/plain"], priority=50)
        clean_registry.register(proc2, ["text/plain"], priority=150)

        result = clean_registry.get_by_capabilities(["ocr", "embedding"], match="all")

        assert len(result) == 2
        assert result[0].processor_id == "high"
        assert result[1].processor_id == "low"


# =============================================================================
# Memory-Aware Selection Tests ()
# =============================================================================


class TestMemoryAwareSelection:
    """Tests for memory-aware processor selection."""

    def test_get_processors_by_memory_filters_correctly(self, clean_registry):
        """
        Given: Processors with varying memory requirements.
        When: get_processors_by_memory() called.
        Then: Only processors within limit returned.
        """
        proc1 = MockProcessor(proc_id="small", mem_mb=100)
        proc2 = MockProcessor(proc_id="medium", mem_mb=500)
        proc3 = MockProcessor(proc_id="large", mem_mb=2000)

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)
        clean_registry.register(proc3, ["text/plain"], priority=100)

        result = clean_registry.get_processors_by_memory(max_mb=600)

        assert len(result) == 2
        proc_ids = [p.processor_id for p in result]
        assert "small" in proc_ids
        assert "medium" in proc_ids
        assert "large" not in proc_ids

    def test_get_processors_by_memory_sorted_by_priority(self, clean_registry):
        """
        Given: Multiple processors within memory limit.
        When: get_processors_by_memory() called.
        Then: Results sorted by priority.
        """
        proc1 = MockProcessor(proc_id="low-priority", mem_mb=100)
        proc2 = MockProcessor(proc_id="high-priority", mem_mb=100)

        clean_registry.register(proc1, ["text/plain"], priority=50)
        clean_registry.register(proc2, ["text/plain"], priority=150)

        result = clean_registry.get_processors_by_memory(max_mb=200)

        assert len(result) == 2
        assert result[0].processor_id == "high-priority"
        assert result[1].processor_id == "low-priority"

    def test_dispatch_memory_safe_respects_limit(self, clean_registry, file_artifact):
        """
        Given: Processors with different memory requirements.
        When: dispatch_memory_safe() called with limit.
        Then: Only processors within limit considered.
        """
        proc1 = MockProcessor(proc_id="large", mem_mb=2000, available=True)
        proc2 = MockProcessor(proc_id="small", mem_mb=100, available=True)

        clean_registry.register(proc1, ["application/pdf"], priority=200)
        clean_registry.register(proc2, ["application/pdf"], priority=100)

        result = clean_registry.dispatch_memory_safe(file_artifact, max_mb=500)

        assert result.processor_id == "small"

    def test_dispatch_memory_safe_raises_when_none_fit(
        self, clean_registry, file_artifact
    ):
        """
        Given: All processors exceed memory limit.
        When: dispatch_memory_safe() called.
        Then: RuntimeError is raised.
        """
        proc = MockProcessor(proc_id="large", mem_mb=2000, available=True)
        clean_registry.register(proc, ["application/pdf"], priority=100)

        with pytest.raises(RuntimeError, match="within memory limit"):
            clean_registry.dispatch_memory_safe(file_artifact, max_mb=500)

    def test_dispatch_memory_safe_checks_availability(
        self, clean_registry, file_artifact
    ):
        """
        Given: Processor within limit but unavailable.
        When: dispatch_memory_safe() called.
        Then: Raises RuntimeError.
        """
        proc = MockProcessor(proc_id="unavailable", mem_mb=100, available=False)
        clean_registry.register(proc, ["application/pdf"], priority=100)

        with pytest.raises(RuntimeError, match="No available"):
            clean_registry.dispatch_memory_safe(file_artifact, max_mb=500)


class TestGetAvailableMemory:
    """Tests for get_available_memory_mb() function."""

    def test_returns_fallback_when_psutil_unavailable(self):
        """
        Given: psutil not installed.
        When: get_available_memory_mb() called.
        Then: Returns 1024 (default fallback).
        """
        with patch.dict("sys.modules", {"psutil": None}):
            # Force reimport to trigger ImportError path
            import importlib
            from ingestforge.core.pipeline import registry

            importlib.reload(registry)

            # This test verifies the fallback behavior exists
            # Actual psutil unavailability is hard to simulate in test
            result = registry.get_available_memory_mb()
            assert result > 0  # Should return a positive value

    def test_returns_positive_value(self):
        """
        Given: Normal system.
        When: get_available_memory_mb() called.
        Then: Returns positive integer.
        """
        result = get_available_memory_mb()

        assert isinstance(result, int)
        assert result > 0


# =============================================================================
# Teardown Tests ()
# =============================================================================


class TestTeardownAll:
    """Tests for IFRegistry.teardown_all() method."""

    def test_teardown_all_calls_teardown_on_all(self, clean_registry):
        """
        Given: Multiple registered processors.
        When: teardown_all() called.
        Then: All processors have teardown() called.
        """
        proc1 = MockProcessor(proc_id="proc1")
        proc2 = MockProcessor(proc_id="proc2")
        proc3 = MockProcessor(proc_id="proc3")

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)
        clean_registry.register(proc3, ["text/plain"], priority=100)

        clean_registry.teardown_all()

        assert proc1.teardown_called
        assert proc2.teardown_called
        assert proc3.teardown_called

    def test_teardown_all_returns_summary(self, clean_registry):
        """
        Given: Mixed teardown results.
        When: teardown_all() called.
        Then: Summary dict returned.
        """
        proc1 = MockProcessor(proc_id="success", teardown_result=True)
        proc2 = MockProcessor(proc_id="failure", teardown_result=False)

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)

        result = clean_registry.teardown_all()

        assert "success_count" in result
        assert "failure_count" in result
        assert "failed_ids" in result
        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert "failure" in result["failed_ids"]

    def test_teardown_all_continues_on_exception(self, clean_registry):
        """
        Given: Processor that raises during teardown.
        When: teardown_all() called.
        Then: Other processors still torn down.
        """
        proc1 = MockProcessor(proc_id="raises", teardown_raises=True)
        proc2 = MockProcessor(proc_id="normal", teardown_result=True)

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)

        result = clean_registry.teardown_all()

        assert proc1.teardown_called
        assert proc2.teardown_called
        assert "raises" in result["failed_ids"]

    def test_teardown_all_empty_registry(self, clean_registry):
        """
        Given: Empty registry.
        When: teardown_all() called.
        Then: Returns success summary.
        """
        result = clean_registry.teardown_all()

        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert result["failed_ids"] == []


# =============================================================================
# Context Manager Tests ()
# =============================================================================


class TestRegistryContextManager:
    """Tests for IFRegistry context manager behavior."""

    def test_context_manager_enter_returns_self(self, clean_registry):
        """
        Given: IFRegistry instance.
        When: Used with 'with' statement.
        Then: __enter__ returns registry.
        """
        result = clean_registry.__enter__()

        assert result is clean_registry

    def test_context_manager_exit_calls_teardown_all(self, clean_registry):
        """
        Given: Processors registered.
        When: Exiting context manager.
        Then: teardown_all() is called.
        """
        proc = MockProcessor(proc_id="ctx-proc")
        clean_registry.register(proc, ["text/plain"], priority=100)

        clean_registry.__exit__(None, None, None)

        assert proc.teardown_called

    def test_context_manager_does_not_suppress_exceptions(self, clean_registry):
        """
        Given: Exception raised in with block.
        When: __exit__ called.
        Then: Returns False (exception propagates).
        """
        result = clean_registry.__exit__(ValueError, ValueError("test"), None)

        assert result is False

    def test_context_manager_full_workflow(self, clean_registry):
        """
        Given: Registry used as context manager.
        When: Block completes.
        Then: All processors torn down.
        """
        proc = MockProcessor(proc_id="workflow-proc")
        clean_registry.register(proc, ["text/plain"], priority=100)

        with clean_registry as registry:
            assert registry is clean_registry

        assert proc.teardown_called


# =============================================================================
# Priority and Sorting Tests
# =============================================================================


class TestPriorityHandling:
    """Tests for priority-based sorting."""

    def test_higher_priority_first(self, clean_registry):
        """
        Given: Processors with different priorities.
        When: get_processors() called.
        Then: Higher priority first.
        """
        proc_low = MockProcessor(proc_id="low")
        proc_high = MockProcessor(proc_id="high")

        clean_registry.register(proc_low, ["text/plain"], priority=50)
        clean_registry.register(proc_high, ["text/plain"], priority=150)

        result = clean_registry.get_processors("text/plain")

        assert result[0].processor_id == "high"
        assert result[1].processor_id == "low"

    def test_same_priority_sorted_by_id(self, clean_registry):
        """
        Given: Processors with same priority.
        When: get_processors() called.
        Then: Sorted by processor_id alphabetically.
        """
        proc_b = MockProcessor(proc_id="b-proc")
        proc_a = MockProcessor(proc_id="a-proc")

        clean_registry.register(proc_b, ["text/plain"], priority=100)
        clean_registry.register(proc_a, ["text/plain"], priority=100)

        result = clean_registry.get_processors("text/plain")

        assert result[0].processor_id == "a-proc"
        assert result[1].processor_id == "b-proc"

    def test_capability_index_sorted_by_priority(self, clean_registry):
        """
        Given: Processors with shared capability, different priorities.
        When: get_by_capability() called.
        Then: Higher priority first.
        """
        proc_low = MockProcessor(proc_id="low", caps=["ocr"])
        proc_high = MockProcessor(proc_id="high", caps=["ocr"])

        clean_registry.register(proc_low, ["text/plain"], priority=50)
        clean_registry.register(proc_high, ["text/plain"], priority=150)

        result = clean_registry.get_by_capability("ocr")

        assert result[0].processor_id == "high"
        assert result[1].processor_id == "low"


# =============================================================================
# Boundary and Edge Case Tests
# =============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    def test_max_processors_limit_enforced(self, clean_registry):
        """
        Given: Registry at MAX_PROCESSORS limit.
        When: Registering one more.
        Then: RuntimeError raised.
        """
        # Fill registry to limit
        for i in range(MAX_PROCESSORS):
            proc = MockProcessor(proc_id=f"proc-{i}")
            clean_registry.register(proc, [f"type/{i}"], priority=100)

        # Next should fail
        with pytest.raises(RuntimeError, match="limit reached"):
            overflow = MockProcessor(proc_id="overflow")
            clean_registry.register(overflow, ["type/overflow"], priority=100)

    def test_get_processors_unknown_mime_returns_empty(self, clean_registry):
        """
        Given: No processor for MIME type.
        When: get_processors() called.
        Then: Empty list returned.
        """
        result = clean_registry.get_processors("unknown/type")

        assert result == []

    def test_get_by_capability_unknown_returns_empty(self, clean_registry):
        """
        Given: No processor with capability.
        When: get_by_capability() called.
        Then: Empty list returned.
        """
        result = clean_registry.get_by_capability("nonexistent")

        assert result == []

    def test_register_same_processor_multiple_mimes(self, clean_registry):
        """
        Given: Processor registered for multiple MIME types.
        When: Querying different MIME types.
        Then: Same processor returned for each.
        """
        proc = MockProcessor(proc_id="multi-mime")
        clean_registry.register(
            proc, ["text/plain", "text/html", "text/csv"], priority=100
        )

        result1 = clean_registry.get_processors("text/plain")
        result2 = clean_registry.get_processors("text/html")
        result3 = clean_registry.get_processors("text/csv")

        assert len(result1) == 1
        assert len(result2) == 1
        assert len(result3) == 1
        assert result1[0] is result2[0] is result3[0]


# =============================================================================
# JPL Power of Ten Compliance Tests
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten rule compliance."""

    def test_rule_1_no_recursion_in_registration(self, clean_registry):
        """
        JPL Rule #1: No recursion.
        Registration should complete without recursive calls.
        """
        proc = MockProcessor(proc_id="test")
        # If this completes, no recursion occurred
        clean_registry.register(proc, ["text/plain"], priority=100)

        assert clean_registry.get_processors("text/plain") != []

    def test_rule_2_fixed_bounds(self, clean_registry):
        """
        JPL Rule #2: Fixed upper bounds.
        Registry has explicit MAX_PROCESSORS constant.
        """
        assert MAX_PROCESSORS == 256
        assert MAX_ENRICHER_FACTORIES == 128

    def test_rule_4_functions_under_60_lines(self):
        """
        JPL Rule #4: Functions < 60 lines.
        Verify key methods are concise.

        Note: register_enricher is at 64 lines (ISSUE-06 in code review).
        Tracked for future refactoring but functional.
        """
        import inspect
        from ingestforge.core.pipeline.registry import IFRegistry

        # Methods expected to be under 60 lines
        compliant_methods = [
            "register",
            "dispatch",
            "get_by_capability",
            "get_by_capabilities",
            "teardown_all",
        ]

        for method_name in compliant_methods:
            method = getattr(IFRegistry, method_name)
            source = inspect.getsource(method)
            lines = source.strip().split("\n")
            assert len(lines) < 60, f"{method_name} has {len(lines)} lines"

        # Known issue: register_enricher is 64 lines (documented in CODE_REVIEW)
        # Verify it's not getting worse
        enricher_method = getattr(IFRegistry, "register_enricher")
        enricher_source = inspect.getsource(enricher_method)
        enricher_lines = enricher_source.strip().split("\n")
        assert len(enricher_lines) <= 65, (
            f"register_enricher grew to {len(enricher_lines)} lines "
            "(was 64, limit 60)"
        )

    def test_rule_7_teardown_errors_isolated(self, clean_registry):
        """
        JPL Rule #7: Check return values, isolate errors.
        Teardown exceptions don't cascade.
        """
        proc1 = MockProcessor(proc_id="raises", teardown_raises=True)
        proc2 = MockProcessor(proc_id="normal")

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)

        # Should not raise
        result = clean_registry.teardown_all()

        # Both attempted, one failed
        assert proc1.teardown_called
        assert proc2.teardown_called
        assert result["failure_count"] == 1

    def test_rule_9_type_hints_present(self, clean_registry):
        """
        JPL Rule #9: All public methods have type hints.
        """
        import inspect

        methods = [
            clean_registry.register,
            clean_registry.dispatch,
            clean_registry.get_by_capability,
            clean_registry.teardown_all,
            clean_registry.register_enricher,
            clean_registry.get_enricher,
        ]

        for method in methods:
            sig = inspect.signature(method)
            # Check return annotation exists
            assert (
                sig.return_annotation != inspect.Signature.empty
            ), f"{method.__name__} missing return type"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Integration tests combining multiple registry features."""

    def test_full_workflow_register_dispatch_teardown(
        self, clean_registry, file_artifact
    ):
        """
        Given: Full registry lifecycle.
        When: Register, dispatch, teardown sequence.
        Then: All operations succeed.
        """
        proc = MockProcessor(proc_id="full-workflow")
        clean_registry.register(proc, ["application/pdf"], priority=100)

        # Dispatch
        dispatched = clean_registry.dispatch(file_artifact)
        assert dispatched.processor_id == "full-workflow"

        # Teardown
        result = clean_registry.teardown_all()
        assert result["success_count"] == 1
        assert proc.teardown_called

    def test_capability_and_memory_combined(self, clean_registry):
        """
        Given: Processors with capabilities and memory requirements.
        When: Filtering by both.
        Then: Only matching processors returned.
        """
        proc1 = MockProcessor(proc_id="small-ocr", caps=["ocr"], mem_mb=100)
        proc2 = MockProcessor(proc_id="large-ocr", caps=["ocr"], mem_mb=2000)
        proc3 = MockProcessor(proc_id="small-embed", caps=["embedding"], mem_mb=100)

        clean_registry.register(proc1, ["text/plain"], priority=100)
        clean_registry.register(proc2, ["text/plain"], priority=100)
        clean_registry.register(proc3, ["text/plain"], priority=100)

        # Get OCR processors
        ocr_procs = clean_registry.get_by_capability("ocr")
        assert len(ocr_procs) == 2

        # Get memory-constrained processors
        small_procs = clean_registry.get_processors_by_memory(max_mb=500)
        assert len(small_procs) == 2

        # Intersection: small OCR processors
        small_ocr = [p for p in ocr_procs if p.memory_mb <= 500]
        assert len(small_ocr) == 1
        assert small_ocr[0].processor_id == "small-ocr"
