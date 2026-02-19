import pytest
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.registry import IFRegistry, register_if_processor
from ingestforge.core.pipeline.artifacts import IFFileArtifact, IFTextArtifact


class MockProcessor(IFProcessor):
    @property
    def processor_id(self) -> str:
        return "mock-1"

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_available(self) -> bool:
        return True

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact


def test_registry_singleton():
    reg1 = IFRegistry()
    reg2 = IFRegistry()
    assert reg1 is reg2


def test_priority_and_tie_breaking():
    """
    GWT:
    Given processors with same priority
    When registered
    Then they must be sorted alphabetically by processor_id.
    """
    reg = IFRegistry()
    reg.clear()

    class ProcB(MockProcessor):
        @property
        def processor_id(self):
            return "proc-b"

    class ProcA(MockProcessor):
        @property
        def processor_id(self):
            return "proc-a"

    reg.register(ProcB(), ["test/tie"], priority=100)
    reg.register(ProcA(), ["test/tie"], priority=100)

    procs = reg.get_processors("test/tie")
    assert procs[0].processor_id == "proc-a"
    assert procs[1].processor_id == "proc-b"


def test_dispatch_with_artifact():
    """
    GWT:
    Given a FileArtifact
    When dispatch() is called
    Then it uses the artifact's mime_type.
    """
    reg = IFRegistry()
    reg.clear()

    @register_if_processor(mime_types=["application/pdf"])
    class PDFProc(MockProcessor):
        @property
        def processor_id(self):
            return "pdf-handler"

    art = IFFileArtifact(
        artifact_id="f1", file_path="test.pdf", mime_type="application/pdf"
    )
    proc = reg.dispatch(art)
    assert proc.processor_id == "pdf-handler"


def test_dispatch_fallback_default_mime():
    """
    GWT:
    Given a TextArtifact (no mime_type)
    When dispatch() is called
    Then it uses the default 'application/octet-stream'.
    """
    reg = IFRegistry()
    reg.clear()

    @register_if_processor(mime_types=["application/octet-stream"])
    class DefaultProc(MockProcessor):
        @property
        def processor_id(self):
            return "default-handler"

    art = IFTextArtifact(artifact_id="t1", content="text")
    proc = reg.dispatch(art)
    assert proc.processor_id == "default-handler"


# ============================================================================
# Capability - Functional Routing Tests
# ============================================================================


class CapabilityProcessor(MockProcessor):
    """Base class for capability-aware test processors."""

    def __init__(self, proc_id: str, caps: list[str]):
        self._id = proc_id
        self._caps = caps

    @property
    def processor_id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> list[str]:
        return self._caps


class TestCapabilityRegistration:
    """Scenario 1: Capability Registration"""

    def test_processor_registered_by_capability(self):
        """
        GWT:
        Given an IFProcessor with declared capabilities,
        When registered with IFRegistry,
        Then the processor is indexed by each capability string.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityProcessor("ocr-proc", ["ocr", "table-extraction"])
        reg.register(proc, ["image/png"])

        # Should be indexed under both capabilities
        assert proc in reg.get_by_capability("ocr")
        assert proc in reg.get_by_capability("table-extraction")

    def test_empty_capabilities_not_indexed(self):
        """
        GWT:
        Given a processor with no capabilities,
        When registered,
        Then the capability index remains empty for that processor.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityProcessor("no-cap-proc", [])
        reg.register(proc, ["text/plain"])

        assert reg.get_by_capability("any-capability") == []


class TestCapabilityQuery:
    """Scenario 2: Capability Query"""

    def test_get_by_capability_returns_sorted(self):
        """
        GWT:
        Given multiple processors with "ocr" capability,
        When get_by_capability("ocr") is called,
        Then all matching processors are returned sorted by priority.
        """
        reg = IFRegistry()
        reg.clear()

        proc_low = CapabilityProcessor("ocr-low", ["ocr"])
        proc_high = CapabilityProcessor("ocr-high", ["ocr"])

        reg.register(proc_low, ["image/png"], priority=50)
        reg.register(proc_high, ["image/png"], priority=200)

        result = reg.get_by_capability("ocr")
        assert len(result) == 2
        assert result[0].processor_id == "ocr-high"  # Higher priority first
        assert result[1].processor_id == "ocr-low"

    def test_get_by_capability_no_match(self):
        """
        GWT:
        Given a request for "unknown-capability",
        When get_by_capability("unknown-capability") is called,
        Then an empty list is returned (not an error).
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.get_by_capability("unknown-capability")
        assert result == []


class TestMultiCapabilityMatch:
    """Scenario 3: Multi-Capability Match"""

    def test_get_by_capabilities_all_match(self):
        """
        GWT:
        Given a request for ["ocr", "table-extraction"] capabilities,
        When get_by_capabilities(["ocr", "table-extraction"]) is called,
        Then only processors with ALL required capabilities are returned.
        """
        reg = IFRegistry()
        reg.clear()

        # Processor with both capabilities
        proc_both = CapabilityProcessor("both-caps", ["ocr", "table-extraction"])
        # Processor with only one
        proc_ocr_only = CapabilityProcessor("ocr-only", ["ocr"])
        # Processor with different
        proc_other = CapabilityProcessor("other", ["embedding"])

        reg.register(proc_both, ["image/png"], priority=100)
        reg.register(proc_ocr_only, ["image/png"], priority=200)
        reg.register(proc_other, ["text/plain"], priority=50)

        result = reg.get_by_capabilities(["ocr", "table-extraction"], match="all")
        assert len(result) == 1
        assert result[0].processor_id == "both-caps"

    def test_get_by_capabilities_any_match(self):
        """
        GWT:
        Given a request for ["ocr", "embedding"] capabilities with match="any",
        When get_by_capabilities(..., match="any") is called,
        Then processors with ANY of the capabilities are returned.
        """
        reg = IFRegistry()
        reg.clear()

        proc_ocr = CapabilityProcessor("ocr-proc", ["ocr"])
        proc_embed = CapabilityProcessor("embed-proc", ["embedding"])
        proc_other = CapabilityProcessor("other-proc", ["summarization"])

        reg.register(proc_ocr, ["image/png"], priority=100)
        reg.register(proc_embed, ["text/plain"], priority=150)
        reg.register(proc_other, ["text/plain"], priority=50)

        result = reg.get_by_capabilities(["ocr", "embedding"], match="any")
        assert len(result) == 2
        # Sorted by priority
        assert result[0].processor_id == "embed-proc"
        assert result[1].processor_id == "ocr-proc"

    def test_get_by_capabilities_empty_list(self):
        """
        GWT:
        Given an empty capabilities list,
        When get_by_capabilities([]) is called,
        Then an empty list is returned.
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.get_by_capabilities([])
        assert result == []


class TestCapabilityDispatch:
    """Scenario 4: Capability-Based Dispatch"""

    def test_dispatch_by_capability_returns_available(self):
        """
        GWT:
        Given an artifact requiring "embedding" capability,
        When dispatch_by_capability("embedding", artifact) is called,
        Then the first available processor with that capability is returned.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityProcessor("embed-proc", ["embedding"])
        reg.register(proc, ["text/plain"], priority=100)

        art = IFTextArtifact(artifact_id="t1", content="test")
        result = reg.dispatch_by_capability("embedding", art)

        assert result.processor_id == "embed-proc"

    def test_dispatch_by_capability_skips_unavailable(self):
        """
        GWT:
        Given multiple processors, one unavailable,
        When dispatch_by_capability is called,
        Then the unavailable processor is skipped.
        """
        reg = IFRegistry()
        reg.clear()

        class UnavailableProc(CapabilityProcessor):
            def is_available(self) -> bool:
                return False

        proc_unavail = UnavailableProc("unavail", ["ocr"])
        proc_avail = CapabilityProcessor("avail", ["ocr"])

        reg.register(proc_unavail, ["image/png"], priority=200)  # Higher priority
        reg.register(proc_avail, ["image/png"], priority=100)

        art = IFFileArtifact(
            artifact_id="f1", file_path="test.png", mime_type="image/png"
        )
        result = reg.dispatch_by_capability("ocr", art)

        assert result.processor_id == "avail"

    def test_dispatch_by_capability_raises_when_none(self):
        """
        GWT:
        Given no processors with required capability,
        When dispatch_by_capability is called,
        Then RuntimeError is raised.
        """
        reg = IFRegistry()
        reg.clear()

        art = IFTextArtifact(artifact_id="t1", content="test")

        with pytest.raises(RuntimeError) as exc_info:
            reg.dispatch_by_capability("nonexistent", art)

        assert "No available IFProcessor" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)


class TestCapabilityClear:
    """Test that clear() also clears capability index."""

    def test_clear_removes_capabilities(self):
        """
        GWT:
        Given processors registered with capabilities,
        When clear() is called,
        Then capability index is also cleared.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityProcessor("test-proc", ["ocr", "embedding"])
        reg.register(proc, ["image/png"])

        assert len(reg.get_by_capability("ocr")) == 1

        reg.clear()

        assert reg.get_by_capability("ocr") == []
        assert reg.get_by_capability("embedding") == []


# ============================================================================
# Resources - Memory-Aware Selection Tests
# ============================================================================


class MemoryAwareProcessor(MockProcessor):
    """Base class for memory-aware test processors."""

    def __init__(self, proc_id: str, memory_mb: int):
        self._id = proc_id
        self._memory_mb = memory_mb

    @property
    def processor_id(self) -> str:
        return self._id

    @property
    def memory_mb(self) -> int:
        return self._memory_mb


class TestMemoryRequirementDeclaration:
    """Scenario 1: Memory Requirement Declaration"""

    def test_processor_default_memory_mb(self):
        """
        GWT:
        Given a processor without declared memory requirements,
        When memory_mb is accessed,
        Then it returns the default 100MB.
        """
        proc = MockProcessor()
        assert proc.memory_mb == 100

    def test_processor_custom_memory_mb(self):
        """
        GWT:
        Given a processor with declared memory requirements,
        When memory_mb is accessed,
        Then it returns the declared value.
        """
        proc = MemoryAwareProcessor("heavy-proc", 2048)
        assert proc.memory_mb == 2048

    def test_processor_memory_stored_after_registration(self):
        """
        GWT:
        Given an IFProcessor with declared memory requirements,
        When registered with IFRegistry,
        Then the memory requirements are stored for later querying.
        """
        reg = IFRegistry()
        reg.clear()

        proc = MemoryAwareProcessor("mem-proc", 512)
        reg.register(proc, ["text/plain"])

        # Verify we can query by memory
        result = reg.get_processors_by_memory(512)
        assert len(result) == 1
        assert result[0].processor_id == "mem-proc"


class TestMemorySafeDispatch:
    """Scenario 2: Memory-Safe Dispatch"""

    def test_dispatch_memory_safe_filters_by_limit(self):
        """
        GWT:
        Given an artifact with estimated size metadata,
        When dispatch_memory_safe(artifact, max_mb) is called,
        Then only processors whose memory requirements fit within limit are considered.
        """
        reg = IFRegistry()
        reg.clear()

        # Heavy processor (too much memory)
        heavy_proc = MemoryAwareProcessor("heavy", 2048)
        # Light processor (fits in memory)
        light_proc = MemoryAwareProcessor("light", 256)

        # Use application/octet-stream (default for non-file artifacts)
        reg.register(heavy_proc, ["application/octet-stream"], priority=200)
        reg.register(light_proc, ["application/octet-stream"], priority=100)

        art = IFTextArtifact(artifact_id="t1", content="test")

        # With 500MB limit, only light processor fits
        result = reg.dispatch_memory_safe(art, max_mb=500)
        assert result.processor_id == "light"

    def test_dispatch_memory_safe_respects_priority_within_limit(self):
        """
        GWT:
        Given multiple processors that fit within memory limit,
        When dispatch_memory_safe is called,
        Then the highest priority processor is returned.
        """
        reg = IFRegistry()
        reg.clear()

        proc_low = MemoryAwareProcessor("low-priority", 200)
        proc_high = MemoryAwareProcessor("high-priority", 300)

        # Use application/octet-stream (default for non-file artifacts)
        reg.register(proc_low, ["application/octet-stream"], priority=50)
        reg.register(proc_high, ["application/octet-stream"], priority=200)

        art = IFTextArtifact(artifact_id="t1", content="test")

        # Both fit within 500MB, so highest priority wins
        result = reg.dispatch_memory_safe(art, max_mb=500)
        assert result.processor_id == "high-priority"

    def test_dispatch_memory_safe_raises_when_none_fit(self):
        """
        GWT:
        Given no processors that fit within memory limit,
        When dispatch_memory_safe is called,
        Then RuntimeError is raised.
        """
        reg = IFRegistry()
        reg.clear()

        heavy_proc = MemoryAwareProcessor("heavy", 4096)
        reg.register(heavy_proc, ["text/plain"])

        art = IFTextArtifact(artifact_id="t1", content="test")

        with pytest.raises(RuntimeError) as exc_info:
            reg.dispatch_memory_safe(art, max_mb=100)

        assert "within memory limit" in str(exc_info.value)


class TestMemoryQuery:
    """Scenario 3: Memory Query"""

    def test_get_processors_by_memory_filters(self):
        """
        GWT:
        Given a registry with processors of varying memory requirements,
        When get_processors_by_memory(max_mb) is called,
        Then only processors requiring <= max_mb are returned.
        """
        reg = IFRegistry()
        reg.clear()

        small = MemoryAwareProcessor("small", 50)
        medium = MemoryAwareProcessor("medium", 200)
        large = MemoryAwareProcessor("large", 1000)

        reg.register(small, ["text/plain"])
        reg.register(medium, ["text/plain"])
        reg.register(large, ["text/plain"])

        result = reg.get_processors_by_memory(250)
        proc_ids = [p.processor_id for p in result]

        assert "small" in proc_ids
        assert "medium" in proc_ids
        assert "large" not in proc_ids

    def test_get_processors_by_memory_sorted_by_priority(self):
        """
        GWT:
        Given multiple processors within memory limit,
        When get_processors_by_memory is called,
        Then results are sorted by priority.
        """
        reg = IFRegistry()
        reg.clear()

        proc_a = MemoryAwareProcessor("a-proc", 100)
        proc_b = MemoryAwareProcessor("b-proc", 100)

        reg.register(proc_a, ["text/plain"], priority=50)
        reg.register(proc_b, ["text/plain"], priority=200)

        result = reg.get_processors_by_memory(500)
        assert result[0].processor_id == "b-proc"  # Higher priority first
        assert result[1].processor_id == "a-proc"

    def test_get_processors_by_memory_returns_empty_when_none_fit(self):
        """
        GWT:
        Given no processors that fit within memory limit,
        When get_processors_by_memory is called,
        Then an empty list is returned.
        """
        reg = IFRegistry()
        reg.clear()

        heavy = MemoryAwareProcessor("heavy", 4096)
        reg.register(heavy, ["text/plain"])

        result = reg.get_processors_by_memory(100)
        assert result == []


class TestDefaultMemoryRequirements:
    """Scenario 4: Default Memory Requirements"""

    def test_processor_without_memory_uses_default(self):
        """
        GWT:
        Given a processor without declared memory requirements,
        When registered,
        Then it uses the default memory estimate (100MB).
        """
        reg = IFRegistry()
        reg.clear()

        # MockProcessor doesn't override memory_mb, so uses default
        proc = MockProcessor()
        reg.register(proc, ["text/plain"])

        # Should appear in results for >= 100MB
        result = reg.get_processors_by_memory(100)
        assert len(result) == 1

        # Should NOT appear in results for < 100MB
        result = reg.get_processors_by_memory(50)
        assert len(result) == 0


class TestSystemMemoryCheck:
    """Scenario 5: System Memory Check"""

    def test_dispatch_memory_safe_with_no_limit_uses_system(self):
        """
        GWT:
        Given a system with available memory,
        When dispatch_memory_safe is called without max_mb,
        Then the registry queries available system memory.
        """
        reg = IFRegistry()
        reg.clear()

        proc = MemoryAwareProcessor("proc", 100)
        # Use application/octet-stream (default for non-file artifacts)
        reg.register(proc, ["application/octet-stream"])

        art = IFTextArtifact(artifact_id="t1", content="test")

        # Without max_mb, should use system memory (which should be > 100MB)
        result = reg.dispatch_memory_safe(art)
        assert result.processor_id == "proc"


class TestGetAvailableMemory:
    """Test the get_available_memory_mb utility function."""

    def test_get_available_memory_returns_int(self):
        """
        GWT:
        Given the system utility function,
        When get_available_memory_mb() is called,
        Then it returns an integer > 0.
        """
        from ingestforge.core.pipeline.registry import get_available_memory_mb

        memory = get_available_memory_mb()
        assert isinstance(memory, int)
        assert memory > 0

    def test_get_available_memory_returns_reasonable_value(self):
        """
        GWT:
        Given a typical system,
        When get_available_memory_mb() is called,
        Then it returns a value between 100MB and 1TB.
        """
        from ingestforge.core.pipeline.registry import get_available_memory_mb

        memory = get_available_memory_mb()
        # Should be at least 100MB (fallback) and less than 1TB
        assert 100 <= memory <= 1_000_000


# ============================================================================
# Teardown - Safe Resource Finalization Tests
# ============================================================================


class TeardownTrackingProcessor(MockProcessor):
    """Processor that tracks teardown calls."""

    def __init__(self, proc_id: str, teardown_result: bool = True):
        self._id = proc_id
        self._teardown_result = teardown_result
        self.teardown_called = False

    @property
    def processor_id(self) -> str:
        return self._id

    def teardown(self) -> bool:
        self.teardown_called = True
        return self._teardown_result


class FailingTeardownProcessor(MockProcessor):
    """Processor whose teardown raises an exception."""

    def __init__(self, proc_id: str):
        self._id = proc_id
        self.teardown_called = False

    @property
    def processor_id(self) -> str:
        return self._id

    def teardown(self) -> bool:
        self.teardown_called = True
        raise RuntimeError("Teardown explosion!")


class TestRegistryTeardownAll:
    """Scenario 1: Registry Teardown All"""

    def test_teardown_all_calls_each_processor(self):
        """
        GWT:
        Given a registry with multiple registered processors,
        When teardown_all() is called,
        Then teardown() is called on each processor.
        """
        reg = IFRegistry()
        reg.clear()

        proc1 = TeardownTrackingProcessor("proc-1")
        proc2 = TeardownTrackingProcessor("proc-2")
        proc3 = TeardownTrackingProcessor("proc-3")

        reg.register(proc1, ["text/plain"])
        reg.register(proc2, ["text/plain"])
        reg.register(proc3, ["text/plain"])

        reg.teardown_all()

        assert proc1.teardown_called
        assert proc2.teardown_called
        assert proc3.teardown_called

    def test_teardown_all_returns_success_count(self):
        """
        GWT:
        Given all processors teardown successfully,
        When teardown_all() is called,
        Then success_count equals total processors.
        """
        reg = IFRegistry()
        reg.clear()

        reg.register(TeardownTrackingProcessor("p1"), ["text/plain"])
        reg.register(TeardownTrackingProcessor("p2"), ["text/plain"])

        result = reg.teardown_all()

        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert result["failed_ids"] == []


class TestTeardownErrorIsolation:
    """Scenario 2: Teardown Error Isolation"""

    def test_teardown_continues_after_failure(self):
        """
        GWT:
        Given a processor whose teardown fails,
        When teardown_all() is called,
        Then other processors are still torn down.
        """
        reg = IFRegistry()
        reg.clear()

        proc_good1 = TeardownTrackingProcessor("good-1")
        proc_bad = FailingTeardownProcessor("bad")
        proc_good2 = TeardownTrackingProcessor("good-2")

        reg.register(proc_good1, ["text/plain"])
        reg.register(proc_bad, ["text/plain"])
        reg.register(proc_good2, ["text/plain"])

        result = reg.teardown_all()

        # All processors should have teardown called
        assert proc_good1.teardown_called
        assert proc_bad.teardown_called
        assert proc_good2.teardown_called

        # Summary should reflect mixed results
        assert result["success_count"] == 2
        assert result["failure_count"] == 1
        assert "bad" in result["failed_ids"]

    def test_teardown_returns_false_tracked(self):
        """
        GWT:
        Given a processor whose teardown returns False,
        When teardown_all() is called,
        Then it's counted as a failure.
        """
        reg = IFRegistry()
        reg.clear()

        proc_ok = TeardownTrackingProcessor("ok", teardown_result=True)
        proc_fail = TeardownTrackingProcessor("fail", teardown_result=False)

        reg.register(proc_ok, ["text/plain"])
        reg.register(proc_fail, ["text/plain"])

        result = reg.teardown_all()

        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert "fail" in result["failed_ids"]


class TestRegistryContextManager:
    """Scenario 3: Registry Context Manager"""

    def test_context_manager_calls_teardown_on_exit(self):
        """
        GWT:
        Given a registry used in a with statement,
        When the context exits normally,
        Then teardown_all() is automatically called.
        """
        reg = IFRegistry()
        reg.clear()

        proc = TeardownTrackingProcessor("ctx-proc")
        reg.register(proc, ["text/plain"])

        with reg:
            # Do some work
            pass

        assert proc.teardown_called

    def test_context_manager_calls_teardown_on_exception(self):
        """
        GWT:
        Given a registry used in a with statement,
        When the context exits via exception,
        Then teardown_all() is still called.
        """
        reg = IFRegistry()
        reg.clear()

        proc = TeardownTrackingProcessor("exc-proc")
        reg.register(proc, ["text/plain"])

        with pytest.raises(ValueError):
            with reg:
                raise ValueError("Something went wrong")

        assert proc.teardown_called

    def test_context_manager_returns_registry(self):
        """
        GWT:
        Given a registry,
        When entering context,
        Then the registry itself is returned.
        """
        reg = IFRegistry()
        reg.clear()

        with reg as r:
            assert r is reg


class TestTeardownSummary:
    """Scenario 4: Teardown Returns Summary"""

    def test_teardown_summary_structure(self):
        """
        GWT:
        Given multiple processors with mixed teardown results,
        When teardown_all() completes,
        Then a summary dict with all fields is returned.
        """
        reg = IFRegistry()
        reg.clear()

        reg.register(TeardownTrackingProcessor("ok-1"), ["text/plain"])
        reg.register(TeardownTrackingProcessor("ok-2"), ["text/plain"])
        reg.register(
            TeardownTrackingProcessor("fail-1", teardown_result=False), ["text/plain"]
        )
        reg.register(FailingTeardownProcessor("error-1"), ["text/plain"])

        result = reg.teardown_all()

        assert "success_count" in result
        assert "failure_count" in result
        assert "failed_ids" in result

        assert result["success_count"] == 2
        assert result["failure_count"] == 2
        assert set(result["failed_ids"]) == {"fail-1", "error-1"}


class TestIdempotentTeardown:
    """Scenario 5: Idempotent Teardown"""

    def test_multiple_teardown_calls_safe(self):
        """
        GWT:
        Given a processor already torn down,
        When teardown() is called again,
        Then it returns True without side effects.
        """
        reg = IFRegistry()
        reg.clear()

        proc = TeardownTrackingProcessor("idem-proc")
        reg.register(proc, ["text/plain"])

        # First teardown
        result1 = reg.teardown_all()
        assert result1["success_count"] == 1

        # Second teardown (should still work)
        result2 = reg.teardown_all()
        assert result2["success_count"] == 1

    def test_teardown_on_empty_registry(self):
        """
        GWT:
        Given an empty registry,
        When teardown_all() is called,
        Then it succeeds with zero counts.
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.teardown_all()

        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert result["failed_ids"] == []


# ============================================================================
# Shared Registry Safety Tests
# ============================================================================


class TestProcessAwareSingleton:
    """Scenario 1: Process-Aware Singleton"""

    def test_registry_tracks_process_id(self):
        """
        GWT:
        Given a fresh registry instance,
        When created,
        Then it tracks the current process ID.
        """
        import os

        reg = IFRegistry()
        reg.clear()

        assert reg.get_process_id() == os.getpid()

    def test_registry_is_healthy_after_creation(self):
        """
        GWT:
        Given a freshly created registry,
        When is_healthy() is called,
        Then it returns True.
        """
        reg = IFRegistry()
        reg.clear()

        assert reg.is_healthy() is True

    def test_assert_healthy_passes_for_valid_registry(self):
        """
        GWT:
        Given a healthy registry,
        When assert_healthy() is called,
        Then no exception is raised.
        """
        reg = IFRegistry()
        reg.clear()

        # Should not raise
        reg.assert_healthy()


class TestResetForWorker:
    """Scenario 2: Worker Reset"""

    def test_reset_for_worker_returns_fresh_registry(self):
        """
        GWT:
        Given a registry with existing processors,
        When reset_for_worker() is called,
        Then a fresh empty registry is returned.
        """
        reg = IFRegistry()
        reg.clear()

        # Register something
        proc = MockProcessor()
        reg.register(proc, ["text/plain"])
        assert len(reg.get_processors("text/plain")) == 1

        # Reset for worker
        new_reg = IFRegistry.reset_for_worker()

        # Should be empty
        assert len(new_reg.get_processors("text/plain")) == 0

    def test_reset_for_worker_is_healthy(self):
        """
        GWT:
        Given a call to reset_for_worker(),
        When it completes,
        Then the returned registry passes health check.
        """
        new_reg = IFRegistry.reset_for_worker()

        assert new_reg.is_healthy() is True
        new_reg.assert_healthy()  # Should not raise

    def test_reset_for_worker_tracks_current_pid(self):
        """
        GWT:
        Given a call to reset_for_worker(),
        When it completes,
        Then the registry tracks the current process ID.
        """
        import os

        new_reg = IFRegistry.reset_for_worker()

        assert new_reg.get_process_id() == os.getpid()


class TestResourceLock:
    """Scenario 3: Global Resource Lock"""

    def test_acquire_resource_lock_succeeds(self):
        """
        GWT:
        Given the global resource lock is available,
        When acquire_resource_lock() is called,
        Then it returns True.
        """
        from ingestforge.core.pipeline.registry import (
            acquire_resource_lock,
            release_resource_lock,
        )

        result = acquire_resource_lock(timeout=1.0)
        assert result is True

        # Clean up
        release_resource_lock()

    def test_release_resource_lock_is_safe(self):
        """
        GWT:
        Given the lock is not held,
        When release_resource_lock() is called,
        Then no exception is raised.
        """
        from ingestforge.core.pipeline.registry import release_resource_lock

        # Should not raise even if lock not held
        release_resource_lock()

    def test_lock_can_be_reacquired(self):
        """
        GWT:
        Given a lock that was acquired and released,
        When acquire is called again,
        Then it succeeds.
        """
        from ingestforge.core.pipeline.registry import (
            acquire_resource_lock,
            release_resource_lock,
        )

        # First acquire/release
        assert acquire_resource_lock(timeout=1.0) is True
        release_resource_lock()

        # Second acquire/release
        assert acquire_resource_lock(timeout=1.0) is True
        release_resource_lock()


class TestHealthCheckAssertions:
    """Scenario 4: JPL Rule #5 Health Assertions"""

    def test_assert_healthy_checks_initialization(self):
        """
        GWT:
        Given the registry._initialized flag is False,
        When assert_healthy() is called,
        Then AssertionError is raised.
        """
        reg = IFRegistry()
        reg.clear()

        # Force uninitialized state
        original = IFRegistry._initialized
        IFRegistry._initialized = False

        try:
            with pytest.raises(AssertionError) as exc_info:
                reg.assert_healthy()
            assert "not initialized" in str(exc_info.value)
        finally:
            IFRegistry._initialized = original

    def test_is_healthy_returns_false_when_uninitialized(self):
        """
        GWT:
        Given the registry is not initialized,
        When is_healthy() is called,
        Then it returns False.
        """
        reg = IFRegistry()
        reg.clear()

        # Force uninitialized state
        original = IFRegistry._initialized
        IFRegistry._initialized = False

        try:
            assert reg.is_healthy() is False
        finally:
            IFRegistry._initialized = original

    def test_clear_maintains_initialized_state(self):
        """
        GWT:
        Given a registry that is initialized,
        When clear() is called,
        Then the registry remains initialized.
        """
        reg = IFRegistry()
        reg.clear()

        # Should still be healthy after clear
        assert reg.is_healthy() is True
