"""
Tests for Extraction Health Monitoring ().

GWT (Given-When-Then) test structure.
NASA JPL Power of Ten compliance verification.
"""

import threading
import pytest

from ingestforge.core.pipeline.extraction_stats import (
    HealthMetric,
    StageStats,
    DocumentTypeStats,
    IFPipelineResult,
    IFExtractionStats,
    TelemetryInterceptor,
    get_extraction_stats,
    create_telemetry_interceptor,
    MAX_METRICS_PER_SESSION,
    MAX_DOCUMENT_TYPES,
    MAX_STAGES_TRACKED,
)
from ingestforge.core.pipeline.artifacts import IFTextArtifact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stats():
    """Create fresh IFExtractionStats instance for each test."""
    IFExtractionStats.reset_instance()
    collector = IFExtractionStats()
    collector.start_session()
    yield collector
    IFExtractionStats.reset_instance()


@pytest.fixture
def sample_metric():
    """Create a sample HealthMetric."""
    return HealthMetric(
        stage_name="extraction",
        document_id="doc-1",
        document_type="pdf",
        token_count=1500,
        latency_ms=250.5,
        success_status=True,
    )


@pytest.fixture
def sample_artifact():
    """Create a sample artifact for testing."""
    return IFTextArtifact(
        artifact_id="art-1",
        content="Test content",
        metadata={"mime_type": "application/pdf", "token_count": 100},
    )


# ---------------------------------------------------------------------------
# HealthMetric Tests
# ---------------------------------------------------------------------------


class TestHealthMetric:
    """Tests for HealthMetric model."""

    def test_health_metric_creation(self):
        """Given valid params, When creating HealthMetric, Then it stores all fields."""
        metric = HealthMetric(
            stage_name="extraction",
            document_id="doc-1",
            document_type="pdf",
            token_count=1000,
            latency_ms=150.0,
            success_status=True,
        )

        assert metric.stage_name == "extraction"
        assert metric.document_id == "doc-1"
        assert metric.document_type == "pdf"
        assert metric.token_count == 1000
        assert metric.latency_ms == 150.0
        assert metric.success_status is True
        assert metric.error_message is None

    def test_health_metric_with_error(self):
        """Given failure, When creating HealthMetric, Then it includes error message."""
        metric = HealthMetric(
            stage_name="extraction",
            document_id="doc-1",
            latency_ms=50.0,
            success_status=False,
            error_message="Connection timeout",
        )

        assert metric.success_status is False
        assert metric.error_message == "Connection timeout"

    def test_health_metric_to_dict(self, sample_metric):
        """Given HealthMetric, When calling to_dict, Then returns complete dict."""
        result = sample_metric.to_dict()

        assert result["stage_name"] == "extraction"
        assert result["document_id"] == "doc-1"
        assert result["document_type"] == "pdf"
        assert result["token_count"] == 1500
        assert result["latency_ms"] == 250.5
        assert result["success_status"] is True
        assert "timestamp" in result

    def test_health_metric_is_immutable(self, sample_metric):
        """Given HealthMetric, When trying to modify, Then raises error."""
        with pytest.raises(Exception):
            sample_metric.stage_name = "modified"

    def test_health_metric_defaults(self):
        """Given minimal params, When creating HealthMetric, Then uses defaults."""
        metric = HealthMetric(
            stage_name="test",
            document_id="doc-1",
        )

        assert metric.token_count == 0
        assert metric.latency_ms == 0.0
        assert metric.success_status is True
        assert metric.document_type == "unknown"
        assert metric.error_message is None
        assert metric.memory_mb is None

    def test_health_metric_with_memory(self):
        """Given memory usage, When creating HealthMetric, Then stores memory_mb."""
        metric = HealthMetric(
            stage_name="extraction",
            document_id="doc-1",
            latency_ms=100.0,
            success_status=True,
            memory_mb=512.5,
        )

        assert metric.memory_mb == 512.5


# ---------------------------------------------------------------------------
# StageStats Tests
# ---------------------------------------------------------------------------


class TestStageStats:
    """Tests for StageStats dataclass."""

    def test_stage_stats_success_rate(self):
        """Given mixed results, When calculating success_rate, Then returns percentage."""
        stats = StageStats(stage_name="test")
        stats.total_calls = 10
        stats.success_count = 8
        stats.failure_count = 2

        assert stats.success_rate == 80.0

    def test_stage_stats_success_rate_zero_calls(self):
        """Given no calls, When calculating success_rate, Then returns 100."""
        stats = StageStats(stage_name="test")

        assert stats.success_rate == 100.0

    def test_stage_stats_avg_latency(self):
        """Given latency data, When calculating avg_latency_ms, Then returns average."""
        stats = StageStats(stage_name="test")
        stats.success_count = 4
        stats.total_latency_ms = 400.0

        assert stats.avg_latency_ms == 100.0

    def test_stage_stats_avg_latency_zero_success(self):
        """Given no successes, When calculating avg_latency_ms, Then returns 0."""
        stats = StageStats(stage_name="test")
        stats.success_count = 0
        stats.total_latency_ms = 100.0

        assert stats.avg_latency_ms == 0.0


# ---------------------------------------------------------------------------
# DocumentTypeStats Tests
# ---------------------------------------------------------------------------


class TestDocumentTypeStats:
    """Tests for DocumentTypeStats dataclass."""

    def test_doc_type_stats_failure_rate(self):
        """Given failures, When calculating failure_rate, Then returns percentage."""
        stats = DocumentTypeStats(document_type="pdf")
        stats.total_processed = 20
        stats.failure_count = 5

        assert stats.failure_rate == 25.0
        assert stats.success_rate == 75.0

    def test_doc_type_stats_failure_rate_zero_processed(self):
        """Given no documents, When calculating failure_rate, Then returns 0."""
        stats = DocumentTypeStats(document_type="pdf")

        assert stats.failure_rate == 0.0
        assert stats.success_rate == 100.0

    def test_doc_type_stats_avg_latency(self):
        """Given latency data, When calculating avg_latency_ms, Then returns average."""
        stats = DocumentTypeStats(document_type="pdf")
        stats.total_processed = 5
        stats.total_latency_ms = 1000.0

        assert stats.avg_latency_ms == 200.0


# ---------------------------------------------------------------------------
# IFPipelineResult Tests
# ---------------------------------------------------------------------------


class TestIFPipelineResult:
    """Tests for IFPipelineResult model."""

    def test_pipeline_result_creation(self):
        """Given valid params, When creating IFPipelineResult, Then stores all fields."""
        result = IFPipelineResult(
            success=True,
            document_id="doc-1",
            total_duration_ms=5000.0,
            stages_completed=3,
            total_stages=3,
            total_tokens=5000,
            peak_memory_mb=256.0,
            success_rate=100.0,
            avg_latency_ms=500.0,
        )

        assert result.success is True
        assert result.document_id == "doc-1"
        assert result.total_duration_ms == 5000.0
        assert result.stages_completed == 3
        assert result.total_tokens == 5000
        assert result.peak_memory_mb == 256.0

    def test_pipeline_result_is_immutable(self):
        """Given IFPipelineResult, When trying to modify, Then raises error."""
        result = IFPipelineResult(
            success=True,
            document_id="doc-1",
        )

        with pytest.raises(Exception):
            result.success = False


# ---------------------------------------------------------------------------
# IFExtractionStats Tests
# ---------------------------------------------------------------------------


class TestIFExtractionStats:
    """Tests for IFExtractionStats collector."""

    def test_singleton_pattern(self):
        """Given two instances, When comparing, Then they are the same object."""
        IFExtractionStats.reset_instance()
        stats1 = IFExtractionStats()
        stats2 = IFExtractionStats()

        assert stats1 is stats2
        IFExtractionStats.reset_instance()

    def test_start_session(self, stats):
        """Given collector, When starting session, Then is_active returns True."""
        assert stats.is_active is True

    def test_end_session(self, stats):
        """Given active session, When ending, Then returns IFPipelineResult."""
        stats.emit(
            stage_name="test",
            document_id="doc-1",
            latency_ms=100.0,
            success_status=True,
        )

        result = stats.end_session()

        assert isinstance(result, IFPipelineResult)
        assert result.success is True
        assert stats.is_active is False

    def test_record_metric(self, stats, sample_metric):
        """Given active session, When recording metric, Then returns True."""
        result = stats.record_metric(sample_metric)

        assert result is True

    def test_record_metric_no_session(self):
        """Given no active session, When recording metric, Then returns False."""
        IFExtractionStats.reset_instance()
        collector = IFExtractionStats()
        metric = HealthMetric(
            stage_name="test",
            document_id="doc-1",
        )

        result = collector.record_metric(metric)

        assert result is False
        IFExtractionStats.reset_instance()

    def test_emit_convenience_method(self, stats):
        """Given active session, When using emit(), Then records metric."""
        result = stats.emit(
            stage_name="extraction",
            document_id="doc-1",
            latency_ms=150.0,
            success_status=True,
            token_count=1000,
            document_type="pdf",
        )

        assert result is True

        stage_stats = stats.get_stage_stats("extraction")
        assert stage_stats is not None
        assert stage_stats.total_calls == 1
        assert stage_stats.success_count == 1

    def test_get_stage_stats(self, stats):
        """Given recorded metrics, When getting stage stats, Then returns StageStats."""
        stats.emit("stage-1", "doc-1", 100.0, True, token_count=500)
        stats.emit("stage-1", "doc-2", 150.0, True, token_count=600)
        stats.emit("stage-1", "doc-3", 200.0, False)

        stage_stats = stats.get_stage_stats("stage-1")

        assert stage_stats is not None
        assert stage_stats.total_calls == 3
        assert stage_stats.success_count == 2
        assert stage_stats.failure_count == 1
        assert stage_stats.total_tokens == 1100

    def test_get_doc_type_stats(self, stats):
        """Given recorded metrics, When getting doc type stats, Then returns stats."""
        stats.emit("stage-1", "doc-1", 100.0, True, document_type="pdf")
        stats.emit("stage-1", "doc-2", 150.0, True, document_type="pdf")
        stats.emit("stage-1", "doc-3", 200.0, False, document_type="pdf")

        doc_stats = stats.get_doc_type_stats("pdf")

        assert doc_stats is not None
        assert doc_stats.total_processed == 3
        assert doc_stats.success_count == 2
        assert doc_stats.failure_count == 1
        assert doc_stats.failure_rate == pytest.approx(33.33, rel=0.01)

    def test_get_failure_rates(self, stats):
        """Given multiple doc types, When getting failure rates, Then returns dict."""
        stats.emit("stage-1", "doc-1", 100.0, True, document_type="pdf")
        stats.emit("stage-1", "doc-2", 100.0, False, document_type="pdf")
        stats.emit("stage-1", "doc-3", 100.0, True, document_type="html")
        stats.emit("stage-1", "doc-4", 100.0, True, document_type="html")

        rates = stats.get_failure_rates()

        assert rates["pdf"] == 50.0
        assert rates["html"] == 0.0

    def test_get_summary(self, stats):
        """Given recorded metrics, When getting summary, Then returns complete dict."""
        stats.emit("stage-1", "doc-1", 100.0, True, token_count=500)
        stats.emit("stage-2", "doc-1", 200.0, True, token_count=600)

        summary = stats.get_summary()

        assert summary["session_active"] is True
        assert summary["total_metrics"] == 2
        assert summary["total_calls"] == 2
        assert summary["success_count"] == 2
        assert summary["success_rate"] == 100.0
        assert summary["total_tokens"] == 1100
        assert summary["stages_tracked"] == 2

    def test_format_cli_summary(self, stats):
        """Given metrics, When formatting CLI summary, Then returns formatted string."""
        stats.emit("stage-1", "doc-1", 1200.0, True)
        stats.emit("stage-2", "doc-1", 1000.0, True)

        summary = stats.format_cli_summary()

        assert "100%" in summary
        assert "success" in summary
        assert "Latency" in summary


# ---------------------------------------------------------------------------
# JPL Power of Ten Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for NASA JPL Power of Ten compliance."""

    def test_jpl_rule_2_max_metrics(self, stats):
        """JPL Rule #2: Verify MAX_METRICS_PER_SESSION bound."""
        assert MAX_METRICS_PER_SESSION == 100000
        assert MAX_DOCUMENT_TYPES == 256
        assert MAX_STAGES_TRACKED == 64

    def test_jpl_rule_2_stage_limit(self, stats):
        """JPL Rule #2: Verify stage tracking has upper bound."""
        # Add more stages than the limit
        for i in range(MAX_STAGES_TRACKED + 10):
            stats.emit(f"stage-{i}", "doc-1", 100.0, True)

        all_stats = stats.get_all_stage_stats()
        assert len(all_stats) <= MAX_STAGES_TRACKED

    def test_jpl_rule_2_doc_type_limit(self, stats):
        """JPL Rule #2: Verify document type tracking has upper bound."""
        # Add more doc types than the limit
        for i in range(MAX_DOCUMENT_TYPES + 10):
            stats.emit("stage-1", f"doc-{i}", 100.0, True, document_type=f"type-{i}")

        all_stats = stats.get_all_doc_type_stats()
        assert len(all_stats) <= MAX_DOCUMENT_TYPES

    def test_jpl_rule_9_type_hints(self):
        """JPL Rule #9: Verify complete type hints on key methods."""
        import inspect

        # Check IFExtractionStats methods have return annotations
        for method_name in ["start_session", "end_session", "record_metric", "emit"]:
            method = getattr(IFExtractionStats, method_name)
            sig = inspect.signature(method)
            assert (
                sig.return_annotation != inspect.Parameter.empty
            ), f"{method_name} missing return type hint"


# ---------------------------------------------------------------------------
# TelemetryInterceptor Tests
# ---------------------------------------------------------------------------


class TestTelemetryInterceptor:
    """Tests for TelemetryInterceptor."""

    def test_interceptor_creation(self, stats):
        """Given stats collector, When creating interceptor, Then initializes correctly."""
        interceptor = TelemetryInterceptor(stats=stats)
        assert interceptor is not None

    def test_pre_stage_records_start_time(self, stats, sample_artifact):
        """Given interceptor, When pre_stage called, Then records start time."""
        interceptor = TelemetryInterceptor(stats=stats)

        interceptor.pre_stage("extraction", sample_artifact, "doc-1")

        # Start time should be recorded (internal state)
        assert "doc-1:extraction" in interceptor._stage_starts

    def test_post_stage_records_metric(self, stats, sample_artifact):
        """Given interceptor, When post_stage called, Then records success metric."""
        interceptor = TelemetryInterceptor(stats=stats)

        interceptor.pre_stage("extraction", sample_artifact, "doc-1")
        interceptor.post_stage("extraction", sample_artifact, "doc-1", 150.0)

        stage_stats = stats.get_stage_stats("extraction")
        assert stage_stats is not None
        assert stage_stats.success_count == 1

    def test_on_error_records_failure(self, stats, sample_artifact):
        """Given interceptor, When on_error called, Then records failure metric."""
        interceptor = TelemetryInterceptor(stats=stats)

        interceptor.pre_stage("extraction", sample_artifact, "doc-1")
        interceptor.on_error(
            "extraction", sample_artifact, "doc-1", RuntimeError("Test error")
        )

        stage_stats = stats.get_stage_stats("extraction")
        assert stage_stats is not None
        assert stage_stats.failure_count == 1

    def test_interceptor_extracts_doc_type_from_mime(self, stats, sample_artifact):
        """Given artifact with mime_type, When post_stage, Then extracts doc type."""
        interceptor = TelemetryInterceptor(stats=stats)

        interceptor.post_stage("extraction", sample_artifact, "doc-1", 100.0)

        doc_stats = stats.get_doc_type_stats("pdf")
        assert doc_stats is not None
        assert doc_stats.total_processed == 1

    def test_interceptor_extracts_token_count(self, stats, sample_artifact):
        """Given artifact with token_count, When post_stage, Then uses token count."""
        interceptor = TelemetryInterceptor(stats=stats)

        interceptor.post_stage("extraction", sample_artifact, "doc-1", 100.0)

        stage_stats = stats.get_stage_stats("extraction")
        assert stage_stats is not None
        assert stage_stats.total_tokens == 100  # From sample_artifact metadata


# ---------------------------------------------------------------------------
# GWT Behavioral Tests
# ---------------------------------------------------------------------------


class TestGWTBehavior:
    """
    Given-When-Then behavioral tests for .

    GWT:
    - Given: An active batch ingestion.
    - When: A processor finishes an extraction run.
    - Then: It must emit a HealthMetric containing token_count, latency_ms, and success_status.
    """

    def test_gwt_active_batch_emits_metric(self, stats):
        """
        Given: An active batch ingestion (session).
        When: A processor finishes an extraction run.
        Then: It must emit a HealthMetric with required fields.
        """
        # Given: Active session
        assert stats.is_active is True

        # When: Processor finishes extraction
        result = stats.emit(
            stage_name="processor-extraction",
            document_id="batch-doc-1",
            latency_ms=250.0,
            success_status=True,
            token_count=1500,
        )

        # Then: Metric is recorded with all required fields
        assert result is True
        summary = stats.get_summary()
        assert summary["total_metrics"] == 1
        assert summary["total_tokens"] == 1500

    def test_gwt_failure_rate_logged_per_doc_type(self, stats):
        """
        Given: Multiple document types being processed.
        When: Some extractions fail.
        Then: Failure rate is tracked per document type.
        """
        # Given & When: Mixed success/failure per document type
        stats.emit("stage", "d1", 100.0, True, document_type="pdf")
        stats.emit("stage", "d2", 100.0, False, document_type="pdf")
        stats.emit("stage", "d3", 100.0, True, document_type="html")
        stats.emit("stage", "d4", 100.0, True, document_type="html")
        stats.emit("stage", "d5", 100.0, True, document_type="html")

        # Then: Failure rates are calculated per type
        pdf_stats = stats.get_doc_type_stats("pdf")
        html_stats = stats.get_doc_type_stats("html")

        assert pdf_stats.failure_rate == 50.0
        assert html_stats.failure_rate == 0.0

    def test_gwt_metrics_exported_to_result(self, stats):
        """
        Given: Metrics recorded during session.
        When: Session ends.
        Then: Metrics are exported to IFPipelineResult.
        """
        # Given: Record metrics
        stats.emit("stage-1", "doc-1", 100.0, True, token_count=500)
        stats.emit("stage-2", "doc-1", 200.0, True, token_count=600)

        # When: Session ends
        result = stats.end_session()

        # Then: Result contains aggregated metrics
        assert result.total_tokens == 1100
        assert result.stages_completed == 2
        assert result.success_rate == 100.0


# ---------------------------------------------------------------------------
# Thread Safety Tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_metric_recording(self, stats):
        """Given multiple threads, When recording metrics, Then all are recorded."""
        recorded_count = [0]
        lock = threading.Lock()

        def record_metrics(thread_id: int):
            for i in range(50):
                result = stats.emit(
                    stage_name=f"stage-{thread_id}",
                    document_id=f"doc-{thread_id}-{i}",
                    latency_ms=100.0,
                    success_status=True,
                )
                if result:
                    with lock:
                        recorded_count[0] += 1

        threads = [threading.Thread(target=record_metrics, args=(i,)) for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 200 metrics should be recorded
        assert recorded_count[0] == 200


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_get_extraction_stats(self):
        """Given factory function, When called, Then returns singleton."""
        IFExtractionStats.reset_instance()
        stats = get_extraction_stats()

        assert isinstance(stats, IFExtractionStats)
        assert stats is IFExtractionStats()
        IFExtractionStats.reset_instance()

    def test_create_telemetry_interceptor(self, stats):
        """Given factory function, When called, Then returns interceptor."""
        interceptor = create_telemetry_interceptor(stats)

        assert isinstance(interceptor, TelemetryInterceptor)
        assert interceptor._stats is stats
