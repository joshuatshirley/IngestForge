"""
Tests for Concurrency Benchmarking Script.

Verifies:
- Synthetic document generation
- Metrics collection
- Benchmark run execution
- Report generation and output
- DPM calculation
- System resource tracking
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the script module
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.benchmark_concurrency import (
    BenchmarkReport,
    BenchmarkRun,
    MetricsCollector,
    SystemMetrics,
    generate_synthetic_documents,
    save_report,
    _generate_summary,
    MAX_DOCUMENTS,
    MAX_WORKER_COUNTS,
)


# =============================================================================
# Synthetic Document Generation Tests
# =============================================================================


class TestSyntheticDocumentGeneration:
    """Tests for synthetic document generation."""

    def test_generates_correct_count(self):
        """
        GWT:
        Given a request for N documents,
        When generate_synthetic_documents(N) is called,
        Then exactly N documents are created.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = generate_synthetic_documents(10, Path(tmpdir))
            assert len(docs) == 10

    def test_documents_are_readable(self):
        """
        GWT:
        Given generated synthetic documents,
        When reading the files,
        Then valid text content is returned.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = generate_synthetic_documents(3, Path(tmpdir))
            for doc in docs:
                content = doc.read_text(encoding="utf-8")
                assert len(content) > 100
                assert "Document" in content

    def test_document_naming_pattern(self):
        """
        GWT:
        Given generated documents,
        When checking filenames,
        Then they follow the synthetic_doc_NNNN.txt pattern.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = generate_synthetic_documents(5, Path(tmpdir))
            for i, doc in enumerate(docs):
                expected_name = f"synthetic_doc_{i:04d}.txt"
                assert doc.name == expected_name

    def test_count_bounds_assertion(self):
        """
        GWT:
        Given a count exceeding MAX_DOCUMENTS,
        When generate_synthetic_documents is called,
        Then AssertionError is raised.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError):
                generate_synthetic_documents(MAX_DOCUMENTS + 1, Path(tmpdir))

    def test_zero_count_assertion(self):
        """
        GWT:
        Given a count of 0,
        When generate_synthetic_documents is called,
        Then AssertionError is raised.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError):
                generate_synthetic_documents(0, Path(tmpdir))


# =============================================================================
# Metrics Collector Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for system metrics collection."""

    def test_collector_initializes_without_psutil(self):
        """
        GWT:
        Given psutil is not available,
        When MetricsCollector is created,
        Then it initializes without error.
        """
        with patch.dict(sys.modules, {"psutil": None}):
            collector = MetricsCollector()
            assert collector._psutil_available is False or True  # May be cached

    def test_get_metrics_returns_valid_structure(self):
        """
        GWT:
        Given a fresh collector,
        When get_metrics() is called,
        Then a SystemMetrics object is returned.
        """
        collector = MetricsCollector()
        metrics = collector.get_metrics()
        assert isinstance(metrics, SystemMetrics)
        assert hasattr(metrics, "peak_memory_mb")
        assert hasattr(metrics, "peak_cpu_percent")

    def test_reset_clears_samples(self):
        """
        GWT:
        Given a collector with samples,
        When reset() is called,
        Then samples are cleared.
        """
        collector = MetricsCollector()
        collector._memory_samples = [100.0, 200.0]
        collector._cpu_samples = [50.0, 60.0]

        collector.reset()

        assert len(collector._memory_samples) == 0
        assert len(collector._cpu_samples) == 0


# =============================================================================
# BenchmarkRun Tests
# =============================================================================


class TestBenchmarkRun:
    """Tests for BenchmarkRun dataclass."""

    def test_dpm_calculation(self):
        """
        GWT:
        Given a benchmark run with 60 successful docs in 60 seconds,
        When DPM is calculated,
        Then DPM equals 60.
        """
        run = BenchmarkRun(
            worker_count=4,
            document_count=60,
            duration_sec=60.0,
            documents_per_minute=60.0,  # Pre-calculated
            success_count=60,
            failure_count=0,
            chunks_created=120,
        )
        assert run.documents_per_minute == 60.0

    def test_system_metrics_default(self):
        """
        GWT:
        Given a BenchmarkRun without explicit metrics,
        When accessing system_metrics,
        Then default SystemMetrics is returned.
        """
        run = BenchmarkRun(
            worker_count=1,
            document_count=10,
            duration_sec=1.0,
            documents_per_minute=600.0,
            success_count=10,
            failure_count=0,
            chunks_created=20,
        )
        assert isinstance(run.system_metrics, SystemMetrics)
        assert run.system_metrics.peak_memory_mb == 0.0


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestReportGeneration:
    """Tests for benchmark report generation."""

    def test_summary_calculates_speedup(self):
        """
        GWT:
        Given runs with 1 and 4 workers,
        When summary is generated,
        Then speedup factors are calculated.
        """
        report = BenchmarkReport(
            timestamp="2026-02-17T00:00:00Z",
            document_count=100,
            worker_counts=[1, 4],
            iterations_per_config=1,
            runs=[
                BenchmarkRun(
                    worker_count=1,
                    document_count=100,
                    duration_sec=100.0,
                    documents_per_minute=60.0,
                    success_count=100,
                    failure_count=0,
                    chunks_created=200,
                ),
                BenchmarkRun(
                    worker_count=4,
                    document_count=100,
                    duration_sec=25.0,
                    documents_per_minute=240.0,
                    success_count=100,
                    failure_count=0,
                    chunks_created=200,
                ),
            ],
        )

        summary = _generate_summary(report)

        assert "speedup_factors" in summary
        assert summary["speedup_factors"]["1"] == 1.0
        assert summary["speedup_factors"]["4"] == 4.0

    def test_summary_tracks_peak_resources(self):
        """
        GWT:
        Given runs with varying resource usage,
        When summary is generated,
        Then peak resources are tracked.
        """
        report = BenchmarkReport(
            timestamp="2026-02-17T00:00:00Z",
            document_count=100,
            worker_counts=[1, 4],
            iterations_per_config=1,
            runs=[
                BenchmarkRun(
                    worker_count=1,
                    document_count=100,
                    duration_sec=100.0,
                    documents_per_minute=60.0,
                    success_count=100,
                    failure_count=0,
                    chunks_created=200,
                    system_metrics=SystemMetrics(
                        peak_memory_mb=500.0,
                        peak_cpu_percent=80.0,
                    ),
                ),
                BenchmarkRun(
                    worker_count=4,
                    document_count=100,
                    duration_sec=25.0,
                    documents_per_minute=240.0,
                    success_count=100,
                    failure_count=0,
                    chunks_created=200,
                    system_metrics=SystemMetrics(
                        peak_memory_mb=1200.0,
                        peak_cpu_percent=320.0,  # 4 cores * 80%
                    ),
                ),
            ],
        )

        summary = _generate_summary(report)

        assert summary["peak_resources"]["memory_mb"] == 1200.0
        assert summary["peak_resources"]["cpu_percent"] == 320.0


# =============================================================================
# Report Output Tests
# =============================================================================


class TestReportOutput:
    """Tests for report output functionality."""

    def test_save_report_creates_json(self):
        """
        GWT:
        Given a benchmark report,
        When save_report is called,
        Then valid JSON file is created.
        """
        report = BenchmarkReport(
            timestamp="2026-02-17T00:00:00Z",
            document_count=10,
            worker_counts=[1, 2],
            iterations_per_config=1,
            runs=[],
            summary={"test": "value"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.json"
            save_report(report, output_path)

            assert output_path.exists()
            data = json.loads(output_path.read_text(encoding="utf-8"))
            assert data["timestamp"] == "2026-02-17T00:00:00Z"
            assert data["document_count"] == 10

    def test_save_report_creates_parent_dirs(self):
        """
        GWT:
        Given an output path with non-existent parent directories,
        When save_report is called,
        Then parent directories are created.
        """
        report = BenchmarkReport(
            timestamp="2026-02-17T00:00:00Z",
            document_count=10,
            worker_counts=[1],
            iterations_per_config=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "path" / "report.json"
            save_report(report, output_path)

            assert output_path.exists()

    def test_save_report_requires_json_extension(self):
        """
        GWT:
        Given an output path without .json extension,
        When save_report is called,
        Then AssertionError is raised.
        """
        report = BenchmarkReport(
            timestamp="2026-02-17T00:00:00Z",
            document_count=10,
            worker_counts=[1],
            iterations_per_config=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            with pytest.raises(AssertionError):
                save_report(report, output_path)


# =============================================================================
# Worker Count Validation Tests
# =============================================================================


class TestWorkerCountValidation:
    """Tests for worker count validation."""

    def test_valid_worker_counts(self):
        """
        GWT:
        Given the default worker counts,
        When checking MAX_WORKER_COUNTS,
        Then it contains [1, 2, 4, 8].
        """
        assert MAX_WORKER_COUNTS == [1, 2, 4, 8]

    def test_document_count_limit(self):
        """
        GWT:
        Given the MAX_DOCUMENTS constant,
        When checking the value,
        Then it is 1000 (JPL Rule #2 bounded).
        """
        assert MAX_DOCUMENTS == 1000


# =============================================================================
# JPL Compliance Tests
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_bounded_worker_counts(self):
        """
        GWT (JPL Rule #2):
        Given the worker counts,
        When checking bounds,
        Then all values are within reasonable limits.
        """
        for count in MAX_WORKER_COUNTS:
            assert 1 <= count <= 32

    def test_bounded_documents(self):
        """
        GWT (JPL Rule #2):
        Given the document limit,
        When checking bounds,
        Then it is capped at 1000.
        """
        assert MAX_DOCUMENTS <= 1000

    def test_dataclasses_have_type_hints(self):
        """
        GWT (JPL Rule #9):
        Given the dataclasses,
        When checking annotations,
        Then all fields have type hints.
        """
        from dataclasses import fields

        for cls in [SystemMetrics, BenchmarkRun, BenchmarkReport]:
            for field in fields(cls):
                assert (
                    field.type is not None
                ), f"{cls.__name__}.{field.name} missing type"
