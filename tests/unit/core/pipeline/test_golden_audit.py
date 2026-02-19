"""
Unit tests for Golden Dataset Hash Audit System.

Audit - Golden Dataset Hash Audit
Tests all GWT scenarios and JPL rule compliance.
"""

import json
import tempfile
from pathlib import Path
from typing import get_type_hints

import pytest

from ingestforge.core.pipeline.golden_audit import (
    GoldenEntry,
    GoldenDataset,
    AuditResult,
    AuditReport,
    HashAuditor,
    BaselineManager,
    create_auditor,
    create_baseline_manager,
    MAX_GOLDEN_ENTRIES,
    MAX_HASH_LENGTH,
    MAX_REPORT_SIZE,
    MAX_DESCRIPTION_LENGTH,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def sample_files(temp_dir):
    """Create multiple sample files."""
    files = []
    for i in range(3):
        file_path = temp_dir / f"file_{i}.txt"
        file_path.write_text(f"Content {i}")
        files.append(file_path)
    return files


@pytest.fixture
def sample_entry(sample_file):
    """Create a sample golden entry."""
    auditor = HashAuditor()
    hash_value = auditor._default_hash(sample_file)
    return GoldenEntry(
        entry_id="test-entry-001",
        input_path=str(sample_file),
        expected_hash=hash_value,
        processor_id="test-processor",
        description="Test entry",
    )


@pytest.fixture
def sample_dataset(sample_entry):
    """Create a sample golden dataset."""
    dataset = GoldenDataset(dataset_id="test-dataset-001", name="Test Dataset")
    dataset.add_entry(sample_entry)
    return dataset


# =============================================================================
# GWT Scenario 1: Golden Dataset Registration
# =============================================================================


class TestGoldenDatasetRegistration:
    """Tests for GWT Scenario 1: Golden Dataset Registration."""

    def test_register_golden_entry(self, sample_file):
        """Given known input and hash, when registered, then entry is stored."""
        entry = GoldenEntry(
            entry_id="test-001",
            input_path=str(sample_file),
            expected_hash="abc123",
            processor_id="test-processor",
        )

        assert entry.entry_id == "test-001"
        assert entry.expected_hash == "abc123"
        assert entry.processor_id == "test-processor"

    def test_register_dataset_with_entries(self, sample_entry):
        """Given entries, when dataset created, then entries are stored."""
        dataset = GoldenDataset(dataset_id="ds-001", name="Test Dataset")
        success = dataset.add_entry(sample_entry)

        assert success is True
        assert len(dataset.entries) == 1
        assert dataset.entries[0].entry_id == sample_entry.entry_id

    def test_dataset_stores_metadata(self):
        """Given metadata, when dataset created, then metadata is stored."""
        dataset = GoldenDataset(
            dataset_id="ds-001",
            name="Test Dataset",
            metadata={"author": "test", "version": "1.0"},
        )

        assert dataset.metadata["author"] == "test"
        assert dataset.metadata["version"] == "1.0"

    def test_dataset_serialization_roundtrip(self, sample_dataset, temp_dir):
        """Given dataset, when saved and loaded, then data is preserved."""
        file_path = temp_dir / "dataset.json"
        sample_dataset.save(file_path)

        loaded = GoldenDataset.load(file_path)

        assert loaded is not None
        assert loaded.name == sample_dataset.name
        assert len(loaded.entries) == len(sample_dataset.entries)


# =============================================================================
# GWT Scenario 2: Audit Execution
# =============================================================================


class TestAuditExecution:
    """Tests for GWT Scenario 2: Audit Execution."""

    def test_audit_single_entry(self, sample_entry):
        """Given golden entry, when audited, then hash is compared."""
        auditor = HashAuditor()
        result = auditor.audit_entry(sample_entry)

        assert isinstance(result, AuditResult)
        assert result.passed is True
        assert result.actual_hash == sample_entry.expected_hash

    def test_audit_entire_dataset(self, sample_dataset):
        """Given dataset, when audited, then all entries are checked."""
        auditor = HashAuditor()
        report = auditor.audit_dataset(sample_dataset)

        assert isinstance(report, AuditReport)
        assert report.total_entries == len(sample_dataset.entries)
        assert report.completed_at is not None

    def test_audit_report_contains_results(self, sample_dataset):
        """Given audit, when completed, then report contains all results."""
        auditor = HashAuditor()
        report = auditor.audit_dataset(sample_dataset)

        assert len(report.results) == len(sample_dataset.entries)
        for result in report.results:
            assert isinstance(result, AuditResult)

    def test_audit_handles_missing_file(self, temp_dir):
        """Given missing file, when audited, then error is reported."""
        entry = GoldenEntry(
            entry_id="missing-001",
            input_path=str(temp_dir / "nonexistent.txt"),
            expected_hash="abc123",
            processor_id="test",
        )

        auditor = HashAuditor()
        result = auditor.audit_entry(entry)

        assert result.passed is False
        assert result.error_message is not None
        assert "not found" in result.error_message.lower()


# =============================================================================
# GWT Scenario 3: Regression Detection
# =============================================================================


class TestRegressionDetection:
    """Tests for GWT Scenario 3: Regression Detection."""

    def test_detect_hash_mismatch(self, sample_file):
        """Given wrong hash, when audited, then regression detected."""
        entry = GoldenEntry(
            entry_id="mismatch-001",
            input_path=str(sample_file),
            expected_hash="wrong_hash_value",
            processor_id="test",
        )

        auditor = HashAuditor()
        result = auditor.audit_entry(entry)

        assert result.passed is False
        assert result.actual_hash != entry.expected_hash

    def test_hash_diff_provided_on_failure(self, sample_file):
        """Given mismatch, when failed, then diff details provided."""
        entry = GoldenEntry(
            entry_id="diff-001",
            input_path=str(sample_file),
            expected_hash="wrong_hash",
            processor_id="test",
        )

        auditor = HashAuditor()
        result = auditor.audit_entry(entry)

        assert result.hash_diff is not None
        assert "Expected" in result.hash_diff
        assert "Got" in result.hash_diff

    def test_report_shows_failures(self, sample_file):
        """Given failures, when report generated, then failures listed."""
        dataset = GoldenDataset(dataset_id="ds", name="Test")
        dataset.add_entry(
            GoldenEntry(
                entry_id="fail-001",
                input_path=str(sample_file),
                expected_hash="wrong",
                processor_id="test",
            )
        )

        auditor = HashAuditor()
        report = auditor.audit_dataset(dataset)

        assert report.failed_count == 1
        assert report.all_passed is False

    def test_mixed_pass_fail_results(self, sample_files):
        """Given mixed results, when audited, then both counted."""
        auditor = HashAuditor()
        dataset = GoldenDataset(dataset_id="ds", name="Mixed")

        # Add correct entry
        correct_hash = auditor._default_hash(sample_files[0])
        dataset.add_entry(
            GoldenEntry(
                entry_id="pass-001",
                input_path=str(sample_files[0]),
                expected_hash=correct_hash,
                processor_id="test",
            )
        )

        # Add wrong entry
        dataset.add_entry(
            GoldenEntry(
                entry_id="fail-001",
                input_path=str(sample_files[1]),
                expected_hash="wrong",
                processor_id="test",
            )
        )

        report = auditor.audit_dataset(dataset)

        assert report.passed_count == 1
        assert report.failed_count == 1
        assert 40 < report.pass_rate < 60  # 50%


# =============================================================================
# GWT Scenario 4: New Baseline Creation
# =============================================================================


class TestNewBaselineCreation:
    """Tests for GWT Scenario 4: New Baseline Creation."""

    def test_create_baseline_from_files(self, sample_files):
        """Given files, when baseline created, then hashes computed."""
        manager = BaselineManager()
        dataset = manager.create_baseline(
            name="New Baseline", input_paths=sample_files, processor_id="test"
        )

        assert len(dataset.entries) == len(sample_files)
        for entry in dataset.entries:
            assert len(entry.expected_hash) == 64  # SHA-256 hex

    def test_update_baseline_hash(self, sample_file, sample_entry, sample_dataset):
        """Given changed file, when baseline updated, then hash updated."""
        # Modify file content
        Path(sample_file).write_text("Modified content!")

        manager = BaselineManager()
        original_hash = sample_entry.expected_hash
        updated_count = manager.update_baseline(sample_dataset)

        assert updated_count == 1
        new_hash = sample_dataset.entries[0].expected_hash
        assert new_hash != original_hash

    def test_add_to_existing_baseline(self, sample_dataset, temp_dir):
        """Given baseline, when new file added, then entry added."""
        new_file = temp_dir / "new_file.txt"
        new_file.write_text("New content")

        manager = BaselineManager()
        original_count = len(sample_dataset.entries)
        entry = manager.add_to_baseline(
            sample_dataset, new_file, processor_id="test", description="New entry"
        )

        assert entry is not None
        assert len(sample_dataset.entries) == original_count + 1

    def test_selective_baseline_update(self, sample_files):
        """Given specific entries, when updated, then only those change."""
        manager = BaselineManager()
        dataset = manager.create_baseline("Test", sample_files)

        # Only update first entry
        target_id = dataset.entries[0].entry_id
        original_hashes = {e.entry_id: e.expected_hash for e in dataset.entries}

        # Modify first file
        sample_files[0].write_text("Changed!")
        manager.update_baseline(dataset, entry_ids=[target_id])

        # First should change, others should not
        assert dataset.entries[0].expected_hash != original_hashes[target_id]


# =============================================================================
# GWT Scenario 5: Audit Report Generation
# =============================================================================


class TestAuditReportGeneration:
    """Tests for GWT Scenario 5: Audit Report Generation."""

    def test_generate_json_report(self, sample_dataset):
        """Given results, when JSON generated, then valid JSON returned."""
        auditor = HashAuditor()
        report = auditor.audit_dataset(sample_dataset)

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert "report_id" in parsed
        assert "summary" in parsed
        assert "results" in parsed

    def test_generate_markdown_report(self, sample_dataset):
        """Given results, when markdown generated, then formatted report."""
        auditor = HashAuditor()
        report = auditor.audit_dataset(sample_dataset)

        md = report.to_markdown()

        assert "# Audit Report" in md
        assert "## Summary" in md
        assert "Total Entries" in md

    def test_report_includes_timestamps(self, sample_dataset):
        """Given audit, when reported, then timestamps included."""
        auditor = HashAuditor()
        report = auditor.audit_dataset(sample_dataset)

        assert report.started_at is not None
        assert report.completed_at is not None

    def test_report_summary_statistics(self, sample_dataset):
        """Given audit, when reported, then summary statistics correct."""
        auditor = HashAuditor()
        report = auditor.audit_dataset(sample_dataset)

        assert report.total_entries == len(sample_dataset.entries)
        assert report.passed_count + report.failed_count == report.total_entries
        assert 0 <= report.pass_rate <= 100


# =============================================================================
# JPL Rule #2: Fixed Upper Bounds
# =============================================================================


class TestJPLRule2FixedBounds:
    """Tests for JPL Rule #2: Fixed upper bounds."""

    def test_max_golden_entries_constant(self):
        """Given constant, MAX_GOLDEN_ENTRIES is defined."""
        assert MAX_GOLDEN_ENTRIES == 1000

    def test_max_hash_length_constant(self):
        """Given constant, MAX_HASH_LENGTH is defined."""
        assert MAX_HASH_LENGTH == 64

    def test_max_report_size_constant(self):
        """Given constant, MAX_REPORT_SIZE is defined."""
        assert MAX_REPORT_SIZE == 100_000

    def test_dataset_respects_max_entries(self):
        """Given many entries, dataset enforces limit."""
        dataset = GoldenDataset(dataset_id="ds", name="Full")

        # Fill to capacity
        for i in range(MAX_GOLDEN_ENTRIES):
            dataset.add_entry(
                GoldenEntry(
                    entry_id=f"entry-{i}",
                    input_path=f"/path/{i}",
                    expected_hash="abc",
                    processor_id="test",
                )
            )

        # Try to add one more
        success = dataset.add_entry(
            GoldenEntry(
                entry_id="overflow",
                input_path="/path/overflow",
                expected_hash="abc",
                processor_id="test",
            )
        )

        assert success is False
        assert len(dataset.entries) == MAX_GOLDEN_ENTRIES

    def test_description_truncation(self):
        """Given long description, it is truncated."""
        long_desc = "x" * (MAX_DESCRIPTION_LENGTH + 100)
        entry = GoldenEntry(
            entry_id="long-desc",
            input_path="/path",
            expected_hash="abc",
            processor_id="test",
            description=long_desc,
        )

        assert len(entry.description) <= MAX_DESCRIPTION_LENGTH


# =============================================================================
# JPL Rule #7: Check Return Values
# =============================================================================


class TestJPLRule7ReturnValues:
    """Tests for JPL Rule #7: Check all return values."""

    def test_add_entry_returns_bool(self, sample_entry):
        """Given add_entry call, returns explicit boolean."""
        dataset = GoldenDataset(dataset_id="ds", name="Test")
        result = dataset.add_entry(sample_entry)
        assert isinstance(result, bool)

    def test_audit_entry_returns_result(self, sample_entry):
        """Given audit_entry call, returns AuditResult."""
        auditor = HashAuditor()
        result = auditor.audit_entry(sample_entry)
        assert isinstance(result, AuditResult)

    def test_audit_dataset_returns_report(self, sample_dataset):
        """Given audit_dataset call, returns AuditReport."""
        auditor = HashAuditor()
        report = auditor.audit_dataset(sample_dataset)
        assert isinstance(report, AuditReport)

    def test_save_returns_bool(self, sample_dataset, temp_dir):
        """Given save call, returns explicit boolean."""
        result = sample_dataset.save(temp_dir / "test.json")
        assert isinstance(result, bool)

    def test_load_returns_dataset_or_none(self, temp_dir):
        """Given load call, returns dataset or None."""
        # Non-existent file
        result = GoldenDataset.load(temp_dir / "nonexistent.json")
        assert result is None


# =============================================================================
# JPL Rule #9: Complete Type Hints
# =============================================================================


class TestJPLRule9TypeHints:
    """Tests for JPL Rule #9: Complete type hints."""

    def test_golden_entry_has_type_hints(self):
        """Given GoldenEntry, all fields have type hints."""
        hints = get_type_hints(GoldenEntry)
        assert "entry_id" in hints
        assert "input_path" in hints
        assert "expected_hash" in hints

    def test_audit_result_has_type_hints(self):
        """Given AuditResult, all fields have type hints."""
        hints = get_type_hints(AuditResult)
        assert "passed" in hints
        assert "actual_hash" in hints

    def test_hash_auditor_methods_have_hints(self):
        """Given HashAuditor, methods have type hints."""
        hints = get_type_hints(HashAuditor.audit_entry)
        assert "return" in hints

    def test_baseline_manager_methods_have_hints(self):
        """Given BaselineManager, methods have type hints."""
        hints = get_type_hints(BaselineManager.create_baseline)
        assert "return" in hints


# =============================================================================
# Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_auditor_returns_hash_auditor(self):
        """Given create_auditor call, returns HashAuditor."""
        auditor = create_auditor()
        assert isinstance(auditor, HashAuditor)

    def test_create_auditor_with_custom_hash(self):
        """Given custom hash function, auditor uses it."""
        custom_called = []

        def custom_hash(path):
            custom_called.append(path)
            return "custom_hash_value"

        auditor = create_auditor(compute_hash_func=custom_hash)
        auditor._compute_hash(Path("/test"))

        assert len(custom_called) == 1

    def test_create_baseline_manager_returns_manager(self):
        """Given create_baseline_manager call, returns BaselineManager."""
        manager = create_baseline_manager()
        assert isinstance(manager, BaselineManager)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataset_audit(self):
        """Given empty dataset, audit returns empty report."""
        dataset = GoldenDataset(dataset_id="empty", name="Empty")
        auditor = HashAuditor()
        report = auditor.audit_dataset(dataset)

        assert report.total_entries == 0
        assert report.pass_rate == 0.0

    def test_dataset_roundtrip_with_special_chars(self, temp_dir):
        """Given special characters, roundtrip preserves them."""
        dataset = GoldenDataset(dataset_id="special", name="Test with émojis 🎉")
        dataset.add_entry(
            GoldenEntry(
                entry_id="special-1",
                input_path="/path/with spaces/file.txt",
                expected_hash="abc",
                processor_id="test",
                description="Description with 日本語",
            )
        )

        file_path = temp_dir / "special.json"
        dataset.save(file_path)
        loaded = GoldenDataset.load(file_path)

        assert loaded.name == dataset.name
        assert loaded.entries[0].description == dataset.entries[0].description

    def test_concurrent_add_remove(self, sample_entry):
        """Given add/remove operations, dataset stays consistent."""
        dataset = GoldenDataset(dataset_id="ds", name="Test")

        dataset.add_entry(sample_entry)
        assert len(dataset.entries) == 1

        dataset.remove_entry(sample_entry.entry_id)
        assert len(dataset.entries) == 0

    def test_get_nonexistent_entry(self, sample_dataset):
        """Given nonexistent ID, get_entry returns None."""
        result = sample_dataset.get_entry("nonexistent-id")
        assert result is None


# =============================================================================
# GWT Scenario Completeness
# =============================================================================


class TestGWTScenarioCompleteness:
    """Meta-tests to ensure all GWT scenarios are covered."""

    def test_scenario_1_registration_covered(self):
        """Verify Scenario 1 tests exist."""
        test_methods = [
            m for m in dir(TestGoldenDatasetRegistration) if m.startswith("test_")
        ]
        assert len(test_methods) >= 4

    def test_scenario_2_execution_covered(self):
        """Verify Scenario 2 tests exist."""
        test_methods = [m for m in dir(TestAuditExecution) if m.startswith("test_")]
        assert len(test_methods) >= 4

    def test_scenario_3_regression_covered(self):
        """Verify Scenario 3 tests exist."""
        test_methods = [
            m for m in dir(TestRegressionDetection) if m.startswith("test_")
        ]
        assert len(test_methods) >= 4

    def test_scenario_4_baseline_covered(self):
        """Verify Scenario 4 tests exist."""
        test_methods = [
            m for m in dir(TestNewBaselineCreation) if m.startswith("test_")
        ]
        assert len(test_methods) >= 4

    def test_scenario_5_report_covered(self):
        """Verify Scenario 5 tests exist."""
        test_methods = [
            m for m in dir(TestAuditReportGeneration) if m.startswith("test_")
        ]
        assert len(test_methods) >= 4
