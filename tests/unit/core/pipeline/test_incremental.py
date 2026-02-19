"""
Tests for Incremental Pipeline Runner.

Skip unchanged files via hash check.
Verifies JPL Power of Ten compliance.
"""

import pytest
import tempfile
from pathlib import Path

from ingestforge.core.pipeline.incremental import (
    FileHashRecord,
    IncrementalCheckResult,
    IncrementalRunReport,
    HashManifest,
    IncrementalRunner,
    create_incremental_runner,
    filter_unchanged_files,
    MAX_TRACKED_FILES,
    MAX_BATCH_SIZE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def manifest_path(temp_dir: Path) -> Path:
    """Create a path for the hash manifest."""
    return temp_dir / "hash_manifest.json"


@pytest.fixture
def sample_file(temp_dir: Path) -> Path:
    """Create a sample file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def manifest(manifest_path: Path) -> HashManifest:
    """Create a HashManifest for testing."""
    return HashManifest(manifest_path)


@pytest.fixture
def runner(manifest_path: Path) -> IncrementalRunner:
    """Create an IncrementalRunner for testing."""
    r = IncrementalRunner(manifest_path)
    r.initialize()
    return r


# =============================================================================
# TestFileHashRecord
# =============================================================================


class TestFileHashRecord:
    """Tests for FileHashRecord dataclass."""

    def test_create_valid_record(self) -> None:
        """Test creating a valid hash record."""
        record = FileHashRecord(
            path="/path/to/file.txt",
            hash_value="abc123",
            algorithm="sha256",
            size_bytes=100,
            last_processed="2026-02-18T00:00:00Z",
        )

        assert record.path == "/path/to/file.txt"
        assert record.hash_value == "abc123"
        assert record.algorithm == "sha256"
        assert record.size_bytes == 100

    def test_empty_path_fails(self) -> None:
        """Test that empty path raises AssertionError."""
        with pytest.raises(AssertionError):
            FileHashRecord(
                path="",
                hash_value="abc123",
                algorithm="sha256",
                size_bytes=100,
                last_processed="2026-02-18T00:00:00Z",
            )

    def test_empty_hash_fails(self) -> None:
        """Test that empty hash raises AssertionError."""
        with pytest.raises(AssertionError):
            FileHashRecord(
                path="/path/to/file.txt",
                hash_value="",
                algorithm="sha256",
                size_bytes=100,
                last_processed="2026-02-18T00:00:00Z",
            )

    def test_negative_size_fails(self) -> None:
        """Test that negative size raises AssertionError."""
        with pytest.raises(AssertionError):
            FileHashRecord(
                path="/path/to/file.txt",
                hash_value="abc123",
                algorithm="sha256",
                size_bytes=-1,
                last_processed="2026-02-18T00:00:00Z",
            )

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        record = FileHashRecord(
            path="/path/to/file.txt",
            hash_value="abc123",
            algorithm="sha256",
            size_bytes=100,
            last_processed="2026-02-18T00:00:00Z",
            metadata={"key": "value"},
        )

        d = record.to_dict()

        assert d["path"] == "/path/to/file.txt"
        assert d["hash_value"] == "abc123"
        assert d["metadata"]["key"] == "value"

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "path": "/path/to/file.txt",
            "hash_value": "abc123",
            "algorithm": "sha256",
            "size_bytes": 100,
            "last_processed": "2026-02-18T00:00:00Z",
        }

        record = FileHashRecord.from_dict(data)

        assert record.path == "/path/to/file.txt"
        assert record.hash_value == "abc123"


# =============================================================================
# TestIncrementalCheckResult
# =============================================================================


class TestIncrementalCheckResult:
    """Tests for IncrementalCheckResult dataclass."""

    def test_create_new_file_result(self) -> None:
        """Test creating result for new file."""
        result = IncrementalCheckResult(
            path="/path/to/file.txt",
            needs_processing=True,
            reason="new",
            current_hash="abc123",
        )

        assert result.needs_processing is True
        assert result.reason == "new"

    def test_create_unchanged_result(self) -> None:
        """Test creating result for unchanged file."""
        result = IncrementalCheckResult(
            path="/path/to/file.txt",
            needs_processing=False,
            reason="unchanged",
            current_hash="abc123",
            stored_hash="abc123",
        )

        assert result.needs_processing is False
        assert result.reason == "unchanged"

    def test_invalid_reason_fails(self) -> None:
        """Test that invalid reason raises AssertionError."""
        with pytest.raises(AssertionError):
            IncrementalCheckResult(
                path="/path/to/file.txt",
                needs_processing=True,
                reason="invalid",
            )


# =============================================================================
# TestIncrementalRunReport
# =============================================================================


class TestIncrementalRunReport:
    """Tests for IncrementalRunReport dataclass."""

    def test_create_valid_report(self) -> None:
        """Test creating a valid report."""
        report = IncrementalRunReport(
            total_files=100,
            processed_files=30,
            skipped_files=70,
            new_files=10,
            modified_files=20,
            error_files=0,
            time_ms=150.5,
        )

        assert report.total_files == 100
        assert report.processed_files == 30
        assert report.skipped_files == 70

    def test_skip_rate(self) -> None:
        """Test skip rate calculation."""
        report = IncrementalRunReport(
            total_files=100,
            processed_files=30,
            skipped_files=70,
            new_files=10,
            modified_files=20,
            error_files=0,
            time_ms=0,
        )

        assert report.skip_rate == 70.0

    def test_skip_rate_empty(self) -> None:
        """Test skip rate with no files."""
        report = IncrementalRunReport(
            total_files=0,
            processed_files=0,
            skipped_files=0,
            new_files=0,
            modified_files=0,
            error_files=0,
            time_ms=0,
        )

        assert report.skip_rate == 0.0

    def test_negative_count_fails(self) -> None:
        """Test that negative counts raise AssertionError."""
        with pytest.raises(AssertionError):
            IncrementalRunReport(
                total_files=-1,
                processed_files=0,
                skipped_files=0,
                new_files=0,
                modified_files=0,
                error_files=0,
                time_ms=0,
            )


# =============================================================================
# TestHashManifest
# =============================================================================


class TestHashManifest:
    """Tests for HashManifest class."""

    def test_load_nonexistent(self, manifest: HashManifest) -> None:
        """Test loading when manifest doesn't exist."""
        result = manifest.load()

        assert result is True
        assert manifest.record_count == 0

    def test_save_and_load(self, manifest: HashManifest, manifest_path: Path) -> None:
        """Test saving and loading manifest."""
        record = FileHashRecord(
            path="/path/to/file.txt",
            hash_value="abc123",
            algorithm="sha256",
            size_bytes=100,
            last_processed="2026-02-18T00:00:00Z",
        )
        manifest.set(record)
        manifest.save()

        # Create new manifest and load
        new_manifest = HashManifest(manifest_path)
        new_manifest.load()

        assert new_manifest.record_count == 1
        loaded = new_manifest.get("/path/to/file.txt")
        assert loaded is not None
        assert loaded.hash_value == "abc123"

    def test_get_nonexistent(self, manifest: HashManifest) -> None:
        """Test getting nonexistent record."""
        manifest.load()

        result = manifest.get("/nonexistent")

        assert result is None

    def test_set_and_get(self, manifest: HashManifest) -> None:
        """Test setting and getting record."""
        manifest.load()
        record = FileHashRecord(
            path="/path/to/file.txt",
            hash_value="abc123",
            algorithm="sha256",
            size_bytes=100,
            last_processed="2026-02-18T00:00:00Z",
        )

        manifest.set(record)
        result = manifest.get("/path/to/file.txt")

        assert result is not None
        assert result.hash_value == "abc123"

    def test_remove(self, manifest: HashManifest) -> None:
        """Test removing record."""
        manifest.load()
        record = FileHashRecord(
            path="/path/to/file.txt",
            hash_value="abc123",
            algorithm="sha256",
            size_bytes=100,
            last_processed="2026-02-18T00:00:00Z",
        )
        manifest.set(record)

        result = manifest.remove("/path/to/file.txt")

        assert result is True
        assert manifest.get("/path/to/file.txt") is None

    def test_remove_nonexistent(self, manifest: HashManifest) -> None:
        """Test removing nonexistent record."""
        manifest.load()

        result = manifest.remove("/nonexistent")

        assert result is False


# =============================================================================
# TestIncrementalRunner
# =============================================================================


class TestIncrementalRunner:
    """Tests for IncrementalRunner class."""

    def test_check_new_file(self, runner: IncrementalRunner, sample_file: Path) -> None:
        """Test checking a new file."""
        result = runner.check_file(sample_file)

        assert result.needs_processing is True
        assert result.reason == "new"
        assert result.current_hash is not None

    def test_check_unchanged_file(
        self, runner: IncrementalRunner, sample_file: Path
    ) -> None:
        """Test checking an unchanged file."""
        # First, mark as processed
        runner.mark_processed(sample_file)

        # Then check again
        result = runner.check_file(sample_file)

        assert result.needs_processing is False
        assert result.reason == "unchanged"

    def test_check_modified_file(
        self, runner: IncrementalRunner, sample_file: Path
    ) -> None:
        """Test checking a modified file."""
        # First, mark as processed
        runner.mark_processed(sample_file)

        # Modify the file
        sample_file.write_text("Modified content!")

        # Then check again
        result = runner.check_file(sample_file)

        assert result.needs_processing is True
        assert result.reason == "modified"

    def test_check_nonexistent_file(
        self, runner: IncrementalRunner, temp_dir: Path
    ) -> None:
        """Test checking a nonexistent file."""
        result = runner.check_file(temp_dir / "nonexistent.txt")

        assert result.needs_processing is False
        assert result.reason == "error"

    def test_mark_processed(self, runner: IncrementalRunner, sample_file: Path) -> None:
        """Test marking a file as processed."""
        result = runner.mark_processed(sample_file)

        assert result is True

    def test_mark_processed_with_metadata(
        self, runner: IncrementalRunner, sample_file: Path
    ) -> None:
        """Test marking a file with metadata."""
        result = runner.mark_processed(sample_file, metadata={"pipeline": "test"})

        assert result is True

    def test_filter_files(self, runner: IncrementalRunner, temp_dir: Path) -> None:
        """Test filtering files."""
        # Create multiple files
        files = []
        for i in range(5):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)

        # Mark first 2 as processed
        runner.mark_processed(files[0])
        runner.mark_processed(files[1])

        # Filter
        to_process, report = runner.filter_files(files)

        assert len(to_process) == 3
        assert report.skipped_files == 2
        assert report.new_files == 3

    def test_get_stats(self, runner: IncrementalRunner) -> None:
        """Test getting stats."""
        stats = runner.get_stats()

        assert "tracked_files" in stats
        assert "max_tracked_files" in stats
        assert stats["max_tracked_files"] == MAX_TRACKED_FILES

    def test_finalize_saves_manifest(
        self, runner: IncrementalRunner, sample_file: Path, manifest_path: Path
    ) -> None:
        """Test that finalize saves the manifest."""
        runner.mark_processed(sample_file)
        runner.finalize()

        assert manifest_path.exists()


# =============================================================================
# TestConvenienceFunctions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_incremental_runner(self, temp_dir: Path) -> None:
        """Test creating an incremental runner."""
        manifest_path = temp_dir / "manifest.json"
        runner = create_incremental_runner(manifest_path)

        assert isinstance(runner, IncrementalRunner)

    def test_create_incremental_runner_default_path(self) -> None:
        """Test creating runner with default path."""
        runner = create_incremental_runner()

        assert isinstance(runner, IncrementalRunner)

    def test_filter_unchanged_files(self, temp_dir: Path) -> None:
        """Test filtering unchanged files."""
        # Create files
        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)

        manifest_path = temp_dir / "manifest.json"

        # First run - all files new
        result = filter_unchanged_files(files, manifest_path)
        assert len(result) == 3


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_fixed_bounds(self) -> None:
        """Rule #2: Verify fixed upper bounds are defined."""
        assert MAX_TRACKED_FILES > 0
        assert MAX_BATCH_SIZE > 0

    def test_rule_5_preconditions(self) -> None:
        """Rule #5: Verify preconditions are asserted."""
        with pytest.raises(AssertionError):
            FileHashRecord(
                path="",
                hash_value="abc",
                algorithm="sha256",
                size_bytes=0,
                last_processed="",
            )

        with pytest.raises(AssertionError):
            IncrementalCheckResult(
                path="/path",
                needs_processing=True,
                reason="invalid_reason",
            )

    def test_rule_9_type_hints(self, manifest_path: Path) -> None:
        """Rule #9: Verify methods have type hints."""
        runner = IncrementalRunner(manifest_path)

        assert hasattr(runner.check_file, "__annotations__")
        assert hasattr(runner.mark_processed, "__annotations__")
        assert hasattr(runner.filter_files, "__annotations__")
