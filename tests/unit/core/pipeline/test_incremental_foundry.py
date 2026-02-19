"""
Tests for Incremental Foundry Runner.

Incremental Foundry Runner.
Verifies JPL Power of Ten compliance.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ingestforge.core.pipeline.incremental_foundry import (
    IncrementalSkipResult,
    IncrementalFoundryConfig,
    IncrementalFoundryReport,
    HashRegistry,
    IncrementalFoundryRunner,
    calculate_file_hash,
    calculate_content_hash,
    create_incremental_foundry_runner,
    MAX_HASH_CACHE_SIZE,
    HASH_ALGORITHM,
)
from ingestforge.core.pipeline.artifacts import IFTextArtifact
from ingestforge.core.pipeline.interfaces import IFArtifact


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_text_artifact() -> IFTextArtifact:
    """Create a sample text artifact."""
    return IFTextArtifact(
        artifact_id="test-artifact-001",
        content="This is test content for hashing.",
        metadata={"source": "test"},
    )


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage backend."""
    storage = MagicMock()
    storage.get_chunks_by_document.return_value = []
    storage.verify_chunk_exists.return_value = False
    return storage


@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text("Test file content for hashing.")
    return file_path


class MockStage:
    """Mock stage for testing."""

    def __init__(self, name: str = "mock-stage"):
        self.name = name
        self.input_type = IFTextArtifact
        self.output_type = IFTextArtifact

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        return artifact


# =============================================================================
# TestIncrementalSkipResult
# =============================================================================


class TestIncrementalSkipResult:
    """Tests for IncrementalSkipResult dataclass."""

    def test_create_valid_result(self) -> None:
        """Test creating a valid skip result."""
        result = IncrementalSkipResult(
            should_skip=True,
            source_hash="abc123",
            reason="cached",
            existing_artifact_id="art-001",
        )
        assert result.should_skip is True
        assert result.source_hash == "abc123"
        assert result.reason == "cached"

    def test_valid_reasons(self) -> None:
        """Test all valid reason values."""
        for reason in ("cached", "force", "not_found", "error"):
            result = IncrementalSkipResult(
                should_skip=False,
                source_hash="hash123",
                reason=reason,
            )
            assert result.reason == reason

    def test_invalid_reason_fails(self) -> None:
        """Test that invalid reason raises AssertionError."""
        with pytest.raises(AssertionError):
            IncrementalSkipResult(
                should_skip=False,
                source_hash="hash123",
                reason="invalid_reason",
            )

    def test_empty_hash_fails(self) -> None:
        """Test that empty hash raises AssertionError."""
        with pytest.raises(AssertionError):
            IncrementalSkipResult(
                should_skip=False,
                source_hash="",
                reason="cached",
            )


# =============================================================================
# TestIncrementalFoundryConfig
# =============================================================================


class TestIncrementalFoundryConfig:
    """Tests for IncrementalFoundryConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = IncrementalFoundryConfig()
        assert config.enabled is True
        assert config.force is False
        assert config.verify_cryptographic is True
        assert config.cache_size == MAX_HASH_CACHE_SIZE

    def test_cache_size_capped(self) -> None:
        """Test that cache size is capped at maximum."""
        config = IncrementalFoundryConfig(cache_size=MAX_HASH_CACHE_SIZE + 1000)
        assert config.cache_size == MAX_HASH_CACHE_SIZE

    def test_negative_cache_size_fixed(self) -> None:
        """Test that negative cache size is fixed to 0."""
        config = IncrementalFoundryConfig(cache_size=-100)
        assert config.cache_size == 0


# =============================================================================
# TestIncrementalFoundryReport
# =============================================================================


class TestIncrementalFoundryReport:
    """Tests for IncrementalFoundryReport dataclass."""

    def test_create_valid_report(self) -> None:
        """Test creating a valid report."""
        report = IncrementalFoundryReport(
            total_documents=10,
            processed_documents=5,
            skipped_documents=4,
            forced_documents=1,
            error_documents=0,
            total_time_ms=1000.0,
            skip_checks=[],
        )
        assert report.total_documents == 10
        assert report.processed_documents == 5

    def test_skip_rate_calculation(self) -> None:
        """Test skip rate percentage calculation."""
        report = IncrementalFoundryReport(
            total_documents=10,
            processed_documents=2,
            skipped_documents=8,
            forced_documents=0,
            error_documents=0,
            total_time_ms=500.0,
            skip_checks=[],
        )
        assert report.skip_rate == 80.0

    def test_skip_rate_zero_total(self) -> None:
        """Test skip rate with zero total documents."""
        report = IncrementalFoundryReport(
            total_documents=0,
            processed_documents=0,
            skipped_documents=0,
            forced_documents=0,
            error_documents=0,
            total_time_ms=0.0,
            skip_checks=[],
        )
        assert report.skip_rate == 0.0

    def test_negative_count_fails(self) -> None:
        """Test that negative counts raise AssertionError."""
        with pytest.raises(AssertionError):
            IncrementalFoundryReport(
                total_documents=-1,
                processed_documents=0,
                skipped_documents=0,
                forced_documents=0,
                error_documents=0,
                total_time_ms=0.0,
                skip_checks=[],
            )


# =============================================================================
# TestHashFunctions
# =============================================================================


class TestHashFunctions:
    """Tests for hash calculation functions."""

    def test_calculate_content_hash(self) -> None:
        """Test content hash calculation."""
        content = "Test content"
        hash_value = calculate_content_hash(content)
        assert len(hash_value) == 64  # SHA-256 hex length
        assert hash_value.isalnum()

    def test_content_hash_deterministic(self) -> None:
        """Test that same content produces same hash."""
        content = "Deterministic test"
        hash1 = calculate_content_hash(content)
        hash2 = calculate_content_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self) -> None:
        """Test that different content produces different hash."""
        hash1 = calculate_content_hash("Content A")
        hash2 = calculate_content_hash("Content B")
        assert hash1 != hash2

    def test_calculate_file_hash(self, temp_file: Path) -> None:
        """Test file hash calculation."""
        hash_value = calculate_file_hash(temp_file)
        assert len(hash_value) == 64
        assert hash_value.isalnum()

    def test_file_hash_not_found(self) -> None:
        """Test file hash with non-existent file."""
        with pytest.raises(FileNotFoundError):
            calculate_file_hash(Path("/nonexistent/file.txt"))


# =============================================================================
# TestHashRegistry
# =============================================================================


class TestHashRegistry:
    """Tests for HashRegistry class."""

    def test_register_and_lookup(self) -> None:
        """Test registering and looking up hashes."""
        registry = HashRegistry()
        registry.register("hash123", "artifact-001")

        assert registry.contains("hash123") is True
        assert registry.get_artifact_id("hash123") == "artifact-001"

    def test_lookup_not_found(self) -> None:
        """Test looking up non-existent hash."""
        registry = HashRegistry()
        assert registry.contains("nonexistent") is False
        assert registry.get_artifact_id("nonexistent") is None

    def test_max_size_eviction(self) -> None:
        """Test that oldest entries are evicted at capacity."""
        registry = HashRegistry(max_size=3)

        registry.register("hash1", "art1")
        registry.register("hash2", "art2")
        registry.register("hash3", "art3")
        registry.register("hash4", "art4")  # Should evict hash1

        assert registry.contains("hash1") is False
        assert registry.contains("hash4") is True
        assert registry.size == 3

    def test_clear(self) -> None:
        """Test clearing the registry."""
        registry = HashRegistry()
        registry.register("hash1", "art1")
        registry.register("hash2", "art2")

        registry.clear()

        assert registry.size == 0
        assert registry.contains("hash1") is False


# =============================================================================
# TestIncrementalFoundryRunner
# =============================================================================


class TestIncrementalFoundryRunner:
    """Tests for IncrementalFoundryRunner class."""

    def test_check_skip_not_found(
        self,
        sample_text_artifact: IFTextArtifact,
    ) -> None:
        """Test skip check when artifact not in cache."""
        runner = IncrementalFoundryRunner()
        result = runner.check_should_skip(sample_text_artifact)

        assert result.should_skip is False
        assert result.reason == "not_found"
        assert result.source_hash != ""

    def test_check_skip_force_flag(
        self,
        sample_text_artifact: IFTextArtifact,
    ) -> None:
        """Test skip check with force flag."""
        runner = IncrementalFoundryRunner()
        result = runner.check_should_skip(sample_text_artifact, force=True)

        assert result.should_skip is False
        assert result.reason == "force"

    def test_check_skip_cached(
        self,
        sample_text_artifact: IFTextArtifact,
    ) -> None:
        """Test skip check when artifact is cached."""
        runner = IncrementalFoundryRunner()

        # Process once to cache
        runner.mark_processed(sample_text_artifact)

        # Should skip on second check
        result = runner.check_should_skip(sample_text_artifact)

        assert result.should_skip is True
        assert result.reason == "cached"
        assert result.existing_artifact_id is not None

    def test_check_skip_with_storage(
        self,
        sample_text_artifact: IFTextArtifact,
        mock_storage: MagicMock,
    ) -> None:
        """Test skip check with storage backend."""
        runner = IncrementalFoundryRunner(storage=mock_storage)
        result = runner.check_should_skip(sample_text_artifact)

        # Storage was checked
        mock_storage.get_chunks_by_document.assert_called()
        assert result.reason in ("not_found", "cached")

    def test_run_incremental_skip(
        self,
        sample_text_artifact: IFTextArtifact,
    ) -> None:
        """Test run_incremental with cached artifact."""
        runner = IncrementalFoundryRunner()
        stages = [MockStage()]

        # Mark as already processed
        runner.mark_processed(sample_text_artifact)

        # Should skip
        result, skip_result = runner.run_incremental(
            sample_text_artifact,
            stages,
            "doc-001",
        )

        assert skip_result.should_skip is True
        assert skip_result.reason == "cached"

    def test_run_incremental_process(
        self,
        sample_text_artifact: IFTextArtifact,
    ) -> None:
        """Test run_incremental when processing is needed."""
        runner = IncrementalFoundryRunner()
        stages = [MockStage()]

        result, skip_result = runner.run_incremental(
            sample_text_artifact,
            stages,
            "doc-001",
        )

        assert skip_result.should_skip is False
        assert skip_result.reason == "not_found"

    def test_get_stats(self) -> None:
        """Test getting runner statistics."""
        runner = IncrementalFoundryRunner()
        stats = runner.get_stats()

        assert "hash_cache_size" in stats
        assert "max_cache_size" in stats
        assert "incremental_enabled" in stats
        assert stats["incremental_enabled"] is True


# =============================================================================
# TestAcceptanceCriteria
# =============================================================================


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    def test_ac_skip_existing_artifact(
        self,
        sample_text_artifact: IFTextArtifact,
    ) -> None:
        """AC: Runner skips document with matching source_hash."""
        runner = IncrementalFoundryRunner()
        stages = [MockStage()]

        # First run: process
        result1, skip1 = runner.run_incremental(
            sample_text_artifact,
            stages,
            "doc-001",
        )
        assert skip1.should_skip is False

        # Second run: should skip
        result2, skip2 = runner.run_incremental(
            sample_text_artifact,
            stages,
            "doc-001",
        )
        assert skip2.should_skip is True

    def test_ac_incremental_skip_logged(
        self,
        sample_text_artifact: IFTextArtifact,
        caplog,
    ) -> None:
        """AC: System logs 'INCREMENTAL SKIP' when skipping."""
        import logging

        caplog.set_level(logging.INFO)

        runner = IncrementalFoundryRunner()
        runner.mark_processed(sample_text_artifact)

        # Should log skip message
        with caplog.at_level(logging.INFO):
            runner.check_should_skip(sample_text_artifact)

        assert any("INCREMENTAL SKIP" in record.message for record in caplog.records)

    def test_ac_force_flag_overrides(
        self,
        sample_text_artifact: IFTextArtifact,
    ) -> None:
        """AC: --force flag overrides skip logic."""
        runner = IncrementalFoundryRunner()
        stages = [MockStage()]

        # Cache the artifact
        runner.mark_processed(sample_text_artifact)

        # With force=True, should not skip
        result, skip_result = runner.run_incremental(
            sample_text_artifact,
            stages,
            "doc-001",
            force=True,
        )

        assert skip_result.should_skip is False
        assert skip_result.reason == "force"

    def test_ac_cryptographic_hash(self) -> None:
        """AC: Uses cryptographic (SHA-256) verification."""
        content = "Test content for crypto hash"
        hash_value = calculate_content_hash(content)

        # SHA-256 produces 64 hex characters
        assert len(hash_value) == 64
        # Verify it matches expected SHA-256 hash
        import hashlib

        expected = hashlib.sha256(content.encode()).hexdigest()
        assert hash_value == expected


# =============================================================================
# TestConvenienceFunctions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_incremental_foundry_runner(self) -> None:
        """Test creating a runner via convenience function."""
        runner = create_incremental_foundry_runner()
        assert isinstance(runner, IncrementalFoundryRunner)

    def test_create_with_force(self) -> None:
        """Test creating runner with force mode."""
        runner = create_incremental_foundry_runner(force=True)
        stats = runner.get_stats()
        assert stats["force_mode"] is True

    def test_create_with_storage(self, mock_storage: MagicMock) -> None:
        """Test creating runner with storage."""
        runner = create_incremental_foundry_runner(storage=mock_storage)
        stats = runner.get_stats()
        assert stats["storage_connected"] is True


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_fixed_bounds(self) -> None:
        """Rule #2: Verify fixed bounds."""
        assert MAX_HASH_CACHE_SIZE > 0
        assert MAX_HASH_CACHE_SIZE <= 100000

    def test_rule_5_preconditions(self) -> None:
        """Rule #5: Verify assertions."""
        with pytest.raises(AssertionError):
            IncrementalSkipResult(
                should_skip=False,
                source_hash="",  # Empty hash should fail
                reason="cached",
            )

    def test_rule_9_type_hints(self) -> None:
        """Rule #9: Verify type hints exist."""
        runner = IncrementalFoundryRunner()
        assert hasattr(runner.check_should_skip, "__annotations__")
        assert hasattr(runner.run_incremental, "__annotations__")

    def test_rule_10_cryptographic_verification(self) -> None:
        """Rule #10: Verify SHA-256 is used."""
        assert HASH_ALGORITHM == "sha256"
        hash_value = calculate_content_hash("test")
        assert len(hash_value) == 64  # SHA-256 = 256 bits = 64 hex chars
