"""
Tests for Background Healer Worker.

Background Healer Worker - Autonomous artifact repair.
Tests follow GWT (Given-When-Then) pattern and NASA JPL Power of Ten rules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from ingestforge.core.maintenance.healer import (
    BackgroundHealer,
    HealingConfig,
    HealingResult,
    HealingSessionResult,
    StaleArtifact,
    StaleReason,
    compute_content_hash,
    heal_artifact,
    scan_for_stale_artifacts,
    verify_hash_write,
    MAX_HEAL_BATCH_SIZE,
    MAX_SCAN_RESULTS,
    DEFAULT_MODEL_VERSION,
)


# =============================================================================
# MOCK CHUNK FOR TESTING
# =============================================================================


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    chunk_id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_hash_empty_content(self):
        """
        GWT:
        Given empty content
        When compute_content_hash is called
        Then empty string is returned.
        """
        result = compute_content_hash("")
        assert result == ""

    def test_hash_valid_content(self):
        """
        GWT:
        Given valid content
        When compute_content_hash is called
        Then SHA-256 hash is returned.
        """
        result = compute_content_hash("test content")
        assert len(result) == 64  # SHA-256 hex digest length
        assert result.isalnum()

    def test_hash_deterministic(self):
        """
        GWT:
        Given same content
        When compute_content_hash is called multiple times
        Then same hash is returned.
        """
        content = "reproducible content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_hash_different_content(self):
        """
        GWT:
        Given different content
        When compute_content_hash is called
        Then different hashes are returned.
        """
        hash1 = compute_content_hash("content A")
        hash2 = compute_content_hash("content B")
        assert hash1 != hash2


class TestVerifyHashWrite:
    """Tests for verify_hash_write function."""

    def test_verify_success(self):
        """
        GWT:
        Given chunk with matching hash
        When verify_hash_write is called
        Then True is returned.
        """
        mock_storage = MagicMock()
        mock_chunk = MockChunk(
            chunk_id="test-1",
            document_id="doc-1",
            content="test",
            metadata={"content_hash": "abc123"},
        )
        mock_storage.get_chunk.return_value = mock_chunk

        result = verify_hash_write(mock_storage, "test-1", "abc123")
        assert result is True

    def test_verify_chunk_not_found(self):
        """
        GWT:
        Given chunk not found in storage
        When verify_hash_write is called
        Then False is returned.
        """
        mock_storage = MagicMock()
        mock_storage.get_chunk.return_value = None

        result = verify_hash_write(mock_storage, "missing-1", "abc123")
        assert result is False

    def test_verify_hash_mismatch(self):
        """
        GWT:
        Given chunk with different hash
        When verify_hash_write is called
        Then False is returned (JPL Rule #7).
        """
        mock_storage = MagicMock()
        mock_chunk = MockChunk(
            chunk_id="test-1",
            document_id="doc-1",
            content="test",
            metadata={"content_hash": "different_hash"},
        )
        mock_storage.get_chunk.return_value = mock_chunk

        result = verify_hash_write(mock_storage, "test-1", "expected_hash")
        assert result is False


# =============================================================================
# STALE ARTIFACT TESTS
# =============================================================================


class TestStaleArtifact:
    """Tests for StaleArtifact dataclass."""

    def test_stale_artifact_creation(self):
        """
        GWT:
        Given valid parameters
        When StaleArtifact is created
        Then all fields are set correctly.
        """
        stale = StaleArtifact(
            chunk_id="chunk-1",
            document_id="doc-1",
            reason=StaleReason.MODEL_VERSION_MISMATCH,
            current_version="0.9.0",
            expected_version="1.0.0",
        )

        assert stale.chunk_id == "chunk-1"
        assert stale.reason == StaleReason.MODEL_VERSION_MISMATCH
        assert stale.current_version == "0.9.0"


class TestStaleReason:
    """Tests for StaleReason enum."""

    def test_all_reasons_defined(self):
        """
        GWT:
        Given StaleReason enum
        When checking members
        Then all expected reasons exist.
        """
        assert StaleReason.MODEL_VERSION_MISMATCH.value == "model_version_mismatch"
        assert StaleReason.MISSING_EMBEDDING.value == "missing_embedding"
        assert StaleReason.CORRUPTED_CONTENT.value == "corrupted_content"
        assert StaleReason.HASH_MISMATCH.value == "hash_mismatch"


# =============================================================================
# HEALING CONFIG TESTS
# =============================================================================


class TestHealingConfig:
    """Tests for HealingConfig dataclass."""

    def test_default_config(self):
        """
        GWT:
        Given no parameters
        When HealingConfig is created
        Then defaults are set.
        """
        config = HealingConfig()

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.model_version == DEFAULT_MODEL_VERSION
        assert config.batch_size <= MAX_HEAL_BATCH_SIZE
        assert config.dry_run is False

    def test_batch_size_bounded(self):
        """
        GWT:
        Given batch_size exceeding MAX_HEAL_BATCH_SIZE (JPL Rule #2)
        When HealingConfig is created
        Then batch_size is capped.
        """
        config = HealingConfig(batch_size=9999)

        assert config.batch_size == MAX_HEAL_BATCH_SIZE

    def test_dry_run_mode(self):
        """
        GWT:
        Given dry_run=True
        When HealingConfig is created
        Then dry_run is set.
        """
        config = HealingConfig(dry_run=True)

        assert config.dry_run is True


# =============================================================================
# HEALING RESULT TESTS
# =============================================================================


class TestHealingResult:
    """Tests for HealingResult dataclass."""

    def test_healing_result_success(self):
        """
        GWT:
        Given successful healing
        When HealingResult is created
        Then hash_verified is True.
        """
        result = HealingResult(
            chunk_id="chunk-1",
            success=True,
            action_taken="re_embed",
            old_hash="old",
            new_hash="new",
        )

        assert result.success is True
        assert result.hash_verified is True

    def test_healing_result_failure(self):
        """
        GWT:
        Given failed healing
        When HealingResult is created
        Then hash_verified is False.
        """
        result = HealingResult(
            chunk_id="chunk-1",
            success=False,
            action_taken="re_embed",
            error_message="Storage error",
        )

        assert result.success is False
        assert result.hash_verified is False


# =============================================================================
# SCAN FUNCTION TESTS
# =============================================================================


class TestScanForStaleArtifacts:
    """Tests for scan_for_stale_artifacts function."""

    def test_scan_empty_storage(self):
        """
        GWT:
        Given empty storage
        When scan_for_stale_artifacts is called
        Then empty list is returned.
        """
        mock_storage = MagicMock()
        mock_storage.get_document_ids.return_value = []

        config = HealingConfig()
        result = scan_for_stale_artifacts(mock_storage, config)

        assert result == []

    def test_scan_finds_missing_embedding(self):
        """
        GWT:
        Given chunk with missing embedding
        When scan_for_stale_artifacts is called
        Then StaleArtifact is returned.
        """
        mock_storage = MagicMock()
        mock_storage.get_document_ids.return_value = ["doc-1"]
        mock_storage.get_chunks_by_document.return_value = [
            MockChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="test",
                embedding=None,  # Missing embedding
                metadata={},
            )
        ]

        config = HealingConfig()
        result = scan_for_stale_artifacts(mock_storage, config)

        assert len(result) == 1
        assert result[0].reason == StaleReason.MISSING_EMBEDDING

    def test_scan_finds_version_mismatch(self):
        """
        GWT:
        Given chunk with old model version
        When scan_for_stale_artifacts is called
        Then StaleArtifact is returned.
        """
        mock_storage = MagicMock()
        mock_storage.get_document_ids.return_value = ["doc-1"]
        mock_storage.get_chunks_by_document.return_value = [
            MockChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="test",
                embedding=[0.1, 0.2],
                metadata={"embedding_model_version": "0.9.0"},
            )
        ]

        config = HealingConfig(model_version="1.0.0")
        result = scan_for_stale_artifacts(mock_storage, config)

        assert len(result) == 1
        assert result[0].reason == StaleReason.MODEL_VERSION_MISMATCH

    def test_scan_bounded_results(self):
        """
        GWT:
        Given more stale artifacts than MAX_SCAN_RESULTS (JPL Rule #2)
        When scan_for_stale_artifacts is called
        Then results are capped.
        """
        mock_storage = MagicMock()
        # Create many documents
        mock_storage.get_document_ids.return_value = [f"doc-{i}" for i in range(2000)]
        # Each with stale chunk
        mock_storage.get_chunks_by_document.return_value = [
            MockChunk(
                chunk_id="chunk", document_id="doc", content="test", embedding=None
            )
        ]

        config = HealingConfig()
        result = scan_for_stale_artifacts(mock_storage, config)

        assert len(result) <= MAX_SCAN_RESULTS


# =============================================================================
# HEAL ARTIFACT TESTS
# =============================================================================


class TestHealArtifact:
    """Tests for heal_artifact function."""

    def test_heal_dry_run(self):
        """
        GWT:
        Given dry_run mode
        When heal_artifact is called
        Then no changes are made.
        """
        mock_storage = MagicMock()
        stale = StaleArtifact(
            chunk_id="chunk-1",
            document_id="doc-1",
            reason=StaleReason.MISSING_EMBEDDING,
        )
        config = HealingConfig(dry_run=True)

        result = heal_artifact(mock_storage, stale, config)

        assert result.success is True
        assert result.action_taken == "dry_run"
        mock_storage.add_chunk.assert_not_called()

    def test_heal_chunk_not_found(self):
        """
        GWT:
        Given chunk not found in storage
        When heal_artifact is called
        Then failure result is returned (JPL Rule #7).
        """
        mock_storage = MagicMock()
        mock_storage.get_chunk.return_value = None

        stale = StaleArtifact(
            chunk_id="missing-1",
            document_id="doc-1",
            reason=StaleReason.MISSING_EMBEDDING,
        )
        config = HealingConfig()

        result = heal_artifact(mock_storage, stale, config)

        assert result.success is False
        assert "not found" in result.error_message.lower()

    def test_heal_missing_embedding_no_fn(self):
        """
        GWT:
        Given missing embedding but no embedding_fn
        When heal_artifact is called
        Then failure result is returned.
        """
        mock_storage = MagicMock()
        mock_storage.get_chunk.return_value = MockChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="test",
            embedding=None,
            metadata={},
        )

        stale = StaleArtifact(
            chunk_id="chunk-1",
            document_id="doc-1",
            reason=StaleReason.MISSING_EMBEDDING,
        )
        config = HealingConfig()

        result = heal_artifact(mock_storage, stale, config, embedding_fn=None)

        assert result.success is False
        assert "embedding" in result.error_message.lower()


# =============================================================================
# BACKGROUND HEALER TESTS
# =============================================================================


class TestBackgroundHealer:
    """Tests for BackgroundHealer class."""

    def test_healer_initialization(self):
        """
        GWT:
        Given storage and config
        When BackgroundHealer is created
        Then healer is initialized.
        """
        mock_storage = MagicMock()
        config = HealingConfig()

        healer = BackgroundHealer(mock_storage, config)

        assert healer._storage == mock_storage
        assert healer._config == config

    def test_run_session_empty_storage(self):
        """
        GWT:
        Given empty storage
        When run_session is called
        Then session completes with zero stale.
        """
        mock_storage = MagicMock()
        mock_storage.get_document_ids.return_value = []

        healer = BackgroundHealer(mock_storage)
        result = healer.run_session()

        assert result.stale_found == 0
        assert result.healed == 0
        assert result.success_rate == 100.0

    def test_heal_document(self):
        """
        GWT:
        Given document with stale chunks
        When heal_document is called
        Then only that document is healed.
        """
        mock_storage = MagicMock()
        mock_storage.get_chunks_by_document.return_value = [
            MockChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="test",
                embedding=[0.1],  # Has embedding
                metadata={},
            )
        ]

        healer = BackgroundHealer(mock_storage)
        result = healer.heal_document("doc-1")

        assert result.total_scanned == 1
        mock_storage.get_chunks_by_document.assert_called_with("doc-1")


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLCompliance:
    """Tests verifying NASA JPL Power of Ten rule compliance."""

    def test_jpl_rule_2_batch_size_bounded(self):
        """
        GWT:
        Given excessive batch_size
        When HealingConfig is created
        Then batch_size is capped (JPL Rule #2).
        """
        config = HealingConfig(batch_size=99999)
        assert config.batch_size <= MAX_HEAL_BATCH_SIZE

    def test_jpl_rule_2_scan_results_bounded(self):
        """
        GWT:
        Given many stale artifacts
        When scanning
        Then results are bounded (JPL Rule #2).
        """
        # Already tested in TestScanForStaleArtifacts
        pass

    def test_jpl_rule_7_verify_write(self):
        """
        GWT:
        Given hash write
        When verify_hash_write is called
        Then return value is checked (JPL Rule #7).
        """
        mock_storage = MagicMock()
        mock_storage.get_chunk.return_value = None

        # Function returns False (failure) rather than raising
        result = verify_hash_write(mock_storage, "test", "hash")
        assert result is False

    def test_jpl_rule_9_type_hints(self):
        """
        GWT:
        Given result dataclasses
        When inspecting fields
        Then all have type hints (JPL Rule #9).
        """
        assert "chunk_id" in HealingResult.__dataclass_fields__
        assert "success" in HealingResult.__dataclass_fields__
        assert "session_id" in HealingSessionResult.__dataclass_fields__


# =============================================================================
# SESSION RESULT TESTS
# =============================================================================


class TestHealingSessionResult:
    """Tests for HealingSessionResult dataclass."""

    def test_success_rate_calculation(self):
        """
        GWT:
        Given session with mixed results
        When success_rate is accessed
        Then correct percentage is returned.
        """
        result = HealingSessionResult(
            session_id="test",
            started_at="2026-01-01",
            completed_at="2026-01-01",
            healed=8,
            failed=2,
        )

        assert result.success_rate == 80.0

    def test_success_rate_no_attempts(self):
        """
        GWT:
        Given session with no attempts
        When success_rate is accessed
        Then 100.0 is returned.
        """
        result = HealingSessionResult(
            session_id="test",
            started_at="2026-01-01",
            completed_at="2026-01-01",
        )

        assert result.success_rate == 100.0
