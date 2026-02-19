"""
Background Healer Worker for IngestForge.

Autonomous process that identifies and fixes stale or corrupted artifacts.
Follows NASA JPL Power of Ten rules.

Features:
- Scans IFStorage for model-version mismatches
- Re-runs stale stages (e.g., re-embedding) during idle time
- Verifies write of new hash after healing
- JPL Rule #7: Checks return value of every database update
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_HEAL_BATCH_SIZE = 100  # Maximum artifacts to heal per batch
MAX_SCAN_RESULTS = 1000  # Maximum stale artifacts to return
MAX_HEALING_TIME_SEC = 300  # Maximum time per healing session (5 min)
DEFAULT_MODEL_VERSION = "1.0.0"  # Default model version for comparison
HASH_ALGORITHM = "sha256"  # Hash algorithm for content verification


# =============================================================================
# ENUMS AND RESULT TYPES
# =============================================================================


class StaleReason(Enum):
    """Reasons why an artifact is considered stale."""

    MODEL_VERSION_MISMATCH = "model_version_mismatch"
    MISSING_EMBEDDING = "missing_embedding"
    CORRUPTED_CONTENT = "corrupted_content"
    SCHEMA_OUTDATED = "schema_outdated"
    HASH_MISMATCH = "hash_mismatch"


@dataclass
class StaleArtifact:
    """
    Represents a stale artifact that needs healing.

    Rule #9: Complete type hints.
    """

    chunk_id: str
    document_id: str
    reason: StaleReason
    current_version: Optional[str] = None
    expected_version: Optional[str] = None
    details: str = ""


@dataclass
class HealingResult:
    """
    Result of a healing operation.

    Rule #9: Complete type hints.
    """

    chunk_id: str
    success: bool
    action_taken: str
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def hash_verified(self) -> bool:
        """Check if new hash was verified after healing."""
        return self.new_hash is not None and self.success


@dataclass
class HealingConfig:
    """
    Configuration for healing operations.

    Rule #2: Bounded parameters.
    Rule #9: Complete type hints.
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    model_version: str = DEFAULT_MODEL_VERSION
    batch_size: int = MAX_HEAL_BATCH_SIZE
    max_time_sec: float = MAX_HEALING_TIME_SEC
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate and bound configuration."""
        self.batch_size = min(self.batch_size, MAX_HEAL_BATCH_SIZE)
        self.max_time_sec = min(self.max_time_sec, MAX_HEALING_TIME_SEC)


@dataclass
class HealingSessionResult:
    """
    Aggregate result from a healing session.

    Rule #9: Complete type hints.
    """

    session_id: str
    started_at: str
    completed_at: str
    total_scanned: int = 0
    stale_found: int = 0
    healed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[HealingResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate healing success rate."""
        total_attempted = self.healed + self.failed
        if total_attempted == 0:
            return 100.0
        return (self.healed / total_attempted) * 100


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of content.

    Rule #4: < 60 lines.
    Rule #7: Check return value.

    Args:
        content: Text content to hash.

    Returns:
        Hex digest of hash.
    """
    if not content:
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_hash_write(storage: Any, chunk_id: str, expected_hash: str) -> bool:
    """
    Verify that hash was written correctly to storage.

    AC: Verified write of new hash after healing.
    Rule #7: Check return value of database read.

    Args:
        storage: Storage backend.
        chunk_id: Chunk identifier.
        expected_hash: Expected hash value.

    Returns:
        True if hash matches, False otherwise.
    """
    try:
        chunk = storage.get_chunk(chunk_id)
        if chunk is None:
            logger.error(f"Chunk {chunk_id} not found after healing")
            return False

        stored_hash = chunk.metadata.get("content_hash", "")
        if stored_hash != expected_hash:
            logger.error(
                f"Hash mismatch for {chunk_id}: expected {expected_hash[:16]}..., "
                f"got {stored_hash[:16]}..."
            )
            return False

        return True
    except Exception as e:
        logger.error(f"Hash verification failed for {chunk_id}: {e}")
        return False


# =============================================================================
# SCANNING FUNCTIONS
# =============================================================================


def scan_for_stale_artifacts(
    storage: Any,
    config: HealingConfig,
) -> List[StaleArtifact]:
    """
    Scan storage for stale or corrupted artifacts.

    AC: Scans IFStorage for model-version mismatches.
    Rule #2: Bounded results.
    Rule #4: < 60 lines.

    Args:
        storage: Storage backend implementing ChunkRepository.
        config: Healing configuration.

    Returns:
        List of stale artifacts (bounded by MAX_SCAN_RESULTS).
    """
    stale_artifacts: List[StaleArtifact] = []

    try:
        # Get all document IDs
        doc_ids = storage.get_document_ids()
        logger.info(f"Scanning {len(doc_ids)} documents for stale artifacts")

        for doc_id in doc_ids:
            if len(stale_artifacts) >= MAX_SCAN_RESULTS:
                logger.warning(f"Scan limit reached ({MAX_SCAN_RESULTS})")
                break

            chunks = storage.get_chunks_by_document(doc_id)
            for chunk in chunks:
                if len(stale_artifacts) >= MAX_SCAN_RESULTS:
                    break

                stale = _check_chunk_staleness(chunk, config)
                if stale:
                    stale_artifacts.append(stale)

    except Exception as e:
        logger.error(f"Scan failed: {e}")

    logger.info(f"Found {len(stale_artifacts)} stale artifacts")
    return stale_artifacts


def _check_chunk_staleness(
    chunk: Any, config: HealingConfig
) -> Optional[StaleArtifact]:
    """
    Check if a single chunk is stale.

    Rule #4: < 60 lines.

    Args:
        chunk: ChunkRecord to check.
        config: Healing configuration.

    Returns:
        StaleArtifact if stale, None otherwise.
    """
    chunk_id = chunk.chunk_id
    doc_id = chunk.document_id
    metadata = chunk.metadata or {}

    # Check 1: Model version mismatch
    stored_version = metadata.get("embedding_model_version", "")
    if stored_version and stored_version != config.model_version:
        return StaleArtifact(
            chunk_id=chunk_id,
            document_id=doc_id,
            reason=StaleReason.MODEL_VERSION_MISMATCH,
            current_version=stored_version,
            expected_version=config.model_version,
            details=f"Model version {stored_version} != {config.model_version}",
        )

    # Check 2: Missing embedding
    if chunk.embedding is None or len(chunk.embedding) == 0:
        return StaleArtifact(
            chunk_id=chunk_id,
            document_id=doc_id,
            reason=StaleReason.MISSING_EMBEDDING,
            details="No embedding vector found",
        )

    # Check 3: Content hash mismatch
    stored_hash = metadata.get("content_hash", "")
    if stored_hash:
        computed_hash = compute_content_hash(chunk.content)
        if computed_hash != stored_hash:
            return StaleArtifact(
                chunk_id=chunk_id,
                document_id=doc_id,
                reason=StaleReason.HASH_MISMATCH,
                details="Content hash mismatch",
            )

    return None


# =============================================================================
# HEALING FUNCTIONS
# =============================================================================


def heal_artifact(
    storage: Any,
    stale: StaleArtifact,
    config: HealingConfig,
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
) -> HealingResult:
    """
    Heal a single stale artifact.

    AC: Re-runs stale stages, verifies hash after healing.
    Rule #4: < 60 lines.
    Rule #7: Check return value of every database update.

    Args:
        storage: Storage backend.
        stale: Stale artifact to heal.
        config: Healing configuration.
        embedding_fn: Optional function to generate embeddings.

    Returns:
        HealingResult with outcome.
    """
    start_time = time.perf_counter()
    chunk_id = stale.chunk_id

    if config.dry_run:
        return HealingResult(
            chunk_id=chunk_id,
            success=True,
            action_taken="dry_run",
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    try:
        # Fetch current chunk
        chunk = storage.get_chunk(chunk_id)
        if chunk is None:
            return HealingResult(
                chunk_id=chunk_id,
                success=False,
                action_taken="fetch_failed",
                error_message="Chunk not found",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Determine healing action based on reason
        result = _execute_healing(storage, chunk, stale, config, embedding_fn)
        result.duration_ms = (time.perf_counter() - start_time) * 1000

        return result

    except Exception as e:
        logger.error(f"Healing failed for {chunk_id}: {e}")
        return HealingResult(
            chunk_id=chunk_id,
            success=False,
            action_taken="exception",
            error_message=str(e),
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )


def _execute_healing(
    storage: Any,
    chunk: Any,
    stale: StaleArtifact,
    config: HealingConfig,
    embedding_fn: Optional[Callable[[str], List[float]]],
) -> HealingResult:
    """
    Execute the healing action for a stale chunk.

    Rule #4: < 60 lines.
    Rule #7: Check return values.

    Args:
        storage: Storage backend.
        chunk: Current chunk data.
        stale: Stale artifact info.
        config: Healing configuration.
        embedding_fn: Optional embedding function.

    Returns:
        HealingResult with outcome.
    """
    chunk_id = stale.chunk_id
    old_hash = chunk.metadata.get("content_hash", "") if chunk.metadata else ""

    if stale.reason == StaleReason.MISSING_EMBEDDING:
        return _heal_missing_embedding(storage, chunk, config, embedding_fn, old_hash)

    elif stale.reason == StaleReason.MODEL_VERSION_MISMATCH:
        return _heal_version_mismatch(storage, chunk, config, embedding_fn, old_hash)

    elif stale.reason == StaleReason.HASH_MISMATCH:
        return _heal_hash_mismatch(storage, chunk, config, old_hash)

    else:
        return HealingResult(
            chunk_id=chunk_id,
            success=False,
            action_taken="unsupported_reason",
            error_message=f"No healing action for {stale.reason.value}",
        )


def _heal_missing_embedding(
    storage: Any,
    chunk: Any,
    config: HealingConfig,
    embedding_fn: Optional[Callable[[str], List[float]]],
    old_hash: str,
) -> HealingResult:
    """Heal a chunk with missing embedding."""
    chunk_id = chunk.chunk_id

    if embedding_fn is None:
        return HealingResult(
            chunk_id=chunk_id,
            success=False,
            action_taken="no_embedding_fn",
            error_message="No embedding function provided",
        )

    try:
        # Generate new embedding
        new_embedding = embedding_fn(chunk.content)

        # Update chunk
        chunk.embedding = new_embedding
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["embedding_model"] = config.embedding_model
        chunk.metadata["embedding_model_version"] = config.model_version
        chunk.metadata["healed_at"] = datetime.now(timezone.utc).isoformat()

        # Compute and store new hash
        new_hash = compute_content_hash(chunk.content)
        chunk.metadata["content_hash"] = new_hash

        # Write to storage (JPL Rule #7: check return value)
        update_success = storage.add_chunk(chunk)
        if not update_success:
            return HealingResult(
                chunk_id=chunk_id,
                success=False,
                action_taken="re_embed",
                error_message="Storage update failed",
                old_hash=old_hash,
            )

        # Verify hash write (AC)
        if not verify_hash_write(storage, chunk_id, new_hash):
            return HealingResult(
                chunk_id=chunk_id,
                success=False,
                action_taken="re_embed",
                error_message="Hash verification failed",
                old_hash=old_hash,
                new_hash=new_hash,
            )

        return HealingResult(
            chunk_id=chunk_id,
            success=True,
            action_taken="re_embed",
            old_hash=old_hash,
            new_hash=new_hash,
        )

    except Exception as e:
        return HealingResult(
            chunk_id=chunk_id,
            success=False,
            action_taken="re_embed",
            error_message=str(e),
            old_hash=old_hash,
        )


def _heal_version_mismatch(
    storage: Any,
    chunk: Any,
    config: HealingConfig,
    embedding_fn: Optional[Callable[[str], List[float]]],
    old_hash: str,
) -> HealingResult:
    """Heal a chunk with model version mismatch by re-embedding."""
    # Same as missing embedding - re-generate with new model
    return _heal_missing_embedding(storage, chunk, config, embedding_fn, old_hash)


def _heal_hash_mismatch(
    storage: Any,
    chunk: Any,
    config: HealingConfig,
    old_hash: str,
) -> HealingResult:
    """Heal a chunk with hash mismatch by recomputing hash."""
    chunk_id = chunk.chunk_id

    try:
        # Recompute hash
        new_hash = compute_content_hash(chunk.content)

        # Update metadata
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["content_hash"] = new_hash
        chunk.metadata["healed_at"] = datetime.now(timezone.utc).isoformat()

        # Write to storage (JPL Rule #7)
        update_success = storage.add_chunk(chunk)
        if not update_success:
            return HealingResult(
                chunk_id=chunk_id,
                success=False,
                action_taken="rehash",
                error_message="Storage update failed",
                old_hash=old_hash,
            )

        # Verify (AC)
        if not verify_hash_write(storage, chunk_id, new_hash):
            return HealingResult(
                chunk_id=chunk_id,
                success=False,
                action_taken="rehash",
                error_message="Hash verification failed",
                old_hash=old_hash,
                new_hash=new_hash,
            )

        return HealingResult(
            chunk_id=chunk_id,
            success=True,
            action_taken="rehash",
            old_hash=old_hash,
            new_hash=new_hash,
        )

    except Exception as e:
        return HealingResult(
            chunk_id=chunk_id,
            success=False,
            action_taken="rehash",
            error_message=str(e),
            old_hash=old_hash,
        )


# =============================================================================
# BACKGROUND HEALER CLASS
# =============================================================================


class BackgroundHealer:
    """
    Background worker that autonomously heals stale artifacts.

    Background Healer Worker.

    Features:
    - Scans for model-version mismatches
    - Re-runs stale stages during idle time
    - Verifies hash after each healing
    - Respects time bounds for idle-time operation

    JPL Rule #2: Bounded batch size and time.
    JPL Rule #4: Methods < 60 lines.
    JPL Rule #7: Checks all return values.
    JPL Rule #9: Complete type hints.
    """

    def __init__(
        self,
        storage: Any,
        config: Optional[HealingConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        """
        Initialize the background healer.

        Args:
            storage: Storage backend implementing ChunkRepository.
            config: Healing configuration (default created if None).
            embedding_fn: Function to generate embeddings.
        """
        self._storage = storage
        self._config = config or HealingConfig()
        self._embedding_fn = embedding_fn

    def run_session(self) -> HealingSessionResult:
        """
        Run a healing session.

        AC: Re-runs stale stages in idle time.
        Rule #2: Bounded by max_time_sec.
        Rule #4: < 60 lines.

        Returns:
            HealingSessionResult with session outcome.
        """
        import uuid

        session_id = str(uuid.uuid4())[:8]
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        logger.info(f"Starting healing session {session_id}")

        # Scan for stale artifacts
        stale_artifacts = scan_for_stale_artifacts(self._storage, self._config)

        result = HealingSessionResult(
            session_id=session_id,
            started_at=started_at,
            completed_at="",
            total_scanned=len(stale_artifacts),
            stale_found=len(stale_artifacts),
        )

        # Process in batches, respecting time limit
        processed = 0
        for stale in stale_artifacts[: self._config.batch_size]:
            # Check time limit
            elapsed = time.perf_counter() - start_time
            if elapsed >= self._config.max_time_sec:
                logger.info(f"Time limit reached ({self._config.max_time_sec}s)")
                break

            heal_result = heal_artifact(
                self._storage, stale, self._config, self._embedding_fn
            )
            result.results.append(heal_result)

            if heal_result.success:
                result.healed += 1
            else:
                result.failed += 1

            processed += 1

        result.skipped = len(stale_artifacts) - processed
        result.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Session {session_id} complete: "
            f"{result.healed} healed, {result.failed} failed, "
            f"{result.skipped} skipped"
        )

        return result

    def heal_document(self, document_id: str) -> HealingSessionResult:
        """
        Heal all stale artifacts for a specific document.

        Rule #4: < 60 lines.

        Args:
            document_id: Document to heal.

        Returns:
            HealingSessionResult for the document.
        """
        session_id = f"doc-{document_id[:8]}"
        started_at = datetime.now(timezone.utc).isoformat()

        chunks = self._storage.get_chunks_by_document(document_id)
        stale_artifacts = []

        for chunk in chunks:
            stale = _check_chunk_staleness(chunk, self._config)
            if stale:
                stale_artifacts.append(stale)

        result = HealingSessionResult(
            session_id=session_id,
            started_at=started_at,
            completed_at="",
            total_scanned=len(chunks),
            stale_found=len(stale_artifacts),
        )

        for stale in stale_artifacts:
            heal_result = heal_artifact(
                self._storage, stale, self._config, self._embedding_fn
            )
            result.results.append(heal_result)

            if heal_result.success:
                result.healed += 1
            else:
                result.failed += 1

        result.completed_at = datetime.now(timezone.utc).isoformat()
        return result
