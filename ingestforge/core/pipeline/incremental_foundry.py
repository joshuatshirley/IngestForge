"""
Incremental Foundry Runner.

Optimize IFPipelineRunner to skip documents already processed.
Checks StorageBackend for existing IFArtifact with matching source_hash.

Follows NASA JPL Power of Ten:
- Rule #1: No recursion
- Rule #2: Fixed bounds on all data structures
- Rule #4: Functions under 60 lines
- Rule #5: Assertions at entry points
- Rule #7: Check return values
- Rule #9: Complete type hints
- Rule #10: Cryptographic verification
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ingestforge.core.pipeline.interfaces import IFArtifact, IFStage
from ingestforge.core.pipeline.artifacts import IFFailureArtifact, IFFileArtifact
from ingestforge.core.pipeline.runner import IFPipelineRunner, ResourceConfig

# JPL Rule #2: Fixed upper bounds
MAX_HASH_CACHE_SIZE = 10000
MAX_DOCUMENT_LOOKUP_RETRIES = 3
HASH_ALGORITHM = "sha256"
HASH_CHUNK_SIZE = 8192

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol for Storage Backend
# =============================================================================


class IFStorageProtocol(Protocol):
    """
    Protocol for storage backends that support hash lookup.

    Rule #9: Complete type hints via Protocol.
    """

    def get_chunks_by_document(self, document_id: str) -> List[Any]:
        """Get all chunks for a document."""
        ...

    def verify_chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk exists."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class IncrementalSkipResult:
    """
    Result of incremental skip check.

    Rule #9: Complete type hints.
    """

    should_skip: bool
    source_hash: str
    reason: str  # "cached", "force", "not_found", "error"
    existing_artifact_id: Optional[str] = None
    check_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.source_hash, "source_hash must be non-empty"
        assert self.reason in (
            "cached",
            "force",
            "not_found",
            "error",
        ), f"invalid reason: {self.reason}"


@dataclass(frozen=True)
class IncrementalFoundryConfig:
    """
    Configuration for incremental foundry runner.

    Rule #2: Fixed upper bounds.
    Rule #9: Complete type hints.
    """

    enabled: bool = True
    force: bool = False
    verify_cryptographic: bool = True
    cache_size: int = MAX_HASH_CACHE_SIZE

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        if self.cache_size > MAX_HASH_CACHE_SIZE:
            object.__setattr__(self, "cache_size", MAX_HASH_CACHE_SIZE)
        if self.cache_size < 0:
            object.__setattr__(self, "cache_size", 0)


@dataclass
class IncrementalFoundryReport:
    """
    Report from incremental foundry run.

    Rule #9: Complete type hints.
    """

    total_documents: int
    processed_documents: int
    skipped_documents: int
    forced_documents: int
    error_documents: int
    total_time_ms: float
    skip_checks: List[IncrementalSkipResult]

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.total_documents >= 0, "total_documents cannot be negative"
        assert self.processed_documents >= 0, "processed_documents cannot be negative"
        assert self.skipped_documents >= 0, "skipped_documents cannot be negative"

    @property
    def skip_rate(self) -> float:
        """Calculate skip rate percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.skipped_documents / self.total_documents) * 100


# =============================================================================
# Hash Calculator
# =============================================================================


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.

    JPL Rule #10: Cryptographic verification.
    Rule #4: Under 60 lines.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hex-encoded SHA-256 hash.

    Raises:
        FileNotFoundError: If file doesn't exist.
        IOError: If file cannot be read.
    """
    assert file_path is not None, "file_path cannot be None"

    if not file_path.exists():
        # SEC-002: Sanitize path disclosure
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError("File not found: [REDACTED]")

    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def calculate_content_hash(content: str) -> str:
    """
    Calculate SHA-256 hash of string content.

    JPL Rule #10: Cryptographic verification.

    Args:
        content: String content to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# Hash Registry
# =============================================================================


class HashRegistry:
    """
    In-memory registry of processed document hashes.

    Rule #2: Bounded by MAX_HASH_CACHE_SIZE.
    Rule #9: Complete type hints.
    """

    def __init__(self, max_size: int = MAX_HASH_CACHE_SIZE) -> None:
        """
        Initialize hash registry.

        Args:
            max_size: Maximum number of hashes to cache.

        Rule #5: Assert preconditions.
        """
        assert max_size > 0, "max_size must be positive"

        self._max_size = min(max_size, MAX_HASH_CACHE_SIZE)
        self._hashes: Dict[str, str] = {}  # hash -> artifact_id
        self._insertion_order: List[str] = []

    @property
    def size(self) -> int:
        """Return current registry size."""
        return len(self._hashes)

    def contains(self, content_hash: str) -> bool:
        """
        Check if hash is in registry.

        Args:
            content_hash: SHA-256 hash to check.

        Returns:
            True if hash exists in registry.
        """
        return content_hash in self._hashes

    def get_artifact_id(self, content_hash: str) -> Optional[str]:
        """
        Get artifact ID for a hash.

        Args:
            content_hash: SHA-256 hash to look up.

        Returns:
            Artifact ID if found, None otherwise.
        """
        return self._hashes.get(content_hash)

    def register(self, content_hash: str, artifact_id: str) -> bool:
        """
        Register a processed hash.

        Rule #2: Evicts oldest entries when limit reached.

        Args:
            content_hash: SHA-256 hash of content.
            artifact_id: ID of the processed artifact.

        Returns:
            True if registered successfully.
        """
        if content_hash in self._hashes:
            # Update existing entry
            self._hashes[content_hash] = artifact_id
            return True

        # Evict if at capacity
        while len(self._hashes) >= self._max_size and self._insertion_order:
            oldest = self._insertion_order.pop(0)
            self._hashes.pop(oldest, None)

        self._hashes[content_hash] = artifact_id
        self._insertion_order.append(content_hash)
        return True

    def clear(self) -> None:
        """Clear all registered hashes."""
        self._hashes.clear()
        self._insertion_order.clear()


# =============================================================================
# Incremental Foundry Runner
# =============================================================================


class IncrementalFoundryRunner:
    """
    Pipeline runner that skips already-processed documents.

    Optimize IFPipelineRunner to skip documents already forged.

    Rule #1: No recursion - linear control flow.
    Rule #2: Bounded hash cache.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        storage: Optional[IFStorageProtocol] = None,
        config: Optional[IncrementalFoundryConfig] = None,
        resource_config: Optional[ResourceConfig] = None,
    ) -> None:
        """
        Initialize incremental foundry runner.

        Args:
            storage: Storage backend for checking existing artifacts.
            config: Incremental processing configuration.
            resource_config: Resource limits for pipeline execution.

        Rule #5: Assert preconditions.
        """
        self._storage = storage
        self._config = config or IncrementalFoundryConfig()
        self._hash_registry = HashRegistry(self._config.cache_size)
        self._base_runner = IFPipelineRunner(resource_config=resource_config)

    def check_should_skip(
        self,
        artifact: IFArtifact,
        force: bool = False,
    ) -> IncrementalSkipResult:
        """
        Check if artifact processing should be skipped.

        Rule #4: Under 60 lines.
        Rule #7: Check return values.
        Rule #10: Cryptographic verification.

        Args:
            artifact: Input artifact to check.
            force: Override skip logic if True.

        Returns:
            IncrementalSkipResult indicating skip decision.
        """
        start_time = time.perf_counter()

        # Calculate source hash
        try:
            content_hash = self._get_artifact_hash(artifact)
        except Exception as e:
            logger.warning(f"Failed to calculate hash: {e}")
            return IncrementalSkipResult(
                should_skip=False,
                source_hash="error",
                reason="error",
                check_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Force flag overrides skip
        if force or self._config.force:
            logger.info(f"FORCE PROCESS: {artifact.artifact_id}")
            return IncrementalSkipResult(
                should_skip=False,
                source_hash=content_hash,
                reason="force",
                check_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check local cache first
        cached_id = self._hash_registry.get_artifact_id(content_hash)
        if cached_id:
            logger.info(f"INCREMENTAL SKIP (cached): {artifact.artifact_id}")
            return IncrementalSkipResult(
                should_skip=True,
                source_hash=content_hash,
                reason="cached",
                existing_artifact_id=cached_id,
                check_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check storage if available
        if self._storage:
            existing_id = self._check_storage_for_hash(
                artifact.artifact_id, content_hash
            )
            if existing_id:
                self._hash_registry.register(content_hash, existing_id)
                logger.info(f"INCREMENTAL SKIP (storage): {artifact.artifact_id}")
                return IncrementalSkipResult(
                    should_skip=True,
                    source_hash=content_hash,
                    reason="cached",
                    existing_artifact_id=existing_id,
                    check_duration_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Not found - needs processing
        duration_ms = (time.perf_counter() - start_time) * 1000
        return IncrementalSkipResult(
            should_skip=False,
            source_hash=content_hash,
            reason="not_found",
            check_duration_ms=duration_ms,
        )

    def _get_artifact_hash(self, artifact: IFArtifact) -> str:
        """
        Get content hash from artifact.

        Rule #4: Under 60 lines.
        Rule #10: Cryptographic verification (SHA-256).

        Args:
            artifact: Artifact to hash.

        Returns:
            SHA-256 hash string.
        """
        # IFFileArtifact: hash from file
        if isinstance(artifact, IFFileArtifact):
            if artifact.content_hash:
                return artifact.content_hash
            return calculate_file_hash(artifact.file_path)

        # Other artifacts: hash from content if available
        if hasattr(artifact, "content_hash") and artifact.content_hash:
            return artifact.content_hash

        if hasattr(artifact, "content") and artifact.content:
            return calculate_content_hash(artifact.content)

        # Fallback: hash artifact_id + metadata
        return calculate_content_hash(
            f"{artifact.artifact_id}:{str(artifact.metadata)}"
        )

    def _check_storage_for_hash(
        self,
        document_id: str,
        content_hash: str,
    ) -> Optional[str]:
        """
        Check storage for existing artifact with matching hash.

        Rule #4: Under 60 lines.
        Rule #7: Check return values.

        Args:
            document_id: Document ID to check.
            content_hash: Content hash to match.

        Returns:
            Artifact ID if found, None otherwise.
        """
        if not self._storage:
            return None

        try:
            chunks = self._storage.get_chunks_by_document(document_id)
            if not chunks:
                return None

            # Check if any chunk has matching content_hash
            for chunk in chunks:
                chunk_hash = None
                if hasattr(chunk, "content_hash"):
                    chunk_hash = chunk.content_hash
                elif hasattr(chunk, "metadata") and chunk.metadata:
                    chunk_hash = chunk.metadata.get("content_hash")

                if chunk_hash == content_hash:
                    return getattr(chunk, "chunk_id", document_id)

            return None

        except Exception as e:
            logger.warning(f"Storage check failed for {document_id}: {e}")
            return None

    def run_incremental(
        self,
        artifact: IFArtifact,
        stages: List[IFStage],
        document_id: str,
        force: bool = False,
    ) -> Tuple[IFArtifact, IncrementalSkipResult]:
        """
        Run pipeline with incremental skip check.

        Main entry point for incremental processing.

        Rule #1: Linear control flow.
        Rule #4: Under 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact.
            stages: Pipeline stages to execute.
            document_id: Document identifier.
            force: Override skip logic.

        Returns:
            Tuple of (result artifact, skip result).
        """
        # Check if should skip
        skip_result = self.check_should_skip(artifact, force)

        if skip_result.should_skip and self._config.enabled:
            # Return original artifact (already processed)
            return artifact, skip_result

        # Process normally
        result = self._base_runner.run(artifact, stages, document_id)

        # Register hash if successful
        if not isinstance(result, IFFailureArtifact):
            self._hash_registry.register(
                skip_result.source_hash,
                result.artifact_id,
            )

        return result, skip_result

    def run_batch_incremental(
        self,
        artifacts: List[Tuple[IFArtifact, str]],
        stages: List[IFStage],
        force: bool = False,
    ) -> IncrementalFoundryReport:
        """
        Run batch processing with incremental skipping.

        Rule #1: Linear control flow.
        Rule #2: Process in order.
        Rule #4: Under 60 lines.

        Args:
            artifacts: List of (artifact, document_id) tuples.
            stages: Pipeline stages to execute.
            force: Override skip logic for all.

        Returns:
            IncrementalFoundryReport with batch results.
        """
        start_time = time.perf_counter()

        processed = 0
        skipped = 0
        forced = 0
        errors = 0
        skip_checks: List[IncrementalSkipResult] = []

        for artifact, doc_id in artifacts:
            result, skip_result = self.run_incremental(artifact, stages, doc_id, force)
            skip_checks.append(skip_result)

            if skip_result.should_skip:
                skipped += 1
            elif skip_result.reason == "force":
                forced += 1
                if not isinstance(result, IFFailureArtifact):
                    processed += 1
                else:
                    errors += 1
            elif isinstance(result, IFFailureArtifact):
                errors += 1
            else:
                processed += 1

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return IncrementalFoundryReport(
            total_documents=len(artifacts),
            processed_documents=processed,
            skipped_documents=skipped,
            forced_documents=forced,
            error_documents=errors,
            total_time_ms=total_time_ms,
            skip_checks=skip_checks,
        )

    def mark_processed(self, artifact: IFArtifact) -> bool:
        """
        Mark an artifact as processed in the hash registry.

        Args:
            artifact: Processed artifact.

        Returns:
            True if registered successfully.
        """
        try:
            content_hash = self._get_artifact_hash(artifact)
            return self._hash_registry.register(content_hash, artifact.artifact_id)
        except Exception as e:
            logger.warning(f"Failed to mark processed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get runner statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "hash_cache_size": self._hash_registry.size,
            "max_cache_size": self._config.cache_size,
            "incremental_enabled": self._config.enabled,
            "force_mode": self._config.force,
            "storage_connected": self._storage is not None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_incremental_foundry_runner(
    storage: Optional[IFStorageProtocol] = None,
    enabled: bool = True,
    force: bool = False,
) -> IncrementalFoundryRunner:
    """
    Create an incremental foundry runner.

    Args:
        storage: Storage backend for hash lookups.
        enabled: Enable incremental skipping.
        force: Force reprocessing (--force flag).

    Returns:
        Configured IncrementalFoundryRunner.
    """
    config = IncrementalFoundryConfig(enabled=enabled, force=force)
    return IncrementalFoundryRunner(storage=storage, config=config)


def run_incremental_batch(
    artifacts: List[Tuple[IFArtifact, str]],
    stages: List[IFStage],
    storage: Optional[IFStorageProtocol] = None,
    force: bool = False,
) -> IncrementalFoundryReport:
    """
    Convenience function for batch incremental processing.

    Args:
        artifacts: List of (artifact, document_id) tuples.
        stages: Pipeline stages.
        storage: Optional storage backend.
        force: Force reprocessing.

    Returns:
        Processing report.
    """
    runner = create_incremental_foundry_runner(storage, force=force)
    return runner.run_batch_incremental(artifacts, stages, force)
