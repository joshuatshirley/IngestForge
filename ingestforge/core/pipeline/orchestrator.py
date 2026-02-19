"""
Multi-Worker Orchestrator for IngestForge (IF).

Multi-Worker Orchestrator - Parallel document processing across CPU cores.
Artifact Serialization IPC - High-fidelity serialization with hash verification.
Worker Fault Recovery - Error containment and automatic retry.
Follows NASA JPL Power of Ten rules.

Features:
- ProcessPoolExecutor for true parallel processing
- Isolated IFRegistry instance per worker
- Thread-safe result aggregation via Queue
- No sub-process recursion (JPL Rule #1)
- model_dump() serialization with SHA-256 hash verification
- Worker watchdog with bounded retry attempts

NASA JPL Power of Ten Rules:
- Rule #2: Fixed upper bounds on workers, batch sizes, and retry attempts
- Rule #7: Check all return values
- Rule #10: Verify SHA-256 hash post-receive ()
"""

import hashlib
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Manager, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import uuid

logger = logging.getLogger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_WORKERS = 32  # Maximum parallel workers
MAX_BATCH_SIZE = 1000  # Maximum documents per batch
MAX_QUEUE_SIZE = 10000  # Maximum pending results
WORKER_TIMEOUT_SEC = 600  # 10-minute timeout per document
MAX_RETRY_COUNT = 3  # Maximum retry attempts per document (JPL Rule #2)


# =============================================================================
# SERIALIZATION AND HASH VERIFICATION
# =============================================================================


def _compute_hash(data: str) -> str:
    """
    Compute SHA-256 hash of data.

    AC / JPL Rule #10: SHA-256 hash for verification.
    Rule #4: <60 lines.

    Args:
        data: String data to hash.

    Returns:
        Hexadecimal hash string.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _serialize_artifact(artifact: Any) -> Tuple[Dict[str, Any], str]:
    """
    Serialize an artifact using model_dump() and compute hash.

    AC: Uses model_dump() for serialization.
    Rule #4: <60 lines.
    Rule #7: Handle serialization errors.

    Args:
        artifact: Pydantic artifact to serialize.

    Returns:
        Tuple of (serialized_dict, sha256_hash).
    """
    try:
        # AC: Uses model_dump() for serialization
        if hasattr(artifact, "model_dump"):
            data = artifact.model_dump(mode="json")
        elif hasattr(artifact, "dict"):
            # Fallback for older Pydantic
            data = artifact.dict()
        else:
            data = {"raw": str(artifact)}

        # Compute hash of serialized JSON
        import json

        serialized = json.dumps(data, sort_keys=True, default=str)
        data_hash = _compute_hash(serialized)

        return data, data_hash

    except Exception as e:
        logger.warning(f"Serialization failed: {e}")
        return {"error": str(e)}, ""


def _verify_worker_result(result: "WorkerResult") -> bool:
    """
    Verify SHA-256 hash of worker result.

    AC / JPL Rule #10: Verify hash post-receive on master process.
    Rule #4: <60 lines.

    Args:
        result: WorkerResult to verify.

    Returns:
        True if hash matches or no hash to verify, False if mismatch.
    """
    if result.artifact_data is None or result.artifact_hash is None:
        return True  # No data to verify

    if result.artifact_hash == "":
        return True  # Empty hash means serialization failed, skip verification

    try:
        import json

        serialized = json.dumps(result.artifact_data, sort_keys=True, default=str)
        computed_hash = _compute_hash(serialized)

        if computed_hash != result.artifact_hash:
            logger.error(
                f"Hash mismatch for {result.document_id}: "
                f"expected {result.artifact_hash[:16]}..., "
                f"got {computed_hash[:16]}..."
            )
            return False

        return True

    except Exception as e:
        logger.warning(f"Hash verification failed: {e}")
        return False


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class WorkerResult:
    """
    Result from a single worker task.

    Includes serialized artifact data with hash for verification.
    Tracks retry attempt count for fault recovery.
    Rule #9: Complete type hints.
    """

    document_id: str
    success: bool
    chunks_created: int = 0
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Serialized artifact data
    artifact_data: Optional[Dict[str, Any]] = None
    artifact_hash: Optional[str] = None  # SHA-256 of serialized data
    # Retry tracking
    retry_count: int = 0
    is_retryable: bool = True  # False if max retries exceeded or non-retryable error


@dataclass
class OrchestratorResult:
    """
    Aggregate result from orchestrator batch.

    Includes retry statistics for fault recovery tracking.
    Rule #9: Complete type hints.
    """

    batch_id: str
    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    total_chunks: int = 0
    total_duration_ms: float = 0.0
    results: List[WorkerResult] = field(default_factory=list)
    # Track hash verification stats
    hash_verified: int = 0
    hash_failed: int = 0
    # Track retry stats
    retries_attempted: int = 0
    retries_succeeded: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_documents == 0:
            return 100.0
        return (self.successful / self.total_documents) * 100


@dataclass
class TelemetryEvent:
    """
    Non-blocking telemetry event for worker monitoring.

    AC: multiprocessing.Manager.Queue for non-blocking telemetry.
    Rule #9: Complete type hints.
    """

    event_type: str  # "started", "completed", "failed", "progress"
    document_id: str
    worker_id: int
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# WORKER WATCHDOG
# =============================================================================

# Non-retryable error patterns (permanent failures)
_NON_RETRYABLE_ERRORS = frozenset(
    [
        "FileNotFoundError",
        "PermissionError",
        "UnicodeDecodeError",
        "IsADirectoryError",
        "NotADirectoryError",
    ]
)


class WorkerWatchdog:
    """
    Tracks retry attempts per document and determines retryability.

    AC: Worker Watchdog logic for fault recovery.
    JPL Rule #2: Strictly caps retry attempts to MAX_RETRY_COUNT.
    Rule #4: Class methods < 60 lines each.
    Rule #9: Complete type hints.
    """

    def __init__(self, max_retries: int = 1) -> None:
        """
        Initialize watchdog with retry limit.

        Args:
            max_retries: Maximum retry attempts per document (default: 1).
        """
        assert isinstance(max_retries, int), "max_retries must be int"
        assert (
            0 <= max_retries <= MAX_RETRY_COUNT
        ), f"max_retries must be 0..{MAX_RETRY_COUNT}"
        self._max_retries = max_retries
        self._retry_counts: Dict[str, int] = {}
        self._failed_permanently: Dict[str, str] = {}  # doc_id -> error_message

    def can_retry(self, document_id: str, error_message: Optional[str]) -> bool:
        """
        Check if a document can be retried.

        AC: Bounded retry attempts.
        JPL Rule #2: Never exceeds MAX_RETRY_COUNT.

        Args:
            document_id: Document identifier.
            error_message: Error that caused failure.

        Returns:
            True if retry is allowed, False otherwise.
        """
        # Check if permanently failed
        if document_id in self._failed_permanently:
            return False

        # Check retry count
        current_retries = self._retry_counts.get(document_id, 0)
        if current_retries >= self._max_retries:
            return False

        # Check if error is non-retryable
        if error_message and self._is_non_retryable(error_message):
            self._failed_permanently[document_id] = error_message
            return False

        return True

    def record_attempt(self, document_id: str) -> int:
        """
        Record a retry attempt for a document.

        Args:
            document_id: Document identifier.

        Returns:
            New retry count after increment.
        """
        current = self._retry_counts.get(document_id, 0)
        new_count = current + 1
        self._retry_counts[document_id] = new_count
        return new_count

    def get_retry_count(self, document_id: str) -> int:
        """Get current retry count for a document."""
        return self._retry_counts.get(document_id, 0)

    def mark_permanent_failure(self, document_id: str, reason: str) -> None:
        """Mark a document as permanently failed (no more retries)."""
        self._failed_permanently[document_id] = reason

    def _is_non_retryable(self, error_message: str) -> bool:
        """Check if error indicates a non-retryable failure."""
        for pattern in _NON_RETRYABLE_ERRORS:
            if pattern in error_message:
                return True
        return False

    @property
    def stats(self) -> Dict[str, int]:
        """Get retry statistics."""
        return {
            "documents_retried": len(self._retry_counts),
            "total_retries": sum(self._retry_counts.values()),
            "permanent_failures": len(self._failed_permanently),
        }


# =============================================================================
# MODULE-LEVEL WORKER FUNCTION (Required for ProcessPoolExecutor)
# =============================================================================


def _worker_process_document(
    document_path: str,
    config_path: Optional[str],
    base_path: str,
) -> WorkerResult:
    """
    Process a single document in an isolated worker process.

    AC: Each worker gets an isolated IFRegistry instance.
    AC: Assert registry health before processing (JPL Rule #5).
    JPL Rule #1: No recursion - this function does NOT call itself or spawn children.

    This is a module-level function (required for pickle/ProcessPoolExecutor).
    Each worker process has its own Python interpreter with isolated state.

    Args:
        document_path: Path to the document to process.
        config_path: Optional path to config file.
        base_path: Base path for project.

    Returns:
        WorkerResult with processing outcome.
    """
    start_time = time.perf_counter()
    doc_path = Path(document_path)

    try:
        # AC: Use reset_for_worker() for safe registry initialization
        # This ensures complete isolation between workers and asserts health
        from ingestforge.core.pipeline.registry import IFRegistry

        # Reset and assert health before processing
        registry = IFRegistry.reset_for_worker()
        assert registry.is_healthy(), "Registry health check failed"

        # Load config in isolated process
        from ingestforge.core.config_loaders import load_config

        config = load_config(base_path=Path(base_path))

        # Create pipeline with fresh registry
        from ingestforge.core.pipeline.pipeline import Pipeline

        pipeline = Pipeline(config=config, base_path=Path(base_path))

        # Disable state updates - main process handles state
        pipeline._skip_state_updates = True

        # Process the document
        result = pipeline.process_file(doc_path)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Serialize result with hash for verification
        artifact_data = None
        artifact_hash = None
        if hasattr(result, "model_dump"):
            artifact_data, artifact_hash = _serialize_artifact(result)

        return WorkerResult(
            document_id=result.document_id,
            success=result.success,
            chunks_created=result.chunks_created,
            duration_ms=duration_ms,
            metadata=result.metadata or {},
            artifact_data=artifact_data,
            artifact_hash=artifact_hash,
        )

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return WorkerResult(
            document_id=str(doc_path),
            success=False,
            error_message=str(e),
            duration_ms=duration_ms,
        )


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================


class MultiWorkerOrchestrator:
    """
    Orchestrates parallel document processing across CPU cores.

    Multi-Worker Orchestrator.
    Artifact Serialization IPC with hash verification.
    Worker Fault Recovery with bounded retry attempts.

    Features:
    - ProcessPoolExecutor for CPU-bound parallelism
    - Isolated registry per worker (subprocess isolation)
    - Thread-safe result aggregation
    - JPL Rule #1: No recursive subprocess spawning
    - model_dump() serialization with SHA-256 verification
    - Non-blocking telemetry via Manager.Queue
    - Worker watchdog with automatic retry on transient failures

    Example:
        orchestrator = MultiWorkerOrchestrator(max_workers=4, max_retry_count=2)
        result = orchestrator.process_batch(document_paths)
        print(f"Processed {result.successful}/{result.total_documents} documents")
        print(f"Retries: {result.retries_attempted}, succeeded: {result.retries_succeeded}")
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        config_path: Optional[str] = None,
        base_path: Optional[Path] = None,
        enable_telemetry: bool = False,
        max_retry_count: int = 1,
    ) -> None:
        """
        Initialize the orchestrator.

        Added telemetry queue support.
        Added max_retry_count for fault recovery.

        Args:
            max_workers: Maximum worker processes (default: cpu_count - 1).
            config_path: Optional path to config file.
            base_path: Base path for project (default: cwd).
            enable_telemetry: Enable non-blocking telemetry queue.
            max_retry_count: Maximum retry attempts per document (default: 1).
        """
        # JPL Rule #2: Bounded workers
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        self._max_workers = min(max_workers, MAX_WORKERS)

        self._config_path = config_path
        self._base_path = base_path or Path.cwd()

        # AC: Worker watchdog with bounded retries
        assert isinstance(max_retry_count, int), "max_retry_count must be int"
        self._max_retry_count = min(max_retry_count, MAX_RETRY_COUNT)
        self._watchdog = WorkerWatchdog(max_retries=self._max_retry_count)

        # AC: multiprocessing.Manager.Queue for non-blocking telemetry
        self._telemetry_queue: Optional[Queue] = None
        if enable_telemetry:
            manager = Manager()
            self._telemetry_queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)

        # Verify not running in subprocess (JPL Rule #1)
        assert (
            multiprocessing.current_process().name == "MainProcess"
        ), "Orchestrator must be created in main process only"

    def get_telemetry_events(self, max_events: int = 100) -> List[TelemetryEvent]:
        """
        Retrieve telemetry events from queue (non-blocking).

        AC: Non-blocking telemetry retrieval.
        Rule #2: Bounded event retrieval.

        Args:
            max_events: Maximum events to retrieve.

        Returns:
            List of telemetry events.
        """
        if self._telemetry_queue is None:
            return []

        events: List[TelemetryEvent] = []
        for _ in range(min(max_events, MAX_QUEUE_SIZE)):
            try:
                event = self._telemetry_queue.get_nowait()
                events.append(event)
            except Exception:
                break

        return events

    def process_batch(
        self,
        document_paths: List[Path],
        timeout_per_doc: float = WORKER_TIMEOUT_SEC,
    ) -> OrchestratorResult:
        """
        Process a batch of documents in parallel with fault recovery.

        AC: Uses ProcessPoolExecutor.
        AC: Results safely merged.
        AC: Worker watchdog with bounded retries.
        JPL Rule #2: Bounded batch size.

        Args:
            document_paths: List of document paths to process.
            timeout_per_doc: Timeout per document in seconds.

        Returns:
            OrchestratorResult with all results.

        Raises:
            ValueError: If batch size exceeds limit.
        """
        # JPL Rule #2: Bounded batch size
        if len(document_paths) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(document_paths)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        batch_id = str(uuid.uuid4())
        batch_result = OrchestratorResult(
            batch_id=batch_id,
            total_documents=len(document_paths),
        )

        if not document_paths:
            return batch_result

        start_time = time.perf_counter()

        logger.info(
            f"Processing batch {batch_id[:8]}: {len(document_paths)} documents "
            f"with {self._max_workers} workers"
        )

        # Reset watchdog for new batch
        self._watchdog = WorkerWatchdog(max_retries=self._max_retry_count)

        # Track pending documents for retry
        pending_docs = list(document_paths)
        processed_docs: Dict[str, WorkerResult] = {}

        # Process with retry loop (JPL Rule #2: bounded iterations)
        for attempt in range(self._max_retry_count + 1):
            if not pending_docs:
                break

            batch_result = self._process_batch_iteration(
                pending_docs, batch_result, timeout_per_doc, attempt
            )

            # Collect failed documents for retry
            pending_docs = self._collect_retryable_failures(batch_result)

            if pending_docs:
                logger.info(
                    f"Batch {batch_id[:8]}: Retrying {len(pending_docs)} failed documents "
                    f"(attempt {attempt + 2}/{self._max_retry_count + 1})"
                )

        batch_result.total_duration_ms = (time.perf_counter() - start_time) * 1000

        # Add watchdog stats
        stats = self._watchdog.stats
        batch_result.retries_attempted = stats["total_retries"]
        batch_result.retries_succeeded = sum(
            1 for r in batch_result.results if r.success and r.retry_count > 0
        )

        logger.info(
            f"Batch {batch_id[:8]} complete: "
            f"{batch_result.successful}/{batch_result.total_documents} successful "
            f"({batch_result.success_rate:.1f}%) in {batch_result.total_duration_ms:.0f}ms "
            f"[retries: {batch_result.retries_attempted}]"
        )

        return batch_result

    def _process_batch_iteration(
        self,
        document_paths: List[Path],
        batch_result: OrchestratorResult,
        timeout_per_doc: float,
        attempt: int,
    ) -> OrchestratorResult:
        """
        Process one iteration of the batch.

        Extracted for JPL Rule #4 compliance.
        Rule #4: Function < 60 lines.

        Args:
            document_paths: Documents to process this iteration.
            batch_result: Aggregate result to update.
            timeout_per_doc: Timeout per document.
            attempt: Current attempt number (0-based).

        Returns:
            Updated batch result.
        """
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    _worker_process_document,
                    str(doc_path),
                    self._config_path,
                    str(self._base_path),
                ): doc_path
                for doc_path in document_paths
            }

            total_timeout = timeout_per_doc * len(document_paths)
            for future in as_completed(futures, timeout=total_timeout):
                doc_path = futures[future]
                doc_id = str(doc_path)

                try:
                    worker_result = future.result(timeout=timeout_per_doc)
                    worker_result.retry_count = self._watchdog.get_retry_count(doc_id)

                    # Verify hash
                    if not _verify_worker_result(worker_result):
                        worker_result = self._create_failure_result(
                            doc_id, "Hash verification failed", attempt
                        )

                    self._aggregate_result(batch_result, worker_result)

                except Exception as e:
                    worker_result = self._create_failure_result(
                        doc_id, f"Worker error: {e}", attempt
                    )
                    self._aggregate_result(batch_result, worker_result)
                    logger.error(f"Worker exception for {doc_path}: {e}")

        return batch_result

    def _create_failure_result(
        self, doc_id: str, error: str, attempt: int
    ) -> WorkerResult:
        """
        Create a failure result with retry tracking.

        Helper for fault recovery.
        Rule #4: Function < 60 lines.
        """
        retry_count = self._watchdog.get_retry_count(doc_id)
        can_retry = self._watchdog.can_retry(doc_id, error)

        return WorkerResult(
            document_id=doc_id,
            success=False,
            error_message=error,
            retry_count=retry_count,
            is_retryable=can_retry,
        )

    def _aggregate_result(
        self, batch_result: OrchestratorResult, worker_result: WorkerResult
    ) -> None:
        """
        Aggregate a worker result into the batch.

        Extracted for code clarity.
        Rule #4: Function < 60 lines.
        """
        # Remove previous result for same document (if retrying)
        batch_result.results = [
            r
            for r in batch_result.results
            if r.document_id != worker_result.document_id
        ]
        batch_result.results.append(worker_result)

        if worker_result.success:
            batch_result.successful += 1
            batch_result.total_chunks += worker_result.chunks_created
        else:
            batch_result.failed += 1
            logger.warning(
                f"Document failed: {worker_result.document_id}: "
                f"{worker_result.error_message}"
            )

    def _collect_retryable_failures(
        self, batch_result: OrchestratorResult
    ) -> List[Path]:
        """
        Collect failed documents that can be retried.

        AC: Worker watchdog with bounded retries.
        Rule #4: Function < 60 lines.

        Returns:
            List of document paths to retry.
        """
        retryable: List[Path] = []

        for result in batch_result.results:
            if result.success:
                continue

            doc_id = result.document_id
            if self._watchdog.can_retry(doc_id, result.error_message):
                self._watchdog.record_attempt(doc_id)
                retryable.append(Path(doc_id))

                # Adjust counts for retry (will be recounted)
                batch_result.failed -= 1

        return retryable

    @property
    def max_workers(self) -> int:
        """Get configured maximum workers."""
        return self._max_workers


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def process_documents_parallel(
    document_paths: List[Path],
    max_workers: Optional[int] = None,
    config_path: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> OrchestratorResult:
    """
    Convenience function to process documents in parallel.

    Multi-Worker Orchestrator.

    Args:
        document_paths: List of document paths to process.
        max_workers: Maximum worker processes (default: cpu_count - 1).
        config_path: Optional path to config file.
        base_path: Base path for project.

    Returns:
        OrchestratorResult with all processing results.

    Example:
        from pathlib import Path
        from ingestforge.core.pipeline.orchestrator import process_documents_parallel

        docs = [Path("doc1.pdf"), Path("doc2.pdf"), Path("doc3.pdf")]
        result = process_documents_parallel(docs, max_workers=4)
        print(f"Success rate: {result.success_rate}%")
    """
    orchestrator = MultiWorkerOrchestrator(
        max_workers=max_workers,
        config_path=config_path,
        base_path=base_path,
    )
    return orchestrator.process_batch(document_paths)
