"""
Extraction Health Monitoring for IngestForge.

Extraction-Health-Monitoring
Provides real-time visibility into extraction success rates and model performance.

NASA JPL Power of Ten compliant.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ingestforge.core.pipeline.interfaces import IFArtifact, IFInterceptor
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_METRICS_PER_SESSION = 100000
MAX_DOCUMENT_TYPES = 256
MAX_STAGES_TRACKED = 64


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class HealthMetric(BaseModel):
    """
    Health metric for a single extraction operation.

    AC: Contains token_count, latency_ms, and success_status.
    Rule #9: Complete type hints.
    """

    model_config = {"frozen": True}

    stage_name: str = Field(
        ..., description="Name of the stage that emitted this metric"
    )
    document_id: str = Field(..., description="Document being processed")
    document_type: str = Field(
        "unknown", description="Type of document (pdf, html, etc.)"
    )
    token_count: int = Field(0, ge=0, description="Number of tokens processed")
    latency_ms: float = Field(0.0, ge=0.0, description="Execution time in milliseconds")
    success_status: bool = Field(True, description="Whether the extraction succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    memory_mb: Optional[float] = Field(
        None, ge=0.0, description="Peak memory usage in MB"
    )
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "success_status": self.success_status,
            "error_message": self.error_message,
            "memory_mb": self.memory_mb,
            "timestamp": self.timestamp,
        }


@dataclass
class StageStats:
    """
    Aggregated statistics for a single stage.

    Rule #9: Complete type hints.
    """

    stage_name: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_calls == 0:
            return 100.0
        return (self.success_count / self.total_calls) * 100.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count


@dataclass
class DocumentTypeStats:
    """
    Statistics aggregated by document type.

    AC: Track "Extraction Failure Rate" per document type.
    Rule #9: Complete type hints.
    """

    document_type: str
    total_processed: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.failure_count / self.total_processed) * 100.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return 100.0 - self.failure_rate

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_processed == 0:
            return 0.0
        return self.total_latency_ms / self.total_processed


class IFPipelineResult(BaseModel):
    """
    Result of pipeline execution with metrics.

    AC: Export metrics to final result.
    Rule #9: Complete type hints.
    """

    model_config = {"frozen": True}

    success: bool = Field(..., description="Overall pipeline success")
    document_id: str = Field(..., description="Document processed")
    total_duration_ms: float = Field(0.0, ge=0.0, description="Total pipeline duration")
    stages_completed: int = Field(0, ge=0, description="Number of stages completed")
    total_stages: int = Field(0, ge=0, description="Total number of stages")
    total_tokens: int = Field(0, ge=0, description="Total tokens processed")
    peak_memory_mb: float = Field(0.0, ge=0.0, description="Peak memory usage")
    success_rate: float = Field(
        100.0, ge=0.0, le=100.0, description="Overall success rate"
    )
    avg_latency_ms: float = Field(0.0, ge=0.0, description="Average stage latency")
    metrics: Tuple[HealthMetric, ...] = Field(
        default_factory=tuple, description="Individual stage metrics"
    )
    error_message: Optional[str] = Field(None, description="Error if pipeline failed")


# ---------------------------------------------------------------------------
# IFExtractionStats Collector
# ---------------------------------------------------------------------------


class IFExtractionStats:
    """
    Thread-safe collector for extraction health metrics.

    AC: Implements IFExtractionStats collector.
    - Tracks metrics per stage and document type
    - Thread-safe for concurrent batch processing
    - Provides CLI summary output

    NASA JPL Power of Ten compliant.
    Rule #2: Fixed upper bounds on metrics.
    Rule #9: Complete type hints.
    """

    _instance: Optional["IFExtractionStats"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "IFExtractionStats":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        """Initialize collector state."""
        self._metrics: List[HealthMetric] = []
        self._stage_stats: Dict[str, StageStats] = {}
        self._doc_type_stats: Dict[str, DocumentTypeStats] = {}
        self._data_lock: threading.RLock = threading.RLock()
        self._session_active: bool = False
        self._session_start: float = 0.0

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance for testing."""
        with cls._lock:
            cls._instance = None

    def start_session(self) -> bool:
        """
        Start a new metrics collection session.

        Clears all existing metrics and marks session as active.

        Returns:
            True if session started successfully.
        """
        with self._data_lock:
            self._metrics.clear()
            self._stage_stats.clear()
            self._doc_type_stats.clear()
            self._session_active = True
            self._session_start = time.time()
            logger.info("Extraction stats session started")
            return True

    def end_session(self) -> IFPipelineResult:
        """
        End the current session and return final result.

        Returns:
            IFPipelineResult with aggregated metrics.
        """
        with self._data_lock:
            self._session_active = False
            duration_ms = (time.time() - self._session_start) * 1000

            total_success = sum(s.success_count for s in self._stage_stats.values())
            total_calls = sum(s.total_calls for s in self._stage_stats.values())
            total_tokens = sum(s.total_tokens for s in self._stage_stats.values())
            peak_memory = max(
                (s.peak_memory_mb for s in self._stage_stats.values()), default=0.0
            )

            success_rate = 100.0
            if total_calls > 0:
                success_rate = (total_success / total_calls) * 100.0

            avg_latency = 0.0
            total_latency = sum(s.total_latency_ms for s in self._stage_stats.values())
            if total_calls > 0:
                avg_latency = total_latency / total_calls

            result = IFPipelineResult(
                success=total_success == total_calls,
                document_id="session",
                total_duration_ms=duration_ms,
                stages_completed=total_success,
                total_stages=total_calls,
                total_tokens=total_tokens,
                peak_memory_mb=peak_memory,
                success_rate=success_rate,
                avg_latency_ms=avg_latency,
                metrics=tuple(self._metrics[:1000]),  # JPL Rule #2: Bounded
            )

            logger.info(
                f"Extraction stats session ended: "
                f"{total_success}/{total_calls} successful, "
                f"{success_rate:.1f}% success rate"
            )
            return result

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._session_active

    def _update_stage_stats(self, metric: HealthMetric) -> None:
        """Update per-stage statistics. Rule #4: Helper < 60 lines."""
        if metric.stage_name not in self._stage_stats:
            if len(self._stage_stats) >= MAX_STAGES_TRACKED:
                logger.warning("Stage stats limit reached")
                return
            self._stage_stats[metric.stage_name] = StageStats(
                stage_name=metric.stage_name
            )

        stats = self._stage_stats[metric.stage_name]
        stats.total_calls += 1
        if metric.success_status:
            stats.success_count += 1
        else:
            stats.failure_count += 1
        stats.total_tokens += metric.token_count
        stats.total_latency_ms += metric.latency_ms
        if metric.memory_mb and metric.memory_mb > stats.peak_memory_mb:
            stats.peak_memory_mb = metric.memory_mb

    def _update_doc_type_stats(self, metric: HealthMetric) -> None:
        """Update per-document-type statistics. Rule #4: Helper < 60 lines."""
        doc_type = metric.document_type
        if doc_type not in self._doc_type_stats:
            if len(self._doc_type_stats) >= MAX_DOCUMENT_TYPES:
                logger.warning("Document type stats limit reached")
                return
            self._doc_type_stats[doc_type] = DocumentTypeStats(document_type=doc_type)

        stats = self._doc_type_stats[doc_type]
        stats.total_processed += 1
        stats.total_latency_ms += metric.latency_ms
        if metric.success_status:
            stats.success_count += 1
        else:
            stats.failure_count += 1
            # AC: Log failure rate per document type
            logger.debug(
                f"Extraction failure for {doc_type}: "
                f"{stats.failure_rate:.1f}% failure rate"
            )

    def record_metric(self, metric: HealthMetric) -> bool:
        """
        Record a single health metric.

        Rule #2: Bounded metrics list.
        Rule #4: Function < 60 lines (uses helpers).

        Args:
            metric: The health metric to record.

        Returns:
            True if recorded, False if limit reached.
        """
        with self._data_lock:
            if not self._session_active:
                logger.warning("Cannot record metric: no active session")
                return False

            if len(self._metrics) >= MAX_METRICS_PER_SESSION:
                logger.warning(f"Metrics limit reached ({MAX_METRICS_PER_SESSION})")
                return False

            self._metrics.append(metric)
            self._update_stage_stats(metric)
            self._update_doc_type_stats(metric)
            return True

    def emit(
        self,
        stage_name: str,
        document_id: str,
        latency_ms: float,
        success_status: bool,
        token_count: int = 0,
        document_type: str = "unknown",
        error_message: Optional[str] = None,
        memory_mb: Optional[float] = None,
    ) -> bool:
        """
        Convenience method to emit a health metric.

        GWT: Processor emits HealthMetric after extraction run.
        Rule #5: Assertions for parameter validation.

        Args:
            stage_name: Name of the stage.
            document_id: Document being processed.
            latency_ms: Execution time in milliseconds.
            success_status: Whether extraction succeeded.
            token_count: Number of tokens processed.
            document_type: Type of document (pdf, html, etc.).
            error_message: Error message if failed.
            memory_mb: Peak memory usage.

        Returns:
            True if recorded successfully.
        """
        # JPL Rule #5: Assert preconditions
        assert stage_name, "stage_name must be non-empty"
        assert document_id, "document_id must be non-empty"
        assert latency_ms >= 0, "latency_ms must be non-negative"
        assert token_count >= 0, "token_count must be non-negative"

        metric = HealthMetric(
            stage_name=stage_name,
            document_id=document_id,
            document_type=document_type,
            token_count=token_count,
            latency_ms=latency_ms,
            success_status=success_status,
            error_message=error_message,
            memory_mb=memory_mb,
        )
        return self.record_metric(metric)

    def get_stage_stats(self, stage_name: str) -> Optional[StageStats]:
        """Get statistics for a specific stage."""
        with self._data_lock:
            return self._stage_stats.get(stage_name)

    def get_all_stage_stats(self) -> List[StageStats]:
        """Get statistics for all stages."""
        with self._data_lock:
            return list(self._stage_stats.values())

    def get_doc_type_stats(self, doc_type: str) -> Optional[DocumentTypeStats]:
        """
        Get statistics for a specific document type.

        AC: Access failure rate per document type.
        """
        with self._data_lock:
            return self._doc_type_stats.get(doc_type)

    def get_all_doc_type_stats(self) -> List[DocumentTypeStats]:
        """Get statistics for all document types."""
        with self._data_lock:
            return list(self._doc_type_stats.values())

    def get_failure_rates(self) -> Dict[str, float]:
        """
        Get failure rates by document type.

        AC: Log "Extraction Failure Rate" per document type.

        Returns:
            Dictionary mapping document_type to failure_rate percentage.
        """
        with self._data_lock:
            return {
                stats.document_type: stats.failure_rate
                for stats in self._doc_type_stats.values()
            }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary metrics.
        """
        with self._data_lock:
            total_calls = sum(s.total_calls for s in self._stage_stats.values())
            total_success = sum(s.success_count for s in self._stage_stats.values())
            total_tokens = sum(s.total_tokens for s in self._stage_stats.values())
            total_latency = sum(s.total_latency_ms for s in self._stage_stats.values())

            success_rate = 100.0
            avg_latency = 0.0
            if total_calls > 0:
                success_rate = (total_success / total_calls) * 100.0
                avg_latency = total_latency / total_calls

            return {
                "session_active": self._session_active,
                "total_metrics": len(self._metrics),
                "total_calls": total_calls,
                "success_count": total_success,
                "success_rate": success_rate,
                "total_tokens": total_tokens,
                "avg_latency_ms": avg_latency,
                "stages_tracked": len(self._stage_stats),
                "document_types_tracked": len(self._doc_type_stats),
            }

    def format_cli_summary(self) -> str:
        """
        Format summary for CLI output.

        AC: Add CLI summary output.
        Example: "Extraction complete: 95% success, Avg Latency 1.2s"

        Returns:
            Formatted summary string.
        """
        summary = self.get_summary()
        success_rate = summary["success_rate"]
        avg_latency_s = summary["avg_latency_ms"] / 1000.0

        return (
            f"Extraction complete: {success_rate:.0f}% success, "
            f"Avg Latency {avg_latency_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# Telemetry Interceptor
# ---------------------------------------------------------------------------


class TelemetryInterceptor(IFInterceptor):
    """
    Interceptor that automatically records health metrics.

    Creates TelemetryStage integration via interceptor pattern.
    Integrates with IFExtractionStats to record metrics automatically.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        stats: Optional[IFExtractionStats] = None,
        extract_doc_type: bool = True,
    ):
        """
        Initialize telemetry interceptor.

        Args:
            stats: Optional stats collector (uses singleton if not provided).
            extract_doc_type: Whether to extract document type from metadata.
        """
        self._stats = stats or IFExtractionStats()
        self._extract_doc_type = extract_doc_type
        self._stage_starts: Dict[str, float] = {}

    def _get_doc_type(self, artifact: IFArtifact) -> str:
        """Extract document type from artifact metadata."""
        if not self._extract_doc_type:
            return "unknown"

        # Try common metadata keys
        metadata = artifact.metadata
        for key in ("mime_type", "document_type", "type", "format"):
            if key in metadata:
                value = metadata[key]
                if isinstance(value, str):
                    # Extract type from mime_type (e.g., "application/pdf" -> "pdf")
                    if "/" in value:
                        return value.split("/")[-1]
                    return value
        return "unknown"

    def pre_stage(
        self, stage_name: str, artifact: IFArtifact, document_id: str
    ) -> None:
        """Record stage start time."""
        key = f"{document_id}:{stage_name}"
        self._stage_starts[key] = time.perf_counter()

    def post_stage(
        self,
        stage_name: str,
        artifact: IFArtifact,
        document_id: str,
        duration_ms: float,
    ) -> None:
        """Record successful stage completion."""
        doc_type = self._get_doc_type(artifact)

        # Extract token count from metadata if available
        token_count = artifact.metadata.get("token_count", 0)
        if not isinstance(token_count, int):
            token_count = 0

        # Extract memory usage if available
        memory_mb = artifact.metadata.get("memory_mb")
        if memory_mb is not None and not isinstance(memory_mb, (int, float)):
            memory_mb = None

        self._stats.emit(
            stage_name=stage_name,
            document_id=document_id,
            latency_ms=duration_ms,
            success_status=True,
            token_count=token_count,
            document_type=doc_type,
            memory_mb=memory_mb,
        )

    def on_error(
        self, stage_name: str, artifact: IFArtifact, document_id: str, error: Exception
    ) -> None:
        """Record stage failure."""
        key = f"{document_id}:{stage_name}"
        start_time = self._stage_starts.pop(key, None)

        latency_ms = 0.0
        if start_time is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000

        doc_type = self._get_doc_type(artifact)

        self._stats.emit(
            stage_name=stage_name,
            document_id=document_id,
            latency_ms=latency_ms,
            success_status=False,
            document_type=doc_type,
            error_message=str(error),
        )

    def on_pipeline_start(self, document_id: str, stage_count: int) -> None:
        """Optional: Log pipeline start."""
        logger.debug(
            f"Telemetry: Pipeline started for {document_id} ({stage_count} stages)"
        )

    def on_pipeline_end(
        self, document_id: str, success: bool, total_duration_ms: float
    ) -> None:
        """Optional: Log pipeline completion."""
        status = "completed" if success else "failed"
        logger.debug(
            f"Telemetry: Pipeline {status} for {document_id} "
            f"in {total_duration_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def get_extraction_stats() -> IFExtractionStats:
    """Get the singleton extraction stats collector."""
    return IFExtractionStats()


def create_telemetry_interceptor(
    stats: Optional[IFExtractionStats] = None,
) -> TelemetryInterceptor:
    """
    Create a telemetry interceptor for pipeline monitoring.

    Factory function for easy integration.

    Args:
        stats: Optional stats collector (uses singleton if not provided).

    Returns:
        TelemetryInterceptor instance.
    """
    return TelemetryInterceptor(stats=stats)
