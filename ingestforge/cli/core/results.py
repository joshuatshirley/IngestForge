"""Standard result types for CLI operations.

This module provides dataclasses for operation results, following
Commandment #9 (Type Safety) with explicit data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class OperationResult:
    """Base class for operation results.

    Provides common fields for tracking success/failure counts
    and calculating success rates.

    Example:
        result = OperationResult()
        result.items_processed = 100
        result.items_failed = 5
        print(f"Success rate: {result.success_rate:.1%}")
    """

    success: bool = True
    items_processed: int = 0
    items_failed: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as fraction (0.0 to 1.0).

        Returns:
            Success rate, or 0.0 if no items processed
        """
        total = self.total_items
        if total == 0:
            return 0.0
        return self.items_processed / total

    @property
    def total_items(self) -> int:
        """Get total items attempted.

        Returns:
            Sum of processed and failed items
        """
        return self.items_processed + self.items_failed

    @property
    def has_failures(self) -> bool:
        """Check if any items failed.

        Returns:
            True if items_failed > 0
        """
        return self.items_failed > 0


@dataclass
class IngestResult(OperationResult):
    """Result of document ingestion operation.

    Tracks documents ingested, chunks created, and any errors.

    Example:
        result = IngestResult()
        result.documents_ingested = 10
        result.chunks_created = 250
        result.errors.append("Failed to process doc.pdf")
    """

    documents_ingested: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def avg_chunks_per_document(self) -> float:
        """Calculate average chunks per document.

        Returns:
            Average chunks, or 0.0 if no documents
        """
        if self.documents_ingested == 0:
            return 0.0
        return self.chunks_created / self.documents_ingested


@dataclass
class QueryResult(OperationResult):
    """Result of knowledge base query.

    Contains query text, results found, and generated answer.

    Example:
        result = QueryResult(
            query="What is IngestForge?",
            results_found=10,
            sources_cited=3,
            answer="IngestForge is a RAG framework..."
        )
    """

    query: str = ""
    results_found: int = 0
    sources_cited: int = 0
    answer: str = ""


@dataclass
class ResearchResult(OperationResult):
    """Result of research/curation session.

    Tracks source discovery, selection, and processing.

    Example:
        result = ResearchResult(session_id="research-123")
        result.sources_found = 20
        result.sources_selected = 5
        result.sources_downloaded = 4
    """

    session_id: str = ""
    topic: str = ""
    sources_found: int = 0
    sources_selected: int = 0
    sources_downloaded: int = 0
    sources_failed: int = 0
    total_chunks: int = 0

    @property
    def selection_rate(self) -> float:
        """Calculate what fraction of found sources were selected.

        Returns:
            Selection rate (0.0 to 1.0)
        """
        if self.sources_found == 0:
            return 0.0
        return self.sources_selected / self.sources_found

    @property
    def download_success_rate(self) -> float:
        """Calculate download success rate.

        Returns:
            Download success rate (0.0 to 1.0)
        """
        total_attempted = self.sources_downloaded + self.sources_failed
        if total_attempted == 0:
            return 0.0
        return self.sources_downloaded / total_attempted


@dataclass
class ValidationResult(OperationResult):
    """Result of validation/audit operation.

    Tracks items checked and issues found.

    Example:
        result = ValidationResult()
        result.items_checked = 100
        result.issues_found = 5
        result.critical_issues = 1
        result.warnings = 4
    """

    items_checked: int = 0
    issues_found: int = 0
    critical_issues: int = 0
    warnings: int = 0
    details: List[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        """Check if critical issues were found.

        Returns:
            True if critical_issues > 0
        """
        return self.critical_issues > 0

    @property
    def is_clean(self) -> bool:
        """Check if no issues were found.

        Returns:
            True if issues_found == 0
        """
        return self.issues_found == 0


@dataclass
class BatchProcessingResult(OperationResult):
    """Result of batch processing operation.

    Generic result for processing multiple items.

    Example:
        result = BatchProcessingResult(operation_name="Processing PDFs")
        result.items_processed = 95
        result.items_failed = 5
        result.processing_time_seconds = 120.5
    """

    operation_name: str = "Processing"
    processing_time_seconds: float = 0.0
    items_skipped: int = 0

    @property
    def items_per_second(self) -> float:
        """Calculate processing rate.

        Returns:
            Items per second, or 0.0 if no time elapsed
        """
        if self.processing_time_seconds == 0:
            return 0.0
        return self.items_processed / self.processing_time_seconds
