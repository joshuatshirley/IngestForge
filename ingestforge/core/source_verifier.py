"""
Source File Verification for Audit Trail.

Verifies that source documents referenced by indexed chunks still exist
and optionally validates their content hasn't changed via hash comparison.

Architecture Context
--------------------
Source verification ensures audit trail integrity:

    Indexed Chunks → SourceVerifier → VerificationResult
                                      ├── exists: True/False
                                      ├── hash_verified: True/False/None
                                      └── status: verified/missing/modified

This module leverages the existing ContentVerifier for hash operations.

Verification Statuses
---------------------
| Status    | Meaning                                          |
|-----------|--------------------------------------------------|
| verified  | File exists and hash matches (if checked)        |
| missing   | File not found at the indexed path               |
| modified  | File exists but hash doesn't match               |
| unknown   | Could not determine status (error during check)  |
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from ingestforge.core.logging import get_logger
from ingestforge.ingest.content_hash_verifier import (
    ContentHasher,
    ContentVerifier,
    ContentHash,
    HashAlgorithm,
    VerificationStatus as HashVerificationStatus,
)

logger = get_logger(__name__)


class VerificationStatus(str, Enum):
    """Status of source file verification."""

    VERIFIED = "verified"  # File exists and hash matches (if checked)
    MISSING = "missing"  # File not found
    MODIFIED = "modified"  # File exists but hash changed
    UNKNOWN = "unknown"  # Error during verification


@dataclass
class SourceVerificationResult:
    """
    Result of verifying a single source file.

    Attributes:
        file_path: Path to the source file
        exists: Whether the file exists on disk
        hash_verified: True if hash matches, False if different, None if not checked
        status: Overall verification status
        chunk_count: Number of chunks referencing this file
        error_message: Error message if status is UNKNOWN
        verified_at: Timestamp of verification
    """

    file_path: str
    exists: bool
    hash_verified: Optional[bool]
    status: VerificationStatus
    chunk_count: int = 0
    error_message: Optional[str] = None
    verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    @property
    def status_symbol(self) -> str:
        """Get a symbol representing the status."""
        symbols = {
            VerificationStatus.VERIFIED: "+",
            VerificationStatus.MISSING: "x",
            VerificationStatus.MODIFIED: "!",
            VerificationStatus.UNKNOWN: "?",
        }
        return symbols.get(self.status, "?")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "exists": self.exists,
            "hash_verified": self.hash_verified,
            "status": self.status.value,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "verified_at": self.verified_at,
        }


@dataclass
class VerificationReport:
    """
    Summary report of source verification.

    Attributes:
        total_files: Total number of unique source files
        verified: Number of verified files
        missing: Number of missing files
        modified: Number of modified files
        unknown: Number of files with unknown status
        results: Individual verification results
        verified_at: Timestamp of verification
    """

    total_files: int
    verified: int
    missing: int
    modified: int
    unknown: int
    results: List[SourceVerificationResult]
    verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    @property
    def all_verified(self) -> bool:
        """Check if all sources are verified."""
        return self.missing == 0 and self.modified == 0 and self.unknown == 0

    @property
    def missing_files(self) -> List[SourceVerificationResult]:
        """Get list of missing files."""
        return [r for r in self.results if r.status == VerificationStatus.MISSING]

    @property
    def modified_files(self) -> List[SourceVerificationResult]:
        """Get list of modified files."""
        return [r for r in self.results if r.status == VerificationStatus.MODIFIED]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "verified": self.verified,
            "missing": self.missing,
            "modified": self.modified,
            "unknown": self.unknown,
            "all_verified": self.all_verified,
            "results": [r.to_dict() for r in self.results],
            "verified_at": self.verified_at,
        }


class SourceVerifier:
    """
    Verify source files for indexed content.

    Checks that source documents still exist at their indexed paths
    and optionally verifies content hasn't changed via hash comparison.

    Example:
        verifier = SourceVerifier()

        # Verify a single source
        result = verifier.verify_source(
            file_path="/docs/thesis.pdf",
            content_hash="abc123...",
        )
        if result.status == VerificationStatus.MISSING:
            print("Source file not found!")

        # Verify all sources from a repository
        report = verifier.verify_all_sources(sources)
        print(f"Verified: {report.verified}/{report.total_files}")
    """

    def __init__(
        self,
        check_hash: bool = False,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ):
        """
        Initialize the verifier.

        Args:
            check_hash: If True, verify file content hash
            algorithm: Hash algorithm to use for verification
        """
        self.check_hash = check_hash
        self.algorithm = algorithm
        self._hasher = ContentHasher()
        self._verifier = ContentVerifier(self._hasher)

    def verify_source(
        self,
        file_path: str,
        content_hash: Optional[str] = None,
        chunk_count: int = 0,
    ) -> SourceVerificationResult:
        """
        Verify a single source file.

        Args:
            file_path: Path to the source file
            content_hash: Expected hash (optional, for hash verification)
            chunk_count: Number of chunks from this source

        Returns:
            SourceVerificationResult with verification status
        """
        path = Path(file_path)

        try:
            if not path.exists():
                return self._create_missing_result(file_path, chunk_count)

            # Check for modified file (returns result if modified, None otherwise)
            modified_result = self._check_if_modified(
                path, file_path, content_hash, chunk_count
            )
            if modified_result:
                return modified_result

            # File exists and is not modified
            hash_verified = self._get_hash_verification_status(path, content_hash)

            return SourceVerificationResult(
                file_path=file_path,
                exists=True,
                hash_verified=hash_verified,
                status=VerificationStatus.VERIFIED,
                chunk_count=chunk_count,
            )

        except Exception as e:
            logger.error(f"Error verifying source {file_path}: {e}")
            return SourceVerificationResult(
                file_path=file_path,
                exists=False,
                hash_verified=None,
                status=VerificationStatus.UNKNOWN,
                chunk_count=chunk_count,
                error_message=str(e),
            )

    def _create_missing_result(
        self, file_path: str, chunk_count: int
    ) -> SourceVerificationResult:
        """Create result for missing file."""
        return SourceVerificationResult(
            file_path=file_path,
            exists=False,
            hash_verified=None,
            status=VerificationStatus.MISSING,
            chunk_count=chunk_count,
        )

    def _check_if_modified(
        self,
        path: Path,
        file_path: str,
        content_hash: Optional[str],
        chunk_count: int,
    ) -> Optional[SourceVerificationResult]:
        """Check if file has been modified. Returns result if modified, None otherwise."""
        if not (self.check_hash and content_hash):
            return None

        try:
            expected = ContentHash(
                algorithm=self.algorithm, hash_value=content_hash, content_size=0
            )
            result = self._verifier.verify_file(path, expected)

            if result.status == HashVerificationStatus.MODIFIED:
                return SourceVerificationResult(
                    file_path=file_path,
                    exists=True,
                    hash_verified=False,
                    status=VerificationStatus.MODIFIED,
                    chunk_count=chunk_count,
                )
        except Exception as e:
            logger.warning(f"Hash verification failed for {file_path}: {e}")

        return None

    def _get_hash_verification_status(
        self, path: Path, content_hash: Optional[str]
    ) -> Optional[bool]:
        """Get hash verification status. Returns True if verified, None if not checked or error."""
        if not (self.check_hash and content_hash):
            return None

        try:
            expected = ContentHash(
                algorithm=self.algorithm, hash_value=content_hash, content_size=0
            )
            result = self._verifier.verify_file(path, expected)
            return result.status == HashVerificationStatus.VERIFIED
        except Exception as e:
            logger.warning(f"Hash verification failed: {e}")
            return None

    def verify_all_sources(
        self,
        sources: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VerificationReport:
        """
        Verify all source files from a list of chunks/sources.

        Rule #1: Reduced nesting with helper methods

        Args:
            sources: List of dicts with 'source_file' or 'file_path' and optional 'content_hash'
            progress_callback: Optional callback(current, total) for progress

        Returns:
            VerificationReport with summary and individual results
        """
        file_chunks = self._group_chunks_by_file(sources)

        total = len(file_chunks)
        results = []
        status_counts = {"verified": 0, "missing": 0, "modified": 0, "unknown": 0}
        for i, (file_path, chunks) in enumerate(file_chunks.items()):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self._verify_file_chunks(file_path, chunks)
            results.append(result)
            self._update_status_counts(result.status, status_counts)

        return VerificationReport(
            total_files=total,
            verified=status_counts["verified"],
            missing=status_counts["missing"],
            modified=status_counts["modified"],
            unknown=status_counts["unknown"],
            results=results,
        )

    def _group_chunks_by_file(
        self, sources: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group chunks by source file path.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        file_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for source in sources:
            file_path = source.get("source_file") or source.get("file_path")
            if not file_path:
                continue

            file_chunks.setdefault(file_path, []).append(source)

        return file_chunks

    def _verify_file_chunks(
        self, file_path: str, chunks: List[Dict[str, Any]]
    ) -> SourceVerificationResult:
        """
        Verify a single file with its chunks.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        # Get content hash from first chunk (if available)
        content_hash = self._extract_content_hash(chunks)

        return self.verify_source(
            file_path=file_path,
            content_hash=content_hash,
            chunk_count=len(chunks),
        )

    def _extract_content_hash(self, chunks: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract content hash from chunks.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        for chunk in chunks:
            source_loc = chunk.get("source_location", {})
            content_hash = source_loc.get("content_hash")
            if content_hash:
                return content_hash
        return None

    def _update_status_counts(
        self, status: VerificationStatus, counts: Dict[str, int]
    ) -> None:
        """
        Update status counts based on verification result.

        Rule #1: Dictionary dispatch eliminates if/elif chain
        Rule #4: Function <60 lines
        """
        status_map = {
            VerificationStatus.VERIFIED: "verified",
            VerificationStatus.MISSING: "missing",
            VerificationStatus.MODIFIED: "modified",
        }

        key = status_map.get(status, "unknown")
        counts[key] += 1

    def verify_from_repository(
        self,
        repository: Any,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VerificationReport:
        """
        Verify all sources from a repository.

        Rule #1: Reduced nesting with helper method

        Args:
            repository: Storage repository with chunks
            progress_callback: Optional progress callback

        Returns:
            VerificationReport with all verification results
        """
        # Get all chunks from repository
        chunks = repository.get_all() if hasattr(repository, "get_all") else []
        sources = self._convert_chunks_to_dicts(chunks)

        return self.verify_all_sources(sources, progress_callback)

    def _convert_chunks_to_dicts(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert chunks to dict format.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        sources = []
        for chunk in chunks:
            chunk_dict = self._chunk_to_dict(chunk)
            if chunk_dict:
                sources.append(chunk_dict)

        return sources

    def _chunk_to_dict(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """
        Convert single chunk to dict.

        Rule #1: Dictionary dispatch eliminates if/elif chain
        Rule #4: Function <60 lines
        """
        if hasattr(chunk, "to_dict"):
            return chunk.to_dict()
        if hasattr(chunk, "__dict__"):
            return vars(chunk)
        if isinstance(chunk, dict):
            return chunk

        return None


def verify_source_file(
    file_path: str,
    content_hash: Optional[str] = None,
    check_hash: bool = False,
) -> SourceVerificationResult:
    """
    Convenience function to verify a single source file.

    Args:
        file_path: Path to the source file
        content_hash: Expected hash for verification
        check_hash: Whether to verify hash

    Returns:
        SourceVerificationResult
    """
    verifier = SourceVerifier(check_hash=check_hash)
    return verifier.verify_source(file_path, content_hash)


def verify_sources(
    sources: List[Dict[str, Any]],
    check_hash: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VerificationReport:
    """
    Convenience function to verify multiple source files.

    Args:
        sources: List of source dicts with file_path/source_file
        check_hash: Whether to verify hashes
        progress_callback: Optional progress callback

    Returns:
        VerificationReport with summary
    """
    verifier = SourceVerifier(check_hash=check_hash)
    return verifier.verify_all_sources(sources, progress_callback)
