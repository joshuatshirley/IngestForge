"""
Content verification and hash storage.

Provides hash verification, persistent storage of hash records,
and convenience functions for common operations.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ingestforge.ingest.content_hash_verifier.hasher import ContentHasher
from ingestforge.ingest.content_hash_verifier.models import (
    ContentHash,
    HashAlgorithm,
    HashRecord,
    MultiHash,
    VerificationResult,
    VerificationStatus,
)


class ContentVerifier:
    """
    Verify content integrity through hash comparison.
    """

    def __init__(self, hasher: Optional[ContentHasher] = None) -> None:
        self.hasher = hasher or ContentHasher()

    def verify_bytes(
        self,
        data: bytes,
        expected: ContentHash,
    ) -> VerificationResult:
        """
        Verify bytes against expected hash.

        Args:
            data: Bytes to verify
            expected: Expected hash

        Returns:
            VerificationResult
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            actual = self.hasher.hash_bytes(data, [expected.algorithm])
            actual_hash = actual.get_hash(expected.algorithm)

            if not actual_hash:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    algorithm=expected.algorithm,
                    expected_hash=expected.hash_value,
                    verified_at=now,
                    error_message="Failed to compute hash",
                )

            if actual_hash.hash_value.lower() == expected.hash_value.lower():
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    algorithm=expected.algorithm,
                    expected_hash=expected.hash_value,
                    actual_hash=actual_hash.hash_value,
                    content_size=len(data),
                    verified_at=now,
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.MODIFIED,
                    algorithm=expected.algorithm,
                    expected_hash=expected.hash_value,
                    actual_hash=actual_hash.hash_value,
                    content_size=len(data),
                    verified_at=now,
                    size_changed=len(data) != expected.content_size,
                    original_size=expected.content_size,
                )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                algorithm=expected.algorithm,
                expected_hash=expected.hash_value,
                verified_at=now,
                error_message=str(e),
            )

    def verify_file(
        self,
        file_path: Path,
        expected: ContentHash,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VerificationResult:
        """
        Verify file against expected hash.

        Rule #4: Reduced from 62 → 35 lines via helper extraction
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not file_path.exists():
            return self._create_error_result(
                expected, now, f"File not found: {file_path}"
            )

        try:
            actual = self.hasher.hash_file(
                file_path, [expected.algorithm], progress_callback
            )
            actual_hash = actual.get_hash(expected.algorithm)
            file_size = actual.content_size

            if not actual_hash:
                return self._create_error_result(
                    expected, now, "Failed to compute hash"
                )
            return self._create_comparison_result(
                expected, actual_hash.hash_value, file_size, now
            )

        except Exception as e:
            return self._create_error_result(expected, now, str(e))

    def _create_comparison_result(
        self, expected: ContentHash, actual_hash: str, file_size: int, now: str
    ) -> VerificationResult:
        """
        Create verification result from hash comparison.

        Rule #4: Extracted to reduce verify_file() size
        """
        if actual_hash.lower() == expected.hash_value.lower():
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                algorithm=expected.algorithm,
                expected_hash=expected.hash_value,
                actual_hash=actual_hash,
                content_size=file_size,
                verified_at=now,
            )
        else:
            return VerificationResult(
                status=VerificationStatus.MODIFIED,
                algorithm=expected.algorithm,
                expected_hash=expected.hash_value,
                actual_hash=actual_hash,
                content_size=file_size,
                verified_at=now,
                size_changed=file_size != expected.content_size,
                original_size=expected.content_size,
            )

    def _create_error_result(
        self, expected: ContentHash, now: str, error_message: str
    ) -> VerificationResult:
        """
        Create error verification result.

        Rule #4: Extracted to reduce verify_file() size
        """
        return VerificationResult(
            status=VerificationStatus.ERROR,
            algorithm=expected.algorithm,
            expected_hash=expected.hash_value,
            verified_at=now,
            error_message=error_message,
        )


class HashStore:
    """
    Store and manage content hashes.
    """

    def __init__(self, store_path: Optional[Path] = None) -> None:
        self.store_path = (
            store_path or Path.home() / ".splitanalyze" / "hash_store.json"
        )
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, HashRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load records from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    data = json.load(f)
                self._records = {
                    k: HashRecord.from_dict(v)
                    for k, v in data.get("records", {}).items()
                }
            except Exception:
                self._records = {}

    def _save(self) -> None:
        """Save records to disk."""
        data = {
            "records": {k: v.to_dict() for k, v in self._records.items()},
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, identifier: str) -> str:
        """Generate record ID from identifier."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def store(
        self,
        identifier: str,
        hashes: MultiHash,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HashRecord:
        """
        Store hash record.

        Args:
            identifier: Source identifier (path, URL, etc.)
            hashes: Hashes to store
            metadata: Optional metadata

        Returns:
            Created/updated HashRecord
        """
        record_id = self._generate_id(identifier)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if record_id in self._records:
            # Update existing
            record = self._records[record_id]
            record.hashes = hashes
            record.updated_at = now
            if metadata:
                record.metadata.update(metadata)
        else:
            # Create new
            record = HashRecord(
                record_id=record_id,
                source_identifier=identifier,
                hashes=hashes,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
            self._records[record_id] = record

        self._save()
        return record

    def get(self, identifier: str) -> Optional[HashRecord]:
        """Get hash record by identifier."""
        record_id = self._generate_id(identifier)
        return self._records.get(record_id)

    def update_verification(
        self,
        identifier: str,
        status: VerificationStatus,
    ) -> Optional[HashRecord]:
        """Update verification status for a record."""
        record = self.get(identifier)
        if not record:
            return None

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record.verification_count += 1
        record.last_verified_at = now
        record.last_verification_status = status
        record.updated_at = now

        self._save()
        return record


class _HashingSingletons:
    """Singleton holder for hashing components.

    Rule #6: Encapsulates singleton state in smallest scope.
    """

    _hasher: ContentHasher = ContentHasher()
    _verifier: Optional[ContentVerifier] = None
    _store: Optional[HashStore] = None

    @classmethod
    def get_hasher(cls) -> ContentHasher:
        """Get the singleton hasher."""
        return cls._hasher

    @classmethod
    def get_verifier(cls) -> ContentVerifier:
        """Get the singleton verifier."""
        if cls._verifier is None:
            cls._verifier = ContentVerifier(cls._hasher)
        return cls._verifier

    @classmethod
    def get_store(cls) -> HashStore:
        """Get the singleton hash store."""
        if cls._store is None:
            cls._store = HashStore()
        return cls._store


def get_store() -> HashStore:
    """Get singleton hash store."""
    return _HashingSingletons.get_store()


# Convenience functions
def hash_content(
    data: Union[bytes, str, Path],
    algorithms: Optional[List[HashAlgorithm]] = None,
) -> MultiHash:
    """Hash content with multiple algorithms."""
    hasher = _HashingSingletons.get_hasher()
    if isinstance(data, Path):
        return hasher.hash_file(data, algorithms)
    elif isinstance(data, str):
        return hasher.hash_string(data, algorithms=algorithms)
    else:
        return hasher.hash_bytes(data, algorithms)


def quick_hash(
    data: Union[bytes, str, Path],
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
) -> str:
    """Quick single-algorithm hash."""
    return _HashingSingletons.get_hasher().quick_hash(data, algorithm)


def verify_content(
    data: Union[bytes, Path],
    expected: ContentHash,
) -> VerificationResult:
    """Verify content against expected hash."""
    verifier = _HashingSingletons.get_verifier()
    if isinstance(data, Path):
        return verifier.verify_file(data, expected)
    return verifier.verify_bytes(data, expected)


def store_hash(
    identifier: str,
    data: Union[bytes, str, Path],
    algorithms: Optional[List[HashAlgorithm]] = None,
) -> HashRecord:
    """Hash content and store the result."""
    hashes = hash_content(data, algorithms)
    return get_store().store(identifier, hashes)


def verify_stored(
    identifier: str,
    data: Union[bytes, Path],
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
) -> VerificationResult:
    """Verify content against stored hash."""
    record = get_store().get(identifier)
    if not record:
        return VerificationResult(
            status=VerificationStatus.MISSING,
            algorithm=algorithm,
            verified_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    expected = record.hashes.get_hash(algorithm)
    if not expected:
        return VerificationResult(
            status=VerificationStatus.MISSING,
            algorithm=algorithm,
            verified_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    result = verify_content(data, expected)

    # Update store
    get_store().update_verification(identifier, result.status)

    return result
