"""
Data models for content hash verification.

Defines enums and dataclasses for hash algorithms, verification status,
content hashes, and verification results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class HashAlgorithm(Enum):
    """Supported hash algorithms."""

    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA384 = "sha384"
    SHA224 = "sha224"
    SHA1 = "sha1"  # Legacy, not recommended
    MD5 = "md5"  # Legacy, not recommended
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class VerificationStatus(Enum):
    """Status of hash verification."""

    VERIFIED = "verified"  # Hash matches
    MODIFIED = "modified"  # Hash doesn't match
    MISSING = "missing"  # No stored hash to compare
    ERROR = "error"  # Error during verification
    UNKNOWN = "unknown"  # Couldn't determine


@dataclass
class ContentHash:
    """A content hash with metadata."""

    algorithm: HashAlgorithm
    hash_value: str  # Hex-encoded hash
    content_size: int = 0
    created_at: str = ""
    # Optional metadata
    source_path: Optional[str] = None
    content_type: Optional[str] = None
    chunk_size: Optional[int] = None  # If hashed in chunks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "hash_value": self.hash_value,
            "content_size": self.content_size,
            "created_at": self.created_at,
            "source_path": self.source_path,
            "content_type": self.content_type,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentHash":
        return cls(
            algorithm=HashAlgorithm(data["algorithm"]),
            hash_value=data["hash_value"],
            content_size=data.get("content_size", 0),
            created_at=data.get("created_at", ""),
            source_path=data.get("source_path"),
            content_type=data.get("content_type"),
            chunk_size=data.get("chunk_size"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContentHash):
            return False
        return (
            self.algorithm == other.algorithm
            and self.hash_value.lower() == other.hash_value.lower()
        )


@dataclass
class MultiHash:
    """Multiple hashes for the same content."""

    hashes: Dict[HashAlgorithm, ContentHash] = field(default_factory=dict)
    content_size: int = 0
    source_path: Optional[str] = None
    created_at: str = ""

    def get_hash(self, algorithm: HashAlgorithm) -> Optional[ContentHash]:
        """Get hash by algorithm."""
        return self.hashes.get(algorithm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hashes": {a.value: h.to_dict() for a, h in self.hashes.items()},
            "content_size": self.content_size,
            "source_path": self.source_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiHash":
        hashes = {
            HashAlgorithm(a): ContentHash.from_dict(h)
            for a, h in data.get("hashes", {}).items()
        }
        return cls(
            hashes=hashes,
            content_size=data.get("content_size", 0),
            source_path=data.get("source_path"),
            created_at=data.get("created_at", ""),
        )


@dataclass
class VerificationResult:
    """Result of hash verification."""

    status: VerificationStatus
    algorithm: HashAlgorithm
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    content_size: int = 0
    verified_at: str = ""
    error_message: Optional[str] = None
    # Diff info if modified
    size_changed: bool = False
    original_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "algorithm": self.algorithm.value,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "content_size": self.content_size,
            "verified_at": self.verified_at,
            "error_message": self.error_message,
            "size_changed": self.size_changed,
            "original_size": self.original_size,
        }

    @property
    def is_verified(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.VERIFIED


@dataclass
class HashRecord:
    """A stored hash record for a file/content."""

    record_id: str
    source_identifier: str  # Path, URL, or content ID
    hashes: MultiHash
    created_at: str
    updated_at: str
    verification_count: int = 0
    last_verified_at: Optional[str] = None
    last_verification_status: Optional[VerificationStatus] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_identifier": self.source_identifier,
            "hashes": self.hashes.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verification_count": self.verification_count,
            "last_verified_at": self.last_verified_at,
            "last_verification_status": (
                self.last_verification_status.value
                if self.last_verification_status
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HashRecord":
        status = data.get("last_verification_status")
        return cls(
            record_id=data["record_id"],
            source_identifier=data["source_identifier"],
            hashes=MultiHash.from_dict(data["hashes"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            verification_count=data.get("verification_count", 0),
            last_verified_at=data.get("last_verified_at"),
            last_verification_status=VerificationStatus(status) if status else None,
            metadata=data.get("metadata", {}),
        )
