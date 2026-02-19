"""
Content hash verification for detecting modifications.

Provides cryptographic hash generation and verification for:
- Files and content blobs
- Incremental hashing for large files
- Multiple hash algorithms
- Hash comparison and integrity verification
- Content fingerprinting

This module has been split into focused submodules for maintainability
while maintaining 100% backward compatibility via re-exports.
"""

# Models
from ingestforge.ingest.content_hash_verifier.models import (
    ContentHash,
    HashAlgorithm,
    HashRecord,
    MultiHash,
    VerificationResult,
    VerificationStatus,
)

# Hasher
from ingestforge.ingest.content_hash_verifier.hasher import ContentHasher

# Verifier and storage
from ingestforge.ingest.content_hash_verifier.verifier import (
    ContentVerifier,
    HashStore,
    get_store,
    # Convenience functions
    hash_content,
    quick_hash,
    store_hash,
    verify_content,
    verify_stored,
)

__all__ = [
    # Enums
    "HashAlgorithm",
    "VerificationStatus",
    # Models
    "ContentHash",
    "MultiHash",
    "VerificationResult",
    "HashRecord",
    # Classes
    "ContentHasher",
    "ContentVerifier",
    "HashStore",
    # Functions
    "get_store",
    "hash_content",
    "quick_hash",
    "verify_content",
    "store_hash",
    "verify_stored",
]
