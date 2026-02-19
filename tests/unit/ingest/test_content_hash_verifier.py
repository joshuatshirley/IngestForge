"""
Tests for Content Hash Verification.

This module tests cryptographic hash generation and verification for content integrity.

Test Strategy
-------------
- Focus on hash generation, comparison, and verification
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test enums, dataclasses, and hash functions
- Use known hash values for verification

Organization
------------
- TestHashAlgorithm: HashAlgorithm enum
- TestVerificationStatus: VerificationStatus enum
- TestContentHash: ContentHash dataclass
- TestMultiHash: MultiHash dataclass
- TestHashGeneration: hash_content and quick_hash functions
"""

import hashlib


from ingestforge.ingest.content_hash_verifier import (
    HashAlgorithm,
    VerificationStatus,
    ContentHash,
    MultiHash,
    hash_content,
    quick_hash,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestHashAlgorithm:
    """Tests for HashAlgorithm enum.

    Rule #4: Focused test class - tests only HashAlgorithm
    """

    def test_hash_algorithm_sha256(self):
        """Test SHA256 algorithm."""
        assert HashAlgorithm.SHA256.value == "sha256"

    def test_hash_algorithm_sha512(self):
        """Test SHA512 algorithm."""
        assert HashAlgorithm.SHA512.value == "sha512"

    def test_hash_algorithm_md5(self):
        """Test MD5 algorithm (legacy)."""
        assert HashAlgorithm.MD5.value == "md5"

    def test_hash_algorithm_blake2b(self):
        """Test BLAKE2B algorithm."""
        assert HashAlgorithm.BLAKE2B.value == "blake2b"


class TestVerificationStatus:
    """Tests for VerificationStatus enum.

    Rule #4: Focused test class - tests only VerificationStatus
    """

    def test_verification_status_verified(self):
        """Test VERIFIED status."""
        assert VerificationStatus.VERIFIED.value == "verified"

    def test_verification_status_modified(self):
        """Test MODIFIED status."""
        assert VerificationStatus.MODIFIED.value == "modified"

    def test_verification_status_missing(self):
        """Test MISSING status."""
        assert VerificationStatus.MISSING.value == "missing"

    def test_verification_status_error(self):
        """Test ERROR status."""
        assert VerificationStatus.ERROR.value == "error"


class TestContentHash:
    """Tests for ContentHash dataclass.

    Rule #4: Focused test class - tests only ContentHash
    """

    def test_create_content_hash(self):
        """Test creating a ContentHash."""
        content_hash = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
            content_size=1024,
            created_at="2024-01-01",
        )

        assert content_hash.algorithm == HashAlgorithm.SHA256
        assert content_hash.hash_value == "abc123"
        assert content_hash.content_size == 1024

    def test_content_hash_with_metadata(self):
        """Test ContentHash with optional metadata."""
        content_hash = ContentHash(
            algorithm=HashAlgorithm.SHA512,
            hash_value="def456",
            source_path="/path/to/file.txt",
            content_type="text/plain",
        )

        assert content_hash.source_path == "/path/to/file.txt"
        assert content_hash.content_type == "text/plain"

    def test_content_hash_to_dict(self):
        """Test converting ContentHash to dict."""
        content_hash = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
            content_size=1024,
        )

        hash_dict = content_hash.to_dict()

        assert hash_dict["algorithm"] == "sha256"
        assert hash_dict["hash_value"] == "abc123"
        assert hash_dict["content_size"] == 1024

    def test_content_hash_from_dict(self):
        """Test creating ContentHash from dict."""
        hash_dict = {
            "algorithm": "sha256",
            "hash_value": "abc123",
            "content_size": 1024,
            "created_at": "2024-01-01",
        }

        content_hash = ContentHash.from_dict(hash_dict)

        assert content_hash.algorithm == HashAlgorithm.SHA256
        assert content_hash.hash_value == "abc123"
        assert content_hash.content_size == 1024

    def test_content_hash_equality(self):
        """Test ContentHash equality comparison."""
        hash1 = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
        )
        hash2 = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
        )

        assert hash1 == hash2

    def test_content_hash_equality_case_insensitive(self):
        """Test ContentHash equality is case-insensitive."""
        hash1 = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="ABC123",
        )
        hash2 = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
        )

        assert hash1 == hash2

    def test_content_hash_inequality(self):
        """Test ContentHash inequality."""
        hash1 = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
        )
        hash2 = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="def456",
        )

        assert hash1 != hash2


class TestMultiHash:
    """Tests for MultiHash dataclass.

    Rule #4: Focused test class - tests only MultiHash
    """

    def test_create_multi_hash(self):
        """Test creating a MultiHash."""
        multi_hash = MultiHash(
            content_size=1024,
            source_path="/path/to/file.txt",
        )

        assert multi_hash.content_size == 1024
        assert multi_hash.source_path == "/path/to/file.txt"
        assert len(multi_hash.hashes) == 0

    def test_multi_hash_with_hashes(self):
        """Test MultiHash with multiple hashes."""
        sha256_hash = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
        )
        sha512_hash = ContentHash(
            algorithm=HashAlgorithm.SHA512,
            hash_value="def456",
        )

        multi_hash = MultiHash(
            hashes={
                HashAlgorithm.SHA256: sha256_hash,
                HashAlgorithm.SHA512: sha512_hash,
            }
        )

        assert len(multi_hash.hashes) == 2
        assert HashAlgorithm.SHA256 in multi_hash.hashes
        assert HashAlgorithm.SHA512 in multi_hash.hashes

    def test_multi_hash_get_hash(self):
        """Test getting hash by algorithm from MultiHash."""
        sha256_hash = ContentHash(
            algorithm=HashAlgorithm.SHA256,
            hash_value="abc123",
        )

        multi_hash = MultiHash(hashes={HashAlgorithm.SHA256: sha256_hash})

        retrieved = multi_hash.get_hash(HashAlgorithm.SHA256)

        assert retrieved is not None
        assert retrieved.hash_value == "abc123"

    def test_multi_hash_get_missing_hash(self):
        """Test getting non-existent hash returns None."""
        multi_hash = MultiHash()

        retrieved = multi_hash.get_hash(HashAlgorithm.SHA256)

        assert retrieved is None


class TestHashGeneration:
    """Tests for hash generation functions.

    Rule #4: Focused test class - tests hash generation only
    """

    def test_hash_content_with_string(self):
        """Test hashing string content."""
        content = "Hello, world!"

        result = hash_content(content)

        # hash_content returns MultiHash
        assert isinstance(result, MultiHash)
        sha256_hash = result.get_hash(HashAlgorithm.SHA256)
        assert sha256_hash is not None

        # Known SHA256 hash of "Hello, world!"
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert sha256_hash.hash_value == expected

    def test_hash_content_with_bytes(self):
        """Test hashing bytes content."""
        content = b"Hello, world!"

        result = hash_content(content)

        assert isinstance(result, MultiHash)
        sha256_hash = result.get_hash(HashAlgorithm.SHA256)
        assert sha256_hash is not None

        expected = hashlib.sha256(content).hexdigest()
        assert sha256_hash.hash_value == expected

    def test_hash_content_with_file(self, temp_dir):
        """Test hashing file content."""
        test_file = temp_dir / "test.txt"
        content = "Hello, world!"
        test_file.write_text(content)

        result = hash_content(test_file)

        assert isinstance(result, MultiHash)
        sha256_hash = result.get_hash(HashAlgorithm.SHA256)
        assert sha256_hash is not None

        expected = hashlib.sha256(content.encode()).hexdigest()
        assert sha256_hash.hash_value == expected

    def test_quick_hash_with_string(self):
        """Test quick hashing string content."""
        content = "Hello, world!"

        result = quick_hash(content)

        # Quick hash should return a hash (implementation may vary)
        assert result is not None
        assert len(result) > 0

    def test_quick_hash_with_bytes(self):
        """Test quick hashing bytes content."""
        content = b"Hello, world!"

        result = quick_hash(content)

        assert result is not None
        assert len(result) > 0

    def test_quick_hash_deterministic(self):
        """Test quick hash is deterministic."""
        content = "Same content"

        result1 = quick_hash(content)
        result2 = quick_hash(content)

        assert result1 == result2

    def test_hash_different_content_produces_different_hashes(self):
        """Test different content produces different hashes."""
        content1 = "Hello, world!"
        content2 = "Goodbye, world!"

        hash1 = hash_content(content1)
        hash2 = hash_content(content2)

        assert hash1 != hash2


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - HashAlgorithm enum: 4 tests (SHA256, SHA512, MD5, BLAKE2B)
    - VerificationStatus enum: 4 tests (VERIFIED, MODIFIED, MISSING, ERROR)
    - ContentHash dataclass: 7 tests (creation, metadata, to_dict, from_dict, equality)
    - MultiHash dataclass: 4 tests (creation, with hashes, get_hash, missing)
    - Hash generation: 7 tests (string, bytes, file, quick_hash, deterministic)

    Total: 26 tests

Design Decisions:
    1. Focus on hash generation and dataclass behavior
    2. Use known hash values for verification
    3. Test serialization (to_dict/from_dict)
    4. Test equality comparison (case-insensitive)
    5. Use temp files for file hashing tests
    6. Simple, clear tests that verify hashing works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - HashAlgorithm enum values
    - VerificationStatus enum values
    - ContentHash creation, serialization, equality
    - MultiHash creation and hash retrieval
    - String content hashing
    - Bytes content hashing
    - File content hashing
    - Quick hash generation
    - Hash determinism and uniqueness

Justification:
    - Content hashing is critical for integrity verification
    - Cryptographic hashes must be deterministic
    - Serialization needed for storage
    - Equality comparison must be case-insensitive (hex values)
    - Multiple algorithms supported for flexibility
    - Simple tests verify hashing system works correctly
"""
