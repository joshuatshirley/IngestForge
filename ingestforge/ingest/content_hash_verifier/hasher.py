"""
Content hashing functionality.

Generates cryptographic hashes for bytes, strings, and files using
multiple algorithms with support for incremental processing.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ingestforge.ingest.content_hash_verifier.models import (
    ContentHash,
    HashAlgorithm,
    MultiHash,
)


class ContentHasher:
    """
    Generate cryptographic hashes for content.
    """

    DEFAULT_ALGORITHMS = [HashAlgorithm.SHA256]
    DEFAULT_CHUNK_SIZE = 8192  # 8KB chunks

    def __init__(
        self,
        algorithms: Optional[List[HashAlgorithm]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Initialize hasher.

        Args:
            algorithms: List of hash algorithms to use
            chunk_size: Chunk size for incremental hashing
        """
        self.algorithms = algorithms or self.DEFAULT_ALGORITHMS
        self.chunk_size = chunk_size

    def hash_bytes(
        self,
        data: bytes,
        algorithms: Optional[List[HashAlgorithm]] = None,
    ) -> MultiHash:
        """
        Hash bytes content.

        Args:
            data: Bytes to hash
            algorithms: Override algorithms (uses instance default if None)

        Returns:
            MultiHash with hashes for all algorithms
        """
        algos = algorithms or self.algorithms
        hashes: Dict[HashAlgorithm, ContentHash] = {}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for algo in algos:
            hasher = self._get_hasher(algo)
            hasher.update(data)
            hash_value = hasher.hexdigest()

            hashes[algo] = ContentHash(
                algorithm=algo,
                hash_value=hash_value,
                content_size=len(data),
                created_at=now,
            )

        return MultiHash(
            hashes=hashes,
            content_size=len(data),
            created_at=now,
        )

    def hash_string(
        self,
        text: str,
        encoding: str = "utf-8",
        algorithms: Optional[List[HashAlgorithm]] = None,
    ) -> MultiHash:
        """Hash string content."""
        return self.hash_bytes(text.encode(encoding), algorithms)

    def hash_file(
        self,
        file_path: Path,
        algorithms: Optional[List[HashAlgorithm]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MultiHash:
        """
        Hash a file incrementally.

        Args:
            file_path: Path to file
            algorithms: Override algorithms
            progress_callback: Optional callback(bytes_processed, total_bytes)

        Returns:
            MultiHash with hashes
        """
        algos = algorithms or self.algorithms
        hashers = {algo: self._get_hasher(algo) for algo in algos}

        file_size = file_path.stat().st_size
        bytes_processed = 0

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break

                for hasher in hashers.values():
                    hasher.update(chunk)

                bytes_processed += len(chunk)
                if progress_callback:
                    progress_callback(bytes_processed, file_size)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        hashes: Dict[HashAlgorithm, ContentHash] = {}

        for algo, hasher in hashers.items():
            hashes[algo] = ContentHash(
                algorithm=algo,
                hash_value=hasher.hexdigest(),
                content_size=file_size,
                created_at=now,
                source_path=str(file_path),
                chunk_size=self.chunk_size,
            )

        return MultiHash(
            hashes=hashes,
            content_size=file_size,
            source_path=str(file_path),
            created_at=now,
        )

    def _get_hasher(self, algorithm: HashAlgorithm) -> Any:
        """Get a hasher instance for the algorithm."""
        return hashlib.new(algorithm.value)

    def quick_hash(
        self,
        data: Union[bytes, str, Path],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> str:
        """
        Quick single-algorithm hash returning just the hex string.

        Args:
            data: Bytes, string, or Path to file
            algorithm: Hash algorithm to use

        Returns:
            Hex-encoded hash string
        """
        if isinstance(data, Path):
            result = self.hash_file(data, [algorithm])
        elif isinstance(data, str):
            result = self.hash_string(data, algorithms=[algorithm])
        else:
            result = self.hash_bytes(data, [algorithm])

        hash_obj = result.get_hash(algorithm)
        if hash_obj is None:
            raise ValueError(f"Hash not found for algorithm {algorithm}")
        return hash_obj.hash_value
