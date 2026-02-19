"""Corpus Packager for portable knowledge base export.

Creates streaming ZIP archives of knowledge base data for sharing.
Supports cross-platform path handling and metadata preservation."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_FILES_PER_ARCHIVE = 10000
MAX_FILE_SIZE_MB = 500
MAX_TOTAL_SIZE_MB = 5000
CHUNK_SIZE = 1024 * 1024  # 1MB streaming chunks


@dataclass
class PackageManifest:
    """Metadata manifest for corpus package."""

    version: str = "1.0"
    created_at: str = ""
    source_platform: str = ""
    file_count: int = 0
    total_size_bytes: int = 0
    storage_type: str = "chromadb"
    includes_embeddings: bool = True
    includes_state: bool = True
    checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "source_platform": self.source_platform,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "storage_type": self.storage_type,
            "includes_embeddings": self.includes_embeddings,
            "includes_state": self.includes_state,
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PackageManifest":
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            source_platform=data.get("source_platform", ""),
            file_count=data.get("file_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            storage_type=data.get("storage_type", "chromadb"),
            includes_embeddings=data.get("includes_embeddings", True),
            includes_state=data.get("includes_state", True),
            checksums=data.get("checksums", {}),
        )


@dataclass
class PackageResult:
    """Result of packaging operation."""

    success: bool
    output_path: Optional[Path] = None
    manifest: Optional[PackageManifest] = None
    error: Optional[str] = None
    files_packaged: int = 0
    bytes_written: int = 0


ProgressCallback = Callable[[int, int, str], None]


class CorpusPackager:
    """Creates portable ZIP archives of knowledge base data.

    Uses streaming I/O to handle large corpora without memory issues.
    Preserves relative paths for cross-platform compatibility.
    """

    def __init__(
        self,
        project_dir: Path,
        include_embeddings: bool = True,
        include_state: bool = True,
    ) -> None:
        """Initialize packager.

        Args:
            project_dir: Project directory containing .ingest/ and .data/
            include_embeddings: Include embedding vectors in package
            include_state: Include pipeline state files
        """
        self.project_dir = project_dir
        self.include_embeddings = include_embeddings
        self.include_state = include_state

    def package(
        self,
        output_path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PackageResult:
        """Create portable corpus package.

        Args:
            output_path: Path for output .zip file
            progress_callback: Optional (current, total, filename) callback

        Returns:
            PackageResult with success status and details
        """
        try:
            files = self._collect_files()
            if not files:
                return PackageResult(success=False, error="No files found to package")

            if len(files) > MAX_FILES_PER_ARCHIVE:
                return PackageResult(
                    success=False,
                    error=f"Too many files: {len(files)} > {MAX_FILES_PER_ARCHIVE}",
                )

            manifest = self._create_manifest(files)
            bytes_written = self._write_archive(
                output_path, files, manifest, progress_callback
            )

            return PackageResult(
                success=True,
                output_path=output_path,
                manifest=manifest,
                files_packaged=len(files),
                bytes_written=bytes_written,
            )
        except Exception as e:
            logger.exception(f"Packaging failed: {e}")
            return PackageResult(success=False, error=str(e))

    def _collect_files(self) -> List[Path]:
        """Collect all files to package.

        Returns:
            List of file paths relative to project directory
        """
        files: List[Path] = []

        # Collect .ingest directory (config and metadata)
        ingest_dir = self.project_dir / ".ingest"
        if ingest_dir.exists():
            files.extend(self._collect_directory(ingest_dir))

        # Collect .data directory (storage)
        data_dir = self.project_dir / ".data"
        if data_dir.exists():
            files.extend(self._collect_directory(data_dir))

        # Include config files
        for config_name in ["ingestforge.yaml", "ingestforge.yml"]:
            config_path = self.project_dir / config_name
            if config_path.exists():
                files.append(config_path)

        return files[:MAX_FILES_PER_ARCHIVE]

    def _collect_directory(self, directory: Path) -> List[Path]:
        """Collect files from directory recursively.

        Args:
            directory: Directory to scan

        Returns:
            List of file paths
        """
        files: List[Path] = []
        count = 0

        for path in directory.rglob("*"):
            if count >= MAX_FILES_PER_ARCHIVE:
                break

            if not path.is_file():
                continue

            # Skip excluded files
            if self._should_exclude(path):
                continue

            # Check file size
            if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Skipping large file: {path}")
                continue

            files.append(path)
            count += 1

        return files

    def _should_exclude(self, path: Path) -> bool:
        """Check if file should be excluded from package.

        Args:
            path: File path to check

        Returns:
            True if file should be excluded
        """
        name = path.name.lower()

        # Exclude lock files
        if name.endswith(".lock"):
            return True

        # Exclude temp files
        if name.startswith(".") or name.endswith(".tmp"):
            return True

        # Conditionally exclude embeddings
        if not self.include_embeddings:
            if "embedding" in name or "vectors" in name:
                return True

        # Conditionally exclude state
        if not self.include_state:
            if "state" in name or "_checkpoint" in name:
                return True

        return False

    def _create_manifest(self, files: List[Path]) -> PackageManifest:
        """Create package manifest.

        Args:
            files: List of files being packaged

        Returns:
            PackageManifest with metadata
        """
        import platform

        total_size = sum(f.stat().st_size for f in files)
        checksums = {}

        # Calculate checksums for key files (not all to save time)
        key_patterns = ["config", "metadata", ".yaml", ".json"]
        for f in files[:100]:
            if any(p in f.name.lower() for p in key_patterns):
                checksums[str(f.relative_to(self.project_dir))] = self._file_checksum(f)

        return PackageManifest(
            version="1.0",
            created_at=datetime.now().isoformat(),
            source_platform=platform.system(),
            file_count=len(files),
            total_size_bytes=total_size,
            storage_type=self._detect_storage_type(),
            includes_embeddings=self.include_embeddings,
            includes_state=self.include_state,
            checksums=checksums,
        )

    def _detect_storage_type(self) -> str:
        """Detect storage backend type.

        Returns:
            Storage type identifier
        """
        data_dir = self.project_dir / ".data"

        if (data_dir / "chroma.sqlite3").exists():
            return "chromadb"
        if (data_dir / "chunks.jsonl").exists():
            return "jsonl"
        if (data_dir / "index.faiss").exists():
            return "faiss"

        return "unknown"

    def _file_checksum(self, path: Path) -> str:
        """Calculate SHA-256 checksum of file.

        Args:
            path: File path

        Returns:
            Hex digest of checksum
        """
        import hashlib

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _write_archive(
        self,
        output_path: Path,
        files: List[Path],
        manifest: PackageManifest,
        progress_callback: Optional[ProgressCallback],
    ) -> int:
        """Write streaming ZIP archive.

        Args:
            output_path: Output path
            files: Files to archive
            manifest: Package manifest
            progress_callback: Progress callback

        Returns:
            Total bytes written
        """
        bytes_written = 0

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write manifest first
            manifest_json = json.dumps(manifest.to_dict(), indent=2)
            zf.writestr("manifest.json", manifest_json)
            bytes_written += len(manifest_json.encode())

            # Write files with streaming
            for idx, file_path in enumerate(files):
                if progress_callback:
                    progress_callback(idx + 1, len(files), file_path.name)

                arc_name = str(file_path.relative_to(self.project_dir))
                zf.write(file_path, arc_name)
                bytes_written += file_path.stat().st_size

        return bytes_written
