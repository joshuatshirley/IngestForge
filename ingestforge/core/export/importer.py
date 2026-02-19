"""Portable Corpus Importer for restoring knowledge bases.

Restores corpus packages with version mismatch detection and
cross-platform path normalization."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from ingestforge.core.export.packager import (
    PackageManifest,
    CHUNK_SIZE,
    MAX_FILE_SIZE_MB,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Current importer version
IMPORTER_VERSION = "1.0"

# Compatible manifest versions
COMPATIBLE_VERSIONS = {"1.0"}


@dataclass
class ImportResult:
    """Result of import operation."""

    success: bool
    target_dir: Optional[Path] = None
    manifest: Optional[PackageManifest] = None
    error: Optional[str] = None
    files_imported: int = 0
    warnings: List[str] = None

    def __post_init__(self) -> None:
        """Initialize warnings list."""
        if self.warnings is None:
            self.warnings = []


@dataclass
class ValidationResult:
    """Result of package validation."""

    valid: bool
    manifest: Optional[PackageManifest] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self) -> None:
        """Initialize lists."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


ProgressCallback = Callable[[int, int, str], None]


class CorpusImporter:
    """Restores corpus packages to target directories.

    Handles version compatibility, path normalization, and
    checksum verification for safe cross-platform transfers.
    """

    def __init__(
        self,
        verify_checksums: bool = True,
        overwrite_existing: bool = False,
    ) -> None:
        """Initialize importer.

        Args:
            verify_checksums: Verify file checksums after extraction
            overwrite_existing: Allow overwriting existing files
        """
        self.verify_checksums = verify_checksums
        self.overwrite_existing = overwrite_existing

    def validate(self, package_path: Path) -> ValidationResult:
        """Validate package before import.

        Args:
            package_path: Path to .zip package

        Returns:
            ValidationResult with validity status
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check file exists
        if not package_path.exists():
            return ValidationResult(
                valid=False, errors=[f"Package not found: {package_path}"]
            )

        # Check it's a valid ZIP
        if not zipfile.is_zipfile(package_path):
            return ValidationResult(
                valid=False, errors=["File is not a valid ZIP archive"]
            )

        try:
            with zipfile.ZipFile(package_path, "r") as zf:
                return self._validate_zip_contents(zf, errors, warnings)

        except json.JSONDecodeError as e:
            logger.exception(f"Invalid manifest JSON in package: {e}")
            return ValidationResult(valid=False, errors=[f"Invalid manifest: {e}"])
        except Exception as e:
            logger.exception(f"Package validation failed: {e}")
            return ValidationResult(valid=False, errors=[f"Validation failed: {e}"])

    def _validate_zip_contents(
        self, zf: zipfile.ZipFile, errors: List[str], warnings: List[str]
    ) -> ValidationResult:
        """Validate ZIP file contents.

        Args:
            zf: Open ZipFile
            errors: Error list (modified in place)
            warnings: Warning list (modified in place)

        Returns:
            ValidationResult
        """
        # Check for manifest
        if "manifest.json" not in zf.namelist():
            return ValidationResult(
                valid=False, errors=["Package missing manifest.json"]
            )

        # Load and parse manifest
        manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
        manifest = PackageManifest.from_dict(manifest_data)

        # Check version compatibility
        if manifest.version not in COMPATIBLE_VERSIONS:
            errors.append(
                f"Incompatible version: {manifest.version} "
                f"(supported: {COMPATIBLE_VERSIONS})"
            )

        # Check for large files
        for info in zf.infolist():
            if info.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                warnings.append(f"Large file: {info.filename}")

        # Validate integrity
        result = zf.testzip()
        if result is not None:
            errors.append(f"Corrupt file in archive: {result}")

        return ValidationResult(
            valid=len(errors) == 0,
            manifest=manifest,
            errors=errors,
            warnings=warnings,
        )

    def import_package(
        self,
        package_path: Path,
        target_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ImportResult:
        """Import corpus package to target directory.

        Args:
            package_path: Path to .zip package
            target_dir: Target directory for extraction
            progress_callback: Optional (current, total, filename) callback

        Returns:
            ImportResult with import status
        """
        # Validate first
        validation = self.validate(package_path)
        if not validation.valid:
            return ImportResult(success=False, error="; ".join(validation.errors))

        warnings = validation.warnings.copy()

        try:
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)

            # Check for existing files
            if not self.overwrite_existing:
                conflict = self._check_conflicts(package_path, target_dir)
                if conflict:
                    return ImportResult(
                        success=False,
                        error=f"Files would be overwritten: {conflict}. "
                        "Use overwrite_existing=True to allow.",
                    )

            # Extract files
            files_imported = self._extract_files(
                package_path, target_dir, progress_callback
            )

            # Verify checksums
            if self.verify_checksums and validation.manifest:
                checksum_warnings = self._verify_checksums(
                    target_dir, validation.manifest
                )
                warnings.extend(checksum_warnings)

            # Normalize paths for platform
            self._normalize_paths(target_dir)

            return ImportResult(
                success=True,
                target_dir=target_dir,
                manifest=validation.manifest,
                files_imported=files_imported,
                warnings=warnings,
            )

        except Exception as e:
            logger.exception(f"Import failed: {e}")
            return ImportResult(success=False, error=str(e))

    def _check_conflicts(self, package_path: Path, target_dir: Path) -> Optional[str]:
        """Check for file conflicts before extraction.

        Args:
            package_path: Package path
            target_dir: Target directory

        Returns:
            First conflicting file name, or None
        """
        with zipfile.ZipFile(package_path, "r") as zf:
            for name in zf.namelist():
                if name == "manifest.json":
                    continue
                target_path = target_dir / name
                if target_path.exists():
                    return name
        return None

    def _extract_files(
        self,
        package_path: Path,
        target_dir: Path,
        progress_callback: Optional[ProgressCallback],
    ) -> int:
        """Extract files from package.

        Args:
            package_path: Package path
            target_dir: Target directory
            progress_callback: Progress callback

        Returns:
            Number of files extracted
        """
        count = 0

        with zipfile.ZipFile(package_path, "r") as zf:
            members = [m for m in zf.namelist() if m != "manifest.json"]
            total = len(members)

            for idx, member in enumerate(members):
                if progress_callback:
                    progress_callback(idx + 1, total, member)

                if self._extract_single_file(zf, member, target_dir):
                    count += 1

        return count

    def _extract_single_file(
        self, zf: zipfile.ZipFile, member: str, target_dir: Path
    ) -> bool:
        """Extract a single file from ZIP.

        Args:
            zf: Open ZipFile
            member: Member name
            target_dir: Target directory

        Returns:
            True if extracted, False if skipped
        """
        # Security: Prevent path traversal
        normalized = self._normalize_zip_path(member)
        if normalized is None:
            logger.warning(f"Skipping unsafe path: {member}")
            return False

        target_path = target_dir / normalized
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract with streaming
        with zf.open(member) as src, open(target_path, "wb") as dst:
            while chunk := src.read(CHUNK_SIZE):
                dst.write(chunk)

        return True

    def _normalize_zip_path(self, path: str) -> Optional[str]:
        """Normalize and validate ZIP path for safety.

        Args:
            path: Path from ZIP archive

        Returns:
            Normalized path, or None if unsafe
        """
        # Reject absolute paths
        if path.startswith("/") or path.startswith("\\"):
            return None

        # Reject path traversal
        if ".." in path:
            return None

        # Normalize separators
        normalized = path.replace("\\", "/")

        # Remove leading ./
        while normalized.startswith("./"):
            normalized = normalized[2:]

        return normalized

    def _verify_checksums(
        self, target_dir: Path, manifest: PackageManifest
    ) -> List[str]:
        """Verify file checksums against manifest.

        Args:
            target_dir: Extracted directory
            manifest: Package manifest

        Returns:
            List of warning messages for mismatches
        """
        import hashlib

        warnings: List[str] = []

        for rel_path, expected_hash in manifest.checksums.items():
            file_path = target_dir / rel_path
            if not file_path.exists():
                warnings.append(f"Missing file: {rel_path}")
                continue

            # Calculate actual hash
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    sha256.update(chunk)

            actual_hash = sha256.hexdigest()
            if actual_hash != expected_hash:
                warnings.append(f"Checksum mismatch: {rel_path}")

        return warnings

    def _normalize_paths(self, target_dir: Path) -> None:
        """Normalize paths for current platform.

        Args:
            target_dir: Directory with extracted files
        """
        # This handles any platform-specific path adjustments needed
        # Most work is done in _normalize_zip_path during extraction
        pass


def get_package_info(package_path: Path) -> Optional[PackageManifest]:
    """Get manifest info from package without extracting.

    Args:
        package_path: Path to .zip package

    Returns:
        PackageManifest if valid, None otherwise
    """
    if not package_path.exists() or not zipfile.is_zipfile(package_path):
        return None

    try:
        with zipfile.ZipFile(package_path, "r") as zf:
            if "manifest.json" not in zf.namelist():
                return None

            manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
            return PackageManifest.from_dict(manifest_data)
    except Exception:
        logger.exception(f"Error reading package manifest from {package_path}")
        return None
