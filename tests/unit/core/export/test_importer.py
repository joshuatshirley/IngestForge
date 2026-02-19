"""Tests for corpus importer module.

Tests portable corpus importing with validation and path normalization."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ingestforge.core.export.importer import (
    CorpusImporter,
    ValidationResult,
    get_package_info,
)
from ingestforge.core.export.packager import PackageManifest

# Test fixtures


@pytest.fixture
def valid_package(tmp_path: Path) -> Path:
    """Create a valid corpus package."""
    package_path = tmp_path / "valid.zip"

    manifest = PackageManifest(
        version="1.0",
        file_count=2,
        storage_type="jsonl",
        created_at="2024-01-01T00:00:00",
        source_platform="test",
    )

    with zipfile.ZipFile(package_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest.to_dict()))
        zf.writestr(".ingest/config.json", '{"version": "1.0"}')
        zf.writestr(".data/chunks.jsonl", '{"id": "1"}\n')

    return package_path


@pytest.fixture
def invalid_package_no_manifest(tmp_path: Path) -> Path:
    """Create package without manifest."""
    package_path = tmp_path / "no_manifest.zip"

    with zipfile.ZipFile(package_path, "w") as zf:
        zf.writestr("data.txt", "some data")

    return package_path


@pytest.fixture
def invalid_package_bad_version(tmp_path: Path) -> Path:
    """Create package with incompatible version."""
    package_path = tmp_path / "bad_version.zip"

    manifest = {"version": "99.0", "file_count": 0}

    with zipfile.ZipFile(package_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))

    return package_path


# ValidationResult tests


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating validation result."""
        result = ValidationResult(valid=True)

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_result_with_errors(self) -> None:
        """Test result with errors."""
        result = ValidationResult(valid=False, errors=["Error 1", "Error 2"])

        assert result.valid is False
        assert len(result.errors) == 2


# CorpusImporter validation tests


class TestCorpusImporterValidation:
    """Tests for package validation."""

    def test_validate_valid_package(self, valid_package: Path) -> None:
        """Test validating a valid package."""
        importer = CorpusImporter()
        result = importer.validate(valid_package)

        assert result.valid is True
        assert result.manifest is not None
        assert result.manifest.version == "1.0"

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Test validating nonexistent file."""
        importer = CorpusImporter()
        result = importer.validate(tmp_path / "nonexistent.zip")

        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_validate_not_a_zip(self, tmp_path: Path) -> None:
        """Test validating non-ZIP file."""
        not_zip = tmp_path / "not_a_zip.txt"
        not_zip.write_text("not a zip file")

        importer = CorpusImporter()
        result = importer.validate(not_zip)

        assert result.valid is False
        assert any("not a valid ZIP" in e for e in result.errors)

    def test_validate_missing_manifest(self, invalid_package_no_manifest: Path) -> None:
        """Test validating package without manifest."""
        importer = CorpusImporter()
        result = importer.validate(invalid_package_no_manifest)

        assert result.valid is False
        assert any("manifest" in e.lower() for e in result.errors)

    def test_validate_incompatible_version(
        self, invalid_package_bad_version: Path
    ) -> None:
        """Test validating package with incompatible version."""
        importer = CorpusImporter()
        result = importer.validate(invalid_package_bad_version)

        assert result.valid is False
        assert any("version" in e.lower() for e in result.errors)


class TestCorpusImporterImport:
    """Tests for package import."""

    def test_import_success(self, valid_package: Path, tmp_path: Path) -> None:
        """Test successful import."""
        target = tmp_path / "imported"
        importer = CorpusImporter()

        result = importer.import_package(valid_package, target)

        assert result.success is True
        assert result.files_imported > 0
        assert target.exists()
        assert (target / ".ingest" / "config.json").exists()

    def test_import_creates_target_directory(
        self, valid_package: Path, tmp_path: Path
    ) -> None:
        """Test that import creates target directory."""
        target = tmp_path / "new" / "nested" / "dir"
        importer = CorpusImporter()

        result = importer.import_package(valid_package, target)

        assert result.success is True
        assert target.exists()

    def test_import_fails_on_conflict_without_force(
        self, valid_package: Path, tmp_path: Path
    ) -> None:
        """Test import fails when files would be overwritten."""
        target = tmp_path / "existing"
        target.mkdir()

        # Create conflicting file
        ingest_dir = target / ".ingest"
        ingest_dir.mkdir()
        (ingest_dir / "config.json").write_text("existing")

        importer = CorpusImporter(overwrite_existing=False)
        result = importer.import_package(valid_package, target)

        assert result.success is False
        assert "overwritten" in result.error.lower()

    def test_import_succeeds_with_force(
        self, valid_package: Path, tmp_path: Path
    ) -> None:
        """Test import succeeds with force flag."""
        target = tmp_path / "existing"
        target.mkdir()

        # Create conflicting file
        ingest_dir = target / ".ingest"
        ingest_dir.mkdir()
        (ingest_dir / "config.json").write_text("existing")

        importer = CorpusImporter(overwrite_existing=True)
        result = importer.import_package(valid_package, target)

        assert result.success is True

    def test_import_progress_callback(
        self, valid_package: Path, tmp_path: Path
    ) -> None:
        """Test progress callback is called during import."""
        target = tmp_path / "imported"
        importer = CorpusImporter()

        progress_calls: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, name: str) -> None:
            progress_calls.append((current, total, name))

        importer.import_package(valid_package, target, progress_callback=callback)

        assert len(progress_calls) > 0


class TestCorpusImporterPathSecurity:
    """Tests for path security during import."""

    def test_rejects_absolute_paths(self, tmp_path: Path) -> None:
        """Test that absolute paths are rejected."""
        package_path = tmp_path / "unsafe.zip"

        with zipfile.ZipFile(package_path, "w") as zf:
            zf.writestr("manifest.json", '{"version": "1.0"}')
            # Attempt to include absolute path (won't work but test the guard)
            zf.writestr("/etc/passwd", "malicious")

        importer = CorpusImporter()
        # Validation should pass but import should skip unsafe files
        target = tmp_path / "imported"
        result = importer.import_package(package_path, target)

        # Should succeed but skip unsafe paths
        assert result.success is True
        # Unsafe file should not exist in target directory
        assert not (target / "etc" / "passwd").exists()

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Test that path traversal is rejected."""
        package_path = tmp_path / "traversal.zip"

        with zipfile.ZipFile(package_path, "w") as zf:
            zf.writestr("manifest.json", '{"version": "1.0"}')
            zf.writestr("../../../etc/passwd", "malicious")

        importer = CorpusImporter()
        target = tmp_path / "imported"
        result = importer.import_package(package_path, target)

        # Should succeed but skip unsafe paths
        # Unsafe file should not exist outside target
        assert not (tmp_path / "etc" / "passwd").exists()


class TestCorpusImporterChecksum:
    """Tests for checksum verification."""

    def test_verify_checksums_success(self, tmp_path: Path) -> None:
        """Test successful checksum verification."""
        import hashlib

        # Create content
        content = b'{"version": "1.0"}'
        checksum = hashlib.sha256(content).hexdigest()

        # Create package with checksum
        package_path = tmp_path / "verified.zip"
        manifest = {
            "version": "1.0",
            "file_count": 1,
            "checksums": {".ingest/config.json": checksum},
        }

        with zipfile.ZipFile(package_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr(".ingest/config.json", content)

        importer = CorpusImporter(verify_checksums=True)
        target = tmp_path / "imported"
        result = importer.import_package(package_path, target)

        assert result.success is True
        # No checksum warnings
        assert not any("mismatch" in w.lower() for w in result.warnings)

    def test_verify_checksums_mismatch(self, tmp_path: Path) -> None:
        """Test checksum mismatch detection."""
        package_path = tmp_path / "mismatch.zip"
        manifest = {
            "version": "1.0",
            "file_count": 1,
            "checksums": {".ingest/config.json": "invalid_checksum"},
        }

        with zipfile.ZipFile(package_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr(".ingest/config.json", '{"version": "1.0"}')

        importer = CorpusImporter(verify_checksums=True)
        target = tmp_path / "imported"
        result = importer.import_package(package_path, target)

        # Should succeed but with warning
        assert result.success is True
        assert any("mismatch" in w.lower() for w in result.warnings)


# get_package_info tests


class TestGetPackageInfo:
    """Tests for get_package_info function."""

    def test_get_info_valid_package(self, valid_package: Path) -> None:
        """Test getting info from valid package."""
        manifest = get_package_info(valid_package)

        assert manifest is not None
        assert manifest.version == "1.0"
        assert manifest.storage_type == "jsonl"

    def test_get_info_nonexistent_file(self, tmp_path: Path) -> None:
        """Test getting info from nonexistent file."""
        result = get_package_info(tmp_path / "nonexistent.zip")

        assert result is None

    def test_get_info_invalid_package(self, invalid_package_no_manifest: Path) -> None:
        """Test getting info from package without manifest."""
        result = get_package_info(invalid_package_no_manifest)

        assert result is None
