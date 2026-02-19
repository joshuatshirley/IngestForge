"""Tests for corpus packager module.

Tests portable corpus packaging with streaming ZIP creation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ingestforge.core.export.packager import (
    CorpusPackager,
    PackageManifest,
)

# Test fixtures


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create sample project structure."""
    # Create .ingest directory
    ingest_dir = tmp_path / ".ingest"
    ingest_dir.mkdir()
    (ingest_dir / "config.json").write_text('{"version": "1.0"}')
    (ingest_dir / "metadata.json").write_text('{"files": []}')

    # Create .data directory
    data_dir = tmp_path / ".data"
    data_dir.mkdir()
    (data_dir / "chunks.jsonl").write_text('{"id": "1", "content": "test"}\n')
    (data_dir / "embeddings.bin").write_bytes(b"\x00" * 100)

    # Create config file
    (tmp_path / "ingestforge.yaml").write_text("storage:\n  type: jsonl\n")

    return tmp_path


@pytest.fixture
def empty_project(tmp_path: Path) -> Path:
    """Create empty project directory."""
    return tmp_path


# PackageManifest tests


class TestPackageManifest:
    """Tests for PackageManifest dataclass."""

    def test_manifest_creation(self) -> None:
        """Test creating manifest with defaults."""
        manifest = PackageManifest()

        assert manifest.version == "1.0"
        assert manifest.file_count == 0
        assert manifest.includes_embeddings is True

    def test_manifest_to_dict(self) -> None:
        """Test converting manifest to dict."""
        manifest = PackageManifest(
            version="1.0",
            file_count=10,
            storage_type="chromadb",
        )

        data = manifest.to_dict()

        assert data["version"] == "1.0"
        assert data["file_count"] == 10
        assert data["storage_type"] == "chromadb"

    def test_manifest_from_dict(self) -> None:
        """Test creating manifest from dict."""
        data = {
            "version": "1.0",
            "file_count": 5,
            "storage_type": "jsonl",
            "includes_embeddings": False,
        }

        manifest = PackageManifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.file_count == 5
        assert manifest.storage_type == "jsonl"
        assert manifest.includes_embeddings is False

    def test_manifest_from_dict_defaults(self) -> None:
        """Test manifest creation with missing fields."""
        data = {}

        manifest = PackageManifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.file_count == 0


# CorpusPackager tests


class TestCorpusPackager:
    """Tests for CorpusPackager."""

    def test_packager_creation(self, sample_project: Path) -> None:
        """Test creating packager instance."""
        packager = CorpusPackager(project_dir=sample_project)

        assert packager.project_dir == sample_project
        assert packager.include_embeddings is True
        assert packager.include_state is True

    def test_package_success(self, sample_project: Path, tmp_path: Path) -> None:
        """Test successful packaging."""
        output_path = tmp_path / "corpus.zip"
        packager = CorpusPackager(project_dir=sample_project)

        result = packager.package(output_path)

        assert result.success is True
        assert result.output_path == output_path
        assert result.files_packaged > 0
        assert output_path.exists()

    def test_package_creates_valid_zip(
        self, sample_project: Path, tmp_path: Path
    ) -> None:
        """Test that created package is a valid ZIP."""
        output_path = tmp_path / "corpus.zip"
        packager = CorpusPackager(project_dir=sample_project)

        packager.package(output_path)

        assert zipfile.is_zipfile(output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            assert zf.testzip() is None  # No corrupt files

    def test_package_includes_manifest(
        self, sample_project: Path, tmp_path: Path
    ) -> None:
        """Test that package includes manifest."""
        output_path = tmp_path / "corpus.zip"
        packager = CorpusPackager(project_dir=sample_project)

        packager.package(output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            assert "manifest.json" in zf.namelist()

            manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
            assert "version" in manifest_data
            assert "file_count" in manifest_data

    def test_package_empty_project(self, empty_project: Path, tmp_path: Path) -> None:
        """Test packaging empty project fails gracefully."""
        output_path = tmp_path / "corpus.zip"
        packager = CorpusPackager(project_dir=empty_project)

        result = packager.package(output_path)

        assert result.success is False
        assert "No files found" in result.error

    def test_package_excludes_lock_files(
        self, sample_project: Path, tmp_path: Path
    ) -> None:
        """Test that lock files are excluded."""
        # Create a lock file
        (sample_project / ".ingest" / "state.lock").write_text("locked")

        output_path = tmp_path / "corpus.zip"
        packager = CorpusPackager(project_dir=sample_project)

        packager.package(output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            names = zf.namelist()
            assert not any("lock" in n.lower() for n in names)

    def test_package_excludes_embeddings_when_disabled(
        self, sample_project: Path, tmp_path: Path
    ) -> None:
        """Test excluding embeddings."""
        output_path = tmp_path / "corpus.zip"
        packager = CorpusPackager(
            project_dir=sample_project,
            include_embeddings=False,
        )

        packager.package(output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            names = zf.namelist()
            assert not any("embedding" in n.lower() for n in names)

    def test_package_detects_storage_type_chromadb(self, tmp_path: Path) -> None:
        """Test ChromaDB storage detection."""
        # Create minimal chromadb structure
        ingest_dir = tmp_path / ".ingest"
        ingest_dir.mkdir()
        (ingest_dir / "config.json").write_text("{}")

        data_dir = tmp_path / ".data"
        data_dir.mkdir()
        (data_dir / "chroma.sqlite3").write_bytes(b"sqlite")

        packager = CorpusPackager(project_dir=tmp_path)
        storage_type = packager._detect_storage_type()

        assert storage_type == "chromadb"

    def test_package_detects_storage_type_jsonl(self, tmp_path: Path) -> None:
        """Test JSONL storage detection."""
        # Create minimal jsonl structure
        ingest_dir = tmp_path / ".ingest"
        ingest_dir.mkdir()
        (ingest_dir / "config.json").write_text("{}")

        data_dir = tmp_path / ".data"
        data_dir.mkdir()
        (data_dir / "chunks.jsonl").write_text("{}\n")

        packager = CorpusPackager(project_dir=tmp_path)
        storage_type = packager._detect_storage_type()

        assert storage_type == "jsonl"

    def test_package_progress_callback(
        self, sample_project: Path, tmp_path: Path
    ) -> None:
        """Test progress callback is called."""
        output_path = tmp_path / "corpus.zip"
        packager = CorpusPackager(project_dir=sample_project)

        progress_calls: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, name: str) -> None:
            progress_calls.append((current, total, name))

        packager.package(output_path, progress_callback=callback)

        assert len(progress_calls) > 0
        # First call should be 1, last should match total
        assert progress_calls[0][0] == 1
        assert progress_calls[-1][0] == progress_calls[-1][1]


class TestPackagerFiltering:
    """Tests for file filtering logic."""

    def test_excludes_temp_files(self, sample_project: Path, tmp_path: Path) -> None:
        """Test temp files are excluded."""
        (sample_project / ".ingest" / "temp.tmp").write_text("temp")

        packager = CorpusPackager(project_dir=sample_project)
        output_path = tmp_path / "corpus.zip"
        packager.package(output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            names = zf.namelist()
            assert not any(".tmp" in n for n in names)

    def test_excludes_hidden_files(self, sample_project: Path, tmp_path: Path) -> None:
        """Test hidden files are excluded."""
        (sample_project / ".ingest" / ".hidden").write_text("hidden")

        packager = CorpusPackager(project_dir=sample_project)
        output_path = tmp_path / "corpus.zip"
        packager.package(output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            names = zf.namelist()
            # Should not have any file starting with .
            non_manifest = [n for n in names if n != "manifest.json"]
            hidden = [n for n in non_manifest if Path(n).name.startswith(".")]
            assert len(hidden) == 0


class TestPackagerChecksum:
    """Tests for checksum calculation."""

    def test_calculates_checksums_for_key_files(
        self, sample_project: Path, tmp_path: Path
    ) -> None:
        """Test checksums are calculated for config files."""
        packager = CorpusPackager(project_dir=sample_project)
        output_path = tmp_path / "corpus.zip"
        result = packager.package(output_path)

        assert result.manifest is not None
        assert len(result.manifest.checksums) > 0

        # Config files should have checksums
        checksums = result.manifest.checksums
        assert any("config" in k for k in checksums.keys())

    def test_checksum_is_sha256(self, sample_project: Path) -> None:
        """Test checksums are SHA-256 hex digests."""
        packager = CorpusPackager(project_dir=sample_project)

        config_path = sample_project / ".ingest" / "config.json"
        checksum = packager._file_checksum(config_path)

        # SHA-256 hex digest is 64 characters
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)
