"""Tests for quickstart wizard."""

from pathlib import Path

from ingestforge.cli.quickstart import (
    check_project_exists,
    get_sample_files,
    STORAGE_PROMPT,
)


class TestQuickstartHelpers:
    """Test quickstart helper functions."""

    def test_check_project_exists_no_project(self, temp_dir: Path, monkeypatch) -> None:
        """Test check_project_exists returns False when no project."""
        monkeypatch.chdir(temp_dir)
        assert check_project_exists() is False

    def test_check_project_exists_with_project(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """Test check_project_exists returns True when .ingestforge exists."""
        monkeypatch.chdir(temp_dir)
        (temp_dir / ".ingestforge").mkdir()
        assert check_project_exists() is True

    def test_get_sample_files_empty_dir(self, temp_dir: Path, monkeypatch) -> None:
        """Test get_sample_files returns empty list for empty directory."""
        monkeypatch.chdir(temp_dir)
        assert get_sample_files() == []

    def test_get_sample_files_finds_supported_types(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        """Test get_sample_files finds supported document types."""
        monkeypatch.chdir(temp_dir)

        # Create sample files
        (temp_dir / "doc1.pdf").touch()
        (temp_dir / "doc2.md").touch()
        (temp_dir / "doc3.txt").touch()
        (temp_dir / "ignore.xyz").touch()  # Unsupported

        files = get_sample_files()
        assert len(files) == 3
        names = [f.name for f in files]
        assert "doc1.pdf" in names
        assert "doc2.md" in names
        assert "doc3.txt" in names
        assert "ignore.xyz" not in names

    def test_get_sample_files_limits_to_10(self, temp_dir: Path, monkeypatch) -> None:
        """Test get_sample_files returns at most 10 files."""
        monkeypatch.chdir(temp_dir)

        # Create 15 files
        for i in range(15):
            (temp_dir / f"doc{i:02d}.txt").touch()

        files = get_sample_files()
        assert len(files) == 10


class TestStoragePrompt:
    """Test storage mode prompt content."""

    def test_storage_prompt_mentions_chromadb(self) -> None:
        """Test that storage prompt mentions ChromaDB for default."""
        assert "ChromaDB" in STORAGE_PROMPT

    def test_storage_prompt_mentions_mobile(self) -> None:
        """Test that storage prompt mentions mobile/JSONL option."""
        assert "Mobile" in STORAGE_PROMPT or "JSONL" in STORAGE_PROMPT

    def test_storage_prompt_mentions_memory_requirements(self) -> None:
        """Test that storage prompt mentions memory requirements."""
        assert "RAM" in STORAGE_PROMPT or "memory" in STORAGE_PROMPT.lower()
