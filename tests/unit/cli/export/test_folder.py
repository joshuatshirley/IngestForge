"""
Tests for Folder Export Command.

This module tests the signature folder export feature that generates
complete study packages.

Test Strategy
-------------
- Focus on study package generation logic
- Mock storage, chunk retrieval, and LLM client
- Test each component generator independently
- Test directory structure creation
- Test error handling and validation
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestFolderExportInit: Initialization
- TestDirectoryValidation: Path validation
- TestPackageStructure: Directory structure creation
- TestComponentGeneration: Individual component tests
- TestIntegration: End-to-end package generation
- TestErrorHandling: Error scenarios
"""

from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

import pytest
import typer

from ingestforge.cli.export.folder import FolderExportCommand


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "Test content",
    source_file: str = "test.txt",
    **metadata,
):
    """Create a mock chunk object.

    Args:
        chunk_id: Chunk identifier
        content: Chunk text content
        source_file: Source file name
        **metadata: Additional metadata fields

    Returns:
        Mock chunk object
    """
    chunk = Mock()
    chunk.chunk_id = chunk_id
    chunk.content = content
    chunk.text = content  # Some methods look for 'text' instead of 'content'
    chunk.source_file = source_file

    # Set up metadata dict with source
    chunk.metadata = {"source": source_file}

    # Add any additional metadata
    for key, value in metadata.items():
        setattr(chunk, key, value)
        if key not in chunk.metadata:
            chunk.metadata[key] = value

    return chunk


def make_mock_context(has_storage: bool = True, has_llm: bool = False):
    """Create a mock context dictionary.

    Args:
        has_storage: Whether to include storage mock
        has_llm: Whether to include LLM client

    Returns:
        Mock context dictionary
    """
    ctx = {}

    if has_storage:
        ctx["storage"] = Mock()
        ctx["config"] = Mock()

    return ctx


def make_mock_llm_client():
    """Create a mock LLM client.

    Returns:
        Mock LLM client
    """
    llm = Mock()
    llm.generate = Mock(return_value="Generated content")
    return llm


# ============================================================================
# Test Classes
# ============================================================================


class TestFolderExportInit:
    """Tests for FolderExportCommand initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_folder_export_command(self):
        """Test creating FolderExportCommand instance."""
        cmd = FolderExportCommand()

        assert cmd is not None

    def test_inherits_from_export_command(self):
        """Test FolderExportCommand inherits from ExportCommand."""
        from ingestforge.cli.export.base import ExportCommand

        cmd = FolderExportCommand()

        assert isinstance(cmd, ExportCommand)


class TestDirectoryValidation:
    """Tests for output directory validation.

    Rule #4: Focused test class - tests validation logic
    """

    def test_validate_nonexistent_directory(self, tmp_path):
        """Test validation accepts nonexistent directory."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "new_folder"

        # Should not raise
        cmd._validate_output_directory(output_dir)

    def test_validate_empty_existing_directory(self, tmp_path):
        """Test validation accepts empty existing directory."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "empty"
        output_dir.mkdir()

        # Should not raise
        cmd._validate_output_directory(output_dir)

    def test_validate_file_path_fails(self, tmp_path):
        """Test validation rejects file path."""
        cmd = FolderExportCommand()
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        with pytest.raises(typer.BadParameter):
            cmd._validate_output_directory(file_path)

    def test_validate_nonempty_directory_warns(self, tmp_path, capsys):
        """Test validation warns for nonempty directory."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "nonempty"
        output_dir.mkdir()
        (output_dir / "existing.txt").write_text("content")

        # Should warn but not raise
        cmd._validate_output_directory(output_dir)


class TestPackageStructure:
    """Tests for package directory structure creation.

    Rule #4: Focused test class - tests structure logic
    """

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "study_package"

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = [make_mock_chunk()]
                    mock_llm.return_value = None

                    cmd.execute(output_dir, include_all=False)

                    assert output_dir.exists()
                    assert output_dir.is_dir()

    def test_creates_all_component_files(self, tmp_path):
        """Test that all component files are created."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "study_package"

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = [make_mock_chunk()]
                    mock_llm.return_value = None

                    cmd.execute(output_dir, include_all=True)

                    # Check all expected files exist
                    expected_files = [
                        "00_START_HERE.md",
                        "01_overview.md",
                        "02_glossary.md",
                        "03_concept_map.md",
                        "05_flashcards.csv",
                        "06_quiz.md",
                        "07_reading_list.md",
                        "bibliography.bib",
                    ]

                    for filename in expected_files:
                        assert (output_dir / filename).exists(), f"Missing: {filename}"

                    # Check study notes directory
                    assert (output_dir / "04_study_notes").exists()
                    assert (output_dir / "04_study_notes").is_dir()


class TestComponentGeneration:
    """Tests for individual component generation.

    Rule #4: Focused test class - tests component generators
    """

    def test_generate_start_here(self, tmp_path):
        """Test START_HERE.md generation."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"
        output_dir.mkdir()

        cmd._generate_start_here(output_dir, "Test Topic")

        start_file = output_dir / "00_START_HERE.md"
        assert start_file.exists()

        content = start_file.read_text()
        assert "Test Topic" in content
        assert "Welcome" in content
        assert "Recommended Study Path" in content

    def test_generate_basic_overview(self, tmp_path):
        """Test overview generation without LLM."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"
        output_dir.mkdir()

        chunks = [
            make_mock_chunk(chunk_id="1", source_file="doc1.txt"),
            make_mock_chunk(chunk_id="2", source_file="doc1.txt"),
            make_mock_chunk(chunk_id="3", source_file="doc2.txt"),
        ]

        cmd._generate_overview(output_dir, chunks, llm_client=None)

        overview_file = output_dir / "01_overview.md"
        assert overview_file.exists()

        content = overview_file.read_text()
        assert "Total Documents" in content
        assert "Total Chunks" in content

    def test_generate_study_notes(self, tmp_path):
        """Test study notes directory generation."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"
        output_dir.mkdir()

        chunks = [
            make_mock_chunk(chunk_id="1", content="Content 1", source_file="doc1.txt"),
            make_mock_chunk(chunk_id="2", content="Content 2", source_file="doc2.txt"),
        ]

        cmd._generate_study_notes(output_dir, chunks)

        notes_dir = output_dir / "04_study_notes"
        assert notes_dir.exists()
        assert notes_dir.is_dir()

        # Should have one file per source
        note_files = list(notes_dir.glob("*.md"))
        assert len(note_files) == 2

    def test_generate_reading_list(self, tmp_path):
        """Test reading list generation."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"
        output_dir.mkdir()

        chunks = [
            make_mock_chunk(source_file="doc1.txt"),
            make_mock_chunk(source_file="doc2.txt"),
            make_mock_chunk(source_file="doc2.txt"),
        ]

        cmd._generate_reading_list(output_dir, chunks)

        reading_file = output_dir / "07_reading_list.md"
        assert reading_file.exists()

        content = reading_file.read_text()
        assert "**Total Sources**: 2" in content or "Total Sources: 2" in content
        assert "doc1.txt" in content
        assert "doc2.txt" in content

    def test_generate_bibliography(self, tmp_path):
        """Test BibTeX bibliography generation."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"
        output_dir.mkdir()

        chunks = [
            make_mock_chunk(source_file="paper1.pdf"),
            make_mock_chunk(source_file="paper2.pdf"),
        ]

        cmd._generate_bibliography(output_dir, chunks)

        bib_file = output_dir / "bibliography.bib"
        assert bib_file.exists()

        content = bib_file.read_text()
        assert "@misc{" in content
        assert "paper1" in content
        assert "paper2" in content


class TestLLMGeneration:
    """Tests for LLM-powered generation.

    Rule #4: Focused test class - tests LLM integration
    """

    def test_llm_overview_generation(self, tmp_path):
        """Test overview generation with LLM."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"
        output_dir.mkdir()

        chunks = [make_mock_chunk(content="Test content")]
        llm_client = make_mock_llm_client()
        llm_client.generate.return_value = "LLM-generated overview"

        with patch("ingestforge.cli.core.ProgressManager") as mock_pm:
            mock_pm.run_with_spinner.side_effect = lambda fn, *args: fn()

            cmd._generate_overview(output_dir, chunks, llm_client)

        overview_file = output_dir / "01_overview.md"
        content = overview_file.read_text()
        assert "LLM-generated overview" in content

    def test_llm_glossary_generation(self, tmp_path):
        """Test glossary generation with LLM."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"
        output_dir.mkdir()

        chunks = [make_mock_chunk(content="Test content")]
        llm_client = make_mock_llm_client()
        llm_client.generate.return_value = "**Term**: Definition"

        with patch("ingestforge.cli.core.ProgressManager") as mock_pm:
            mock_pm.run_with_spinner.side_effect = lambda fn, *args: fn()

            cmd._generate_glossary(output_dir, chunks, llm_client)

        glossary_file = output_dir / "02_glossary.md"
        content = glossary_file.read_text()
        assert "Term" in content


class TestErrorHandling:
    """Tests for error handling.

    Rule #4: Focused test class - tests error scenarios
    """

    def test_handles_no_chunks(self, tmp_path):
        """Test handling of empty knowledge base."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                mock_init.return_value = make_mock_context()
                mock_chunks.return_value = []

                result = cmd.execute(output_dir)

                assert result == 0  # Success but nothing to export

    def test_handles_llm_client_failure(self, tmp_path):
        """Test graceful handling when LLM unavailable."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = [make_mock_chunk()]
                    mock_llm.return_value = None  # No LLM available

                    result = cmd.execute(output_dir, include_all=True)

                    # Should still succeed with basic exports
                    assert result == 0
                    assert (output_dir / "00_START_HERE.md").exists()


class TestHelperFunctions:
    """Tests for helper/utility functions.

    Rule #4: Focused test class - tests utilities
    """

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        cmd = FolderExportCommand()

        # Test special characters
        result = cmd._sanitize_filename("file/with\\bad:chars.txt")
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result
        assert ".txt" not in result  # Extension removed

    def test_sanitize_filename_length_limit(self):
        """Test filename length limiting."""
        cmd = FolderExportCommand()

        long_name = "a" * 200 + ".txt"
        result = cmd._sanitize_filename(long_name)

        assert len(result) <= 100

    def test_generate_cite_key(self):
        """Test BibTeX citation key generation."""
        cmd = FolderExportCommand()

        key = cmd._generate_cite_key("My Paper Title.pdf")

        assert key.islower()
        assert len(key) <= 30
        assert key.isalnum()

    def test_build_context_sample_limits_chunks(self):
        """Test context sampling limits number of chunks."""
        cmd = FolderExportCommand()

        chunks = [make_mock_chunk(chunk_id=str(i)) for i in range(100)]

        context = cmd._build_context_sample(chunks, max_chunks=10)

        # Should only include 10 chunks worth of content
        assert context.count("---") <= 10


class TestNewCLIOptions:
    """Tests for new CLI options (LLM provider, flashcards, quiz flags).

    Rule #4: Focused test class - tests new options
    """

    def test_llm_provider_option(self, tmp_path):
        """Test --llm option for provider selection."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"

        chunks = [make_mock_chunk()]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = chunks
                    mock_llm.return_value = None

                    result = cmd.execute(
                        output_dir,
                        include_all=True,
                        llm_provider="claude",
                    )

                    # Verify LLM client was called with provider
                    mock_llm.assert_called_once()
                    call_args = mock_llm.call_args
                    assert call_args[0][1] == "claude"  # provider argument

    def test_no_flashcards_flag(self, tmp_path):
        """Test --no-flashcards excludes flashcard file."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"

        chunks = [make_mock_chunk()]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = chunks
                    mock_llm.return_value = None

                    result = cmd.execute(
                        output_dir,
                        include_all=True,
                        include_flashcards=False,
                    )

                    assert result == 0
                    # Flashcards should not be generated
                    assert not (output_dir / "05_flashcards.csv").exists()

    def test_no_quiz_flag(self, tmp_path):
        """Test --no-quiz excludes quiz file."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"

        chunks = [make_mock_chunk()]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = chunks
                    mock_llm.return_value = None

                    result = cmd.execute(
                        output_dir,
                        include_all=True,
                        include_quiz=False,
                    )

                    assert result == 0
                    # Quiz should not be generated
                    assert not (output_dir / "06_quiz.md").exists()

    def test_both_flashcards_and_quiz_excluded(self, tmp_path):
        """Test excluding both flashcards and quiz."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "pkg"

        chunks = [make_mock_chunk()]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = chunks
                    mock_llm.return_value = None

                    result = cmd.execute(
                        output_dir,
                        include_all=True,
                        include_flashcards=False,
                        include_quiz=False,
                    )

                    assert result == 0
                    # Other files should still exist
                    assert (output_dir / "00_START_HERE.md").exists()
                    assert (output_dir / "01_overview.md").exists()


class TestIntegration:
    """Integration tests for complete package generation.

    Rule #4: Focused test class - tests full workflow
    """

    def test_full_package_generation_basic(self, tmp_path):
        """Test complete package generation without LLM."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "full_package"

        chunks = [
            make_mock_chunk(chunk_id="1", content="Content 1", source_file="doc1.txt"),
            make_mock_chunk(chunk_id="2", content="Content 2", source_file="doc2.txt"),
            make_mock_chunk(chunk_id="3", content="Content 3", source_file="doc2.txt"),
        ]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    mock_init.return_value = make_mock_context()
                    mock_chunks.return_value = chunks
                    mock_llm.return_value = None

                    result = cmd.execute(
                        output_dir, topic="Test Topic", include_all=True
                    )

                    assert result == 0

                    # Verify all files created
                    assert (output_dir / "00_START_HERE.md").exists()
                    assert (output_dir / "01_overview.md").exists()
                    assert (output_dir / "04_study_notes").is_dir()
                    assert (output_dir / "07_reading_list.md").exists()
                    assert (output_dir / "bibliography.bib").exists()

    def test_full_package_with_llm(self, tmp_path):
        """Test complete package generation with LLM."""
        cmd = FolderExportCommand()
        output_dir = tmp_path / "full_package_llm"

        chunks = [make_mock_chunk(content="Test content")]
        llm_client = make_mock_llm_client()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_all_chunks_from_storage") as mock_chunks:
                with patch.object(cmd, "_get_llm_client") as mock_llm:
                    with patch("ingestforge.cli.core.ProgressManager") as mock_pm:
                        mock_init.return_value = make_mock_context()
                        mock_chunks.return_value = chunks
                        mock_llm.return_value = llm_client
                        mock_pm.run_with_spinner.side_effect = lambda fn, *args: fn()

                        result = cmd.execute(output_dir, include_all=True)

                        assert result == 0

                        # Verify LLM was called for generation
                        assert llm_client.generate.called


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory.

    Yields:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
