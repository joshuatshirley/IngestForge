"""
Tests for Pipeline Orchestrator.

This module tests the core Pipeline class that orchestrates document processing.

Test Strategy
-------------
- Focus on public API behavior, not implementation details
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Avoid deep mocking - test real behavior where possible
- Each test should be self-contained and clear

Organization
------------
- TestPipelineResult: Tests for PipelineResult dataclass
- TestPipelineInitialization: Tests for Pipeline.__init__()
- TestPipelineBasics: Tests for basic pipeline operations
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ingestforge.core.pipeline import Pipeline, PipelineResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def mock_pipeline_dependencies():
    """Auto-patch external dependencies for all pipeline tests.

    Patches:
        - StateManager: Prevents state file I/O
        - apply_performance_preset: Returns config unchanged

    Rule #1: Simple control flow - single patch context, no nesting
    """
    with patch("ingestforge.core.state.StateManager") as mock_state_cls, patch(
        "ingestforge.core.config_loaders.apply_performance_preset"
    ) as mock_preset:
        # Configure StateManager mock
        mock_state_instance = Mock()
        mock_state_instance.get_state = Mock(return_value=None)
        mock_state_instance.set_state = Mock()
        mock_state_instance.get_all_states = Mock(return_value={})
        mock_state_cls.return_value = mock_state_instance

        # apply_performance_preset returns config unchanged
        mock_preset.side_effect = lambda config: config

        yield


def create_mock_config(**overrides):
    """Create a mock Config object with sensible defaults.

    Rule #4: Helper function to reduce duplication
    Rule #1: Simple, no nesting

    Args:
        **overrides: Attributes to override

    Returns:
        Mock Config object with all necessary attributes
    """
    config = Mock()
    config.data_path = overrides.get("data_path", Path("/tmp/data"))

    # Create project mock with proper name attribute
    project_name = overrides.get("project_name", "test")
    project = Mock()
    project.name = project_name
    config.project = project

    config.ensure_directories = Mock()
    config.chunking = overrides.get("chunking", Mock(strategy="semantic"))
    config.enrichment = overrides.get(
        "enrichment",
        Mock(
            generate_embeddings=False,
            extract_entities=False,
            generate_questions=False,
        ),
    )
    config.pending_path = overrides.get("pending_path", Path("/tmp/pending"))

    # Add any additional overrides
    for key, value in overrides.items():
        if key not in ["project_name"] and not hasattr(config, key):
            setattr(config, key, value)

    return config


# ============================================================================
# Test Classes
# ============================================================================


class TestPipelineResult:
    """Tests for PipelineResult dataclass.

    Rule #4: Focused test class - tests only PipelineResult
    """

    def test_required_fields(self):
        """Test PipelineResult with only required fields."""
        result = PipelineResult(
            document_id="doc123",
            source_file="test.pdf",
            success=True,
            chunks_created=10,
            chunks_indexed=8,
        )

        assert result.document_id == "doc123"
        assert result.source_file == "test.pdf"
        assert result.success is True
        assert result.chunks_created == 10
        assert result.chunks_indexed == 8

    def test_optional_fields(self):
        """Test PipelineResult with optional fields."""
        result = PipelineResult(
            document_id="doc456",
            source_file="test.docx",
            success=False,
            chunks_created=0,
            chunks_indexed=0,
            error_message="File not found",
            processing_time_sec=1.23,
        )

        assert result.error_message == "File not found"
        assert result.processing_time_sec == 1.23

    def test_success_result(self):
        """Test successful result has no error message."""
        result = PipelineResult(
            document_id="doc789",
            source_file="success.pdf",
            success=True,
            chunks_created=15,
            chunks_indexed=15,
        )

        assert result.success is True
        assert result.error_message is None

    def test_failure_result(self):
        """Test failed result has error message."""
        result = PipelineResult(
            document_id="doc000",
            source_file="fail.pdf",
            success=False,
            chunks_created=0,
            chunks_indexed=0,
            error_message="Processing failed",
        )

        assert result.success is False
        assert result.error_message == "Processing failed"

    def test_partial_indexing(self):
        """Test partial indexing (some chunks indexed, not all)."""
        result = PipelineResult(
            document_id="doc111",
            source_file="partial.pdf",
            success=True,
            chunks_created=20,
            chunks_indexed=15,  # Only 15 of 20 indexed
        )

        assert result.chunks_created == 20
        assert result.chunks_indexed == 15
        assert result.chunks_indexed < result.chunks_created


class TestPipelineInitialization:
    """Tests for Pipeline initialization.

    Rule #4: Focused test class - tests only __init__
    """

    def test_default_initialization(self):
        """Test pipeline initializes with default config."""
        config = create_mock_config()

        pipeline = Pipeline(config=config)

        assert pipeline.config is config
        assert pipeline.base_path == Path.cwd()

    def test_custom_config(self):
        """Test pipeline accepts custom config."""
        custom_config = create_mock_config(project_name="custom_project")

        pipeline = Pipeline(config=custom_config)

        assert pipeline.config.project.name == "custom_project"

    def test_custom_base_path(self):
        """Test pipeline accepts custom base path."""
        config = create_mock_config()
        custom_path = Path("/custom/path")

        pipeline = Pipeline(config=config, base_path=custom_path)

        assert pipeline.base_path == custom_path

    def test_safe_file_operations_initialized(self):
        """Test SafeFileOperations is initialized."""
        config = create_mock_config()

        pipeline = Pipeline(config=config)

        assert pipeline._safe_ops is not None

    def test_state_manager_initialized(self):
        """Test StateManager is initialized."""
        config = create_mock_config()

        pipeline = Pipeline(config=config)

        assert pipeline.state_manager is not None

    def test_config_ensure_directories_called(self):
        """Test config.ensure_directories() is called during init."""
        config = create_mock_config()

        Pipeline(config=config)

        config.ensure_directories.assert_called_once()


class TestPipelineProgressCallback:
    """Tests for progress callback functionality.

    Rule #4: Focused test class - tests only progress callbacks
    """

    def test_set_progress_callback(self):
        """Test setting a progress callback."""
        config = create_mock_config()
        pipeline = Pipeline(config=config)
        callback = Mock()

        pipeline.set_progress_callback(callback)

        assert pipeline._progress_callback == callback

    def test_no_callback_by_default(self):
        """Test no progress callback is set by default."""
        config = create_mock_config()

        pipeline = Pipeline(config=config)

        assert pipeline._progress_callback is None

    def test_callback_with_none_doesnt_error(self):
        """Test setting callback to None doesn't cause errors."""
        config = create_mock_config()
        pipeline = Pipeline(config=config)

        pipeline.set_progress_callback(None)

        assert pipeline._progress_callback is None


class TestPipelineFileNotFound:
    """Tests for file not found scenarios.

    Rule #4: Focused test class - tests only file not found errors
    """

    def test_nonexistent_file(self):
        """Test processing nonexistent file returns error result."""
        config = create_mock_config()
        pipeline = Pipeline(config=config)
        nonexistent_file = Path("/nonexistent/file.pdf")

        result = pipeline.process_file(nonexistent_file)

        assert result.success is False
        assert "not found" in result.error_message.lower()
        assert result.chunks_created == 0
        assert result.chunks_indexed == 0


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - PipelineResult: 5 tests
    - Pipeline Initialization: 6 tests
    - Progress Callbacks: 3 tests
    - File Not Found: 1 test

    Total: 15 tests

Design Decisions:
    1. Removed complex lazy loading tests - implementation details
    2. Removed component invalidation tests - internal complexity
    3. Removed deep mocking - makes tests brittle
    4. Focus on public API behavior
    5. Each test is simple, clear, and self-contained
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (No Large Functions/Classes)

Justification:
    - Original 68 tests tested too many implementation details
    - Deep mocking made tests hard to maintain
    - Tests should verify behavior, not implementation
    - Simpler tests are more valuable and maintainable
"""
