"""
Pipeline Context Manager Tests.

Tests for Pipeline.__enter__ and __exit__ context manager support.
GWT-compliant behavioral specifications.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestforge.core.pipeline.pipeline import Pipeline


# =============================================================================
# Mock Config
# =============================================================================


def create_mock_config():
    """Create a minimal mock config for Pipeline tests."""
    config = MagicMock()
    config.data_path = Path("/tmp/test_data")
    config.pending_path = Path("/tmp/test_pending")
    config.project.name = "test-project"
    config.ensure_directories = MagicMock()
    config.ingest.supported_formats = [".pdf", ".txt"]
    config.enrichment = MagicMock()
    config.chunking = MagicMock()
    config.storage = MagicMock()
    config.retrieval = MagicMock()
    config.refinement = MagicMock()
    config.performance_mode = "balanced"  # Required for apply_performance_preset
    return config


# =============================================================================
# Scenario 3: Context Manager Usage Tests
# =============================================================================


class TestPipelineContextManager:
    """Tests for Pipeline context manager support."""

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_enter_returns_self(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT - Scenario 3:
        Given a pipeline used as a context manager.
        When entering the with block.
        Then self is returned.
        """
        config = create_mock_config()
        mock_load.return_value = config
        mock_preset.return_value = config

        with Pipeline() as pipeline:
            assert isinstance(pipeline, Pipeline)

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_exit_calls_teardown_on_enricher(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT - Scenario 3:
        Given a pipeline used as a context manager with an initialized enricher.
        When exiting the with block.
        Then teardown() is called on the enricher.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        mock_enricher = MagicMock()
        mock_enricher.teardown.return_value = True

        pipeline = Pipeline()
        pipeline._enricher = mock_enricher

        # Exit context
        pipeline.__exit__(None, None, None)

        mock_enricher.teardown.assert_called_once()

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_exit_does_not_suppress_exceptions(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline used as a context manager.
        When an exception is raised inside the with block.
        Then the exception is not suppressed.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        with pytest.raises(ValueError):
            with Pipeline() as pipeline:
                raise ValueError("test error")

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_exit_calls_teardown_even_on_exception(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline used as a context manager.
        When an exception is raised inside the with block.
        Then teardown is still called.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        mock_enricher = MagicMock()
        mock_enricher.teardown.return_value = True

        pipeline = Pipeline()
        pipeline._enricher = mock_enricher

        try:
            with pipeline:
                raise ValueError("test error")
        except ValueError:
            pass

        mock_enricher.teardown.assert_called_once()

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_exit_handles_teardown_exception(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline where teardown raises an exception.
        When exiting the with block.
        Then the exception is logged but not propagated.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        mock_enricher = MagicMock()
        mock_enricher.teardown.side_effect = RuntimeError("teardown boom")

        pipeline = Pipeline()
        pipeline._enricher = mock_enricher

        # Should not raise
        pipeline.__exit__(None, None, None)

        mock_enricher.teardown.assert_called_once()

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_exit_calls_storage_close(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline with initialized storage.
        When exiting the with block.
        Then close() is called on storage.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        mock_storage = MagicMock()
        mock_storage.close.return_value = None

        pipeline = Pipeline()
        pipeline._storage = mock_storage

        pipeline.__exit__(None, None, None)

        mock_storage.close.assert_called_once()

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_exit_skips_none_components(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline with no initialized components.
        When exiting the with block.
        Then no errors occur.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        pipeline = Pipeline()
        # Components are None by default

        # Should not raise
        pipeline.__exit__(None, None, None)


# =============================================================================
# Explicit Teardown Tests
# =============================================================================


class TestPipelineTeardown:
    """Tests for explicit Pipeline.teardown() method."""

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_teardown_returns_true_on_success(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline with components.
        When teardown() is called and all succeed.
        Then True is returned.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        mock_enricher = MagicMock()
        mock_enricher.teardown.return_value = True

        pipeline = Pipeline()
        pipeline._enricher = mock_enricher

        result = pipeline.teardown()

        assert result is True
        mock_enricher.teardown.assert_called_once()

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_teardown_returns_false_on_exception(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline where component teardown raises.
        When teardown() is called.
        Then False is returned.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        mock_enricher = MagicMock()
        mock_enricher.teardown.side_effect = RuntimeError("boom")

        pipeline = Pipeline()
        pipeline._enricher = mock_enricher

        result = pipeline.teardown()

        assert result is False

    @patch("ingestforge.core.pipeline.pipeline.load_config")
    @patch("ingestforge.core.pipeline.pipeline.apply_performance_preset")
    @patch("ingestforge.core.pipeline.pipeline.StateManager")
    @patch("ingestforge.core.pipeline.pipeline.SafeFileOperations")
    def test_teardown_can_be_called_multiple_times(
        self, mock_safe_ops, mock_state, mock_preset, mock_load
    ):
        """
        GWT:
        Given a pipeline.
        When teardown() is called multiple times.
        Then no errors occur.
        """
        mock_load.return_value = create_mock_config()
        mock_preset.return_value = mock_load.return_value

        pipeline = Pipeline()

        # First call
        result1 = pipeline.teardown()
        # Second call
        result2 = pipeline.teardown()

        assert result1 is True
        assert result2 is True
