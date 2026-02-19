"""
Tests for Pipeline Registry Integration.

Verifies Pipeline._select_enricher_by_capability() prefers registry factories.
Follows GWT (Given-When-Then) test naming convention.
"""

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.registry import IFRegistry


# =============================================================================
# Test Fixtures
# =============================================================================


class MockEnricherBase(IFProcessor):
    """Base mock enricher for testing."""

    def __init__(self, config=None):
        self.config = config
        self._source = "unknown"

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        return True

    @property
    def processor_id(self) -> str:
        return f"mock-{self._source}"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return ["mock"]

    @property
    def memory_mb(self) -> int:
        return 100

    def teardown(self) -> bool:
        return True


class RegistryEnricher(MockEnricherBase):
    """Enricher that comes from registry."""

    def __init__(self, config=None):
        super().__init__(config)
        self._source = "registry"

    @property
    def processor_id(self) -> str:
        return "registry-enricher"


class FallbackEnricher(MockEnricherBase):
    """Enricher that comes from fallback factory."""

    def __init__(self):
        super().__init__()
        self._source = "fallback"

    @property
    def processor_id(self) -> str:
        return "fallback-enricher"


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    registry = IFRegistry()
    registry.clear()
    yield registry
    registry.clear()


@pytest.fixture
def mock_config():
    """Provide a mock config object."""
    config = MagicMock()
    config.enrichment = MagicMock()
    config.enrichment.generate_embeddings = True
    config.enrichment.extract_entities = True
    config.enrichment.generate_questions = False
    config.enrichment.generate_summaries = False
    config.enrichment.use_instructor_citation = False
    config.enrichment.compute_quality = False
    config.enrichment.embedding_model = "test-model"
    config.performance_mode = "balanced"
    return config


# =============================================================================
# Registry-First Selection Tests
# =============================================================================


class TestPipelineRegistryIntegration:
    """Tests for Pipeline registry integration ()."""

    def test_registry_factory_preferred_over_fallback(
        self, clean_registry, mock_config
    ):
        """
        Given: Enricher registered via @register_enricher with factory.
        When: Pipeline._select_enricher_by_capability() is called.
        Then: Registry factory is used instead of fallback.
        """
        # Register enricher via registry
        clean_registry.register_enricher(
            cls=RegistryEnricher,
            capabilities=["test-capability"],
            priority=100,
            factory=lambda cfg: RegistryEnricher(cfg),
        )

        # Import Pipeline and mock its dependencies
        with patch("ingestforge.core.pipeline.pipeline.load_config"):
            with patch("ingestforge.core.pipeline.pipeline.apply_performance_preset"):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    # Create pipeline with mock config
                    pipeline = Pipeline.__new__(Pipeline)
                    pipeline.config = mock_config
                    pipeline._progress_callback = None

                    # Call selection method
                    result = pipeline._select_enricher_by_capability(
                        "test-capability", lambda: FallbackEnricher()
                    )

                    # Verify registry enricher was selected
                    assert result is not None
                    assert result.processor_id == "registry-enricher"
                    assert result._source == "registry"
                    assert result.config is mock_config

    def test_fallback_used_when_registry_empty(self, clean_registry, mock_config):
        """
        Given: No enricher registered for capability.
        When: Pipeline._select_enricher_by_capability() is called.
        Then: Fallback factory is used.
        """
        with patch("ingestforge.core.pipeline.pipeline.load_config"):
            with patch("ingestforge.core.pipeline.pipeline.apply_performance_preset"):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline.__new__(Pipeline)
                    pipeline.config = mock_config
                    pipeline._progress_callback = None

                    result = pipeline._select_enricher_by_capability(
                        "unregistered-capability", lambda: FallbackEnricher()
                    )

                    assert result is not None
                    assert result.processor_id == "fallback-enricher"
                    assert result._source == "fallback"

    def test_config_passed_to_registry_factory(self, clean_registry, mock_config):
        """
        Given: Enricher registered with factory requiring config.
        When: Pipeline requests enricher.
        Then: Config is passed to factory.
        """
        received_config = []

        def capture_config(cfg):
            received_config.append(cfg)
            return RegistryEnricher(cfg)

        clean_registry.register_enricher(
            cls=RegistryEnricher,
            capabilities=["config-test"],
            priority=100,
            factory=capture_config,
        )

        with patch("ingestforge.core.pipeline.pipeline.load_config"):
            with patch("ingestforge.core.pipeline.pipeline.apply_performance_preset"):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline.__new__(Pipeline)
                    pipeline.config = mock_config
                    pipeline._progress_callback = None

                    pipeline._select_enricher_by_capability(
                        "config-test", lambda: FallbackEnricher()
                    )

                    assert len(received_config) == 1
                    assert received_config[0] is mock_config

    def test_factory_failure_falls_back_gracefully(self, clean_registry, mock_config):
        """
        Given: Registry factory raises exception.
        When: Pipeline requests enricher.
        Then: Fallback factory is used.
        """

        def failing_factory(cfg):
            raise RuntimeError("Factory failed")

        clean_registry.register_enricher(
            cls=RegistryEnricher,
            capabilities=["failing"],
            priority=100,
            factory=failing_factory,
        )

        with patch("ingestforge.core.pipeline.pipeline.load_config"):
            with patch("ingestforge.core.pipeline.pipeline.apply_performance_preset"):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline.__new__(Pipeline)
                    pipeline.config = mock_config
                    pipeline._progress_callback = None

                    result = pipeline._select_enricher_by_capability(
                        "failing", lambda: FallbackEnricher()
                    )

                    # Should gracefully fall back
                    assert result is not None
                    assert result.processor_id == "fallback-enricher"


class TestPipelineRegistryPriority:
    """Tests for priority-based selection in Pipeline."""

    def test_highest_priority_factory_selected(self, clean_registry, mock_config):
        """
        Given: Multiple enrichers registered with different priorities.
        When: Pipeline requests enricher.
        Then: Highest priority enricher is selected.
        """

        class LowPriorityEnricher(MockEnricherBase):
            def __init__(self, config=None):
                super().__init__(config)
                self.priority_level = "low"

            @property
            def processor_id(self) -> str:
                return "low-priority"

        class HighPriorityEnricher(MockEnricherBase):
            def __init__(self, config=None):
                super().__init__(config)
                self.priority_level = "high"

            @property
            def processor_id(self) -> str:
                return "high-priority"

        clean_registry.register_enricher(
            cls=LowPriorityEnricher,
            capabilities=["priority-test"],
            priority=50,
            factory=lambda cfg: LowPriorityEnricher(cfg),
        )

        clean_registry.register_enricher(
            cls=HighPriorityEnricher,
            capabilities=["priority-test"],
            priority=150,
            factory=lambda cfg: HighPriorityEnricher(cfg),
        )

        with patch("ingestforge.core.pipeline.pipeline.load_config"):
            with patch("ingestforge.core.pipeline.pipeline.apply_performance_preset"):
                with patch("ingestforge.core.pipeline.pipeline.StateManager"):
                    from ingestforge.core.pipeline.pipeline import Pipeline

                    pipeline = Pipeline.__new__(Pipeline)
                    pipeline.config = mock_config
                    pipeline._progress_callback = None

                    result = pipeline._select_enricher_by_capability(
                        "priority-test", lambda: FallbackEnricher()
                    )

                    assert result.processor_id == "high-priority"
                    assert result.priority_level == "high"
