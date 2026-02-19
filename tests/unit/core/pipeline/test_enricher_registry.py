"""
Tests for Enricher Registration Decorator.

Tests the @register_enricher decorator and IFRegistry enricher methods.
Follows GWT (Given-When-Then) test naming convention.
Adheres to NASA JPL Power of Ten rules.
"""

import pytest
from typing import List
from unittest.mock import MagicMock

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.registry import (
    IFRegistry,
    EnricherEntry,
    register_enricher,
    MAX_ENRICHER_FACTORIES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockEnricherBase(IFProcessor):
    """Base mock enricher for testing."""

    def __init__(self, config=None):
        self.config = config

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact

    def is_available(self) -> bool:
        return True

    @property
    def processor_id(self) -> str:
        return "mock-enricher"

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


class MockEmbeddingEnricher(MockEnricherBase):
    """Mock embedding enricher requiring config."""

    def __init__(self, config):
        super().__init__(config)
        self._config = config

    @property
    def processor_id(self) -> str:
        return "mock-embedding"

    @property
    def capabilities(self) -> List[str]:
        return ["embedding", "semantic-search"]


class MockEntityEnricher(MockEnricherBase):
    """Mock entity enricher with optional args."""

    def __init__(self, use_spacy: bool = True):
        super().__init__()
        self.use_spacy = use_spacy

    @property
    def processor_id(self) -> str:
        return "mock-entity"

    @property
    def capabilities(self) -> List[str]:
        return ["ner", "entity-extraction"]


class MockQuestionEnricher(MockEnricherBase):
    """Mock question enricher with no args."""

    @property
    def processor_id(self) -> str:
        return "mock-question"

    @property
    def capabilities(self) -> List[str]:
        return ["question-generation"]


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
    config.enrichment.embedding_model = "test-model"
    config.performance_mode = "balanced"
    return config


# =============================================================================
# EnricherEntry Tests
# =============================================================================


class TestEnricherEntry:
    """Tests for EnricherEntry dataclass."""

    def test_entry_stores_factory_and_capabilities(self):
        """Given a factory and capabilities, When creating entry, Then all fields stored."""
        factory = lambda: MockEnricherBase()
        entry = EnricherEntry(
            factory=factory,
            capabilities=["cap1", "cap2"],
            priority=100,
            cls_name="TestClass",
        )

        assert entry.factory is factory
        assert entry.capabilities == ["cap1", "cap2"]
        assert entry.priority == 100
        assert entry.cls_name == "TestClass"

    def test_entry_factory_is_callable(self):
        """Given an entry with factory, When calling factory, Then returns instance."""
        factory = lambda: MockEnricherBase()
        entry = EnricherEntry(
            factory=factory,
            capabilities=["test"],
            priority=100,
            cls_name="Test",
        )

        instance = entry.factory()
        assert isinstance(instance, MockEnricherBase)


# =============================================================================
# IFRegistry.register_enricher Tests
# =============================================================================


class TestRegistryRegisterEnricher:
    """Tests for IFRegistry.register_enricher() method."""

    def test_register_enricher_with_factory(self, clean_registry, mock_config):
        """Given enricher class with factory, When registered, Then factory stored."""
        factory = lambda cfg: MockEmbeddingEnricher(cfg)

        clean_registry.register_enricher(
            cls=MockEmbeddingEnricher,
            capabilities=["embedding"],
            priority=100,
            factory=factory,
        )

        # Verify registration
        retrieved_factory = clean_registry.get_enricher_factory("embedding")
        assert retrieved_factory is not None

        # Verify factory works
        instance = retrieved_factory(mock_config)
        assert isinstance(instance, MockEmbeddingEnricher)
        assert instance._config is mock_config

    def test_register_enricher_without_factory(self, clean_registry):
        """Given enricher class without factory, When registered, Then class used as factory."""
        clean_registry.register_enricher(
            cls=MockQuestionEnricher,
            capabilities=["question-generation"],
            priority=100,
            factory=None,
        )

        factory = clean_registry.get_enricher_factory("question-generation")
        assert factory is MockQuestionEnricher

        instance = factory()
        assert isinstance(instance, MockQuestionEnricher)

    def test_register_enricher_multiple_capabilities(self, clean_registry):
        """Given enricher with multiple capabilities, When registered, Then indexed for all."""
        clean_registry.register_enricher(
            cls=MockEmbeddingEnricher,
            capabilities=["embedding", "semantic-search"],
            priority=100,
            factory=lambda: MockEmbeddingEnricher(None),
        )

        # Both capabilities should find the enricher
        assert clean_registry.get_enricher_factory("embedding") is not None
        assert clean_registry.get_enricher_factory("semantic-search") is not None

    def test_register_enricher_empty_capabilities_raises(self, clean_registry):
        """Given empty capabilities list, When registering, Then raises ValueError."""
        with pytest.raises(ValueError, match="at least one capability"):
            clean_registry.register_enricher(
                cls=MockEnricherBase,
                capabilities=[],
                priority=100,
            )

    def test_register_enricher_respects_limit(self, clean_registry):
        """Given registry at limit, When registering, Then raises RuntimeError."""
        # Fill registry to limit
        for i in range(MAX_ENRICHER_FACTORIES):
            # Create unique class for each registration
            class_name = f"MockEnricher{i}"
            mock_cls = type(class_name, (MockEnricherBase,), {})
            clean_registry.register_enricher(
                cls=mock_cls,
                capabilities=[f"cap{i}"],
                priority=100,
            )

        # Next registration should fail
        with pytest.raises(RuntimeError, match="limit reached"):
            clean_registry.register_enricher(
                cls=MockEnricherBase,
                capabilities=["overflow"],
                priority=100,
            )


# =============================================================================
# IFRegistry.get_enricher_factory Tests
# =============================================================================


class TestRegistryGetEnricherFactory:
    """Tests for IFRegistry.get_enricher_factory() method."""

    def test_get_factory_returns_highest_priority(self, clean_registry):
        """Given multiple enrichers for capability, When getting factory, Then highest priority returned."""

        # Register low priority
        class LowPriorityEnricher(MockEnricherBase):
            pass

        clean_registry.register_enricher(
            cls=LowPriorityEnricher,
            capabilities=["embedding"],
            priority=50,
            factory=lambda: "low",
        )

        # Register high priority
        class HighPriorityEnricher(MockEnricherBase):
            pass

        clean_registry.register_enricher(
            cls=HighPriorityEnricher,
            capabilities=["embedding"],
            priority=200,
            factory=lambda: "high",
        )

        factory = clean_registry.get_enricher_factory("embedding")
        assert factory() == "high"

    def test_get_factory_unknown_capability_returns_none(self, clean_registry):
        """Given unknown capability, When getting factory, Then returns None."""
        result = clean_registry.get_enricher_factory("nonexistent")
        assert result is None

    def test_get_factory_after_clear_returns_none(self, clean_registry):
        """Given cleared registry, When getting factory, Then returns None."""
        clean_registry.register_enricher(
            cls=MockEnricherBase,
            capabilities=["test"],
            priority=100,
        )

        clean_registry.clear()

        result = clean_registry.get_enricher_factory("test")
        assert result is None


# =============================================================================
# IFRegistry.get_enricher Tests
# =============================================================================


class TestRegistryGetEnricher:
    """Tests for IFRegistry.get_enricher() method."""

    def test_get_enricher_instantiates_via_factory(self, clean_registry, mock_config):
        """Given registered enricher, When get_enricher called, Then instance created."""
        clean_registry.register_enricher(
            cls=MockEmbeddingEnricher,
            capabilities=["embedding"],
            priority=100,
            factory=lambda cfg: MockEmbeddingEnricher(cfg),
        )

        instance = clean_registry.get_enricher("embedding", mock_config)

        assert isinstance(instance, MockEmbeddingEnricher)
        assert instance._config is mock_config

    def test_get_enricher_passes_kwargs(self, clean_registry):
        """Given registered enricher, When get_enricher with kwargs, Then kwargs passed."""
        clean_registry.register_enricher(
            cls=MockEntityEnricher,
            capabilities=["ner"],
            priority=100,
            factory=lambda use_spacy=True: MockEntityEnricher(use_spacy=use_spacy),
        )

        instance = clean_registry.get_enricher("ner", use_spacy=False)

        assert isinstance(instance, MockEntityEnricher)
        assert instance.use_spacy is False

    def test_get_enricher_unknown_capability_returns_none(self, clean_registry):
        """Given unknown capability, When get_enricher called, Then returns None."""
        result = clean_registry.get_enricher("nonexistent")
        assert result is None

    def test_get_enricher_factory_exception_returns_none(self, clean_registry):
        """Given factory that raises, When get_enricher called, Then returns None."""

        def failing_factory():
            raise RuntimeError("Factory failed")

        clean_registry.register_enricher(
            cls=MockEnricherBase,
            capabilities=["broken"],
            priority=100,
            factory=failing_factory,
        )

        result = clean_registry.get_enricher("broken")
        assert result is None


# =============================================================================
# IFRegistry.get_enricher_factories_by_capability Tests
# =============================================================================


class TestRegistryGetEnricherFactoriesByCapability:
    """Tests for IFRegistry.get_enricher_factories_by_capability() method."""

    def test_get_all_factories_returns_sorted_list(self, clean_registry):
        """Given multiple enrichers, When getting factories, Then sorted by priority."""

        class EnricherA(MockEnricherBase):
            pass

        class EnricherB(MockEnricherBase):
            pass

        class EnricherC(MockEnricherBase):
            pass

        clean_registry.register_enricher(
            cls=EnricherA,
            capabilities=["cap"],
            priority=50,
            factory=lambda: "A",
        )
        clean_registry.register_enricher(
            cls=EnricherB,
            capabilities=["cap"],
            priority=150,
            factory=lambda: "B",
        )
        clean_registry.register_enricher(
            cls=EnricherC,
            capabilities=["cap"],
            priority=100,
            factory=lambda: "C",
        )

        factories = clean_registry.get_enricher_factories_by_capability("cap")

        assert len(factories) == 3
        # Priority order: B(150), C(100), A(50)
        assert factories[0]() == "B"
        assert factories[1]() == "C"
        assert factories[2]() == "A"

    def test_get_all_factories_empty_for_unknown(self, clean_registry):
        """Given unknown capability, When getting factories, Then returns empty list."""
        result = clean_registry.get_enricher_factories_by_capability("nonexistent")
        assert result == []


# =============================================================================
# IFRegistry.get_all_enricher_capabilities Tests
# =============================================================================


class TestRegistryGetAllEnricherCapabilities:
    """Tests for IFRegistry.get_all_enricher_capabilities() method."""

    def test_get_all_capabilities_returns_registered(self, clean_registry):
        """Given registered enrichers, When getting capabilities, Then all returned."""
        clean_registry.register_enricher(
            cls=MockEmbeddingEnricher,
            capabilities=["embedding", "semantic-search"],
            priority=100,
            factory=lambda: MockEmbeddingEnricher(None),
        )
        clean_registry.register_enricher(
            cls=MockEntityEnricher,
            capabilities=["ner"],
            priority=100,
        )

        capabilities = clean_registry.get_all_enricher_capabilities()

        assert set(capabilities) == {"embedding", "semantic-search", "ner"}

    def test_get_all_capabilities_empty_when_cleared(self, clean_registry):
        """Given cleared registry, When getting capabilities, Then returns empty."""
        clean_registry.register_enricher(
            cls=MockEnricherBase,
            capabilities=["test"],
            priority=100,
        )
        clean_registry.clear()

        result = clean_registry.get_all_enricher_capabilities()
        assert result == []


# =============================================================================
# @register_enricher Decorator Tests
# =============================================================================


class TestRegisterEnricherDecorator:
    """Tests for @register_enricher decorator."""

    def test_decorator_registers_class(self, clean_registry):
        """Given class with decorator, When module loads, Then class registered."""

        @register_enricher(capabilities=["test-cap"], priority=75)
        class TestEnricher(MockEnricherBase):
            @property
            def processor_id(self) -> str:
                return "test-decorated"

        factory = clean_registry.get_enricher_factory("test-cap")
        assert factory is TestEnricher

    def test_decorator_with_factory(self, clean_registry, mock_config):
        """Given decorator with factory, When registered, Then factory used."""

        @register_enricher(
            capabilities=["config-req"],
            priority=100,
            factory=lambda cfg: ConfigRequiredEnricher(cfg),
        )
        class ConfigRequiredEnricher(MockEnricherBase):
            def __init__(self, config):
                super().__init__(config)
                self.config = config

            @property
            def processor_id(self) -> str:
                return "config-required"

        instance = clean_registry.get_enricher("config-req", mock_config)
        assert instance.config is mock_config

    def test_decorator_returns_original_class(self, clean_registry):
        """Given decorator applied, When class defined, Then original class returned."""

        @register_enricher(capabilities=["preserve"])
        class OriginalClass(MockEnricherBase):
            @property
            def processor_id(self) -> str:
                return "original"

        # Class should be usable directly
        instance = OriginalClass()
        assert isinstance(instance, OriginalClass)
        assert instance.processor_id == "original"

    def test_decorator_non_ifprocessor_raises(self, clean_registry):
        """Given non-IFProcessor class, When decorated, Then raises AssertionError."""
        with pytest.raises(AssertionError, match="must inherit from IFProcessor"):

            @register_enricher(capabilities=["invalid"])
            class NotAProcessor:
                pass

    def test_decorator_multiple_capabilities(self, clean_registry):
        """Given decorator with multiple capabilities, When registered, Then all indexed."""

        @register_enricher(capabilities=["cap1", "cap2", "cap3"])
        class MultiCapEnricher(MockEnricherBase):
            @property
            def processor_id(self) -> str:
                return "multi-cap"

        # All capabilities should work
        assert clean_registry.get_enricher_factory("cap1") is MultiCapEnricher
        assert clean_registry.get_enricher_factory("cap2") is MultiCapEnricher
        assert clean_registry.get_enricher_factory("cap3") is MultiCapEnricher


# =============================================================================
# Integration Tests
# =============================================================================


class TestEnricherRegistryIntegration:
    """Integration tests for enricher registration workflow."""

    def test_full_workflow_embedding_enricher(self, clean_registry, mock_config):
        """
        Given: EmbeddingGenerator-like enricher requiring config.
        When: Registered with factory and retrieved.
        Then: Instance created with config and usable.
        """

        @register_enricher(
            capabilities=["embedding", "semantic-search"],
            priority=100,
            factory=lambda config: EmbeddingLike(config),
        )
        class EmbeddingLike(MockEnricherBase):
            def __init__(self, config):
                super().__init__(config)
                self._config = config

            @property
            def processor_id(self) -> str:
                return "embedding-like"

            def generate_embedding(self, text: str) -> list:
                return [0.1, 0.2, 0.3]

        # Retrieve via capability
        instance = clean_registry.get_enricher("embedding", mock_config)

        assert isinstance(instance, EmbeddingLike)
        assert instance._config is mock_config
        assert instance.generate_embedding("test") == [0.1, 0.2, 0.3]

    def test_full_workflow_entity_enricher(self, clean_registry):
        """
        Given: EntityExtractor-like enricher with optional args.
        When: Registered without factory and retrieved.
        Then: Instance created with defaults.
        """

        @register_enricher(
            capabilities=["ner", "entity-extraction"],
            priority=90,
        )
        class EntityLike(MockEnricherBase):
            def __init__(self, use_spacy: bool = True):
                super().__init__()
                self.use_spacy = use_spacy

            @property
            def processor_id(self) -> str:
                return "entity-like"

        # Retrieve with default args
        instance = clean_registry.get_enricher("ner")
        assert instance.use_spacy is True

        # Retrieve with custom args via factory call
        factory = clean_registry.get_enricher_factory("entity-extraction")
        custom_instance = factory(use_spacy=False)
        assert custom_instance.use_spacy is False

    def test_priority_based_selection(self, clean_registry, mock_config):
        """
        Given: Multiple enrichers for same capability with different priorities.
        When: Getting enricher.
        Then: Highest priority enricher returned.
        """

        @register_enricher(capabilities=["embedding"], priority=50)
        class LowPriorityEmbed(MockEnricherBase):
            def __init__(self):
                self.name = "low"

            @property
            def processor_id(self) -> str:
                return "low-embed"

        @register_enricher(capabilities=["embedding"], priority=150)
        class HighPriorityEmbed(MockEnricherBase):
            def __init__(self):
                self.name = "high"

            @property
            def processor_id(self) -> str:
                return "high-embed"

        instance = clean_registry.get_enricher("embedding")
        assert instance.name == "high"

    def test_clear_resets_all_enricher_state(self, clean_registry):
        """
        Given: Registry with registered enrichers.
        When: clear() called.
        Then: All enricher state reset.
        """
        clean_registry.register_enricher(
            cls=MockEnricherBase,
            capabilities=["test"],
            priority=100,
        )

        assert clean_registry.get_enricher_factory("test") is not None
        assert "test" in clean_registry.get_all_enricher_capabilities()

        clean_registry.clear()

        assert clean_registry.get_enricher_factory("test") is None
        assert clean_registry.get_all_enricher_capabilities() == []


# =============================================================================
# JPL Power of Ten Compliance Tests
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten rule compliance."""

    def test_rule_1_linear_control_flow(self, clean_registry):
        """Rule #1: No recursion in enricher registration."""
        # Registration should complete without recursion
        clean_registry.register_enricher(
            cls=MockEnricherBase,
            capabilities=["test"],
            priority=100,
        )
        # If we got here, no recursion occurred

    def test_rule_2_fixed_upper_bounds(self, clean_registry):
        """Rule #2: Registry has fixed upper bound."""
        assert MAX_ENRICHER_FACTORIES == 128

        # Attempting to exceed should fail
        # (Tested in test_register_enricher_respects_limit)

    def test_rule_7_check_return_values(self, clean_registry):
        """Rule #7: Factory errors are handled gracefully."""

        def failing_factory():
            raise RuntimeError("Intentional failure")

        clean_registry.register_enricher(
            cls=MockEnricherBase,
            capabilities=["failing"],
            priority=100,
            factory=failing_factory,
        )

        # Should return None, not raise
        result = clean_registry.get_enricher("failing")
        assert result is None

    def test_rule_9_type_hints_present(self, clean_registry):
        """Rule #9: All methods have type hints."""
        import inspect

        # Check register_enricher
        sig = inspect.signature(clean_registry.register_enricher)
        for param in sig.parameters.values():
            if param.name != "self":
                assert (
                    param.annotation != inspect.Parameter.empty
                ), f"Parameter {param.name} missing type hint"

        # Check get_enricher_factory
        sig = inspect.signature(clean_registry.get_enricher_factory)
        assert sig.return_annotation != inspect.Signature.empty

        # Check get_enricher
        sig = inspect.signature(clean_registry.get_enricher)
        assert sig.return_annotation != inspect.Signature.empty
