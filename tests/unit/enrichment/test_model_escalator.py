"""
Tests for Multi-Model Fallback Escalator ().

GWT (Given-When-Then) test structure.
NASA JPL Power of Ten compliance verification.
"""

import pytest
from typing import List
from dataclasses import dataclass

from ingestforge.enrichment.model_escalator import (
    ModelEscalator,
    EscalationEvent,
    EscalationStats,
    create_model_escalator,
    DEFAULT_FALLBACK_THRESHOLD,
    MAX_ESCALATION_ATTEMPTS,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockEntity:
    """Mock entity for testing."""

    text: str
    confidence: float


@pytest.fixture
def escalator():
    """Create a ModelEscalator with default settings."""
    return ModelEscalator(
        fast_model="gpt-4o-mini",
        smart_model="gpt-4o",
        fallback_threshold=0.6,
    )


@pytest.fixture
def high_confidence_entities():
    """Entities with confidence above threshold."""
    return [
        MockEntity(text="John Smith", confidence=0.9),
        MockEntity(text="Acme Corp", confidence=0.85),
        MockEntity(text="New York", confidence=0.95),
    ]


@pytest.fixture
def low_confidence_entities():
    """Entities with confidence below threshold."""
    return [
        MockEntity(text="John Smith", confidence=0.4),
        MockEntity(text="Acme Corp", confidence=0.35),
        MockEntity(text="New York", confidence=0.5),
    ]


def get_confidence(entity: MockEntity) -> float:
    """Extract confidence from mock entity."""
    return entity.confidence


# ---------------------------------------------------------------------------
# EscalationEvent Tests
# ---------------------------------------------------------------------------


class TestEscalationEvent:
    """Tests for EscalationEvent model."""

    def test_escalation_event_creation(self):
        """Given valid params, When creating EscalationEvent, Then stores all fields."""
        event = EscalationEvent(
            fast_model="gpt-4o-mini",
            smart_model="gpt-4o",
            fast_confidence=0.4,
            smart_confidence=0.85,
            threshold=0.6,
            latency_fast_ms=100.0,
            latency_smart_ms=500.0,
        )

        assert event.fast_model == "gpt-4o-mini"
        assert event.smart_model == "gpt-4o"
        assert event.fast_confidence == 0.4
        assert event.smart_confidence == 0.85
        assert event.threshold == 0.6

    def test_escalation_event_provenance_entry(self):
        """Given EscalationEvent, When generating provenance, Then formats correctly."""
        event = EscalationEvent(
            fast_model="gpt-4o-mini",
            smart_model="gpt-4o",
            fast_confidence=0.4,
            smart_confidence=0.85,
            threshold=0.6,
        )

        provenance = event.to_provenance_entry()

        assert "escalated" in provenance
        assert "gpt-4o-mini" in provenance
        assert "gpt-4o" in provenance
        assert "0.40" in provenance  # fast confidence
        assert "0.6" in provenance  # threshold

    def test_escalation_event_is_immutable(self):
        """Given EscalationEvent, When trying to modify, Then raises error."""
        event = EscalationEvent(
            fast_model="gpt-4o-mini",
            smart_model="gpt-4o",
            fast_confidence=0.5,
            smart_confidence=0.8,
            threshold=0.6,
        )

        with pytest.raises(Exception):
            event.fast_model = "modified"


# ---------------------------------------------------------------------------
# EscalationStats Tests
# ---------------------------------------------------------------------------


class TestEscalationStats:
    """Tests for EscalationStats dataclass."""

    def test_escalation_rate_calculation(self):
        """Given stats, When calculating escalation_rate, Then returns percentage."""
        stats = EscalationStats(
            total_extractions=10,
            escalation_count=3,
        )

        assert stats.escalation_rate == 30.0

    def test_escalation_rate_zero_extractions(self):
        """Given no extractions, When calculating rate, Then returns 0."""
        stats = EscalationStats()

        assert stats.escalation_rate == 0.0

    def test_avg_latency_calculations(self):
        """Given latency data, When calculating averages, Then returns correct values."""
        stats = EscalationStats(
            total_extractions=10,
            escalation_count=2,
            total_fast_latency_ms=500.0,
            total_smart_latency_ms=800.0,
        )

        assert stats.avg_fast_latency_ms == 50.0
        assert stats.avg_smart_latency_ms == 400.0


# ---------------------------------------------------------------------------
# ModelEscalator Tests
# ---------------------------------------------------------------------------


class TestModelEscalator:
    """Tests for ModelEscalator class."""

    def test_escalator_creation(self, escalator):
        """Given valid params, When creating escalator, Then initializes correctly."""
        assert escalator.fast_model == "gpt-4o-mini"
        assert escalator.smart_model == "gpt-4o"
        assert escalator.fallback_threshold == 0.6

    def test_escalator_default_threshold(self):
        """Given no threshold, When creating escalator, Then uses default."""
        escalator = ModelEscalator(fast_model="fast", smart_model="smart")

        assert escalator.fallback_threshold == DEFAULT_FALLBACK_THRESHOLD

    def test_escalator_assertion_empty_fast_model(self):
        """Given empty fast_model, When creating escalator, Then raises assertion."""
        with pytest.raises(AssertionError):
            ModelEscalator(fast_model="", smart_model="smart")

    def test_escalator_assertion_empty_smart_model(self):
        """Given empty smart_model, When creating escalator, Then raises assertion."""
        with pytest.raises(AssertionError):
            ModelEscalator(fast_model="fast", smart_model="")

    def test_escalator_assertion_invalid_threshold(self):
        """Given invalid threshold, When creating escalator, Then raises assertion."""
        with pytest.raises(AssertionError):
            ModelEscalator(
                fast_model="fast", smart_model="smart", fallback_threshold=1.5
            )

    def test_calculate_confidence_average(self, escalator, high_confidence_entities):
        """Given entities, When calculating confidence, Then returns average."""
        confidence = escalator._calculate_confidence(
            high_confidence_entities, get_confidence
        )

        expected = (0.9 + 0.85 + 0.95) / 3
        assert abs(confidence - expected) < 0.01

    def test_calculate_confidence_empty_list(self, escalator):
        """Given empty list, When calculating confidence, Then returns 0."""
        confidence = escalator._calculate_confidence([], get_confidence)

        assert confidence == 0.0

    def test_should_escalate_low_confidence(self, escalator, low_confidence_entities):
        """Given low confidence, When checking escalation, Then returns True."""
        confidence = escalator._calculate_confidence(
            low_confidence_entities, get_confidence
        )

        assert escalator._should_escalate(confidence) is True

    def test_should_escalate_high_confidence(self, escalator, high_confidence_entities):
        """Given high confidence, When checking escalation, Then returns False."""
        confidence = escalator._calculate_confidence(
            high_confidence_entities, get_confidence
        )

        assert escalator._should_escalate(confidence) is False

    def test_reset_stats(self, escalator):
        """Given stats, When resetting, Then clears all counters."""
        escalator._stats.total_extractions = 10
        escalator._stats.escalation_count = 3

        escalator.reset_stats()

        assert escalator.stats.total_extractions == 0
        assert escalator.stats.escalation_count == 0

    def test_get_summary(self, escalator):
        """Given escalator, When getting summary, Then returns complete dict."""
        summary = escalator.get_summary()

        assert "fast_model" in summary
        assert "smart_model" in summary
        assert "fallback_threshold" in summary
        assert "total_extractions" in summary
        assert "escalation_count" in summary
        assert "escalation_rate" in summary


# ---------------------------------------------------------------------------
# GWT Behavioral Tests
# ---------------------------------------------------------------------------


class TestGWTBehavior:
    """
    Given-When-Then behavioral tests for .

    GWT:
    - Given: A low-confidence extraction result from a fast model.
    - When: The confidence score is < fallback_threshold.
    - Then: Re-run extraction using smart model and log "Escalation" event.
    """

    def test_gwt_high_confidence_no_escalation(
        self, escalator, high_confidence_entities
    ):
        """
        Given: High-confidence extraction result.
        When: Confidence >= threshold.
        Then: No escalation occurs.
        """

        # Mock extract function that returns high confidence entities
        def mock_extract(text: str) -> List[MockEntity]:
            return high_confidence_entities

        result = escalator.extract_with_fallback(
            extract_fn=mock_extract,
            text="Test text",
            confidence_extractor=get_confidence,
        )

        assert result.escalated is False
        assert result.model_used == "gpt-4o-mini"
        assert result.escalation_event is None

    def test_gwt_low_confidence_triggers_escalation(
        self, escalator, low_confidence_entities, high_confidence_entities
    ):
        """
        Given: Low-confidence extraction result from fast model.
        When: Confidence < fallback_threshold.
        Then: Re-run with smart model and log escalation.
        """
        call_count = [0]

        def mock_extract(text: str) -> List[MockEntity]:
            call_count[0] += 1
            # First call (fast model) returns low confidence
            if call_count[0] == 1:
                return low_confidence_entities
            # Second call (smart model) returns high confidence
            return high_confidence_entities

        result = escalator.extract_with_fallback(
            extract_fn=mock_extract,
            text="Test text",
            confidence_extractor=get_confidence,
        )

        assert result.escalated is True
        assert result.model_used == "gpt-4o"
        assert result.escalation_event is not None
        assert call_count[0] == 2  # Both models were called

    def test_gwt_escalation_recorded_in_provenance(
        self, escalator, low_confidence_entities, high_confidence_entities
    ):
        """
        Given: Escalation occurs.
        When: Getting provenance entries.
        Then: Escalation event is in provenance.
        """
        call_count = [0]

        def mock_extract(text: str) -> List[MockEntity]:
            call_count[0] += 1
            if call_count[0] == 1:
                return low_confidence_entities
            return high_confidence_entities

        result = escalator.extract_with_fallback(
            extract_fn=mock_extract,
            text="Test text",
            confidence_extractor=get_confidence,
        )

        # AC: Fallback event recorded in provenance
        assert len(result.provenance_entries) >= 2
        assert any("escalated" in entry for entry in result.provenance_entries)

    def test_gwt_escalation_rate_tracked(self, escalator, low_confidence_entities):
        """
        Given: Multiple extractions with escalations.
        When: Getting escalation rate.
        Then: Rate is correctly calculated.
        """

        def mock_low_extract(text: str) -> List[MockEntity]:
            return low_confidence_entities

        # Run 3 extractions that will all escalate
        for _ in range(3):
            escalator.extract_with_fallback(
                extract_fn=mock_low_extract,
                text="Test text",
                confidence_extractor=get_confidence,
            )

        # All 3 should have escalated
        assert escalator.get_escalation_rate() == 100.0
        assert escalator.stats.escalation_count == 3


# ---------------------------------------------------------------------------
# JPL Power of Ten Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for NASA JPL Power of Ten compliance."""

    def test_jpl_rule_2_max_escalation_attempts(self):
        """JPL Rule #2: Verify MAX_ESCALATION_ATTEMPTS bound."""
        assert MAX_ESCALATION_ATTEMPTS == 3

    def test_jpl_rule_2_escalation_bound_enforced(self):
        """JPL Rule #2: Escalation attempts are bounded."""
        escalator = ModelEscalator(
            fast_model="fast",
            smart_model="smart",
            max_escalation_attempts=100,  # Try to exceed
        )

        # Should be capped at MAX_ESCALATION_ATTEMPTS
        assert escalator.max_escalation_attempts <= MAX_ESCALATION_ATTEMPTS

    def test_jpl_rule_5_assertions_in_init(self):
        """JPL Rule #5: Assert preconditions in __init__."""
        import inspect

        source = inspect.getsource(ModelEscalator.__init__)

        assert "assert" in source
        assert "fast_model" in source
        assert "smart_model" in source

    def test_jpl_rule_5_assertions_in_extract(self):
        """JPL Rule #5: Assert preconditions in extract_with_fallback."""
        import inspect

        source = inspect.getsource(ModelEscalator.extract_with_fallback)

        assert "assert" in source

    def test_jpl_rule_9_type_hints(self):
        """JPL Rule #9: Verify complete type hints."""
        import inspect

        # Check key methods have return annotations
        methods = [
            "extract_with_fallback",
            "get_escalation_rate",
            "get_summary",
            "reset_stats",
        ]

        for method_name in methods:
            method = getattr(ModelEscalator, method_name)
            sig = inspect.signature(method)
            assert (
                sig.return_annotation != inspect.Parameter.empty
            ), f"{method_name} missing return type hint"


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------


class TestFactoryFunction:
    """Tests for create_model_escalator factory function."""

    def test_create_with_defaults(self):
        """Given no args, When creating, Then uses defaults."""
        escalator = create_model_escalator()

        assert escalator.fast_model == "gpt-4o-mini"
        assert escalator.smart_model == "gpt-4o"
        assert escalator.fallback_threshold == DEFAULT_FALLBACK_THRESHOLD

    def test_create_with_custom_models(self):
        """Given custom models, When creating, Then uses them."""
        escalator = create_model_escalator(
            fast_model="claude-3-haiku",
            smart_model="claude-3-opus",
            fallback_threshold=0.7,
        )

        assert escalator.fast_model == "claude-3-haiku"
        assert escalator.smart_model == "claude-3-opus"
        assert escalator.fallback_threshold == 0.7


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_extraction_result(self, escalator):
        """Given extract returns empty, When extracting, Then no escalation."""

        def mock_extract(text: str) -> List[MockEntity]:
            return []

        result = escalator.extract_with_fallback(
            extract_fn=mock_extract,
            text="Test",
            confidence_extractor=get_confidence,
        )

        # Empty result = confidence 0 = should escalate
        # But since smart model also returns empty, escalation happens
        assert result.result == []

    def test_extract_with_exception(self, escalator):
        """Given extract raises, When extracting, Then exception propagates."""

        def mock_extract(text: str) -> List[MockEntity]:
            raise ValueError("Extraction failed")

        with pytest.raises(ValueError):
            escalator.extract_with_fallback(
                extract_fn=mock_extract,
                text="Test",
                confidence_extractor=get_confidence,
            )

    def test_latency_tracking(self, escalator, high_confidence_entities):
        """Given extractions, When checking latency, Then tracked correctly."""

        def mock_extract(text: str) -> List[MockEntity]:
            return high_confidence_entities

        result = escalator.extract_with_fallback(
            extract_fn=mock_extract,
            text="Test",
            confidence_extractor=get_confidence,
        )

        assert result.latency_ms >= 0
        assert escalator.stats.total_fast_latency_ms >= 0
