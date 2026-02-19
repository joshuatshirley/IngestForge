"""Tests for legal facts vs. opinions classifier.

Tests multi-agent legal text classification."""

from __future__ import annotations

import pytest

from ingestforge.agent.legal_classifier import (
    LegalRole,
    LegalClassification,
    ClassificationPrompts,
    LegalClassifier,
    create_legal_classifier,
    classify_legal_text,
    MAX_TEXT_LENGTH,
    MAX_CHUNKS_PER_BATCH,
    MAX_SECONDARY_ROLES,
)

# Import mock LLM from test fixtures
from tests.fixtures.agents import MockLLM

# LegalRole tests


class TestLegalRole:
    """Tests for LegalRole enum."""

    def test_roles_defined(self) -> None:
        """Test all roles are defined."""
        roles = [r.value for r in LegalRole]

        assert "FACTS" in roles
        assert "HOLDING" in roles
        assert "DICTA" in roles
        assert "DISSENT" in roles
        assert "CONCURRENCE" in roles
        assert "ANALYSIS" in roles
        assert "PROCEDURAL" in roles
        assert "UNKNOWN" in roles

    def test_role_count(self) -> None:
        """Test correct number of roles."""
        assert len(LegalRole) == 8


# LegalClassification tests


class TestLegalClassification:
    """Tests for LegalClassification dataclass."""

    def test_classification_creation(self) -> None:
        """Test creating a classification."""
        classification = LegalClassification(
            text="The court held that the defendant was liable.",
            role="HOLDING",
            confidence=0.9,
        )

        assert classification.text == "The court held that the defendant was liable."
        assert classification.role == "HOLDING"
        assert classification.confidence == 0.9

    def test_classification_with_secondary_roles(self) -> None:
        """Test classification with secondary roles."""
        classification = LegalClassification(
            text="Test text",
            role="ANALYSIS",
            secondary_roles=["FACTS", "HOLDING"],
        )

        assert len(classification.secondary_roles) == 2
        assert "FACTS" in classification.secondary_roles

    def test_text_truncation(self) -> None:
        """Test long text is truncated."""
        long_text = "x" * 10000
        classification = LegalClassification(
            text=long_text,
            role="FACTS",
        )

        assert len(classification.text) == MAX_TEXT_LENGTH

    def test_secondary_roles_truncation(self) -> None:
        """Test secondary roles list is truncated."""
        many_roles = ["FACTS", "HOLDING", "DICTA", "ANALYSIS", "PROCEDURAL"]
        classification = LegalClassification(
            text="Test",
            role="HOLDING",
            secondary_roles=many_roles,
        )

        assert len(classification.secondary_roles) == MAX_SECONDARY_ROLES

    def test_confidence_clamping(self) -> None:
        """Test confidence is clamped to 0-1 range."""
        too_high = LegalClassification(text="Test", role="FACTS", confidence=1.5)
        too_low = LegalClassification(text="Test", role="FACTS", confidence=-0.5)

        assert too_high.confidence == 1.0
        assert too_low.confidence == 0.0

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        classification = LegalClassification(
            text="Test text",
            role="HOLDING",
            confidence=0.85,
            secondary_roles=["ANALYSIS"],
            reasoning="Clear ruling language",
        )

        d = classification.to_dict()

        assert d["role"] == "HOLDING"
        assert d["confidence"] == 0.85
        assert d["reasoning"] == "Clear ruling language"

    def test_is_binding_property(self) -> None:
        """Test is_binding property."""
        holding = LegalClassification(text="Test", role="HOLDING")
        facts = LegalClassification(text="Test", role="FACTS")
        dicta = LegalClassification(text="Test", role="DICTA")

        assert holding.is_binding is True
        assert facts.is_binding is True
        assert dicta.is_binding is False

    def test_is_opinion_property(self) -> None:
        """Test is_opinion property."""
        dicta = LegalClassification(text="Test", role="DICTA")
        dissent = LegalClassification(text="Test", role="DISSENT")
        holding = LegalClassification(text="Test", role="HOLDING")

        assert dicta.is_opinion is True
        assert dissent.is_opinion is True
        assert holding.is_opinion is False


# ClassificationPrompts tests


class TestClassificationPrompts:
    """Tests for ClassificationPrompts class."""

    def test_primary_classifier_prompt(self) -> None:
        """Test primary classifier prompt generation."""
        prompt = ClassificationPrompts.primary_classifier_prompt(
            text="The court held that the statute was unconstitutional.",
        )

        assert "unconstitutional" in prompt
        assert "FACTS" in prompt
        assert "HOLDING" in prompt
        assert "Role:" in prompt

    def test_primary_classifier_prompt_with_context(self) -> None:
        """Test prompt generation with context."""
        prompt = ClassificationPrompts.primary_classifier_prompt(
            text="The court held that...",
            context="This case involves a dispute over property rights.",
        )

        assert "property rights" in prompt
        assert "Surrounding Context" in prompt

    def test_context_aware_prompt(self) -> None:
        """Test context-aware prompt generation."""
        prompt = ClassificationPrompts.context_aware_prompt(
            text="Therefore, we reverse the lower court's decision.",
            preceding="The evidence clearly shows that...",
            following="The case is remanded for further proceedings.",
        )

        assert "[TARGET TEXT]" in prompt
        assert "[PRECEDING]" in prompt
        assert "[FOLLOWING]" in prompt
        assert "reverse" in prompt


# LegalClassifier tests


class TestLegalClassifier:
    """Tests for LegalClassifier class."""

    def test_classifier_creation(self) -> None:
        """Test creating a classifier."""
        llm = MockLLM()
        classifier = LegalClassifier(llm_client=llm)

        assert classifier is not None

    def test_classifier_requires_llm(self) -> None:
        """Test classifier requires LLM client."""
        with pytest.raises(ValueError, match="llm_client cannot be None"):
            LegalClassifier(llm_client=None)  # type: ignore


class TestFactsClassification:
    """Tests for facts classification."""

    def test_classify_facts(self) -> None:
        """Test classifying factual statements."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: FACTS
Confidence: 0.92
Secondary: None
Reasoning: This passage describes the events leading to the case."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "On January 15, 2020, the plaintiff entered into a contract with the defendant."
        )

        assert result.role == "FACTS"
        assert result.confidence == 0.92
        assert "events" in result.reasoning.lower()

    def test_classify_procedural_history(self) -> None:
        """Test classifying procedural history as facts."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: FACTS
Confidence: 0.88
Secondary: PROCEDURAL
Reasoning: Describes case procedural history."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "The district court granted summary judgment in favor of the plaintiff."
        )

        assert result.role == "FACTS"
        assert "PROCEDURAL" in result.secondary_roles


class TestHoldingClassification:
    """Tests for holding classification."""

    def test_classify_holding(self) -> None:
        """Test classifying court holdings."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: HOLDING
Confidence: 0.95
Secondary: None
Reasoning: This is the court's explicit ruling on the matter."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "We hold that the statute violates the First Amendment."
        )

        assert result.role == "HOLDING"
        assert result.confidence == 0.95
        assert result.is_binding is True

    def test_classify_ruling_language(self) -> None:
        """Test classifying various ruling language patterns."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: HOLDING
Confidence: 0.91
Secondary: None
Reasoning: Contains definitive ruling language."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "Accordingly, the judgment of the lower court is reversed."
        )

        assert result.role == "HOLDING"


class TestDictaClassification:
    """Tests for dicta classification."""

    def test_classify_dicta(self) -> None:
        """Test classifying obiter dicta."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: DICTA
Confidence: 0.87
Secondary: ANALYSIS
Reasoning: Hypothetical discussion not necessary for the holding."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "Even if the defendant had raised this argument, it would likely fail."
        )

        assert result.role == "DICTA"
        assert result.is_binding is False
        assert result.is_opinion is True

    def test_dicta_vs_holding_distinction(self) -> None:
        """Test distinguishing dicta from holding."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: DICTA
Confidence: 0.82
Secondary: HOLDING
Reasoning: Contains commentary beyond what is necessary for decision.""",
                """Role: HOLDING
Confidence: 0.93
Secondary: None
Reasoning: This is the actual binding ruling.""",
            ]
        )

        classifier = LegalClassifier(llm_client=llm)

        dicta_result = classifier.classify(
            "It might be noted that in a different factual scenario..."
        )
        assert dicta_result.role == "DICTA"
        assert "HOLDING" in dicta_result.secondary_roles

        holding_result = classifier.classify("We therefore affirm the judgment below.")
        assert holding_result.role == "HOLDING"


class TestDissentClassification:
    """Tests for dissent classification."""

    def test_classify_dissent(self) -> None:
        """Test classifying dissenting opinions."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: DISSENT
Confidence: 0.94
Secondary: None
Reasoning: Clearly labeled as dissenting opinion with contrary view."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "I respectfully dissent. The majority's analysis fails to consider..."
        )

        assert result.role == "DISSENT"
        assert result.is_opinion is True
        assert result.is_binding is False


class TestConcurrenceClassification:
    """Tests for concurrence classification."""

    def test_classify_concurrence(self) -> None:
        """Test classifying concurring opinions."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: CONCURRENCE
Confidence: 0.89
Secondary: ANALYSIS
Reasoning: Agrees with result but provides different reasoning."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "I concur in the judgment but write separately to address..."
        )

        assert result.role == "CONCURRENCE"
        assert result.is_opinion is True


class TestAnalysisClassification:
    """Tests for analysis classification."""

    def test_classify_analysis(self) -> None:
        """Test classifying legal analysis."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: ANALYSIS
Confidence: 0.86
Secondary: HOLDING
Reasoning: Applies legal framework to facts."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify(
            "Under the three-prong test established in Lemon v. Kurtzman..."
        )

        assert result.role == "ANALYSIS"


class TestMultiRoleHandling:
    """Tests for multi-role handling."""

    def test_multiple_secondary_roles(self) -> None:
        """Test parsing multiple secondary roles."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: ANALYSIS
Confidence: 0.75
Secondary: [FACTS, HOLDING]
Reasoning: Contains both factual recitation and reasoning toward holding."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify("Complex legal passage...")

        assert result.role == "ANALYSIS"
        assert "FACTS" in result.secondary_roles
        assert "HOLDING" in result.secondary_roles

    def test_ambiguous_passage(self) -> None:
        """Test handling ambiguous passages."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: ANALYSIS
Confidence: 0.55
Secondary: [DICTA, HOLDING]
Reasoning: Could be interpreted as either binding or non-binding."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify("While not strictly necessary, we note that...")

        assert result.confidence < 0.7
        assert len(result.secondary_roles) == 2


class TestContextAwareClassification:
    """Tests for context-aware classification."""

    def test_classify_with_context(self) -> None:
        """Test context-aware classification."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Role: HOLDING
Confidence: 0.90
Secondary: None
Reasoning: Position in document after analysis suggests this is the ruling."""
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify_with_context(
            text="We therefore reverse.",
            preceding="Having analyzed the statutory language...",
            following="The case is remanded for further proceedings.",
        )

        assert result.role == "HOLDING"


class TestDocumentClassification:
    """Tests for full document classification."""

    def test_classify_document(self) -> None:
        """Test classifying document chunks."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Role: FACTS\nConfidence: 0.9\nSecondary: None\nReasoning: Background",
                "Role: ANALYSIS\nConfidence: 0.85\nSecondary: None\nReasoning: Reasoning",
                "Role: HOLDING\nConfidence: 0.95\nSecondary: None\nReasoning: Decision",
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        chunks = [
            {"text": "On January 1, plaintiff filed suit."},
            {"text": "Applying the standard from prior cases..."},
            {"text": "We therefore affirm the judgment."},
        ]

        results = classifier.classify_document(chunks)

        assert len(results) == 3
        assert results[0].role == "FACTS"
        assert results[1].role == "ANALYSIS"
        assert results[2].role == "HOLDING"

    def test_classify_empty_document(self) -> None:
        """Test classifying empty document."""
        llm = MockLLM()
        classifier = LegalClassifier(llm_client=llm)

        results = classifier.classify_document([])

        assert results == []


class TestChunkEnrichment:
    """Tests for chunk enrichment."""

    def test_enrich_chunks(self) -> None:
        """Test enriching chunks with legal metadata."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Role: FACTS\nConfidence: 0.88\nSecondary: None\nReasoning: Background facts",
                "Role: HOLDING\nConfidence: 0.94\nSecondary: None\nReasoning: Court ruling",
            ]
        )

        classifier = LegalClassifier(llm_client=llm)
        chunks = [
            {"text": "The plaintiff was injured.", "id": "chunk_1"},
            {"text": "We hold for the plaintiff.", "id": "chunk_2"},
        ]

        enriched = classifier.enrich_chunks(chunks)

        assert len(enriched) == 2
        assert enriched[0]["legal_role"] == "FACTS"
        assert enriched[0]["legal_confidence"] == 0.88
        assert enriched[0]["id"] == "chunk_1"  # Original data preserved
        assert enriched[1]["legal_role"] == "HOLDING"
        assert enriched[1]["legal_is_binding"] is True

    def test_enrich_empty_chunks(self) -> None:
        """Test enriching empty chunk list."""
        llm = MockLLM()
        classifier = LegalClassifier(llm_client=llm)

        result = classifier.enrich_chunks([])

        assert result == []


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self) -> None:
        """Test classifying empty text."""
        llm = MockLLM()
        classifier = LegalClassifier(llm_client=llm)

        result = classifier.classify("")

        assert result.role == "UNKNOWN"
        assert result.confidence == 0.0

    def test_whitespace_text(self) -> None:
        """Test classifying whitespace-only text."""
        llm = MockLLM()
        classifier = LegalClassifier(llm_client=llm)

        result = classifier.classify("   \n\t  ")

        assert result.role == "UNKNOWN"

    def test_malformed_llm_response(self) -> None:
        """Test handling malformed LLM response."""
        llm = MockLLM()
        llm.set_responses(["This response has no proper format."])

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify("Some legal text.")

        # Should fallback gracefully
        assert result.role in [r.value for r in LegalRole]

    def test_empty_llm_response(self) -> None:
        """Test handling empty LLM response."""
        llm = MockLLM()
        llm.set_responses([""])

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify("Some legal text.")

        assert result.role == "UNKNOWN"

    def test_missing_confidence(self) -> None:
        """Test handling response without confidence."""
        llm = MockLLM()
        llm.set_responses(
            ["Role: FACTS\nSecondary: None\nReasoning: No confidence provided"]
        )

        classifier = LegalClassifier(llm_client=llm)
        result = classifier.classify("Some text.")

        assert result.role == "FACTS"
        assert result.confidence == 0.5  # Default

    def test_very_long_text(self) -> None:
        """Test handling very long text."""
        llm = MockLLM()
        llm.set_responses(
            ["Role: ANALYSIS\nConfidence: 0.8\nSecondary: None\nReasoning: Long text"]
        )

        classifier = LegalClassifier(llm_client=llm)
        long_text = "legal content " * 1000
        result = classifier.classify(long_text)

        assert result.role == "ANALYSIS"
        assert len(result.text) == MAX_TEXT_LENGTH


# Factory function tests


class TestCreateLegalClassifier:
    """Tests for create_legal_classifier factory."""

    def test_create(self) -> None:
        """Test creating classifier via factory."""
        llm = MockLLM()
        classifier = create_legal_classifier(llm_client=llm)

        assert isinstance(classifier, LegalClassifier)

    def test_create_with_config(self) -> None:
        """Test creating with custom config."""
        from ingestforge.llm.base import GenerationConfig

        llm = MockLLM()
        config = GenerationConfig(max_tokens=1000, temperature=0.1)
        classifier = create_legal_classifier(llm_client=llm, config=config)

        assert classifier._config.max_tokens == 1000
        assert classifier._config.temperature == 0.1


class TestClassifyLegalTextFunction:
    """Tests for classify_legal_text convenience function."""

    def test_classify_legal_text(self) -> None:
        """Test convenience function."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Role: HOLDING\nConfidence: 0.92\nSecondary: None\nReasoning: Court ruling"
            ]
        )

        result = classify_legal_text(
            text="We hold that the statute is unconstitutional.",
            llm_client=llm,
        )

        assert result.role == "HOLDING"
        assert result.confidence == 0.92


# Constant tests


class TestConstants:
    """Tests for module constants."""

    def test_max_text_length(self) -> None:
        """Test MAX_TEXT_LENGTH is reasonable."""
        assert MAX_TEXT_LENGTH > 0
        assert MAX_TEXT_LENGTH == 5000

    def test_max_chunks_per_batch(self) -> None:
        """Test MAX_CHUNKS_PER_BATCH is reasonable."""
        assert MAX_CHUNKS_PER_BATCH > 0
        assert MAX_CHUNKS_PER_BATCH == 50

    def test_max_secondary_roles(self) -> None:
        """Test MAX_SECONDARY_ROLES is reasonable."""
        assert MAX_SECONDARY_ROLES > 0
        assert MAX_SECONDARY_ROLES == 3
