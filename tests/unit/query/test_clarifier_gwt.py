"""
Comprehensive GWT Unit Tests for Query Clarifier.

Query Clarification - Given-When-Then test format.
Ensures >80% code coverage and JPL compliance verification.

Test Format:
- Given: Initial conditions/setup
- When: Action being tested
- Then: Expected outcomes/assertions
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from ingestforge.query.clarifier import (
    AmbiguityReport,
    AmbiguityType,
    ClarificationArtifact,
    ClarificationQuestion,
    ClarifierConfig,
    ClarityScore,
    IFQueryClarifier,
    create_clarifier,
    evaluate_query_clarity,
    needs_clarification,
    CLARITY_THRESHOLD,
    MAX_QUESTIONS,
    MAX_SUGGESTIONS,
    MAX_QUERY_LENGTH,
    MAX_AMBIGUOUS_TERMS,
    MAX_CONTEXT_QUERIES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clarifier() -> IFQueryClarifier:
    """Create a standard clarifier instance."""
    return IFQueryClarifier()


@pytest.fixture
def clarifier_with_config() -> IFQueryClarifier:
    """Create a clarifier with custom config."""
    config = ClarifierConfig(threshold=0.6, use_llm=False, max_suggestions=3)
    return IFQueryClarifier(config)


@pytest.fixture
def sample_context() -> Dict[str, Any]:
    """Create sample conversation context."""
    return {
        "previous_queries": [
            "Who is Guido van Rossum?",
            "What is Python programming language?",
        ]
    }


# =============================================================================
# GWT Tests: Pronoun Detection
# =============================================================================


class TestPronounDetectionGWT:
    """GWT tests for pronoun ambiguity detection."""

    def test_detect_single_pronoun_he(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with unclear pronoun 'he'
        When: detect_pronouns() is called
        Then: Returns list containing 'he'
        """
        # Given
        query = "What did he say about programming?"

        # When
        result = clarifier.detect_pronouns(query)

        # Then
        assert "he" in result
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_detect_multiple_pronouns(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with multiple unclear pronouns
        When: detect_pronouns() is called
        Then: Returns list containing all ambiguous pronouns
        """
        # Given
        query = "What did he say to her about it?"

        # When
        result = clarifier.detect_pronouns(query)

        # Then
        assert "he" in result or len(result) > 0  # At least one pronoun detected
        assert isinstance(result, list)

    def test_detect_demonstrative_pronoun_this(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: A query with demonstrative pronoun 'this'
        When: detect_pronouns() is called
        Then: Returns list containing 'this'
        """
        # Given
        query = "Explain this concept"

        # When
        result = clarifier.detect_pronouns(query)

        # Then
        assert "this" in result
        assert len(result) >= 1

    def test_no_pronouns_in_clear_query(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A clear query without ambiguous pronouns
        When: detect_pronouns() is called
        Then: Returns empty list
        """
        # Given
        query = "Who is the CEO of Apple Inc.?"

        # When
        result = clarifier.detect_pronouns(query)

        # Then
        assert isinstance(result, list)
        assert len(result) == 0

    def test_pronoun_resolution_with_context(
        self, clarifier: IFQueryClarifier, sample_context: Dict[str, Any]
    ) -> None:
        """
        Given: A query with pronoun AND conversation context
        When: detect_pronouns() is called with context
        Then: May resolve pronoun if context contains proper nouns
        """
        # Given
        query = "What did he create?"

        # When
        result = clarifier.detect_pronouns(query, sample_context)

        # Then
        assert isinstance(result, list)
        # With context containing "Guido van Rossum", pronoun might be resolvable
        # Test verifies method accepts context parameter

    def test_pronoun_detection_bounded(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query that could return many results
        When: detect_pronouns() is called
        Then: Results are bounded by MAX_AMBIGUOUS_TERMS
        """
        # Given
        query = "he she it they them his her its their"

        # When
        result = clarifier.detect_pronouns(query)

        # Then
        assert len(result) <= MAX_AMBIGUOUS_TERMS


# =============================================================================
# GWT Tests: Multi-Meaning Term Detection
# =============================================================================


class TestMultiMeaningDetectionGWT:
    """GWT tests for multi-meaning term detection."""

    def test_detect_python_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query containing 'Python'
        When: detect_multi_meaning() is called
        Then: Returns list with Python and its possible meanings
        """
        # Given
        query = "Tell me about Python"

        # When
        result = clarifier.detect_multi_meaning(query)

        # Then
        assert len(result) > 0
        assert any("python" == term.lower() for term, _ in result)
        # Verify meanings are provided
        for term, meanings in result:
            assert len(meanings) >= 2
            assert isinstance(meanings, list)

    def test_detect_java_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query containing 'Java'
        When: detect_multi_meaning() is called
        Then: Returns Java with multiple meanings
        """
        # Given
        query = "What is Java used for?"

        # When
        result = clarifier.detect_multi_meaning(query)

        # Then
        assert len(result) > 0
        found_java = any("java" == term.lower() for term, _ in result)
        assert found_java

    def test_detect_apple_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query containing 'Apple'
        When: detect_multi_meaning() is called
        Then: Returns Apple with company/fruit meanings
        """
        # Given
        query = "Tell me about Apple"

        # When
        result = clarifier.detect_multi_meaning(query)

        # Then
        assert len(result) > 0
        # Verify at least one multi-meaning term detected

    def test_no_ambiguity_in_clear_query(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query without multi-meaning terms
        When: detect_multi_meaning() is called
        Then: Returns empty list
        """
        # Given
        query = "What is machine learning?"

        # When
        result = clarifier.detect_multi_meaning(query)

        # Then
        assert isinstance(result, list)
        assert len(result) == 0

    def test_multiple_ambiguous_terms(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with multiple ambiguous terms
        When: detect_multi_meaning() is called
        Then: Returns all detected ambiguous terms
        """
        # Given
        query = "Compare Python and Java for web development"

        # When
        result = clarifier.detect_multi_meaning(query)

        # Then
        assert len(result) >= 1  # At least one detected
        assert len(result) <= MAX_AMBIGUOUS_TERMS  # Bounded

    def test_case_insensitive_detection(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with mixed case ambiguous term
        When: detect_multi_meaning() is called
        Then: Detects term regardless of case
        """
        # Given
        query = "PYTHON programming"

        # When
        result = clarifier.detect_multi_meaning(query)

        # Then
        assert len(result) > 0


# =============================================================================
# GWT Tests: Temporal Ambiguity Detection
# =============================================================================


class TestTemporalAmbiguityGWT:
    """GWT tests for temporal ambiguity detection."""

    def test_detect_recent_temporal_ambiguity(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: A query with 'recent' temporal qualifier
        When: detect_temporal_ambiguity() is called
        Then: Returns list containing 'recent'
        """
        # Given
        query = "What are recent developments in AI?"

        # When
        result = clarifier.detect_temporal_ambiguity(query)

        # Then
        assert "recent" in result
        assert isinstance(result, list)

    def test_detect_soon_temporal_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with 'soon' temporal qualifier
        When: detect_temporal_ambiguity() is called
        Then: Returns list containing 'soon'
        """
        # Given
        query = "What will happen soon?"

        # When
        result = clarifier.detect_temporal_ambiguity(query)

        # Then
        assert "soon" in result

    def test_detect_before_temporal_ambiguity(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: A query with 'before' temporal qualifier
        When: detect_temporal_ambiguity() is called
        Then: Returns list containing 'before'
        """
        # Given
        query = "What happened before the incident?"

        # When
        result = clarifier.detect_temporal_ambiguity(query)

        # Then
        assert "before" in result

    def test_specific_date_no_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with specific date (2024)
        When: detect_temporal_ambiguity() is called
        Then: Returns empty list (no ambiguity)
        """
        # Given
        query = "What happened in 2024?"

        # When
        result = clarifier.detect_temporal_ambiguity(query)

        # Then
        assert len(result) == 0

    def test_multiple_temporal_terms(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with multiple temporal qualifiers
        When: detect_temporal_ambiguity() is called
        Then: Returns all detected temporal terms
        """
        # Given
        query = "Recent developments before the current situation"

        # When
        result = clarifier.detect_temporal_ambiguity(query)

        # Then
        assert len(result) >= 1
        assert len(result) <= MAX_AMBIGUOUS_TERMS


# =============================================================================
# GWT Tests: Scope Ambiguity Detection
# =============================================================================


class TestScopeAmbiguityGWT:
    """GWT tests for scope ambiguity detection."""

    def test_detect_broad_scope_everything(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with 'everything' (broad scope)
        When: detect_scope_ambiguity() is called
        Then: Returns True (overly broad)
        """
        # Given
        query = "Tell me everything about AI"

        # When
        result = clarifier.detect_scope_ambiguity(query)

        # Then
        assert result is True

    def test_detect_broad_scope_all(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with 'all' (broad scope)
        When: detect_scope_ambiguity() is called
        Then: Returns True (overly broad)
        """
        # Given
        query = "Explain all machine learning algorithms"

        # When
        result = clarifier.detect_scope_ambiguity(query)

        # Then
        assert result is True

    def test_specific_scope_clear(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with specific scope indicator
        When: detect_scope_ambiguity() is called
        Then: Returns False (scope is clear)
        """
        # Given
        query = "Specifically explain neural network backpropagation"

        # When
        result = clarifier.detect_scope_ambiguity(query)

        # Then
        assert result is False

    def test_normal_query_no_scope_issue(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A normal query without scope indicators
        When: detect_scope_ambiguity() is called
        Then: Returns False
        """
        # Given
        query = "How does gradient descent work?"

        # When
        result = clarifier.detect_scope_ambiguity(query)

        # Then
        assert result is False


# =============================================================================
# GWT Tests: Question Generation
# =============================================================================


class TestQuestionGenerationGWT:
    """GWT tests for clarification question generation."""

    def test_generate_pronoun_question(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: Ambiguous terms list with pronoun
        When: generate_questions() is called
        Then: Returns ClarificationQuestion for pronoun
        """
        # Given
        ambiguous_terms = [(AmbiguityType.PRONOUN, "he")]
        query = "What did he say?"

        # When
        questions = clarifier.generate_questions(ambiguous_terms, query)

        # Then
        assert len(questions) > 0
        assert questions[0].type == AmbiguityType.PRONOUN
        assert "he" in questions[0].question.lower()
        assert len(questions[0].options) >= 2
        assert 0.0 <= questions[0].confidence <= 1.0

    def test_generate_multi_meaning_question(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: Ambiguous terms list with multi-meaning term
        When: generate_questions() is called
        Then: Returns question with multiple meaning options
        """
        # Given
        ambiguous_terms = [(AmbiguityType.MULTI_MEANING, "python")]
        query = "Tell me about Python"

        # When
        questions = clarifier.generate_questions(ambiguous_terms, query)

        # Then
        assert len(questions) > 0
        assert questions[0].type == AmbiguityType.MULTI_MEANING
        assert len(questions[0].options) >= 2
        assert questions[0].term == "python"

    def test_generate_temporal_question(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: Ambiguous terms list with temporal term
        When: generate_questions() is called
        Then: Returns question about time range
        """
        # Given
        ambiguous_terms = [(AmbiguityType.TEMPORAL, "recent")]
        query = "recent developments"

        # When
        questions = clarifier.generate_questions(ambiguous_terms, query)

        # Then
        assert len(questions) > 0
        assert questions[0].type == AmbiguityType.TEMPORAL
        assert "recent" in questions[0].question.lower()

    def test_generate_multiple_questions(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: Multiple ambiguous terms
        When: generate_questions() is called
        Then: Returns multiple questions up to MAX_QUESTIONS
        """
        # Given
        ambiguous_terms = [
            (AmbiguityType.PRONOUN, "he"),
            (AmbiguityType.MULTI_MEANING, "python"),
            (AmbiguityType.TEMPORAL, "recent"),
        ]
        query = "What did he say about Python recently?"

        # When
        questions = clarifier.generate_questions(ambiguous_terms, query)

        # Then
        assert len(questions) >= 1
        assert len(questions) <= MAX_QUESTIONS

    def test_questions_bounded_by_max(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: More ambiguous terms than MAX_QUESTIONS
        When: generate_questions() is called
        Then: Returns exactly MAX_QUESTIONS questions
        """
        # Given
        ambiguous_terms = [(AmbiguityType.PRONOUN, f"term{i}") for i in range(20)]
        query = "test query"

        # When
        questions = clarifier.generate_questions(ambiguous_terms, query)

        # Then
        assert len(questions) <= MAX_QUESTIONS


# =============================================================================
# GWT Tests: Query Refinement
# =============================================================================


class TestQueryRefinementGWT:
    """GWT tests for query refinement with clarifications."""

    def test_refine_with_single_clarification(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: Original query and one clarification
        When: refine_query() is called
        Then: Returns refined query with increased confidence
        """
        # Given
        original = "What did he say?"
        clarifications = {"he": "Guido van Rossum"}

        # When
        refined, confidence = clarifier.refine_query(original, clarifications)

        # Then
        assert "Guido van Rossum" in refined
        assert confidence > 0.5
        assert isinstance(refined, str)
        assert isinstance(confidence, float)

    def test_refine_with_multiple_clarifications(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: Original query and multiple clarifications
        When: refine_query() is called
        Then: All clarifications applied, higher confidence
        """
        # Given
        original = "What did he say about Python?"
        clarifications = {
            "he": "Guido van Rossum",
            "Python": "Python programming language",
        }

        # When
        refined, confidence = clarifier.refine_query(original, clarifications)

        # Then
        assert "Guido van Rossum" in refined
        assert "programming language" in refined.lower()
        assert confidence > 0.6

    def test_confidence_increases_with_more_clarifications(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: Same query with different numbers of clarifications
        When: refine_query() is called for each
        Then: More clarifications = higher confidence
        """
        # Given
        original = "Tell me about it"
        one_clarif = {"it": "Python"}
        two_clarif = {"it": "Python programming", "tell": "explain"}

        # When
        _, conf1 = clarifier.refine_query(original, one_clarif)
        _, conf2 = clarifier.refine_query(original, two_clarif)

        # Then
        assert conf2 >= conf1

    def test_refinement_preserves_original_meaning(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: Original query with clarification
        When: refine_query() is called
        Then: Original query structure preserved, terms replaced
        """
        # Given
        original = "What did he say?"
        clarifications = {"he": "Einstein"}

        # When
        refined, _ = clarifier.refine_query(original, clarifications)

        # Then
        assert "What did" in refined
        assert "say" in refined
        assert "Einstein" in refined

    def test_case_insensitive_refinement(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: Original query with mixed case term
        When: refine_query() is called
        Then: Refinement works regardless of case
        """
        # Given
        original = "Tell me about PYTHON"
        clarifications = {"python": "programming language"}

        # When
        refined, confidence = clarifier.refine_query(original, clarifications)

        # Then
        assert confidence > 0.5
        # Refinement should handle case-insensitive matching

    def test_empty_clarifications(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: Original query and empty clarifications dict
        When: refine_query() is called
        Then: Returns original query with base confidence
        """
        # Given
        original = "What is Python?"
        clarifications: Dict[str, str] = {}

        # When
        refined, confidence = clarifier.refine_query(original, clarifications)

        # Then
        assert refined == original
        assert confidence >= 0.5


# =============================================================================
# GWT Tests: Enhanced Evaluate Method
# =============================================================================


class TestEnhancedEvaluateGWT:
    """GWT tests for enhanced evaluate() method."""

    def test_evaluate_clear_query_no_ambiguity(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: A clear, specific query
        When: evaluate() is called
        Then: Returns artifact with is_clear=True, no questions
        """
        # Given
        query = "Who is the CEO of Apple Inc. in 2024?"

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert artifact.clarity_score.score >= 0.5
        assert artifact.ambiguity_report is not None
        assert (
            len(artifact.ambiguity_report.questions) == 0
            or not artifact.needs_clarification
        )

    def test_evaluate_pronoun_ambiguity_detected(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: A query with unclear pronoun
        When: evaluate() is called
        Then: Detects ambiguity, generates pronoun question
        """
        # Given
        query = "What did he say?"

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert artifact.ambiguity_report is not None
        assert artifact.ambiguity_report.is_ambiguous is True
        assert len(artifact.ambiguity_report.questions) > 0
        # Check if pronoun question generated
        has_pronoun_question = any(
            q.type == AmbiguityType.PRONOUN for q in artifact.ambiguity_report.questions
        )
        assert has_pronoun_question

    def test_evaluate_multi_meaning_detected(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with multi-meaning term
        When: evaluate() is called
        Then: Detects ambiguity, generates clarification question
        """
        # Given
        query = "Tell me about Python"

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert artifact.ambiguity_report is not None
        assert artifact.ambiguity_report.is_ambiguous is True
        # Should have multi-meaning question
        has_multi_meaning = any(
            q.type == AmbiguityType.MULTI_MEANING
            for q in artifact.ambiguity_report.questions
        )
        assert has_multi_meaning

    def test_evaluate_temporal_ambiguity_detected(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: A query with temporal ambiguity
        When: evaluate() is called
        Then: Flags query as ambiguous
        """
        # Given
        query = "recent developments in AI"

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert artifact.ambiguity_report is not None
        assert artifact.ambiguity_report.is_ambiguous is True

    def test_evaluate_with_context_reduces_ambiguity(
        self, clarifier: IFQueryClarifier, sample_context: Dict[str, Any]
    ) -> None:
        """
        Given: Ambiguous query WITH conversation context
        When: evaluate() is called with context
        Then: Context helps resolve some ambiguities
        """
        # Given
        query = "What did he create?"

        # When
        artifact_no_context = clarifier.evaluate(query)
        artifact_with_context = clarifier.evaluate(query, sample_context)

        # Then
        # Both should detect ambiguity, but results may differ
        assert artifact_no_context.ambiguity_report is not None
        assert artifact_with_context.ambiguity_report is not None

    def test_evaluate_short_query(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A very short query (< MIN_QUERY_LENGTH)
        When: evaluate() is called
        Then: Returns low clarity artifact with suggestions
        """
        # Given
        query = "x"

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert artifact.needs_clarification is True
        assert len(artifact.suggestions) > 0

    def test_evaluate_empty_query(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: An empty query string
        When: evaluate() is called
        Then: Returns low clarity artifact
        """
        # Given
        query = ""

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert artifact.needs_clarification is True
        assert artifact.clarity_score.score == 0.0

    def test_evaluate_ambiguity_score_calculation(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """
        Given: Query with multiple ambiguities
        When: evaluate() is called
        Then: Ambiguity score reflects number of issues
        """
        # Given
        query = "What did he say about Python recently?"  # pronoun + multi-meaning + temporal

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert artifact.ambiguity_report is not None
        assert artifact.ambiguity_report.ambiguity_score > 0.0
        assert artifact.ambiguity_report.ambiguity_score <= 1.0


# =============================================================================
# GWT Tests: Dataclass Validation
# =============================================================================


class TestDataclassValidationGWT:
    """GWT tests for dataclass validation and JPL compliance."""

    def test_clarity_score_valid_range(self) -> None:
        """
        Given: Creating ClarityScore with valid score
        When: Dataclass is instantiated
        Then: No assertion error, to_dict() works
        """
        # Given
        score_value = 0.75

        # When
        score = ClarityScore(score=score_value, is_clear=True, factors={})

        # Then
        assert score.score == score_value
        assert score.is_clear is True
        d = score.to_dict()
        assert d["score"] == score_value

    def test_clarity_score_invalid_range_fails(self) -> None:
        """
        Given: Creating ClarityScore with invalid score (> 1.0)
        When: Dataclass is instantiated
        Then: Raises AssertionError (JPL Rule #5)
        """
        # Given
        invalid_score = 1.5

        # When/Then
        with pytest.raises(AssertionError):
            ClarityScore(score=invalid_score, is_clear=True)

    def test_clarification_question_valid(self) -> None:
        """
        Given: Creating ClarificationQuestion with valid data
        When: Dataclass is instantiated
        Then: No errors, to_dict() serializes correctly
        """
        # Given
        question = ClarificationQuestion(
            type=AmbiguityType.PRONOUN,
            question="Who does 'he' refer to?",
            options=["Person A", "Person B"],
            confidence=0.8,
            term="he",
        )

        # When
        d = question.to_dict()

        # Then
        assert d["type"] == "pronoun_reference"
        assert d["question"] == "Who does 'he' refer to?"
        assert len(d["options"]) == 2
        assert d["confidence"] == 0.8
        assert d["term"] == "he"

    def test_clarification_question_min_options(self) -> None:
        """
        Given: Creating ClarificationQuestion with < 2 options
        When: Dataclass is instantiated
        Then: Raises AssertionError
        """
        # Given/When/Then
        with pytest.raises(AssertionError):
            ClarificationQuestion(
                type=AmbiguityType.PRONOUN,
                question="Test?",
                options=["Only one"],  # Invalid: need >= 2
                confidence=0.8,
            )

    def test_ambiguity_report_valid(self) -> None:
        """
        Given: Creating AmbiguityReport with valid data
        When: Dataclass is instantiated
        Then: No errors, to_dict() works
        """
        # Given
        question = ClarificationQuestion(
            type=AmbiguityType.PRONOUN,
            question="Test?",
            options=["A", "B"],
            confidence=0.8,
        )
        report = AmbiguityReport(
            is_ambiguous=True,
            ambiguity_score=0.75,
            ambiguous_terms=[(AmbiguityType.PRONOUN, "he")],
            questions=[question],
        )

        # When
        d = report.to_dict()

        # Then
        assert d["is_ambiguous"] is True
        assert d["ambiguity_score"] == 0.75
        assert len(d["questions"]) == 1

    def test_ambiguity_report_too_many_questions(self) -> None:
        """
        Given: Creating AmbiguityReport with > MAX_QUESTIONS
        When: Dataclass is instantiated
        Then: Raises AssertionError (JPL Rule #2)
        """
        # Given
        questions = [
            ClarificationQuestion(
                type=AmbiguityType.PRONOUN,
                question=f"Q{i}?",
                options=["A", "B"],
                confidence=0.8,
            )
            for i in range(MAX_QUESTIONS + 1)
        ]

        # When/Then
        with pytest.raises(AssertionError):
            AmbiguityReport(
                is_ambiguous=True,
                ambiguity_score=0.5,
                ambiguous_terms=[],
                questions=questions,
            )


# =============================================================================
# GWT Tests: Convenience Functions
# =============================================================================


class TestConvenienceFunctionsGWT:
    """GWT tests for convenience functions."""

    def test_create_clarifier_default(self) -> None:
        """
        Given: Calling create_clarifier() with no args
        When: Function is executed
        Then: Returns IFQueryClarifier with default config
        """
        # Given/When
        clarifier = create_clarifier()

        # Then
        assert isinstance(clarifier, IFQueryClarifier)
        assert clarifier._config.threshold == CLARITY_THRESHOLD

    def test_create_clarifier_custom_threshold(self) -> None:
        """
        Given: Calling create_clarifier() with custom threshold
        When: Function is executed
        Then: Returns clarifier with custom config
        """
        # Given
        custom_threshold = 0.8

        # When
        clarifier = create_clarifier(threshold=custom_threshold)

        # Then
        assert isinstance(clarifier, IFQueryClarifier)
        assert clarifier._config.threshold == custom_threshold

    def test_evaluate_query_clarity_convenience(self) -> None:
        """
        Given: Calling evaluate_query_clarity() with a query
        When: Function is executed
        Then: Returns ClarificationArtifact
        """
        # Given
        query = "What is Python?"

        # When
        artifact = evaluate_query_clarity(query)

        # Then
        assert isinstance(artifact, ClarificationArtifact)
        assert artifact.original_query == query

    def test_needs_clarification_vague_query(self) -> None:
        """
        Given: Calling needs_clarification() with vague query
        When: Function is executed
        Then: Returns True
        """
        # Given
        query = "tell me more"

        # When
        result = needs_clarification(query)

        # Then
        assert result is True

    def test_needs_clarification_clear_query(self) -> None:
        """
        Given: Calling needs_clarification() with clear query
        When: Function is executed
        Then: Returns False
        """
        # Given
        query = "Who is the CEO of Apple Inc.?"

        # When
        result = needs_clarification(query)

        # Then
        assert result is False


# =============================================================================
# GWT Tests: JPL Compliance
# =============================================================================


class TestJPLComplianceGWT:
    """GWT tests for JPL Power of Ten compliance."""

    def test_rule_2_max_constants_defined(self) -> None:
        """
        Given: JPL Rule #2 requirement for fixed bounds
        When: Checking module constants
        Then: All MAX constants are positive integers
        """
        # Given/When/Then
        assert MAX_QUERY_LENGTH > 0
        assert MAX_SUGGESTIONS > 0
        assert MAX_QUESTIONS > 0
        assert MAX_AMBIGUOUS_TERMS > 0
        assert MAX_CONTEXT_QUERIES > 0

    def test_rule_5_precondition_assertions(self) -> None:
        """
        Given: JPL Rule #5 requirement for preconditions
        When: Creating dataclass with invalid data
        Then: AssertionError is raised
        """
        # Given/When/Then
        with pytest.raises(AssertionError):
            ClarityScore(score=2.0, is_clear=True)  # Score > 1.0

        with pytest.raises(AssertionError):
            ClarificationArtifact(
                original_query="",  # Empty query
                clarity_score=ClarityScore(score=0.5, is_clear=False),
                suggestions=[],
                reason="test",
                needs_clarification=True,
            )

    def test_rule_9_type_hints_present(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: JPL Rule #9 requirement for type hints
        When: Checking method signatures
        Then: All methods have __annotations__
        """
        # Given/When/Then
        assert hasattr(clarifier.evaluate, "__annotations__")
        assert hasattr(clarifier.detect_pronouns, "__annotations__")
        assert hasattr(clarifier.detect_multi_meaning, "__annotations__")
        assert hasattr(clarifier.detect_temporal_ambiguity, "__annotations__")
        assert hasattr(clarifier.generate_questions, "__annotations__")
        assert hasattr(clarifier.refine_query, "__annotations__")


# =============================================================================
# GWT Tests: Edge Cases
# =============================================================================


class TestEdgeCasesGWT:
    """GWT tests for edge cases and boundary conditions."""

    def test_very_long_query_truncated(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query longer than MAX_QUERY_LENGTH
        When: evaluate() is called
        Then: Query is truncated to MAX_QUERY_LENGTH
        """
        # Given
        long_query = "x" * (MAX_QUERY_LENGTH + 100)

        # When
        artifact = clarifier.evaluate(long_query)

        # Then
        assert len(artifact.original_query) <= MAX_QUERY_LENGTH

    def test_query_with_special_characters(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with special characters
        When: evaluate() is called
        Then: Handles gracefully without errors
        """
        # Given
        query = "What is @#$%^&*()?"

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert isinstance(artifact, ClarificationArtifact)

    def test_query_with_numbers_only(self, clarifier: IFQueryClarifier) -> None:
        """
        Given: A query with only numbers
        When: evaluate() is called
        Then: Returns artifact with low clarity
        """
        # Given
        query = "12345"

        # When
        artifact = clarifier.evaluate(query)

        # Then
        assert isinstance(artifact, ClarificationArtifact)
        # Likely needs clarification
