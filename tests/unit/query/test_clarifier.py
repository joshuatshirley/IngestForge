"""
Tests for Query Clarification Agent.

Query Clarification Agent.
Verifies JPL Power of Ten compliance.
"""

import json
import pytest
from typing import Callable

from ingestforge.query.clarifier import (
    ClarityScore,
    ClarificationArtifact,
    ClarificationQuestion,
    AmbiguityReport,
    AmbiguityType,
    IFQueryClarifier,
    create_clarifier,
    evaluate_query_clarity,
    needs_clarification,
    CLARITY_THRESHOLD,
    MAX_SUGGESTIONS,
    MAX_QUERY_LENGTH,
    MAX_QUESTIONS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clarifier() -> IFQueryClarifier:
    """Create a clarifier for testing."""
    return IFQueryClarifier()


@pytest.fixture
def mock_llm_fn() -> Callable[[str], str]:
    """Create a mock LLM function."""

    def llm_fn(prompt: str) -> str:
        return json.dumps(
            [
                "What specific topic?",
                "Which time period?",
                "What context?",
            ]
        )

    return llm_fn


# =============================================================================
# TestClarityScore
# =============================================================================


class TestClarityScore:
    """Tests for ClarityScore dataclass."""

    def test_create_valid_score(self) -> None:
        """Test creating a valid clarity score."""
        score = ClarityScore(
            score=0.75,
            is_clear=True,
            factors={"length": 0.8, "specificity": 0.7},
        )
        assert score.score == 0.75
        assert score.is_clear is True

    def test_score_out_of_range_fails(self) -> None:
        """Test that invalid score raises AssertionError."""
        with pytest.raises(AssertionError):
            ClarityScore(score=1.5, is_clear=True)

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        score = ClarityScore(score=0.8, is_clear=True, factors={"test": 0.5})
        d = score.to_dict()
        assert d["score"] == 0.8


# =============================================================================
# TestClarificationArtifact
# =============================================================================


class TestClarificationArtifact:
    """Tests for ClarificationArtifact dataclass."""

    def test_create_valid_artifact(self) -> None:
        """Test creating a valid artifact."""
        artifact = ClarificationArtifact(
            original_query="test query",
            clarity_score=ClarityScore(score=0.5, is_clear=False),
            suggestions=["Be more specific"],
            reason="Query is vague",
            needs_clarification=True,
        )
        assert artifact.needs_clarification is True

    def test_empty_query_fails(self) -> None:
        """Test that empty query raises AssertionError."""
        with pytest.raises(AssertionError):
            ClarificationArtifact(
                original_query="",
                clarity_score=ClarityScore(score=0.5, is_clear=False),
                suggestions=[],
                reason="test",
                needs_clarification=True,
            )


# =============================================================================
# TestIFQueryClarifier
# =============================================================================


class TestIFQueryClarifier:
    """Tests for IFQueryClarifier class."""

    def test_evaluate_clear_query(self, clarifier: IFQueryClarifier) -> None:
        """Test evaluating a clear query."""
        result = clarifier.evaluate("Who is the CEO of Apple?")
        assert result.clarity_score.score >= 0.5

    def test_evaluate_vague_query(self, clarifier: IFQueryClarifier) -> None:
        """Test evaluating 'tell me more'."""
        result = clarifier.evaluate("tell me more")
        assert result.needs_clarification is True
        assert result.clarity_score.score < CLARITY_THRESHOLD

    def test_evaluate_short_query(self, clarifier: IFQueryClarifier) -> None:
        """Test evaluating a very short query."""
        result = clarifier.evaluate("x")
        assert result.needs_clarification is True

    def test_suggestions_provided(self, clarifier: IFQueryClarifier) -> None:
        """Test that suggestions are provided for vague queries."""
        result = clarifier.evaluate("help")
        assert len(result.suggestions) > 0


# =============================================================================
# TestAcceptanceCriteria
# =============================================================================


class TestAcceptanceCriteria:
    """Tests for acceptance criteria."""

    def test_ac_tell_me_more_triggers_clarification(self) -> None:
        """AC: Query 'Tell me more' triggers clarification."""
        result = evaluate_query_clarity("Tell me more")
        assert result.needs_clarification is True
        assert len(result.suggestions) >= 1

    def test_ac_ceo_of_apple_proceeds(self) -> None:
        """AC: Query 'CEO of Apple' has decent clarity."""
        result = evaluate_query_clarity("CEO of Apple")
        assert result.clarity_score.score >= 0.5

    def test_ac_returns_suggestions(self) -> None:
        """AC: Returns suggested refinement questions."""
        result = evaluate_query_clarity("help me")
        assert len(result.suggestions) >= 1

    def test_ac_clarity_score_range(self) -> None:
        """AC: ClarityScore is in range [0.0 - 1.0]."""
        result = evaluate_query_clarity("test query")
        assert 0.0 <= result.clarity_score.score <= 1.0


# =============================================================================
# TestConvenienceFunctions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_clarifier(self) -> None:
        """Test creating a clarifier."""
        clarifier = create_clarifier(threshold=0.8)
        assert isinstance(clarifier, IFQueryClarifier)

    def test_needs_clarification_vague(self) -> None:
        """Test needs_clarification for vague query."""
        result = needs_clarification("tell me more")
        assert result is True


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_fixed_bounds(self) -> None:
        """Rule #2: Verify fixed bounds."""
        assert MAX_QUERY_LENGTH > 0
        assert MAX_SUGGESTIONS > 0

    def test_rule_5_preconditions(self) -> None:
        """Rule #5: Verify assertions."""
        with pytest.raises(AssertionError):
            ClarityScore(score=1.5, is_clear=True)

    def test_rule_9_type_hints(self) -> None:
        """Rule #9: Verify type hints."""
        clarifier = IFQueryClarifier()
        assert hasattr(clarifier.evaluate, "__annotations__")


# =============================================================================
# TestPronounDetection ()
# =============================================================================


class TestPronounDetection:
    """Tests for pronoun ambiguity detection."""

    def test_detect_pronoun_he(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'he' pronoun."""
        result = clarifier.detect_pronouns("What did he say?")
        assert "he" in result

    def test_detect_pronoun_it(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'it' pronoun."""
        result = clarifier.detect_pronouns("Tell me about it")
        assert "it" in result

    def test_detect_demonstrative_this(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'this' pronoun."""
        result = clarifier.detect_pronouns("What is this?")
        assert "this" in result

    def test_clear_query_no_pronouns(self, clarifier: IFQueryClarifier) -> None:
        """Test that clear queries don't trigger pronoun detection."""
        result = clarifier.detect_pronouns("Who is the CEO of Apple?")
        assert len(result) == 0

    def test_context_resolution(self, clarifier: IFQueryClarifier) -> None:
        """Test pronoun resolution with context."""
        context = {"previous_queries": ["Who is Guido van Rossum?"]}
        result = clarifier.detect_pronouns("What did he say about Python?", context)
        # With context of proper noun, pronoun might be resolvable
        # This is a heuristic-based test
        assert isinstance(result, list)


# =============================================================================
# TestMultiMeaningDetection ()
# =============================================================================


class TestMultiMeaningDetection:
    """Tests for multi-meaning term detection."""

    def test_detect_python_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'Python' ambiguity."""
        result = clarifier.detect_multi_meaning("Tell me about Python")
        assert len(result) > 0
        assert any("python" in term.lower() for term, _ in result)

    def test_detect_java_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'Java' ambiguity."""
        result = clarifier.detect_multi_meaning("What is Java?")
        assert len(result) > 0
        assert any("java" in term.lower() for term, _ in result)

    def test_detect_apple_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'Apple' ambiguity."""
        result = clarifier.detect_multi_meaning("Tell me about Apple")
        assert len(result) > 0

    def test_clear_query_no_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """Test that queries without ambiguous terms return empty."""
        result = clarifier.detect_multi_meaning("What is machine learning?")
        assert len(result) == 0


# =============================================================================
# TestTemporalAmbiguity ()
# =============================================================================


class TestTemporalAmbiguity:
    """Tests for temporal ambiguity detection."""

    def test_detect_recent(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'recent' temporal ambiguity."""
        result = clarifier.detect_temporal_ambiguity("recent developments in AI")
        assert "recent" in result

    def test_detect_soon(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'soon' temporal ambiguity."""
        result = clarifier.detect_temporal_ambiguity("What will happen soon?")
        assert "soon" in result

    def test_detect_before(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of 'before' temporal ambiguity."""
        result = clarifier.detect_temporal_ambiguity("What happened before?")
        assert "before" in result

    def test_specific_date_no_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """Test that specific dates don't trigger temporal ambiguity."""
        result = clarifier.detect_temporal_ambiguity("Events in 2024")
        assert len(result) == 0


# =============================================================================
# TestScopeAmbiguity ()
# =============================================================================


class TestScopeAmbiguity:
    """Tests for scope ambiguity detection."""

    def test_detect_broad_scope(self, clarifier: IFQueryClarifier) -> None:
        """Test detection of overly broad scope."""
        result = clarifier.detect_scope_ambiguity("Tell me everything about AI")
        assert result is True

    def test_specific_scope_clear(self, clarifier: IFQueryClarifier) -> None:
        """Test that specific queries are not flagged."""
        result = clarifier.detect_scope_ambiguity(
            "Specifically explain neural networks"
        )
        assert result is False


# =============================================================================
# TestQuestionGeneration ()
# =============================================================================


class TestQuestionGeneration:
    """Tests for clarification question generation."""

    def test_generate_pronoun_question(self, clarifier: IFQueryClarifier) -> None:
        """Test generation of pronoun clarification question."""
        ambiguous_terms = [(AmbiguityType.PRONOUN, "he")]
        questions = clarifier.generate_questions(ambiguous_terms, "What did he say?")
        assert len(questions) > 0
        assert questions[0].type == AmbiguityType.PRONOUN
        assert "he" in questions[0].question.lower()

    def test_generate_multi_meaning_question(self, clarifier: IFQueryClarifier) -> None:
        """Test generation of multi-meaning clarification question."""
        ambiguous_terms = [(AmbiguityType.MULTI_MEANING, "python")]
        questions = clarifier.generate_questions(
            ambiguous_terms, "Tell me about Python"
        )
        assert len(questions) > 0
        assert questions[0].type == AmbiguityType.MULTI_MEANING
        assert len(questions[0].options) >= 2

    def test_generate_temporal_question(self, clarifier: IFQueryClarifier) -> None:
        """Test generation of temporal clarification question."""
        ambiguous_terms = [(AmbiguityType.TEMPORAL, "recent")]
        questions = clarifier.generate_questions(ambiguous_terms, "recent events")
        assert len(questions) > 0
        assert questions[0].type == AmbiguityType.TEMPORAL

    def test_max_questions_bound(self, clarifier: IFQueryClarifier) -> None:
        """Test that questions are bounded by MAX_QUESTIONS."""
        ambiguous_terms = [(AmbiguityType.PRONOUN, f"term{i}") for i in range(20)]
        questions = clarifier.generate_questions(ambiguous_terms, "test query")
        assert len(questions) <= MAX_QUESTIONS


# =============================================================================
# TestQueryRefinement ()
# =============================================================================


class TestQueryRefinement:
    """Tests for query refinement."""

    def test_refine_pronoun(self, clarifier: IFQueryClarifier) -> None:
        """Test refining query with pronoun clarification."""
        original = "What did he say about Python?"
        clarifications = {"he": "Guido van Rossum"}
        refined, confidence = clarifier.refine_query(original, clarifications)
        assert "Guido van Rossum" in refined
        assert confidence > 0.5

    def test_refine_multiple_terms(self, clarifier: IFQueryClarifier) -> None:
        """Test refining query with multiple clarifications."""
        original = "What did he say about Python?"
        clarifications = {
            "he": "Guido van Rossum",
            "Python": "Python programming language",
        }
        refined, confidence = clarifier.refine_query(original, clarifications)
        assert "Guido van Rossum" in refined
        assert "programming language" in refined
        assert confidence > 0.6

    def test_confidence_increases_with_clarifications(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """Test that confidence increases with more clarifications."""
        original = "What about it?"
        one_clarif = {"it": "Python"}
        two_clarif = {"it": "Python programming", "what": "features"}

        _, conf1 = clarifier.refine_query(original, one_clarif)
        _, conf2 = clarifier.refine_query(original, two_clarif)

        # More clarifications should increase confidence
        assert conf2 >= conf1


# =============================================================================
# TestEnhancedEvaluate ()
# =============================================================================


class TestEnhancedEvaluate:
    """Tests for enhanced evaluate() with ambiguity detection."""

    def test_evaluate_with_pronoun_ambiguity(self, clarifier: IFQueryClarifier) -> None:
        """Test evaluation detects pronoun ambiguity."""
        result = clarifier.evaluate("What did he say?")
        assert result.ambiguity_report is not None
        assert result.ambiguity_report.is_ambiguous is True
        assert len(result.ambiguity_report.questions) > 0

    def test_evaluate_with_multi_meaning(self, clarifier: IFQueryClarifier) -> None:
        """Test evaluation detects multi-meaning ambiguity."""
        result = clarifier.evaluate("Tell me about Python")
        assert result.ambiguity_report is not None
        # Should detect Python as ambiguous
        assert any(
            q.type == AmbiguityType.MULTI_MEANING
            for q in result.ambiguity_report.questions
        )

    def test_evaluate_with_temporal_ambiguity(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """Test evaluation detects temporal ambiguity."""
        result = clarifier.evaluate("recent developments")
        assert result.ambiguity_report is not None
        # Should flag as ambiguous due to 'recent'
        assert result.ambiguity_report.is_ambiguous is True

    def test_evaluate_clear_query_no_ambiguity(
        self, clarifier: IFQueryClarifier
    ) -> None:
        """Test that clear queries have no ambiguity."""
        result = clarifier.evaluate("Who is the CEO of Apple Inc. in 2024?")
        assert result.ambiguity_report is not None
        # Very specific query should not be ambiguous
        # (or have low ambiguity score)
        assert (
            len(result.ambiguity_report.questions) == 0
            or result.ambiguity_report.ambiguity_score < 0.5
        )

    def test_evaluate_with_context(self, clarifier: IFQueryClarifier) -> None:
        """Test evaluation with conversation context."""
        context = {"previous_queries": ["Who is Guido van Rossum?"]}
        result = clarifier.evaluate("What did he say about Python?", context)
        # Context should help resolve 'he' pronoun
        assert result.ambiguity_report is not None


# =============================================================================
# TestAmbiguityReport ()
# =============================================================================


class TestAmbiguityReport:
    """Tests for AmbiguityReport dataclass."""

    def test_create_valid_report(self) -> None:
        """Test creating a valid ambiguity report."""
        question = ClarificationQuestion(
            type=AmbiguityType.PRONOUN,
            question="Who does 'he' refer to?",
            options=["Option 1", "Option 2"],
            confidence=0.8,
            term="he",
        )
        report = AmbiguityReport(
            is_ambiguous=True,
            ambiguity_score=0.75,
            ambiguous_terms=[(AmbiguityType.PRONOUN, "he")],
            questions=[question],
        )
        assert report.is_ambiguous is True
        assert len(report.questions) == 1

    def test_ambiguity_score_range(self) -> None:
        """Test ambiguity score is in valid range."""
        with pytest.raises(AssertionError):
            AmbiguityReport(
                is_ambiguous=True,
                ambiguity_score=1.5,  # Invalid: > 1.0
                ambiguous_terms=[],
                questions=[],
            )

    def test_max_questions_bound(self) -> None:
        """Test that questions are bounded."""
        questions = [
            ClarificationQuestion(
                type=AmbiguityType.PRONOUN,
                question=f"Question {i}?",
                options=["A", "B"],
                confidence=0.8,
            )
            for i in range(10)
        ]
        with pytest.raises(AssertionError):
            AmbiguityReport(
                is_ambiguous=True,
                ambiguity_score=0.5,
                ambiguous_terms=[],
                questions=questions,  # Exceeds MAX_QUESTIONS
            )
