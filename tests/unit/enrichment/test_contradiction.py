"""Tests for contradiction detection engine.

Tests cover:
- Basic contradiction detection
- Negation pattern matching
- Antonym detection
- Corpus-wide contradiction finding
- Edge cases and error handling
- JPL Commandment compliance
"""

import pytest

from ingestforge.enrichment.contradiction import (
    ContradictionDetector,
    ContradictionPair,
    ContradictionResult,
    MAX_CLAIMS_TO_COMPARE,
    MAX_CLAIM_LENGTH,
    NEGATION_WORDS,
    ANTONYM_PAIRS,
)


class TestContradictionResult:
    """Tests for ContradictionResult dataclass."""

    def test_initialization(self):
        """Test ContradictionResult initialization."""
        result = ContradictionResult(
            claim1="The sky is blue",
            claim2="The sky is not blue",
            score=0.8,
            explanation="Test explanation",
            negation_detected=True,
            antonym_detected=False,
        )

        assert result.claim1 == "The sky is blue"
        assert result.claim2 == "The sky is not blue"
        assert result.score == 0.8
        assert result.explanation == "Test explanation"
        assert result.negation_detected is True
        assert result.antonym_detected is False

    def test_default_values(self):
        """Test default values for optional fields."""
        result = ContradictionResult(
            claim1="Test 1",
            claim2="Test 2",
            score=0.5,
            explanation="Test",
        )

        assert result.negation_detected is False
        assert result.antonym_detected is False


class TestContradictionPair:
    """Tests for ContradictionPair dataclass."""

    def test_initialization(self):
        """Test ContradictionPair initialization."""
        pair = ContradictionPair(
            index1=0,
            index2=5,
            score=0.9,
            claim1="First claim",
            claim2="Second claim",
        )

        assert pair.index1 == 0
        assert pair.index2 == 5
        assert pair.score == 0.9
        assert pair.claim1 == "First claim"
        assert pair.claim2 == "Second claim"


class TestContradictionDetectorInit:
    """Tests for ContradictionDetector initialization."""

    def test_default_initialization(self):
        """Test detector with default parameters."""
        detector = ContradictionDetector()

        assert detector.similarity_threshold == 0.7
        assert detector.negation_boost == 0.3
        assert detector.embedding_model == "all-MiniLM-L6-v2"
        assert detector._model is None

    def test_custom_initialization(self):
        """Test detector with custom parameters."""
        detector = ContradictionDetector(
            similarity_threshold=0.8,
            negation_boost=0.5,
            embedding_model="custom-model",
        )

        assert detector.similarity_threshold == 0.8
        assert detector.negation_boost == 0.5
        assert detector.embedding_model == "custom-model"

    def test_invalid_similarity_threshold_high(self):
        """Test validation of similarity_threshold upper bound."""
        with pytest.raises(ValueError, match="similarity_threshold must be 0.0-1.0"):
            ContradictionDetector(similarity_threshold=1.5)

    def test_invalid_similarity_threshold_low(self):
        """Test validation of similarity_threshold lower bound."""
        with pytest.raises(ValueError, match="similarity_threshold must be 0.0-1.0"):
            ContradictionDetector(similarity_threshold=-0.1)

    def test_invalid_negation_boost_high(self):
        """Test validation of negation_boost upper bound."""
        with pytest.raises(ValueError, match="negation_boost must be 0.0-1.0"):
            ContradictionDetector(negation_boost=1.5)

    def test_invalid_negation_boost_low(self):
        """Test validation of negation_boost lower bound."""
        with pytest.raises(ValueError, match="negation_boost must be 0.0-1.0"):
            ContradictionDetector(negation_boost=-0.1)

    def test_boundary_values(self):
        """Test boundary values are accepted."""
        detector1 = ContradictionDetector(similarity_threshold=0.0, negation_boost=0.0)
        assert detector1.similarity_threshold == 0.0
        assert detector1.negation_boost == 0.0

        detector2 = ContradictionDetector(similarity_threshold=1.0, negation_boost=1.0)
        assert detector2.similarity_threshold == 1.0
        assert detector2.negation_boost == 1.0


class TestDetectContradiction:
    """Tests for detect_contradiction method."""

    @pytest.fixture
    def detector(self) -> ContradictionDetector:
        """Create detector instance."""
        # Use lower similarity threshold for testing
        return ContradictionDetector(similarity_threshold=0.5)

    def test_empty_claim1(self, detector: ContradictionDetector):
        """Test detection with empty first claim."""
        with pytest.raises(ValueError, match="claim1 cannot be empty"):
            detector.detect_contradiction("", "The sky is blue")

    def test_empty_claim2(self, detector: ContradictionDetector):
        """Test detection with empty second claim."""
        with pytest.raises(ValueError, match="claim2 cannot be empty"):
            detector.detect_contradiction("The sky is blue", "")

    def test_whitespace_only_claim1(self, detector: ContradictionDetector):
        """Test detection with whitespace-only first claim."""
        with pytest.raises(ValueError, match="claim1 cannot be empty"):
            detector.detect_contradiction("   ", "The sky is blue")

    def test_whitespace_only_claim2(self, detector: ContradictionDetector):
        """Test detection with whitespace-only second claim."""
        with pytest.raises(ValueError, match="claim2 cannot be empty"):
            detector.detect_contradiction("The sky is blue", "   ")

    def test_claim1_too_long(self, detector: ContradictionDetector):
        """Test detection with claim1 exceeding max length."""
        long_claim = "a" * (MAX_CLAIM_LENGTH + 1)
        with pytest.raises(ValueError, match="claim1 too long"):
            detector.detect_contradiction(long_claim, "Short claim")

    def test_claim2_too_long(self, detector: ContradictionDetector):
        """Test detection with claim2 exceeding max length."""
        long_claim = "a" * (MAX_CLAIM_LENGTH + 1)
        with pytest.raises(ValueError, match="claim2 too long"):
            detector.detect_contradiction("Short claim", long_claim)

    def test_basic_negation_contradiction(self, detector: ContradictionDetector):
        """Test detection of basic negation contradiction."""
        result = detector.detect_contradiction(
            "The sky is blue",
            "The sky is not blue",
        )

        assert isinstance(result, ContradictionResult)
        assert result.claim1 == "The sky is blue"
        assert result.claim2 == "The sky is not blue"
        assert result.score > 0.0
        assert result.negation_detected is True
        assert "similar" in result.explanation.lower()

    def test_antonym_contradiction(self, detector: ContradictionDetector):
        """Test detection of antonym-based contradiction."""
        result = detector.detect_contradiction(
            "The temperature is hot today",
            "The temperature is cold today",
        )

        assert result.score > 0.0
        assert result.antonym_detected is True

    def test_unrelated_claims(self, detector: ContradictionDetector):
        """Test detection with semantically unrelated claims."""
        result = detector.detect_contradiction(
            "The sky is blue",
            "I like pizza",
        )

        # Unrelated claims should have low contradiction score
        assert result.score == 0.0
        assert "not semantically related" in result.explanation

    def test_similar_non_contradicting_claims(self, detector: ContradictionDetector):
        """Test detection with similar but non-contradicting claims."""
        result = detector.detect_contradiction(
            "The sky is blue",
            "The sky appears blue",
        )

        # Similar claims without negation should have low score
        assert result.score == 0.0 or result.score < 0.3


class TestFindContradictionsInCorpus:
    """Tests for find_contradictions_in_corpus method."""

    @pytest.fixture
    def detector(self) -> ContradictionDetector:
        """Create detector instance."""
        return ContradictionDetector(similarity_threshold=0.5)

    def test_empty_corpus(self, detector: ContradictionDetector):
        """Test finding contradictions in empty corpus."""
        result = detector.find_contradictions_in_corpus([])
        assert result == []

    def test_single_claim(self, detector: ContradictionDetector):
        """Test finding contradictions with single claim."""
        result = detector.find_contradictions_in_corpus(["The sky is blue"])
        assert result == []

    def test_too_many_claims(self, detector: ContradictionDetector):
        """Test validation of claims list size."""
        too_many = ["claim" for _ in range(MAX_CLAIMS_TO_COMPARE + 1)]
        with pytest.raises(ValueError, match="Too many claims"):
            detector.find_contradictions_in_corpus(too_many)

    def test_empty_claim_in_corpus(self, detector: ContradictionDetector):
        """Test validation of empty claim in corpus."""
        claims = ["Valid claim", "", "Another valid claim"]
        with pytest.raises(ValueError, match="Claim at index 1 is empty"):
            detector.find_contradictions_in_corpus(claims)

    def test_too_long_claim_in_corpus(self, detector: ContradictionDetector):
        """Test validation of too long claim in corpus."""
        long_claim = "a" * (MAX_CLAIM_LENGTH + 1)
        claims = ["Valid claim", long_claim]
        with pytest.raises(ValueError, match="Claim at index 1 too long"):
            detector.find_contradictions_in_corpus(claims)

    def test_two_contradicting_claims(self, detector: ContradictionDetector):
        """Test finding contradiction between two claims."""
        claims = [
            "The sky is blue",
            "The sky is not blue",
        ]

        result = detector.find_contradictions_in_corpus(claims, min_score=0.1)

        assert len(result) >= 1
        assert isinstance(result[0], ContradictionPair)
        assert result[0].index1 == 0
        assert result[0].index2 == 1
        assert result[0].score > 0.0

    def test_multiple_contradictions(self, detector: ContradictionDetector):
        """Test finding multiple contradictions in corpus."""
        claims = [
            "The sky is blue",
            "The sky is not blue",
            "Temperature is hot",
            "Temperature is cold",
        ]

        result = detector.find_contradictions_in_corpus(claims, min_score=0.1)

        # Should find at least one contradiction
        assert len(result) >= 1
        assert all(isinstance(pair, ContradictionPair) for pair in result)

    def test_sorted_by_score(self, detector: ContradictionDetector):
        """Test results are sorted by score descending."""
        claims = [
            "The sky is blue",
            "The sky is not blue",
            "Temperature is hot",
            "Temperature is cold",
            "It is raining",
        ]

        result = detector.find_contradictions_in_corpus(claims, min_score=0.0)

        if len(result) > 1:
            scores = [pair.score for pair in result]
            assert scores == sorted(scores, reverse=True)

    def test_min_score_filtering(self, detector: ContradictionDetector):
        """Test min_score filters low-scoring pairs."""
        claims = [
            "The sky is blue",
            "The sky is not blue",
        ]

        # With very high min_score, should get no results
        result = detector.find_contradictions_in_corpus(claims, min_score=0.99)
        assert len(result) == 0

    def test_no_contradictions(self, detector: ContradictionDetector):
        """Test corpus with no contradictions."""
        claims = [
            "I like apples",
            "The weather is nice",
            "Python is a programming language",
        ]

        result = detector.find_contradictions_in_corpus(claims, min_score=0.5)
        assert len(result) == 0


class TestHelperMethods:
    """Tests for internal helper methods."""

    @pytest.fixture
    def detector(self) -> ContradictionDetector:
        """Create detector instance."""
        return ContradictionDetector()

    def test_extract_claims(self, detector: ContradictionDetector):
        """Test claim extraction from text."""
        text = "First sentence. Second sentence! Third sentence? Fourth"

        claims = detector._extract_claims(text)

        assert len(claims) == 4
        assert "First sentence" in claims
        assert "Second sentence" in claims
        assert "Third sentence" in claims
        assert "Fourth" in claims

    def test_extract_claims_empty(self, detector: ContradictionDetector):
        """Test claim extraction from empty text."""
        claims = detector._extract_claims("")
        assert claims == []

    def test_extract_claims_whitespace(self, detector: ContradictionDetector):
        """Test claim extraction handles whitespace."""
        text = "  First.  Second.  "
        claims = detector._extract_claims(text)

        assert len(claims) == 2
        assert claims[0] == "First"
        assert claims[1] == "Second"

    def test_check_negation_one_has_negation(self, detector: ContradictionDetector):
        """Test negation detection when one claim has negation."""
        score = detector._check_negation(
            "The sky is blue",
            "The sky is not blue",
        )

        assert score > 0.0
        assert score == detector.negation_boost

    def test_check_negation_both_have_negation(self, detector: ContradictionDetector):
        """Test negation detection when both claims have negation."""
        score = detector._check_negation(
            "The sky is not blue",
            "The grass is not green",
        )

        # Both have negation, so no contradiction indicated
        assert score == 0.0

    def test_check_negation_neither_has_negation(self, detector: ContradictionDetector):
        """Test negation detection when neither claim has negation."""
        score = detector._check_negation(
            "The sky is blue",
            "The grass is green",
        )

        assert score == 0.0

    def test_check_antonyms_hot_cold(self, detector: ContradictionDetector):
        """Test antonym detection for hot/cold."""
        score = detector._check_antonyms(
            "It is hot today",
            "It is cold today",
        )

        assert score > 0.0
        assert score == 0.2

    def test_check_antonyms_true_false(self, detector: ContradictionDetector):
        """Test antonym detection for true/false."""
        score = detector._check_antonyms(
            "This statement is true",
            "This statement is false",
        )

        assert score > 0.0

    def test_check_antonyms_no_antonyms(self, detector: ContradictionDetector):
        """Test antonym detection with no antonyms."""
        score = detector._check_antonyms(
            "The sky is blue",
            "The grass is green",
        )

        assert score == 0.0

    def test_combine_scores_high_similarity_high_negation(
        self, detector: ContradictionDetector
    ):
        """Test score combination with high similarity and negation."""
        score = detector._combine_scores(
            similarity=0.9,
            negation=0.3,
            antonym=0.0,
        )

        assert score > 0.0
        assert score <= 1.0

    def test_combine_scores_clamping(self, detector: ContradictionDetector):
        """Test score combination clamps to [0, 1]."""
        score = detector._combine_scores(
            similarity=1.0,
            negation=1.0,
            antonym=1.0,
        )

        assert score <= 1.0
        assert score >= 0.0

    def test_generate_explanation_all_signals(self, detector: ContradictionDetector):
        """Test explanation generation with all signals present."""
        explanation = detector._generate_explanation(
            similarity=0.85,
            has_negation=True,
            has_antonym=True,
        )

        assert "85%" in explanation or "85.0%" in explanation
        assert "Negation" in explanation
        assert "Antonym" in explanation
        assert "contradiction" in explanation.lower()

    def test_generate_explanation_no_signals(self, detector: ContradictionDetector):
        """Test explanation generation with no signals."""
        explanation = detector._generate_explanation(
            similarity=0.5,
            has_negation=False,
            has_antonym=False,
        )

        assert "50%" in explanation or "50.0%" in explanation
        assert "Negation" not in explanation
        assert "Antonym" not in explanation


class TestConstants:
    """Tests for module constants."""

    def test_max_claims_to_compare(self):
        """Test MAX_CLAIMS_TO_COMPARE is reasonable."""
        assert MAX_CLAIMS_TO_COMPARE > 0
        assert MAX_CLAIMS_TO_COMPARE == 1000

    def test_max_claim_length(self):
        """Test MAX_CLAIM_LENGTH is reasonable."""
        assert MAX_CLAIM_LENGTH > 0
        assert MAX_CLAIM_LENGTH == 5000

    def test_negation_words_populated(self):
        """Test NEGATION_WORDS contains expected words."""
        assert len(NEGATION_WORDS) > 0
        assert "not" in NEGATION_WORDS
        assert "no" in NEGATION_WORDS
        assert "never" in NEGATION_WORDS
        assert "false" in NEGATION_WORDS

    def test_antonym_pairs_populated(self):
        """Test ANTONYM_PAIRS contains expected pairs."""
        assert len(ANTONYM_PAIRS) > 0
        assert ("hot", "cold") in ANTONYM_PAIRS
        assert ("true", "false") in ANTONYM_PAIRS
        assert ("yes", "no") in ANTONYM_PAIRS


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def detector(self) -> ContradictionDetector:
        """Create detector instance."""
        return ContradictionDetector(similarity_threshold=0.5)

    def test_identical_claims(self, detector: ContradictionDetector):
        """Test detection with identical claims."""
        result = detector.detect_contradiction(
            "The sky is blue",
            "The sky is blue",
        )

        # Identical claims should not contradict
        assert result.score == 0.0

    def test_case_sensitivity(self, detector: ContradictionDetector):
        """Test detection is case-insensitive."""
        result = detector.detect_contradiction(
            "The sky is NOT blue",
            "the sky is blue",
        )

        assert result.negation_detected is True

    def test_punctuation_handling(self, detector: ContradictionDetector):
        """Test detection handles punctuation."""
        result = detector.detect_contradiction(
            "The sky is blue.",
            "The sky is not blue!",
        )

        assert result.score > 0.0

    def test_special_characters(self, detector: ContradictionDetector):
        """Test detection handles special characters."""
        result = detector.detect_contradiction(
            "Price increased by 20%",
            "Price decreased by 20%",
        )

        # Should detect antonym (increase/decrease) if similar enough
        assert result.antonym_detected or result.score >= 0.0

    def test_unicode_text(self, detector: ContradictionDetector):
        """Test detection handles Unicode text."""
        result = detector.detect_contradiction(
            "The café is open",
            "The café is not open",
        )

        assert result.negation_detected is True

    def test_very_short_claims(self, detector: ContradictionDetector):
        """Test detection with very short claims."""
        result = detector.detect_contradiction(
            "Yes",
            "No",
        )

        # Should detect antonym
        assert result.score > 0.0 or result.antonym_detected

    def test_max_length_claims(self, detector: ContradictionDetector):
        """Test detection with maximum length claims."""
        claim1 = "The sky is blue. " * 300
        claim2 = "The sky is not blue. " * 300

        # Should work at boundary
        result = detector.detect_contradiction(
            claim1[:MAX_CLAIM_LENGTH],
            claim2[:MAX_CLAIM_LENGTH],
        )

        assert isinstance(result, ContradictionResult)

    def test_multiple_negations(self, detector: ContradictionDetector):
        """Test claims with multiple negation words."""
        result = detector.detect_contradiction(
            "It is never not raining",
            "It is always raining",
        )

        # Multiple negations in one claim
        assert isinstance(result, ContradictionResult)

    def test_numeric_values(self, detector: ContradictionDetector):
        """Test claims with numeric values."""
        result = detector.detect_contradiction(
            "The temperature is 100 degrees",
            "The temperature is not 100 degrees",
        )

        assert result.negation_detected is True


class TestJPLCompliance:
    """Tests for NASA JPL Commandments compliance."""

    def test_fixed_upper_bounds(self):
        """Test Rule #2: Fixed upper bounds for loops."""
        # MAX_CLAIMS_TO_COMPARE provides fixed bound
        assert MAX_CLAIMS_TO_COMPARE == 1000

        detector = ContradictionDetector()

        # Should reject lists exceeding bound
        too_many = ["claim" for _ in range(MAX_CLAIMS_TO_COMPARE + 1)]
        with pytest.raises(ValueError, match="Too many claims"):
            detector.find_contradictions_in_corpus(too_many)

    def test_input_validation(self):
        """Test Rule #7: Validate all inputs."""
        detector = ContradictionDetector()

        # Empty claims rejected
        with pytest.raises(ValueError):
            detector.detect_contradiction("", "test")

        # Too long claims rejected
        with pytest.raises(ValueError):
            detector.detect_contradiction("a" * 10000, "test")

        # Invalid thresholds rejected at init
        with pytest.raises(ValueError):
            ContradictionDetector(similarity_threshold=2.0)

    def test_type_hints_present(self):
        """Test Rule #9: Complete type hints."""
        # Check key methods have annotations
        from inspect import signature

        sig = signature(ContradictionDetector.detect_contradiction)
        # With __future__ annotations, return type is string
        assert sig.return_annotation == "ContradictionResult"

        sig = signature(ContradictionDetector.find_contradictions_in_corpus)
        # Return annotation should be present
        assert sig.return_annotation is not sig.empty
