"""
Tests for Chunk Quality Scorer.

This module tests quality metrics calculation for text chunks.

Test Strategy
-------------
- Focus on quality scoring logic and metric calculation
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test dataclass, scoring functions, and metric calculations
- Use simple ChunkRecord mocks

Organization
------------
- TestQualityMetrics: QualityMetrics dataclass
- TestQualityScorerInit: Initialization and parameters
- TestCoherenceScoring: _score_coherence function
- TestCompletenessScoring: _score_completeness function
- TestInformationDensity: _score_information_density function
- TestReadabilityScoring: _score_readability function
- TestSentenceSplitting: _split_sentences function
- TestSyllableCounting: _count_syllables function
- TestChunkScoring: Main score() and score_batch() methods
"""


from ingestforge.chunking.quality_scorer import QualityScorer, QualityMetrics
from ingestforge.chunking.semantic_chunker import ChunkRecord


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(content: str, chunk_id: str = "test_1") -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="test_doc",
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass.

    Rule #4: Focused test class - tests only QualityMetrics
    """

    def test_create_quality_metrics(self):
        """Test creating QualityMetrics."""
        metrics = QualityMetrics(
            coherence_score=0.9,
            completeness_score=0.85,
            information_density=0.75,
            readability_score=0.8,
            overall_score=0.825,
        )

        assert metrics.coherence_score == 0.9
        assert metrics.completeness_score == 0.85
        assert metrics.information_density == 0.75
        assert metrics.readability_score == 0.8
        assert metrics.overall_score == 0.825


class TestQualityScorerInit:
    """Tests for QualityScorer initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_scorer_with_defaults(self):
        """Test creating QualityScorer with default weights."""
        scorer = QualityScorer()

        assert scorer.coherence_weight == 0.3
        assert scorer.completeness_weight == 0.3
        assert scorer.density_weight == 0.2
        assert scorer.readability_weight == 0.2
        # Weights sum to 1.0
        total = sum(
            [
                scorer.coherence_weight,
                scorer.completeness_weight,
                scorer.density_weight,
                scorer.readability_weight,
            ]
        )
        assert abs(total - 1.0) < 0.01

    def test_create_scorer_with_custom_weights(self):
        """Test creating QualityScorer with custom weights."""
        scorer = QualityScorer(
            coherence_weight=0.4,
            completeness_weight=0.3,
            density_weight=0.2,
            readability_weight=0.1,
        )

        assert scorer.coherence_weight == 0.4
        assert scorer.completeness_weight == 0.3
        assert scorer.density_weight == 0.2
        assert scorer.readability_weight == 0.1


class TestCoherenceScoring:
    """Tests for coherence scoring.

    Rule #4: Focused test class - tests _score_coherence only
    """

    def test_score_coherence_well_formed_text(self):
        """Test coherence score for well-formed text."""
        scorer = QualityScorer()
        text = "This is a sentence. This is another sentence. This is a third sentence."

        score = scorer._score_coherence(text)

        # All sentences capitalized and punctuated properly
        assert score > 0.8

    def test_score_coherence_empty_text(self):
        """Test coherence score for empty text."""
        scorer = QualityScorer()
        text = ""

        score = scorer._score_coherence(text)

        assert score == 0.0

    def test_score_coherence_with_transition_words(self):
        """Test coherence bonus for transition words."""
        scorer = QualityScorer()
        text_without = "This is good. This is also good."
        text_with = "This is good. However, this is also good."

        score_without = scorer._score_coherence(text_without)
        score_with = scorer._score_coherence(text_with)

        # Transition words should increase score
        assert score_with >= score_without

    def test_score_coherence_poor_capitalization(self):
        """Test coherence penalty for poor capitalization."""
        scorer = QualityScorer()
        text = "this is bad. also this. and this."

        score = scorer._score_coherence(text)

        # Lowercase sentences should reduce score
        assert score < 0.5


class TestCompletenessScoring:
    """Tests for completeness scoring.

    Rule #4: Focused test class - tests _score_completeness only
    """

    def test_score_completeness_complete_sentences(self):
        """Test completeness score for complete sentences."""
        scorer = QualityScorer()
        text = "The quick brown fox jumps over the lazy dog."

        score = scorer._score_completeness(text)

        # Complete sentence should score high
        assert score > 0.8

    def test_score_completeness_empty_text(self):
        """Test completeness score for empty text."""
        scorer = QualityScorer()
        text = ""

        score = scorer._score_completeness(text)

        assert score == 0.0

    def test_score_completeness_fragments(self):
        """Test completeness penalty for sentence fragments."""
        scorer = QualityScorer()
        text = "The. Quick. Brown."

        score = scorer._score_completeness(text)

        # Short fragments should score lower than complete sentences
        assert score < 0.8


class TestInformationDensity:
    """Tests for information density scoring.

    Rule #4: Focused test class - tests _score_information_density only
    """

    def test_score_information_density_high_content(self):
        """Test density score for content-rich text."""
        scorer = QualityScorer()
        text = "Python programming language framework architecture"

        score = scorer._score_information_density(text)

        # Few stop words, high content density
        assert score > 0.7

    def test_score_information_density_low_content(self):
        """Test density score for stop-word-heavy text."""
        scorer = QualityScorer()
        text = "the the the and or but if it is was"

        score = scorer._score_information_density(text)

        # Mostly stop words, low content density
        assert score < 0.5

    def test_score_information_density_empty_text(self):
        """Test density score for empty text."""
        scorer = QualityScorer()
        text = ""

        score = scorer._score_information_density(text)

        assert score == 0.0

    def test_score_information_density_balanced(self):
        """Test density score for balanced text."""
        scorer = QualityScorer()
        text = "The system processes data with some additional context here."

        score = scorer._score_information_density(text)

        # Score should be valid (between 0 and 1)
        assert 0.0 <= score <= 1.0


class TestReadabilityScoring:
    """Tests for readability scoring.

    Rule #4: Focused test class - tests _score_readability only
    """

    def test_score_readability_simple_text(self):
        """Test readability score for simple text."""
        scorer = QualityScorer()
        text = "The cat sat on the mat. The dog ran fast."

        score = scorer._score_readability(text)

        # Simple words and sentences should score high
        assert score > 0.5

    def test_score_readability_empty_text(self):
        """Test readability score for empty text."""
        scorer = QualityScorer()
        text = ""

        score = scorer._score_readability(text)

        assert score == 0.0

    def test_score_readability_complex_text(self):
        """Test readability score for complex text."""
        scorer = QualityScorer()
        text = "The implementation utilizes sophisticated algorithmic methodologies."

        score = scorer._score_readability(text)

        # Complex words should reduce readability
        assert score >= 0.0  # Should still be valid


class TestSentenceSplitting:
    """Tests for sentence splitting.

    Rule #4: Focused test class - tests _split_sentences only
    """

    def test_split_sentences_basic(self):
        """Test basic sentence splitting."""
        scorer = QualityScorer()
        text = "First sentence. Second sentence. Third sentence."

        sentences = scorer._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence." in sentences
        assert "Second sentence." in sentences

    def test_split_sentences_with_abbreviations(self):
        """Test sentence splitting preserves abbreviations."""
        scorer = QualityScorer()
        text = "Dr. Smith went to the store. He bought milk."

        sentences = scorer._split_sentences(text)

        # Should handle "Dr." without splitting
        assert len(sentences) == 2
        assert any("Dr. Smith" in s for s in sentences)

    def test_split_sentences_with_numbers(self):
        """Test sentence splitting preserves decimal numbers."""
        scorer = QualityScorer()
        text = "The value is 3.14 exactly. This is correct."

        sentences = scorer._split_sentences(text)

        # Should preserve "3.14" as one number
        assert len(sentences) == 2
        assert any("3.14" in s for s in sentences)


class TestSyllableCounting:
    """Tests for syllable counting.

    Rule #4: Focused test class - tests _count_syllables only
    """

    def test_count_syllables_single_syllable(self):
        """Test counting single-syllable words."""
        scorer = QualityScorer()

        assert scorer._count_syllables("cat") == 1
        assert scorer._count_syllables("dog") == 1
        assert scorer._count_syllables("run") == 1

    def test_count_syllables_multi_syllable(self):
        """Test counting multi-syllable words."""
        scorer = QualityScorer()

        assert scorer._count_syllables("running") >= 2
        assert scorer._count_syllables("computer") >= 2
        assert scorer._count_syllables("information") >= 3

    def test_count_syllables_silent_e(self):
        """Test syllable counting handles silent e."""
        scorer = QualityScorer()

        # "make" should be 1 syllable (silent e)
        assert scorer._count_syllables("make") == 1
        assert scorer._count_syllables("time") == 1


class TestChunkScoring:
    """Tests for main chunk scoring methods.

    Rule #4: Focused test class - tests score() and score_batch()
    """

    def test_score_single_chunk(self):
        """Test scoring a single chunk."""
        scorer = QualityScorer()
        chunk = make_chunk("This is a well-formed sentence with good content.")

        metrics = scorer.score(chunk)

        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.coherence_score <= 1.0
        assert 0.0 <= metrics.completeness_score <= 1.0
        assert 0.0 <= metrics.information_density <= 1.0
        assert 0.0 <= metrics.readability_score <= 1.0
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_score_chunk_overall_is_weighted_average(self):
        """Test overall score is weighted average of metrics."""
        scorer = QualityScorer(
            coherence_weight=0.25,
            completeness_weight=0.25,
            density_weight=0.25,
            readability_weight=0.25,
        )
        chunk = make_chunk("This is test content.")

        metrics = scorer.score(chunk)

        # Calculate expected overall (equal weights)
        expected = (
            metrics.coherence_score
            + metrics.completeness_score
            + metrics.information_density
            + metrics.readability_score
        ) / 4.0

        assert abs(metrics.overall_score - expected) < 0.01

    def test_score_batch_updates_chunks(self):
        """Test score_batch updates quality_score field."""
        scorer = QualityScorer()
        chunks = [
            make_chunk("First chunk content.", "chunk_1"),
            make_chunk("Second chunk content here.", "chunk_2"),
            make_chunk("Third chunk with more content.", "chunk_3"),
        ]

        result = scorer.score_batch(chunks)

        # Should return same chunks
        assert result is chunks
        # All chunks should have quality_score set
        for chunk in chunks:
            assert hasattr(chunk, "quality_score")
            assert 0.0 <= chunk.quality_score <= 1.0

    def test_score_batch_empty_list(self):
        """Test score_batch with empty list."""
        scorer = QualityScorer()
        chunks = []

        result = scorer.score_batch(chunks)

        assert result == []

    def test_score_high_quality_chunk(self):
        """Test scoring high-quality chunk."""
        scorer = QualityScorer()
        chunk = make_chunk(
            "The quick brown fox jumps over the lazy dog. "
            "This sentence demonstrates proper structure. "
            "Furthermore, it contains transition words."
        )

        metrics = scorer.score(chunk)

        # High quality text should score well overall
        assert metrics.overall_score > 0.5

    def test_score_low_quality_chunk(self):
        """Test scoring low-quality chunk."""
        scorer = QualityScorer()
        chunk = make_chunk("the and or but if is was")

        metrics = scorer.score(chunk)

        # Poor quality text (all stop words) should score lower
        assert metrics.information_density < 0.5


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - QualityMetrics: 1 test (dataclass creation)
    - QualityScorer init: 2 tests (defaults, custom weights)
    - Coherence scoring: 4 tests (well-formed, empty, transitions, poor caps)
    - Completeness scoring: 3 tests (complete, empty, fragments)
    - Information density: 4 tests (high content, low content, empty, balanced)
    - Readability scoring: 3 tests (simple, empty, complex)
    - Sentence splitting: 3 tests (basic, abbreviations, numbers)
    - Syllable counting: 3 tests (single, multi, silent e)
    - Chunk scoring: 6 tests (single, weighted avg, batch, empty, high/low quality)

    Total: 29 tests

Design Decisions:
    1. Focus on quality scoring logic and metric calculation
    2. Use simple ChunkRecord mocks (no external dependencies)
    3. Test each scoring function separately
    4. Test main scoring workflow (score, score_batch)
    5. Test edge cases (empty text, poor quality, high quality)
    6. Simple, clear tests that verify scoring works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - QualityMetrics dataclass creation
    - QualityScorer initialization with default and custom weights
    - Coherence scoring (capitalization, punctuation, transition words)
    - Completeness scoring (sentence completeness, fragments)
    - Information density (content words vs stop words)
    - Readability scoring (Flesch-Kincaid simplified)
    - Sentence splitting (basic, abbreviations, decimal numbers)
    - Syllable counting (single, multi, silent e)
    - Single chunk scoring with all metrics
    - Batch scoring updates quality_score field
    - Overall score as weighted average
    - High vs low quality chunk differentiation

Justification:
    - Quality scoring is critical for filtering and ranking chunks
    - Multiple metrics provide comprehensive quality assessment
    - Scoring functions are deterministic and testable
    - Main workflow tests cover common scoring scenarios
    - Simple tests verify quality scoring system works correctly
"""
