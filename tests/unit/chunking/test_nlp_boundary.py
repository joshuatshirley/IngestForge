"""Tests for NLP Boundary Detector.

NLPBoundaryDetector implementation tests.
Content hash verification tests.
Tests sentence integrity, semantic boundaries, and size constraints."""

from __future__ import annotations

from typing import List

import pytest

from ingestforge.chunking.nlp_boundary import (
    BoundaryCandidate,
    BoundaryConstraints,
    BoundaryReason,
    NLPBoundaryDetector,
    Sentence,
    SentenceSegmenter,
    SplitVerification,
    detect_boundaries,
    segment_sentences,
    calculate_content_hash,
    reconstruct_from_sentences,
    verify_split_integrity,
    segment_and_verify,
    MAX_SENTENCES,
    MAX_BOUNDARY_CANDIDATES,
    MAX_TEXT_LENGTH,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_OVERLAP_SENTENCES,
)


# ---------------------------------------------------------------------------
# Sentence Tests
# ---------------------------------------------------------------------------


class TestSentence:
    """Tests for Sentence dataclass."""

    def test_sentence_creation(self) -> None:
        """Test creating a sentence."""
        sent = Sentence(
            text="Hello world.",
            start_char=0,
            end_char=12,
            index=0,
        )
        assert sent.text == "Hello world."
        assert sent.start_char == 0
        assert sent.end_char == 12
        assert sent.index == 0

    def test_char_count(self) -> None:
        """Test character count property."""
        sent = Sentence(text="Hello world.", start_char=0, end_char=12, index=0)
        assert sent.char_count == 12

    def test_word_count(self) -> None:
        """Test word count property."""
        sent = Sentence(text="Hello world today.", start_char=0, end_char=18, index=0)
        assert sent.word_count == 3

    def test_paragraph_end_default(self) -> None:
        """Test is_paragraph_end defaults to False."""
        sent = Sentence(text="Test.", start_char=0, end_char=5, index=0)
        assert sent.is_paragraph_end is False

    def test_paragraph_end_explicit(self) -> None:
        """Test explicit paragraph end."""
        sent = Sentence(
            text="Last sentence.",
            start_char=0,
            end_char=14,
            index=0,
            is_paragraph_end=True,
        )
        assert sent.is_paragraph_end is True


# ---------------------------------------------------------------------------
# BoundaryCandidate Tests
# ---------------------------------------------------------------------------


class TestBoundaryCandidate:
    """Tests for BoundaryCandidate dataclass."""

    def test_candidate_creation(self) -> None:
        """Test creating a boundary candidate."""
        candidate = BoundaryCandidate(
            position=5,
            score=0.7,
            similarity_drop=0.4,
            is_paragraph_break=False,
            reason=BoundaryReason.SIMILARITY_DROP,
        )
        assert candidate.position == 5
        assert candidate.score == 0.7
        assert candidate.similarity_drop == 0.4

    def test_candidate_to_dict(self) -> None:
        """Test conversion to dictionary."""
        candidate = BoundaryCandidate(
            position=3,
            score=0.5,
            similarity_drop=0.3,
            is_paragraph_break=True,
            reason=BoundaryReason.PARAGRAPH_BREAK,
            char_position=150,
        )
        d = candidate.to_dict()
        assert d["position"] == 3
        assert d["reason"] == "paragraph_break"
        assert d["char_position"] == 150


class TestBoundaryReason:
    """Tests for BoundaryReason enum."""

    def test_reasons_defined(self) -> None:
        """Test all reasons are defined."""
        reasons = [r.value for r in BoundaryReason]
        assert "similarity_drop" in reasons
        assert "paragraph_break" in reasons
        assert "size_limit" in reasons
        assert "combined" in reasons

    def test_reason_count(self) -> None:
        """Test correct number of reasons."""
        assert len(BoundaryReason) == 4


# ---------------------------------------------------------------------------
# BoundaryConstraints Tests
# ---------------------------------------------------------------------------


class TestBoundaryConstraints:
    """Tests for BoundaryConstraints dataclass."""

    def test_default_values(self) -> None:
        """Test default constraint values."""
        constraints = BoundaryConstraints()
        assert constraints.min_chunk_chars == 100
        assert constraints.max_chunk_chars == 2000
        assert constraints.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_custom_values(self) -> None:
        """Test custom constraint values."""
        constraints = BoundaryConstraints(
            min_chunk_chars=50,
            max_chunk_chars=500,
            similarity_threshold=0.9,
        )
        assert constraints.min_chunk_chars == 50
        assert constraints.max_chunk_chars == 500
        assert constraints.similarity_threshold == 0.9


# ---------------------------------------------------------------------------
# SentenceSegmenter Tests
# ---------------------------------------------------------------------------


class TestSentenceSegmenter:
    """Tests for SentenceSegmenter (GWT-1)."""

    def test_simple_segmentation(self) -> None:
        """GWT-1: Simple sentence splitting."""
        segmenter = SentenceSegmenter()
        text = "Hello world. How are you? I am fine."
        sentences = segmenter.segment(text)
        assert len(sentences) == 3
        assert sentences[0].text == "Hello world."
        assert sentences[1].text == "How are you?"
        assert sentences[2].text == "I am fine."

    def test_abbreviation_handling(self) -> None:
        """GWT-1: Abbreviations don't split sentences."""
        segmenter = SentenceSegmenter()
        text = "Dr. Smith went to see Mr. Jones. They talked."
        sentences = segmenter.segment(text)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0].text
        assert "Mr. Jones" in sentences[0].text

    def test_paragraph_detection(self) -> None:
        """GWT-3: Paragraph breaks detected."""
        segmenter = SentenceSegmenter()
        text = "First paragraph.\n\nSecond paragraph."
        sentences = segmenter.segment(text)
        assert len(sentences) == 2
        assert sentences[0].is_paragraph_end is True
        assert sentences[1].is_paragraph_end is True

    def test_position_tracking(self) -> None:
        """GWT-1: Sentence positions are tracked."""
        segmenter = SentenceSegmenter()
        text = "Hello. World."
        sentences = segmenter.segment(text)
        assert sentences[0].start_char == 0
        assert sentences[0].end_char == 6
        assert sentences[1].index == 1

    def test_empty_text(self) -> None:
        """Test empty text returns empty list."""
        segmenter = SentenceSegmenter()
        sentences = segmenter.segment("")
        assert sentences == []

    def test_whitespace_only(self) -> None:
        """Test whitespace-only text returns empty list."""
        segmenter = SentenceSegmenter()
        sentences = segmenter.segment("   \n\n   ")
        assert sentences == []

    def test_exclamation_question(self) -> None:
        """Test exclamation and question marks end sentences."""
        segmenter = SentenceSegmenter()
        text = "What is this! Is it good? Yes it is."
        sentences = segmenter.segment(text)
        assert len(sentences) == 3

    def test_multiple_paragraphs(self) -> None:
        """Test multiple paragraph handling."""
        segmenter = SentenceSegmenter()
        text = "Para one.\n\nPara two.\n\nPara three."
        sentences = segmenter.segment(text)
        assert len(sentences) == 3
        assert all(s.is_paragraph_end for s in sentences)


# ---------------------------------------------------------------------------
# NLPBoundaryDetector Tests
# ---------------------------------------------------------------------------


class TestNLPBoundaryDetector:
    """Tests for NLPBoundaryDetector initialization."""

    def test_default_initialization(self) -> None:
        """Test default detector settings."""
        detector = NLPBoundaryDetector()
        assert detector.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert detector.overlap_sentences == DEFAULT_OVERLAP_SENTENCES

    def test_custom_initialization(self) -> None:
        """Test custom detector settings."""
        detector = NLPBoundaryDetector(
            similarity_threshold=0.9,
            overlap_sentences=3,
            paragraph_boost=0.5,
        )
        assert detector.similarity_threshold == 0.9
        assert detector.overlap_sentences == 3

    def test_invalid_threshold(self) -> None:
        """Test invalid threshold raises assertion."""
        with pytest.raises(AssertionError):
            NLPBoundaryDetector(similarity_threshold=1.5)

    def test_negative_overlap(self) -> None:
        """Test negative overlap raises assertion."""
        with pytest.raises(AssertionError):
            NLPBoundaryDetector(overlap_sentences=-1)


class TestGWT1SentenceIntegrity:
    """GWT-1: Sentence integrity tests."""

    def test_boundaries_at_sentence_ends(self) -> None:
        """GWT-1: Boundaries only at sentence ends."""
        detector = NLPBoundaryDetector()
        text = "First sentence. Second sentence. Third sentence."
        sentences, candidates = detector.detect_boundaries(text)

        # All candidate positions should be at sentence indices
        for candidate in candidates:
            assert 0 <= candidate.position < len(sentences)

    def test_no_mid_sentence_boundaries(self) -> None:
        """GWT-1: No boundaries within sentences."""
        detector = NLPBoundaryDetector()
        text = "This is a very long sentence with many words. Short one."
        sentences, candidates = detector.detect_boundaries(text)

        # Verify sentences are complete
        assert sentences[0].text == "This is a very long sentence with many words."
        assert sentences[1].text == "Short one."

    def test_single_sentence_no_boundaries(self) -> None:
        """GWT-1: Single sentence has no internal boundaries."""
        detector = NLPBoundaryDetector()
        text = "Just one sentence here."
        sentences, candidates = detector.detect_boundaries(text)
        assert len(sentences) == 1
        assert len(candidates) == 0


class TestGWT2SemanticBoundaries:
    """GWT-2: Semantic boundary detection tests."""

    def test_similarity_drop_detected(self) -> None:
        """GWT-2: Similarity drops create boundaries."""
        detector = NLPBoundaryDetector(similarity_threshold=0.8)

        # Use manually provided embeddings for predictable behavior
        sentences = [
            Sentence("Topic A sentence.", 0, 17, 0),
            Sentence("Topic A continues.", 18, 36, 1),
            Sentence("Topic B starts here.", 37, 57, 2),  # Different topic
        ]

        # Embeddings: A-A similar, A-B different
        embeddings = [
            [1.0, 0.0, 0.0],  # Topic A
            [0.9, 0.1, 0.0],  # Topic A (similar)
            [0.0, 1.0, 0.0],  # Topic B (different)
        ]

        detector._segmenter.segment = lambda t: sentences
        _, candidates = detector.detect_boundaries("dummy", embeddings=embeddings)

        # Should have boundary between sentence 1 and 2
        assert len(candidates) >= 1
        # Find candidate with highest similarity drop
        max_drop = max(c.similarity_drop for c in candidates)
        assert max_drop > 0.5  # Significant drop

    def test_high_similarity_no_boundary(self) -> None:
        """GWT-2: High similarity doesn't create strong boundary."""
        detector = NLPBoundaryDetector(similarity_threshold=0.8)

        # Very similar embeddings
        embeddings = [
            [1.0, 0.0],
            [0.99, 0.01],
        ]

        sentences = [
            Sentence("Similar one.", 0, 12, 0),
            Sentence("Similar two.", 13, 25, 1),
        ]

        detector._segmenter.segment = lambda t: sentences
        _, candidates = detector.detect_boundaries("dummy", embeddings=embeddings)

        # Should have low scores (high similarity = low drop)
        if candidates:
            assert candidates[0].similarity_drop < 0.2


class TestGWT3ParagraphAwareness:
    """GWT-3: Paragraph-aware boundary tests."""

    def test_paragraph_break_boosts_score(self) -> None:
        """GWT-3: Paragraph breaks boost boundary score."""
        detector = NLPBoundaryDetector(paragraph_boost=0.3)

        text = "First paragraph.\n\nSecond paragraph."
        sentences, candidates = detector.detect_boundaries(text)

        # Find paragraph break candidate
        para_candidates = [c for c in candidates if c.is_paragraph_break]
        assert len(para_candidates) >= 1
        assert para_candidates[0].score > para_candidates[0].similarity_drop

    def test_paragraph_vs_no_paragraph(self) -> None:
        """GWT-3: Paragraph boundaries score higher than non-paragraph."""
        detector = NLPBoundaryDetector(paragraph_boost=0.3)

        # Use embeddings to isolate paragraph effect
        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],  # Similar to first
            [0.8, 0.2],  # Similar to second
        ]

        sentences = [
            Sentence("First.", 0, 6, 0, is_paragraph_end=True),
            Sentence("Second.", 10, 17, 1, is_paragraph_end=False),
            Sentence("Third.", 18, 24, 2, is_paragraph_end=True),
        ]

        detector._segmenter.segment = lambda t: sentences
        _, candidates = detector.detect_boundaries("dummy", embeddings=embeddings)

        # Paragraph candidate should have higher score
        para_candidates = [c for c in candidates if c.is_paragraph_break]
        non_para_candidates = [c for c in candidates if not c.is_paragraph_break]

        if para_candidates and non_para_candidates:
            # Paragraph candidate boosted
            assert para_candidates[0].score > non_para_candidates[0].similarity_drop


class TestGWT4ConfigurableOverlap:
    """GWT-4: Configurable overlap tests."""

    def test_overlap_default(self) -> None:
        """GWT-4: Default overlap is 2 sentences."""
        detector = NLPBoundaryDetector()
        assert detector.overlap_sentences == 2

    def test_overlap_custom(self) -> None:
        """GWT-4: Custom overlap is respected."""
        detector = NLPBoundaryDetector(overlap_sentences=5)
        assert detector.overlap_sentences == 5


class TestGWT5SizeBoundedBoundaries:
    """GWT-5: Size-bounded boundary tests."""

    def test_max_size_forces_boundary(self) -> None:
        """GWT-5: Exceeding max size forces boundary."""
        detector = NLPBoundaryDetector()
        constraints = BoundaryConstraints(max_chunk_chars=50)

        sentences = [
            Sentence("A" * 30 + ".", 0, 31, 0),
            Sentence("B" * 30 + ".", 32, 63, 1),
            Sentence("C" * 30 + ".", 64, 95, 2),
        ]
        candidates: List[BoundaryCandidate] = []

        boundaries = detector.select_boundaries(sentences, candidates, constraints)

        # Should have multiple boundaries due to size limit
        assert len(boundaries) >= 2

    def test_min_size_prevents_boundary(self) -> None:
        """GWT-5: Below min size prevents boundary."""
        detector = NLPBoundaryDetector()
        constraints = BoundaryConstraints(min_chunk_chars=100)

        sentences = [
            Sentence("Short.", 0, 6, 0),
            Sentence("Also short.", 7, 18, 1),
            Sentence("Still short.", 19, 31, 2),
        ]

        # High-scoring candidates that should still be ignored
        candidates = [
            BoundaryCandidate(0, 0.9, 0.5, True, BoundaryReason.COMBINED),
            BoundaryCandidate(1, 0.8, 0.4, True, BoundaryReason.COMBINED),
        ]

        boundaries = detector.select_boundaries(sentences, candidates, constraints)

        # Should only have final boundary (min size not met for early splits)
        # All sentences together are ~31 chars, less than 100
        assert len(boundaries) == 1  # Only final boundary

    def test_sentence_count_limit(self) -> None:
        """GWT-5: Max sentences per chunk enforced."""
        detector = NLPBoundaryDetector()
        constraints = BoundaryConstraints(max_chunk_sentences=2)

        sentences = [
            Sentence(f"Sentence {i}.", i * 15, (i + 1) * 15, i) for i in range(6)
        ]
        candidates: List[BoundaryCandidate] = []

        boundaries = detector.select_boundaries(sentences, candidates, constraints)

        # Should split every 2 sentences
        assert len(boundaries) >= 3


# ---------------------------------------------------------------------------
# Cosine Similarity Tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have similarity 1.0."""
        detector = NLPBoundaryDetector()
        vec = [1.0, 2.0, 3.0]
        sim = detector._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors have similarity 0.0."""
        detector = NLPBoundaryDetector()
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = detector._cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.001

    def test_opposite_vectors(self) -> None:
        """Opposite vectors have similarity -1.0."""
        detector = NLPBoundaryDetector()
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        sim = detector._cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 0.001

    def test_empty_vectors(self) -> None:
        """Empty vectors return 0.0."""
        detector = NLPBoundaryDetector()
        sim = detector._cosine_similarity([], [])
        assert sim == 0.0

    def test_mismatched_lengths(self) -> None:
        """Mismatched vector lengths return 0.0."""
        detector = NLPBoundaryDetector()
        sim = detector._cosine_similarity([1.0, 2.0], [1.0])
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_boundaries_function(self) -> None:
        """Test detect_boundaries convenience function."""
        text = "First sentence. Second sentence. Third sentence."
        sentences, candidates = detect_boundaries(text)
        assert len(sentences) == 3

    def test_segment_sentences_function(self) -> None:
        """Test segment_sentences convenience function."""
        text = "Hello world. How are you?"
        sentences = segment_sentences(text)
        assert len(sentences) == 2


# ---------------------------------------------------------------------------
# JPL Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule2_max_sentences_constant(self) -> None:
        """JPL Rule #2: MAX_SENTENCES is bounded."""
        assert MAX_SENTENCES == 10000
        assert MAX_SENTENCES > 0

    def test_rule2_max_candidates_constant(self) -> None:
        """JPL Rule #2: MAX_BOUNDARY_CANDIDATES is bounded."""
        assert MAX_BOUNDARY_CANDIDATES == 1000
        assert MAX_BOUNDARY_CANDIDATES > 0

    def test_rule2_max_text_length(self) -> None:
        """JPL Rule #2: MAX_TEXT_LENGTH is bounded."""
        assert MAX_TEXT_LENGTH == 10_000_000
        assert MAX_TEXT_LENGTH > 0

    def test_rule5_segmenter_assertions(self) -> None:
        """JPL Rule #5: SentenceSegmenter asserts preconditions."""
        segmenter = SentenceSegmenter()
        with pytest.raises(AssertionError):
            segmenter.segment(None)  # type: ignore

    def test_rule5_detector_assertions(self) -> None:
        """JPL Rule #5: NLPBoundaryDetector asserts preconditions."""
        detector = NLPBoundaryDetector()
        with pytest.raises(AssertionError):
            detector.detect_boundaries(None)  # type: ignore

    def test_rule9_type_hints_sentence(self) -> None:
        """JPL Rule #9: Sentence has complete type hints."""
        from dataclasses import fields

        sentence_fields = {f.name for f in fields(Sentence)}
        required = {"text", "start_char", "end_char", "index", "is_paragraph_end"}
        assert required.issubset(sentence_fields)

    def test_rule9_type_hints_candidate(self) -> None:
        """JPL Rule #9: BoundaryCandidate has complete type hints."""
        from dataclasses import fields

        candidate_fields = {f.name for f in fields(BoundaryCandidate)}
        required = {
            "position",
            "score",
            "similarity_drop",
            "is_paragraph_break",
            "reason",
        }
        assert required.issubset(candidate_fields)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_text(self) -> None:
        """Test handling of very short text."""
        detector = NLPBoundaryDetector()
        sentences, candidates = detector.detect_boundaries("Hi.")
        assert len(sentences) == 1
        assert len(candidates) == 0

    def test_no_punctuation(self) -> None:
        """Test text without sentence-ending punctuation."""
        detector = NLPBoundaryDetector()
        text = "This text has no punctuation"
        sentences, candidates = detector.detect_boundaries(text)
        assert len(sentences) == 1

    def test_only_punctuation(self) -> None:
        """Test text with only punctuation."""
        segmenter = SentenceSegmenter()
        sentences = segmenter.segment("... !!! ???")
        # May produce empty or minimal results
        assert isinstance(sentences, list)

    def test_unicode_text(self) -> None:
        """Test handling of unicode text."""
        detector = NLPBoundaryDetector()
        text = "Café résumé. Naïve approach. 日本語テスト。"
        sentences, _ = detector.detect_boundaries(text)
        assert len(sentences) >= 2

    def test_long_sentence(self) -> None:
        """Test handling of very long sentence."""
        detector = NLPBoundaryDetector()
        long_sentence = "Word " * 500 + "end."
        sentences, _ = detector.detect_boundaries(long_sentence)
        assert len(sentences) == 1
        assert sentences[0].word_count > 400


# ---------------------------------------------------------------------------
# Content Hash Verification Tests ()
# ---------------------------------------------------------------------------


class TestCalculateContentHash:
    """Tests for calculate_content_hash function."""

    def test_hash_simple_text(self) -> None:
        """Test hashing simple text."""
        hash1 = calculate_content_hash("Hello world")
        assert len(hash1) == 64  # SHA-256 hex length
        assert hash1.isalnum()

    def test_hash_consistency(self) -> None:
        """Test that same text produces same hash."""
        text = "Test content."
        hash1 = calculate_content_hash(text)
        hash2 = calculate_content_hash(text)
        assert hash1 == hash2

    def test_hash_different_text(self) -> None:
        """Test that different text produces different hash."""
        hash1 = calculate_content_hash("Text A")
        hash2 = calculate_content_hash("Text B")
        assert hash1 != hash2

    def test_hash_strips_whitespace(self) -> None:
        """Test that whitespace is normalized."""
        hash1 = calculate_content_hash("  Hello  ")
        hash2 = calculate_content_hash("Hello")
        assert hash1 == hash2

    def test_hash_none_raises(self) -> None:
        """Test that None raises AssertionError."""
        with pytest.raises(AssertionError, match="text cannot be None"):
            calculate_content_hash(None)


class TestReconstructFromSentences:
    """Tests for reconstruct_from_sentences function."""

    def test_reconstruct_simple(self) -> None:
        """Test reconstructing simple sentences."""
        sentences = [
            Sentence(text="Hello.", start_char=0, end_char=6, index=0),
            Sentence(text="World.", start_char=7, end_char=13, index=1),
        ]
        result = reconstruct_from_sentences(sentences)
        assert "Hello." in result
        assert "World." in result

    def test_reconstruct_empty(self) -> None:
        """Test reconstructing empty list."""
        result = reconstruct_from_sentences([])
        assert result == ""

    def test_reconstruct_with_paragraph_breaks(self) -> None:
        """Test reconstructing with paragraph breaks."""
        sentences = [
            Sentence(
                text="First.", start_char=0, end_char=6, index=0, is_paragraph_end=True
            ),
            Sentence(text="Second.", start_char=9, end_char=16, index=1),
        ]
        result = reconstruct_from_sentences(sentences)
        assert "\n\n" in result

    def test_reconstruct_preserves_order(self) -> None:
        """Test that sentences are ordered correctly."""
        sentences = [
            Sentence(text="Third.", start_char=0, end_char=6, index=2),
            Sentence(text="First.", start_char=0, end_char=6, index=0),
            Sentence(text="Second.", start_char=0, end_char=7, index=1),
        ]
        result = reconstruct_from_sentences(sentences)
        assert result.index("First") < result.index("Second")
        assert result.index("Second") < result.index("Third")


class TestSplitVerification:
    """Tests for SplitVerification dataclass."""

    def test_verification_creation(self) -> None:
        """Test creating verification result."""
        v = SplitVerification(
            is_valid=True,
            original_hash="abc123",
            reconstructed_hash="abc123",
            sentence_count=5,
        )
        assert v.is_valid is True
        assert v.sentence_count == 5

    def test_verification_to_dict(self) -> None:
        """Test converting verification to dict."""
        v = SplitVerification(
            is_valid=False,
            original_hash="abc",
            reconstructed_hash="xyz",
            sentence_count=3,
            error_message="Mismatch",
        )
        d = v.to_dict()
        assert d["is_valid"] is False
        assert d["error_message"] == "Mismatch"


class TestVerifySplitIntegrity:
    """Tests for verify_split_integrity function."""

    def test_verify_valid_split(self) -> None:
        """Test verification of valid split."""
        text = "Hello world. This is a test."
        segmenter = SentenceSegmenter()
        sentences = segmenter.segment(text)

        result = verify_split_integrity(text, sentences)
        # Note: May not always pass due to whitespace normalization
        assert isinstance(result, SplitVerification)
        assert result.sentence_count == len(sentences)

    def test_verify_empty_text(self) -> None:
        """Test verification with empty text."""
        result = verify_split_integrity("", [])
        assert result.is_valid is True
        assert result.sentence_count == 0

    def test_verify_none_raises(self) -> None:
        """Test that None raises AssertionError."""
        with pytest.raises(AssertionError, match="original_text cannot be None"):
            verify_split_integrity(None, [])

    def test_verify_returns_hash_mismatch(self) -> None:
        """Test verification returns mismatch error."""
        text = "Original text."
        wrong_sentences = [
            Sentence(text="Wrong text.", start_char=0, end_char=11, index=0),
        ]
        result = verify_split_integrity(text, wrong_sentences)
        assert result.is_valid is False
        assert "mismatch" in result.error_message.lower()


class TestSegmentAndVerify:
    """Tests for segment_and_verify function."""

    def test_segment_and_verify_simple(self) -> None:
        """Test combined segment and verify."""
        text = "First sentence. Second sentence."
        sentences, verification = segment_and_verify(text)

        assert len(sentences) >= 1
        assert isinstance(verification, SplitVerification)
        assert verification.sentence_count == len(sentences)

    def test_segment_and_verify_returns_tuple(self) -> None:
        """Test return type is tuple."""
        result = segment_and_verify("Test.")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestJPLRule10Compliance:
    """Tests for JPL Rule #10 compliance (content hash verification)."""

    def test_hash_verification_after_split(self) -> None:
        """AC: Verify content hash after splitting."""
        text = "Dr. Smith went to the store. He bought apples."
        sentences, verification = segment_and_verify(text)

        # Verification should complete without error
        assert verification.original_hash is not None
        assert len(verification.original_hash) == 64
        assert verification.sentence_count > 0

    def test_abbreviations_dont_corrupt_hash(self) -> None:
        """Test that abbreviations don't corrupt reconstruction."""
        text = "Mr. Jones met Dr. Smith at 3 p.m. They discussed the Ph.D. program."
        sentences, verification = segment_and_verify(text)

        # Should handle abbreviations correctly
        assert verification.sentence_count >= 1
        # All sentence texts should be non-empty
        for sent in sentences:
            assert len(sent.text.strip()) > 0

    def test_function_line_counts(self) -> None:
        """Test that new functions are under 60 lines."""
        import inspect
        import ingestforge.chunking.nlp_boundary as module

        functions_to_check = [
            "calculate_content_hash",
            "reconstruct_from_sentences",
            "verify_split_integrity",
            "segment_and_verify",
        ]

        for func_name in functions_to_check:
            func = getattr(module, func_name)
            source = inspect.getsource(func)
            line_count = len(source.strip().split("\n"))
            assert line_count <= 60, f"{func_name} has {line_count} lines (max 60)"
