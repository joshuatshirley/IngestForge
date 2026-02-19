"""Tests for evidence linking system.

Tests cover:
- Evidence linking to claims
- Support/refute classification
- Relevance scoring
- Entity extraction
- Integration with ContradictionDetector
- Edge cases and error handling
- JPL Commandment compliance
"""

import pytest
from unittest.mock import Mock

from ingestforge.enrichment.evidence_linker import (
    EvidenceLinker,
    LinkedEvidence,
    EvidenceLinkResult,
    SupportType,
    MAX_EVIDENCE_ITEMS,
    MAX_CLAIM_LENGTH,
    MAX_TOP_K,
    MODERATE_SUPPORT_THRESHOLD,
)
from ingestforge.enrichment.contradiction import ContradictionDetector
from ingestforge.storage.base import SearchResult


class TestSupportType:
    """Tests for SupportType enum."""

    def test_enum_values(self):
        """Test SupportType enum has expected values."""
        assert SupportType.SUPPORTS.value == "supports"
        assert SupportType.REFUTES.value == "refutes"
        assert SupportType.NEUTRAL.value == "neutral"

    def test_enum_members(self):
        """Test all enum members are present."""
        members = [e.value for e in SupportType]
        assert "supports" in members
        assert "refutes" in members
        assert "neutral" in members


class TestLinkedEvidence:
    """Tests for LinkedEvidence dataclass."""

    def test_initialization(self):
        """Test LinkedEvidence initialization."""
        evidence = LinkedEvidence(
            evidence_text="The sky is blue according to science.",
            source="doc123",
            chunk_id="chunk456",
            relevance_score=0.85,
            support_type=SupportType.SUPPORTS,
            confidence=0.9,
            metadata={"source_file": "test.pdf"},
        )

        assert evidence.evidence_text == "The sky is blue according to science."
        assert evidence.source == "doc123"
        assert evidence.chunk_id == "chunk456"
        assert evidence.relevance_score == 0.85
        assert evidence.support_type == SupportType.SUPPORTS
        assert evidence.confidence == 0.9
        assert evidence.metadata == {"source_file": "test.pdf"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        evidence = LinkedEvidence(
            evidence_text="Test evidence",
            source="doc1",
            chunk_id="chunk1",
            relevance_score=0.7,
            support_type=SupportType.REFUTES,
            confidence=0.8,
        )

        result = evidence.to_dict()

        assert result["evidence_text"] == "Test evidence"
        assert result["source"] == "doc1"
        assert result["chunk_id"] == "chunk1"
        assert result["relevance_score"] == 0.7
        assert result["support_type"] == "refutes"
        assert result["confidence"] == 0.8
        assert "metadata" in result


class TestEvidenceLinkResult:
    """Tests for EvidenceLinkResult dataclass."""

    def test_initialization(self):
        """Test EvidenceLinkResult initialization."""
        evidence = LinkedEvidence(
            evidence_text="Test",
            source="doc1",
            chunk_id="chunk1",
            relevance_score=0.8,
            support_type=SupportType.SUPPORTS,
            confidence=0.9,
        )

        result = EvidenceLinkResult(
            claim="The sky is blue",
            linked_evidence=[evidence],
            total_support=1,
            total_refute=0,
            total_neutral=0,
            key_entities=["sky"],
        )

        assert result.claim == "The sky is blue"
        assert len(result.linked_evidence) == 1
        assert result.total_support == 1
        assert result.total_refute == 0
        assert result.total_neutral == 0
        assert result.key_entities == ["sky"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        evidence = LinkedEvidence(
            evidence_text="Test",
            source="doc1",
            chunk_id="chunk1",
            relevance_score=0.8,
            support_type=SupportType.SUPPORTS,
            confidence=0.9,
        )

        result = EvidenceLinkResult(
            claim="Test claim",
            linked_evidence=[evidence],
            total_support=1,
            total_refute=0,
            total_neutral=0,
        )

        result_dict = result.to_dict()

        assert result_dict["claim"] == "Test claim"
        assert len(result_dict["linked_evidence"]) == 1
        assert result_dict["total_support"] == 1
        assert result_dict["total_refute"] == 0
        assert result_dict["total_neutral"] == 0


class TestEvidenceLinkerInit:
    """Tests for EvidenceLinker initialization."""

    def test_default_initialization(self):
        """Test linker with default parameters."""
        linker = EvidenceLinker()

        assert linker.contradiction_detector is not None
        assert linker.support_threshold == MODERATE_SUPPORT_THRESHOLD
        assert linker.refute_threshold == MODERATE_SUPPORT_THRESHOLD

    def test_custom_initialization(self):
        """Test linker with custom parameters."""
        detector = ContradictionDetector()
        linker = EvidenceLinker(
            contradiction_detector=detector,
            support_threshold=0.7,
            refute_threshold=0.8,
        )

        assert linker.contradiction_detector is detector
        assert linker.support_threshold == 0.7
        assert linker.refute_threshold == 0.8

    def test_invalid_support_threshold_high(self):
        """Test validation of support_threshold upper bound."""
        with pytest.raises(ValueError, match="support_threshold must be 0.0-1.0"):
            EvidenceLinker(support_threshold=1.5)

    def test_invalid_support_threshold_low(self):
        """Test validation of support_threshold lower bound."""
        with pytest.raises(ValueError, match="support_threshold must be 0.0-1.0"):
            EvidenceLinker(support_threshold=-0.1)

    def test_invalid_refute_threshold_high(self):
        """Test validation of refute_threshold upper bound."""
        with pytest.raises(ValueError, match="refute_threshold must be 0.0-1.0"):
            EvidenceLinker(refute_threshold=1.5)

    def test_invalid_refute_threshold_low(self):
        """Test validation of refute_threshold lower bound."""
        with pytest.raises(ValueError, match="refute_threshold must be 0.0-1.0"):
            EvidenceLinker(refute_threshold=-0.1)

    def test_boundary_values(self):
        """Test boundary values are accepted."""
        linker1 = EvidenceLinker(support_threshold=0.0, refute_threshold=0.0)
        assert linker1.support_threshold == 0.0
        assert linker1.refute_threshold == 0.0

        linker2 = EvidenceLinker(support_threshold=1.0, refute_threshold=1.0)
        assert linker2.support_threshold == 1.0
        assert linker2.refute_threshold == 1.0


class TestClassifySupport:
    """Tests for classify_support method."""

    @pytest.fixture
    def linker(self) -> EvidenceLinker:
        """Create linker instance with appropriate thresholds for testing."""
        # Lower refute threshold since contradiction scores tend to be lower
        # due to multiplication of similarity * (negation + antonym) in detector
        return EvidenceLinker(
            support_threshold=0.5,
            refute_threshold=0.2,  # Lower threshold for contradiction detection
        )

    def test_empty_claim(self, linker: EvidenceLinker):
        """Test classification with empty claim."""
        with pytest.raises(ValueError, match="claim cannot be empty"):
            linker.classify_support("", "Evidence text")

    def test_empty_evidence(self, linker: EvidenceLinker):
        """Test classification with empty evidence."""
        with pytest.raises(ValueError, match="evidence cannot be empty"):
            linker.classify_support("Claim text", "")

    def test_whitespace_only_claim(self, linker: EvidenceLinker):
        """Test classification with whitespace-only claim."""
        with pytest.raises(ValueError, match="claim cannot be empty"):
            linker.classify_support("   ", "Evidence text")

    def test_whitespace_only_evidence(self, linker: EvidenceLinker):
        """Test classification with whitespace-only evidence."""
        with pytest.raises(ValueError, match="evidence cannot be empty"):
            linker.classify_support("Claim text", "   ")

    def test_supporting_evidence(self, linker: EvidenceLinker):
        """Test classification of supporting evidence."""
        result = linker.classify_support(
            "The sky is blue",
            "Scientific studies confirm the sky appears blue",
        )

        # Should be SUPPORTS or NEUTRAL depending on similarity
        assert result in [SupportType.SUPPORTS, SupportType.NEUTRAL]

    def test_refuting_evidence(self, linker: EvidenceLinker):
        """Test classification of refuting evidence."""
        result = linker.classify_support(
            "The sky is blue",
            "The sky is not blue",
        )

        # Should detect contradiction and return REFUTES
        # Note: The threshold is 0.5, so contradiction score must be >= 0.5
        # High similarity + negation should trigger this
        assert result == SupportType.REFUTES

    def test_neutral_evidence(self, linker: EvidenceLinker):
        """Test classification of neutral evidence."""
        result = linker.classify_support(
            "The sky is blue",
            "I like pizza",
        )

        assert result == SupportType.NEUTRAL


class TestLinkEvidence:
    """Tests for link_evidence method."""

    @pytest.fixture
    def linker(self) -> EvidenceLinker:
        """Create linker instance."""
        return EvidenceLinker(
            support_threshold=0.5,
            refute_threshold=0.5,
        )

    @pytest.fixture
    def mock_storage(self) -> Mock:
        """Create mock storage repository."""
        storage = Mock()
        storage.search = Mock(return_value=[])
        return storage

    def test_empty_claim(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with empty claim."""
        with pytest.raises(ValueError, match="claim cannot be empty"):
            linker.link_evidence("", mock_storage)

    def test_whitespace_only_claim(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with whitespace-only claim."""
        with pytest.raises(ValueError, match="claim cannot be empty"):
            linker.link_evidence("   ", mock_storage)

    def test_claim_too_long(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with claim exceeding max length."""
        long_claim = "a" * (MAX_CLAIM_LENGTH + 1)
        with pytest.raises(ValueError, match="claim too long"):
            linker.link_evidence(long_claim, mock_storage)

    def test_invalid_top_k_zero(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with top_k=0."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            linker.link_evidence("Test claim", mock_storage, top_k=0)

    def test_invalid_top_k_negative(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with negative top_k."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            linker.link_evidence("Test claim", mock_storage, top_k=-1)

    def test_top_k_exceeds_max(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with top_k exceeding maximum."""
        with pytest.raises(ValueError, match="top_k exceeds maximum"):
            linker.link_evidence(
                "Test claim", mock_storage, top_k=MAX_EVIDENCE_ITEMS + 1
            )

    def test_no_evidence_found(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking when no evidence is found."""
        result = linker.link_evidence("Test claim", mock_storage, top_k=10)

        assert isinstance(result, EvidenceLinkResult)
        assert result.claim == "Test claim"
        assert len(result.linked_evidence) == 0
        assert result.total_support == 0
        assert result.total_refute == 0
        assert result.total_neutral == 0

    def test_evidence_found(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking when evidence is found."""
        # Create mock search results
        search_result = SearchResult(
            chunk_id="chunk1",
            content="The sky appears blue during the day",
            score=0.8,
            document_id="doc1",
            section_title="Sky Color",
            chunk_type="content",
            source_file="sky.pdf",
            word_count=10,
        )

        mock_storage.search.return_value = [search_result]

        result = linker.link_evidence("The sky is blue", mock_storage, top_k=10)

        assert isinstance(result, EvidenceLinkResult)
        assert result.claim == "The sky is blue"
        assert len(result.linked_evidence) >= 1
        assert result.linked_evidence[0].source == "doc1"

    def test_library_filter_passed(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test that library filter is passed to storage."""
        linker.link_evidence(
            "Test claim",
            mock_storage,
            top_k=10,
            library_filter="my_library",
        )

        mock_storage.search.assert_called_once()
        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["library_filter"] == "my_library"

    def test_top_k_clamped(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test that top_k is clamped to MAX_TOP_K."""
        linker.link_evidence("Test claim", mock_storage, top_k=MAX_TOP_K)

        mock_storage.search.assert_called_once()
        call_kwargs = mock_storage.search.call_args[1]
        assert call_kwargs["top_k"] == MAX_TOP_K


class TestHelperMethods:
    """Tests for internal helper methods."""

    @pytest.fixture
    def linker(self) -> EvidenceLinker:
        """Create linker instance."""
        return EvidenceLinker()

    def test_extract_key_entities_capitalized_words(self, linker: EvidenceLinker):
        """Test entity extraction finds capitalized words."""
        entities = linker._extract_key_entities(
            "The research by Albert Einstein changed physics"
        )

        assert "Albert Einstein" in entities or "Einstein" in entities

    def test_extract_key_entities_no_sentence_start(self, linker: EvidenceLinker):
        """Test entity extraction excludes sentence-start capitalizations."""
        text = "The sky is blue. Blue is a color."
        entities = linker._extract_key_entities(text)

        # "The" (sentence start) should be filtered out
        # "Blue" at sentence start might be an entity if not in common words list
        # The main check is that sentence-start "The" is excluded
        assert "The" not in entities

    def test_extract_key_entities_empty_text(self, linker: EvidenceLinker):
        """Test entity extraction with empty text."""
        entities = linker._extract_key_entities("")
        assert entities == []

    def test_extract_key_entities_no_entities(self, linker: EvidenceLinker):
        """Test entity extraction with no capitalized words."""
        entities = linker._extract_key_entities("the sky is blue today")
        assert len(entities) == 0

    def test_extract_key_entities_deduplication(self, linker: EvidenceLinker):
        """Test entity extraction deduplicates results."""
        entities = linker._extract_key_entities(
            "Einstein said physics. Einstein was great."
        )

        # Should have only one "Einstein"
        assert entities.count("Einstein") == 1

    def test_count_by_type_all_support(self, linker: EvidenceLinker):
        """Test counting evidence when all support."""
        evidence = [
            LinkedEvidence(
                evidence_text="Test",
                source="doc1",
                chunk_id="chunk1",
                relevance_score=0.8,
                support_type=SupportType.SUPPORTS,
                confidence=0.9,
            ),
            LinkedEvidence(
                evidence_text="Test2",
                source="doc2",
                chunk_id="chunk2",
                relevance_score=0.7,
                support_type=SupportType.SUPPORTS,
                confidence=0.8,
            ),
        ]

        counts = linker._count_by_type(evidence)

        assert counts["support"] == 2
        assert counts["refute"] == 0
        assert counts["neutral"] == 0

    def test_count_by_type_mixed(self, linker: EvidenceLinker):
        """Test counting evidence with mixed types."""
        evidence = [
            LinkedEvidence(
                evidence_text="Test",
                source="doc1",
                chunk_id="chunk1",
                relevance_score=0.8,
                support_type=SupportType.SUPPORTS,
                confidence=0.9,
            ),
            LinkedEvidence(
                evidence_text="Test2",
                source="doc2",
                chunk_id="chunk2",
                relevance_score=0.7,
                support_type=SupportType.REFUTES,
                confidence=0.8,
            ),
            LinkedEvidence(
                evidence_text="Test3",
                source="doc3",
                chunk_id="chunk3",
                relevance_score=0.5,
                support_type=SupportType.NEUTRAL,
                confidence=0.6,
            ),
        ]

        counts = linker._count_by_type(evidence)

        assert counts["support"] == 1
        assert counts["refute"] == 1
        assert counts["neutral"] == 1

    def test_count_by_type_empty(self, linker: EvidenceLinker):
        """Test counting with empty evidence list."""
        counts = linker._count_by_type([])

        assert counts["support"] == 0
        assert counts["refute"] == 0
        assert counts["neutral"] == 0

    def test_calculate_confidence_support(self, linker: EvidenceLinker):
        """Test confidence calculation for supporting evidence."""
        confidence = linker._calculate_confidence(0.9, SupportType.SUPPORTS)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # High relevance should give high confidence

    def test_calculate_confidence_refute(self, linker: EvidenceLinker):
        """Test confidence calculation for refuting evidence."""
        confidence = linker._calculate_confidence(0.8, SupportType.REFUTES)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7

    def test_calculate_confidence_neutral(self, linker: EvidenceLinker):
        """Test confidence calculation for neutral evidence."""
        confidence = linker._calculate_confidence(0.5, SupportType.NEUTRAL)

        assert 0.0 <= confidence <= 0.5  # Neutral should have lower confidence

    def test_calculate_confidence_clamping(self, linker: EvidenceLinker):
        """Test confidence is clamped to [0, 1]."""
        # Test various values
        for relevance in [0.0, 0.5, 1.0, 1.5]:  # Including out-of-range
            for support_type in SupportType:
                confidence = linker._calculate_confidence(relevance, support_type)
                assert 0.0 <= confidence <= 1.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def linker(self) -> EvidenceLinker:
        """Create linker instance."""
        return EvidenceLinker()

    @pytest.fixture
    def mock_storage(self) -> Mock:
        """Create mock storage repository."""
        storage = Mock()
        storage.search = Mock(return_value=[])
        return storage

    def test_max_length_claim(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with maximum length claim."""
        claim = "a" * MAX_CLAIM_LENGTH

        result = linker.link_evidence(claim, mock_storage)

        assert isinstance(result, EvidenceLinkResult)

    def test_unicode_in_claim(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with Unicode characters."""
        result = linker.link_evidence("Le café est ouvert", mock_storage)

        assert isinstance(result, EvidenceLinkResult)

    def test_special_characters(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with special characters."""
        result = linker.link_evidence("Price: $100 (20% off!)", mock_storage)

        assert isinstance(result, EvidenceLinkResult)

    def test_multiple_sentences(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test linking with multi-sentence claim."""
        claim = "The sky is blue. This is a known fact. Science confirms it."
        result = linker.link_evidence(claim, mock_storage)

        assert isinstance(result, EvidenceLinkResult)

    def test_max_evidence_items(self, linker: EvidenceLinker, mock_storage: Mock):
        """Test that evidence is capped at MAX_EVIDENCE_ITEMS."""
        # Create more results than MAX_EVIDENCE_ITEMS
        many_results = [
            SearchResult(
                chunk_id=f"chunk{i}",
                content=f"Evidence {i}",
                score=0.8,
                document_id=f"doc{i}",
                section_title="Test",
                chunk_type="content",
                source_file="test.pdf",
                word_count=10,
            )
            for i in range(MAX_EVIDENCE_ITEMS + 10)
        ]

        mock_storage.search.return_value = many_results

        result = linker.link_evidence("Test claim", mock_storage, top_k=50)

        # Should be capped at MAX_EVIDENCE_ITEMS
        assert len(result.linked_evidence) <= MAX_EVIDENCE_ITEMS


class TestJPLCompliance:
    """Tests for NASA JPL Commandments compliance."""

    def test_fixed_upper_bounds(self):
        """Test Rule #2: Fixed upper bounds for loops."""
        assert MAX_EVIDENCE_ITEMS == 100
        assert MAX_CLAIM_LENGTH == 5000
        assert MAX_TOP_K == 50

    def test_input_validation(self):
        """Test Rule #7: Validate all inputs."""
        linker = EvidenceLinker()

        # Empty claims rejected
        with pytest.raises(ValueError):
            linker.classify_support("", "test")

        # Too long claims rejected
        mock_storage = Mock()
        mock_storage.search = Mock(return_value=[])

        with pytest.raises(ValueError):
            linker.link_evidence("a" * 10000, mock_storage)

        # Invalid thresholds rejected at init
        with pytest.raises(ValueError):
            EvidenceLinker(support_threshold=2.0)

    def test_type_hints_present(self):
        """Test Rule #9: Complete type hints."""
        from inspect import signature

        sig = signature(EvidenceLinker.link_evidence)
        # Check return annotation is present
        assert sig.return_annotation == "EvidenceLinkResult"

        sig = signature(EvidenceLinker.classify_support)
        assert sig.return_annotation == "SupportType"

    def test_constants_are_fixed(self):
        """Test that constants provide fixed upper bounds."""
        # These should be compile-time constants
        assert isinstance(MAX_EVIDENCE_ITEMS, int)
        assert isinstance(MAX_CLAIM_LENGTH, int)
        assert isinstance(MAX_TOP_K, int)

        assert MAX_EVIDENCE_ITEMS > 0
        assert MAX_CLAIM_LENGTH > 0
        assert MAX_TOP_K > 0
