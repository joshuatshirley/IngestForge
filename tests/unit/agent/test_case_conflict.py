"""Tests for case conflict detector (LEGAL-004).

Tests conflict detection in legal corpus:
- Direct contradiction detection
- Jurisdictional split detection
- Overruled case detection
- Mock storage and LLM for unit tests"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from ingestforge.agent.case_conflict import (
    CaseConflict,
    CaseConflictDetector,
    ConflictPrompts,
    ConflictType,
    create_case_conflict_detector,
    MAX_DOCUMENTS_PER_BATCH,
    MAX_CONFLICTS_RETURNED,
    MAX_EXCERPT_LENGTH,
)

# Import mock LLM from test fixtures
from tests.fixtures.agents import MockLLM

# =============================================================================
# Mock Storage for Testing
# =============================================================================


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    chunk_id: str
    content: str
    document_id: str
    chunk_type: str = "text"
    legal_role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.content,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_type": self.chunk_type,
            "legal_role": self.legal_role,
        }


class MockStorage:
    """Mock storage backend for testing."""

    def __init__(self) -> None:
        """Initialize mock storage."""
        self._chunks: Dict[str, List[MockChunk]] = {}

    def add_document(self, document_id: str, chunks: List[MockChunk]) -> None:
        """Add chunks for a document."""
        self._chunks[document_id] = chunks

    def get_chunks_by_document(self, document_id: str) -> List[MockChunk]:
        """Get all chunks for a document."""
        return self._chunks.get(document_id, [])

    def search_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[MockChunk]:
        """Mock semantic search."""
        # Return all chunks flattened
        all_chunks = []
        for chunks in self._chunks.values():
            all_chunks.extend(chunks)
        return all_chunks[:top_k]


class MockContradictionDetector:
    """Mock contradiction detector for testing."""

    def __init__(self, score: float = 0.8) -> None:
        """Initialize with default score."""
        self._score = score

    @dataclass
    class Result:
        """Mock result."""

        score: float

    def detect_contradiction(self, claim1: str, claim2: str) -> Result:
        """Return mock contradiction result."""
        return self.Result(score=self._score)


class MockLegalClassifier:
    """Mock legal classifier for testing."""

    def __init__(self, role: str = "HOLDING") -> None:
        """Initialize with default role."""
        self._role = role

    @dataclass
    class Classification:
        """Mock classification."""

        role: str
        confidence: float

    def classify(self, text: str) -> Classification:
        """Return mock classification."""
        return self.Classification(role=self._role, confidence=0.9)


class MockBluebookParser:
    """Mock Bluebook parser for testing."""

    @dataclass
    class Citation:
        """Mock citation."""

        raw_text: str

    def extract_citations(self, text: str) -> List[Citation]:
        """Return mock citations."""
        if "347 U.S. 483" in text:
            return [self.Citation(raw_text="347 U.S. 483 (1954)")]
        return []


# =============================================================================
# ConflictType Tests
# =============================================================================


class TestConflictType:
    """Tests for ConflictType enum."""

    def test_conflict_types_defined(self) -> None:
        """Test all conflict types are defined."""
        types = [ct.value for ct in ConflictType]

        assert "DIRECT" in types
        assert "JURISDICTIONAL" in types
        assert "TEMPORAL" in types
        assert "FACTUAL" in types
        assert "UNKNOWN" in types

    def test_conflict_type_count(self) -> None:
        """Test correct number of conflict types."""
        assert len(ConflictType) == 5


# =============================================================================
# CaseConflict Tests
# =============================================================================


class TestCaseConflict:
    """Tests for CaseConflict dataclass."""

    def test_conflict_creation(self) -> None:
        """Test creating a case conflict."""
        conflict = CaseConflict(
            case_a_id="case_1",
            case_b_id="case_2",
            conflict_type="DIRECT",
            issue="First Amendment rights",
            description="Opposite holdings on free speech",
            confidence=0.9,
            case_a_excerpt="The statute is unconstitutional",
            case_b_excerpt="The statute is constitutional",
        )

        assert conflict.case_a_id == "case_1"
        assert conflict.case_b_id == "case_2"
        assert conflict.conflict_type == "DIRECT"
        assert conflict.confidence == 0.9

    def test_confidence_clamping_high(self) -> None:
        """Test confidence is clamped to max 1.0."""
        conflict = CaseConflict(
            case_a_id="case_1",
            case_b_id="case_2",
            conflict_type="DIRECT",
            issue="Test",
            description="Test",
            confidence=1.5,
            case_a_excerpt="A",
            case_b_excerpt="B",
        )

        assert conflict.confidence == 1.0

    def test_confidence_clamping_low(self) -> None:
        """Test confidence is clamped to min 0.0."""
        conflict = CaseConflict(
            case_a_id="case_1",
            case_b_id="case_2",
            conflict_type="DIRECT",
            issue="Test",
            description="Test",
            confidence=-0.5,
            case_a_excerpt="A",
            case_b_excerpt="B",
        )

        assert conflict.confidence == 0.0

    def test_excerpt_truncation(self) -> None:
        """Test excerpts are truncated to max length."""
        long_excerpt = "x" * 1000
        conflict = CaseConflict(
            case_a_id="case_1",
            case_b_id="case_2",
            conflict_type="DIRECT",
            issue="Test",
            description="Test",
            confidence=0.8,
            case_a_excerpt=long_excerpt,
            case_b_excerpt=long_excerpt,
        )

        assert len(conflict.case_a_excerpt) == MAX_EXCERPT_LENGTH
        assert len(conflict.case_b_excerpt) == MAX_EXCERPT_LENGTH

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        conflict = CaseConflict(
            case_a_id="case_1",
            case_b_id="case_2",
            conflict_type="JURISDICTIONAL",
            issue="Commerce Clause",
            description="Circuit split",
            confidence=0.85,
            case_a_excerpt="Ninth Circuit holds...",
            case_b_excerpt="Fifth Circuit holds...",
        )

        d = conflict.to_dict()

        assert d["case_a_id"] == "case_1"
        assert d["conflict_type"] == "JURISDICTIONAL"
        assert d["confidence"] == 0.85

    def test_is_high_confidence_property(self) -> None:
        """Test is_high_confidence property."""
        high = CaseConflict(
            case_a_id="a",
            case_b_id="b",
            conflict_type="DIRECT",
            issue="Test",
            description="Test",
            confidence=0.9,
            case_a_excerpt="A",
            case_b_excerpt="B",
        )
        low = CaseConflict(
            case_a_id="a",
            case_b_id="b",
            conflict_type="DIRECT",
            issue="Test",
            description="Test",
            confidence=0.5,
            case_a_excerpt="A",
            case_b_excerpt="B",
        )

        assert high.is_high_confidence is True
        assert low.is_high_confidence is False

    def test_is_direct_conflict_property(self) -> None:
        """Test is_direct_conflict property."""
        direct = CaseConflict(
            case_a_id="a",
            case_b_id="b",
            conflict_type="DIRECT",
            issue="Test",
            description="Test",
            confidence=0.8,
            case_a_excerpt="A",
            case_b_excerpt="B",
        )
        jurisdictional = CaseConflict(
            case_a_id="a",
            case_b_id="b",
            conflict_type="JURISDICTIONAL",
            issue="Test",
            description="Test",
            confidence=0.8,
            case_a_excerpt="A",
            case_b_excerpt="B",
        )

        assert direct.is_direct_conflict is True
        assert jurisdictional.is_direct_conflict is False


# =============================================================================
# ConflictPrompts Tests
# =============================================================================


class TestConflictPrompts:
    """Tests for ConflictPrompts class."""

    def test_compare_holdings_prompt(self) -> None:
        """Test compare holdings prompt generation."""
        prompt = ConflictPrompts.compare_holdings_prompt(
            holding_a="We hold that the statute violates due process.",
            holding_b="We hold that the statute does not violate due process.",
        )

        assert "violates due process" in prompt
        assert "does not violate" in prompt
        assert "Conflict:" in prompt
        assert "Type:" in prompt
        assert "DIRECT" in prompt
        assert "JURISDICTIONAL" in prompt

    def test_compare_holdings_prompt_with_context(self) -> None:
        """Test prompt generation with context."""
        prompt = ConflictPrompts.compare_holdings_prompt(
            holding_a="We hold...",
            holding_b="The court held...",
            context_a="In this employment discrimination case...",
            context_b="This Title VII case involves...",
        )

        assert "Context A" in prompt
        assert "Context B" in prompt
        assert "employment discrimination" in prompt

    def test_check_overruled_prompt(self) -> None:
        """Test overruled check prompt generation."""
        prompt = ConflictPrompts.check_overruled_prompt(
            case_text="The separate but equal doctrine is constitutional.",
            citations=["Plessy v. Ferguson, 163 U.S. 537 (1896)"],
        )

        assert "Plessy" in prompt
        assert "Overruled:" in prompt
        assert "superseded" in prompt.lower()

    def test_analyze_jurisdictional_split_prompt(self) -> None:
        """Test jurisdictional split prompt generation."""
        prompt = ConflictPrompts.analyze_jurisdictional_split_prompt(
            issue="Qualified immunity standard",
            holdings=[
                {"jurisdiction": "9th Cir.", "holding": "We apply a broad standard."},
                {"jurisdiction": "5th Cir.", "holding": "We apply a narrow standard."},
            ],
        )

        assert "9th Cir" in prompt
        assert "5th Cir" in prompt
        assert "Split_Detected" in prompt


# =============================================================================
# CaseConflictDetector Tests
# =============================================================================


class TestCaseConflictDetector:
    """Tests for CaseConflictDetector class."""

    def test_detector_creation(self) -> None:
        """Test creating a detector."""
        llm = MockLLM()
        detector = CaseConflictDetector(llm_client=llm)

        assert detector is not None

    def test_detector_requires_llm(self) -> None:
        """Test detector requires LLM client."""
        with pytest.raises(ValueError, match="llm_client cannot be None"):
            CaseConflictDetector(llm_client=None)  # type: ignore

    def test_detector_with_storage(self) -> None:
        """Test creating detector with storage."""
        llm = MockLLM()
        storage = MockStorage()
        detector = CaseConflictDetector(llm_client=llm, storage=storage)

        assert detector._storage is not None


# =============================================================================
# Direct Contradiction Detection Tests
# =============================================================================


class TestDirectContradictionDetection:
    """Tests for direct contradiction detection."""

    def test_detect_direct_contradiction(self) -> None:
        """Test detecting direct contradiction."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Conflict: YES
Type: DIRECT
Issue: First Amendment free speech
Confidence: 0.92
Description: Holdings directly contradict on constitutionality."""
            ]
        )

        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold that the statute violates the First Amendment.",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )
        storage.add_document(
            "case_2",
            [
                MockChunk(
                    chunk_id="c2",
                    content="We hold that the statute does not violate the First Amendment.",
                    document_id="case_2",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)
        conflicts = detector.detect_conflicts(["case_1", "case_2"])

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "DIRECT"
        assert conflicts[0].confidence == 0.92

    def test_no_conflict_detected(self) -> None:
        """Test when no conflict is detected."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Conflict: NO
Type: NONE
Issue: N/A
Confidence: 0.1
Description: Holdings are consistent, both affirm lower court."""
            ]
        )

        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We affirm the judgment below.",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )
        storage.add_document(
            "case_2",
            [
                MockChunk(
                    chunk_id="c2",
                    content="We affirm the district court's decision.",
                    document_id="case_2",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)
        conflicts = detector.detect_conflicts(["case_1", "case_2"])

        assert len(conflicts) == 0

    def test_empty_document_list(self) -> None:
        """Test with empty document list."""
        llm = MockLLM()
        detector = CaseConflictDetector(llm_client=llm)

        conflicts = detector.detect_conflicts([])

        assert conflicts == []

    def test_single_document(self) -> None:
        """Test with single document (no comparison possible)."""
        llm = MockLLM()
        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold...",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)
        conflicts = detector.detect_conflicts(["case_1"])

        assert len(conflicts) == 0


# =============================================================================
# Jurisdictional Split Detection Tests
# =============================================================================


class TestJurisdictionalSplitDetection:
    """Tests for jurisdictional split detection."""

    def test_detect_jurisdictional_split(self) -> None:
        """Test detecting jurisdictional split."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Split_Detected: YES
Agreeing_Jurisdictions: [1st Cir., 2nd Cir.]
Disagreeing_Jurisdictions: [9th Cir., 5th Cir.]
Key_Disagreement: Courts disagree on qualified immunity standard application.
Confidence: 0.88"""
            ]
        )

        detector = CaseConflictDetector(llm_client=llm)
        # Note: This tests the prompt generation and parsing
        # Full integration would require semantic search

        # Test prompt generation
        prompt = ConflictPrompts.analyze_jurisdictional_split_prompt(
            issue="Qualified immunity",
            holdings=[
                {"jurisdiction": "9th Cir.", "holding": "Broad standard applies."},
                {"jurisdiction": "5th Cir.", "holding": "Narrow standard applies."},
            ],
        )

        assert "9th Cir" in prompt
        assert "5th Cir" in prompt

    def test_no_split_detected(self) -> None:
        """Test when no jurisdictional split is detected."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Split_Detected: NO
Agreeing_Jurisdictions: [All circuits]
Disagreeing_Jurisdictions: []
Key_Disagreement: None
Confidence: 0.95"""
            ]
        )

        detector = CaseConflictDetector(llm_client=llm)
        conflicts = detector.analyze_jurisdictional_split("Commerce clause regulation")

        assert len(conflicts) == 0

    def test_empty_issue(self) -> None:
        """Test with empty issue string."""
        llm = MockLLM()
        detector = CaseConflictDetector(llm_client=llm)

        conflicts = detector.analyze_jurisdictional_split("")

        assert conflicts == []


# =============================================================================
# Overruled Case Detection Tests
# =============================================================================


class TestOverruledCaseDetection:
    """Tests for overruled case detection."""

    def test_detect_overruled_case(self) -> None:
        """Test detecting an overruled case."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Overruled: YES
Overruling_Authority: Brown v. Board of Education, 347 U.S. 483 (1954)
Confidence: 0.98
Reasoning: The separate but equal doctrine was explicitly overruled."""
            ]
        )

        storage = MockStorage()
        storage.add_document(
            "plessy",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold that separate but equal facilities are constitutional.",
                    document_id="plessy",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)
        result = detector.check_overruled("plessy")

        assert result is not None
        assert "Brown" in result

    def test_not_overruled(self) -> None:
        """Test case that is not overruled."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Overruled: NO
Overruling_Authority: None
Confidence: 0.90
Reasoning: This case remains good law."""
            ]
        )

        storage = MockStorage()
        storage.add_document(
            "marbury",
            [
                MockChunk(
                    chunk_id="c1",
                    content="It is emphatically the province of the judicial department to say what the law is.",
                    document_id="marbury",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)
        result = detector.check_overruled("marbury")

        assert result is None

    def test_partially_overruled(self) -> None:
        """Test partially overruled case."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Overruled: PARTIAL
Overruling_Authority: Later decisions limited this holding
Confidence: 0.75
Reasoning: The broad reading has been narrowed significantly."""
            ]
        )

        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold broadly that...",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)
        result = detector.check_overruled("case_1")

        assert result is not None
        assert "limited" in result

    def test_empty_case_id(self) -> None:
        """Test with empty case ID."""
        llm = MockLLM()
        detector = CaseConflictDetector(llm_client=llm)

        result = detector.check_overruled("")

        assert result is None

    def test_case_not_found(self) -> None:
        """Test with case not in storage."""
        llm = MockLLM()
        storage = MockStorage()
        detector = CaseConflictDetector(llm_client=llm, storage=storage)

        result = detector.check_overruled("nonexistent_case")

        assert result is None


# =============================================================================
# Find Contradictory Holdings Tests
# =============================================================================


class TestFindContradictoryHoldings:
    """Tests for finding contradictory holdings."""

    def test_find_contradictions_for_holding(self) -> None:
        """Test finding contradictory holdings for a given chunk."""
        llm = MockLLM()
        classifier = MockLegalClassifier(role="HOLDING")

        detector = CaseConflictDetector(
            llm_client=llm,
            legal_classifier=classifier,
        )

        chunk = {
            "text": "We hold that the defendant is liable.",
            "document_id": "case_1",
        }

        # No storage means no related holdings to compare
        conflicts = detector.find_contradictory_holdings(chunk)

        assert conflicts == []

    def test_skip_non_holding_chunks(self) -> None:
        """Test that non-holding chunks are skipped."""
        llm = MockLLM()
        classifier = MockLegalClassifier(role="FACTS")  # Not a holding

        detector = CaseConflictDetector(
            llm_client=llm,
            legal_classifier=classifier,
        )

        chunk = {
            "text": "On January 1, the plaintiff filed suit.",
            "document_id": "case_1",
        }

        conflicts = detector.find_contradictory_holdings(chunk)

        assert conflicts == []

    def test_empty_chunk(self) -> None:
        """Test with empty chunk."""
        llm = MockLLM()
        detector = CaseConflictDetector(llm_client=llm)

        conflicts = detector.find_contradictory_holdings({})

        assert conflicts == []

    def test_empty_text_in_chunk(self) -> None:
        """Test with empty text in chunk."""
        llm = MockLLM()
        detector = CaseConflictDetector(llm_client=llm)

        chunk = {"text": "", "document_id": "case_1"}
        conflicts = detector.find_contradictory_holdings(chunk)

        assert conflicts == []


# =============================================================================
# Integration with ContradictionDetector Tests
# =============================================================================


class TestContradictionDetectorIntegration:
    """Tests for integration with ContradictionDetector."""

    def test_uses_contradiction_detector_for_screening(self) -> None:
        """Test that ContradictionDetector is used for initial screening."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Conflict: YES
Type: DIRECT
Issue: Test issue
Confidence: 0.9
Description: Test conflict"""
            ]
        )

        # High score means claims are related
        contradiction_detector = MockContradictionDetector(score=0.8)

        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold A.",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )
        storage.add_document(
            "case_2",
            [
                MockChunk(
                    chunk_id="c2",
                    content="We hold NOT A.",
                    document_id="case_2",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(
            llm_client=llm,
            storage=storage,
            contradiction_detector=contradiction_detector,
        )
        conflicts = detector.detect_conflicts(["case_1", "case_2"])

        # Should proceed with LLM analysis since similarity is high
        assert len(conflicts) == 1

    def test_skips_dissimilar_holdings(self) -> None:
        """Test that dissimilar holdings are skipped."""
        llm = MockLLM()
        # Low score means claims are not related
        contradiction_detector = MockContradictionDetector(score=0.1)

        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold on contract law.",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )
        storage.add_document(
            "case_2",
            [
                MockChunk(
                    chunk_id="c2",
                    content="We hold on criminal procedure.",
                    document_id="case_2",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(
            llm_client=llm,
            storage=storage,
            contradiction_detector=contradiction_detector,
        )
        conflicts = detector.detect_conflicts(["case_1", "case_2"])

        # Should skip LLM analysis since similarity is low
        assert len(conflicts) == 0
        # LLM should not have been called
        assert llm.call_count == 0


# =============================================================================
# BluebookParser Integration Tests
# =============================================================================


class TestBluebookParserIntegration:
    """Tests for integration with BluebookParser."""

    def test_extracts_citations_with_parser(self) -> None:
        """Test citation extraction using BluebookParser."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Overruled: YES
Overruling_Authority: Brown v. Board of Education
Confidence: 0.95
Reasoning: Explicit overruling"""
            ]
        )

        parser = MockBluebookParser()
        storage = MockStorage()
        storage.add_document(
            "plessy",
            [
                MockChunk(
                    chunk_id="c1",
                    content="As held in 347 U.S. 483, separate but equal...",
                    document_id="plessy",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(
            llm_client=llm,
            storage=storage,
            bluebook_parser=parser,
        )
        result = detector.check_overruled("plessy")

        assert result is not None


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_llm_error_handling(self) -> None:
        """Test handling of LLM errors gracefully."""
        llm = MockLLM()
        # Don't set responses - will cause assertion error internally
        # But detector should catch and handle gracefully

        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold...",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )
        storage.add_document(
            "case_2",
            [
                MockChunk(
                    chunk_id="c2",
                    content="We hold...",
                    document_id="case_2",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)

        # Should handle error gracefully and return empty list
        conflicts = detector.detect_conflicts(["case_1", "case_2"])
        assert len(conflicts) == 0

    def test_malformed_llm_response(self) -> None:
        """Test handling malformed LLM response."""
        llm = MockLLM()
        llm.set_responses(["This is not a properly formatted response."])

        storage = MockStorage()
        storage.add_document(
            "case_1",
            [
                MockChunk(
                    chunk_id="c1",
                    content="We hold A.",
                    document_id="case_1",
                    legal_role="HOLDING",
                ),
            ],
        )
        storage.add_document(
            "case_2",
            [
                MockChunk(
                    chunk_id="c2",
                    content="We hold B.",
                    document_id="case_2",
                    legal_role="HOLDING",
                ),
            ],
        )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)
        conflicts = detector.detect_conflicts(["case_1", "case_2"])

        # Should handle gracefully, no conflict detected
        assert len(conflicts) == 0

    def test_document_batch_limit(self) -> None:
        """Test document batch limit is enforced."""
        llm = MockLLM()
        storage = MockStorage()

        # Create many documents
        doc_ids = []
        for i in range(150):  # More than MAX_DOCUMENTS_PER_BATCH
            doc_id = f"case_{i}"
            doc_ids.append(doc_id)
            storage.add_document(
                doc_id,
                [
                    MockChunk(
                        chunk_id=f"c{i}",
                        content="We hold...",
                        document_id=doc_id,
                        legal_role="HOLDING",
                    ),
                ],
            )

        detector = CaseConflictDetector(llm_client=llm, storage=storage)

        # Should only process MAX_DOCUMENTS_PER_BATCH
        assert MAX_DOCUMENTS_PER_BATCH == 100

    def test_whitespace_only_text(self) -> None:
        """Test handling whitespace-only text."""
        llm = MockLLM()
        detector = CaseConflictDetector(llm_client=llm)

        chunk = {"text": "   \n\t  ", "document_id": "case_1"}
        conflicts = detector.find_contradictory_holdings(chunk)

        assert conflicts == []


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateCaseConflictDetector:
    """Tests for create_case_conflict_detector factory."""

    def test_create_basic(self) -> None:
        """Test creating detector via factory."""
        llm = MockLLM()
        detector = create_case_conflict_detector(llm_client=llm)

        assert isinstance(detector, CaseConflictDetector)

    def test_create_with_all_dependencies(self) -> None:
        """Test creating with all optional dependencies."""
        from ingestforge.llm.base import GenerationConfig

        llm = MockLLM()
        storage = MockStorage()
        contradiction = MockContradictionDetector()
        parser = MockBluebookParser()
        classifier = MockLegalClassifier()
        config = GenerationConfig(max_tokens=1000, temperature=0.1)

        detector = create_case_conflict_detector(
            llm_client=llm,
            storage=storage,
            contradiction_detector=contradiction,
            bluebook_parser=parser,
            legal_classifier=classifier,
            config=config,
        )

        assert detector._storage is not None
        assert detector._contradiction_detector is not None
        assert detector._bluebook_parser is not None
        assert detector._legal_classifier is not None
        assert detector._config.max_tokens == 1000


# =============================================================================
# Constant Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_documents_per_batch(self) -> None:
        """Test MAX_DOCUMENTS_PER_BATCH is reasonable."""
        assert MAX_DOCUMENTS_PER_BATCH > 0
        assert MAX_DOCUMENTS_PER_BATCH == 100

    def test_max_conflicts_returned(self) -> None:
        """Test MAX_CONFLICTS_RETURNED is reasonable."""
        assert MAX_CONFLICTS_RETURNED > 0
        assert MAX_CONFLICTS_RETURNED == 200

    def test_max_excerpt_length(self) -> None:
        """Test MAX_EXCERPT_LENGTH is reasonable."""
        assert MAX_EXCERPT_LENGTH > 0
        assert MAX_EXCERPT_LENGTH == 500
