"""
Unit tests for Proactive Scout Agent - 

Tests gap detection, priority scoring, and discovery intent generation.

JPL Power of Ten Compliance:
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints

Epic: EP-06 (Agentic Intelligence)
Feature: (Proactive Scout)
Test Date: 2026-02-18
"""

import pytest
from unittest.mock import Mock

from ingestforge.agent.proactive_scout import (
    ProactiveScout,
    GapAnalysisResult,
    run_scout_analysis,
    MAX_ENTITIES_TO_ANALYZE,
    MIN_REFERENCES_FOR_WEAK_NODE,
)
from ingestforge.core.pipeline.knowledge_manifest import (
    IFKnowledgeManifest,
    ManifestEntry,
    EntityReference,
)
from ingestforge.core.pipeline.artifacts import IFDiscoveryIntentArtifact


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_manifest() -> Mock:
    """Create mock knowledge manifest."""
    manifest = Mock(spec=IFKnowledgeManifest)
    manifest.is_active = True
    return manifest


@pytest.fixture
def sample_entity_dangling() -> ManifestEntry:
    """Create dangling entity (no cross-document links)."""
    ref = EntityReference(
        document_id="doc1",
        artifact_id="art1",
        chunk_id="chunk1",
        confidence=0.95,
    )
    entry = ManifestEntry(
        entity_hash="hash1",
        entity_text="John Doe",
        entity_type="PERSON",
        references=[ref],
        first_seen_document="doc1",
    )
    return entry


@pytest.fixture
def sample_entity_weak() -> ManifestEntry:
    """Create weak entity (<2 references)."""
    ref = EntityReference(
        document_id="doc1",
        artifact_id="art1",
        chunk_id="chunk1",
        confidence=0.95,
    )
    entry = ManifestEntry(
        entity_hash="hash2",
        entity_text="Apple Inc",
        entity_type="ORG",
        references=[ref],
        first_seen_document="doc1",
    )
    return entry


@pytest.fixture
def sample_entity_strong() -> ManifestEntry:
    """Create strong entity (cross-document, multiple refs)."""
    refs = [
        EntityReference(
            document_id="doc1",
            artifact_id="art1",
            chunk_id="chunk1",
            confidence=0.95,
        ),
        EntityReference(
            document_id="doc2",
            artifact_id="art2",
            chunk_id="chunk2",
            confidence=0.92,
        ),
        EntityReference(
            document_id="doc3",
            artifact_id="art3",
            chunk_id="chunk3",
            confidence=0.88,
        ),
    ]
    entry = ManifestEntry(
        entity_hash="hash3",
        entity_text="Python",
        entity_type="TECH",
        references=refs,
        first_seen_document="doc1",
    )
    return entry


# =============================================================================
# INITIALIZATION TESTS (2 tests)
# =============================================================================


def test_given_no_args_when_scout_created_then_initializes_with_session_id() -> None:
    """
    GIVEN no arguments
    WHEN ProactiveScout is created
    THEN initializes with a unique session ID
    """
    scout = ProactiveScout()

    assert scout._session_id is not None
    assert len(scout._session_id) > 0


def test_given_two_scouts_when_created_then_have_different_session_ids() -> None:
    """
    GIVEN two ProactiveScout instances
    WHEN both are created
    THEN they have different session IDs
    """
    scout1 = ProactiveScout()
    scout2 = ProactiveScout()

    assert scout1._session_id != scout2._session_id


# =============================================================================
# DANGLING NODE DETECTION TESTS (3 tests)
# =============================================================================


def test_given_dangling_entity_when_analyzed_then_identified_as_dangling(
    sample_entity_dangling: ManifestEntry,
) -> None:
    """
    GIVEN an entity with no cross-document links
    WHEN entity is analyzed
    THEN is_dangling flag is True
    """
    scout = ProactiveScout()
    results = scout._analyze_entities([sample_entity_dangling])

    assert len(results) == 1
    assert results[0].is_dangling is True
    assert results[0].document_count == 1


def test_given_cross_document_entity_when_analyzed_then_not_dangling(
    sample_entity_strong: ManifestEntry,
) -> None:
    """
    GIVEN an entity with cross-document links
    WHEN entity is analyzed
    THEN is_dangling flag is False
    """
    scout = ProactiveScout()
    results = scout._analyze_entities([sample_entity_strong])

    assert len(results) == 1
    assert results[0].is_dangling is False
    assert results[0].document_count == 3


def test_given_multiple_entities_when_analyzed_then_correctly_identifies_dangling(
    sample_entity_dangling: ManifestEntry,
    sample_entity_strong: ManifestEntry,
) -> None:
    """
    GIVEN mix of dangling and cross-document entities
    WHEN entities are analyzed
    THEN correctly identifies each type
    """
    scout = ProactiveScout()
    results = scout._analyze_entities([sample_entity_dangling, sample_entity_strong])

    assert len(results) == 2
    assert results[0].is_dangling is True  # dangling
    assert results[1].is_dangling is False  # strong


# =============================================================================
# WEAK NODE DETECTION TESTS (3 tests)
# =============================================================================


def test_given_weak_entity_when_analyzed_then_identified_as_weak(
    sample_entity_weak: ManifestEntry,
) -> None:
    """
    GIVEN an entity with <2 references
    WHEN entity is analyzed
    THEN is_weak flag is True
    """
    scout = ProactiveScout()
    results = scout._analyze_entities([sample_entity_weak])

    assert len(results) == 1
    assert results[0].is_weak is True
    assert results[0].reference_count < MIN_REFERENCES_FOR_WEAK_NODE


def test_given_strong_entity_when_analyzed_then_not_weak(
    sample_entity_strong: ManifestEntry,
) -> None:
    """
    GIVEN an entity with >=2 references
    WHEN entity is analyzed
    THEN is_weak flag is False
    """
    scout = ProactiveScout()
    results = scout._analyze_entities([sample_entity_strong])

    assert len(results) == 1
    assert results[0].is_weak is False
    assert results[0].reference_count >= MIN_REFERENCES_FOR_WEAK_NODE


def test_given_exactly_2_refs_when_analyzed_then_not_weak() -> None:
    """
    GIVEN an entity with exactly 2 references (threshold)
    WHEN entity is analyzed
    THEN is_weak flag is False
    """
    refs = [
        EntityReference(
            document_id="doc1",
            artifact_id="art1",
            chunk_id="chunk1",
            confidence=0.95,
        ),
        EntityReference(
            document_id="doc2",
            artifact_id="art2",
            chunk_id="chunk2",
            confidence=0.92,
        ),
    ]
    entry = ManifestEntry(
        entity_hash="hash_threshold",
        entity_text="Test Entity",
        entity_type="TEST",
        references=refs,
    )

    scout = ProactiveScout()
    results = scout._analyze_entities([entry])

    assert len(results) == 1
    assert results[0].is_weak is False


# =============================================================================
# PRIORITY SCORING TESTS (4 tests)
# =============================================================================


def test_given_dangling_entity_when_priority_calculated_then_gets_boost() -> None:
    """
    GIVEN a dangling entity
    WHEN priority is calculated
    THEN receives dangling boost (+0.3)
    """
    refs = [
        EntityReference(
            document_id="doc1",
            artifact_id="art1",
            chunk_id="chunk1",
            confidence=0.95,
        ),
    ]
    entry = ManifestEntry(
        entity_hash="hash",
        entity_text="Entity",
        entity_type="TYPE",
        references=refs,
    )

    scout = ProactiveScout()
    priority = scout._calculate_priority(entry, is_dangling=True, is_weak=False)

    # Base priority for 1 ref: 0.1, dangling boost: +0.3 = 0.4
    assert priority >= 0.3
    assert priority <= 0.5


def test_given_weak_entity_when_priority_calculated_then_gets_boost() -> None:
    """
    GIVEN a weak entity (not dangling)
    WHEN priority is calculated
    THEN receives weak boost (+0.2)
    """
    refs = [
        EntityReference(
            document_id="doc1",
            artifact_id="art1",
            chunk_id="chunk1",
            confidence=0.95,
        ),
    ]
    entry = ManifestEntry(
        entity_hash="hash",
        entity_text="Entity",
        entity_type="TYPE",
        references=refs,
    )

    scout = ProactiveScout()
    priority = scout._calculate_priority(entry, is_dangling=False, is_weak=True)

    # Base priority for 1 ref: 0.1, weak boost: +0.2 = 0.3
    assert priority >= 0.2
    assert priority <= 0.4


def test_given_strong_entity_when_priority_calculated_then_base_priority() -> None:
    """
    GIVEN a strong entity
    WHEN priority is calculated
    THEN uses only base priority (no boosts)
    """
    refs = [
        EntityReference(
            document_id=f"doc{i}",
            artifact_id=f"art{i}",
            chunk_id=f"chunk{i}",
            confidence=0.95,
        )
        for i in range(10)
    ]
    entry = ManifestEntry(
        entity_hash="hash",
        entity_text="Entity",
        entity_type="TYPE",
        references=refs,
    )

    scout = ProactiveScout()
    priority = scout._calculate_priority(entry, is_dangling=False, is_weak=False)

    # Base priority for 10 refs: min(10/10.0, 0.5) = 0.5
    assert priority == pytest.approx(0.5, abs=0.01)


def test_given_priority_calculation_when_computed_then_capped_at_1() -> None:
    """
    GIVEN priority calculation with very high reference count
    WHEN priority is computed
    THEN capped at 1.0
    """
    refs = [
        EntityReference(
            document_id=f"doc{i}",
            artifact_id=f"art{i}",
            chunk_id=f"chunk{i}",
            confidence=0.95,
        )
        for i in range(50)
    ]
    entry = ManifestEntry(
        entity_hash="hash",
        entity_text="Entity",
        entity_type="TYPE",
        references=refs,
    )

    scout = ProactiveScout()
    priority = scout._calculate_priority(entry, is_dangling=True, is_weak=False)

    # Should be capped at 1.0
    assert priority == 1.0


# =============================================================================
# RATIONALE GENERATION TESTS (3 tests)
# =============================================================================


def test_given_dangling_entity_when_rationale_generated_then_mentions_cross_document() -> (
    None
):
    """
    GIVEN a dangling entity
    WHEN rationale is generated
    THEN mentions cross-document connections
    """
    refs = [
        EntityReference(
            document_id="doc1",
            artifact_id="art1",
            chunk_id="chunk1",
            confidence=0.95,
        ),
    ]
    entry = ManifestEntry(
        entity_hash="hash",
        entity_text="Test Entity",
        entity_type="TYPE",
        references=refs,
    )

    scout = ProactiveScout()
    rationale = scout._generate_rationale(entry, is_dangling=True, is_weak=False)

    assert "Test Entity" in rationale
    assert "cross-document" in rationale.lower()
    assert "one document" in rationale


def test_given_weak_entity_when_rationale_generated_then_mentions_references() -> None:
    """
    GIVEN a weak entity
    WHEN rationale is generated
    THEN mentions reference count
    """
    refs = [
        EntityReference(
            document_id="doc1",
            artifact_id="art1",
            chunk_id="chunk1",
            confidence=0.95,
        ),
    ]
    entry = ManifestEntry(
        entity_hash="hash",
        entity_text="Weak Entity",
        entity_type="TYPE",
        references=refs,
    )

    scout = ProactiveScout()
    rationale = scout._generate_rationale(entry, is_dangling=False, is_weak=True)

    assert "Weak Entity" in rationale
    assert "1 reference" in rationale
    assert "context" in rationale.lower()


def test_given_strong_entity_when_rationale_generated_then_suggests_additional_sources() -> (
    None
):
    """
    GIVEN a strong entity
    WHEN rationale is generated
    THEN suggests additional sources
    """
    refs = [
        EntityReference(
            document_id=f"doc{i}",
            artifact_id=f"art{i}",
            chunk_id=f"chunk{i}",
            confidence=0.95,
        )
        for i in range(5)
    ]
    entry = ManifestEntry(
        entity_hash="hash",
        entity_text="Strong Entity",
        entity_type="TYPE",
        references=refs,
    )

    scout = ProactiveScout()
    rationale = scout._generate_rationale(entry, is_dangling=False, is_weak=False)

    assert "Strong Entity" in rationale
    assert "5 time" in rationale
    assert "additional" in rationale.lower() or "new" in rationale.lower()


# =============================================================================
# INTENT GENERATION TESTS (3 tests)
# =============================================================================


def test_given_gap_results_when_intents_generated_then_creates_artifacts() -> None:
    """
    GIVEN gap analysis results
    WHEN discovery intents are generated
    THEN creates IFDiscoveryIntentArtifact objects
    """
    gaps = [
        GapAnalysisResult(
            entity_hash="hash1",
            entity_text="Entity1",
            entity_type="TYPE1",
            reference_count=1,
            document_count=1,
            is_dangling=True,
            is_weak=False,
            priority_score=0.7,
            rationale="Test rationale",
        ),
    ]

    scout = ProactiveScout()
    intents = scout._generate_intents(gaps)

    assert len(intents) == 1
    assert isinstance(intents[0], IFDiscoveryIntentArtifact)
    assert intents[0].target_entity == "Entity1"
    assert intents[0].confidence == 0.7


def test_given_low_confidence_gaps_when_filtered_then_excluded() -> None:
    """
    GIVEN gaps with low confidence (<0.6)
    WHEN intents are generated
    THEN low-confidence gaps are excluded
    """
    gaps = [
        GapAnalysisResult(
            entity_hash="hash1",
            entity_text="Entity1",
            entity_type="TYPE1",
            reference_count=1,
            document_count=1,
            is_dangling=False,
            is_weak=False,
            priority_score=0.5,  # Below MIN_CONFIDENCE (0.6)
            rationale="Low confidence",
        ),
        GapAnalysisResult(
            entity_hash="hash2",
            entity_text="Entity2",
            entity_type="TYPE2",
            reference_count=1,
            document_count=1,
            is_dangling=True,
            is_weak=False,
            priority_score=0.8,  # Above MIN_CONFIDENCE
            rationale="High confidence",
        ),
    ]

    scout = ProactiveScout()
    intents = scout._generate_intents(gaps)

    assert len(intents) == 1
    assert intents[0].target_entity == "Entity2"


def test_given_many_gaps_when_generated_then_respects_max_limit() -> None:
    """
    GIVEN more gaps than MAX_DISCOVERY_INTENTS
    WHEN intents are generated
    THEN respects the maximum limit
    """
    gaps = [
        GapAnalysisResult(
            entity_hash=f"hash{i}",
            entity_text=f"Entity{i}",
            entity_type="TYPE",
            reference_count=1,
            document_count=1,
            is_dangling=True,
            is_weak=False,
            priority_score=0.7,
            rationale=f"Rationale {i}",
        )
        for i in range(150)  # More than MAX_DISCOVERY_INTENTS (100)
    ]

    scout = ProactiveScout()
    intents = scout._generate_intents(gaps)

    assert len(intents) <= 100  # MAX_DISCOVERY_INTENTS


# =============================================================================
# TOP ENTITIES SELECTION TESTS (2 tests)
# =============================================================================


def test_given_many_entities_when_get_top_then_returns_max_50() -> None:
    """
    GIVEN more than 50 entities
    WHEN get_top_entities is called
    THEN returns exactly 50 (MAX_ENTITIES_TO_ANALYZE)
    """
    entities = [
        ManifestEntry(
            entity_hash=f"hash{i}",
            entity_text=f"Entity{i}",
            entity_type="TYPE",
            references=[
                EntityReference(
                    document_id="doc1",
                    artifact_id="art1",
                    chunk_id="chunk1",
                    confidence=0.95,
                )
            ]
            * (100 - i),  # Descending reference counts
        )
        for i in range(75)
    ]

    manifest = Mock(spec=IFKnowledgeManifest)
    manifest.get_all_entities.return_value = entities

    scout = ProactiveScout()
    top_entities = scout._get_top_entities(manifest)

    assert len(top_entities) == MAX_ENTITIES_TO_ANALYZE  # Should be 50


def test_given_few_entities_when_get_top_then_returns_all() -> None:
    """
    GIVEN fewer than 50 entities
    WHEN get_top_entities is called
    THEN returns all entities
    """
    entities = [
        ManifestEntry(
            entity_hash=f"hash{i}",
            entity_text=f"Entity{i}",
            entity_type="TYPE",
            references=[
                EntityReference(
                    document_id="doc1",
                    artifact_id="art1",
                    chunk_id="chunk1",
                    confidence=0.95,
                )
            ],
        )
        for i in range(10)
    ]

    manifest = Mock(spec=IFKnowledgeManifest)
    manifest.get_all_entities.return_value = entities

    scout = ProactiveScout()
    top_entities = scout._get_top_entities(manifest)

    assert len(top_entities) == 10


# =============================================================================
# MANIFEST ANALYSIS TESTS (3 tests)
# =============================================================================


def test_given_active_manifest_when_analyzed_then_returns_intents() -> None:
    """
    GIVEN an active knowledge manifest
    WHEN analyze_manifest is called
    THEN returns list of discovery intents
    """
    entities = [
        ManifestEntry(
            entity_hash="hash1",
            entity_text="Entity1",
            entity_type="TYPE1",
            references=[
                EntityReference(
                    document_id="doc1",
                    artifact_id="art1",
                    chunk_id="chunk1",
                    confidence=0.95,
                )
            ],
        ),
    ]

    manifest = Mock(spec=IFKnowledgeManifest)
    manifest.is_active = True
    manifest.get_all_entities.return_value = entities

    scout = ProactiveScout()
    intents = scout.analyze_manifest(manifest)

    assert isinstance(intents, list)
    assert all(isinstance(i, IFDiscoveryIntentArtifact) for i in intents)


def test_given_inactive_manifest_when_analyzed_then_returns_empty_list() -> None:
    """
    GIVEN an inactive knowledge manifest
    WHEN analyze_manifest is called
    THEN returns empty list
    """
    manifest = Mock(spec=IFKnowledgeManifest)
    manifest.is_active = False

    scout = ProactiveScout()
    intents = scout.analyze_manifest(manifest)

    assert intents == []


def test_given_manifest_when_helper_function_called_then_creates_scout_and_analyzes() -> (
    None
):
    """
    GIVEN a knowledge manifest
    WHEN run_scout_analysis helper is called
    THEN creates scout and returns analysis results
    """
    entities = [
        ManifestEntry(
            entity_hash="hash1",
            entity_text="Entity1",
            entity_type="TYPE1",
            references=[
                EntityReference(
                    document_id="doc1",
                    artifact_id="art1",
                    chunk_id="chunk1",
                    confidence=0.95,
                )
            ],
        ),
    ]

    manifest = Mock(spec=IFKnowledgeManifest)
    manifest.is_active = True
    manifest.get_all_entities.return_value = entities

    intents = run_scout_analysis(manifest)

    assert isinstance(intents, list)
