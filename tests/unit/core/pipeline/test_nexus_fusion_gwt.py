"""
GWT Unit Tests for Nexus Fusion (RRF) - Task 129.
NASA JPL Compliance: Rule #4, Rule #9.
"""

import pytest
from ingestforge.core.pipeline.nexus_fusion import NexusResultFusion
from ingestforge.core.models.search import SearchResult


@pytest.fixture
def fusion_engine():
    return NexusResultFusion()


# =============================================================================
# GIVEN: Two sources with overlapping results
# =============================================================================


def test_fusion_given_overlapping_results_when_merged_then_high_rank_boosted(
    fusion_engine,
):
    # Given
    # Result A is rank 0 in Local, rank 1 in Remote
    res_a_local = SearchResult(
        content="A",
        score=0.9,
        confidence=0.9,
        artifact_id="art1",
        nexus_id="local",
        document_id="d1",
    )
    res_a_remote = SearchResult(
        content="A",
        score=0.8,
        confidence=0.9,
        artifact_id="art1",
        nexus_id="local",
        document_id="d1",
    )  # Same identity

    # Result B is rank 0 in Remote only
    res_b_remote = SearchResult(
        content="B",
        score=0.95,
        confidence=0.9,
        artifact_id="art2",
        nexus_id="remote-1",
        document_id="d2",
    )

    data = {"local": [res_a_local], "remote-1": [res_b_remote, res_a_remote]}

    # When
    merged = fusion_engine.merge(data)

    # Then
    # Result A should be #1 because it appeared in both sources
    assert merged[0].content == "A"
    assert merged[1].content == "B"
    # Unified RRF score should be applied
    assert merged[0].score > merged[1].score


def test_fusion_given_single_source_when_merged_then_preserves_order(fusion_engine):
    # Given
    r1 = SearchResult(
        content="1",
        score=0.9,
        confidence=0.9,
        artifact_id="1",
        nexus_id="loc",
        document_id="d",
    )
    r2 = SearchResult(
        content="2",
        score=0.8,
        confidence=0.9,
        artifact_id="2",
        nexus_id="loc",
        document_id="d",
    )

    data = {"local": [r1, r2]}

    # When
    merged = fusion_engine.merge(data)

    # Then
    assert len(merged) == 2
    assert merged[0].content == "1"
    assert merged[1].content == "2"


def test_fusion_given_empty_input_when_merged_then_returns_empty(fusion_engine):
    # When
    merged = fusion_engine.merge({})

    # Then
    assert merged == []


def test_fusion_given_large_input_when_merged_then_enforces_bounds(fusion_engine):
    # Given
    from ingestforge.core.models.search import MAX_TOP_K

    many_results = [
        SearchResult(
            content=str(i),
            score=0.1,
            confidence=0.1,
            artifact_id=str(i),
            nexus_id="l",
            document_id="d",
        )
        for i in range(MAX_TOP_K + 50)
    ]

    # When
    merged = fusion_engine.merge({"local": many_results})

    # Then
    assert len(merged) == MAX_TOP_K
