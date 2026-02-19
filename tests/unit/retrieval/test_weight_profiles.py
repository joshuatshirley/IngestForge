"""
Tests for Hybrid Weight Profiles.

This module tests intent-aware weight profiles for hybrid retrieval.

Test Strategy
-------------
- Focus on weight profile configuration and retrieval
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test dataclass validation and profile lookup
- Test all predefined intent profiles

Organization
------------
- TestHybridWeightProfile: HybridWeightProfile dataclass
- TestHybridProfiles: HYBRID_PROFILES registry
- TestGetProfile: get_profile function
- TestGetAvailableIntents: get_available_intents function
"""

import pytest

from ingestforge.retrieval.weight_profiles import (
    HybridWeightProfile,
    HYBRID_PROFILES,
    get_profile,
    get_available_intents,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestHybridWeightProfile:
    """Tests for HybridWeightProfile dataclass.

    Rule #4: Focused test class - tests only HybridWeightProfile
    """

    def test_create_weight_profile_valid(self):
        """Test creating HybridWeightProfile with valid weights."""
        profile = HybridWeightProfile(bm25_weight=0.6, semantic_weight=0.4)

        assert profile.bm25_weight == 0.6
        assert profile.semantic_weight == 0.4

    def test_create_weight_profile_edge_cases(self):
        """Test creating profile with edge case values (0.0, 1.0)."""
        profile_zero = HybridWeightProfile(bm25_weight=0.0, semantic_weight=1.0)
        profile_one = HybridWeightProfile(bm25_weight=1.0, semantic_weight=0.0)

        assert profile_zero.bm25_weight == 0.0
        assert profile_zero.semantic_weight == 1.0
        assert profile_one.bm25_weight == 1.0
        assert profile_one.semantic_weight == 0.0

    def test_validate_bm25_weight_out_of_range_high(self):
        """Test validation rejects bm25_weight > 1.0."""
        with pytest.raises(ValueError) as exc_info:
            HybridWeightProfile(bm25_weight=1.5, semantic_weight=0.4)

        assert "bm25_weight must be 0.0-1.0" in str(exc_info.value)

    def test_validate_bm25_weight_out_of_range_low(self):
        """Test validation rejects bm25_weight < 0.0."""
        with pytest.raises(ValueError) as exc_info:
            HybridWeightProfile(bm25_weight=-0.1, semantic_weight=0.4)

        assert "bm25_weight must be 0.0-1.0" in str(exc_info.value)

    def test_validate_semantic_weight_out_of_range_high(self):
        """Test validation rejects semantic_weight > 1.0."""
        with pytest.raises(ValueError) as exc_info:
            HybridWeightProfile(bm25_weight=0.6, semantic_weight=1.2)

        assert "semantic_weight must be 0.0-1.0" in str(exc_info.value)

    def test_validate_semantic_weight_out_of_range_low(self):
        """Test validation rejects semantic_weight < 0.0."""
        with pytest.raises(ValueError) as exc_info:
            HybridWeightProfile(bm25_weight=0.6, semantic_weight=-0.3)

        assert "semantic_weight must be 0.0-1.0" in str(exc_info.value)


class TestHybridProfiles:
    """Tests for HYBRID_PROFILES registry.

    Rule #4: Focused test class - tests HYBRID_PROFILES only
    """

    def test_profiles_contain_all_intents(self):
        """Test HYBRID_PROFILES contains all expected intents."""
        expected_intents = [
            "factual",
            "procedural",
            "conceptual",
            "comparative",
            "exploratory",
            "literary",
            "default",
        ]

        for intent in expected_intents:
            assert intent in HYBRID_PROFILES

    def test_factual_profile_high_bm25(self):
        """Test factual profile favors BM25 (exact matching)."""
        profile = HYBRID_PROFILES["factual"]

        assert profile.bm25_weight > profile.semantic_weight
        assert profile.bm25_weight == 0.65
        assert profile.semantic_weight == 0.35

    def test_procedural_profile_balanced(self):
        """Test procedural profile is balanced toward BM25."""
        profile = HYBRID_PROFILES["procedural"]

        assert profile.bm25_weight >= 0.5
        assert profile.bm25_weight == 0.55
        assert profile.semantic_weight == 0.45

    def test_conceptual_profile_balanced(self):
        """Test conceptual profile is balanced."""
        profile = HYBRID_PROFILES["conceptual"]

        assert profile.bm25_weight == 0.50
        assert profile.semantic_weight == 0.50

    def test_comparative_profile_favors_semantic(self):
        """Test comparative profile favors semantic."""
        profile = HYBRID_PROFILES["comparative"]

        assert profile.semantic_weight > profile.bm25_weight
        assert profile.bm25_weight == 0.45
        assert profile.semantic_weight == 0.55

    def test_exploratory_profile_high_semantic(self):
        """Test exploratory profile strongly favors semantic."""
        profile = HYBRID_PROFILES["exploratory"]

        assert profile.semantic_weight > profile.bm25_weight
        assert profile.bm25_weight == 0.40
        assert profile.semantic_weight == 0.60

    def test_literary_profile_highest_semantic(self):
        """Test literary profile has highest semantic weight."""
        profile = HYBRID_PROFILES["literary"]

        assert profile.semantic_weight > profile.bm25_weight
        assert profile.bm25_weight == 0.35
        assert profile.semantic_weight == 0.65

    def test_default_profile_favors_semantic(self):
        """Test default profile favors semantic."""
        profile = HYBRID_PROFILES["default"]

        assert profile.semantic_weight > profile.bm25_weight
        assert profile.bm25_weight == 0.40
        assert profile.semantic_weight == 0.60

    def test_all_profiles_valid_weights(self):
        """Test all profiles have valid weights in range."""
        for intent, profile in HYBRID_PROFILES.items():
            assert (
                0.0 <= profile.bm25_weight <= 1.0
            ), f"{intent} has invalid bm25_weight"
            assert (
                0.0 <= profile.semantic_weight <= 1.0
            ), f"{intent} has invalid semantic_weight"


class TestGetProfile:
    """Tests for get_profile function.

    Rule #4: Focused test class - tests get_profile only
    """

    def test_get_profile_known_intent(self):
        """Test getting profile for known intent."""
        profile = get_profile("factual")

        assert isinstance(profile, HybridWeightProfile)
        assert profile.bm25_weight == 0.65
        assert profile.semantic_weight == 0.35

    def test_get_profile_unknown_intent_returns_default(self):
        """Test getting profile for unknown intent returns default."""
        profile = get_profile("unknown_intent_type")

        assert isinstance(profile, HybridWeightProfile)
        assert profile == HYBRID_PROFILES["default"]

    def test_get_profile_case_insensitive(self):
        """Test get_profile is case-insensitive."""
        profile_lower = get_profile("factual")
        profile_upper = get_profile("FACTUAL")
        profile_mixed = get_profile("FaCtuAl")

        assert profile_lower == profile_upper
        assert profile_lower == profile_mixed

    def test_get_profile_none_returns_default(self):
        """Test get_profile with None returns default."""
        profile = get_profile(None)

        assert profile == HYBRID_PROFILES["default"]

    def test_get_profile_empty_string_returns_default(self):
        """Test get_profile with empty string returns default."""
        profile = get_profile("")

        assert profile == HYBRID_PROFILES["default"]

    def test_get_profile_all_intents(self):
        """Test get_profile works for all registered intents."""
        for intent in HYBRID_PROFILES.keys():
            profile = get_profile(intent)

            assert isinstance(profile, HybridWeightProfile)
            assert profile == HYBRID_PROFILES[intent]


class TestGetAvailableIntents:
    """Tests for get_available_intents function.

    Rule #4: Focused test class - tests get_available_intents only
    """

    def test_get_available_intents_returns_list(self):
        """Test get_available_intents returns a list."""
        intents = get_available_intents()

        assert isinstance(intents, list)

    def test_get_available_intents_contains_all_intents(self):
        """Test get_available_intents contains all profile types."""
        intents = get_available_intents()

        expected = [
            "factual",
            "procedural",
            "conceptual",
            "comparative",
            "exploratory",
            "literary",
            "default",
        ]

        for intent in expected:
            assert intent in intents

    def test_get_available_intents_count(self):
        """Test get_available_intents returns expected count."""
        intents = get_available_intents()

        # Should have 7 intent types
        assert len(intents) == 7


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - HybridWeightProfile: 6 tests (creation, edge cases, validation x4)
    - HYBRID_PROFILES: 9 tests (all intents, weight verification)
    - get_profile: 6 tests (known, unknown, case, None, empty, all)
    - get_available_intents: 3 tests (returns list, contains all, count)

    Total: 24 tests

Design Decisions:
    1. Focus on weight profile configuration and retrieval
    2. Test dataclass validation thoroughly
    3. Verify all predefined intent profiles
    4. Test profile lookup edge cases
    5. Simple, clear tests that verify profiles work
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - HybridWeightProfile dataclass creation
    - Weight validation (0.0-1.0 range)
    - All intent profiles exist and have correct weights
    - Intent-specific weight distributions (factual high BM25, literary high semantic)
    - Profile retrieval by intent name
    - Case-insensitive intent lookup
    - Unknown intent fallback to default
    - Available intents enumeration

Justification:
    - Weight profiles are critical for hybrid retrieval accuracy
    - Intent-aware weighting improves retrieval quality
    - Validation ensures weights stay in valid range
    - Profile lookup needs edge case handling
    - Simple tests verify weight profile system works correctly
"""
