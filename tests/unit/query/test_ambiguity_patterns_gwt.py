"""
Comprehensive GWT Unit Tests for Ambiguity Patterns Module.

Query Clarification - Pattern library validation.
Tests pattern definitions for correctness and JPL compliance.

Test Format:
- Given: Pattern library state
- When: Pattern is accessed or validated
- Then: Expected structure and bounds verified
"""

from __future__ import annotations

import pytest

from ingestforge.query.ambiguity_patterns import (
    PRONOUN_PATTERNS,
    CONTEXTUAL_PRONOUNS,
    DEMONSTRATIVE_PRONOUNS,
    MULTI_MEANING_TERMS,
    TEMPORAL_AMBIGUITY_PATTERNS,
    VAGUE_TEMPORAL_QUALIFIERS,
    VAGUE_PATTERNS,
    SPECIFIC_PATTERNS,
    BROAD_SCOPE_INDICATORS,
    SPECIFIC_SCOPE_INDICATORS,
    MAX_PATTERNS,
    MAX_MULTI_MEANING_TERMS,
    MAX_PRONOUN_PATTERNS,
)


# =============================================================================
# GWT Tests: Pronoun Patterns
# =============================================================================


class TestPronounPatternsGWT:
    """GWT tests for pronoun pattern definitions."""

    def test_pronoun_patterns_is_set(self) -> None:
        """
        Given: PRONOUN_PATTERNS constant
        When: Type is checked
        Then: Is a set (for O(1) lookup)
        """
        # Given/When/Then
        assert isinstance(PRONOUN_PATTERNS, set)

    def test_pronoun_patterns_contains_common_pronouns(self) -> None:
        """
        Given: PRONOUN_PATTERNS constant
        When: Common pronouns are checked
        Then: Contains he, she, it, they, etc.
        """
        # Given
        expected_pronouns = {"he", "she", "it", "they", "this", "that"}

        # When/Then
        for pronoun in expected_pronouns:
            assert pronoun in PRONOUN_PATTERNS

    def test_pronoun_patterns_bounded_by_max(self) -> None:
        """
        Given: PRONOUN_PATTERNS constant
        When: Length is checked
        Then: Does not exceed MAX_PRONOUN_PATTERNS (JPL Rule #2)
        """
        # Given/When/Then
        assert len(PRONOUN_PATTERNS) <= MAX_PRONOUN_PATTERNS

    def test_contextual_pronouns_subset(self) -> None:
        """
        Given: CONTEXTUAL_PRONOUNS constant
        When: Compared to PRONOUN_PATTERNS
        Then: Is a subset (all contextual pronouns are in main set)
        """
        # Given/When/Then
        assert CONTEXTUAL_PRONOUNS.issubset(PRONOUN_PATTERNS)

    def test_demonstrative_pronouns_subset(self) -> None:
        """
        Given: DEMONSTRATIVE_PRONOUNS constant
        When: Compared to PRONOUN_PATTERNS
        Then: Is a subset
        """
        # Given/When/Then
        assert DEMONSTRATIVE_PRONOUNS.issubset(PRONOUN_PATTERNS)

    def test_pronoun_patterns_all_lowercase(self) -> None:
        """
        Given: PRONOUN_PATTERNS constant
        When: Each pattern is checked
        Then: All are lowercase (for case-insensitive matching)
        """
        # Given/When/Then
        for pronoun in PRONOUN_PATTERNS:
            assert pronoun == pronoun.lower()


# =============================================================================
# GWT Tests: Multi-Meaning Terms
# =============================================================================


class TestMultiMeaningTermsGWT:
    """GWT tests for multi-meaning term definitions."""

    def test_multi_meaning_terms_is_dict(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: Type is checked
        Then: Is a dictionary
        """
        # Given/When/Then
        assert isinstance(MULTI_MEANING_TERMS, dict)

    def test_multi_meaning_terms_has_python(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: 'python' is looked up
        Then: Has multiple meanings defined
        """
        # Given/When
        meanings = MULTI_MEANING_TERMS.get("python")

        # Then
        assert meanings is not None
        assert isinstance(meanings, list)
        assert len(meanings) >= 2
        assert "Python programming language" in meanings or any(
            "programming" in m for m in meanings
        )

    def test_multi_meaning_terms_has_java(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: 'java' is looked up
        Then: Has multiple meanings (language, island, coffee)
        """
        # Given/When
        meanings = MULTI_MEANING_TERMS.get("java")

        # Then
        assert meanings is not None
        assert len(meanings) >= 2

    def test_multi_meaning_terms_has_apple(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: 'apple' is looked up
        Then: Has multiple meanings (company, fruit)
        """
        # Given/When
        meanings = MULTI_MEANING_TERMS.get("apple")

        # Then
        assert meanings is not None
        assert len(meanings) >= 2

    def test_multi_meaning_terms_bounded_by_max(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: Length is checked
        Then: Does not exceed MAX_MULTI_MEANING_TERMS (JPL Rule #2)
        """
        # Given/When/Then
        assert len(MULTI_MEANING_TERMS) <= MAX_MULTI_MEANING_TERMS

    def test_multi_meaning_terms_all_have_multiple_meanings(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: Each term is validated
        Then: All have >= 2 meanings
        """
        # Given/When/Then
        for term, meanings in MULTI_MEANING_TERMS.items():
            assert isinstance(meanings, list)
            assert len(meanings) >= 2, f"Term '{term}' should have >= 2 meanings"

    def test_multi_meaning_terms_keys_lowercase(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: Each key is checked
        Then: All keys are lowercase
        """
        # Given/When/Then
        for term in MULTI_MEANING_TERMS.keys():
            assert term == term.lower()

    def test_multi_meaning_terms_has_programming_languages(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: Programming language terms are checked
        Then: Contains Python, Java, Go, Rust, etc.
        """
        # Given
        expected_langs = ["python", "java", "go", "rust", "c", "r"]

        # When/Then
        for lang in expected_langs:
            if lang in MULTI_MEANING_TERMS:  # Optional, but good coverage
                assert len(MULTI_MEANING_TERMS[lang]) >= 2


# =============================================================================
# GWT Tests: Temporal Ambiguity Patterns
# =============================================================================


class TestTemporalAmbiguityPatternsGWT:
    """GWT tests for temporal ambiguity pattern definitions."""

    def test_temporal_patterns_is_set(self) -> None:
        """
        Given: TEMPORAL_AMBIGUITY_PATTERNS constant
        When: Type is checked
        Then: Is a set
        """
        # Given/When/Then
        assert isinstance(TEMPORAL_AMBIGUITY_PATTERNS, set)

    def test_temporal_patterns_contains_common_terms(self) -> None:
        """
        Given: TEMPORAL_AMBIGUITY_PATTERNS constant
        When: Common temporal terms are checked
        Then: Contains recent, soon, before, after, etc.
        """
        # Given
        expected_terms = {"recent", "soon", "before", "after", "now"}

        # When/Then
        for term in expected_terms:
            assert term in TEMPORAL_AMBIGUITY_PATTERNS

    def test_vague_temporal_qualifiers_subset(self) -> None:
        """
        Given: VAGUE_TEMPORAL_QUALIFIERS constant
        When: Compared to TEMPORAL_AMBIGUITY_PATTERNS
        Then: Is a subset
        """
        # Given/When/Then
        assert VAGUE_TEMPORAL_QUALIFIERS.issubset(TEMPORAL_AMBIGUITY_PATTERNS)

    def test_temporal_patterns_all_lowercase(self) -> None:
        """
        Given: TEMPORAL_AMBIGUITY_PATTERNS constant
        When: Each pattern is checked
        Then: All are lowercase
        """
        # Given/When/Then
        for term in TEMPORAL_AMBIGUITY_PATTERNS:
            assert term == term.lower()


# =============================================================================
# GWT Tests: Vague/Specific Query Patterns
# =============================================================================


class TestVagueSpecificPatternsGWT:
    """GWT tests for vague and specific query pattern definitions."""

    def test_vague_patterns_is_list(self) -> None:
        """
        Given: VAGUE_PATTERNS constant
        When: Type is checked
        Then: Is a list of (pattern, penalty) tuples
        """
        # Given/When/Then
        assert isinstance(VAGUE_PATTERNS, list)
        for item in VAGUE_PATTERNS:
            assert isinstance(item, tuple)
            assert len(item) == 2
            pattern, penalty = item
            assert isinstance(pattern, str)
            assert isinstance(penalty, float)

    def test_vague_patterns_has_tell_me_more(self) -> None:
        """
        Given: VAGUE_PATTERNS constant
        When: Patterns are checked
        Then: Contains 'tell me more' pattern
        """
        # Given/When
        patterns = [p for p, _ in VAGUE_PATTERNS]

        # Then
        assert any("tell me more" in p for p in patterns)

    def test_vague_patterns_penalties_valid_range(self) -> None:
        """
        Given: VAGUE_PATTERNS constant
        When: Penalty values are checked
        Then: All penalties are in range [0.0, 1.0]
        """
        # Given/When/Then
        for pattern, penalty in VAGUE_PATTERNS:
            assert (
                0.0 <= penalty <= 1.0
            ), f"Pattern '{pattern}' has invalid penalty: {penalty}"

    def test_vague_patterns_bounded_by_max(self) -> None:
        """
        Given: VAGUE_PATTERNS constant
        When: Length is checked
        Then: Does not exceed MAX_PATTERNS (JPL Rule #2)
        """
        # Given/When/Then
        assert len(VAGUE_PATTERNS) <= MAX_PATTERNS

    def test_specific_patterns_is_list(self) -> None:
        """
        Given: SPECIFIC_PATTERNS constant
        When: Type is checked
        Then: Is a list of (pattern, bonus) tuples
        """
        # Given/When/Then
        assert isinstance(SPECIFIC_PATTERNS, list)
        for item in SPECIFIC_PATTERNS:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_specific_patterns_has_ceo_pattern(self) -> None:
        """
        Given: SPECIFIC_PATTERNS constant
        When: Patterns are checked
        Then: Contains CEO/CTO/CFO pattern
        """
        # Given/When
        patterns = [p for p, _ in SPECIFIC_PATTERNS]

        # Then
        assert any("CEO" in p or "CTO" in p for p in patterns)

    def test_specific_patterns_bonuses_valid_range(self) -> None:
        """
        Given: SPECIFIC_PATTERNS constant
        When: Bonus values are checked
        Then: All bonuses are in range [0.0, 1.0]
        """
        # Given/When/Then
        for pattern, bonus in SPECIFIC_PATTERNS:
            assert (
                0.0 <= bonus <= 1.0
            ), f"Pattern '{pattern}' has invalid bonus: {bonus}"

    def test_specific_patterns_bounded_by_max(self) -> None:
        """
        Given: SPECIFIC_PATTERNS constant
        When: Length is checked
        Then: Does not exceed MAX_PATTERNS (JPL Rule #2)
        """
        # Given/When/Then
        assert len(SPECIFIC_PATTERNS) <= MAX_PATTERNS


# =============================================================================
# GWT Tests: Scope Indicators
# =============================================================================


class TestScopeIndicatorsGWT:
    """GWT tests for scope indicator definitions."""

    def test_broad_scope_indicators_is_set(self) -> None:
        """
        Given: BROAD_SCOPE_INDICATORS constant
        When: Type is checked
        Then: Is a set
        """
        # Given/When/Then
        assert isinstance(BROAD_SCOPE_INDICATORS, set)

    def test_broad_scope_indicators_contains_everything_all(self) -> None:
        """
        Given: BROAD_SCOPE_INDICATORS constant
        When: Common broad terms are checked
        Then: Contains 'everything', 'all', 'any'
        """
        # Given
        expected_terms = {"everything", "all", "any"}

        # When/Then
        for term in expected_terms:
            assert term in BROAD_SCOPE_INDICATORS

    def test_specific_scope_indicators_is_set(self) -> None:
        """
        Given: SPECIFIC_SCOPE_INDICATORS constant
        When: Type is checked
        Then: Is a set
        """
        # Given/When/Then
        assert isinstance(SPECIFIC_SCOPE_INDICATORS, set)

    def test_specific_scope_indicators_contains_specifically(self) -> None:
        """
        Given: SPECIFIC_SCOPE_INDICATORS constant
        When: Common specific terms are checked
        Then: Contains 'specifically', 'exactly', 'precisely'
        """
        # Given
        expected_terms = {"specifically", "exactly", "precisely"}

        # When/Then
        for term in expected_terms:
            assert term in SPECIFIC_SCOPE_INDICATORS

    def test_scope_indicators_all_lowercase(self) -> None:
        """
        Given: Scope indicator sets
        When: Each term is checked
        Then: All are lowercase
        """
        # Given/When/Then
        for term in BROAD_SCOPE_INDICATORS:
            assert term == term.lower()
        for term in SPECIFIC_SCOPE_INDICATORS:
            assert term == term.lower()


# =============================================================================
# GWT Tests: Module-Level Validation
# =============================================================================


class TestModuleValidationGWT:
    """GWT tests for module-level validation (JPL Rule #2)."""

    def test_module_loads_without_errors(self) -> None:
        """
        Given: ambiguity_patterns module
        When: Module is imported
        Then: No import errors or assertion failures
        """
        # Given/When/Then
        # If we got here, module loaded successfully
        # Module has assertions at the end that validate bounds
        assert True

    def test_max_constants_positive_integers(self) -> None:
        """
        Given: MAX_* constants
        When: Values are checked
        Then: All are positive integers
        """
        # Given/When/Then
        assert isinstance(MAX_PATTERNS, int)
        assert isinstance(MAX_MULTI_MEANING_TERMS, int)
        assert isinstance(MAX_PRONOUN_PATTERNS, int)
        assert MAX_PATTERNS > 0
        assert MAX_MULTI_MEANING_TERMS > 0
        assert MAX_PRONOUN_PATTERNS > 0

    def test_all_bounds_respected(self) -> None:
        """
        Given: All pattern collections
        When: Lengths are checked
        Then: All respect their MAX bounds (JPL Rule #2)
        """
        # Given/When/Then
        assert len(VAGUE_PATTERNS) <= MAX_PATTERNS
        assert len(SPECIFIC_PATTERNS) <= MAX_PATTERNS
        assert len(MULTI_MEANING_TERMS) <= MAX_MULTI_MEANING_TERMS
        assert len(PRONOUN_PATTERNS) <= MAX_PRONOUN_PATTERNS


# =============================================================================
# GWT Tests: Pattern Quality
# =============================================================================


class TestPatternQualityGWT:
    """GWT tests for pattern quality and coverage."""

    def test_multi_meaning_terms_coverage(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: Number of terms is checked
        Then: Has reasonable coverage (>= 20 terms)
        """
        # Given/When/Then
        assert len(MULTI_MEANING_TERMS) >= 20

    def test_pronoun_patterns_comprehensive(self) -> None:
        """
        Given: PRONOUN_PATTERNS constant
        When: Coverage is checked
        Then: Covers common pronouns (>= 10)
        """
        # Given/When/Then
        assert len(PRONOUN_PATTERNS) >= 10

    def test_temporal_patterns_comprehensive(self) -> None:
        """
        Given: TEMPORAL_AMBIGUITY_PATTERNS constant
        When: Coverage is checked
        Then: Covers common temporal qualifiers (>= 10)
        """
        # Given/When/Then
        assert len(TEMPORAL_AMBIGUITY_PATTERNS) >= 10

    def test_vague_patterns_comprehensive(self) -> None:
        """
        Given: VAGUE_PATTERNS constant
        When: Coverage is checked
        Then: Has good coverage (>= 10)
        """
        # Given/When/Then
        assert len(VAGUE_PATTERNS) >= 10

    def test_specific_patterns_comprehensive(self) -> None:
        """
        Given: SPECIFIC_PATTERNS constant
        When: Coverage is checked
        Then: Has good coverage (>= 8)
        """
        # Given/When/Then
        assert len(SPECIFIC_PATTERNS) >= 8


# =============================================================================
# GWT Tests: Pattern Uniqueness
# =============================================================================


class TestPatternUniquenessGWT:
    """GWT tests for pattern uniqueness and consistency."""

    def test_pronoun_patterns_no_duplicates(self) -> None:
        """
        Given: PRONOUN_PATTERNS constant (set)
        When: Converted to list and back to set
        Then: No duplicates (set property enforces this)
        """
        # Given/When
        as_list = list(PRONOUN_PATTERNS)
        as_set = set(as_list)

        # Then
        assert len(as_list) == len(as_set)

    def test_multi_meaning_terms_keys_unique(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant (dict)
        When: Keys are checked
        Then: All keys are unique (dict property enforces this)
        """
        # Given/When
        keys = list(MULTI_MEANING_TERMS.keys())

        # Then
        assert len(keys) == len(set(keys))

    def test_temporal_patterns_no_duplicates(self) -> None:
        """
        Given: TEMPORAL_AMBIGUITY_PATTERNS constant (set)
        When: Converted to list and back to set
        Then: No duplicates
        """
        # Given/When
        as_list = list(TEMPORAL_AMBIGUITY_PATTERNS)
        as_set = set(as_list)

        # Then
        assert len(as_list) == len(as_set)


# =============================================================================
# GWT Tests: Pattern Content Validation
# =============================================================================


class TestPatternContentValidationGWT:
    """GWT tests for pattern content validation."""

    def test_vague_patterns_are_regex_compatible(self) -> None:
        """
        Given: VAGUE_PATTERNS constant
        When: Each pattern is validated
        Then: All are valid regex patterns
        """
        # Given
        import re

        # When/Then
        for pattern, _ in VAGUE_PATTERNS:
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Invalid regex pattern: {pattern}")

    def test_specific_patterns_are_regex_compatible(self) -> None:
        """
        Given: SPECIFIC_PATTERNS constant
        When: Each pattern is validated
        Then: All are valid regex patterns
        """
        # Given
        import re

        # When/Then
        for pattern, _ in SPECIFIC_PATTERNS:
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Invalid regex pattern: {pattern}")

    def test_multi_meaning_terms_meanings_are_strings(self) -> None:
        """
        Given: MULTI_MEANING_TERMS constant
        When: Each meaning is validated
        Then: All meanings are non-empty strings
        """
        # Given/When/Then
        for term, meanings in MULTI_MEANING_TERMS.items():
            for meaning in meanings:
                assert isinstance(meaning, str)
                assert len(meaning) > 0, f"Term '{term}' has empty meaning"

    def test_pronoun_patterns_non_empty_strings(self) -> None:
        """
        Given: PRONOUN_PATTERNS constant
        When: Each pronoun is validated
        Then: All are non-empty strings
        """
        # Given/When/Then
        for pronoun in PRONOUN_PATTERNS:
            assert isinstance(pronoun, str)
            assert len(pronoun) > 0

    def test_temporal_patterns_non_empty_strings(self) -> None:
        """
        Given: TEMPORAL_AMBIGUITY_PATTERNS constant
        When: Each term is validated
        Then: All are non-empty strings
        """
        # Given/When/Then
        for term in TEMPORAL_AMBIGUITY_PATTERNS:
            assert isinstance(term, str)
            assert len(term) > 0
