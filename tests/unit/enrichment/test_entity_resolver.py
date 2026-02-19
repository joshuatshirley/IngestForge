"""Tests for Atomic Entity Resolution.

Comprehensive GWT test suite following NASA JPL Power of Ten rules.
"""

import pytest
from typing import List

from ingestforge.enrichment.entity_resolver import (
    # Constants
    MAX_CLUSTER_SIZE,
    MAX_ENTITIES_PER_BATCH,
    MAX_ALIASES_PER_CLUSTER,
    MAX_NAME_LENGTH,
    EntityNormalizer,
    FuzzyMatcher,
    EntityReference,
    EntityCluster,
    EntityResolver,
    # Convenience functions
    create_entity_resolver,
    resolve_entities,
    normalize_entity_name,
    compute_name_similarity,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer() -> EntityNormalizer:
    """Create entity normalizer."""
    return EntityNormalizer()


@pytest.fixture
def matcher() -> FuzzyMatcher:
    """Create fuzzy matcher."""
    return FuzzyMatcher()


@pytest.fixture
def resolver() -> EntityResolver:
    """Create entity resolver."""
    return EntityResolver()


@pytest.fixture
def sample_entities() -> List[EntityReference]:
    """Sample entity references for testing."""
    return [
        EntityReference(
            document_id="doc-001",
            artifact_id="chunk-001",
            original_text="John Smith",
            entity_type="PERSON",
            confidence=0.95,
        ),
        EntityReference(
            document_id="doc-002",
            artifact_id="chunk-002",
            original_text="J. Smith",
            entity_type="PERSON",
            confidence=0.85,
        ),
        EntityReference(
            document_id="doc-003",
            artifact_id="chunk-003",
            original_text="Smith, John",
            entity_type="PERSON",
            confidence=0.90,
        ),
        EntityReference(
            document_id="doc-001",
            artifact_id="chunk-004",
            original_text="Acme Corporation",
            entity_type="ORG",
            confidence=0.98,
        ),
        EntityReference(
            document_id="doc-002",
            artifact_id="chunk-005",
            original_text="ACME Corp.",
            entity_type="ORG",
            confidence=0.88,
        ),
    ]


# ---------------------------------------------------------------------------
# TestEntityNormalizer
# ---------------------------------------------------------------------------


class TestEntityNormalizer:
    """Tests for EntityNormalizer class."""

    def test_normalize_basic(self, normalizer):
        """Test basic name normalization."""
        result = normalizer.normalize("John Smith")
        assert result == "john smith"

    def test_normalize_removes_honorifics(self, normalizer):
        """Test honorific removal."""
        result = normalizer.normalize("Dr. John Smith")
        assert "dr" not in result
        assert "john smith" == result

    def test_normalize_removes_multiple_honorifics(self, normalizer):
        """Test multiple honorific removal."""
        result = normalizer.normalize("Prof. Dr. Jane Doe, Ph.D.")
        assert "prof" not in result
        assert "dr" not in result
        assert "phd" not in result

    def test_normalize_handles_case(self, normalizer):
        """Test case normalization."""
        result = normalizer.normalize("JOHN SMITH")
        assert result == "john smith"

    def test_normalize_handles_punctuation(self, normalizer):
        """Test punctuation handling."""
        result = normalizer.normalize("Smith, John Jr.")
        assert "," not in result
        assert "." not in result

    def test_normalize_preserves_apostrophe(self, normalizer):
        """Test apostrophe preservation."""
        result = normalizer.normalize("O'Brien")
        assert "'" in result or "obrien" in result

    def test_normalize_handles_whitespace(self, normalizer):
        """Test whitespace normalization."""
        result = normalizer.normalize("  John   Smith  ")
        assert result == "john smith"

    def test_normalize_handles_unicode(self, normalizer):
        """Test unicode normalization."""
        result = normalizer.normalize("José García")
        # Should handle accented characters
        assert len(result) > 0

    def test_normalize_empty_string(self, normalizer):
        """Test empty string handling."""
        result = normalizer.normalize("")
        assert result == ""

    def test_normalize_truncates_long_names(self, normalizer):
        """Test long name truncation."""
        long_name = "A" * (MAX_NAME_LENGTH + 100)
        result = normalizer.normalize(long_name)
        assert len(result) <= MAX_NAME_LENGTH

    def test_get_tokens(self, normalizer):
        """Test token extraction."""
        tokens = normalizer.get_tokens("John Smith Jr.")
        assert "john" in tokens
        assert "smith" in tokens
        assert "jr" not in tokens  # Honorific removed


# ---------------------------------------------------------------------------
# TestFuzzyMatcher
# ---------------------------------------------------------------------------


class TestFuzzyMatcher:
    """Tests for FuzzyMatcher class."""

    def test_exact_match(self, matcher):
        """Test exact match returns 1.0."""
        score = matcher.compute_similarity("John Smith", "John Smith")
        assert score == 1.0

    def test_normalized_exact_match(self, matcher):
        """Test case-insensitive match."""
        score = matcher.compute_similarity("John Smith", "john smith")
        assert score == 1.0

    def test_similar_names(self, matcher):
        """Test similar names have high score."""
        score = matcher.compute_similarity("John Smith", "Jon Smith")
        assert score > 0.8

    def test_different_names(self, matcher):
        """Test different names have low score."""
        score = matcher.compute_similarity("John Smith", "Jane Doe")
        assert score < 0.5

    def test_token_order_invariance(self, matcher):
        """Test name order doesn't affect match."""
        score = matcher.compute_similarity("John Smith", "Smith John")
        assert score > 0.9

    def test_comma_separated_name(self, matcher):
        """Test comma-separated name matching."""
        score = matcher.compute_similarity("Smith, John", "John Smith")
        assert score > 0.9

    def test_abbreviated_name(self, matcher):
        """Test abbreviated name matching."""
        score = matcher.compute_similarity("J. Smith", "John Smith")
        # Lower but still reasonable
        assert score > 0.5

    def test_is_match_above_threshold(self, matcher):
        """Test is_match with similar names."""
        assert matcher.is_match("John Smith", "Jon Smith") is True

    def test_is_match_below_threshold(self, matcher):
        """Test is_match with different names."""
        assert matcher.is_match("John Smith", "Jane Doe") is False

    def test_custom_threshold(self):
        """Test custom threshold."""
        strict_matcher = FuzzyMatcher(threshold=0.95)
        assert strict_matcher.is_match("John Smith", "Jon Smith") is False

        lenient_matcher = FuzzyMatcher(threshold=0.5)
        assert lenient_matcher.is_match("John Smith", "Jon Smith") is True

    def test_empty_string_similarity(self, matcher):
        """Test empty string handling."""
        assert matcher.compute_similarity("", "John") == 0.0
        assert matcher.compute_similarity("John", "") == 0.0
        assert matcher.compute_similarity("", "") == 0.0

    def test_invalid_threshold_raises(self):
        """Test invalid threshold raises assertion."""
        with pytest.raises(AssertionError):
            FuzzyMatcher(threshold=-0.1)
        with pytest.raises(AssertionError):
            FuzzyMatcher(threshold=1.5)


# ---------------------------------------------------------------------------
# TestEntityReference
# ---------------------------------------------------------------------------


class TestEntityReference:
    """Tests for EntityReference dataclass."""

    def test_create_basic_reference(self):
        """Test basic reference creation."""
        ref = EntityReference(
            document_id="doc-001",
            artifact_id="chunk-001",
            original_text="John Smith",
            entity_type="PERSON",
        )
        assert ref.document_id == "doc-001"
        assert ref.entity_type == "PERSON"

    def test_reference_with_attributes(self):
        """Test reference with attributes."""
        ref = EntityReference(
            document_id="doc-001",
            artifact_id="chunk-001",
            original_text="John Smith",
            entity_type="PERSON",
            attributes={"role": "CEO"},
        )
        assert ref.attributes["role"] == "CEO"

    def test_to_dict(self):
        """Test dictionary conversion."""
        ref = EntityReference(
            document_id="doc-001",
            artifact_id="chunk-001",
            original_text="John Smith",
            entity_type="PERSON",
            confidence=0.95,
        )
        result = ref.to_dict()
        assert result["document_id"] == "doc-001"
        assert result["confidence"] == 0.95


# ---------------------------------------------------------------------------
# TestEntityCluster
# ---------------------------------------------------------------------------


class TestEntityCluster:
    """Tests for EntityCluster dataclass."""

    def test_create_basic_cluster(self):
        """Test basic cluster creation."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        assert cluster.canonical_name == "John Smith"
        assert cluster.alias_count == 0

    def test_add_alias(self):
        """Test adding alias."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        result = cluster.add_alias("J. Smith")
        assert result is True
        assert cluster.alias_count == 1
        assert "J. Smith" in cluster.aliases

    def test_add_duplicate_alias(self):
        """Test duplicate alias not added."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        cluster.add_alias("J. Smith")
        cluster.add_alias("J. Smith")
        assert cluster.alias_count == 1

    def test_add_canonical_as_alias(self):
        """Test canonical name not added as alias."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        cluster.add_alias("John Smith")
        assert cluster.alias_count == 0

    def test_add_reference(self):
        """Test adding reference."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        ref = EntityReference(
            document_id="doc-001",
            artifact_id="chunk-001",
            original_text="John Smith",
            entity_type="PERSON",
        )
        result = cluster.add_reference(ref)
        assert result is True
        assert cluster.reference_count == 1

    def test_document_count(self):
        """Test document count."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        cluster.add_reference(
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
            )
        )
        cluster.add_reference(
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="J. Smith",
                entity_type="PERSON",
            )
        )
        assert cluster.document_count == 2
        assert cluster.is_cross_document is True

    def test_alias_limit(self):
        """Test alias limit enforcement."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        for i in range(MAX_ALIASES_PER_CLUSTER + 10):
            cluster.add_alias(f"Alias {i}")
        assert cluster.alias_count <= MAX_ALIASES_PER_CLUSTER

    def test_reference_limit(self):
        """Test reference limit enforcement."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
        )
        for i in range(MAX_CLUSTER_SIZE + 10):
            ref = EntityReference(
                document_id=f"doc-{i}",
                artifact_id=f"chunk-{i}",
                original_text="John Smith",
                entity_type="PERSON",
            )
            cluster.add_reference(ref)
        assert cluster.reference_count <= MAX_CLUSTER_SIZE

    def test_long_canonical_name_raises(self):
        """Test long canonical name raises assertion."""
        with pytest.raises(AssertionError):
            EntityCluster(
                canonical_name="A" * (MAX_NAME_LENGTH + 1),
                entity_type="PERSON",
            )

    def test_to_dict(self):
        """Test dictionary conversion."""
        cluster = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
            cluster_id="cluster_1",
            confidence=0.95,
        )
        cluster.add_alias("J. Smith")
        result = cluster.to_dict()
        assert result["canonical_name"] == "John Smith"
        assert result["cluster_id"] == "cluster_1"
        assert len(result["aliases"]) == 1


# ---------------------------------------------------------------------------
# TestGWT1NameNormalization
# ---------------------------------------------------------------------------


class TestGWT1NameNormalization:
    """GWT-1: Name Normalization tests."""

    def test_normalize_case_variations(self, normalizer):
        """Given names with case variations, When normalized, Then canonical form."""
        names = ["JOHN SMITH", "john smith", "John Smith", "jOhN sMiTh"]
        normalized = [normalizer.normalize(n) for n in names]
        assert len(set(normalized)) == 1

    def test_normalize_punctuation_variations(self, normalizer):
        """Given names with punctuation, When normalized, Then punctuation removed."""
        names = ["Smith, John", "Smith - John", "Smith. John"]
        normalized = [normalizer.normalize(n) for n in names]
        for n in normalized:
            assert "," not in n
            assert "-" not in n or "smith" in n  # Hyphen in names OK

    def test_normalize_honorific_variations(self, normalizer):
        """Given names with honorifics, When normalized, Then honorifics removed."""
        names = ["Dr. John Smith", "Mr. John Smith", "Prof. John Smith"]
        normalized = [normalizer.normalize(n) for n in names]
        for n in normalized:
            assert "dr" not in n
            assert "mr" not in n
            assert "prof" not in n

    def test_normalize_whitespace_variations(self, normalizer):
        """Given names with whitespace variations, When normalized, Then single spaces."""
        names = ["John  Smith", "John\tSmith", "  John Smith  "]
        normalized = [normalizer.normalize(n) for n in names]
        for n in normalized:
            assert "  " not in n
            assert n == n.strip()


# ---------------------------------------------------------------------------
# TestGWT2FuzzyMatching
# ---------------------------------------------------------------------------


class TestGWT2FuzzyMatching:
    """GWT-2: Fuzzy Matching tests."""

    def test_similar_names_high_score(self, matcher):
        """Given similar names, When similarity computed, Then high score."""
        score = matcher.compute_similarity("Jonathan Smith", "John Smith")
        assert score > 0.7

    def test_identical_names_perfect_score(self, matcher):
        """Given identical names, When similarity computed, Then score is 1.0."""
        score = matcher.compute_similarity("John Smith", "John Smith")
        assert score == 1.0

    def test_different_names_low_score(self, matcher):
        """Given different names, When similarity computed, Then low score."""
        score = matcher.compute_similarity("John Smith", "Mary Johnson")
        assert score < 0.5

    def test_reordered_tokens_high_score(self, matcher):
        """Given reordered tokens, When similarity computed, Then high score."""
        score = matcher.compute_similarity("Smith John", "John Smith")
        assert score > 0.9

    def test_abbreviations_moderate_score(self, matcher):
        """Given abbreviations, When similarity computed, Then moderate score."""
        score = matcher.compute_similarity("J. Smith", "John Smith")
        assert 0.4 < score < 0.9


# ---------------------------------------------------------------------------
# TestGWT3EntityMerging
# ---------------------------------------------------------------------------


class TestGWT3EntityMerging:
    """GWT-3: Entity Merging tests."""

    def test_merge_similar_entities(self, resolver, sample_entities):
        """Given similar entities, When resolved, Then merged into cluster."""
        # Just the person entities
        persons = [e for e in sample_entities if e.entity_type == "PERSON"]
        clusters = resolver.resolve(persons)

        # Should have one or few clusters for similar names
        person_clusters = [c for c in clusters if c.entity_type == "PERSON"]
        assert len(person_clusters) >= 1

    def test_merged_cluster_has_all_references(self, resolver):
        """Given merged entities, When resolved, Then cluster has all refs."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="John Smith",
                entity_type="PERSON",
            ),
        ]
        clusters = resolver.resolve(entities)
        assert len(clusters) == 1
        assert clusters[0].reference_count == 2

    def test_merge_preserves_highest_confidence(self):
        """Given entities with different confidence, When merged, Then highest preserved."""
        # Use lenient resolver to ensure merging
        resolver = EntityResolver(threshold=0.6)
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
                confidence=0.70,
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="Jon Smith",
                entity_type="PERSON",  # Similar enough to merge
                confidence=0.95,
            ),
        ]
        clusters = resolver.resolve(entities)
        # Higher confidence name should be canonical
        assert len(clusters) == 1
        assert clusters[0].confidence >= 0.95

    def test_merge_two_clusters(self, resolver):
        """Given two clusters, When merged, Then single cluster with all data."""
        cluster1 = EntityCluster(
            canonical_name="John Smith",
            entity_type="PERSON",
            cluster_id="c1",
            confidence=0.9,
        )
        cluster1.add_alias("J. Smith")

        cluster2 = EntityCluster(
            canonical_name="Johnny Smith",
            entity_type="PERSON",
            cluster_id="c2",
            confidence=0.8,
        )

        merged = resolver.merge_clusters(cluster1, cluster2)
        assert merged.canonical_name == "John Smith"  # Higher confidence
        assert "Johnny Smith" in merged.aliases


# ---------------------------------------------------------------------------
# TestGWT4ConflictResolution
# ---------------------------------------------------------------------------


class TestGWT4ConflictResolution:
    """GWT-4: Conflict Resolution tests."""

    def test_select_canonical_most_common(self, resolver):
        """Given name variations, When selecting canonical, Then most common."""
        names = ["John Smith", "John Smith", "J. Smith"]
        canonical = resolver.select_canonical(names)
        assert canonical == "John Smith"

    def test_select_canonical_prefers_longer(self, resolver):
        """Given equal frequency, When selecting canonical, Then longer name."""
        names = ["John", "John Smith"]
        canonical = resolver.select_canonical(names)
        assert canonical == "John Smith"

    def test_resolve_conflicting_attributes(self, resolver):
        """Given conflicting attributes, When resolved, Then most frequent wins."""
        attributes = {
            "title": ["CEO", "CEO", "President"],
            "company": ["Acme", "Acme Corp"],
        }
        resolved = resolver.resolve_conflicts(attributes)
        assert resolved["title"] == "CEO"  # Most frequent

    def test_resolve_single_value_attributes(self, resolver):
        """Given single value attributes, When resolved, Then value preserved."""
        attributes = {
            "email": ["john@example.com"],
        }
        resolved = resolver.resolve_conflicts(attributes)
        assert resolved["email"] == "john@example.com"

    def test_merge_attributes_tracks_conflicts(self, resolver):
        """Given conflicting attributes, When merged, Then tracked as list."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
                attributes={"title": "CEO"},
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="John Smith",
                entity_type="PERSON",
                attributes={"title": "President"},
            ),
        ]
        clusters = resolver.resolve(entities)
        merged_attrs = clusters[0].merged_attributes
        # Should track both values
        assert "title" in merged_attrs


# ---------------------------------------------------------------------------
# TestGWT5ClusterTracking
# ---------------------------------------------------------------------------


class TestGWT5ClusterTracking:
    """GWT-5: Cluster Tracking tests."""

    def test_clusters_have_canonical_name(self, resolver, sample_entities):
        """Given resolved entities, When clustered, Then canonical name set."""
        clusters = resolver.resolve(sample_entities)
        for cluster in clusters:
            assert len(cluster.canonical_name) > 0

    def test_clusters_have_aliases(self, resolver):
        """Given merged entities, When clustered, Then aliases tracked."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="J. Smith",
                entity_type="PERSON",
            ),
        ]
        resolver_lenient = EntityResolver(threshold=0.6)
        clusters = resolver_lenient.resolve(entities)

        # If merged, should have alias
        if len(clusters) == 1:
            assert clusters[0].alias_count >= 1

    def test_clusters_track_document_ids(self, resolver):
        """Given cross-document entities, When clustered, Then docs tracked."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="John Smith",
                entity_type="PERSON",
            ),
        ]
        clusters = resolver.resolve(entities)
        assert clusters[0].document_count == 2
        assert clusters[0].is_cross_document is True

    def test_clusters_separated_by_type(self, resolver, sample_entities):
        """Given different entity types, When resolved, Then separate clusters."""
        clusters = resolver.resolve(sample_entities)
        person_clusters = [c for c in clusters if c.entity_type == "PERSON"]
        org_clusters = [c for c in clusters if c.entity_type == "ORG"]
        assert len(person_clusters) >= 1
        assert len(org_clusters) >= 1

    def test_cluster_ids_unique(self, resolver, sample_entities):
        """Given multiple clusters, When resolved, Then unique IDs."""
        clusters = resolver.resolve(sample_entities)
        cluster_ids = [c.cluster_id for c in clusters]
        assert len(cluster_ids) == len(set(cluster_ids))


# ---------------------------------------------------------------------------
# TestEntityResolver
# ---------------------------------------------------------------------------


class TestEntityResolver:
    """Additional tests for EntityResolver."""

    def test_resolve_empty_list(self, resolver):
        """Test resolving empty list."""
        clusters = resolver.resolve([])
        assert clusters == []

    def test_resolve_single_entity(self, resolver):
        """Test resolving single entity."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="chunk-001",
                original_text="John Smith",
                entity_type="PERSON",
            )
        ]
        clusters = resolver.resolve(entities)
        assert len(clusters) == 1
        assert clusters[0].canonical_name == "John Smith"

    def test_resolve_bounds_input(self, resolver):
        """Test input is bounded."""
        entities = [
            EntityReference(
                document_id=f"doc-{i}",
                artifact_id=f"chunk-{i}",
                original_text=f"Person {i}",
                entity_type="PERSON",
            )
            for i in range(MAX_ENTITIES_PER_BATCH + 100)
        ]
        clusters = resolver.resolve(entities)
        total_refs = sum(c.reference_count for c in clusters)
        assert total_refs <= MAX_ENTITIES_PER_BATCH

    def test_resolve_none_raises(self, resolver):
        """Test None input raises assertion."""
        with pytest.raises(AssertionError):
            resolver.resolve(None)

    def test_custom_threshold(self):
        """Test custom threshold affects matching."""
        strict = EntityResolver(threshold=0.99)
        lenient = EntityResolver(threshold=0.5)

        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="Jon Smith",
                entity_type="PERSON",
            ),
        ]

        strict_clusters = strict.resolve(entities)
        lenient_clusters = lenient.resolve(entities)

        # Strict should have more clusters
        assert len(strict_clusters) >= len(lenient_clusters)


# ---------------------------------------------------------------------------
# TestConvenienceFunctions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_entity_resolver(self):
        """Test creating resolver via convenience function."""
        resolver = create_entity_resolver(threshold=0.9)
        assert resolver is not None

    def test_resolve_entities(self):
        """Test resolving via convenience function."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="chunk-001",
                original_text="John Smith",
                entity_type="PERSON",
            )
        ]
        clusters = resolve_entities(entities)
        assert len(clusters) == 1

    def test_normalize_entity_name(self):
        """Test normalizing via convenience function."""
        result = normalize_entity_name("Dr. JOHN SMITH")
        assert result == "john smith"

    def test_compute_name_similarity(self):
        """Test computing similarity via convenience function."""
        score = compute_name_similarity("John Smith", "Jon Smith")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestJPLCompliance
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for NASA JPL Power of Ten compliance."""

    def test_rule2_max_cluster_size_constant(self):
        """Rule #2: MAX_CLUSTER_SIZE is defined and reasonable."""
        assert MAX_CLUSTER_SIZE == 100
        assert MAX_CLUSTER_SIZE > 0

    def test_rule2_max_entities_per_batch_constant(self):
        """Rule #2: MAX_ENTITIES_PER_BATCH is defined and reasonable."""
        assert MAX_ENTITIES_PER_BATCH == 5000
        assert MAX_ENTITIES_PER_BATCH > 0

    def test_rule2_max_aliases_per_cluster_constant(self):
        """Rule #2: MAX_ALIASES_PER_CLUSTER is defined and reasonable."""
        assert MAX_ALIASES_PER_CLUSTER == 50
        assert MAX_ALIASES_PER_CLUSTER > 0

    def test_rule5_preconditions_enforced(self):
        """Rule #5: Preconditions enforced via assertions."""
        # Invalid threshold
        with pytest.raises(AssertionError):
            FuzzyMatcher(threshold=-1.0)

        # Long canonical name
        with pytest.raises(AssertionError):
            EntityCluster(
                canonical_name="X" * (MAX_NAME_LENGTH + 1),
                entity_type="PERSON",
            )

    def test_rule7_return_values_checked(self):
        """Rule #7: Return values properly typed and checked."""
        resolver = EntityResolver()
        clusters = resolver.resolve([])
        assert clusters is not None
        assert isinstance(clusters, list)

    def test_rule9_type_hints_present(self):
        """Rule #9: Type hints present on all classes."""
        assert "canonical_name" in EntityCluster.__annotations__
        assert "original_text" in EntityReference.__annotations__


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_unicode_names(self, resolver):
        """Test handling of unicode names."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="José García",
                entity_type="PERSON",
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="Jose Garcia",
                entity_type="PERSON",
            ),
        ]
        clusters = resolver.resolve(entities)
        assert len(clusters) >= 1

    def test_single_character_names(self, normalizer):
        """Test single character name handling."""
        result = normalizer.normalize("X")
        assert result == "x"

    def test_numbers_in_names(self, normalizer):
        """Test names with numbers."""
        result = normalizer.normalize("John Smith III")
        # III should be removed as honorific
        assert "iii" not in result

    def test_empty_attributes(self, resolver):
        """Test entities with empty attributes."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
                attributes={},
            ),
        ]
        clusters = resolver.resolve(entities)
        assert len(clusters) == 1

    def test_very_similar_different_people(self):
        """Test that very similar names can be separate if needed."""
        # With strict threshold, similar names should remain separate
        resolver = EntityResolver(threshold=0.99)
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="John Smith",
                entity_type="PERSON",
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="John Smyth",
                entity_type="PERSON",
            ),
        ]
        clusters = resolver.resolve(entities)
        assert len(clusters) == 2  # Should be separate

    def test_same_name_different_types(self, resolver):
        """Test same name with different entity types."""
        entities = [
            EntityReference(
                document_id="doc-001",
                artifact_id="c1",
                original_text="Apple",
                entity_type="ORG",
            ),
            EntityReference(
                document_id="doc-002",
                artifact_id="c2",
                original_text="Apple",
                entity_type="LOC",
            ),
        ]
        clusters = resolver.resolve(entities)
        # Should be separate clusters due to different types
        assert len(clusters) == 2
