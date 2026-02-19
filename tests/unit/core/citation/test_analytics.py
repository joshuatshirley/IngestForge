"""Tests for analytics module (CITE-001.2).

Tests the graph analytics engine:
- Node metrics calculation
- HITS hub/authority scores
- PageRank scores
- Cycle detection
"""


from ingestforge.core.citation.link_extractor import LinkMap, InternalReference
from ingestforge.core.citation.analytics import (
    NodeRole,
    NodeMetrics,
    GraphAnalytics,
)


class TestNodeRole:
    """Test NodeRole enum."""

    def test_all_roles_defined(self) -> None:
        """All node roles should be defined."""
        assert NodeRole.HUB
        assert NodeRole.AUTHORITY
        assert NodeRole.CONNECTOR
        assert NodeRole.ISOLATED


class TestNodeMetrics:
    """Test NodeMetrics dataclass."""

    def test_total_degree(self) -> None:
        """Should calculate total degree."""
        metrics = NodeMetrics(
            chunk_id="chunk1",
            in_degree=3,
            out_degree=5,
        )

        assert metrics.total_degree == 8

    def test_determine_role_hub(self) -> None:
        """Should identify hub nodes."""
        metrics = NodeMetrics(
            chunk_id="chunk1",
            hub_score=0.8,
            authority_score=0.2,
        )

        role = metrics.determine_role()
        assert role == NodeRole.HUB

    def test_determine_role_authority(self) -> None:
        """Should identify authority nodes."""
        metrics = NodeMetrics(
            chunk_id="chunk1",
            hub_score=0.2,
            authority_score=0.8,
        )

        role = metrics.determine_role()
        assert role == NodeRole.AUTHORITY

    def test_determine_role_connector(self) -> None:
        """Should identify connector nodes."""
        metrics = NodeMetrics(
            chunk_id="chunk1",
            hub_score=0.8,
            authority_score=0.8,
        )

        role = metrics.determine_role()
        assert role == NodeRole.CONNECTOR

    def test_determine_role_isolated(self) -> None:
        """Should identify isolated nodes."""
        metrics = NodeMetrics(
            chunk_id="chunk1",
            hub_score=0.1,
            authority_score=0.1,
        )

        role = metrics.determine_role()
        assert role == NodeRole.ISOLATED


def _create_test_link_map() -> LinkMap:
    """Create a test link map: A -> B -> C, A -> C"""
    link_map = LinkMap()

    refs = [
        InternalReference(source_chunk_id="A", target_chunk_id="B", resolved=True),
        InternalReference(source_chunk_id="B", target_chunk_id="C", resolved=True),
        InternalReference(source_chunk_id="A", target_chunk_id="C", resolved=True),
    ]

    for ref in refs:
        link_map.add_reference(ref)

    return link_map


class TestGraphAnalyticsInit:
    """Test GraphAnalytics initialization."""

    def test_init_with_link_map(self) -> None:
        """Should initialize with link map."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)

        assert analytics is not None

    def test_empty_link_map(self) -> None:
        """Should handle empty link map."""
        link_map = LinkMap()
        analytics = GraphAnalytics(link_map)

        assert analytics is not None


class TestDegreeCalculation:
    """Test degree calculation."""

    def test_calculate_degrees(self) -> None:
        """Should calculate in/out degrees."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)

        analytics.calculate_degrees()

        metrics_a = analytics.get_metrics("A")
        metrics_b = analytics.get_metrics("B")
        metrics_c = analytics.get_metrics("C")

        assert metrics_a.out_degree == 2
        assert metrics_a.in_degree == 0
        assert metrics_b.out_degree == 1
        assert metrics_b.in_degree == 1
        assert metrics_c.out_degree == 0
        assert metrics_c.in_degree == 2


class TestHITSCalculation:
    """Test HITS algorithm."""

    def test_calculate_hits(self) -> None:
        """Should calculate HITS scores."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)

        analytics.calculate_hits()

        metrics_a = analytics.get_metrics("A")
        metrics_c = analytics.get_metrics("C")

        assert metrics_a.hub_score > 0
        assert metrics_c.authority_score > 0


class TestPageRank:
    """Test PageRank algorithm."""

    def test_calculate_pagerank(self) -> None:
        """Should calculate PageRank scores."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)

        analytics.calculate_pagerank()

        metrics_a = analytics.get_metrics("A")
        metrics_c = analytics.get_metrics("C")

        assert metrics_a.pagerank > 0
        assert metrics_c.pagerank > 0


class TestGraphStatsCalculation:
    """Test graph statistics."""

    def test_calculate_stats(self) -> None:
        """Should calculate graph statistics."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)

        stats = analytics.calculate_stats()

        assert stats.node_count == 3
        assert stats.edge_count == 3
        assert stats.density > 0


class TestTopNodes:
    """Test top node retrieval."""

    def test_get_hubs(self) -> None:
        """Should return top hubs."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)
        analytics.calculate_all_metrics()

        hubs = analytics.get_hubs(top_k=2)

        assert len(hubs) == 2

    def test_get_authorities(self) -> None:
        """Should return top authorities."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)
        analytics.calculate_all_metrics()

        authorities = analytics.get_authorities(top_k=2)

        assert len(authorities) == 2


class TestCircularReferences:
    """Test circular reference detection."""

    def test_find_no_cycles(self) -> None:
        """Should find no cycles in DAG."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)

        cycles = analytics.find_circular_references()

        assert len(cycles) == 0

    def test_find_cycle(self) -> None:
        """Should find cycles."""
        link_map = LinkMap()

        refs = [
            InternalReference(source_chunk_id="A", target_chunk_id="B", resolved=True),
            InternalReference(source_chunk_id="B", target_chunk_id="C", resolved=True),
            InternalReference(source_chunk_id="C", target_chunk_id="A", resolved=True),
        ]

        for ref in refs:
            link_map.add_reference(ref)

        analytics = GraphAnalytics(link_map)
        cycles = analytics.find_circular_references()

        assert len(cycles) > 0


class TestCalculateAllMetrics:
    """Test combined metrics calculation."""

    def test_calculate_all(self) -> None:
        """Should calculate all metrics."""
        link_map = _create_test_link_map()
        analytics = GraphAnalytics(link_map)

        metrics = analytics.calculate_all_metrics()

        assert len(metrics) == 3
        assert all(m.role is not None for m in metrics.values())
