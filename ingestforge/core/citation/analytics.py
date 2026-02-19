"""Citation graph analytics engine (CITE-001.2).

Provides graph analysis to identify hubs and authorities
in the citation network.

NASA JPL Commandments compliance:
- Rule #1: Simple control flow, no deep nesting
- Rule #2: Fixed upper bounds on iterations (max 100)
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.core.citation.analytics import (
        GraphAnalytics,
        NodeMetrics,
    )

    analytics = GraphAnalytics(link_map)
    metrics = analytics.calculate_metrics()
    hubs = analytics.get_hubs(top_k=10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from enum import Enum

from ingestforge.core.logging import get_logger

if TYPE_CHECKING:
    from ingestforge.core.citation.link_extractor import LinkMap

logger = get_logger(__name__)
MAX_HITS_ITERATIONS = 100
MAX_PAGERANK_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_DAMPING = 0.85


class NodeRole(Enum):
    """Role of a node in the citation graph."""

    HUB = "hub"  # Many outgoing links (cites many)
    AUTHORITY = "authority"  # Many incoming links (cited often)
    CONNECTOR = "connector"  # Both hub and authority
    ISOLATED = "isolated"  # Few connections


@dataclass
class NodeMetrics:
    """Metrics for a single node in the citation graph.

    Attributes:
        chunk_id: ID of the chunk
        in_degree: Number of incoming links (citations received)
        out_degree: Number of outgoing links (citations made)
        hub_score: HITS hub score (0-1)
        authority_score: HITS authority score (0-1)
        pagerank: PageRank score
        role: Determined role of the node
    """

    chunk_id: str
    in_degree: int = 0
    out_degree: int = 0
    hub_score: float = 0.0
    authority_score: float = 0.0
    pagerank: float = 0.0
    role: NodeRole = NodeRole.ISOLATED

    @property
    def total_degree(self) -> int:
        """Total number of connections."""
        return self.in_degree + self.out_degree

    def determine_role(
        self,
        hub_threshold: float = 0.7,
        authority_threshold: float = 0.7,
    ) -> NodeRole:
        """Determine the node's role based on scores."""
        is_hub = self.hub_score >= hub_threshold
        is_authority = self.authority_score >= authority_threshold

        if is_hub and is_authority:
            return NodeRole.CONNECTOR
        if is_hub:
            return NodeRole.HUB
        if is_authority:
            return NodeRole.AUTHORITY
        return NodeRole.ISOLATED


@dataclass
class GraphStats:
    """Overall statistics for the citation graph.

    Attributes:
        node_count: Number of nodes
        edge_count: Number of edges
        density: Graph density (edges / possible edges)
        avg_in_degree: Average incoming connections
        avg_out_degree: Average outgoing connections
        max_in_degree: Maximum incoming connections
        max_out_degree: Maximum outgoing connections
        connected_components: Number of connected components
    """

    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_in_degree: float = 0.0
    avg_out_degree: float = 0.0
    max_in_degree: int = 0
    max_out_degree: int = 0
    connected_components: int = 0


class GraphAnalytics:
    """Analyzes citation graph structure.

    Calculates degree counts, HITS hub/authority scores,
    and PageRank to identify important nodes.

    Args:
        link_map: LinkMap from LinkExtractor
    """

    def __init__(self, link_map: LinkMap) -> None:
        """Initialize graph analytics."""
        self._link_map = link_map
        self._metrics: Dict[str, NodeMetrics] = {}
        self._stats: Optional[GraphStats] = None
        self._nodes: Set[str] = set()
        self._initialize_nodes()

    def _initialize_nodes(self) -> None:
        """Initialize node set from link map."""
        self._nodes = set(self._link_map.links.keys())
        for targets in self._link_map.links.values():
            self._nodes.update(targets)

        # Initialize metrics for each node
        for node_id in self._nodes:
            self._metrics[node_id] = NodeMetrics(chunk_id=node_id)

    def calculate_degrees(self) -> None:
        """Calculate in-degree and out-degree for all nodes."""
        for node_id in self._nodes:
            outgoing = self._link_map.get_outgoing(node_id)
            incoming = self._link_map.get_incoming(node_id)

            self._metrics[node_id].out_degree = len(outgoing)
            self._metrics[node_id].in_degree = len(incoming)

    def calculate_hits(
        self,
        max_iterations: int = MAX_HITS_ITERATIONS,
    ) -> None:
        """Calculate HITS hub and authority scores.

        Uses the Hyperlink-Induced Topic Search algorithm.
        Hubs point to many good authorities.
        Authorities are pointed to by many good hubs.

        Args:
            max_iterations: Maximum iterations (Rule #2)
        """
        if not self._nodes:
            return

        n = len(self._nodes)
        node_list = list(self._nodes)
        node_idx = {node: i for i, node in enumerate(node_list)}

        # Initialize scores
        hub = [1.0 / n] * n
        auth = [1.0 / n] * n

        for iteration in range(max_iterations):
            old_hub = hub.copy()
            old_auth = auth.copy()

            # Update authority scores
            for i, node in enumerate(node_list):
                incoming = self._link_map.get_incoming(node)
                auth[i] = sum(hub[node_idx[src]] for src in incoming if src in node_idx)

            # Update hub scores
            for i, node in enumerate(node_list):
                outgoing = self._link_map.get_outgoing(node)
                hub[i] = sum(auth[node_idx[tgt]] for tgt in outgoing if tgt in node_idx)

            # Normalize
            hub_sum = sum(hub) or 1.0
            auth_sum = sum(auth) or 1.0
            hub = [h / hub_sum for h in hub]
            auth = [a / auth_sum for a in auth]

            # Check convergence
            hub_diff = sum(abs(hub[i] - old_hub[i]) for i in range(n))
            auth_diff = sum(abs(auth[i] - old_auth[i]) for i in range(n))

            if hub_diff < CONVERGENCE_THRESHOLD and auth_diff < CONVERGENCE_THRESHOLD:
                logger.debug(f"HITS converged at iteration {iteration}")
                break

        # Store results
        for i, node in enumerate(node_list):
            self._metrics[node].hub_score = hub[i]
            self._metrics[node].authority_score = auth[i]

    def calculate_pagerank(
        self,
        damping: float = DEFAULT_DAMPING,
        max_iterations: int = MAX_PAGERANK_ITERATIONS,
    ) -> None:
        """Calculate PageRank scores.

        Args:
            damping: Damping factor (0.85 standard)
            max_iterations: Maximum iterations (Rule #2)
        """
        if not self._nodes:
            return

        n = len(self._nodes)
        node_list = list(self._nodes)
        node_idx = {node: i for i, node in enumerate(node_list)}

        # Initialize scores
        rank = [1.0 / n] * n

        for iteration in range(max_iterations):
            old_rank = rank.copy()
            rank = [(1 - damping) / n] * n

            self._update_pagerank_scores(rank, old_rank, node_list, node_idx, damping)

            # Check convergence
            diff = sum(abs(rank[i] - old_rank[i]) for i in range(n))
            if diff < CONVERGENCE_THRESHOLD:
                logger.debug(f"PageRank converged at iteration {iteration}")
                break

        # Store results
        for i, node in enumerate(node_list):
            self._metrics[node].pagerank = rank[i]

    def _update_pagerank_scores(
        self,
        rank: List[float],
        old_rank: List[float],
        node_list: List[str],
        node_idx: Dict[str, int],
        damping: float,
    ) -> None:
        """Update PageRank scores for one iteration.

        Args:
            rank: Current rank scores (modified in place)
            old_rank: Previous rank scores
            node_list: List of nodes
            node_idx: Node to index mapping
            damping: Damping factor
        """
        for i, node in enumerate(node_list):
            incoming = self._link_map.get_incoming(node)

            for src in incoming:
                if src not in node_idx:
                    continue
                src_idx = node_idx[src]
                out_count = len(self._link_map.get_outgoing(src)) or 1
                rank[i] += damping * old_rank[src_idx] / out_count

    def calculate_all_metrics(self) -> Dict[str, NodeMetrics]:
        """Calculate all metrics for all nodes.

        Returns:
            Dict mapping chunk_id to NodeMetrics
        """
        self.calculate_degrees()
        self.calculate_hits()
        self.calculate_pagerank()

        # Determine roles
        for metrics in self._metrics.values():
            metrics.role = metrics.determine_role()

        return self._metrics

    def calculate_stats(self) -> GraphStats:
        """Calculate overall graph statistics.

        Returns:
            GraphStats with summary statistics
        """
        if self._stats:
            return self._stats

        # Ensure degrees are calculated
        if not any(m.in_degree or m.out_degree for m in self._metrics.values()):
            self.calculate_degrees()

        n = len(self._nodes)
        e = self._link_map.edge_count

        # Calculate density
        possible_edges = n * (n - 1) if n > 1 else 1
        density = e / possible_edges if possible_edges > 0 else 0

        # Degree statistics
        in_degrees = [m.in_degree for m in self._metrics.values()]
        out_degrees = [m.out_degree for m in self._metrics.values()]

        self._stats = GraphStats(
            node_count=n,
            edge_count=e,
            density=density,
            avg_in_degree=sum(in_degrees) / n if n > 0 else 0,
            avg_out_degree=sum(out_degrees) / n if n > 0 else 0,
            max_in_degree=max(in_degrees) if in_degrees else 0,
            max_out_degree=max(out_degrees) if out_degrees else 0,
            connected_components=self._count_components(),
        )

        return self._stats

    def _count_components(self) -> int:
        """Count connected components using BFS."""
        visited: Set[str] = set()
        components = 0

        for node in self._nodes:
            if node in visited:
                continue

            # BFS from this node
            components += 1
            self._bfs_component(node, visited)

        return components

    def _bfs_component(self, start_node: str, visited: Set[str]) -> None:
        """Perform BFS to mark all nodes in a component.

        Args:
            start_node: Starting node
            visited: Set of visited nodes (modified in place)
        """
        queue = [start_node]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Add neighbors (both directions for undirected connectivity)
            neighbors = set(self._link_map.get_outgoing(current))
            neighbors.update(self._link_map.get_incoming(current))

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    def get_hubs(self, top_k: int = 10) -> List[NodeMetrics]:
        """Get top hub nodes.

        Args:
            top_k: Number of top hubs to return

        Returns:
            List of NodeMetrics sorted by hub score
        """
        if not any(m.hub_score for m in self._metrics.values()):
            self.calculate_hits()

        sorted_nodes = sorted(
            self._metrics.values(),
            key=lambda m: m.hub_score,
            reverse=True,
        )
        return sorted_nodes[:top_k]

    def get_authorities(self, top_k: int = 10) -> List[NodeMetrics]:
        """Get top authority nodes.

        Args:
            top_k: Number of top authorities to return

        Returns:
            List of NodeMetrics sorted by authority score
        """
        if not any(m.authority_score for m in self._metrics.values()):
            self.calculate_hits()

        sorted_nodes = sorted(
            self._metrics.values(),
            key=lambda m: m.authority_score,
            reverse=True,
        )
        return sorted_nodes[:top_k]

    def get_by_pagerank(self, top_k: int = 10) -> List[NodeMetrics]:
        """Get top nodes by PageRank.

        Args:
            top_k: Number of top nodes to return

        Returns:
            List of NodeMetrics sorted by PageRank
        """
        if not any(m.pagerank for m in self._metrics.values()):
            self.calculate_pagerank()

        sorted_nodes = sorted(
            self._metrics.values(),
            key=lambda m: m.pagerank,
            reverse=True,
        )
        return sorted_nodes[:top_k]

    def get_metrics(self, chunk_id: str) -> Optional[NodeMetrics]:
        """Get metrics for a specific chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            NodeMetrics or None if not found
        """
        return self._metrics.get(chunk_id)

    def find_circular_references(
        self,
        max_depth: int = 10,
    ) -> List[List[str]]:
        """Find circular reference chains.

        Args:
            max_depth: Maximum depth to search (Rule #2)

        Returns:
            List of circular chains (each is a list of chunk IDs)
        """
        cycles: List[List[str]] = []

        for start_node in self._nodes:
            visited: Set[str] = set()
            path: List[str] = []

            self._find_cycles_dfs(
                start_node, start_node, visited, path, cycles, 0, max_depth
            )

        return cycles

    def _find_cycles_dfs(
        self,
        current: str,
        start: str,
        visited: Set[str],
        path: List[str],
        cycles: List[List[str]],
        depth: int,
        max_depth: int,
    ) -> None:
        """DFS helper for cycle detection."""
        if depth > max_depth:
            return

        if current in visited:
            if current == start and len(path) > 1:
                # Found a cycle
                cycles.append(path.copy() + [current])
            return

        visited.add(current)
        path.append(current)

        for neighbor in self._link_map.get_outgoing(current):
            self._find_cycles_dfs(
                neighbor, start, visited, path, cycles, depth + 1, max_depth
            )

        path.pop()
        visited.discard(current)
