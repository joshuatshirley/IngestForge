"""Mermaid graph visualization for citation networks (CITE-001.3).

Generates valid Mermaid.js diagrams for TUI and Markdown export.

NASA JPL Commandments compliance:
- Rule #1: Simple control flow, no deep nesting
- Rule #2: Fixed upper bounds on node counts
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.cli.citation.graph_viz import (
        MermaidExporter,
        export_to_mermaid,
    )

    exporter = MermaidExporter(link_map, analytics)
    diagram = exporter.generate_diagram()
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from ingestforge.core.logging import get_logger

if TYPE_CHECKING:
    from ingestforge.core.citation.link_extractor import LinkMap
    from ingestforge.core.citation.analytics import (
        GraphAnalytics,
        NodeMetrics,
        NodeRole,
    )

logger = get_logger(__name__)
MAX_NODES_DEFAULT = 50
MAX_NODES_FULL = 200
MAX_LABEL_LENGTH = 30


class GraphDirection(Enum):
    """Direction of the Mermaid graph."""

    TOP_DOWN = "TD"
    BOTTOM_UP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


class NodeShape(Enum):
    """Shapes for Mermaid nodes."""

    RECTANGLE = "rect"
    ROUNDED = "rounded"
    STADIUM = "stadium"
    CIRCLE = "circle"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"


@dataclass
class MermaidStyle:
    """Styling options for Mermaid diagrams.

    Attributes:
        direction: Graph direction
        show_scores: Show hub/authority scores
        highlight_hubs: Highlight hub nodes
        highlight_authorities: Highlight authority nodes
        max_nodes: Maximum nodes to display
        include_legend: Include a legend
        node_labels: Custom node labels (id -> label)
    """

    direction: GraphDirection = GraphDirection.TOP_DOWN
    show_scores: bool = False
    highlight_hubs: bool = True
    highlight_authorities: bool = True
    max_nodes: int = MAX_NODES_DEFAULT
    include_legend: bool = True
    node_labels: Optional[Dict[str, str]] = None


# Color classes for different node types
STYLE_CLASSES = """
    classDef hub fill:#f9d71c,stroke:#333,stroke-width:2px
    classDef authority fill:#00b894,stroke:#333,stroke-width:2px
    classDef connector fill:#e17055,stroke:#333,stroke-width:2px
    classDef default fill:#74b9ff,stroke:#333,stroke-width:1px
"""


class MermaidExporter:
    """Exports citation graphs to Mermaid.js format.

    Args:
        link_map: LinkMap with citation links
        analytics: GraphAnalytics with computed metrics (optional)
        style: MermaidStyle options
    """

    def __init__(
        self,
        link_map: LinkMap,
        analytics: Optional[GraphAnalytics] = None,
        style: Optional[MermaidStyle] = None,
    ) -> None:
        """Initialize the Mermaid exporter."""
        self._link_map = link_map
        self._analytics = analytics
        self._style = style or MermaidStyle()
        self._node_labels: Dict[str, str] = {}

    def generate_diagram(
        self,
        title: Optional[str] = None,
    ) -> str:
        """Generate a complete Mermaid diagram.

        Args:
            title: Optional title for the diagram

        Returns:
            Mermaid diagram as string
        """
        lines: List[str] = []

        # Add title if provided
        if title:
            lines.append("---")
            lines.append(f"title: {title}")
            lines.append("---")

        # Start flowchart
        lines.append(f"flowchart {self._style.direction.value}")

        # Generate nodes and edges
        nodes = self._select_nodes()
        node_lines = self._generate_nodes(nodes)
        edge_lines = self._generate_edges(nodes)

        lines.extend(node_lines)
        lines.extend(edge_lines)

        # Add style classes
        if self._style.highlight_hubs or self._style.highlight_authorities:
            lines.append("")
            lines.append(STYLE_CLASSES.strip())
            lines.extend(self._generate_class_assignments(nodes))

        # Add legend if requested
        if self._style.include_legend:
            lines.extend(self._generate_legend())

        return "\n".join(lines)

    def _select_nodes(self) -> Set[str]:
        """Select nodes to include in the diagram."""
        all_nodes: Set[str] = set(self._link_map.links.keys())
        for targets in self._link_map.links.values():
            all_nodes.update(targets)
        if len(all_nodes) <= self._style.max_nodes:
            return all_nodes

        # Prioritize by importance if analytics available
        if self._analytics:
            metrics = self._analytics.calculate_all_metrics()
            sorted_nodes = sorted(
                all_nodes,
                key=lambda n: metrics.get(n, _default_metrics(n)).total_degree,
                reverse=True,
            )
            return set(sorted_nodes[: self._style.max_nodes])

        # Otherwise take first N
        return set(list(all_nodes)[: self._style.max_nodes])

    def _generate_nodes(self, nodes: Set[str]) -> List[str]:
        """Generate node definitions."""
        lines: List[str] = []

        for node_id in sorted(nodes):
            label = self._get_node_label(node_id)
            safe_id = self._sanitize_id(node_id)

            # Add scores if requested
            if self._style.show_scores and self._analytics:
                metrics = self._analytics.get_metrics(node_id)
                if metrics:
                    label += (
                        f"\\nH:{metrics.hub_score:.2f} A:{metrics.authority_score:.2f}"
                    )

            lines.append(f"    {safe_id}[{label}]")

        return lines

    def _generate_edges(self, nodes: Set[str]) -> List[str]:
        """Generate edge definitions."""
        lines: List[str] = []
        lines.append("")

        for source in sorted(nodes):
            targets = self._link_map.get_outgoing(source)

            for target in targets:
                if target not in nodes:
                    continue

                safe_source = self._sanitize_id(source)
                safe_target = self._sanitize_id(target)
                lines.append(f"    {safe_source} --> {safe_target}")

        return lines

    def _generate_class_assignments(self, nodes: Set[str]) -> List[str]:
        """Generate class assignments for styling.

        Rule #1: Max 3 nesting levels.
        """
        lines: List[str] = []

        if not self._analytics:
            return lines

        # Import here to avoid circular imports
        from ingestforge.core.citation.analytics import NodeRole

        hubs: List[str] = []
        authorities: List[str] = []
        connectors: List[str] = []

        for node_id in nodes:
            metrics = self._analytics.get_metrics(node_id)
            if not metrics:
                continue

            safe_id = self._sanitize_id(node_id)
            self._classify_node_by_role(
                metrics.role, safe_id, hubs, authorities, connectors, NodeRole
            )

        lines.append("")

        if hubs and self._style.highlight_hubs:
            lines.append(f"    class {','.join(hubs)} hub")

        if authorities and self._style.highlight_authorities:
            lines.append(f"    class {','.join(authorities)} authority")

        if connectors:
            lines.append(f"    class {','.join(connectors)} connector")

        return lines

    def _classify_node_by_role(
        self,
        role: "NodeRole",
        safe_id: str,
        hubs: List[str],
        authorities: List[str],
        connectors: List[str],
        NodeRole: type,
    ) -> None:
        """Classify node into category by role.

        Helper for _generate_class_assignments to reduce nesting.

        Args:
            role: Node role
            safe_id: Sanitized node ID
            hubs: List to append hub nodes to
            authorities: List to append authority nodes to
            connectors: List to append connector nodes to
            NodeRole: NodeRole enum type
        """
        if role == NodeRole.CONNECTOR:
            connectors.append(safe_id)
        elif role == NodeRole.HUB:
            hubs.append(safe_id)
        elif role == NodeRole.AUTHORITY:
            authorities.append(safe_id)

    def _generate_legend(self) -> List[str]:
        """Generate a legend subgraph."""
        lines: List[str] = []
        lines.append("")
        lines.append("    subgraph Legend")
        lines.append("        direction LR")
        lines.append("        leg_hub[Hub]:::hub")
        lines.append("        leg_auth[Authority]:::authority")
        lines.append("        leg_conn[Connector]:::connector")
        lines.append("    end")
        return lines

    def _get_node_label(self, node_id: str) -> str:
        """Get display label for a node."""
        # Check custom labels
        if self._style.node_labels and node_id in self._style.node_labels:
            label = self._style.node_labels[node_id]
        else:
            label = node_id

        # Truncate if too long
        if len(label) > MAX_LABEL_LENGTH:
            label = label[: MAX_LABEL_LENGTH - 3] + "..."

        # Escape special characters
        label = label.replace('"', "'")
        label = label.replace("[", "(")
        label = label.replace("]", ")")

        return label

    def _sanitize_id(self, node_id: str) -> str:
        """Sanitize node ID for Mermaid."""
        # Replace special characters with underscores
        safe = node_id
        for char in " -/.[](){}:;'\"":
            safe = safe.replace(char, "_")

        # Ensure starts with letter
        if safe and safe[0].isdigit():
            safe = "n" + safe

        return safe or "unknown"

    def export_to_file(
        self,
        filepath: str,
        title: Optional[str] = None,
    ) -> None:
        """Export diagram to a Markdown file.

        Args:
            filepath: Path to output file
            title: Optional title
        """
        diagram = self.generate_diagram(title)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("```mermaid\n")
            f.write(diagram)
            f.write("\n```\n")

        logger.info(f"Exported Mermaid diagram to {filepath}")


def _default_metrics(node_id: str) -> "NodeMetrics":
    """Create default metrics for a node."""
    from ingestforge.core.citation.analytics import NodeMetrics

    return NodeMetrics(chunk_id=node_id)


def export_to_mermaid(
    link_map: LinkMap,
    analytics: Optional[GraphAnalytics] = None,
    title: Optional[str] = None,
    max_nodes: int = MAX_NODES_DEFAULT,
) -> str:
    """Convenience function to export to Mermaid.

    Args:
        link_map: LinkMap with citation links
        analytics: GraphAnalytics with metrics
        title: Optional diagram title
        max_nodes: Maximum nodes to include

    Returns:
        Mermaid diagram string
    """
    style = MermaidStyle(max_nodes=max_nodes)
    exporter = MermaidExporter(link_map, analytics, style)
    return exporter.generate_diagram(title)


def generate_citation_graph_markdown(
    link_map: LinkMap,
    analytics: Optional[GraphAnalytics] = None,
    title: str = "Citation Graph",
) -> str:
    """Generate complete Markdown with embedded Mermaid.

    Args:
        link_map: LinkMap with citation links
        analytics: GraphAnalytics with metrics
        title: Title for the document

    Returns:
        Markdown document with embedded diagram
    """
    lines: List[str] = []

    # Add header
    lines.append(f"# {title}")
    lines.append("")

    # Add statistics if analytics available
    if analytics:
        stats = analytics.calculate_stats()
        lines.append("## Graph Statistics")
        lines.append("")
        lines.append(f"- **Nodes**: {stats.node_count}")
        lines.append(f"- **Edges**: {stats.edge_count}")
        lines.append(f"- **Density**: {stats.density:.4f}")
        lines.append(f"- **Connected Components**: {stats.connected_components}")
        lines.append("")

        # Top hubs
        lines.append("## Top Hubs")
        lines.append("")
        hubs = analytics.get_hubs(top_k=5)
        for i, hub in enumerate(hubs, 1):
            lines.append(f"{i}. `{hub.chunk_id}` (score: {hub.hub_score:.3f})")
        lines.append("")

        # Top authorities
        lines.append("## Top Authorities")
        lines.append("")
        authorities = analytics.get_authorities(top_k=5)
        for i, auth in enumerate(authorities, 1):
            lines.append(f"{i}. `{auth.chunk_id}` (score: {auth.authority_score:.3f})")
        lines.append("")

    # Add diagram
    lines.append("## Visualization")
    lines.append("")
    lines.append("```mermaid")
    diagram = export_to_mermaid(link_map, analytics, title=None)
    lines.append(diagram)
    lines.append("```")

    return "\n".join(lines)
