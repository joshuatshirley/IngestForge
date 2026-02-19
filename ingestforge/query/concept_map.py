"""Concept map generation from corpus."""
import re
from collections import Counter, defaultdict
from typing import Any, List, Dict, Set, Tuple, Optional


class ConceptMapGenerator:
    """Generate concept maps showing relationships between ideas."""

    def __init__(self, min_relevance: float = 0.5) -> None:
        """
        Initialize generator.

        Args:
            min_relevance: Minimum relevance score for including concepts (0-1)
        """
        self.min_relevance = min_relevance

    def extract_concepts(
        self, chunks: List[Dict[str, Any]], max_concepts: int = 15
    ) -> List[Tuple[str, int]]:
        """
        Extract key concepts from chunks.

        Args:
            chunks: List of chunk dictionaries
            max_concepts: Maximum number of concepts to extract

        Returns:
            List of (concept, frequency) tuples
        """
        # Extract all meaningful terms (2+ chars, not common words)
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "are",
            "was",
            "has",
            "have",
            "been",
            "will",
            "can",
            "may",
            "also",
            "their",
            "which",
        }

        term_freq: Counter[str] = Counter[str]()

        for chunk in chunks:
            content = chunk.get("content", "").lower()

            # Extract capitalized phrases (likely important concepts)
            original_content = chunk.get("content", "")
            capitalized = re.findall(
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", original_content
            )
            for phrase in capitalized:
                if len(phrase) > 2:
                    term_freq[phrase] += 2  # Weight capitalized terms higher

            # Extract individual words
            words = re.findall(r"\b\w{3,}\b", content)
            for word in words:
                if word not in stopwords and not word.isdigit():
                    term_freq[word] += 1

        # Return top concepts
        return term_freq.most_common(max_concepts)

    def find_relationships(
        self, chunks: List[Dict[str, Any]], concepts: List[str]
    ) -> Dict[str, Set[str]]:
        """
        Find co-occurrence relationships between concepts.

        Args:
            chunks: List of chunk dictionaries
            concepts: List of concept terms

        Returns:
            Dictionary mapping concept -> set of related concepts
        """
        relationships = defaultdict(set)
        concept_set = set(c.lower() for c in concepts)

        for chunk in chunks:
            content = chunk.get("content", "").lower()

            # Find which concepts appear in this chunk
            found_concepts = [c for c in concepts if c.lower() in content]

            # Create relationships between co-occurring concepts
            for i, concept1 in enumerate(found_concepts):
                for concept2 in found_concepts[i + 1 :]:
                    relationships[concept1].add(concept2)
                    relationships[concept2].add(concept1)

        return relationships

    def _add_concept_edges(
        self,
        concept: str,
        related: set[str],
        node_ids: Dict[str, str],
        lines: List[str],
        added_edges: set[tuple[str, str]],
    ) -> None:
        """
        Add edges for concept relationships.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            concept: Source concept
            related: Set of related concepts
            node_ids: Mapping of concepts to node IDs
            lines: List to append edges to (mutated)
            added_edges: Set to track added edges (mutated)
        """
        for rel_concept in related:
            if rel_concept not in node_ids:
                continue

            edge: tuple[str, str] = tuple(sorted([concept, rel_concept]))  # type: ignore[assignment]
            if edge in added_edges:
                continue

            lines.append(f"    {node_ids[concept]} --- {node_ids[rel_concept]}")
            added_edges.add(edge)

    def _add_concept_nodes(
        self, concepts: List[tuple[str, int]], lines: List[str]
    ) -> dict[str, str]:
        """Add concept nodes to Mermaid diagram.

        Rule #4: No large functions - Extracted from generate_mermaid

        Returns:
            Dictionary mapping concept names to node IDs
        """
        node_ids = {}
        for i, (concept, freq) in enumerate(concepts):
            node_id = f"N{i}"
            node_ids[concept] = node_id

            # Escape quotes and special chars
            safe_concept = concept.replace('"', "'")

            # Size/style based on frequency
            if freq > 10:
                lines.append(f'    {node_id}["{safe_concept}"]')
                lines.append(
                    f"    style {node_id} fill:#bbf,stroke:#333,stroke-width:2px"
                )
            elif freq > 5:
                lines.append(f'    {node_id}["{safe_concept}"]')
            else:
                lines.append(f'    {node_id}["{safe_concept}"]')

        return node_ids

    def _connect_topic_to_concepts(
        self,
        concepts: List[tuple[str, int]],
        node_ids: dict[str, str],
        lines: List[str],
    ) -> None:
        """Connect topic node to top concepts.

        Rule #4: No large functions - Extracted from generate_mermaid
        """
        for concept, _ in concepts[:5]:  # Top 5 concepts
            if concept in node_ids:
                lines.append(f"    TOPIC -.-> {node_ids[concept]}")

    def generate_mermaid(
        self,
        chunks: List[Dict[str, Any]],
        topic: Optional[str] = None,
        max_concepts: int = 15,
    ) -> str:
        """
        Generate Mermaid diagram from chunks.

        Rule #4: Function <60 lines (refactored to 48 lines)

        Args:
            chunks: List of chunk dictionaries
            topic: Optional topic to center the map around
            max_concepts: Maximum number of concepts to show

        Returns:
            Mermaid diagram as string
        """
        # Extract concepts
        concepts = self.extract_concepts(chunks, max_concepts)
        if not concepts:
            return "graph TD\n    A[No concepts found]"

        concept_terms = [c[0] for c in concepts]

        # Find relationships
        relationships = self.find_relationships(chunks, concept_terms)

        # Build Mermaid diagram
        lines = ["graph TD"]

        # Add topic node if provided
        if topic:
            lines.append(f'    TOPIC["{topic}"]')
            lines.append("    style TOPIC fill:#f9f,stroke:#333,stroke-width:4px")

        # Add concept nodes with IDs
        node_ids = self._add_concept_nodes(concepts, lines)

        # Add relationships (edges)
        added_edges: set[tuple[str, str]] = set()
        for concept, related in relationships.items():
            if concept in node_ids:
                self._add_concept_edges(concept, related, node_ids, lines, added_edges)

        # Connect topic to top concepts if topic provided
        if topic:
            self._connect_topic_to_concepts(concepts, node_ids, lines)

        return "\n".join(lines)

    def _add_graphml_edges(
        self,
        concept: str,
        related: set[str],
        node_ids: Dict[str, str],
        lines: List[str],
        added_edges: set[tuple[str, str]],
        edge_id: int,
    ) -> int:
        """
        Add GraphML edges for concept relationships.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            concept: Source concept
            related: Set of related concepts
            node_ids: Mapping of concepts to node IDs
            lines: List to append edges to (mutated)
            added_edges: Set to track added edges (mutated)
            edge_id: Current edge ID counter

        Returns:
            Updated edge ID counter
        """
        for rel_concept in related:
            if rel_concept not in node_ids:
                continue

            pair = sorted([concept, rel_concept])
            edge = (str(pair[0]), str(pair[1]))
            if edge in added_edges:
                continue

            lines.append(
                f'    <edge id="e{edge_id}" source="{node_ids[concept]}" '
                f'target="{node_ids[rel_concept]}"/>'
            )
            edge_id += 1
            added_edges.add(edge)

        return edge_id

    def generate_graphml(
        self, chunks: List[Dict[str, Any]], topic: Optional[str] = None
    ) -> str:
        """
        Generate GraphML format (for tools like yEd).

        Args:
            chunks: List of chunk dictionaries
            topic: Optional topic

        Returns:
            GraphML XML as string
        """
        concepts = self.extract_concepts(chunks)
        concept_terms = [c[0] for c in concepts]
        relationships = self.find_relationships(chunks, concept_terms)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <graph id="G" edgedefault="undirected">',
        ]

        # Add nodes
        for i, (concept, freq) in enumerate(concepts):
            lines.append(f'    <node id="n{i}">')
            lines.append(f'      <data key="label">{concept}</data>')
            lines.append(f'      <data key="weight">{freq}</data>')
            lines.append("    </node>")

        # Add edges
        node_ids = {c[0]: f"n{i}" for i, c in enumerate(concepts)}
        edge_id = 0
        added_edges: set[tuple[str, str]] = set()

        for concept, related in relationships.items():
            if concept in node_ids:
                edge_id = self._add_graphml_edges(
                    concept, related, node_ids, lines, added_edges, edge_id
                )

        lines.extend(
            [
                "  </graph>",
                "</graphml>",
            ]
        )

        return "\n".join(lines)
