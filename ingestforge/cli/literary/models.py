"""Data models for literary analysis.

Provides structured data classes for literary analysis results:
- Character: Character with aliases and relationships
- CharacterProfile: Extended character profile with arc summary
- Appearance: Character appearance in a chunk
- Relationship: Character-to-character relationship
- Theme: Detected theme with confidence and keywords
- ThemeArc: Theme development across narrative
- ThemePoint: Theme intensity at a position
- Evidence: Theme evidence with quote and location
- PlotPoint: Story structure plot point
- StoryArc: Complete story arc with plot points
- Symbol: Detected symbolic element
- SymbolAnalysis: Extended symbol analysis"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# ============================================================================
# Character Models
# ============================================================================


@dataclass
class Appearance:
    """Character appearance in a chunk.

    Attributes:
        chunk_index: Index of chunk containing appearance
        chunk_id: ID of the chunk
        context: Surrounding text context
        position: Position within narrative (0.0 to 1.0)
    """

    chunk_index: int
    chunk_id: str
    context: str
    position: float = 0.0


@dataclass
class Relationship:
    """Relationship between two characters.

    Attributes:
        target: Name of related character
        relationship_type: Type (family, friend, enemy, romantic, mentor, etc.)
        confidence: Confidence score (0.0 to 1.0)
        evidence: Text evidence for relationship
    """

    target: str
    relationship_type: str
    confidence: float = 0.8
    evidence: str = ""


@dataclass
class Character:
    """Literary character with tracking metadata.

    Attributes:
        name: Primary character name
        aliases: Alternative names/nicknames
        first_appearance: Index of first chunk appearance
        mention_count: Total number of mentions
        relationships: Map of character name to relationship type
    """

    name: str
    aliases: List[str] = field(default_factory=list)
    first_appearance: int = 0
    mention_count: int = 1
    relationships: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "aliases": self.aliases,
            "first_appearance": self.first_appearance,
            "mention_count": self.mention_count,
            "relationships": self.relationships,
        }


@dataclass
class CharacterProfile:
    """Extended character profile with analysis.

    Attributes:
        character: Base character data
        description: Character description
        arc_summary: Summary of character's arc
        key_moments: List of key moments
        relationships: Detailed relationship objects
        traits: Character traits
        motivations: Character motivations
    """

    character: Character
    description: str = ""
    arc_summary: str = ""
    key_moments: List[str] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "character": self.character.to_dict(),
            "description": self.description,
            "arc_summary": self.arc_summary,
            "key_moments": self.key_moments,
            "relationships": [
                {"target": r.target, "type": r.relationship_type}
                for r in self.relationships
            ],
            "traits": self.traits,
            "motivations": self.motivations,
        }


# ============================================================================
# Theme Models
# ============================================================================


@dataclass
class ThemePoint:
    """Theme intensity at a narrative position.

    Attributes:
        position: Position in narrative (0.0 to 1.0)
        intensity: Theme intensity (0.0 to 1.0)
        chunk_id: Associated chunk ID
    """

    position: float
    intensity: float
    chunk_id: str = ""


@dataclass
class Evidence:
    """Evidence supporting a theme.

    Attributes:
        quote: Relevant text quote
        chunk_id: Source chunk ID
        explanation: Why this supports the theme
        position: Position in narrative
    """

    quote: str
    chunk_id: str
    explanation: str = ""
    position: float = 0.0


@dataclass
class Theme:
    """Detected literary theme.

    Attributes:
        name: Theme name
        confidence: Detection confidence (0.0 to 1.0)
        keywords: Associated keywords
        evidence_count: Number of supporting passages
        description: Theme description
    """

    name: str
    confidence: float = 0.8
    keywords: List[str] = field(default_factory=list)
    evidence_count: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "confidence": self.confidence,
            "keywords": self.keywords,
            "evidence_count": self.evidence_count,
            "description": self.description,
        }


@dataclass
class ThemeArc:
    """Theme development across the narrative.

    Attributes:
        theme: Base theme data
        development: Theme intensity over narrative
        peak_moments: Key moments of theme prominence
    """

    theme: Theme
    development: List[ThemePoint] = field(default_factory=list)
    peak_moments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theme": self.theme.to_dict(),
            "development": [
                {"position": p.position, "intensity": p.intensity}
                for p in self.development
            ],
            "peak_moments": self.peak_moments,
        }


@dataclass
class ThemeComparison:
    """Comparison between multiple themes.

    Attributes:
        themes: Themes being compared
        correlations: Theme correlation matrix
        summary: Comparison summary
    """

    themes: List[Theme] = field(default_factory=list)
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: str = ""


# ============================================================================
# Story Arc Models
# ============================================================================


@dataclass
class PlotPoint:
    """Identified plot point in narrative structure.

    Attributes:
        name: Plot point name
        type: Type (inciting_incident, climax, resolution, etc.)
        position: Position in narrative (0.0 to 1.0)
        description: Description of the plot point
        chunk_index: Associated chunk index
    """

    name: str
    type: str
    position: float
    description: str
    chunk_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "position": self.position,
            "description": self.description,
            "chunk_index": self.chunk_index,
        }


# Valid structure types for story arc analysis
STRUCTURE_TYPES = [
    "three-act",
    "hero-journey",
    "five-act",
    "freytag",
]


@dataclass
class StoryArc:
    """Complete story arc analysis.

    Attributes:
        structure_type: Type of structure detected/applied
        plot_points: Identified plot points
        tension_curve: Tension values across narrative
        summary: Arc analysis summary
        acts: Act/section divisions
    """

    structure_type: str
    plot_points: List[PlotPoint] = field(default_factory=list)
    tension_curve: List[float] = field(default_factory=list)
    summary: str = ""
    acts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "structure_type": self.structure_type,
            "plot_points": [p.to_dict() for p in self.plot_points],
            "tension_curve": self.tension_curve,
            "summary": self.summary,
            "acts": self.acts,
        }

    def to_mermaid(self) -> str:
        """Export arc as Mermaid diagram."""
        lines = ["graph LR"]

        for i, point in enumerate(self.plot_points):
            node_id = f"P{i}"
            label = f"{point.name}"
            lines.append(f"    {node_id}[{label}]")

            if i > 0:
                prev_id = f"P{i-1}"
                lines.append(f"    {prev_id} --> {node_id}")

        return "\n".join(lines)

    def to_ascii_art(self) -> str:
        """Export tension curve as ASCII art."""
        if not self.tension_curve:
            return "No tension data available"

        height = 10
        width = min(60, len(self.tension_curve))
        max_tension = max(self.tension_curve) if self.tension_curve else 1

        # Sample tension curve to fit width
        step = len(self.tension_curve) / width
        sampled = [self.tension_curve[int(i * step)] for i in range(width)]

        lines = []
        for row in range(height, 0, -1):
            threshold = (row / height) * max_tension
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += "*"
                else:
                    line += " "
            lines.append(f"|{line}|")

        lines.append("+" + "-" * width + "+")
        lines.append(" Tension Curve (left=start, right=end)")

        return "\n".join(lines)


# ============================================================================
# Symbol Models
# ============================================================================


@dataclass
class Symbol:
    """Detected symbolic element.

    Attributes:
        name: Symbol name/identifier
        occurrences: Number of occurrences
        first_appearance: First chunk index
        contexts: Sample contexts where symbol appears
        associated_themes: Themes this symbol relates to
    """

    name: str
    occurrences: int = 1
    first_appearance: int = 0
    contexts: List[str] = field(default_factory=list)
    associated_themes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "occurrences": self.occurrences,
            "first_appearance": self.first_appearance,
            "contexts": self.contexts[:5],  # Limit contexts
            "associated_themes": self.associated_themes,
        }


@dataclass
class SymbolAnalysis:
    """Extended symbol analysis.

    Attributes:
        symbol: Base symbol data
        literal_meaning: Literal meaning in text
        symbolic_meaning: Symbolic/metaphorical meaning
        significance: Overall significance to the work
        development: How the symbol develops
        connections: Connections to characters/themes
    """

    symbol: Symbol
    literal_meaning: str = ""
    symbolic_meaning: str = ""
    significance: str = ""
    development: str = ""
    connections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol.to_dict(),
            "literal_meaning": self.literal_meaning,
            "symbolic_meaning": self.symbolic_meaning,
            "significance": self.significance,
            "development": self.development,
            "connections": self.connections,
        }


# ============================================================================
# Relationship Graph
# ============================================================================


@dataclass
class RelationshipGraph:
    """Graph of character relationships.

    Attributes:
        characters: List of characters
        edges: List of relationship edges
    """

    characters: List[Character] = field(default_factory=list)
    edges: List[Relationship] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "characters": [c.to_dict() for c in self.characters],
            "edges": [
                {
                    "source": e.target,  # Note: edge has target, source implied
                    "type": e.relationship_type,
                    "confidence": e.confidence,
                }
                for e in self.edges
            ],
        }

    def to_mermaid(self) -> str:
        """Export as Mermaid diagram."""
        lines = ["graph TD"]

        # Add character nodes
        for char in self.characters:
            node_id = char.name.replace(" ", "_")
            lines.append(f"    {node_id}[{char.name}]")

        # Add relationship edges
        for char in self.characters:
            source_id = char.name.replace(" ", "_")
            for target, rel_type in char.relationships.items():
                target_id = target.replace(" ", "_")
                lines.append(f"    {source_id} -->|{rel_type}| {target_id}")

        return "\n".join(lines)
