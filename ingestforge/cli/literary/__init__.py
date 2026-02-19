"""Literary analysis commands package.

Provides literary analysis tools:
- themes: Theme extraction and analysis
- character: Character development and relationship analysis
- symbols: Symbol detection and analysis
- arc: Story structure analysis
- outline: Narrative outline generation

Models:
- Character, CharacterProfile, Relationship
- Theme, ThemeArc, Evidence
- PlotPoint, StoryArc
- Symbol, SymbolAnalysis

Usage:
    from ingestforge.cli.literary import literary_app
    from ingestforge.cli.literary.models import Character, Theme, StoryArc
"""

from __future__ import annotations

from ingestforge.cli.literary.main import app as literary_app

# Import models for convenient access
from ingestforge.cli.literary.models import (
    # Character models
    Appearance,
    Character,
    CharacterProfile,
    Relationship,
    RelationshipGraph,
    # Theme models
    Evidence,
    Theme,
    ThemeArc,
    ThemeComparison,
    ThemePoint,
    # Story arc models
    PlotPoint,
    StoryArc,
    STRUCTURE_TYPES,
    # Symbol models
    Symbol,
    SymbolAnalysis,
)

# Import analysis classes
from ingestforge.cli.literary.character import CharacterExtractor
from ingestforge.cli.literary.themes import ThemeDetector
from ingestforge.cli.literary.arc import StoryArcAnalyzer
from ingestforge.cli.literary.symbols import SymbolDetector

__all__ = [
    # Main app
    "literary_app",
    # Character models
    "Appearance",
    "Character",
    "CharacterProfile",
    "Relationship",
    "RelationshipGraph",
    "CharacterExtractor",
    # Theme models
    "Evidence",
    "Theme",
    "ThemeArc",
    "ThemeComparison",
    "ThemePoint",
    "ThemeDetector",
    # Story arc models
    "PlotPoint",
    "StoryArc",
    "STRUCTURE_TYPES",
    "StoryArcAnalyzer",
    # Symbol models
    "Symbol",
    "SymbolAnalysis",
    "SymbolDetector",
]
