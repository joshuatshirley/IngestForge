"""
Tests for Literary Analysis Data Models.

Tests all dataclasses and their methods:
- Character models (Character, CharacterProfile, Relationship)
- Theme models (Theme, ThemeArc, Evidence)
- Story arc models (PlotPoint, StoryArc)
- Symbol models (Symbol, SymbolAnalysis)

Test Strategy
-------------
- Test dataclass instantiation
- Test to_dict() methods
- Test visualization methods (to_mermaid, to_ascii_art)
- Keep tests simple (NASA JPL Rule #1)
"""


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


# ============================================================================
# Character Model Tests
# ============================================================================


class TestAppearance:
    """Tests for Appearance dataclass."""

    def test_create_appearance(self):
        """Test creating Appearance instance."""
        app = Appearance(chunk_index=0, chunk_id="chunk_1", context="test")
        assert app.chunk_index == 0
        assert app.chunk_id == "chunk_1"
        assert app.context == "test"
        assert app.position == 0.0

    def test_appearance_with_position(self):
        """Test Appearance with explicit position."""
        app = Appearance(
            chunk_index=5, chunk_id="chunk_5", context="middle", position=0.5
        )
        assert app.position == 0.5


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_create_relationship(self):
        """Test creating Relationship instance."""
        rel = Relationship(target="Juliet", relationship_type="romantic")
        assert rel.target == "Juliet"
        assert rel.relationship_type == "romantic"
        assert rel.confidence == 0.8  # default
        assert rel.evidence == ""

    def test_relationship_with_confidence(self):
        """Test Relationship with custom confidence."""
        rel = Relationship(
            target="Tybalt",
            relationship_type="enemy",
            confidence=0.95,
            evidence="They fought a duel",
        )
        assert rel.confidence == 0.95
        assert rel.evidence == "They fought a duel"


class TestCharacter:
    """Tests for Character dataclass."""

    def test_create_character(self):
        """Test creating Character instance."""
        char = Character(name="Hamlet")
        assert char.name == "Hamlet"
        assert char.aliases == []
        assert char.first_appearance == 0
        assert char.mention_count == 1
        assert char.relationships == {}

    def test_character_with_aliases(self):
        """Test Character with aliases."""
        char = Character(
            name="Harry Potter",
            aliases=["The Boy Who Lived", "The Chosen One"],
            mention_count=500,
        )
        assert len(char.aliases) == 2
        assert char.mention_count == 500

    def test_character_to_dict(self):
        """Test Character.to_dict() method."""
        char = Character(
            name="Sherlock Holmes",
            aliases=["Holmes"],
            first_appearance=5,
            mention_count=100,
            relationships={"Watson": "friend"},
        )
        data = char.to_dict()

        assert data["name"] == "Sherlock Holmes"
        assert data["aliases"] == ["Holmes"]
        assert data["first_appearance"] == 5
        assert data["mention_count"] == 100
        assert data["relationships"] == {"Watson": "friend"}

    def test_character_with_relationships(self):
        """Test Character with relationships."""
        char = Character(name="Romeo")
        char.relationships["Juliet"] = "romantic"
        char.relationships["Tybalt"] = "enemy"

        assert len(char.relationships) == 2
        assert char.relationships["Juliet"] == "romantic"


class TestCharacterProfile:
    """Tests for CharacterProfile dataclass."""

    def test_create_profile(self):
        """Test creating CharacterProfile instance."""
        char = Character(name="Hamlet")
        profile = CharacterProfile(character=char)

        assert profile.character.name == "Hamlet"
        assert profile.description == ""
        assert profile.arc_summary == ""
        assert profile.key_moments == []
        assert profile.relationships == []
        assert profile.traits == []
        assert profile.motivations == []

    def test_profile_with_details(self):
        """Test CharacterProfile with full details."""
        char = Character(name="Hamlet")
        profile = CharacterProfile(
            character=char,
            description="Prince of Denmark",
            arc_summary="From indecision to action",
            key_moments=["Ghost scene", "To be or not to be", "Final duel"],
            traits=["indecisive", "philosophical", "vengeful"],
            motivations=["revenge", "justice", "truth"],
        )

        assert profile.description == "Prince of Denmark"
        assert len(profile.key_moments) == 3
        assert "indecisive" in profile.traits

    def test_profile_to_dict(self):
        """Test CharacterProfile.to_dict() method."""
        char = Character(name="Macbeth")
        rel = Relationship(target="Lady Macbeth", relationship_type="spouse")

        profile = CharacterProfile(
            character=char,
            description="Scottish lord",
            relationships=[rel],
        )

        data = profile.to_dict()
        assert data["character"]["name"] == "Macbeth"
        assert data["description"] == "Scottish lord"
        assert len(data["relationships"]) == 1


class TestRelationshipGraph:
    """Tests for RelationshipGraph dataclass."""

    def test_create_empty_graph(self):
        """Test creating empty RelationshipGraph."""
        graph = RelationshipGraph()
        assert graph.characters == []
        assert graph.edges == []

    def test_graph_with_characters(self):
        """Test RelationshipGraph with characters."""
        char1 = Character(name="Romeo", relationships={"Juliet": "romantic"})
        char2 = Character(name="Juliet", relationships={"Romeo": "romantic"})

        graph = RelationshipGraph(characters=[char1, char2])
        assert len(graph.characters) == 2

    def test_graph_to_mermaid(self):
        """Test RelationshipGraph.to_mermaid() method."""
        char1 = Character(name="Romeo", relationships={"Juliet": "loves"})
        graph = RelationshipGraph(characters=[char1])

        mermaid = graph.to_mermaid()
        assert "graph TD" in mermaid
        assert "Romeo" in mermaid

    def test_graph_to_dict(self):
        """Test RelationshipGraph.to_dict() method."""
        char = Character(name="Test")
        graph = RelationshipGraph(characters=[char])

        data = graph.to_dict()
        assert "characters" in data
        assert "edges" in data


# ============================================================================
# Theme Model Tests
# ============================================================================


class TestThemePoint:
    """Tests for ThemePoint dataclass."""

    def test_create_theme_point(self):
        """Test creating ThemePoint instance."""
        point = ThemePoint(position=0.5, intensity=0.8)
        assert point.position == 0.5
        assert point.intensity == 0.8
        assert point.chunk_id == ""

    def test_theme_point_with_chunk(self):
        """Test ThemePoint with chunk_id."""
        point = ThemePoint(position=0.25, intensity=0.6, chunk_id="chunk_10")
        assert point.chunk_id == "chunk_10"


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_create_evidence(self):
        """Test creating Evidence instance."""
        ev = Evidence(quote="To be or not to be", chunk_id="chunk_5")
        assert ev.quote == "To be or not to be"
        assert ev.chunk_id == "chunk_5"
        assert ev.explanation == ""
        assert ev.position == 0.0

    def test_evidence_with_explanation(self):
        """Test Evidence with explanation."""
        ev = Evidence(
            quote="Something is rotten in the state of Denmark",
            chunk_id="chunk_3",
            explanation="Indicates theme of corruption",
            position=0.1,
        )
        assert "corruption" in ev.explanation


class TestTheme:
    """Tests for Theme dataclass."""

    def test_create_theme(self):
        """Test creating Theme instance."""
        theme = Theme(name="Love")
        assert theme.name == "Love"
        assert theme.confidence == 0.8  # default
        assert theme.keywords == []
        assert theme.evidence_count == 0
        assert theme.description == ""

    def test_theme_with_details(self):
        """Test Theme with full details."""
        theme = Theme(
            name="Revenge",
            confidence=0.95,
            keywords=["avenge", "vengeance", "retribution"],
            evidence_count=25,
            description="Central theme of Hamlet",
        )

        assert theme.confidence == 0.95
        assert len(theme.keywords) == 3
        assert theme.evidence_count == 25

    def test_theme_to_dict(self):
        """Test Theme.to_dict() method."""
        theme = Theme(
            name="Justice",
            confidence=0.85,
            keywords=["fair", "right", "judge"],
            evidence_count=10,
        )

        data = theme.to_dict()
        assert data["name"] == "Justice"
        assert data["confidence"] == 0.85
        assert len(data["keywords"]) == 3


class TestThemeArc:
    """Tests for ThemeArc dataclass."""

    def test_create_theme_arc(self):
        """Test creating ThemeArc instance."""
        theme = Theme(name="Power")
        arc = ThemeArc(theme=theme)

        assert arc.theme.name == "Power"
        assert arc.development == []
        assert arc.peak_moments == []

    def test_theme_arc_with_development(self):
        """Test ThemeArc with development points."""
        theme = Theme(name="Corruption")
        points = [
            ThemePoint(position=0.0, intensity=0.2),
            ThemePoint(position=0.5, intensity=0.8),
            ThemePoint(position=1.0, intensity=0.3),
        ]

        arc = ThemeArc(
            theme=theme,
            development=points,
            peak_moments=["Murder of Duncan", "Banquet scene"],
        )

        assert len(arc.development) == 3
        assert len(arc.peak_moments) == 2

    def test_theme_arc_to_dict(self):
        """Test ThemeArc.to_dict() method."""
        theme = Theme(name="Test")
        arc = ThemeArc(theme=theme)

        data = arc.to_dict()
        assert "theme" in data
        assert "development" in data
        assert "peak_moments" in data


class TestThemeComparison:
    """Tests for ThemeComparison dataclass."""

    def test_create_comparison(self):
        """Test creating ThemeComparison instance."""
        comp = ThemeComparison()
        assert comp.themes == []
        assert comp.correlations == {}
        assert comp.summary == ""

    def test_comparison_with_themes(self):
        """Test ThemeComparison with themes."""
        themes = [Theme(name="Love"), Theme(name="Death")]
        comp = ThemeComparison(
            themes=themes,
            correlations={"Love": {"Death": 0.7}},
            summary="Love and death are intertwined",
        )

        assert len(comp.themes) == 2
        assert comp.correlations["Love"]["Death"] == 0.7


# ============================================================================
# Story Arc Model Tests
# ============================================================================


class TestPlotPoint:
    """Tests for PlotPoint dataclass."""

    def test_create_plot_point(self):
        """Test creating PlotPoint instance."""
        point = PlotPoint(
            name="Inciting Incident",
            type="inciting_incident",
            position=0.1,
            description="The murder is revealed",
        )

        assert point.name == "Inciting Incident"
        assert point.type == "inciting_incident"
        assert point.position == 0.1
        assert point.chunk_index == 0  # default

    def test_plot_point_to_dict(self):
        """Test PlotPoint.to_dict() method."""
        point = PlotPoint(
            name="Climax",
            type="climax",
            position=0.8,
            description="Final confrontation",
            chunk_index=50,
        )

        data = point.to_dict()
        assert data["name"] == "Climax"
        assert data["type"] == "climax"
        assert data["position"] == 0.8
        assert data["chunk_index"] == 50


class TestStoryArc:
    """Tests for StoryArc dataclass."""

    def test_create_story_arc(self):
        """Test creating StoryArc instance."""
        arc = StoryArc(structure_type="three-act")

        assert arc.structure_type == "three-act"
        assert arc.plot_points == []
        assert arc.tension_curve == []
        assert arc.summary == ""
        assert arc.acts == []

    def test_story_arc_with_points(self):
        """Test StoryArc with plot points."""
        points = [
            PlotPoint(name="Start", type="exposition", position=0.0, description=""),
            PlotPoint(name="End", type="resolution", position=1.0, description=""),
        ]

        arc = StoryArc(
            structure_type="three-act",
            plot_points=points,
            tension_curve=[0.2, 0.5, 0.8, 0.3],
            summary="Classic three-act structure",
        )

        assert len(arc.plot_points) == 2
        assert len(arc.tension_curve) == 4

    def test_story_arc_to_dict(self):
        """Test StoryArc.to_dict() method."""
        arc = StoryArc(structure_type="hero-journey")

        data = arc.to_dict()
        assert data["structure_type"] == "hero-journey"
        assert "plot_points" in data
        assert "tension_curve" in data

    def test_story_arc_to_mermaid(self):
        """Test StoryArc.to_mermaid() method."""
        points = [
            PlotPoint(name="Start", type="exposition", position=0.0, description=""),
            PlotPoint(name="Middle", type="climax", position=0.5, description=""),
        ]

        arc = StoryArc(structure_type="three-act", plot_points=points)
        mermaid = arc.to_mermaid()

        assert "graph LR" in mermaid
        assert "Start" in mermaid
        assert "Middle" in mermaid

    def test_story_arc_to_ascii_art(self):
        """Test StoryArc.to_ascii_art() method."""
        arc = StoryArc(
            structure_type="three-act",
            tension_curve=[0.2, 0.4, 0.6, 0.8, 0.6, 0.3],
        )

        ascii_art = arc.to_ascii_art()
        assert "Tension Curve" in ascii_art
        assert "|" in ascii_art

    def test_story_arc_empty_tension(self):
        """Test StoryArc.to_ascii_art() with empty tension."""
        arc = StoryArc(structure_type="three-act")
        result = arc.to_ascii_art()
        assert "No tension data" in result


class TestStructureTypes:
    """Tests for STRUCTURE_TYPES constant."""

    def test_structure_types_defined(self):
        """Test STRUCTURE_TYPES is defined."""
        assert len(STRUCTURE_TYPES) > 0

    def test_three_act_in_types(self):
        """Test three-act is a valid structure type."""
        assert "three-act" in STRUCTURE_TYPES

    def test_hero_journey_in_types(self):
        """Test hero-journey is a valid structure type."""
        assert "hero-journey" in STRUCTURE_TYPES


# ============================================================================
# Symbol Model Tests
# ============================================================================


class TestSymbol:
    """Tests for Symbol dataclass."""

    def test_create_symbol(self):
        """Test creating Symbol instance."""
        symbol = Symbol(name="Water")

        assert symbol.name == "Water"
        assert symbol.occurrences == 1
        assert symbol.first_appearance == 0
        assert symbol.contexts == []
        assert symbol.associated_themes == []

    def test_symbol_with_details(self):
        """Test Symbol with full details."""
        symbol = Symbol(
            name="Green Light",
            occurrences=15,
            first_appearance=3,
            contexts=["He stretched out his arms", "The green light"],
            associated_themes=["hope", "dreams", "future"],
        )

        assert symbol.occurrences == 15
        assert len(symbol.contexts) == 2
        assert "hope" in symbol.associated_themes

    def test_symbol_to_dict(self):
        """Test Symbol.to_dict() method."""
        symbol = Symbol(
            name="Rose",
            occurrences=10,
            contexts=["a", "b", "c", "d", "e", "f"],  # More than 5
        )

        data = symbol.to_dict()
        assert data["name"] == "Rose"
        assert data["occurrences"] == 10
        assert len(data["contexts"]) <= 5  # Limited in to_dict


class TestSymbolAnalysis:
    """Tests for SymbolAnalysis dataclass."""

    def test_create_symbol_analysis(self):
        """Test creating SymbolAnalysis instance."""
        symbol = Symbol(name="Mirror")
        analysis = SymbolAnalysis(symbol=symbol)

        assert analysis.symbol.name == "Mirror"
        assert analysis.literal_meaning == ""
        assert analysis.symbolic_meaning == ""
        assert analysis.significance == ""
        assert analysis.development == ""
        assert analysis.connections == []

    def test_symbol_analysis_with_details(self):
        """Test SymbolAnalysis with full details."""
        symbol = Symbol(name="Whale")
        analysis = SymbolAnalysis(
            symbol=symbol,
            literal_meaning="A large sea creature",
            symbolic_meaning="Obsession, the unknowable, nature's power",
            significance="Central to the novel's themes",
            development="Grows more ominous as story progresses",
            connections=["Ahab", "Fate", "Nature vs Man"],
        )

        assert "obsession" in analysis.symbolic_meaning.lower()
        assert len(analysis.connections) == 3

    def test_symbol_analysis_to_dict(self):
        """Test SymbolAnalysis.to_dict() method."""
        symbol = Symbol(name="Test")
        analysis = SymbolAnalysis(
            symbol=symbol,
            literal_meaning="A literal thing",
            symbolic_meaning="A symbolic thing",
        )

        data = analysis.to_dict()
        assert "symbol" in data
        assert data["literal_meaning"] == "A literal thing"
        assert data["symbolic_meaning"] == "A symbolic thing"
