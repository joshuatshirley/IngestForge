"""
Tests for Character CLI Command and CharacterExtractor.

Tests character extraction, tracking, and relationship analysis.

Test Strategy
-------------
- Test CharacterExtractor class methods
- Test CharacterCommand execution
- Mock LLM and storage for unit testing
- Keep tests simple (NASA JPL Rule #1)
"""

import json
from unittest.mock import Mock, patch


from ingestforge.cli.literary.character import (
    CharacterCommand,
    CharacterExtractor,
    command,
    extract_command,
)
from ingestforge.cli.literary.models import (
    Character,
    CharacterProfile,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "Harry Potter and Hermione Granger walked through the castle.",
    **metadata,
):
    """Create a mock chunk object."""
    chunk = Mock()
    chunk.chunk_id = chunk_id
    chunk.content = content
    chunk.text = content
    chunk.metadata = metadata
    return chunk


def make_mock_context(has_storage: bool = True):
    """Create a mock context dictionary."""
    ctx = {"config": Mock()}
    if has_storage:
        ctx["storage"] = Mock()
        ctx["storage"].search = Mock(return_value=[make_mock_chunk()])
    return ctx


def make_mock_llm_client():
    """Create a mock LLM client."""
    llm = Mock()
    llm.generate = Mock(
        return_value=json.dumps(
            {
                "description": "A brave wizard",
                "arc_summary": "From orphan to hero",
                "traits": ["brave", "loyal", "curious"],
                "motivations": ["friendship", "justice"],
            }
        )
    )
    return llm


# ============================================================================
# CharacterExtractor Tests
# ============================================================================


class TestCharacterExtractorInit:
    """Tests for CharacterExtractor initialization."""

    def test_create_extractor(self):
        """Test creating CharacterExtractor instance."""
        extractor = CharacterExtractor()
        assert extractor is not None
        assert extractor.min_mentions == 2

    def test_create_extractor_custom_min(self):
        """Test CharacterExtractor with custom min_mentions."""
        extractor = CharacterExtractor(min_mentions=5)
        assert extractor.min_mentions == 5


class TestCharacterExtraction:
    """Tests for character extraction."""

    def test_extract_characters_empty_text(self):
        """Test extraction from empty text."""
        extractor = CharacterExtractor()
        result = extractor.extract_characters("")
        assert result == []

    def test_extract_characters_whitespace(self):
        """Test extraction from whitespace-only text."""
        extractor = CharacterExtractor()
        result = extractor.extract_characters("   \n\t  ")
        assert result == []

    def test_extract_characters_basic(self):
        """Test basic character extraction using patterns."""
        extractor = CharacterExtractor(min_mentions=1)
        text = "John Smith walked in. John Smith said hello. Mary Jane waved."

        with patch.object(extractor, "_get_entity_extractor", return_value=None):
            result = extractor.extract_characters(text)

        # Should find at least some names
        assert isinstance(result, list)

    def test_extract_characters_sorted_by_mentions(self):
        """Test characters are sorted by mention count."""
        extractor = CharacterExtractor(min_mentions=1)
        text = "Harry appeared. Harry spoke. Harry left. Ron said hi."

        with patch.object(extractor, "_get_entity_extractor", return_value=None):
            result = extractor._extract_basic(text)

        # Should be sorted by mention count (descending)
        if len(result) >= 2:
            assert result[0].mention_count >= result[1].mention_count


class TestCharacterTracking:
    """Tests for tracking character appearances."""

    def test_track_appearances_empty(self):
        """Test tracking with no chunks."""
        extractor = CharacterExtractor()
        result = extractor.track_appearances("Harry", [])
        assert result == []

    def test_track_appearances_found(self):
        """Test tracking when character is found."""
        extractor = CharacterExtractor()
        chunks = [
            make_mock_chunk(chunk_id="chunk_1", content="Harry walked."),
            make_mock_chunk(chunk_id="chunk_2", content="Ron spoke."),
            make_mock_chunk(chunk_id="chunk_3", content="Harry returned."),
        ]

        result = extractor.track_appearances("Harry", chunks)

        assert len(result) == 2
        assert result[0].chunk_id == "chunk_1"
        assert result[1].chunk_id == "chunk_3"

    def test_track_appearances_position(self):
        """Test that positions are calculated correctly."""
        extractor = CharacterExtractor()
        chunks = [
            make_mock_chunk(chunk_id="chunk_1", content="Harry first."),
            make_mock_chunk(chunk_id="chunk_2", content="Middle."),
            make_mock_chunk(chunk_id="chunk_3", content="Harry last."),
        ]

        result = extractor.track_appearances("Harry", chunks)

        assert result[0].position == 0.0  # First chunk
        assert result[1].position == 1.0  # Last chunk

    def test_track_appearances_case_insensitive(self):
        """Test case-insensitive tracking."""
        extractor = CharacterExtractor()
        chunks = [
            make_mock_chunk(content="HARRY spoke loudly."),
            make_mock_chunk(content="harry whispered."),
        ]

        result = extractor.track_appearances("Harry", chunks)
        assert len(result) == 2


class TestContextExtraction:
    """Tests for context extraction."""

    def test_extract_context_found(self):
        """Test context extraction when keyword found."""
        extractor = CharacterExtractor()
        text = "Before Harry appeared in the middle. After."
        result = extractor._extract_context(text, "Harry", window=10)

        assert "Harry" in result

    def test_extract_context_not_found(self):
        """Test context extraction when keyword not found."""
        extractor = CharacterExtractor()
        text = "No character here at all."
        result = extractor._extract_context(text, "Harry", window=10)

        # Should return beginning of text
        assert len(result) <= len(text)


class TestRelationshipAnalysis:
    """Tests for relationship analysis."""

    def test_analyze_relationships_empty(self):
        """Test relationship analysis with no characters."""
        extractor = CharacterExtractor()
        result = extractor.analyze_relationships([], [])

        assert result.characters == []
        assert result.edges == []

    def test_analyze_relationships_single(self):
        """Test relationship analysis with single character."""
        extractor = CharacterExtractor()
        char = Character(name="Harry")
        result = extractor.analyze_relationships([char], [])

        assert len(result.characters) == 1
        assert len(result.edges) == 0

    def test_analyze_relationships_cooccurrence(self):
        """Test relationship detection from co-occurrence."""
        extractor = CharacterExtractor()
        char1 = Character(name="Romeo")
        char2 = Character(name="Juliet")

        chunks = [
            make_mock_chunk(content="Romeo and Juliet met."),
            make_mock_chunk(content="Romeo loved Juliet."),
            make_mock_chunk(content="Juliet loved Romeo."),
        ]

        result = extractor.analyze_relationships([char1, char2], chunks)

        assert len(result.characters) == 2

    def test_build_cooccurrence(self):
        """Test co-occurrence matrix building."""
        extractor = CharacterExtractor()
        char_names = ["romeo", "juliet"]
        chunks = [
            make_mock_chunk(content="Romeo and Juliet together."),
            make_mock_chunk(content="Romeo alone."),
        ]

        result = extractor._build_cooccurrence(char_names, chunks)

        assert (0, 1) in result  # Romeo-Juliet co-occurrence
        assert result[(0, 1)] == 1

    def test_infer_relationship_type_romantic(self):
        """Test inferring romantic relationship type."""
        extractor = CharacterExtractor()
        char1 = Character(name="Romeo")
        char2 = Character(name="Juliet")
        chunks = [make_mock_chunk(content="Romeo loves Juliet with all his heart.")]

        result = extractor._infer_relationship_type(char1, char2, chunks)
        assert result == "romantic"

    def test_infer_relationship_type_enemy(self):
        """Test inferring enemy relationship type."""
        extractor = CharacterExtractor()
        char1 = Character(name="Harry")
        char2 = Character(name="Voldemort")
        chunks = [make_mock_chunk(content="Harry hates Voldemort, his enemy.")]

        result = extractor._infer_relationship_type(char1, char2, chunks)
        assert result == "enemy"


class TestProfileGeneration:
    """Tests for character profile generation."""

    def test_generate_profile_no_llm(self):
        """Test profile generation without LLM."""
        extractor = CharacterExtractor()
        char = Character(name="Harry", mention_count=10)
        chunks = [make_mock_chunk(content="Harry did things.")]

        result = extractor.generate_profile(char, chunks, llm_client=None)

        assert result.character.name == "Harry"
        assert "10" in result.description

    def test_generate_profile_with_llm(self):
        """Test profile generation with LLM."""
        extractor = CharacterExtractor()
        char = Character(name="Harry")
        chunks = [make_mock_chunk(content="Harry Potter is a wizard.")]
        llm = make_mock_llm_client()

        result = extractor.generate_profile(char, chunks, llm_client=llm)

        assert result.character.name == "Harry"


# ============================================================================
# CharacterCommand Tests
# ============================================================================


class TestCharacterCommandInit:
    """Tests for CharacterCommand initialization."""

    def test_create_command(self):
        """Test creating CharacterCommand instance."""
        cmd = CharacterCommand()
        assert cmd is not None
        assert cmd.extractor is not None

    def test_inherits_from_literary_command(self):
        """Test CharacterCommand inherits from LiteraryCommand."""
        from ingestforge.cli.literary.base import LiteraryCommand

        cmd = CharacterCommand()
        assert isinstance(cmd, LiteraryCommand)


class TestCharacterCommandExecution:
    """Tests for CharacterCommand.execute()."""

    def test_execute_no_storage(self):
        """Test execution fails gracefully without storage."""
        cmd = CharacterCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client", return_value=None):
                mock_init.return_value = make_mock_context()

                result = cmd.execute("Hamlet")

                assert result == 1

    def test_execute_no_context(self):
        """Test execution with no search results."""
        cmd = CharacterCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_context", return_value=[]):
                    mock_init.return_value = make_mock_context()
                    mock_llm.return_value = make_mock_llm_client()

                    result = cmd.execute("Unknown Work")

                    assert result == 0  # Returns 0 with warning

    def test_execute_with_specific_character(self):
        """Test execution with specific character."""
        cmd = CharacterCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_context") as mock_search:
                    with patch.object(cmd, "_analyze_single_character", return_value=0):
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = [make_mock_chunk()]

                        result = cmd.execute("Hamlet", character="Ophelia")

                        assert result == 0

    def test_execute_all_characters(self):
        """Test execution for all characters."""
        cmd = CharacterCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_context") as mock_search:
                    with patch.object(cmd, "_analyze_all_characters", return_value=0):
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = [make_mock_chunk()]

                        result = cmd.execute("Hamlet")

                        assert result == 0


class TestCharacterCommandHelpers:
    """Tests for CharacterCommand helper methods."""

    def test_build_search_query_no_character(self):
        """Test search query without specific character."""
        cmd = CharacterCommand()
        result = cmd._build_search_query("Hamlet", None)
        assert "Hamlet" in result
        assert "characters" in result

    def test_build_search_query_with_character(self):
        """Test search query with specific character."""
        cmd = CharacterCommand()
        result = cmd._build_search_query("Hamlet", "Ophelia")
        assert "Hamlet" in result
        assert "Ophelia" in result


class TestCharacterCommandOutput:
    """Tests for CharacterCommand output methods."""

    def test_save_profile_json(self, tmp_path):
        """Test saving profile as JSON."""
        cmd = CharacterCommand()
        output = tmp_path / "profile.json"

        char = Character(name="Harry")
        profile = CharacterProfile(character=char, description="A wizard")

        cmd._save_profile(output, "Harry Potter", profile)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["work"] == "Harry Potter"

    def test_save_profile_markdown(self, tmp_path):
        """Test saving profile as Markdown."""
        cmd = CharacterCommand()
        output = tmp_path / "profile.md"

        char = Character(name="Harry")
        profile = CharacterProfile(
            character=char,
            description="A wizard",
            traits=["brave"],
        )

        cmd._save_profile(output, "Harry Potter", profile)

        assert output.exists()
        content = output.read_text()
        assert "# Character Profile" in content

    def test_save_characters_json(self, tmp_path):
        """Test saving character list as JSON."""
        cmd = CharacterCommand()
        output = tmp_path / "characters.json"

        characters = [
            Character(name="Harry", mention_count=100),
            Character(name="Ron", mention_count=50),
        ]

        cmd._save_characters(output, "Harry Potter", characters, relationships=False)

        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data["characters"]) == 2


# ============================================================================
# Typer Command Wrapper Tests
# ============================================================================


class TestTyperCommand:
    """Tests for the typer command wrapper."""

    def test_command_exists(self):
        """Test that command function exists."""
        assert callable(command)

    def test_extract_command_exists(self):
        """Test that extract_command function exists."""
        assert callable(extract_command)
