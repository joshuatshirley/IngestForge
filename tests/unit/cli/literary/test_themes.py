"""
Tests for Themes CLI Command and ThemeDetector.

Tests theme detection, evidence finding, and development tracking.

Test Strategy
-------------
- Test ThemeDetector class methods
- Test ThemesCommand execution
- Mock LLM and storage for unit testing
- Keep tests simple (NASA JPL Rule #1)
"""

import json
from unittest.mock import Mock, patch


from ingestforge.cli.literary.themes import (
    ThemesCommand,
    ThemeDetector,
    command,
    detect_command,
    analyze_command,
    COMMON_THEMES,
)
from ingestforge.cli.literary.models import (
    Evidence,
    Theme,
    ThemeArc,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "The theme of love and death pervades this work.",
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
            [
                {"name": "love", "confidence": 0.9},
                {"name": "death", "confidence": 0.8},
            ]
        )
    )
    return llm


# ============================================================================
# ThemeDetector Tests
# ============================================================================


class TestThemeDetectorInit:
    """Tests for ThemeDetector initialization."""

    def test_create_detector(self):
        """Test creating ThemeDetector instance."""
        detector = ThemeDetector()
        assert detector is not None
        assert detector.llm_client is None

    def test_create_detector_with_llm(self):
        """Test ThemeDetector with LLM client."""
        llm = make_mock_llm_client()
        detector = ThemeDetector(llm_client=llm)
        assert detector.llm_client is not None

    def test_theme_keywords_built(self):
        """Test theme keywords are initialized."""
        detector = ThemeDetector()
        assert len(detector._theme_keywords) > 0
        assert "love" in detector._theme_keywords
        assert "death" in detector._theme_keywords


class TestThemeDetection:
    """Tests for theme detection."""

    def test_detect_themes_empty(self):
        """Test detection with empty chunks."""
        detector = ThemeDetector()
        result = detector.detect_themes([])
        assert result == []

    def test_detect_themes_basic(self):
        """Test basic theme detection."""
        detector = ThemeDetector()
        chunks = [
            make_mock_chunk(content="Love conquers all in this story of passion."),
            make_mock_chunk(content="The lover's heart beats with passion."),
        ]

        result = detector.detect_themes(chunks, top_n=5)

        assert len(result) > 0
        assert all(isinstance(t, Theme) for t in result)

    def test_detect_themes_sorted_by_confidence(self):
        """Test themes are sorted by confidence."""
        detector = ThemeDetector()
        chunks = [
            make_mock_chunk(content="Love love love. Death death death death death."),
        ]

        result = detector.detect_themes(chunks, top_n=5)

        if len(result) >= 2:
            assert result[0].confidence >= result[1].confidence

    def test_detect_themes_with_llm(self):
        """Test theme detection with LLM enhancement."""
        llm = make_mock_llm_client()
        detector = ThemeDetector(llm_client=llm)
        chunks = [make_mock_chunk()]

        result = detector.detect_themes(chunks, top_n=3)

        # LLM should be called
        llm.generate.assert_called_once()


class TestKeywordScoring:
    """Tests for keyword-based theme scoring."""

    def test_score_themes_by_keywords(self):
        """Test keyword scoring."""
        detector = ThemeDetector()
        text = "love love love death"

        scores = detector._score_themes_by_keywords(text)

        assert "love" in scores
        assert "death" in scores
        assert scores["love"] > scores["death"]  # More occurrences

    def test_score_themes_empty_text(self):
        """Test scoring with empty text."""
        detector = ThemeDetector()
        scores = detector._score_themes_by_keywords("")

        assert isinstance(scores, dict)


class TestLLMDetection:
    """Tests for LLM-based detection."""

    def test_detect_with_llm_success(self):
        """Test LLM detection with valid response."""
        llm = Mock()
        llm.generate = Mock(
            return_value=json.dumps(
                [
                    {"name": "redemption", "confidence": 0.85},
                ]
            )
        )

        detector = ThemeDetector(llm_client=llm)
        chunks = [make_mock_chunk()]

        result = detector._detect_with_llm(chunks, top_n=5)

        assert "redemption" in result

    def test_detect_with_llm_invalid_json(self):
        """Test LLM detection with invalid JSON."""
        llm = Mock()
        llm.generate = Mock(return_value="not valid json")

        detector = ThemeDetector(llm_client=llm)
        chunks = [make_mock_chunk()]

        result = detector._detect_with_llm(chunks, top_n=5)

        assert result == {}  # Empty on failure

    def test_merge_theme_scores(self):
        """Test merging keyword and LLM scores."""
        detector = ThemeDetector()

        keyword_scores = {"love": 5.0, "death": 3.0}
        llm_scores = {"love": 0.9, "redemption": 0.8}

        result = detector._merge_theme_scores(keyword_scores, llm_scores)

        assert "love" in result
        assert "death" in result
        assert "redemption" in result


class TestEvidenceFinding:
    """Tests for evidence finding."""

    def test_find_evidence_empty(self):
        """Test evidence finding with no chunks."""
        detector = ThemeDetector()
        result = detector.find_evidence("love", [])
        assert result == []

    def test_find_evidence_found(self):
        """Test evidence finding when theme present."""
        detector = ThemeDetector()
        chunks = [
            make_mock_chunk(chunk_id="c1", content="True love conquers all."),
            make_mock_chunk(chunk_id="c2", content="No love here."),
        ]

        result = detector.find_evidence("love", chunks, max_evidence=10)

        assert len(result) > 0
        assert all(isinstance(e, Evidence) for e in result)

    def test_find_evidence_max_limit(self):
        """Test evidence respects max limit."""
        detector = ThemeDetector()
        chunks = [
            make_mock_chunk(content="Love love love love love.") for _ in range(20)
        ]

        result = detector.find_evidence("love", chunks, max_evidence=5)

        assert len(result) <= 5

    def test_find_evidence_position(self):
        """Test evidence positions are calculated."""
        detector = ThemeDetector()
        chunks = [
            make_mock_chunk(chunk_id="c1", content="Love at start."),
            make_mock_chunk(chunk_id="c2", content="No theme here."),
            make_mock_chunk(chunk_id="c3", content="Love at end."),
        ]

        result = detector.find_evidence("love", chunks)

        if len(result) >= 2:
            assert result[0].position < result[1].position


class TestQuoteExtraction:
    """Tests for quote extraction."""

    def test_extract_quote_found(self):
        """Test quote extraction when keyword found."""
        detector = ThemeDetector()
        text = "Before the theme of love appears. After the theme."

        result = detector._extract_quote(text, "love", window=20)

        assert "love" in result

    def test_extract_quote_with_ellipsis(self):
        """Test quote extraction adds ellipsis."""
        detector = ThemeDetector()
        text = "A" * 100 + " love " + "B" * 100

        result = detector._extract_quote(text, "love", window=30)

        assert "..." in result


class TestThemeDevelopment:
    """Tests for theme development tracking."""

    def test_track_theme_development(self):
        """Test tracking theme development."""
        detector = ThemeDetector()
        chunks = [
            make_mock_chunk(content="Love appears here."),
            make_mock_chunk(content="No theme."),
            make_mock_chunk(content="Love returns strongly, love love."),
        ]

        result = detector.track_theme_development("love", chunks)

        assert isinstance(result, ThemeArc)
        assert len(result.development) == 3
        assert result.theme.name == "Love"

    def test_track_development_positions(self):
        """Test development point positions."""
        detector = ThemeDetector()
        chunks = [make_mock_chunk() for _ in range(5)]

        result = detector.track_theme_development("love", chunks)

        positions = [p.position for p in result.development]
        assert positions[0] == 0.0
        assert positions[-1] == 1.0


class TestThemeComparison:
    """Tests for theme comparison."""

    def test_compare_themes(self):
        """Test comparing themes."""
        detector = ThemeDetector()
        themes = [
            Theme(name="love"),
            Theme(name="death"),
        ]

        result = detector.compare_themes(themes)

        assert len(result.themes) == 2
        assert result.summary != ""

    def test_compare_themes_empty(self):
        """Test comparing empty theme list."""
        detector = ThemeDetector()
        result = detector.compare_themes([])

        assert "No themes" in result.summary


# ============================================================================
# ThemesCommand Tests
# ============================================================================


class TestThemesCommandInit:
    """Tests for ThemesCommand initialization."""

    def test_create_command(self):
        """Test creating ThemesCommand instance."""
        cmd = ThemesCommand()
        assert cmd is not None

    def test_inherits_from_literary_command(self):
        """Test ThemesCommand inherits from LiteraryCommand."""
        from ingestforge.cli.literary.base import LiteraryCommand

        cmd = ThemesCommand()
        assert isinstance(cmd, LiteraryCommand)


class TestThemesCommandExecution:
    """Tests for ThemesCommand.execute()."""

    def test_execute_no_llm(self):
        """Test execution fails without LLM."""
        cmd = ThemesCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client", return_value=None):
                mock_init.return_value = make_mock_context()

                result = cmd.execute("Hamlet")

                assert result == 1

    def test_execute_no_context(self):
        """Test execution with no search results."""
        cmd = ThemesCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "search_literary_context", return_value=[]):
                    mock_init.return_value = make_mock_context()
                    mock_llm.return_value = make_mock_llm_client()

                    result = cmd.execute("Unknown Work")

                    assert result == 0  # Returns 0 with warning

    def test_execute_detect_all_themes(self):
        """Test execution to detect all themes."""
        cmd = ThemesCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "search_literary_context") as mock_search:
                    with patch.object(cmd, "_detect_all_themes", return_value=0):
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = [make_mock_chunk()]

                        result = cmd.execute("Hamlet")

                        assert result == 0

    def test_execute_analyze_single_theme(self):
        """Test execution to analyze single theme."""
        cmd = ThemesCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "search_literary_context") as mock_search:
                    with patch.object(cmd, "_analyze_single_theme", return_value=0):
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = [make_mock_chunk()]

                        result = cmd.execute("Hamlet", theme="revenge")

                        assert result == 0


class TestThemesCommandOutput:
    """Tests for ThemesCommand output methods."""

    def test_save_themes_json(self, tmp_path):
        """Test saving themes as JSON."""
        cmd = ThemesCommand()
        output = tmp_path / "themes.json"

        themes = [
            Theme(name="Love", confidence=0.9),
            Theme(name="Death", confidence=0.8),
        ]

        cmd._save_themes(output, "Hamlet", themes, {})

        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data["themes"]) == 2

    def test_save_themes_markdown(self, tmp_path):
        """Test saving themes as Markdown."""
        cmd = ThemesCommand()
        output = tmp_path / "themes.md"

        themes = [Theme(name="Love", confidence=0.9)]

        cmd._save_themes(output, "Hamlet", themes, {})

        assert output.exists()
        content = output.read_text()
        assert "# Themes" in content

    def test_save_theme_analysis_json(self, tmp_path):
        """Test saving theme analysis as JSON."""
        cmd = ThemesCommand()
        output = tmp_path / "analysis.json"

        theme = Theme(name="Revenge")
        arc = ThemeArc(theme=theme)
        evidence = [Evidence(quote="Test", chunk_id="c1")]

        cmd._save_theme_analysis(output, "Hamlet", arc, evidence)

        assert output.exists()


class TestMiniAsciiChart:
    """Tests for the mini ASCII chart method."""

    def test_mini_ascii_chart(self):
        """Test ASCII chart generation."""
        cmd = ThemesCommand()
        values = [0.2, 0.5, 0.8, 0.3]

        result = cmd._mini_ascii_chart(values, width=4)

        assert "[" in result
        assert "]" in result

    def test_mini_ascii_chart_empty(self):
        """Test ASCII chart with empty values."""
        cmd = ThemesCommand()
        result = cmd._mini_ascii_chart([])

        assert "No data" in result


# ============================================================================
# Typer Command Tests
# ============================================================================


class TestTyperCommands:
    """Tests for typer command wrappers."""

    def test_command_exists(self):
        """Test main command exists."""
        assert callable(command)

    def test_detect_command_exists(self):
        """Test detect command exists."""
        assert callable(detect_command)

    def test_analyze_command_exists(self):
        """Test analyze command exists."""
        assert callable(analyze_command)


class TestCommonThemes:
    """Tests for COMMON_THEMES constant."""

    def test_common_themes_defined(self):
        """Test COMMON_THEMES is defined."""
        assert len(COMMON_THEMES) > 0

    def test_common_themes_contains_love(self):
        """Test love is a common theme."""
        assert "love" in COMMON_THEMES

    def test_common_themes_contains_death(self):
        """Test death is a common theme."""
        assert "death" in COMMON_THEMES
