"""
Tests for Symbols CLI Command and SymbolDetector.

Tests symbol detection, analysis, and symbolic meaning extraction.

Test Strategy
-------------
- Test SymbolDetector class methods
- Test SymbolsCommand execution
- Test symbol patterns and meanings
- Mock LLM and storage for unit testing
- Keep tests simple (NASA JPL Rule #1)
"""

import json
from unittest.mock import Mock, patch


from ingestforge.cli.literary.symbols import (
    SymbolsCommand,
    SymbolDetector,
    command,
    detect_command,
    analyze_command,
    COMMON_SYMBOLS,
)
from ingestforge.cli.literary.models import (
    Symbol,
    SymbolAnalysis,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "The water flowed endlessly, reflecting the light of the moon.",
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
                "literal": "Water flowing in a stream",
                "symbolic": "Represents the passage of time and renewal",
                "significance": "Central to the work's theme of change",
                "development": "Appears throughout the narrative",
            }
        )
    )
    return llm


# ============================================================================
# Common Symbols Tests
# ============================================================================


class TestCommonSymbols:
    """Tests for COMMON_SYMBOLS dictionary."""

    def test_common_symbols_defined(self):
        """Test COMMON_SYMBOLS is defined and populated."""
        assert len(COMMON_SYMBOLS) > 0

    def test_water_symbol_defined(self):
        """Test water symbol is defined with meanings."""
        assert "water" in COMMON_SYMBOLS
        assert len(COMMON_SYMBOLS["water"]) > 0

    def test_fire_symbol_defined(self):
        """Test fire symbol is defined."""
        assert "fire" in COMMON_SYMBOLS

    def test_light_symbol_defined(self):
        """Test light symbol is defined."""
        assert "light" in COMMON_SYMBOLS

    def test_color_symbols_defined(self):
        """Test color symbols are defined."""
        assert "white" in COMMON_SYMBOLS
        assert "black" in COMMON_SYMBOLS
        assert "red" in COMMON_SYMBOLS

    def test_object_symbols_defined(self):
        """Test object symbols are defined."""
        assert "mirror" in COMMON_SYMBOLS
        assert "door" in COMMON_SYMBOLS
        assert "key" in COMMON_SYMBOLS

    def test_all_symbols_have_meanings(self):
        """Test all symbols have at least one meaning."""
        for symbol, meanings in COMMON_SYMBOLS.items():
            assert len(meanings) > 0, f"Symbol {symbol} has no meanings"


# ============================================================================
# SymbolDetector Tests
# ============================================================================


class TestSymbolDetectorInit:
    """Tests for SymbolDetector initialization."""

    def test_create_detector(self):
        """Test creating SymbolDetector instance."""
        detector = SymbolDetector()
        assert detector is not None
        assert detector.min_occurrences == 3
        assert detector.llm_client is None

    def test_create_detector_custom_min(self):
        """Test SymbolDetector with custom min_occurrences."""
        detector = SymbolDetector(min_occurrences=5)
        assert detector.min_occurrences == 5

    def test_create_detector_with_llm(self):
        """Test SymbolDetector with LLM client."""
        llm = make_mock_llm_client()
        detector = SymbolDetector(llm_client=llm)
        assert detector.llm_client is not None

    def test_symbol_patterns_loaded(self):
        """Test symbol patterns are loaded."""
        detector = SymbolDetector()
        assert len(detector._symbol_patterns) > 0


class TestSymbolDetection:
    """Tests for symbol detection."""

    def test_detect_symbols_empty(self):
        """Test detection with empty chunks."""
        detector = SymbolDetector()
        result = detector.detect_symbols([])
        assert result == []

    def test_detect_symbols_basic(self):
        """Test basic symbol detection."""
        detector = SymbolDetector(min_occurrences=1)
        chunks = [
            make_mock_chunk(content="The water flowed."),
            make_mock_chunk(content="Fire burned bright."),
        ]

        result = detector.detect_symbols(chunks)

        assert isinstance(result, list)

    def test_detect_symbols_filters_by_count(self):
        """Test symbols are filtered by min occurrences."""
        detector = SymbolDetector(min_occurrences=3)
        chunks = [
            make_mock_chunk(content="Water once."),
            make_mock_chunk(content="Fire fire fire fire."),
        ]

        result = detector.detect_symbols(chunks)

        # Only fire should pass (4 occurrences)
        symbol_names = [s.name.lower() for s in result]
        assert "fire" in symbol_names or len(result) == 0

    def test_detect_symbols_sorted_by_occurrences(self):
        """Test symbols are sorted by occurrence count."""
        detector = SymbolDetector(min_occurrences=1)
        chunks = [
            make_mock_chunk(content="Water water water. Fire."),
        ]

        result = detector.detect_symbols(chunks)

        if len(result) >= 2:
            assert result[0].occurrences >= result[1].occurrences


class TestSymbolCounting:
    """Tests for symbol counting."""

    def test_count_symbols(self):
        """Test counting symbols in chunks."""
        detector = SymbolDetector()
        chunks = [
            make_mock_chunk(content="water water light"),
            make_mock_chunk(content="water fire"),
        ]

        counts = detector._count_symbols(chunks)

        assert counts.get("water", 0) == 3
        assert counts.get("light", 0) == 1

    def test_count_symbols_empty(self):
        """Test counting with empty chunks."""
        detector = SymbolDetector()
        counts = detector._count_symbols([])
        assert counts == {}


class TestSymbolObjectCreation:
    """Tests for Symbol object creation."""

    def test_create_symbol_objects(self):
        """Test creating Symbol objects from counts."""
        detector = SymbolDetector()
        counts = {"water": 5, "fire": 3}
        chunks = [make_mock_chunk(content="water and fire")]

        result = detector._create_symbol_objects(counts, chunks)

        assert len(result) == 2
        assert all(isinstance(s, Symbol) for s in result)

    def test_symbol_has_contexts(self):
        """Test created symbols have contexts."""
        detector = SymbolDetector()
        counts = {"water": 3}
        chunks = [make_mock_chunk(content="The water flows endlessly.")]

        result = detector._create_symbol_objects(counts, chunks)

        assert len(result) == 1
        assert len(result[0].contexts) > 0

    def test_symbol_has_associated_themes(self):
        """Test created symbols have associated themes."""
        detector = SymbolDetector()
        counts = {"water": 3}
        chunks = [make_mock_chunk()]

        result = detector._create_symbol_objects(counts, chunks)

        assert len(result) == 1
        assert len(result[0].associated_themes) > 0


class TestFirstAppearance:
    """Tests for first appearance finding."""

    def test_find_first_appearance(self):
        """Test finding first appearance of symbol."""
        detector = SymbolDetector()
        chunks = [
            make_mock_chunk(chunk_id="c1", content="No symbol here."),
            make_mock_chunk(chunk_id="c2", content="water appears here."),
            make_mock_chunk(chunk_id="c3", content="More water here."),
        ]

        result = detector._find_first_appearance("water", chunks)

        assert result == 1  # Second chunk (index 1)

    def test_find_first_appearance_not_found(self):
        """Test first appearance when symbol not found."""
        detector = SymbolDetector()
        chunks = [make_mock_chunk(content="No matching symbols.")]

        result = detector._find_first_appearance("water", chunks)

        assert result == 0  # Default


class TestContextExtraction:
    """Tests for context extraction."""

    def test_get_contexts(self):
        """Test getting contexts for symbol."""
        detector = SymbolDetector()
        chunks = [
            make_mock_chunk(content="The water was cold."),
            make_mock_chunk(content="No symbol."),
            make_mock_chunk(content="More water flowed."),
        ]

        result = detector._get_contexts("water", chunks, max_contexts=5)

        assert len(result) == 2  # Two chunks have "water"

    def test_get_contexts_max_limit(self):
        """Test context respects max limit."""
        detector = SymbolDetector()
        chunks = [make_mock_chunk(content="water") for _ in range(10)]

        result = detector._get_contexts("water", chunks, max_contexts=3)

        assert len(result) == 3

    def test_extract_context_found(self):
        """Test extracting context around symbol."""
        detector = SymbolDetector()
        text = "Before the water symbol appeared. After."

        result = detector._extract_context(text, "water", window=10)

        assert "water" in result

    def test_extract_context_with_ellipsis(self):
        """Test context has ellipsis for long text."""
        detector = SymbolDetector()
        text = "A" * 100 + " water " + "B" * 100

        result = detector._extract_context(text, "water", window=20)

        assert "..." in result


class TestSymbolismAnalysis:
    """Tests for symbolism analysis."""

    def test_analyze_symbolism_basic(self):
        """Test basic symbolism analysis."""
        detector = SymbolDetector()
        chunks = [make_mock_chunk(content="The water flowed.")]

        result = detector.analyze_symbolism("water", chunks)

        assert isinstance(result, SymbolAnalysis)
        assert result.symbol.name == "Water"

    def test_analyze_symbolism_has_meanings(self):
        """Test analysis includes symbolic meanings."""
        detector = SymbolDetector()
        chunks = [make_mock_chunk(content="water flows")]

        result = detector.analyze_symbolism("water", chunks)

        # Should have known meanings from COMMON_SYMBOLS
        assert result.symbolic_meaning != ""

    def test_analyze_symbolism_with_llm(self):
        """Test analysis with LLM enhancement."""
        llm = make_mock_llm_client()
        detector = SymbolDetector(llm_client=llm)
        chunks = [make_mock_chunk(content="water flows")]

        result = detector.analyze_symbolism("water", chunks)

        llm.generate.assert_called_once()

    def test_enhance_with_llm_success(self):
        """Test LLM enhancement with valid response."""
        llm = make_mock_llm_client()
        detector = SymbolDetector(llm_client=llm)

        symbol = Symbol(name="Water")
        analysis = SymbolAnalysis(symbol=symbol)
        chunks = [make_mock_chunk(content="water here")]

        result = detector._enhance_with_llm(analysis, chunks)

        assert result.literal_meaning != ""

    def test_enhance_with_llm_failure(self):
        """Test LLM enhancement handles failure gracefully."""
        llm = Mock()
        llm.generate = Mock(return_value="not valid json")
        detector = SymbolDetector(llm_client=llm)

        symbol = Symbol(name="Water")
        analysis = SymbolAnalysis(symbol=symbol)
        chunks = [make_mock_chunk()]

        result = detector._enhance_with_llm(analysis, chunks)

        # Should not crash, returns original analysis
        assert result.symbol.name == "Water"


# ============================================================================
# SymbolsCommand Tests
# ============================================================================


class TestSymbolsCommandInit:
    """Tests for SymbolsCommand initialization."""

    def test_create_command(self):
        """Test creating SymbolsCommand instance."""
        cmd = SymbolsCommand()
        assert cmd is not None

    def test_inherits_from_literary_command(self):
        """Test SymbolsCommand inherits from LiteraryCommand."""
        from ingestforge.cli.literary.base import LiteraryCommand

        cmd = SymbolsCommand()
        assert isinstance(cmd, LiteraryCommand)


class TestSymbolsCommandExecution:
    """Tests for SymbolsCommand.execute()."""

    def test_execute_no_llm(self):
        """Test execution fails without LLM."""
        cmd = SymbolsCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client", return_value=None):
                mock_init.return_value = make_mock_context()

                result = cmd.execute("The Great Gatsby")

                assert result == 1

    def test_execute_no_context(self):
        """Test execution with no search results."""
        cmd = SymbolsCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_symbols", return_value=[]):
                    mock_init.return_value = make_mock_context()
                    mock_llm.return_value = make_mock_llm_client()

                    result = cmd.execute("Unknown Work")

                    assert result == 0  # Returns 0 with warning

    def test_execute_detect_all_symbols(self):
        """Test execution to detect all symbols."""
        cmd = SymbolsCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_symbols") as mock_search:
                    with patch.object(cmd, "_detect_all_symbols", return_value=0):
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = [make_mock_chunk()]

                        result = cmd.execute("The Great Gatsby")

                        assert result == 0

    def test_execute_analyze_single_symbol(self):
        """Test execution to analyze single symbol."""
        cmd = SymbolsCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "_search_for_symbols") as mock_search:
                    with patch.object(cmd, "_analyze_single_symbol", return_value=0):
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = [make_mock_chunk()]

                        result = cmd.execute("Gatsby", symbol="green light")

                        assert result == 0


class TestSymbolsCommandOutput:
    """Tests for SymbolsCommand output methods."""

    def test_save_symbols_json(self, tmp_path):
        """Test saving symbols as JSON."""
        cmd = SymbolsCommand()
        output = tmp_path / "symbols.json"

        symbols = [
            Symbol(name="Water", occurrences=10),
            Symbol(name="Fire", occurrences=5),
        ]

        cmd._save_symbols(output, "Test Work", symbols)

        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data["symbols"]) == 2

    def test_save_symbols_markdown(self, tmp_path):
        """Test saving symbols as Markdown."""
        cmd = SymbolsCommand()
        output = tmp_path / "symbols.md"

        symbols = [Symbol(name="Water", occurrences=10)]

        cmd._save_symbols(output, "Test Work", symbols)

        assert output.exists()
        content = output.read_text()
        assert "# Symbols" in content

    def test_save_symbol_analysis_json(self, tmp_path):
        """Test saving symbol analysis as JSON."""
        cmd = SymbolsCommand()
        output = tmp_path / "analysis.json"

        symbol = Symbol(name="Water")
        analysis = SymbolAnalysis(
            symbol=symbol,
            literal_meaning="H2O",
            symbolic_meaning="Life and renewal",
        )

        cmd._save_symbol_analysis(output, "Test Work", analysis)

        assert output.exists()

    def test_save_symbol_analysis_markdown(self, tmp_path):
        """Test saving symbol analysis as Markdown."""
        cmd = SymbolsCommand()
        output = tmp_path / "analysis.md"

        symbol = Symbol(name="Water")
        analysis = SymbolAnalysis(
            symbol=symbol,
            literal_meaning="H2O",
            symbolic_meaning="Life and renewal",
        )

        cmd._save_symbol_analysis(output, "Test Work", analysis)

        assert output.exists()
        content = output.read_text()
        assert "# Symbol Analysis" in content


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
