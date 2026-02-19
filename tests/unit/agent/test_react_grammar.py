"""Tests for ReAct Grammar.

Tests the GBNF grammar generation for constrained LLM output."""

from __future__ import annotations


from ingestforge.agent.react_grammar import (
    build_react_grammar,
    get_react_grammar,
    ReActGrammarHelper,
    REACT_GRAMMAR_GENERIC,
    REACT_GRAMMAR_LENIENT,
)


class TestBuildReactGrammar:
    """Tests for build_react_grammar function."""

    def test_build_with_no_tools(self) -> None:
        """Test grammar with no tool restriction."""
        grammar = build_react_grammar(None)

        assert "root ::=" in grammar
        assert "thought ::=" in grammar
        assert "action ::=" in grammar
        assert "action-input ::=" in grammar
        assert "identifier" in grammar  # Allows any identifier

    def test_build_with_tools(self) -> None:
        """Test grammar with specific tools."""
        tools = ["search_web", "search_docs", "FINISH"]
        grammar = build_react_grammar(tools)

        assert '"search_web"' in grammar
        assert '"search_docs"' in grammar
        assert '"FINISH"' in grammar

    def test_build_adds_finish_if_missing(self) -> None:
        """Test FINISH is added if not in tools list."""
        tools = ["search_web", "analyze"]
        grammar = build_react_grammar(tools)

        assert '"FINISH"' in grammar

    def test_build_escapes_special_chars(self) -> None:
        """Test tool names with special chars are escaped."""
        tools = ['tool_with_"quote"']
        grammar = build_react_grammar(tools)

        # Should be escaped
        assert '\\"' in grammar


class TestGetReactGrammar:
    """Tests for get_react_grammar function."""

    def test_get_strict_with_tools(self) -> None:
        """Test strict grammar with tools."""
        grammar = get_react_grammar(tools=["search"], strict=True)

        assert '"search"' in grammar
        assert "json-object" in grammar

    def test_get_lenient(self) -> None:
        """Test lenient grammar."""
        grammar = get_react_grammar(strict=False)

        assert grammar == REACT_GRAMMAR_LENIENT

    def test_get_generic_no_tools(self) -> None:
        """Test generic grammar when no tools specified."""
        grammar = get_react_grammar(tools=None, strict=True)

        assert "identifier" in grammar


class TestReActGrammarHelper:
    """Tests for ReActGrammarHelper class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        helper = ReActGrammarHelper()

        assert helper.tools == []
        assert helper._strict is True

    def test_set_tools(self) -> None:
        """Test setting tools."""
        helper = ReActGrammarHelper()
        helper.set_tools(["search", "analyze"])

        assert helper.tools == ["search", "analyze"]

    def test_get_grammar_caches(self) -> None:
        """Test grammar is cached."""
        helper = ReActGrammarHelper()
        helper.set_tools(["search"])

        grammar1 = helper.get_grammar()
        grammar2 = helper.get_grammar()

        assert grammar1 is grammar2  # Same object (cached)

    def test_set_tools_invalidates_cache(self) -> None:
        """Test setting tools invalidates cache."""
        helper = ReActGrammarHelper()
        helper.set_tools(["search"])
        grammar1 = helper.get_grammar()

        helper.set_tools(["analyze"])
        grammar2 = helper.get_grammar()

        assert grammar1 != grammar2

    def test_get_grammar_with_tools(self) -> None:
        """Test grammar includes set tools."""
        helper = ReActGrammarHelper()
        helper.set_tools(["my_tool", "other_tool"])

        grammar = helper.get_grammar()

        assert '"my_tool"' in grammar
        assert '"other_tool"' in grammar


class TestPrebuiltGrammars:
    """Tests for pre-built grammar constants."""

    def test_generic_grammar_exists(self) -> None:
        """Test generic grammar is defined."""
        assert REACT_GRAMMAR_GENERIC is not None
        assert len(REACT_GRAMMAR_GENERIC) > 0

    def test_lenient_grammar_exists(self) -> None:
        """Test lenient grammar is defined."""
        assert REACT_GRAMMAR_LENIENT is not None
        assert len(REACT_GRAMMAR_LENIENT) > 0

    def test_generic_has_required_rules(self) -> None:
        """Test generic grammar has all required rules."""
        assert "root ::=" in REACT_GRAMMAR_GENERIC
        assert "thought" in REACT_GRAMMAR_GENERIC
        assert "action" in REACT_GRAMMAR_GENERIC
        assert "json" in REACT_GRAMMAR_GENERIC.lower()

    def test_lenient_has_required_rules(self) -> None:
        """Test lenient grammar has all required rules."""
        assert "root ::=" in REACT_GRAMMAR_LENIENT
        assert "thought" in REACT_GRAMMAR_LENIENT
        assert "action" in REACT_GRAMMAR_LENIENT


class TestGrammarValidity:
    """Tests for grammar syntax validity."""

    def test_grammar_generates_without_error(self) -> None:
        """Test grammar generates without raising exceptions."""
        # Should not raise any exceptions
        grammar = build_react_grammar(["test"])
        assert len(grammar) > 100  # Non-trivial output

    def test_grammar_has_valid_structure(self) -> None:
        """Test grammar has expected GBNF structure."""
        grammar = build_react_grammar(["search", "analyze"])

        # Should have rule definitions
        assert "::=" in grammar
        # Should have our main rules
        assert "root ::=" in grammar
        assert "thought ::=" in grammar
        assert "action ::=" in grammar
        # Should have the tools
        assert '"search"' in grammar
        assert '"analyze"' in grammar
