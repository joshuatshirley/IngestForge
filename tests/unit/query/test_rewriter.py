"""Tests for query rewriting functionality.

Tests cover:
- Query expansion strategy
- Query simplification strategy
- Query clarification strategy
- Stop word removal
- Edge cases and boundary conditions
"""

import pytest
from ingestforge.query.rewriter import (
    QueryRewriter,
    rewrite_query,
    multi_strategy_rewrite,
)


class TestQueryRewriter:
    """Test suite for QueryRewriter class."""

    @pytest.fixture
    def rewriter(self):
        """Create QueryRewriter instance."""
        return QueryRewriter()

    # Initialization Tests
    def test_initialization(self, rewriter):
        """Test rewriter initializes with stop words."""
        assert hasattr(rewriter, "stop_words")
        assert isinstance(rewriter.stop_words, set)
        assert len(rewriter.stop_words) > 0

    def test_stop_words_loaded(self, rewriter):
        """Test common stop words are loaded."""
        assert "the" in rewriter.stop_words
        assert "a" in rewriter.stop_words
        assert "and" in rewriter.stop_words
        assert "or" in rewriter.stop_words

    # Rewrite Method Tests
    def test_rewrite_returns_dict(self, rewriter):
        """Test rewrite returns dictionary with expected keys."""
        result = rewriter.rewrite("test query")
        assert isinstance(result, dict)
        assert "original" in result
        assert "strategy" in result
        assert "rewritten" in result
        assert "count" in result

    def test_rewrite_preserves_original(self, rewriter):
        """Test original query is preserved in result."""
        query = "test query"
        result = rewriter.rewrite(query)
        assert result["original"] == query

    def test_rewrite_includes_strategy(self, rewriter):
        """Test strategy is included in result."""
        result = rewriter.rewrite("test", strategy="expand")
        assert result["strategy"] == "expand"

    def test_rewrite_count_matches_length(self, rewriter):
        """Test count matches length of rewritten queries."""
        result = rewriter.rewrite("test query")
        assert result["count"] == len(result["rewritten"])

    # Expand Strategy Tests
    def test_expand_strategy(self, rewriter):
        """Test expand strategy generates variations."""
        result = rewriter.rewrite("machine learning", strategy="expand")
        assert len(result["rewritten"]) > 1
        assert result["rewritten"][0] == "machine learning"

    def test_expand_adds_question_variations(self, rewriter):
        """Test expand adds question variations."""
        rewritten = rewriter._expand_query("Python")
        assert any("What is" in q for q in rewritten)
        assert any("Explain" in q for q in rewritten)
        assert any("How does" in q for q in rewritten)

    def test_expand_skips_questions_if_present(self, rewriter):
        """Test expand handles queries with existing question marks."""
        rewritten = rewriter._expand_query("What is Python?")
        # Should not add "What is What is Python?"
        assert rewritten[0] == "What is Python?"

    def test_expand_adds_context_variations(self, rewriter):
        """Test expand adds context variations."""
        rewritten = rewriter._expand_query("Python")
        assert any("overview" in q for q in rewritten)
        # Note: rewriter adds "examples" and "definition" but they may not be in the first 5
        assert len(rewritten) > 1

    def test_expand_limits_to_five(self, rewriter):
        """Test expand limits to 5 variations."""
        rewritten = rewriter._expand_query("test")
        assert len(rewritten) <= 5

    # Simplify Strategy Tests
    def test_simplify_strategy(self, rewriter):
        """Test simplify strategy removes stop words."""
        result = rewriter.rewrite("what is the best way to learn", strategy="simplify")
        assert len(result["rewritten"]) >= 1

    def test_simplify_removes_stop_words(self, rewriter):
        """Test simplify removes common stop words."""
        rewritten = rewriter._simplify_query("what is the best way")
        simplified = rewritten[-1] if len(rewritten) > 1 else rewritten[0]
        # Stop words should be removed
        assert "the" not in simplified.split()
        assert "is" not in simplified.split()

    def test_simplify_preserves_key_words(self, rewriter):
        """Test simplify preserves key words."""
        rewritten = rewriter._simplify_query("what is machine learning")
        simplified = rewritten[-1] if len(rewritten) > 1 else rewritten[0]
        assert "machine" in simplified.lower()
        assert "learning" in simplified.lower()

    def test_simplify_all_stop_words(self, rewriter):
        """Test simplify handles query with only stop words."""
        rewritten = rewriter._simplify_query("the a an")
        assert len(rewritten) == 1
        assert rewritten[0] == "the a an"

    def test_simplify_no_stop_words(self, rewriter):
        """Test simplify handles query without stop words."""
        query = "machine learning"
        rewritten = rewriter._simplify_query(query)
        # Should return original if no changes
        assert query in [r.lower() for r in rewritten]

    def test_simplify_returns_both_versions(self, rewriter):
        """Test simplify returns original and simplified if different."""
        rewritten = rewriter._simplify_query("what is Python")
        assert len(rewritten) == 2
        assert rewritten[0] == "what is Python"
        assert "python" in rewritten[1].lower()

    # Clarify Strategy Tests
    def test_clarify_strategy(self, rewriter):
        """Test clarify strategy adds specificity."""
        result = rewriter.rewrite("AI", strategy="clarify")
        assert len(result["rewritten"]) >= 1

    def test_clarify_short_query(self, rewriter):
        """Test clarify adds context for short queries."""
        rewritten = rewriter._clarify_query("AI")
        assert len(rewritten) > 1
        assert any("detailed explanation" in q for q in rewritten)
        assert any("key concepts" in q for q in rewritten)

    def test_clarify_question_query(self, rewriter):
        """Test clarify adds depth for questions."""
        rewritten = rewriter._clarify_query("What is AI")
        assert len(rewritten) > 1
        assert any("in detail" in q for q in rewritten)

    def test_clarify_long_query(self, rewriter):
        """Test clarify handles longer queries."""
        query = "machine learning algorithms for natural language processing"
        rewritten = rewriter._clarify_query(query)
        assert query in rewritten

    def test_clarify_limits_to_three(self, rewriter):
        """Test clarify limits to 3 variations."""
        rewritten = rewriter._clarify_query("test")
        assert len(rewritten) <= 3

    # Unknown Strategy Tests
    def test_unknown_strategy(self, rewriter):
        """Test unknown strategy returns original query."""
        result = rewriter.rewrite("test", strategy="unknown")
        assert result["rewritten"] == ["test"]
        assert result["count"] == 1

    def test_invalid_strategy(self, rewriter):
        """Test invalid strategy handled gracefully."""
        result = rewriter.rewrite("test", strategy="invalid")
        assert result["original"] == "test"
        assert len(result["rewritten"]) >= 1

    # Module-Level Functions Tests
    def test_rewrite_query_function(self):
        """Test module-level rewrite_query function."""
        result = rewrite_query("test query", strategy="expand")
        assert isinstance(result, dict)
        assert "original" in result
        assert "rewritten" in result

    def test_rewrite_query_default_strategy(self):
        """Test rewrite_query uses expand by default."""
        result = rewrite_query("test")
        assert result["strategy"] == "expand"

    def test_multi_strategy_rewrite_function(self):
        """Test multi_strategy_rewrite applies all strategies."""
        result = multi_strategy_rewrite("test query")
        assert isinstance(result, dict)
        assert "original" in result
        assert "expand" in result
        assert "simplify" in result
        assert "clarify" in result

    def test_multi_strategy_all_lists(self):
        """Test multi_strategy_rewrite returns lists for all strategies."""
        result = multi_strategy_rewrite("test")
        assert isinstance(result["original"], list)
        assert isinstance(result["expand"], list)
        assert isinstance(result["simplify"], list)
        assert isinstance(result["clarify"], list)

    def test_multi_strategy_includes_original(self):
        """Test multi_strategy_rewrite preserves original."""
        query = "test query"
        result = multi_strategy_rewrite(query)
        assert result["original"] == [query]

    # Edge Cases
    def test_empty_query(self, rewriter):
        """Test rewriting empty query."""
        result = rewriter.rewrite("")
        assert result["original"] == ""
        assert isinstance(result["rewritten"], list)

    def test_single_word_query(self, rewriter):
        """Test rewriting single word query."""
        result = rewriter.rewrite("Python")
        assert isinstance(result["rewritten"], list)
        assert len(result["rewritten"]) > 0

    def test_very_long_query(self, rewriter):
        """Test rewriting very long query."""
        query = " ".join(["word"] * 100)
        result = rewriter.rewrite(query)
        assert result["original"] == query
        assert isinstance(result["rewritten"], list)

    def test_special_characters(self, rewriter):
        """Test rewriting with special characters."""
        query = "what is @#$%?"
        result = rewriter.rewrite(query)
        assert result["original"] == query

    def test_unicode_characters(self, rewriter):
        """Test rewriting with unicode characters."""
        query = "café résumé"
        result = rewriter.rewrite(query)
        assert result["original"] == query

    def test_mixed_case_query(self, rewriter):
        """Test rewriting preserves case in original."""
        query = "TeSt QuErY"
        result = rewriter.rewrite(query)
        assert result["original"] == query

    def test_query_with_numbers(self, rewriter):
        """Test rewriting query with numbers."""
        query = "Python 3.9 features"
        result = rewriter.rewrite(query)
        assert result["original"] == query

    def test_query_with_punctuation(self, rewriter):
        """Test rewriting query with punctuation."""
        query = "What is AI/ML?"
        result = rewriter.rewrite(query)
        assert result["original"] == query

    # Case Sensitivity Tests
    def test_simplify_case_insensitive(self, rewriter):
        """Test simplify is case-insensitive for stop words."""
        rewritten = rewriter._simplify_query("THE BEST WAY")
        simplified = rewritten[-1] if len(rewritten) > 1 else rewritten[0]
        # "THE" should be removed despite being uppercase
        assert "the" not in simplified.lower().split()

    def test_clarify_detects_questions_case_insensitive(self, rewriter):
        """Test clarify detects question words case-insensitively."""
        rewritten = rewriter._clarify_query("WHAT is this")
        # Should detect "WHAT" as question word
        assert len(rewritten) > 1

    # Integration Tests
    def test_all_strategies_produce_output(self, rewriter):
        """Test all strategies produce valid output."""
        query = "machine learning"
        for strategy in ["expand", "simplify", "clarify"]:
            result = rewriter.rewrite(query, strategy=strategy)
            assert len(result["rewritten"]) > 0
            assert all(isinstance(q, str) for q in result["rewritten"])

    def test_strategies_produce_different_results(self, rewriter):
        """Test different strategies produce different results."""
        query = "what is the best way to learn machine learning"
        expand = rewriter.rewrite(query, strategy="expand")
        simplify = rewriter.rewrite(query, strategy="simplify")
        clarify = rewriter.rewrite(query, strategy="clarify")

        # Results should differ
        assert expand["rewritten"] != simplify["rewritten"]

    def test_rewriter_idempotent(self, rewriter):
        """Test rewriting same query produces consistent results."""
        query = "test query"
        result1 = rewriter.rewrite(query, strategy="expand")
        result2 = rewriter.rewrite(query, strategy="expand")
        assert result1["rewritten"] == result2["rewritten"]

    def test_whitespace_handling(self, rewriter):
        """Test queries with extra whitespace."""
        query = "test   query   with   spaces"
        result = rewriter.rewrite(query)
        assert result["original"] == query
        assert isinstance(result["rewritten"], list)

    def test_newline_handling(self, rewriter):
        """Test queries with newlines."""
        query = "test\nquery"
        result = rewriter.rewrite(query)
        assert result["original"] == query
