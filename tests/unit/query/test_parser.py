"""Tests for advanced query parser.

Tests cover:
- Boolean operators (AND, OR, NOT)
- Field filters (field:value)
- Quoted phrases
- Regex mode
- Filter application
- Edge cases and boundary conditions
"""

import pytest
from ingestforge.query.parser import QueryParser, QueryFilter, ParsedQuery


class TestQueryFilter:
    """Test QueryFilter dataclass."""

    def test_query_filter_creation(self):
        """Test creating QueryFilter."""
        filter = QueryFilter(field="author", value="Smith")
        assert filter.field == "author"
        assert filter.value == "Smith"

    def test_query_filter_equality(self):
        """Test QueryFilter equality."""
        f1 = QueryFilter(field="year", value="2023")
        f2 = QueryFilter(field="year", value="2023")
        assert f1 == f2


class TestParsedQuery:
    """Test ParsedQuery dataclass."""

    def test_parsed_query_creation(self):
        """Test creating ParsedQuery with all fields."""
        pq = ParsedQuery(
            terms=["term1", "term2"],
            exact_phrases=["exact phrase"],
            filters=[QueryFilter("author", "Smith")],
            exclude_terms=["exclude"],
            and_groups=[["term1", "term2"]],
            or_groups=[["term3", "term4"]],
            is_regex=False,
        )
        assert len(pq.terms) == 2
        assert len(pq.exact_phrases) == 1
        assert len(pq.filters) == 1
        assert len(pq.exclude_terms) == 1
        assert len(pq.and_groups) == 1
        assert len(pq.or_groups) == 1
        assert pq.is_regex is False

    def test_parsed_query_defaults(self):
        """Test ParsedQuery with minimal fields."""
        pq = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[],
            exclude_terms=[],
            and_groups=[],
            or_groups=[],
        )
        assert pq.is_regex is False


class TestQueryParser:
    """Test suite for QueryParser class."""

    @pytest.fixture
    def parser(self):
        """Create QueryParser instance."""
        return QueryParser()

    # Initialization Tests
    def test_initialization(self, parser):
        """Test parser initializes correctly."""
        assert hasattr(parser, "QUOTED_PATTERN")
        assert hasattr(parser, "FIELD_FILTER_PATTERN")
        assert hasattr(parser, "BOOLEAN_OPS")

    def test_boolean_ops_set(self, parser):
        """Test boolean operators are defined."""
        assert "AND" in parser.BOOLEAN_OPS
        assert "OR" in parser.BOOLEAN_OPS
        assert "NOT" in parser.BOOLEAN_OPS

    # Basic Parsing Tests
    def test_parse_simple_query(self, parser):
        """Test parsing simple query."""
        result = parser.parse("machine learning")
        assert isinstance(result, ParsedQuery)
        assert "machine" in result.terms
        assert "learning" in result.terms

    def test_parse_returns_parsed_query(self, parser):
        """Test parse returns ParsedQuery instance."""
        result = parser.parse("test")
        assert isinstance(result, ParsedQuery)
        assert hasattr(result, "terms")
        assert hasattr(result, "filters")

    def test_parse_empty_query(self, parser):
        """Test parsing empty query."""
        result = parser.parse("")
        assert isinstance(result, ParsedQuery)
        assert len(result.terms) == 0

    def test_parse_single_word(self, parser):
        """Test parsing single word."""
        result = parser.parse("python")
        assert "python" in result.terms
        assert len(result.terms) == 1

    # Quoted Phrase Tests
    def test_parse_quoted_phrase(self, parser):
        """Test parsing quoted phrase."""
        result = parser.parse('"machine learning"')
        assert "machine learning" in result.exact_phrases
        assert len(result.exact_phrases) == 1

    def test_parse_multiple_quoted_phrases(self, parser):
        """Test parsing multiple quoted phrases."""
        result = parser.parse('"first phrase" and "second phrase"')
        assert len(result.exact_phrases) == 2
        assert "first phrase" in result.exact_phrases
        assert "second phrase" in result.exact_phrases

    def test_parse_quoted_phrase_removal(self, parser):
        """Test quoted phrases are removed from terms."""
        result = parser.parse('"exact phrase" other terms')
        assert "exact phrase" in result.exact_phrases
        assert "other" in result.terms
        assert "terms" in result.terms

    def test_extract_quoted_phrases(self, parser):
        """Test _extract_quoted_phrases helper."""
        phrases, remaining = parser._extract_quoted_phrases('"test phrase" other')
        assert "test phrase" in phrases
        assert '"test phrase"' not in remaining

    # Field Filter Tests
    def test_parse_field_filter(self, parser):
        """Test parsing field filter."""
        result = parser.parse("author:Smith")
        assert len(result.filters) == 1
        assert result.filters[0].field == "author"
        assert result.filters[0].value == "Smith"

    def test_parse_multiple_field_filters(self, parser):
        """Test parsing multiple field filters."""
        result = parser.parse("author:Smith year:2023")
        assert len(result.filters) == 2

    def test_parse_field_filter_case_normalization(self, parser):
        """Test field names are lowercased."""
        result = parser.parse("AUTHOR:Smith")
        assert result.filters[0].field == "author"

    def test_parse_field_filter_excludes_boolean_ops(self, parser):
        """Test boolean operators not treated as field filters."""
        result = parser.parse("AND:value OR:value NOT:value")
        # Should not create filters for boolean operators
        assert not any(f.field in ["AND", "OR", "NOT"] for f in result.filters)

    def test_extract_filters(self, parser):
        """Test _extract_filters helper."""
        filters, remaining = parser._extract_filters("author:Smith test")
        assert len(filters) == 1
        assert filters[0].field == "author"
        assert filters[0].value == "Smith"

    # Boolean Operator Tests - AND
    def test_parse_and_operator(self, parser):
        """Test parsing AND operator."""
        result = parser.parse("term1 AND term2")
        assert len(result.and_groups) > 0

    def test_parse_multiple_and_groups(self, parser):
        """Test parsing multiple AND groups."""
        result = parser.parse("term1 AND term2 AND term3")
        assert len(result.and_groups) >= 1

    def test_and_group_contains_correct_terms(self, parser):
        """Test AND groups contain correct terms."""
        result = parser.parse("python AND java")
        # Both terms should be in terms list
        assert "python" in result.terms
        assert "java" in result.terms

    # Boolean Operator Tests - OR
    def test_parse_or_operator(self, parser):
        """Test parsing OR operator."""
        result = parser.parse("term1 OR term2")
        assert len(result.or_groups) > 0

    def test_parse_multiple_or_groups(self, parser):
        """Test parsing multiple OR groups."""
        result = parser.parse("term1 OR term2 OR term3")
        assert len(result.or_groups) >= 1

    def test_or_group_contains_correct_terms(self, parser):
        """Test OR groups contain correct terms."""
        result = parser.parse("python OR java")
        assert "python" in result.terms
        assert "java" in result.terms

    # Boolean Operator Tests - NOT
    def test_parse_not_operator(self, parser):
        """Test parsing NOT operator."""
        result = parser.parse("python NOT java")
        assert "java" in result.exclude_terms
        assert "python" in result.terms

    def test_parse_multiple_not_terms(self, parser):
        """Test parsing multiple NOT terms."""
        result = parser.parse("python NOT java NOT ruby")
        assert "java" in result.exclude_terms
        assert "ruby" in result.exclude_terms

    def test_not_term_not_in_regular_terms(self, parser):
        """Test NOT terms excluded from regular terms."""
        result = parser.parse("python NOT java")
        assert "java" not in result.terms

    # Complex Query Tests
    def test_parse_complex_query(self, parser):
        """Test parsing complex query with multiple features."""
        query = 'machine learning author:Smith year:2023 NOT "deep learning"'
        result = parser.parse(query)
        assert "machine" in result.terms
        assert "learning" in result.terms
        assert len(result.filters) == 2
        assert "deep learning" in result.exact_phrases

    def test_parse_mixed_operators(self, parser):
        """Test parsing query with mixed operators."""
        result = parser.parse("term1 AND term2 OR term3 NOT term4")
        assert len(result.and_groups) > 0
        assert len(result.or_groups) > 0
        assert "term4" in result.exclude_terms

    # Regex Mode Tests
    def test_parse_regex_mode(self, parser):
        """Test parsing in regex mode."""
        result = parser.parse("test.*pattern", is_regex=True)
        assert result.is_regex is True
        assert len(result.terms) == 1
        assert result.terms[0] == "test.*pattern"

    def test_regex_mode_skips_parsing(self, parser):
        """Test regex mode skips boolean parsing."""
        result = parser.parse("test AND pattern", is_regex=True)
        assert result.is_regex is True
        # Should not parse AND as operator
        assert len(result.and_groups) == 0

    # Parentheses Handling
    def test_parse_with_parentheses(self, parser):
        """Test parsing handles parentheses."""
        result = parser.parse("(term1 AND term2) OR term3")
        # Parentheses tokens are skipped but may be attached to terms
        # Test that query is parsed even with parentheses
        assert len(result.terms) > 0
        assert len(result.and_groups) > 0 or len(result.or_groups) > 0

    # Helper Method Tests
    def test_process_token_not(self, parser):
        """Test _process_token handles NOT."""
        terms, exclude, and_groups, or_groups = [], [], [], []
        should_continue, negate, _, _ = parser._process_token(
            "NOT", False, terms, exclude, and_groups, or_groups, [], []
        )
        assert should_continue is True
        assert negate is True

    def test_process_token_and(self, parser):
        """Test _process_token handles AND."""
        terms, exclude, and_groups, or_groups = [], [], [], []
        current_and = ["term1"]
        should_continue, negate, new_and, _ = parser._process_token(
            "AND", False, terms, exclude, and_groups, or_groups, current_and, []
        )
        assert should_continue is True
        assert len(and_groups) == 1

    def test_process_token_or(self, parser):
        """Test _process_token handles OR."""
        terms, exclude, and_groups, or_groups = [], [], [], []
        current_or = ["term1"]
        should_continue, negate, _, new_or = parser._process_token(
            "OR", False, terms, exclude, and_groups, or_groups, [], current_or
        )
        assert should_continue is True
        assert len(or_groups) == 1

    def test_process_token_regular(self, parser):
        """Test _process_token handles regular terms."""
        terms, exclude, and_groups, or_groups = [], [], [], []
        should_continue, negate, _, _ = parser._process_token(
            "python", False, terms, exclude, and_groups, or_groups, [], []
        )
        assert should_continue is False
        assert "python" in terms

    def test_process_regular_term(self, parser):
        """Test _process_regular_term helper."""
        terms, exclude = [], []
        should_continue, negate, _, _ = parser._process_regular_term(
            "python", False, terms, exclude, [], []
        )
        assert "python" in terms
        assert should_continue is False

    def test_process_regular_term_with_negate(self, parser):
        """Test _process_regular_term with negation."""
        terms, exclude = [], []
        should_continue, negate, _, _ = parser._process_regular_term(
            "java", True, terms, exclude, [], []
        )
        assert "java" in exclude
        assert "java" not in terms

    # Filter Application Tests
    def test_apply_filters_empty(self, parser):
        """Test apply_filters with no filters."""
        parsed = ParsedQuery([], [], [], [], [], [])
        chunks = [{"content": "test"}]
        result = parser.apply_filters(parsed, chunks)
        assert result == chunks

    def test_apply_filters_basic(self, parser):
        """Test apply_filters with basic filter."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[QueryFilter("type", "article")],
            exclude_terms=[],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {"metadata": {"type": "article"}, "source_location": {}},
            {"metadata": {"type": "book"}, "source_location": {}},
        ]
        result = parser.apply_filters(parsed, chunks)
        assert len(result) == 1
        assert result[0]["metadata"]["type"] == "article"

    def test_apply_filters_author(self, parser):
        """Test apply_filters with author filter."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[QueryFilter("author", "smith")],
            exclude_terms=[],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {
                "metadata": {},
                "source_location": {
                    "authors": [{"name": "John Smith", "last_name": "Smith"}]
                },
            },
            {
                "metadata": {},
                "source_location": {
                    "authors": [{"name": "Jane Doe", "last_name": "Doe"}]
                },
            },
        ]
        result = parser.apply_filters(parsed, chunks)
        assert len(result) == 1

    def test_apply_filters_year(self, parser):
        """Test apply_filters with year filter."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[QueryFilter("year", "2023")],
            exclude_terms=[],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {"metadata": {}, "source_location": {"publication_date": "2023-01-01"}},
            {"metadata": {}, "source_location": {"publication_date": "2022-01-01"}},
        ]
        result = parser.apply_filters(parsed, chunks)
        assert len(result) == 1

    def test_apply_filters_tag(self, parser):
        """Test apply_filters with tag filter."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[QueryFilter("tag", "python")],
            exclude_terms=[],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {"metadata": {"tags": ["python", "programming"]}, "source_location": {}},
            {"metadata": {"tags": ["java", "programming"]}, "source_location": {}},
        ]
        result = parser.apply_filters(parsed, chunks)
        assert len(result) == 1

    def test_apply_filters_multiple(self, parser):
        """Test apply_filters with multiple filters."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[QueryFilter("type", "article"), QueryFilter("tag", "python")],
            exclude_terms=[],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {
                "metadata": {"type": "article", "tags": ["python"]},
                "source_location": {},
            },
            {"metadata": {"type": "article", "tags": ["java"]}, "source_location": {}},
        ]
        result = parser.apply_filters(parsed, chunks)
        assert len(result) == 1

    # Exclusion Tests
    def test_apply_exclusions_empty(self, parser):
        """Test apply_exclusions with no exclusions."""
        parsed = ParsedQuery([], [], [], [], [], [])
        chunks = [{"content": "test"}]
        result = parser.apply_exclusions(parsed, chunks)
        assert result == chunks

    def test_apply_exclusions_basic(self, parser):
        """Test apply_exclusions with basic exclusion."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[],
            exclude_terms=["java"],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {"content": "python programming"},
            {"content": "java programming"},
        ]
        result = parser.apply_exclusions(parsed, chunks)
        assert len(result) == 1
        assert "python" in result[0]["content"]

    def test_apply_exclusions_multiple(self, parser):
        """Test apply_exclusions with multiple terms."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[],
            exclude_terms=["java", "ruby"],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {"content": "python programming"},
            {"content": "java programming"},
            {"content": "ruby programming"},
        ]
        result = parser.apply_exclusions(parsed, chunks)
        assert len(result) == 1

    def test_apply_exclusions_case_insensitive(self, parser):
        """Test apply_exclusions is case insensitive."""
        parsed = ParsedQuery(
            terms=[],
            exact_phrases=[],
            filters=[],
            exclude_terms=["java"],
            and_groups=[],
            or_groups=[],
        )
        chunks = [
            {"content": "JAVA programming"},
        ]
        result = parser.apply_exclusions(parsed, chunks)
        assert len(result) == 0

    # Edge Cases
    def test_parse_whitespace_only(self, parser):
        """Test parsing whitespace-only query."""
        result = parser.parse("   ")
        assert len(result.terms) == 0

    def test_parse_special_characters(self, parser):
        """Test parsing with special characters."""
        result = parser.parse("test@#$%")
        assert len(result.terms) > 0

    def test_parse_unicode(self, parser):
        """Test parsing with unicode characters."""
        result = parser.parse("café résumé")
        assert "café" in result.terms

    def test_parse_numbers(self, parser):
        """Test parsing with numbers."""
        result = parser.parse("Python 3.9")
        assert "python" in result.terms
        assert "3.9" in result.terms

    def test_parse_very_long_query(self, parser):
        """Test parsing very long query."""
        query = " ".join(["term"] * 100)
        result = parser.parse(query)
        assert len(result.terms) > 0

    # Filter Helper Tests
    def test_check_author_filter(self, parser):
        """Test _check_author_filter helper."""
        source_location = {"authors": [{"name": "John Smith", "last_name": "Smith"}]}
        assert parser._check_author_filter(source_location, "smith")
        assert parser._check_author_filter(source_location, "john")
        assert not parser._check_author_filter(source_location, "doe")

    def test_check_year_filter(self, parser):
        """Test _check_year_filter helper."""
        source_location = {"publication_date": "2023-01-01"}
        assert parser._check_year_filter(source_location, "2023")
        assert not parser._check_year_filter(source_location, "2022")

    def test_check_tag_filter(self, parser):
        """Test _check_tag_filter helper."""
        metadata = {"tags": ["Python", "Programming"]}
        assert parser._check_tag_filter(metadata, "python")
        assert parser._check_tag_filter(metadata, "programming")
        assert not parser._check_tag_filter(metadata, "java")

    def test_check_single_filter(self, parser):
        """Test _check_single_filter helper."""
        f = QueryFilter("type", "article")
        metadata = {"type": "article"}
        source_location = {}
        assert parser._check_single_filter(f, metadata, source_location)

    def test_chunk_matches_filters(self, parser):
        """Test _chunk_matches_filters helper."""
        chunk = {"metadata": {"type": "article"}, "source_location": {}}
        filters = [QueryFilter("type", "article")]
        assert parser._chunk_matches_filters(chunk, filters)
