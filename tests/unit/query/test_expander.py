"""Tests for query expansion functionality.

Tests cover:
- Synonym expansion
- Acronym expansion
- Query paraphrasing (procedural, factual, conceptual)
- Edge cases and boundary conditions
"""

import pytest
from ingestforge.query.expander import QueryExpander


class TestQueryExpander:
    """Test suite for QueryExpander class."""

    @pytest.fixture
    def expander(self):
        """Create QueryExpander instance."""
        return QueryExpander()

    # Basic Functionality Tests
    def test_initialization(self, expander):
        """Test expander initializes with synonym and acronym dictionaries."""
        assert isinstance(expander.synonyms, dict)
        assert isinstance(expander.acronyms, dict)
        assert len(expander.synonyms) > 0
        assert len(expander.acronyms) > 0

    def test_expand_returns_list(self, expander):
        """Test expand returns a list of strings."""
        result = expander.expand("test query")
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_expand_respects_max_expansions(self, expander):
        """Test max_expansions parameter limits results."""
        result = expander.expand("how to obtain requirements", max_expansions=2)
        assert len(result) <= 2

    def test_expand_removes_duplicates(self, expander):
        """Test expansions don't include duplicates."""
        result = expander.expand("test query", max_expansions=10)
        assert len(result) == len(set(result))

    # Synonym Expansion Tests
    def test_synonym_expansion_basic(self, expander):
        """Test basic synonym replacement."""
        # "requirements" should expand to "prerequisites"
        result = expander._expand_synonyms("check requirements")
        assert "prerequisites" in result

    def test_synonym_expansion_multiple_matches(self, expander):
        """Test only one synonym replacement per query."""
        result = expander._expand_synonyms("requirements process")
        # Should replace only first match
        assert "prerequisites" in result or "procedure" in result

    def test_synonym_expansion_no_match(self, expander):
        """Test queries without synonyms return unchanged."""
        original = "some random text"
        result = expander._expand_synonyms(original)
        assert result == original.lower()

    def test_synonym_expansion_case_insensitive(self, expander):
        """Test synonym matching is case-insensitive."""
        result = expander._expand_synonyms("REQUIREMENTS")
        assert "prerequisites" in result

    # Acronym Expansion Tests
    def test_acronym_expansion_basic(self, expander):
        """Test basic acronym expansion."""
        result = expander._expand_acronyms("What is FAQ")
        assert "frequently asked questions" in result.lower()

    def test_acronym_expansion_word_boundary(self, expander):
        """Test acronym expansion respects word boundaries."""
        # "FAQ" in "FAQTEST" should not match
        result = expander._expand_acronyms("FAQTEST")
        assert result == "FAQTEST"

    def test_acronym_expansion_case_insensitive(self, expander):
        """Test acronym matching is case-insensitive."""
        result = expander._expand_acronyms("check faq")
        assert "frequently asked questions" in result.lower()

    def test_acronym_expansion_multiple(self, expander):
        """Test multiple acronyms in query."""
        result = expander._expand_acronyms("FAQ and ASAP")
        # Should expand first acronym only
        assert (
            "frequently asked questions" in result.lower()
            or "as soon as possible" in result.lower()
        )

    def test_acronym_expansion_no_match(self, expander):
        """Test queries without acronyms return unchanged."""
        original = "some random text"
        result = expander._expand_acronyms(original)
        assert result == original

    # Procedural Query Paraphrasing Tests
    def test_paraphrase_procedural_how_to(self, expander):
        """Test paraphrasing 'how to' queries."""
        paraphrases = expander._paraphrase_procedural("how to apply")
        assert len(paraphrases) > 0
        assert any("steps to" in p for p in paraphrases)
        assert any("process for" in p for p in paraphrases)

    def test_paraphrase_procedural_how_do_i(self, expander):
        """Test paraphrasing 'how do i' queries."""
        paraphrases = expander._paraphrase_procedural("how do i submit")
        assert len(paraphrases) > 0
        assert any("how to" in p for p in paraphrases)
        assert any("steps to" in p for p in paraphrases)

    def test_paraphrase_procedural_no_match(self, expander):
        """Test non-procedural queries return empty list."""
        paraphrases = expander._paraphrase_procedural("what is this")
        assert paraphrases == []

    def test_paraphrase_procedural_validation(self, expander):
        """Test parameter validation for procedural paraphrasing."""
        with pytest.raises(AssertionError):
            expander._paraphrase_procedural(None)

        with pytest.raises(AssertionError):
            expander._paraphrase_procedural(123)

    # Factual Query Paraphrasing Tests
    def test_paraphrase_factual_what_is(self, expander):
        """Test paraphrasing 'what is' queries."""
        paraphrases = expander._paraphrase_factual("what is python")
        assert len(paraphrases) > 0
        assert any("define" in p for p in paraphrases)
        assert any("definition" in p for p in paraphrases)

    def test_paraphrase_factual_what_are(self, expander):
        """Test paraphrasing 'what are' queries."""
        paraphrases = expander._paraphrase_factual("what are requirements")
        assert len(paraphrases) > 0
        assert any("list of" in p for p in paraphrases)

    def test_paraphrase_factual_no_match(self, expander):
        """Test non-factual queries return empty list."""
        paraphrases = expander._paraphrase_factual("how to do this")
        assert paraphrases == []

    def test_paraphrase_factual_validation(self, expander):
        """Test parameter validation for factual paraphrasing."""
        with pytest.raises(AssertionError):
            expander._paraphrase_factual(None)

        with pytest.raises(AssertionError):
            expander._paraphrase_factual(123)

    # Conceptual Query Paraphrasing Tests
    def test_paraphrase_conceptual_explain(self, expander):
        """Test paraphrasing 'explain' queries."""
        paraphrases = expander._paraphrase_conceptual("explain machine learning")
        assert len(paraphrases) > 0
        assert any("what is" in p for p in paraphrases)
        assert any("overview" in p for p in paraphrases)

    def test_paraphrase_conceptual_no_match(self, expander):
        """Test non-conceptual queries return empty list."""
        paraphrases = expander._paraphrase_conceptual("what is this")
        assert paraphrases == []

    def test_paraphrase_conceptual_validation(self, expander):
        """Test parameter validation for conceptual paraphrasing."""
        with pytest.raises(AssertionError):
            expander._paraphrase_conceptual(None)

        with pytest.raises(AssertionError):
            expander._paraphrase_conceptual(123)

    # Question Mark Handling Tests
    def test_add_question_marks_with_mark(self, expander):
        """Test adding question marks when original had one."""
        paraphrases = ["what is it", "define it"]
        result = expander._add_question_marks(paraphrases, has_question_mark=True)
        assert all(p.endswith("?") for p in result)

    def test_add_question_marks_without_mark(self, expander):
        """Test no question marks when original didn't have one."""
        paraphrases = ["what is it", "define it"]
        result = expander._add_question_marks(paraphrases, has_question_mark=False)
        assert result == paraphrases

    def test_add_question_marks_validation(self, expander):
        """Test parameter validation for question mark addition."""
        with pytest.raises(AssertionError):
            expander._add_question_marks(None, True)

        with pytest.raises(AssertionError):
            expander._add_question_marks("not a list", True)

    # Full Query Generation Tests
    def test_generate_paraphrases_procedural(self, expander):
        """Test full paraphrase generation for procedural queries."""
        result = expander._generate_paraphrases("How to apply?", "procedural")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(p.endswith("?") for p in result)

    def test_generate_paraphrases_factual(self, expander):
        """Test full paraphrase generation for factual queries."""
        result = expander._generate_paraphrases("What is Python?", "factual")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(p.endswith("?") for p in result)

    def test_generate_paraphrases_conceptual(self, expander):
        """Test full paraphrase generation for conceptual queries."""
        result = expander._generate_paraphrases("Explain AI?", "conceptual")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(p.endswith("?") for p in result)

    def test_generate_paraphrases_unknown_type(self, expander):
        """Test unknown query type returns empty list."""
        result = expander._generate_paraphrases("test query", "unknown")
        assert result == []

    def test_generate_paraphrases_no_type(self, expander):
        """Test no query type returns empty list."""
        result = expander._generate_paraphrases("test query", None)
        assert result == []

    def test_generate_paraphrases_validation(self, expander):
        """Test parameter validation for paraphrase generation."""
        with pytest.raises(AssertionError):
            expander._generate_paraphrases(None, "procedural")

        with pytest.raises(AssertionError):
            expander._generate_paraphrases(123, "procedural")

    # Integration Tests
    def test_expand_procedural_query(self, expander):
        """Test full expansion of procedural query."""
        query = "How to obtain requirements?"
        result = expander.expand(query, query_type="procedural", max_expansions=5)
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 5

    def test_expand_factual_query(self, expander):
        """Test full expansion of factual query."""
        query = "What is machine learning?"
        result = expander.expand(query, query_type="factual", max_expansions=5)
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 5

    def test_expand_with_acronym(self, expander):
        """Test expansion with acronym."""
        query = "What is FAQ"
        result = expander.expand(query, max_expansions=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_expand_with_synonym(self, expander):
        """Test expansion with synonym."""
        query = "check requirements"
        result = expander.expand(query, max_expansions=5)
        assert isinstance(result, list)
        assert len(result) > 0

    # Edge Cases
    def test_expand_empty_query(self, expander):
        """Test expansion of empty query."""
        result = expander.expand("")
        assert isinstance(result, list)

    def test_expand_very_long_query(self, expander):
        """Test expansion of very long query."""
        query = " ".join(["word"] * 100)
        result = expander.expand(query, max_expansions=3)
        assert len(result) <= 3

    def test_expand_special_characters(self, expander):
        """Test expansion with special characters."""
        query = "what is @#$%?"
        result = expander.expand(query)
        assert isinstance(result, list)

    def test_expand_unicode_query(self, expander):
        """Test expansion with unicode characters."""
        query = "what is résumé"
        result = expander.expand(query)
        assert isinstance(result, list)

    def test_expand_max_expansions_zero(self, expander):
        """Test expansion with max_expansions=0."""
        result = expander.expand("test query", max_expansions=0)
        assert result == []

    def test_expand_max_expansions_negative(self, expander):
        """Test expansion with negative max_expansions."""
        result = expander.expand("test query", max_expansions=-1)
        assert result == []
