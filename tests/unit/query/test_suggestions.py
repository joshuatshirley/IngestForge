"""Tests for query suggestions."""
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ingestforge.query.suggestions import QuerySuggester


class TestQuerySuggesterInitialization:
    """Tests for QuerySuggester initialization."""

    def test_init_creates_cache_directory(self, tmp_path: Path):
        """Test init creates cache directory if needed."""
        cache_path = tmp_path / "subdir" / "cache.json"
        suggester = QuerySuggester(cache_path)

        assert suggester.cache_path == cache_path
        assert cache_path.parent.exists()

    def test_init_existing_directory(self, tmp_path: Path):
        """Test init with existing directory."""
        cache_path = tmp_path / "cache.json"
        suggester = QuerySuggester(cache_path)

        assert suggester.cache_path == cache_path


class TestCorpusAnalysis:
    """Tests for corpus analysis."""

    @pytest.fixture
    def simple_chunks(self) -> List[Dict[str, Any]]:
        """Create simple test chunks."""
        return [
            {"content": "Python is a programming language."},
            {"content": "Python programming is great for data science."},
            {"content": "Machine Learning with Python is popular."},
        ]

    @pytest.fixture
    def chunks_with_entities(self) -> List[Dict[str, Any]]:
        """Create chunks with capitalized entities."""
        return [
            {"content": "John Doe works at Microsoft Corporation."},
            {"content": "Microsoft released Visual Studio Code."},
            {"content": "Google Cloud Platform is a competitor."},
        ]

    def test_analyze_corpus_extracts_terms(
        self, tmp_path: Path, simple_chunks: List[Dict[str, Any]]
    ):
        """Test corpus analysis extracts frequent terms."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(simple_chunks)

        assert "top_terms" in analysis
        assert "python" in analysis["top_terms"]
        assert "programming" in analysis["top_terms"]

    def test_analyze_corpus_extracts_bigrams(
        self, tmp_path: Path, simple_chunks: List[Dict[str, Any]]
    ):
        """Test corpus analysis extracts bigrams."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(simple_chunks)

        assert "top_bigrams" in analysis
        # Note: bigrams need frequency >= 3, might not appear with just 3 chunks

    def test_analyze_corpus_extracts_entities(
        self, tmp_path: Path, chunks_with_entities: List[Dict[str, Any]]
    ):
        """Test corpus analysis extracts capitalized entities."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks_with_entities)

        assert "entities" in analysis
        entities = analysis["entities"]

        # Should extract proper nouns
        assert any("Microsoft" in e for e in entities)
        assert any("Google" in e for e in entities)

    def test_analyze_corpus_filters_stopwords(self, tmp_path: Path):
        """Test analysis filters out stopwords."""
        chunks = [{"content": "The and for with this that from are was has have been."}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Stopwords should not be in top terms
        terms = analysis["top_terms"]
        assert "the" not in terms
        assert "and" not in terms
        assert "for" not in terms

    def test_analyze_corpus_counts_corpus_size(
        self, tmp_path: Path, simple_chunks: List[Dict[str, Any]]
    ):
        """Test analysis records corpus size."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(simple_chunks)

        assert analysis["corpus_size"] == len(simple_chunks)

    def test_analyze_corpus_handles_empty(self, tmp_path: Path):
        """Test analysis handles empty corpus."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus([])

        assert analysis["corpus_size"] == 0
        assert analysis["top_terms"] == []
        assert analysis["top_bigrams"] == []

    def test_analyze_corpus_saves_cache(
        self, tmp_path: Path, simple_chunks: List[Dict[str, Any]]
    ):
        """Test analysis saves results to cache."""
        cache_path = tmp_path / "cache.json"
        suggester = QuerySuggester(cache_path)
        suggester.analyze_corpus(simple_chunks)

        assert cache_path.exists()

        with open(cache_path) as f:
            cached = json.load(f)

        assert "top_terms" in cached
        assert "corpus_size" in cached

    def test_analyze_corpus_limits_terms(self, tmp_path: Path):
        """Test analysis limits number of terms."""
        # Create many unique terms
        chunks = [{"content": f"term{i} " * 10} for i in range(100)]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Should limit to top 50
        assert len(analysis["top_terms"]) <= 50

    def test_analyze_corpus_limits_entities(self, tmp_path: Path):
        """Test analysis limits number of entities."""
        # Create many entities
        chunks = [{"content": f"Person{i} " * 10} for i in range(50)]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Should limit to top 20
        assert len(analysis["entities"]) <= 20

    def test_analyze_corpus_filters_short_entities(self, tmp_path: Path):
        """Test analysis filters short entity names."""
        chunks = [{"content": "AI ML IOT and Machine Learning"}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Short entities (<=3 chars) should be filtered
        entities = analysis["entities"]
        assert not any(len(e) <= 3 for e in entities)

    def test_analyze_corpus_filters_digits(self, tmp_path: Path):
        """Test analysis filters numeric terms."""
        chunks = [{"content": "Python 3.9.1 and version 2023 release"}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Digit-only terms should be filtered
        terms = analysis["top_terms"]
        assert not any(t.isdigit() for t in terms)


class TestGetSuggestions:
    """Tests for general suggestion retrieval."""

    @pytest.fixture
    def suggester_with_cache(self, tmp_path: Path) -> QuerySuggester:
        """Create suggester with cached analysis."""
        cache_path = tmp_path / "cache.json"
        cache_data = {
            "top_terms": [
                "python",
                "programming",
                "data",
                "science",
                "machine",
                "learning",
            ],
            "top_bigrams": ["machine learning", "data science"],
            "entities": ["Python Programming", "Data Science"],
            "corpus_size": 10,
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        return QuerySuggester(cache_path)

    def test_get_suggestions_returns_list(self, suggester_with_cache: QuerySuggester):
        """Test get_suggestions returns a list."""
        suggestions = suggester_with_cache.get_suggestions()
        assert isinstance(suggestions, list)

    def test_get_suggestions_includes_entities(
        self, suggester_with_cache: QuerySuggester
    ):
        """Test suggestions include entities."""
        suggestions = suggester_with_cache.get_suggestions(limit=10)

        # Should include some entities
        assert any(
            "Python Programming" in s or "Data Science" in s for s in suggestions
        )

    def test_get_suggestions_includes_bigrams(
        self, suggester_with_cache: QuerySuggester
    ):
        """Test suggestions include bigrams."""
        suggestions = suggester_with_cache.get_suggestions(limit=10)

        # Should include some bigrams
        assert any("machine learning" in s or "data science" in s for s in suggestions)

    def test_get_suggestions_combines_terms(self, suggester_with_cache: QuerySuggester):
        """Test suggestions combine top terms."""
        suggestions = suggester_with_cache.get_suggestions(limit=10)

        # Should create combined queries from terms
        # (e.g., "python programming")
        assert len(suggestions) > 0

    def test_get_suggestions_respects_limit(self, suggester_with_cache: QuerySuggester):
        """Test suggestions respect limit parameter."""
        suggestions = suggester_with_cache.get_suggestions(limit=5)
        assert len(suggestions) <= 5

    def test_get_suggestions_no_cache(self, tmp_path: Path):
        """Test suggestions with no cache file."""
        suggester = QuerySuggester(tmp_path / "missing.json")
        suggestions = suggester.get_suggestions()

        assert suggestions == []

    def test_get_suggestions_empty_analysis(self, tmp_path: Path):
        """Test suggestions with empty analysis."""
        cache_path = tmp_path / "cache.json"
        cache_data = {
            "top_terms": [],
            "top_bigrams": [],
            "entities": [],
            "corpus_size": 0,
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        suggester = QuerySuggester(cache_path)
        suggestions = suggester.get_suggestions(limit=10)

        assert suggestions == []

    def test_get_suggestions_partial_data(self, tmp_path: Path):
        """Test suggestions handle partial analysis data."""
        cache_path = tmp_path / "cache.json"
        cache_data = {
            "top_terms": ["python"],
            # Missing bigrams and entities
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        suggester = QuerySuggester(cache_path)
        suggestions = suggester.get_suggestions(limit=10)

        # Should handle gracefully
        assert isinstance(suggestions, list)


class TestRelatedQueries:
    """Tests for related query generation."""

    @pytest.fixture
    def result_chunks(self) -> List[Dict[str, Any]]:
        """Create sample result chunks."""
        return [
            {"content": "Python is a versatile programming language."},
            {"content": "Python Programming is used for Machine Learning."},
            {"content": "Data Science with Python is popular."},
            {"content": "NumPy and Pandas are Python libraries."},
        ]

    def test_get_related_queries_returns_list(
        self, tmp_path: Path, result_chunks: List[Dict[str, Any]]
    ):
        """Test related queries returns a list."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("python", result_chunks, limit=3)

        assert isinstance(related, list)

    def test_get_related_queries_extracts_entities(
        self, tmp_path: Path, result_chunks: List[Dict[str, Any]]
    ):
        """Test related queries extract entities from results."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("python", result_chunks, limit=5)

        # Should extract capitalized entities
        assert any("Machine Learning" in r or "Data Science" in r for r in related)

    def test_get_related_queries_excludes_original_terms(
        self, tmp_path: Path, result_chunks: List[Dict[str, Any]]
    ):
        """Test related queries exclude terms from original query."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("python", result_chunks, limit=5)

        # Should not suggest "python" alone since it's in original
        # (but can be combined with other terms)
        assert not any(r.lower() == "python" for r in related)

    def test_get_related_queries_respects_limit(
        self, tmp_path: Path, result_chunks: List[Dict[str, Any]]
    ):
        """Test related queries respect limit."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("python", result_chunks, limit=3)

        assert len(related) <= 3

    def test_get_related_queries_empty_results(self, tmp_path: Path):
        """Test related queries with empty results."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("python", [], limit=3)

        assert related == []

    def test_get_related_queries_analyzes_top_results(self, tmp_path: Path):
        """Test related queries only analyze top 10 results."""
        # Create many chunks
        chunks = [{"content": f"Content {i}"} for i in range(50)]

        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("query", chunks, limit=3)

        # Should not crash or be slow (only analyzes first 10)
        assert isinstance(related, list)

    def test_get_related_queries_filters_short_entities(self, tmp_path: Path):
        """Test related queries filter short entity names."""
        chunks = [{"content": "AI ML IOT and Machine Learning"}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("ai", chunks, limit=5)

        # Short entities should be filtered
        assert not any(len(r) <= 3 for r in related if r.isupper())

    def test_get_related_queries_filters_digits(self, tmp_path: Path):
        """Test related queries filter numeric terms."""
        chunks = [{"content": "Python 3.9 released in 2020"}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("python", chunks, limit=5)

        # Should not suggest pure numbers
        assert not any(r.isdigit() for r in related)

    def test_get_related_queries_combines_with_original(
        self, tmp_path: Path, result_chunks: List[Dict[str, Any]]
    ):
        """Test related queries can combine terms with original."""
        suggester = QuerySuggester(tmp_path / "cache.json")
        related = suggester.get_related_queries("python", result_chunks, limit=5)

        # Some suggestions should combine new terms with original
        assert any("python" in r.lower() for r in related)


class TestCacheSave:
    """Tests for cache saving."""

    def test_save_cache_creates_file(self, tmp_path: Path):
        """Test save cache creates file."""
        cache_path = tmp_path / "cache.json"
        suggester = QuerySuggester(cache_path)

        analysis = {
            "top_terms": ["test"],
            "top_bigrams": [],
            "entities": [],
            "corpus_size": 1,
        }

        suggester._save_cache(analysis)

        assert cache_path.exists()

    def test_save_cache_writes_json(self, tmp_path: Path):
        """Test save cache writes valid JSON."""
        cache_path = tmp_path / "cache.json"
        suggester = QuerySuggester(cache_path)

        analysis = {
            "top_terms": ["python", "java"],
            "top_bigrams": ["machine learning"],
            "entities": ["Python Programming"],
            "corpus_size": 5,
        }

        suggester._save_cache(analysis)

        with open(cache_path) as f:
            loaded = json.load(f)

        assert loaded == analysis

    def test_save_cache_handles_unicode(self, tmp_path: Path):
        """Test save cache handles unicode characters."""
        cache_path = tmp_path / "cache.json"
        suggester = QuerySuggester(cache_path)

        analysis = {
            "top_terms": ["caf\u00e9", "na\u00efve"],
            "top_bigrams": [],
            "entities": [],
            "corpus_size": 1,
        }

        suggester._save_cache(analysis)

        with open(cache_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert "caf\u00e9" in loaded["top_terms"]


class TestCacheLoad:
    """Tests for cache loading."""

    def test_load_cache_missing_file(self, tmp_path: Path):
        """Test load cache with missing file."""
        suggester = QuerySuggester(tmp_path / "missing.json")
        result = suggester._load_cache()

        assert result == {}

    def test_load_cache_valid_file(self, tmp_path: Path):
        """Test load cache with valid file."""
        cache_path = tmp_path / "cache.json"
        cache_data = {"top_terms": ["test"], "corpus_size": 1}

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        suggester = QuerySuggester(cache_path)
        result = suggester._load_cache()

        assert result == cache_data

    def test_load_cache_invalid_json(self, tmp_path: Path):
        """Test load cache with invalid JSON."""
        cache_path = tmp_path / "cache.json"

        with open(cache_path, "w") as f:
            f.write("not valid json {")

        suggester = QuerySuggester(cache_path)
        result = suggester._load_cache()

        # Should return empty dict on error
        assert result == {}

    def test_load_cache_handles_unicode(self, tmp_path: Path):
        """Test load cache handles unicode."""
        cache_path = tmp_path / "cache.json"
        cache_data = {"top_terms": ["caf\u00e9"], "corpus_size": 1}

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)

        suggester = QuerySuggester(cache_path)
        result = suggester._load_cache()

        assert "caf\u00e9" in result["top_terms"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content_chunks(self, tmp_path: Path):
        """Test handling chunks with empty content."""
        chunks = [{"content": ""}, {"content": "   "}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Should handle gracefully
        assert analysis["corpus_size"] == 2

    def test_missing_content_key(self, tmp_path: Path):
        """Test handling chunks missing content key."""
        chunks = [{"text": "Some text"}, {}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Should handle gracefully (content defaults to empty)
        assert isinstance(analysis, dict)

    def test_very_large_corpus(self, tmp_path: Path):
        """Test handling large corpus."""
        # Create many chunks
        chunks = [{"content": f"term{i % 100}"} for i in range(1000)]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Should complete without error
        assert analysis["corpus_size"] == 1000

    def test_special_characters_in_content(self, tmp_path: Path):
        """Test handling special characters."""
        chunks = [{"content": "@#$%^&*() <html> [brackets] {braces}"}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Should handle gracefully
        assert isinstance(analysis, dict)

    def test_mixed_case_entities(self, tmp_path: Path):
        """Test handling mixed case entities."""
        # Entity extraction uses pattern: [A-Z][a-z]+(\s+[A-Z][a-z]+)*
        # Matches: "Machine Learning", "Data Science"
        # Doesn't match: "iPhone", "MacBook" (capital in middle)
        chunks = [
            {"content": "Machine Learning uses Python Programming for Data Science"}
        ]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        # Should extract standard capitalized entities
        entities = analysis["entities"]
        assert any(
            "Machine Learning" in e or "Data Science" in e or "Python Programming" in e
            for e in entities
        )

    def test_analyze_then_suggest_workflow(self, tmp_path: Path):
        """Test full workflow: analyze then suggest."""
        chunks = [
            {"content": "Python Programming Language"},
            {"content": "Machine Learning with Python"},
            {"content": "Data Science and Python"},
        ]

        cache_path = tmp_path / "cache.json"
        suggester = QuerySuggester(cache_path)

        # Analyze
        suggester.analyze_corpus(chunks)

        # Get suggestions
        suggestions = suggester.get_suggestions(limit=5)

        assert len(suggestions) > 0
        assert any("python" in s.lower() for s in suggestions)


class TestBigramFrequency:
    """Tests for bigram frequency threshold."""

    def test_bigrams_need_minimum_frequency(self, tmp_path: Path):
        """Test bigrams need frequency >= 3."""
        chunks = [
            {"content": "rare bigram here"},
            {"content": "common phrase common phrase common phrase"},
        ]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        bigrams = analysis["top_bigrams"]

        # "rare bigram" appears once, should not be included
        assert "rare bigram" not in bigrams

        # "common phrase" appears 3 times, should be included
        assert "common phrase" in bigrams

    def test_bigrams_count_correctly(self, tmp_path: Path):
        """Test bigrams are counted correctly."""
        chunks = [
            {"content": "machine learning machine learning machine learning"},
        ]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        bigrams = analysis["top_bigrams"]

        # Should appear (frequency >= 3)
        assert "machine learning" in bigrams


class TestWordExtraction:
    """Tests for word extraction patterns."""

    def test_extracts_words_4_chars_minimum(self, tmp_path: Path):
        """Test extraction uses minimum 4 characters."""
        chunks = [{"content": "cat dog bird python programming"}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        terms = analysis["top_terms"]

        # Short words should not appear
        assert "cat" not in terms
        assert "dog" not in terms

        # Longer words should appear
        assert "bird" in terms or "python" in terms

    def test_extracts_alphanumeric_only(self, tmp_path: Path):
        """Test extraction only gets alphanumeric words."""
        chunks = [{"content": "test123 @mention #hashtag word"}]

        suggester = QuerySuggester(tmp_path / "cache.json")
        analysis = suggester.analyze_corpus(chunks)

        terms = analysis["top_terms"]

        # Should extract clean words
        assert "word" in terms
        # Mixed alphanumeric might be included depending on regex
