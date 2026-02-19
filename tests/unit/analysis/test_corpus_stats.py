"""Tests for corpus statistics and analytics.

Tests cover:
- CorpusStatistics dataclass
- CorpusAnalyzer analysis methods
- Entity extraction
- Topic extraction
- Temporal analysis
- Coverage gap detection
- Edge cases and boundary conditions
"""

import pytest
from collections import Counter
from ingestforge.analysis.corpus_stats import CorpusAnalyzer, CorpusStatistics


class TestCorpusStatistics:
    """Test CorpusStatistics dataclass."""

    def test_corpus_statistics_creation(self):
        """Test creating CorpusStatistics."""
        stats = CorpusStatistics(
            total_documents=10,
            total_chunks=100,
            avg_chunk_size=500,
            document_types={"pdf": 5, "txt": 5},
            top_entities=[("Python", 10)],
            top_topics=[("programming", 20)],
            source_diversity=10,
            temporal_distribution={"2023": 5, "2024": 5},
            reading_time_minutes=30,
            coverage_gaps=[],
        )
        assert stats.total_documents == 10
        assert stats.total_chunks == 100
        assert stats.avg_chunk_size == 500

    def test_corpus_statistics_all_fields(self):
        """Test CorpusStatistics has all expected fields."""
        stats = CorpusStatistics(
            total_documents=0,
            total_chunks=0,
            avg_chunk_size=0,
            document_types={},
            top_entities=[],
            top_topics=[],
            source_diversity=0,
            temporal_distribution={},
            reading_time_minutes=0,
            coverage_gaps=[],
        )
        assert hasattr(stats, "total_documents")
        assert hasattr(stats, "total_chunks")
        assert hasattr(stats, "avg_chunk_size")
        assert hasattr(stats, "document_types")
        assert hasattr(stats, "top_entities")
        assert hasattr(stats, "top_topics")
        assert hasattr(stats, "source_diversity")
        assert hasattr(stats, "temporal_distribution")
        assert hasattr(stats, "reading_time_minutes")
        assert hasattr(stats, "coverage_gaps")


class TestCorpusAnalyzer:
    """Test suite for CorpusAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create CorpusAnalyzer instance."""
        return CorpusAnalyzer()

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            {
                "source_file": "doc1.pdf",
                "content": "Python is a programming language. Machine Learning uses Python.",
                "source_location": {"publication_date": "2023-01-01"},
            },
            {
                "source_file": "doc2.txt",
                "content": "Natural Language Processing is important. Deep Learning models are powerful.",
                "source_location": {"publication_date": "2023-06-01"},
            },
            {
                "source_file": "doc3.md",
                "content": "Data Science requires statistical knowledge. Python is widely used.",
                "source_location": {"publication_date": "2024-01-01"},
            },
        ]

    # Initialization Tests
    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert hasattr(analyzer, "WORDS_PER_MINUTE")
        assert analyzer.WORDS_PER_MINUTE == 200

    # Basic Analysis Tests
    def test_analyze_returns_statistics(self, analyzer, sample_chunks):
        """Test analyze returns CorpusStatistics."""
        result = analyzer.analyze(sample_chunks)
        assert isinstance(result, CorpusStatistics)

    def test_analyze_empty_chunks(self, analyzer):
        """Test analyze handles empty chunk list."""
        result = analyzer.analyze([])
        assert isinstance(result, CorpusStatistics)
        assert result.total_chunks == 0
        assert result.total_documents == 0

    def test_analyze_total_chunks(self, analyzer, sample_chunks):
        """Test analyze counts total chunks correctly."""
        result = analyzer.analyze(sample_chunks)
        assert result.total_chunks == 3

    def test_analyze_total_documents(self, analyzer, sample_chunks):
        """Test analyze counts unique documents."""
        result = analyzer.analyze(sample_chunks)
        assert result.total_documents == 3

    def test_analyze_source_diversity(self, analyzer, sample_chunks):
        """Test analyze calculates source diversity."""
        result = analyzer.analyze(sample_chunks)
        assert result.source_diversity == 3

    # Chunk Source Analysis Tests
    def test_analyze_chunk_source(self, analyzer):
        """Test _analyze_chunk_source extracts source info."""
        sources = set()
        doc_types = Counter()
        chunk = {"source_file": "test.pdf"}
        analyzer._analyze_chunk_source(chunk, sources, doc_types)
        assert "test.pdf" in sources
        assert doc_types["pdf"] == 1

    def test_analyze_chunk_source_no_extension(self, analyzer):
        """Test _analyze_chunk_source handles files without extension."""
        sources = set()
        doc_types = Counter()
        chunk = {"source_file": "testfile"}
        analyzer._analyze_chunk_source(chunk, sources, doc_types)
        assert "testfile" in sources
        assert len(doc_types) == 0

    def test_analyze_chunk_source_unknown(self, analyzer):
        """Test _analyze_chunk_source handles missing source."""
        sources = set()
        doc_types = Counter()
        chunk = {}
        analyzer._analyze_chunk_source(chunk, sources, doc_types)
        assert "unknown" in sources

    def test_analyze_document_types(self, analyzer, sample_chunks):
        """Test analyze extracts document types."""
        result = analyzer.analyze(sample_chunks)
        assert "pdf" in result.document_types
        assert "txt" in result.document_types
        assert "md" in result.document_types
        assert result.document_types["pdf"] == 1

    # Entity Extraction Tests
    def test_extract_entities_basic(self, analyzer):
        """Test _extract_entities extracts capitalized words."""
        entities = Counter()
        content = "Python is great. Machine Learning is important."
        analyzer._extract_entities(content, entities)
        assert "Python" in entities
        assert "Machine Learning" in entities

    def test_extract_entities_min_length(self, analyzer):
        """Test _extract_entities filters short entities."""
        entities = Counter()
        content = "A Big Test Word"
        analyzer._extract_entities(content, entities)
        # "A" should be filtered (length <= 3), but multi-word phrases extracted
        assert "A" not in entities
        # "Big Test Word" is extracted as multi-word entity
        assert len(entities) > 0

    def test_extract_entities_multi_word(self, analyzer):
        """Test _extract_entities handles multi-word entities."""
        entities = Counter()
        content = "Natural Language Processing"
        analyzer._extract_entities(content, entities)
        assert "Natural Language Processing" in entities

    def test_extract_entities_empty_content(self, analyzer):
        """Test _extract_entities handles empty content."""
        entities = Counter()
        analyzer._extract_entities("", entities)
        assert len(entities) == 0

    def test_analyze_top_entities(self, analyzer, sample_chunks):
        """Test analyze returns top entities."""
        result = analyzer.analyze(sample_chunks)
        assert isinstance(result.top_entities, list)
        assert len(result.top_entities) <= 15

    # Topic Extraction Tests
    def test_extract_topics_basic(self, analyzer):
        """Test _extract_topics extracts meaningful words."""
        topics = Counter()
        stopwords = {"the", "and", "for"}
        content = "programming language machine learning"
        analyzer._extract_topics(content, topics, stopwords)
        assert "programming" in topics
        assert "language" in topics
        assert "machine" in topics
        assert "learning" in topics

    def test_extract_topics_min_length(self, analyzer):
        """Test _extract_topics filters short words."""
        topics = Counter()
        stopwords = set()
        content = "test data code it"
        analyzer._extract_topics(content, topics, stopwords)
        # Words shorter than 5 chars should be filtered
        assert "test" not in topics
        assert "data" not in topics
        assert "code" not in topics

    def test_extract_topics_stopwords(self, analyzer):
        """Test _extract_topics filters stopwords."""
        topics = Counter()
        stopwords = {"programming", "language"}
        content = "programming language testing"
        analyzer._extract_topics(content, topics, stopwords)
        assert "programming" not in topics
        assert "language" not in topics
        assert "testing" in topics

    def test_extract_topics_digits(self, analyzer):
        """Test _extract_topics filters digit-only words."""
        topics = Counter()
        stopwords = set()
        content = "testing 12345 python"
        analyzer._extract_topics(content, topics, stopwords)
        assert "12345" not in topics
        assert "testing" in topics
        assert "python" in topics

    def test_extract_topics_case_insensitive(self, analyzer):
        """Test _extract_topics normalizes case."""
        topics = Counter()
        stopwords = set()
        content = "Programming PROGRAMMING programming"
        analyzer._extract_topics(content, topics, stopwords)
        assert topics["programming"] == 3

    def test_analyze_top_topics(self, analyzer, sample_chunks):
        """Test analyze returns top topics."""
        result = analyzer.analyze(sample_chunks)
        assert isinstance(result.top_topics, list)
        assert len(result.top_topics) <= 20

    # Temporal Analysis Tests
    def test_extract_temporal_data_basic(self, analyzer):
        """Test _extract_temporal_data extracts year."""
        years = Counter()
        chunk = {"source_location": {"publication_date": "2023-01-01"}}
        analyzer._extract_temporal_data(chunk, years)
        assert years["2023"] == 1

    def test_extract_temporal_data_no_date(self, analyzer):
        """Test _extract_temporal_data handles missing date."""
        years = Counter()
        chunk = {"source_location": {}}
        analyzer._extract_temporal_data(chunk, years)
        assert len(years) == 0

    def test_extract_temporal_data_short_date(self, analyzer):
        """Test _extract_temporal_data handles short date."""
        years = Counter()
        chunk = {"source_location": {"publication_date": "202"}}
        analyzer._extract_temporal_data(chunk, years)
        assert len(years) == 0

    def test_extract_temporal_data_invalid_year(self, analyzer):
        """Test _extract_temporal_data filters non-digit years."""
        years = Counter()
        chunk = {"source_location": {"publication_date": "abcd-01-01"}}
        analyzer._extract_temporal_data(chunk, years)
        assert len(years) == 0

    def test_extract_temporal_data_no_source_location(self, analyzer):
        """Test _extract_temporal_data handles missing source_location."""
        years = Counter()
        chunk = {}
        analyzer._extract_temporal_data(chunk, years)
        assert len(years) == 0

    def test_analyze_temporal_distribution(self, analyzer, sample_chunks):
        """Test analyze calculates temporal distribution."""
        result = analyzer.analyze(sample_chunks)
        assert isinstance(result.temporal_distribution, dict)
        assert "2023" in result.temporal_distribution
        assert "2024" in result.temporal_distribution

    # Reading Time Tests
    def test_calculate_reading_time_basic(self, analyzer):
        """Test _calculate_reading_time calculates correctly."""
        # 1000 chars = ~200 words = 1 minute
        reading_time = analyzer._calculate_reading_time(1000)
        assert reading_time == 1

    def test_calculate_reading_time_zero(self, analyzer):
        """Test _calculate_reading_time handles zero chars."""
        reading_time = analyzer._calculate_reading_time(0)
        assert reading_time == 0

    def test_calculate_reading_time_large(self, analyzer):
        """Test _calculate_reading_time handles large numbers."""
        # 100,000 chars = ~20,000 words = 100 minutes
        reading_time = analyzer._calculate_reading_time(100000)
        assert reading_time == 100

    def test_analyze_reading_time(self, analyzer, sample_chunks):
        """Test analyze calculates reading time."""
        result = analyzer.analyze(sample_chunks)
        assert result.reading_time_minutes >= 0

    # Average Chunk Size Tests
    def test_analyze_avg_chunk_size(self, analyzer, sample_chunks):
        """Test analyze calculates average chunk size."""
        result = analyzer.analyze(sample_chunks)
        total_chars = sum(len(c.get("content", "")) for c in sample_chunks)
        expected_avg = total_chars // len(sample_chunks)
        assert result.avg_chunk_size == expected_avg

    def test_analyze_avg_chunk_size_empty_content(self, analyzer):
        """Test analyze handles chunks with empty content."""
        chunks = [
            {"source_file": "test.txt", "content": "", "source_location": {}},
        ]
        result = analyzer.analyze(chunks)
        assert result.avg_chunk_size == 0

    # Coverage Gap Detection Tests
    def test_identify_gaps_empty_chunks(self, analyzer):
        """Test _identify_gaps handles empty chunks."""
        gaps = analyzer._identify_gaps([], Counter())
        assert len(gaps) > 0
        assert "too small" in gaps[0].lower()

    def test_identify_gaps_no_topics(self, analyzer):
        """Test _identify_gaps handles no topics."""
        chunks = [{"content": "", "source_location": {}}]
        gaps = analyzer._identify_gaps(chunks, Counter())
        assert len(gaps) > 0

    def test_identify_gaps_imbalanced_coverage(self, analyzer):
        """Test _identify_gaps detects imbalanced coverage."""
        topics = Counter(
            {
                "topic1": 100,
                "topic2": 2,
                "topic3": 1,
                "topic4": 1,
                "topic5": 1,
            }
        )
        chunks = [{"content": "test", "source_location": {}}]
        gaps = analyzer._identify_gaps(chunks, topics)
        # Should detect low coverage topics
        assert any("low coverage" in gap.lower() for gap in gaps)

    def test_identify_gaps_temporal(self, analyzer):
        """Test _identify_gaps detects temporal gaps."""
        chunks = [
            {"content": "test", "source_location": {"publication_date": "2020-01-01"}},
            {"content": "test", "source_location": {"publication_date": "2022-01-01"}},
            {"content": "test", "source_location": {"publication_date": "2024-01-01"}},
        ]
        topics = Counter({"test": 3})
        gaps = analyzer._identify_gaps(chunks, topics)
        # Should detect missing 2021, 2023
        assert any("missing coverage for years" in gap.lower() for gap in gaps)

    def test_identify_gaps_no_gaps(self, analyzer):
        """Test _identify_gaps returns no gaps message."""
        topics = Counter({"topic1": 10, "topic2": 9, "topic3": 8})
        chunks = [
            {"content": "test", "source_location": {"publication_date": "2023-01-01"}},
            {"content": "test", "source_location": {"publication_date": "2023-06-01"}},
        ]
        gaps = analyzer._identify_gaps(chunks, topics)
        # Should find no significant gaps
        assert any("no significant" in gap.lower() for gap in gaps)

    def test_analyze_coverage_gaps(self, analyzer, sample_chunks):
        """Test analyze includes coverage gaps."""
        result = analyzer.analyze(sample_chunks)
        assert isinstance(result.coverage_gaps, list)
        assert len(result.coverage_gaps) > 0

    # Helper Method Tests
    def test_create_empty_statistics(self, analyzer):
        """Test _create_empty_statistics returns valid stats."""
        stats = analyzer._create_empty_statistics()
        assert isinstance(stats, CorpusStatistics)
        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.avg_chunk_size == 0
        assert stats.document_types == {}
        assert stats.top_entities == []
        assert stats.top_topics == []
        assert stats.source_diversity == 0
        assert stats.temporal_distribution == {}
        assert stats.reading_time_minutes == 0
        assert stats.coverage_gaps == []

    def test_analyze_all_chunks(self, analyzer, sample_chunks):
        """Test _analyze_all_chunks collects statistics."""
        counters, total_chars = analyzer._analyze_all_chunks(sample_chunks)
        assert "sources" in counters
        assert "doc_types" in counters
        assert "entities" in counters
        assert "topics" in counters
        assert "years" in counters
        assert total_chars > 0

    def test_analyze_all_chunks_sources(self, analyzer, sample_chunks):
        """Test _analyze_all_chunks tracks sources."""
        counters, _ = analyzer._analyze_all_chunks(sample_chunks)
        assert len(counters["sources"]) == 3

    def test_analyze_all_chunks_char_count(self, analyzer, sample_chunks):
        """Test _analyze_all_chunks counts characters."""
        counters, total_chars = analyzer._analyze_all_chunks(sample_chunks)
        expected = sum(len(c.get("content", "")) for c in sample_chunks)
        assert total_chars == expected

    # Edge Cases
    def test_analyze_single_chunk(self, analyzer):
        """Test analyze with single chunk."""
        chunks = [
            {
                "source_file": "test.txt",
                "content": "Test content here",
                "source_location": {},
            }
        ]
        result = analyzer.analyze(chunks)
        assert result.total_chunks == 1
        assert result.total_documents == 1

    def test_analyze_chunks_without_content(self, analyzer):
        """Test analyze with chunks missing content."""
        chunks = [
            {"source_file": "test.txt", "source_location": {}},
        ]
        result = analyzer.analyze(chunks)
        assert result.total_chunks == 1
        assert result.avg_chunk_size == 0

    def test_analyze_chunks_without_source_location(self, analyzer):
        """Test analyze with chunks missing source_location."""
        chunks = [
            {"source_file": "test.txt", "content": "test content"},
        ]
        result = analyzer.analyze(chunks)
        assert result.total_chunks == 1

    def test_analyze_very_large_corpus(self, analyzer):
        """Test analyze with many chunks."""
        chunks = [
            {
                "source_file": f"doc{i}.txt",
                "content": f"Content {i} with some words",
                "source_location": {"publication_date": f"202{i % 4}-01-01"},
            }
            for i in range(100)
        ]
        result = analyzer.analyze(chunks)
        assert result.total_chunks == 100
        assert result.total_documents == 100

    def test_analyze_unicode_content(self, analyzer):
        """Test analyze with unicode content."""
        chunks = [
            {
                "source_file": "test.txt",
                "content": "Café résumé naïve façade",
                "source_location": {},
            }
        ]
        result = analyzer.analyze(chunks)
        assert result.total_chunks == 1

    def test_analyze_special_characters(self, analyzer):
        """Test analyze with special characters."""
        chunks = [
            {
                "source_file": "test.txt",
                "content": "@#$%^&*() special characters here",
                "source_location": {},
            }
        ]
        result = analyzer.analyze(chunks)
        assert result.total_chunks == 1

    def test_analyze_mixed_content(self, analyzer):
        """Test analyze with mixed content types."""
        chunks = [
            {
                "source_file": "test.pdf",
                "content": "PDF content",
                "source_location": {"publication_date": "2023-01-01"},
            },
            {
                "source_file": "test.docx",
                "content": "DOCX content",
                "source_location": {},
            },
            {
                "source_file": "no_extension",
                "content": "Plain content",
                "source_location": {"publication_date": "invalid"},
            },
        ]
        result = analyzer.analyze(chunks)
        assert result.total_chunks == 3
        assert len(result.document_types) > 0

    # Integration Tests
    def test_full_analysis_pipeline(self, analyzer, sample_chunks):
        """Test complete analysis pipeline."""
        result = analyzer.analyze(sample_chunks)

        # Verify all fields are populated
        assert result.total_documents > 0
        assert result.total_chunks > 0
        assert result.avg_chunk_size > 0
        assert len(result.document_types) > 0
        assert isinstance(result.top_entities, list)
        assert isinstance(result.top_topics, list)
        assert result.source_diversity > 0
        assert len(result.temporal_distribution) > 0
        assert result.reading_time_minutes >= 0
        assert len(result.coverage_gaps) > 0

    def test_analyze_idempotent(self, analyzer, sample_chunks):
        """Test analyze produces consistent results."""
        result1 = analyzer.analyze(sample_chunks)
        result2 = analyzer.analyze(sample_chunks)

        assert result1.total_chunks == result2.total_chunks
        assert result1.avg_chunk_size == result2.avg_chunk_size
        assert result1.reading_time_minutes == result2.reading_time_minutes
