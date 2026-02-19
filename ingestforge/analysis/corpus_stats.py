"""Corpus statistics and analytics."""
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any
import re


@dataclass
class CorpusStatistics:
    """Statistics about the corpus."""

    total_documents: int
    total_chunks: int
    avg_chunk_size: int
    document_types: Dict[str, int]
    top_entities: List[tuple[str, int]]
    top_topics: List[tuple[str, int]]
    source_diversity: int
    temporal_distribution: Dict[str, int]
    reading_time_minutes: int
    coverage_gaps: List[str]


class CorpusAnalyzer:
    """Analyze corpus and generate statistics."""

    WORDS_PER_MINUTE = 200  # Average reading speed

    def _analyze_chunk_source(
        self, chunk: Dict[str, Any], sources: set[str], doc_types: Counter[str]
    ) -> None:
        """
        Analyze chunk source and document type.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            chunk: Chunk dictionary
            sources: Set of sources (mutated)
            doc_types: Document type counter (mutated)
        """
        source = chunk.get("source_file", "unknown")
        sources.add(source)

        if "." in source:
            ext = source.rsplit(".", 1)[-1].lower()
            doc_types[ext] += 1

    def _extract_entities(self, content: str, entities: Counter[str]) -> None:
        """
        Extract entities (capitalized phrases) from content.

        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            content: Text content
            entities: Entity counter (mutated)
        """
        capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content)
        for entity in capitalized:
            if len(entity) > 3:
                entities[entity] += 1

    def _extract_topics(
        self, content: str, topics: Counter[str], stopwords: set[str]
    ) -> None:
        """
        Extract topics (important terms) from content.

        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            content: Text content
            topics: Topic counter (mutated)
            stopwords: Set of stopwords to filter
        """
        words = re.findall(r"\b\w{5,}\b", content.lower())
        for word in words:
            if word not in stopwords and not word.isdigit():
                topics[word] += 1

    def _extract_temporal_data(
        self, chunk: Dict[str, Any], years: Counter[str]
    ) -> None:
        """
        Extract temporal data (publication year) from chunk.

        Rule #1: Early return pattern
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            chunk: Chunk dictionary
            years: Year counter (mutated)
        """
        source_loc = chunk.get("source_location", {})
        pub_date = source_loc.get("publication_date", "")
        if not pub_date or len(pub_date) < 4:
            return

        year = pub_date[:4]
        if year.isdigit():
            years[year] += 1

    def _calculate_reading_time(self, total_chars: int) -> int:
        """
        Calculate reading time from character count.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            total_chars: Total character count

        Returns:
            Reading time in minutes
        """
        total_words = total_chars // 5  # Estimate words
        return total_words // self.WORDS_PER_MINUTE

    def analyze(self, chunks: List[Dict[str, Any]]) -> CorpusStatistics:
        """
        Analyze corpus and generate statistics.

        Rule #1: Early return for empty input
        Rule #4: Reduced from 64 → 38 lines

        Args:
            chunks: List of chunk dictionaries

        Returns:
            CorpusStatistics with analysis results
        """
        if not chunks:
            return self._create_empty_statistics()
        counters, total_chars = self._analyze_all_chunks(chunks)

        # Calculate final metrics
        avg_chunk_size = total_chars // len(chunks)
        reading_time = self._calculate_reading_time(total_chars)
        gaps = self._identify_gaps(chunks, counters["topics"])

        return CorpusStatistics(
            total_documents=len(counters["sources"]),
            total_chunks=len(chunks),
            avg_chunk_size=avg_chunk_size,
            document_types=dict(counters["doc_types"]),
            top_entities=counters["entities"].most_common(15),
            top_topics=counters["topics"].most_common(20),
            source_diversity=len(counters["sources"]),
            temporal_distribution=dict(counters["years"]),
            reading_time_minutes=reading_time,
            coverage_gaps=gaps,
        )

    def _create_empty_statistics(self) -> CorpusStatistics:
        """Rule #4: Extracted empty statistics creation (<60 lines)."""
        return CorpusStatistics(
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

    def _analyze_all_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> tuple[Dict[str, Any], int]:
        """
        Analyze all chunks and collect statistics.

        Rule #4: Extracted analysis loop (<60 lines)

        Returns:
            Tuple of (counters dict, total_chars)
        """
        sources: set[str] = set[Any]()
        doc_types: Counter[str] = Counter()
        entities: Counter[str] = Counter()
        topics: Counter[str] = Counter()
        years: Counter[str] = Counter()
        total_chars = 0

        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "are",
            "was",
            "has",
            "have",
            "been",
            "will",
            "can",
            "may",
            "also",
            "their",
            "which",
        }

        for chunk in chunks:
            self._analyze_chunk_source(chunk, sources, doc_types)
            content = chunk.get("content", "")
            total_chars += len(content)
            self._extract_entities(content, entities)
            self._extract_topics(content, topics, stopwords)
            self._extract_temporal_data(chunk, years)

        counters = {
            "sources": sources,
            "doc_types": doc_types,
            "entities": entities,
            "topics": topics,
            "years": years,
        }

        return counters, total_chars

    def _identify_gaps(
        self, chunks: List[Dict[str, Any]], topics: Counter[str]
    ) -> List[str]:
        """
        Identify potential coverage gaps in the corpus.

        Args:
            chunks: List of chunks
            topics: Topic frequency counter

        Returns:
            List of identified gaps
        """
        gaps = []

        # Check for imbalanced coverage
        if not topics:
            return ["Corpus may be too small for meaningful analysis"]

        # Get topic frequencies
        top_topic_count = topics.most_common(1)[0][1] if topics else 0
        bottom_topics = [
            t for t, c in topics.items() if c < max(2, top_topic_count // 20)
        ]

        if len(bottom_topics) > len(topics) // 2:
            gaps.append(
                "Many topics have low coverage - consider adding more diverse content"
            )

        # Check for temporal gaps (if temporal data exists)
        years_with_data = set[Any]()
        for chunk in chunks:
            source_loc = chunk.get("source_location", {})
            pub_date = source_loc.get("publication_date", "")
            if pub_date and len(pub_date) >= 4:
                year = pub_date[:4]
                if year.isdigit():
                    years_with_data.add(int(year))

        if years_with_data:
            min_year, max_year = min(years_with_data), max(years_with_data)
            if max_year - min_year > 3:
                year_range = set(range(min_year, max_year + 1))
                missing_years = year_range - years_with_data
                if missing_years and len(missing_years) < 5:
                    gaps.append(f"Missing coverage for years: {sorted(missing_years)}")

        if not gaps:
            gaps.append("No significant coverage gaps detected")

        return gaps
