"""Smart query suggestions based on corpus analysis."""
import json
from collections import Counter
from pathlib import Path
from typing import Any, List, Dict
import re


class QuerySuggester:
    """Generate smart query suggestions from corpus."""

    def __init__(self, cache_path: Path) -> None:
        """
        Initialize suggester.

        Args:
            cache_path: Path to suggestions cache file
        """
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def analyze_corpus(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze corpus to extract suggestion candidates.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Analysis results with topics, entities, patterns
        """
        # Extract terms and their frequencies
        term_freq: Counter[str] = Counter()
        bigrams: Counter[str] = Counter()
        entities: set[str] = set()

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
            "more",
            "than",
            "were",
            "when",
            "they",
            "what",
            "how",
            "about",
            "into",
        }

        for chunk in chunks:
            content = chunk.get("content", "").lower()

            # Extract capitalized phrases (likely entities/topics)
            original = chunk.get("content", "")
            capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", original)
            for entity in capitalized:
                if len(entity) > 3:
                    entities.add(entity)

            # Extract individual terms
            words = re.findall(r"\b\w{4,}\b", content)
            for word in words:
                if word not in stopwords and not word.isdigit():
                    term_freq[word] += 1

            # Extract bigrams
            for i in range(len(words) - 1):
                if words[i] not in stopwords and words[i + 1] not in stopwords:
                    bigram = f"{words[i]} {words[i+1]}"
                    bigrams[bigram] += 1

        # Get top suggestions
        top_terms = [term for term, _ in term_freq.most_common(50)]
        top_bigrams = [bg for bg, count in bigrams.most_common(30) if count >= 3]
        top_entities = list(entities)[:20]

        analysis = {
            "top_terms": top_terms,
            "top_bigrams": top_bigrams,
            "entities": top_entities,
            "corpus_size": len(chunks),
        }

        # Cache results
        self._save_cache(analysis)

        return analysis

    def get_suggestions(self, limit: int = 10) -> List[str]:
        """
        Get general query suggestions from corpus.

        Args:
            limit: Number of suggestions to return

        Returns:
            List of suggested queries
        """
        analysis = self._load_cache()
        if not analysis:
            return []

        suggestions = []

        # Add top entities
        suggestions.extend(analysis.get("entities", [])[: limit // 3])

        # Add top bigrams
        suggestions.extend(analysis.get("top_bigrams", [])[: limit // 3])

        # Combine terms into queries
        top_terms = analysis.get("top_terms", [])
        for i in range(0, min(len(top_terms) - 1, limit // 3 * 2), 2):
            suggestions.append(f"{top_terms[i]} {top_terms[i+1]}")

        return suggestions[:limit]

    def get_related_queries(
        self, original_query: str, chunks: List[Dict[str, Any]], limit: int = 3
    ) -> List[str]:
        """
        Generate related query suggestions based on search results.

        Args:
            original_query: The original query
            chunks: Result chunks from the query
            limit: Number of related queries to generate

        Returns:
            List of related query suggestions
        """
        if not chunks:
            return []

        # Extract key terms from results
        term_freq: Counter[str] = Counter()
        entities: set[str] = set()
        query_terms = set(original_query.lower().split())

        for chunk in chunks[:10]:  # Analyze top 10 results
            content = chunk.get("content", "")

            # Extract capitalized entities
            capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content)
            for entity in capitalized:
                if len(entity) > 3 and entity.lower() not in query_terms:
                    entities.add(entity)

            # Extract terms
            words = re.findall(r"\b\w{4,}\b", content.lower())
            for word in words:
                if word not in query_terms and not word.isdigit():
                    term_freq[word] += 1

        suggestions = []

        # Add entity-based suggestions
        for entity in list(entities)[:limit]:
            suggestions.append(entity)

        # Add term-based suggestions
        top_terms = [term for term, _ in term_freq.most_common(limit * 2)]
        for term in top_terms:
            if len(suggestions) >= limit:
                break
            suggestions.append(f"{original_query} {term}")

        return suggestions[:limit]

    def _save_cache(self, analysis: Dict[str, Any]) -> None:
        """Save analysis to cache file."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

    def _load_cache(self) -> Dict[str, Any]:
        """Load analysis from cache file."""
        if not self.cache_path.exists():
            return {}

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                result: dict[str, Any] = json.load(f)
                return result
        except (json.JSONDecodeError, IOError):
            return {}
