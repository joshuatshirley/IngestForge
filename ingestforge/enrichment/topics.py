"""Topic modeling enrichment.

Identifies topics and themes in text chunks using simple frequency-based
and pattern-based approaches."""

from __future__ import annotations

from typing import List, Dict, Any, Set
from collections import Counter
import re


class TopicModeler:
    """Extract topics from text chunks.

    Aliases: TopicDetector, TopicExtractor (for backwards compatibility)
    """

    def __init__(self, min_term_freq: int = 2) -> None:
        """Initialize topic modeler.

        Args:
            min_term_freq: Minimum term frequency for topics
        """
        self.min_term_freq = min_term_freq
        self.stop_words = self._load_stop_words()

    def extract_topics(
        self, chunks: List[Dict[str, Any]], num_topics: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract topics from chunks.

        Args:
            chunks: List of chunks
            num_topics: Number of top topics to return

        Returns:
            List of topic dictionaries
        """
        # Combine all text
        all_text = " ".join([c.get("text", "") for c in chunks])

        # Extract terms
        terms = self._extract_terms(all_text)

        # Count term frequencies
        term_counts = Counter(terms)

        # Filter by minimum frequency
        frequent_terms = {
            term: count
            for term, count in term_counts.items()
            if count >= self.min_term_freq
        }

        # Get top topics
        top_topics = sorted(frequent_terms.items(), key=lambda x: x[1], reverse=True)[
            :num_topics
        ]

        return [
            {
                "topic": term,
                "frequency": count,
                "weight": count / len(terms) if terms else 0,
            }
            for term, count in top_topics
        ]

    def enrich_chunk(
        self, chunk: Dict[str, Any], global_topics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enrich chunk with topic labels.

        Args:
            chunk: Chunk dictionary
            global_topics: List of global topics

        Returns:
            Enriched chunk with topics
        """
        text = chunk.get("text", "").lower()

        # Find which topics appear in this chunk
        chunk_topics = []
        for topic in global_topics:
            topic_term = topic["topic"].lower()
            if topic_term in text:
                chunk_topics.append(topic["topic"])

        chunk["topics"] = chunk_topics
        chunk["primary_topic"] = chunk_topics[0] if chunk_topics else None

        return chunk

    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text.

        Args:
            text: Input text

        Returns:
            List of terms
        """
        # Convert to lowercase
        text = text.lower()

        # Extract potential terms (2-3 word phrases and single words)
        terms = []

        # Extract bigrams
        words = re.findall(r"\b[a-z]{3,}\b", text)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if self._is_valid_term(bigram):
                terms.append(bigram)

        # Extract trigrams
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if self._is_valid_term(trigram):
                terms.append(trigram)

        # Extract single words
        terms.extend([w for w in words if self._is_valid_term(w)])

        return terms

    def _is_valid_term(self, term: str) -> bool:
        """Check if term is valid.

        Args:
            term: Term to check

        Returns:
            True if valid
        """
        # Remove stop words
        words = term.split()
        if any(w in self.stop_words for w in words):
            return False

        # Minimum length
        if len(term) < 4:
            return False

        return True

    def _load_stop_words(self) -> Set[str]:
        """Load stop words.

        Returns:
            Set of stop words
        """
        return {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "man",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
            "this",
            "that",
            "with",
            "have",
            "from",
            "they",
            "what",
            "been",
            "more",
            "when",
            "will",
            "your",
            "there",
            "their",
        }


def extract_topics(
    chunks: List[Dict[str, Any]], num_topics: int = 10
) -> List[Dict[str, Any]]:
    """Extract topics from chunks.

    Args:
        chunks: List of chunks
        num_topics: Number of topics

    Returns:
        List of topics
    """
    modeler = TopicModeler()
    return modeler.extract_topics(chunks, num_topics)


def enrich_with_topics(
    chunks: List[Dict[str, Any]], topics: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Enrich chunks with topic labels.

    Args:
        chunks: List of chunks
        topics: List of global topics

    Returns:
        Enriched chunks
    """
    modeler = TopicModeler()
    return [modeler.enrich_chunk(chunk, topics) for chunk in chunks]


# Aliases for backwards compatibility
TopicDetector = TopicModeler
TopicExtractor = TopicModeler
