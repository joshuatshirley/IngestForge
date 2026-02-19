"""Sentiment analysis enrichment.

Analyzes sentiment polarity (positive/negative/neutral) of text chunks
using lexicon-based approach."""

from __future__ import annotations

from typing import Dict, Any, Set, List


class SentimentAnalyzer:
    """Analyze sentiment polarity of text."""

    def __init__(self) -> None:
        """Initialize sentiment analyzer."""
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()

    def analyze(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of chunk.

        Args:
            chunk: Chunk dictionary with 'text' field

        Returns:
            Enriched chunk with sentiment fields
        """
        text = chunk.get("text", "").lower()

        if not text:
            chunk["sentiment"] = "neutral"
            chunk["sentiment_score"] = 0.0
            chunk["positive_words"] = []
            chunk["negative_words"] = []
            return chunk

        # Count sentiment words
        words = text.split()
        pos_found = [w for w in words if w in self.positive_words]
        neg_found = [w for w in words if w in self.negative_words]

        # Calculate score
        pos_count = len(pos_found)
        neg_count = len(neg_found)
        total_words = len(words)

        score = (pos_count - neg_count) / total_words if total_words > 0 else 0.0

        # Determine sentiment
        if score > 0.05:
            sentiment = "positive"
        elif score < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Add to chunk
        chunk["sentiment"] = sentiment
        chunk["sentiment_score"] = round(score, 3)
        chunk["positive_words"] = pos_found[:5]  # Top 5
        chunk["negative_words"] = neg_found[:5]  # Top 5

        return chunk

    def _load_positive_words(self) -> Set[str]:
        """Load positive sentiment words.

        Returns:
            Set of positive words
        """
        return {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "best",
            "better",
            "love",
            "loved",
            "happy",
            "joy",
            "pleased",
            "success",
            "successful",
            "win",
            "won",
            "brilliant",
            "outstanding",
            "perfect",
            "beautiful",
            "nice",
            "awesome",
            "impressive",
            "positive",
            "beneficial",
            "advantage",
            "gain",
            "improve",
            "superior",
            "effective",
        }

    def _load_negative_words(self) -> Set[str]:
        """Load negative sentiment words.

        Returns:
            Set of negative words
        """
        return {
            "bad",
            "terrible",
            "horrible",
            "awful",
            "worst",
            "worse",
            "hate",
            "hated",
            "sad",
            "unhappy",
            "fail",
            "failed",
            "failure",
            "loss",
            "poor",
            "negative",
            "problem",
            "issue",
            "wrong",
            "error",
            "mistake",
            "difficult",
            "hard",
            "impossible",
            "weak",
            "inferior",
            "ineffective",
            "harm",
            "damage",
            "risk",
            "threat",
            "danger",
            "crisis",
        }


def analyze_sentiment(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze sentiment of chunk.

    Args:
        chunk: Chunk dictionary

    Returns:
        Enriched chunk with sentiment
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(chunk)


def batch_analyze_sentiment(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze sentiment for multiple chunks.

    Args:
        chunks: List of chunks

    Returns:
        Enriched chunks with sentiment
    """
    analyzer = SentimentAnalyzer()
    return [analyzer.analyze(chunk) for chunk in chunks]
